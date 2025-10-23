"""FEBio metadata extraction utilities.

This module provides a typed interface for crawling FEBio headers, gathering
`BEGIN_FECORE_CLASS` macro blocks, and exposing the results as structured
dataclasses that can be reused by the downstream code generator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from clang.cindex import Cursor, CursorKind, Index

from .config import FEBIO_ROOT


def _existing_paths(paths: Iterable[Path]) -> list[Path]:
    """Return the subset of ``paths`` that exist on disk."""

    result: list[Path] = []
    for candidate in paths:
        try:
            if candidate.exists():
                result.append(candidate)
        except OSError:
            continue
    return result


def _default_febio_include_dirs() -> list[Path]:
    """Best-effort include directories relative to the FEBio checkout."""

    candidates = [FEBIO_ROOT, FEBIO_ROOT / "src", FEBIO_ROOT / "include"]
    return _existing_paths([path.resolve() for path in candidates])


MANIFEST_CATEGORY_DEFAULTS: dict[str, tuple[str | None, str | None]] = {
    "material": ("material", "Materials"),
    "core": (None, None),
    "boundaryconditions": ("bc", "Boundary Conditions"),
    "loads": ("load", "Loads"),
    "constraints": ("constraint", "Constraints"),
    "solver": (None, None),
    "control": (None, None),
}


def _manifest_defaults(
    category: str | None, xml_tag: str | None, xml_section: str | None
) -> tuple[str | None, str | None]:
    if category:
        defaults = MANIFEST_CATEGORY_DEFAULTS.get(category.lower())
    else:
        defaults = None
    if defaults:
        default_tag, default_section = defaults
    else:
        default_tag, default_section = ("material", category or "Materials")
    tag = xml_tag if xml_tag is not None else default_tag
    section = xml_section if xml_section is not None else default_section
    return tag, section


def _manifest_filename(category: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "", category)
    token = token or "Manifest"
    return f"{token}.json"


def _relative_header_path(header_path: Path) -> str:
    resolved = header_path.resolve()
    try:
        return resolved.relative_to(FEBIO_ROOT).as_posix()
    except (ValueError, RuntimeError):
        return resolved.as_posix()


def _build_manifest_entries(
    metadata: Sequence[ClassMetadata],
    header_path: Path,
    *,
    xml_tag: str,
    xml_section: str,
) -> list[dict[str, Any]]:
    header_value = _relative_header_path(header_path)
    entries: list[dict[str, Any]] = []
    for item in metadata:
        entry: dict[str, Any] = {
            "class": item.class_name,
            "python_name": item.class_name,
            "header": header_value,
        }
        if xml_tag:
            entry["xml_tag"] = xml_tag
        entries.append(entry)
    return entries


def _print_manifest_suggestion(
    entries: Sequence[dict[str, Any]], category: str, manifest_path: Path | None = None
) -> None:
    if not entries:
        return
    location = f" at {manifest_path}" if manifest_path else ""
    print(f"[manifest] Suggested entries for '{category}'{location}:")
    print(json.dumps({category: list(entries)}, indent=2))


def _update_manifest_file(
    manifest_path: Path,
    category: str,
    entries: Sequence[dict[str, Any]],
    *,
    dry_run: bool,
) -> None:
    if not entries:
        return
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text())
    else:
        data = {}
    existing_list: list[dict[str, Any]] = data.get(category, [])
    mapping = {item["class"]: item for item in existing_list}
    changed = False
    for entry in entries:
        current = mapping.get(entry["class"])
        if current != entry:
            mapping[entry["class"]] = entry
            changed = True
    if not changed:
        message = "already up to date" if not dry_run else "would remain unchanged"
        print(f"[manifest] {manifest_path} ({category}) {message}.")
        return
    new_list = sorted(
        mapping.values(),
        key=lambda item: item.get("python_name", item.get("class", "")),
    )
    if dry_run:
        print(
            f"[manifest] Dry run: {manifest_path} would be updated with {len(entries)} entr"
            f"{'y' if len(entries) == 1 else 'ies'}."
        )
        return
    data[category] = new_list
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(data, indent=2) + "\n")
    print(
        f"[manifest] Updated {manifest_path} with {len(entries)} entr"
        f"{'y' if len(entries) == 1 else 'ies'}."
    )


def _maybe_process_manifest(
    metadata: Sequence[ClassMetadata], header_path: Path, args: argparse.Namespace
) -> None:
    if not metadata:
        return
    category = args.category or "Materials"
    xml_tag, xml_section = _manifest_defaults(category, args.xml_tag, args.xml_section)
    entries = _build_manifest_entries(
        metadata, header_path, xml_tag=xml_tag, xml_section=xml_section
    )
    manifest_path: Path | None = None
    if args.manifest_dir:
        manifest_dir = Path(args.manifest_dir)
        manifest_filename = _manifest_filename(category)
        manifest_path = manifest_dir / manifest_filename
    _print_manifest_suggestion(entries, category, manifest_path)
    if manifest_path:
        _update_manifest_file(
            manifest_path, category, entries, dry_run=args.manifest_dry_run
        )


def _decode_c_string(literal: str) -> str:
    """Decode a C-style string literal into Python."""

    try:
        return bytes(literal, encoding="utf-8").decode("unicode_escape")
    except Exception:
        return literal


def _normalise_cpp_class_name(ctype: str | None) -> str | None:
    if not ctype:
        return None
    cleaned = (
        ctype.replace("*", "")
        .replace("&", "")
        .replace("const", "")
        .replace("class", "")
        .strip()
    )
    if not cleaned:
        return None
    tokens = cleaned.split()
    if not tokens:
        return None
    return tokens[-1].split("::")[-1]


def _extract_enum_tokens_from_text(enum_name: str, text: str) -> list[str] | None:
    pattern = re.compile(rf"enum\s+(?:class\s+)?{re.escape(enum_name)}\b")
    for match in pattern.finditer(text):
        idx = match.end()
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text) or text[idx] != '{':
            continue
        idx += 1
        depth = 1
        start = idx
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
            idx += 1
        body = text[start : idx - 1]
        body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
        lines = [re.sub(r"//.*$", "", line) for line in body.splitlines()]
        cleaned = "\n".join(lines)
        tokens: list[str] = []
        for entry in cleaned.split(','):
            token = entry.strip()
            if not token:
                continue
            if '=' in token:
                token = token.split('=', 1)[0].strip()
            if token:
                token = token.split('::')[-1]
                tokens.append(token)
        if tokens:
            return tokens
    return None


def _parse_flag_value(tokens: list[str]) -> int:
    total = 0
    for token in tokens:
        for part in str(token).split('|'):
            entry = part.strip()
            if not entry:
                continue
            try:
                total |= int(entry, 0)
                continue
            except ValueError:
                pass
            name = entry.split('::')[-1]
            flag = _PARAM_FLAG_MAP.get(name)
            if flag is not None:
                total |= flag
    return total


def _collect_base_defaults(base_class: str, class_map: dict[str, ClassMetadata]) -> dict[str, Any]:
    defaults: dict[str, Any] = {}
    seen: set[str] = set()
    current = base_class
    while current and current not in seen:
        seen.add(current)
        metadata = class_map.get(current)
        if metadata is None:
            break
        for param in metadata.params:
            member = param.member
            if not member:
                continue
            if member not in defaults and param.default is not None:
                defaults[member] = param.default
        current = metadata.base_class
    return defaults


_REGISTRATION_CACHE: dict[str, str] | None = None

_PARAM_FLAG_MAP = {
    "FE_PARAM_ATTRIBUTE": 0x01,
    "FE_PARAM_USER": 0x02,
    "FE_PARAM_HIDDEN": 0x04,
    "FE_PARAM_ADDLC": 0x08,
    "FE_PARAM_VOLATILE": 0x10,
    "FE_PARAM_TOPLEVEL": 0x20,
    "FE_PARAM_WATCH": 0x40,
    "FE_PARAM_OBSOLETE": 0x80,
}


def _load_global_registration_map() -> dict[str, str]:
    """Scan the FEBio tree for REGISTER_FECORE_CLASS invocations."""

    global _REGISTRATION_CACHE
    if _REGISTRATION_CACHE is not None:
        return _REGISTRATION_CACHE

    pattern = re.compile(
        r"REGISTER_FECORE_CLASS\s*\(\s*([A-Za-z0-9_:]+)\s*,\s*\"((?:\\.|[^\"])*)\"\s*\)"
    )
    result: dict[str, str] = {}
    search_roots = [FEBIO_ROOT]
    for root in search_roots:
        if not root.exists():
            continue
        for extension in ("*.cpp", "*.cxx", "*.cc"):
            for path in root.rglob(extension):
                try:
                    text = path.read_text(errors="ignore")
                except Exception:
                    continue
                for match in pattern.finditer(text):
                    result[match.group(1)] = _decode_c_string(match.group(2))

    _REGISTRATION_CACHE = result
    return result


RangeDict = dict[str, int | float | str]


_DOF_ENUM_CACHE: dict[str, str] | None = None
_DOF_SELECTOR_CACHE: dict[str, list[str]] | None = None


def _build_variable_name_lookup(root: Path) -> dict[str, str]:
    global _DOF_ENUM_CACHE
    if _DOF_ENUM_CACHE is not None:
        return _DOF_ENUM_CACHE
    mapping: dict[str, str] = {}
    if not root.exists():
        return mapping
    function_pattern = re.compile(
        r"const\s+char\s*\*\s+([A-Za-z0-9_:]+)::GetVariableName\s*\([^)]*\)\s*\{(.*?)\}",
        re.S,
    )
    case_pattern = re.compile(r"case\s+([A-Za-z0-9_:]+)\s*:\s*return\s+\"((?:\\.|[^\"])*)\"")
    for path in root.rglob("*.cpp"):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        for func_match in function_pattern.finditer(text):
            owner = func_match.group(1)
            body = func_match.group(2)
            for case_match in case_pattern.finditer(body):
                enumerator = case_match.group(1).strip()
                label = _decode_c_string(case_match.group(2))
                if not label:
                    continue
                if "::" not in enumerator and owner:
                    key = f"{owner}::{enumerator}"
                else:
                    key = enumerator
                mapping[key] = label
                short = key.split("::")[-1]
                mapping.setdefault(short, label)
    _DOF_ENUM_CACHE = mapping
    return mapping


def _resolve_add_variable_label(expr: str, enum_lookup: Mapping[str, str]) -> str | None:
    literal_match = re.search(r'"((?:\\.|[^\"])*)"', expr)
    if literal_match:
        return _decode_c_string(literal_match.group(1)).strip()
    getter_match = re.search(r'GetVariableName\(([^)]+)\)', expr)
    if getter_match:
        token = getter_match.group(1).strip()
        label = enum_lookup.get(token)
        if label:
            return label.strip()
        if "::" not in token:
            for prefix in ("FEBioMech", "FEBioFluid", "FEBioFSI", "FEBioFSI2", "FEBioMix", "FEBioPolarFluid"):
                candidate = f"{prefix}::{token}"
                label = enum_lookup.get(candidate)
                if label:
                    return label.strip()
    return None


def _collect_dof_selectors(root: Path) -> dict[str, list[str]]:
    global _DOF_SELECTOR_CACHE
    if _DOF_SELECTOR_CACHE is not None:
        return _DOF_SELECTOR_CACHE
    if not root.exists():
        return {}
    enum_lookup = _build_variable_name_lookup(root)
    add_var_pattern = re.compile(
        r'(?:const\s+)?(?:auto|int)\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*?AddVariable\((.*?)\);',
        re.S,
    )
    assign_pattern = re.compile(
        r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*[^;]*?AddVariable\((.*?)\);',
        re.S,
    )
    set_name_pattern = re.compile(
        r'SetDOFName\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,\s*(\d+)\s*,\s*"((?:\\.|[^\"])*)"\s*\)'
    )
    selector_map: dict[str, list[str | None]] = {}
    for path in root.rglob("*.cpp"):
        try:
            text = path.read_text(errors="ignore")
        except OSError:
            continue
        var_labels: dict[str, str] = {}
        for pattern in (add_var_pattern, assign_pattern):
            for match in pattern.finditer(text):
                var_name = match.group(1)
                expr = match.group(2)
                if var_name in var_labels:
                    continue
                label = _resolve_add_variable_label(expr, enum_lookup)
                if not label:
                    continue
                normalized = label.strip().lower()
                if normalized:
                    var_labels[var_name] = normalized
        if not var_labels:
            continue
        for match in set_name_pattern.finditer(text):
            var_name, index_text, value = match.groups()
            label = var_labels.get(var_name)
            if not label:
                continue
            idx = int(index_text)
            decoded_value = _decode_c_string(value)
            bucket = selector_map.setdefault(label, [])
            while len(bucket) <= idx:
                bucket.append(None)
            if bucket[idx] is None:
                bucket[idx] = decoded_value
    cleaned: dict[str, list[str]] = {}
    for label, entries in selector_map.items():
        values = [token for token in entries if token]
        if values:
            cleaned[label] = values
    union: list[str] = []
    for values in cleaned.values():
        for token in values:
            if token not in union:
                union.append(token)
    if union:
        cleaned["dof_list"] = union
    _DOF_SELECTOR_CACHE = cleaned
    return cleaned


def _resolve_dof_selector(selector: str) -> list[str] | None:
    match = re.fullmatch(r"\$\((dof_list)(?::([^\)]+))?\)", selector.strip())
    if not match:
        return None
    _, variant = match.groups()
    mapping = _collect_dof_selectors(FEBIO_ROOT)
    if not mapping:
        return None
    if variant is None:
        return None
    key = _decode_c_string(variant).strip().lower()
    if not key:
        return None
    return mapping.get(key)


@dataclass(slots=True)
class MacroChainCall:
    """Represents a chained macro call such as `->setUnits(...)`.

    Attributes:
        func: Name of the chained function.
        args: Raw string arguments captured from the call.
    """

    func: str
    args: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialise the call.

        Returns:
            Mapping containing ``func`` and ``args`` keys.
        """

        return {"func": self.func, "args": self.args}


@dataclass(slots=True)
class ParameterInfo:
    """Captures the information of a single `ADD_PARAMETER` or `ADD_PROPERTY`.

    Attributes:
        macro: Name of the macro that introduced the entry.
        args: Raw macro arguments.
        member: Field/member name associated with the parameter, when known.
        ctype: C++ type of the field when it can be resolved.
        chain: Chained calls that follow the macro invocation.
        default: Parsed default value if it can be inferred.
        strings: Literal string arguments found inside the macro call.
        range: Interval or selector information extracted from FE_RANGE macros.
        name: Human readable name (typically the parameter label).
        enum: Selector enum identifier when present.
        long_name: Long name metadata pulled from chained calls.
        units: Units metadata pulled from chained calls.
        definition: Nested property definition when resolved via `ADD_PROPERTY`.
    """

    macro: str
    args: list[str]
    member: str | None
    ctype: str | None
    chain: list[MacroChainCall]
    default: int | float | str | bool | None
    strings: list[str]
    range: RangeDict | str | None
    name: str | None
    enum: str | list[str] | None
    long_name: str | None
    units: str | None
    definition: list[dict[str, Any]] | None
    hidden: bool = False

    def to_json(self) -> dict[str, Any]:
        """Serialise the parameter.

        Returns:
            Dictionary matching the legacy JSON structure.
        """

        data = {
            "macro": self.macro,
            "args": self.args,
            "chain": [call.to_json() for call in self.chain],
            "member": self.member,
            "ctype": self.ctype,
            "default": self.default,
            "strings": self.strings,
            "range": self.range,
            "name": self.name,
            "enum": self.enum,
            "long_name": self.long_name,
            "units": self.units,
        }
        if self.definition is not None:
            data["definition"] = self.definition
        if self.hidden:
            data["hidden"] = True
        return data


@dataclass(slots=True)
class ClassMetadata:
    """Aggregated metadata for a FEBio class declaration."""

    class_name: str
    base_class: str
    registration: str | None = None
    params: list[ParameterInfo] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        """Serialise the class metadata.

        Returns:
            Dictionary with keys ``class``, ``base`` and ``params``.
        """

        return {
            "class": self.class_name,
            "base": self.base_class,
            "registration": self.registration,
            "params": [param.to_json() for param in self.params],
        }


def _find_matching_paren(text: str, open_idx: int) -> int:
    """Find the closing parenthesis index for a balanced pair.

    Args:
        text: String to search.
        open_idx: Index of the opening parenthesis.

    Returns:
        Index of the matching closing parenthesis or ``-1`` when not found.
    """

    depth = 0
    in_single = False
    in_double = False
    idx = open_idx
    while idx < len(text):
        char = text[idx]
        if char == "'" and not in_double:
            in_single = not in_single
            idx += 1
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            idx += 1
            continue
        if char == "\\" and (in_single or in_double) and idx + 1 < len(text):
            idx += 2
            continue
        if in_single or in_double:
            idx += 1
            continue
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return idx
        idx += 1
    return -1


def _split_args(serialised: str) -> list[str]:
    """Split an argument list while respecting nested parentheses.

    Args:
        serialised: Raw argument list without surrounding parentheses.

    Returns:
        A list of argument fragments.
    """

    arguments: list[str] = []
    buffer = []
    depth = 0
    in_single = False
    in_double = False
    escape = False
    for char in serialised:
        if escape:
            buffer.append(char)
            escape = False
            continue
        if char == "\\" and (in_single or in_double):
            buffer.append(char)
            escape = True
            continue
        if char == "'" and not in_double:
            in_single = not in_single
            buffer.append(char)
            continue
        if char == '"' and not in_single:
            in_double = not in_double
            buffer.append(char)
            continue
        if char == "," and depth == 0 and not in_single and not in_double:
            token = "".join(buffer).strip()
            if token:
                arguments.append(token)
            buffer.clear()
            continue
        buffer.append(char)
        if char == "(" and not in_single and not in_double:
            depth += 1
        elif char == ")" and not in_single and not in_double:
            depth -= 1
    trailing = "".join(buffer).strip()
    if trailing:
        arguments.append(trailing)
    return arguments


def _remove_cpp_comments(text: str) -> str:
    """Remove C/C++ comments while preserving literals and newlines."""

    result: list[str] = []
    i = 0
    length = len(text)
    in_single_quote = False
    in_double_quote = False
    while i < length:
        char = text[i]
        nxt = text[i + 1] if i + 1 < length else ""
        if not in_single_quote and not in_double_quote:
            if char == "/" and nxt == "/":
                i += 2
                while i < length and text[i] != "\n":
                    i += 1
                continue
            if char == "/" and nxt == "*":
                i += 2
                while i < length:
                    if i + 1 < length and text[i] == "*" and text[i + 1] == "/":
                        i += 2
                        break
                    if text[i] == "\n":
                        result.append("\n")
                    i += 1
                continue
        if char == "\"" and not in_single_quote:
            in_double_quote = not in_double_quote
            result.append(char)
            i += 1
            continue
        if char == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
            result.append(char)
            i += 1
            continue
        if char == "\\" and (in_single_quote or in_double_quote) and i + 1 < length:
            result.append(char)
            result.append(text[i + 1])
            i += 2
            continue
        result.append(char)
        i += 1
    return "".join(result)


def _find_matching_brace(text: str, open_idx: int) -> int:
    """Return the closing brace index matching ``open_idx`` or ``-1``."""

    depth = 0
    for idx in range(open_idx, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return idx
    return -1


def _split_initializers(serialised: str) -> list[str]:
    """Split constructor initializer lists while balancing delimiters."""

    entries: list[str] = []
    buffer: list[str] = []
    depth_round = depth_curly = depth_square = 0
    for char in serialised:
        if (
            char == ","
            and depth_round == 0
            and depth_curly == 0
            and depth_square == 0
        ):
            token = "".join(buffer).strip()
            if token:
                entries.append(token)
            buffer.clear()
            continue
        buffer.append(char)
        if char == "(":
            depth_round += 1
        elif char == ")":
            depth_round -= 1
        elif char == "{":
            depth_curly += 1
        elif char == "}":
            depth_curly -= 1
        elif char == "[":
            depth_square += 1
        elif char == "]":
            depth_square -= 1
    trailing = "".join(buffer).strip()
    if trailing:
        entries.append(trailing)
    return entries


def _parse_literal_value(token: str) -> int | float | str | bool:
    """Convert a simple C++ literal into a Python value."""

    cleaned = token.strip()
    lower = cleaned.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if cleaned.startswith("\"") and cleaned.endswith("\"") and len(cleaned) >= 2:
        return _decode_c_string(cleaned.strip("\""))
    if cleaned.startswith("'") and cleaned.endswith("'") and len(cleaned) >= 2:
        return _decode_c_string(cleaned.strip("'"))
    if cleaned.startswith("0x") or cleaned.startswith("0X"):
        try:
            return int(cleaned, 16)
        except ValueError:
            return cleaned
    try:
        return _try_number(cleaned)
    except Exception:
        return cleaned


def _parse_initializer_defaults(serialised: str) -> dict[str, int | float | str | bool]:
    """Extract member defaults from constructor initializer lists."""

    defaults: dict[str, int | float | str | bool] = {}
    if not serialised:
        return defaults
    for entry in _split_initializers(serialised):
        if not entry:
            continue
        entry = entry.strip()
        if entry.endswith(")"):
            open_idx = entry.find("(")
            close_idx = len(entry) - 1
        elif entry.endswith("}"):
            open_idx = entry.find("{")
            close_idx = len(entry) - 1
        else:
            continue
        if open_idx == -1:
            continue
        name = entry[:open_idx].strip()
        if not name.startswith("m_"):
            continue
        value_expr = entry[open_idx + 1 : close_idx].strip()
        if not value_expr:
            continue
        if "," in value_expr:
            defaults[name] = entry
        else:
            defaults[name] = _parse_literal_value(value_expr)
    return defaults



def _extract_constructor_defaults(text: str, class_name: str) -> dict[str, int | float | str | bool]:
    """Gather default member assignments for ``class_name`` from ``text``."""

    pattern = re.compile(
        rf"{re.escape(class_name)}::\s*{re.escape(class_name)}\s*\([^)]*\)",
        re.MULTILINE,
    )
    defaults: dict[str, int | float | str | bool] = {}
    for match in pattern.finditer(text):
        idx = match.end()
        while idx < len(text) and text[idx].isspace():
            idx += 1
        initializer_blob = ""
        if idx < len(text) and text[idx] == ':':
            idx += 1
            init_start = idx
            depth = 0
            while idx < len(text):
                char = text[idx]
                if char == '{' and depth == 0:
                    break
                if char == '(':
                    depth += 1
                elif char == ')':
                    depth -= 1
                idx += 1
            initializer_blob = text[init_start:idx].strip()
        body_defaults = {}
        if idx < len(text) and text[idx] == '{':
            brace_start = idx
            depth = 0
            while idx < len(text):
                char = text[idx]
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        idx += 1
                        break
                idx += 1
            body = text[brace_start + 1 : idx - 1]
            body_defaults = _parse_body_defaults(_remove_cpp_comments(body))
        init_defaults = _parse_initializer_defaults(initializer_blob) if initializer_blob else {}
        combined = {**init_defaults, **body_defaults}
        defaults.update(combined)
    return defaults



def _parse_body_defaults(body: str) -> dict[str, int | float | str | bool]:
    """Extract simple member assignments from a constructor body."""

    defaults: dict[str, int | float | str | bool] = {}
    array_assign = re.compile(r"(m_[A-Za-z0-9_]+)\s*\[\s*(\d+)\s*\]\s*=\s*([^;]+);")
    array_values: dict[str, dict[int, int | float | str | bool]] = {}
    for match in array_assign.finditer(body):
        name, index_text, rhs = match.groups()
        rhs = rhs.strip()
        if not rhs:
            continue
        if "," in rhs or "(" in rhs or ")" in rhs or "{" in rhs or "}" in rhs:
            value = rhs
        else:
            value = _parse_literal_value(rhs)
        array_values.setdefault(name, {})[int(index_text)] = value
    body = array_assign.sub("", body)

    assignment = re.compile(r"((?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*)+)([^;]+);")
    for match in assignment.finditer(body):
        targets_blob, rhs = match.groups()
        rhs = rhs.strip()
        if not rhs:
            continue
        if "," in rhs or "(" in rhs or ")" in rhs or "{" in rhs or "}" in rhs:
            value = rhs
        else:
            value = _parse_literal_value(rhs)
        targets = [token.strip() for token in targets_blob.split("=") if token.strip()]
        for target in targets:
            if not target.startswith("m_"):
                continue
            defaults[target] = value
    for name, mapping in array_values.items():
        if not mapping:
            continue
        max_index = max(mapping)
        values: list[int | float | str | bool | None] = []
        for idx in range(max_index + 1):
            values.append(mapping.get(idx))
        defaults[name] = tuple(values)
    return defaults


def _extract_combined_defaults(text: str, class_name: str) -> dict[str, int | float | str | bool]:
    """Gather default member assignments for ``class_name`` from ``text``."""

    pattern = re.compile(
        rf"{re.escape(class_name)}::\s*{re.escape(class_name)}\s*\([^)]*\)",
        re.MULTILINE,
    )
    defaults: dict[str, int | float | str | bool] = {}
    for match in pattern.finditer(text):
        idx = match.end()
        while idx < len(text) and text[idx].isspace():
            idx += 1
        initializer_blob = ""
        if idx < len(text) and text[idx] == ":":
            idx += 1
            init_start = idx
            depth = 0
            while idx < len(text):
                char = text[idx]
                if char == "{" and depth == 0:
                    break
                if char in "([{":
                    depth += 1
                elif char in ")]}":
                    depth = max(depth - 1, 0)
                idx += 1
            initializer_blob = text[init_start:idx].strip()
        while idx < len(text) and text[idx] != "{":
            idx += 1
        if idx >= len(text) or text[idx] != "{":
            continue
        body_start = idx
        body_end = _find_matching_brace(text, body_start)
        if body_end == -1:
            continue
        body = text[body_start + 1 : body_end]
        defaults.update(_parse_initializer_defaults(initializer_blob))
        defaults.update(_parse_body_defaults(body))
    return defaults


def _try_number(token: str) -> int | float | str:
    """Coerce a scalar token into a numeric value when possible.

    Args:
        token: Raw token as a string.

    Returns:
        Parsed ``int`` or ``float`` when possible, otherwise the stripped string.
    """

    try:
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token)
    except ValueError:
        return token.strip('"')


def _map_range(kind: str, args: list[int | float | str]) -> RangeDict | None:
    """Transform FE_RANGE macro names into structured dictionaries.

    Args:
        kind: Name of the FE_RANGE macro (e.g. ``FE_RANGE_CLOSED``).
        args: Parsed arguments inside the macro.

    Returns:
        Dictionary describing the interval semantics or ``None`` when the macro
        indicates no constraint.
    """

    upper = kind.upper()
    if "DONT_CARE" in upper:
        return None
    if "GREATER_OR_EQUAL" in upper:
        return {"min": args[0], "min_inclusive": True}
    if "GREATER" in upper:
        return {"min": args[0], "min_inclusive": False}
    if "LESS_OR_EQUAL" in upper:
        return {"max": args[0], "max_inclusive": True}
    if "LESS" in upper:
        return {"max": args[0], "max_inclusive": False}
    if "CLOSED" in upper:
        return {
            "min": args[0],
            "max": args[1],
            "min_inclusive": True,
            "max_inclusive": True,
        }
    if "LEFT_OPEN" in upper:
        return {
            "min": args[0],
            "max": args[1],
            "min_inclusive": False,
            "max_inclusive": True,
        }
    if "RIGHT_OPEN" in upper:
        return {
            "min": args[0],
            "max": args[1],
            "min_inclusive": True,
            "max_inclusive": False,
        }
    if "OPEN" in upper and "LEFT" not in upper and "RIGHT" not in upper:
        return {
            "min": args[0],
            "max": args[1],
            "min_inclusive": False,
            "max_inclusive": False,
        }
    if "NOT_EQUAL" in upper:
        return {"not_equal": args[0]}
    return {"raw_args": args}


def _parse_fe_range(text: str, position: int) -> RangeDict | str | None:
    """Parse an FE_RANGE macro invocation.

    Args:
        text: Line containing the macro call.
        position: Index of the first character of the macro name.

    Returns:
        Parsed range dictionary or selector string.
    """

    match = re.match(r"FE_RANGE_[A-Z_]*", text[position:])
    if not match:
        return None
    kind = match.group(0)
    start = position + match.end()
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "(":
        return None
    end = _find_matching_paren(text, start)
    if end == -1:
        return None
    inner = text[start + 1 : end]
    parsed_args = [_try_number(token) for token in _split_args(inner)]
    interval = _map_range(kind, parsed_args)
    return interval


class FEBioExtractor:
    """High-level API for extracting FEBio metadata from headers."""

    def __init__(self, *, include_dirs: Sequence[Path] | None = None) -> None:
        """Initialise the extractor.

        Args:
            include_dirs: Optional list of include directories forwarded to
                libclang during parsing. When omitted the extractor adds the
                FEBio checkout paths discovered via ``classExposer.config``.
        """

        self._index = Index.create()
        default_dirs = _default_febio_include_dirs()
        user_dirs = [Path(p).resolve() for p in include_dirs] if include_dirs else []

        combined: list[Path] = []
        for candidate in default_dirs + user_dirs:
            if candidate not in combined:
                combined.append(candidate)
        self._include_dirs = combined

    # ------------------------------------------------------------------
    def extract(self, headers: Sequence[Path | str]) -> list[ClassMetadata]:
        """Extract metadata from a collection of header files.

        Args:
            headers: Iterable of header paths.

        Returns:
            List of :class:`ClassMetadata` objects.
        """

        results: list[ClassMetadata] = []
        class_map: dict[str, ClassMetadata] = {}
        for header in headers:
            results.extend(self._inspect_file_recursive(Path(header), class_map))
        return results

    # ------------------------------------------------------------------
    def _build_clang_args(self, header_path: Path) -> list[str]:
        """Build compiler arguments for libclang.

        Args:
            header_path: Path to the header currently being parsed.

        Returns:
            List of command-line arguments.
        """
        include_paths = {
            header_path.resolve().parent,
            Path(".").resolve(),
            *self._include_dirs,
        }
        resource_dir = os.environ.get("CLANG_RESOURCE_DIR")
        if not resource_dir:
            try:
                resource_dir = subprocess.check_output(
                    ["clang", "-print-resource-dir"], text=True
                ).strip()
            except Exception:
                resource_dir = None
        if resource_dir:
            clang_include = Path(resource_dir) / "include"
            if clang_include.exists():
                include_paths.add(clang_include.resolve())
        args: list[str] = ["-x", "c++", "-std=c++17"]
        for include in sorted(include_paths):
            args.extend(["-I", str(include)])
        return args

    # ------------------------------------------------------------------
    def _inspect_file_recursive(
        self, header_path: Path, class_map: dict[str, ClassMetadata]
    ) -> list[ClassMetadata]:
        """Inspect a header and its companion implementation file.

        Args:
            header_path: Header file to inspect.
            class_map: Cache of already processed classes.

        Returns:
            List of discovered class metadata entries.
        """
        args = self._build_clang_args(header_path)
        translation_unit = self._index.parse(str(header_path), args=args)
        classes = self._find_classes_in_translation_unit(
            translation_unit.cursor, header_path
        )
        members_map: dict[str, dict[str, str]] = {}
        for cls in classes:
            members_map[cls.spelling] = self._collect_class_members(cls)
        includes = [
            inc.include.name for inc in translation_unit.get_includes() if inc.include
        ]
        cpp_file = header_path.with_suffix(".cpp")
        if cpp_file.exists():
            return self._parse_cpp_macros(cpp_file, members_map, includes, class_map)
        return []

    # ------------------------------------------------------------------
    def _collect_class_members(
        self, cursor: Cursor, visited: set[str] | None = None
    ) -> dict[str, str]:
        """Collect field declarations, following inheritance recursively.

        Args:
            cursor: Class cursor to analyse.
            visited: Optional set of visited USRs to prevent cycles.

        Returns:
            Mapping from member names to C++ type strings.
        """
        visited = visited or set()
        usr = (
            cursor.get_usr()
            or f"{cursor.spelling}@{cursor.location.file}:{cursor.location.line}"
        )
        if usr in visited:
            return {}
        visited.add(usr)
        members: dict[str, str] = {}
        for child in cursor.get_children():
            if child.kind == CursorKind.FIELD_DECL:
                members[child.spelling] = child.type.spelling
            elif child.kind == CursorKind.CXX_BASE_SPECIFIER:
                referenced = child.referenced
                base_definition = referenced.get_definition() if referenced else None
                if (
                    base_definition is None
                    and referenced
                    and referenced.kind == CursorKind.CLASS_DECL
                ):
                    base_definition = referenced
                if base_definition:
                    members.update(
                        self._collect_class_members(base_definition, visited)
                    )
        return members

    # ------------------------------------------------------------------
    def _find_classes_in_translation_unit(
        self, root: Cursor, header_path: Path
    ) -> list[Cursor]:
        """Locate classes defined in ``header_path`` within the translation unit.

        Args:
            root: Root cursor of the translation unit.
            header_path: Path of the header being scanned.

        Returns:
            List of class cursors owned by ``header_path``.
        """
        target = header_path.resolve()
        result: list[Cursor] = []

        def walk(cursor: Cursor) -> None:
            if cursor.kind == CursorKind.CLASS_DECL and cursor.is_definition():
                location = cursor.location
                if location.file and Path(location.file.name).resolve() == target:
                    result.append(cursor)
            for child in cursor.get_children():
                walk(child)

        walk(root)
        return result

    # ------------------------------------------------------------------
    def _parse_cpp_macros(
        self,
        cpp_path: Path,
        members_map: Mapping[str, Mapping[str, str]],
        includes: Sequence[str],
        class_map: dict[str, ClassMetadata],
    ) -> list[ClassMetadata]:
        """Parse ``BEGIN_FECORE_CLASS`` blocks from a companion source file.

        Args:
            cpp_path: Path to the implementation file.
            members_map: Mapping from class names to member/type dictionaries.
            includes: Collection of include directives discovered during parsing.
            class_map: Cache of already processed classes.

        Returns:
            List of :class:`ClassMetadata` entries discovered in the source file.
        """
        text = cpp_path.read_text(errors="ignore")
        blocks: list[ClassMetadata] = []
        pattern = re.compile(
            r"BEGIN_FECORE_CLASS\s*\(\s*([A-Za-z0-9_:]+)\s*,\s*([A-Za-z0-9_:]+)\s*\)"
        )
        register_pattern = re.compile(
            r"REGISTER_FECORE_CLASS\s*\(\s*([A-Za-z0-9_:]+)\s*,\s*\"((?:\\.|[^\"])*)\"\s*\)"
        )
        register_map = {
            match.group(1): _decode_c_string(match.group(2))
            for match in register_pattern.finditer(text)
        }
        global_registration = _load_global_registration_map()
        local_includes = list(includes)
        local_includes.append(str(cpp_path))
        header_candidate = cpp_path.with_suffix(".h")
        if header_candidate.exists():
            local_includes.append(str(header_candidate))
        for match in pattern.finditer(text):
            class_name, base_class = match.groups()
            if class_name in class_map:
                continue
            end_match = re.search(r"END_FECORE_CLASS", text[match.end() :])
            if not end_match:
                continue
            block_text = text[match.end() : match.end() + end_match.start()]
            clean_block = _remove_cpp_comments(block_text)
            lines = [line.strip() for line in clean_block.splitlines() if line.strip()]
            parameters: list[ParameterInfo] = []
            members = members_map.get(class_name, {})
            combined_defaults = _extract_constructor_defaults(text, class_name)
            for member, value in _collect_base_defaults(base_class, class_map).items():
                combined_defaults.setdefault(member, value)
            for line in lines:
                if "ADD_PARAMETER" in line or "ADD_PROPERTY" in line:
                    parsed = self._parse_macro_line(
                        line, members, local_includes, class_map, combined_defaults
                    )
                    if parsed:
                        parameters.append(parsed)
            registration = register_map.get(class_name) or global_registration.get(
                class_name
            )
            if registration is None:
                print(
                    f"[classExposer] WARNING: No registration found for {class_name}",
                    flush=True,
                )
            else:
                print(
                    f"[classExposer] Using registration '{registration}' for {class_name}",
                    flush=True,
                )
            metadata = ClassMetadata(
                class_name=class_name,
                base_class=base_class,
                registration=registration,
                params=parameters,
            )
            blocks.append(metadata)
            class_map[class_name] = metadata
        return blocks

    # ------------------------------------------------------------------
    def _parse_macro_line(
        self,
        line: str,
        members: Mapping[str, str],
        includes: Sequence[str],
        class_map: dict[str, ClassMetadata],
        defaults: Mapping[str, int | float | str | bool] | None,
    ) -> ParameterInfo | None:
        """Parse individual macro invocations inside a block.

        Args:
            line: Raw line containing the macro invocation.
            members: Mapping from member names to C++ types.
            includes: Include paths associated with the translation unit.
            class_map: Cache of processed classes.

        Returns:
            Parameter info or ``None`` when the line does not represent a
            supported macro invocation.
        """
        chain_calls: list[MacroChainCall] = []
        prefix_match = re.match(r"(ADD_PARAMETER|ADD_PROPERTY)\s*\(", line)
        if not prefix_match:
            return None
        macro = prefix_match.group(1)
        open_idx = prefix_match.end() - 1
        close_idx = _find_matching_paren(line, open_idx)
        if close_idx == -1:
            return None
        arguments_blob = line[open_idx + 1 : close_idx]
        postamble = line[close_idx + 1 :].strip().rstrip(";")
        while postamble.startswith("->"):
            postamble = postamble[2:].strip()
            name_match = re.match(r"([A-Za-z_][A-Za-z0-9_]*)", postamble)
            if not name_match:
                break
            func = name_match.group(1)
            open_idx = postamble.find("(", name_match.end(0))
            if open_idx == -1:
                break
            close_idx = _find_matching_paren(postamble, open_idx)
            if close_idx == -1:
                break
            args_blob = postamble[open_idx + 1 : close_idx]
            chain_calls.append(
                MacroChainCall(
                    func=func,
                    args=[arg.strip() for arg in _split_args(args_blob)],
                )
            )
            postamble = postamble[close_idx + 1 :].strip().lstrip(";").strip()

        arguments = [_arg.strip() for _arg in _split_args(arguments_blob)]
        member = arguments[0].strip() if arguments else None
        ctype = members.get(member) if member else None
        nested_param: ParameterInfo | None = None
        if member and "->" in member:
            base_member, attr_member = [part.strip() for part in member.split("->", 1)]
            nested_param = self._resolve_nested_parameter(
                members.get(base_member), attr_member, includes, class_map
            )
            if nested_param and ctype is None:
                ctype = nested_param.ctype

        default: int | float | str | bool | None = None
        strings: list[str] = []
        range_info: RangeDict | str | None = None
        human_name: str | None = None
        selector: str | None = None
        enum_values: list[str] | None = None
        long_name: str | None = None
        units: str | None = None
        hidden = False

        if macro == "ADD_PARAMETER":
            if member and defaults:
                default = defaults.get(member)
            if default is None and nested_param:
                default = nested_param.default
            for argument in arguments[1:]:
                if "FE_RANGE_" in argument:
                    parsed_range = _parse_fe_range(argument, argument.find("FE_RANGE_"))
                    if parsed_range is not None:
                        range_info = parsed_range
                    continue
                if argument.startswith('"') and argument.endswith('"'):
                    literal = argument.strip('"')
                    strings.append(literal)
                    continue
            if strings:
                human_name = strings[0]
            if not strings and nested_param and nested_param.name:
                human_name = nested_param.name
                strings.append(nested_param.name)
            for literal in strings[1:]:
                decoded_literal = _decode_c_string(literal)
                if "\0" in decoded_literal:
                    tokens = [token for token in decoded_literal.split("\0") if token]
                    if tokens:
                        enum_values = tokens
                    continue
                if literal.startswith("$(") and literal.endswith(")"):
                    selector = literal
                    if range_info is None:
                        range_info = selector
                    break
            if selector:
                resolved = _resolve_dof_selector(selector)
                if resolved:
                    enum_values = resolved
            for call in chain_calls:
                if call.func == "setLongName" and call.args:
                    long_name = call.args[0].strip('"')
                if call.func == "setUnits" and call.args:
                    units = call.args[0]
            if nested_param:
                if nested_param.range and range_info is None:
                    range_info = nested_param.range
                nested_enum = nested_param.enum
                if nested_enum and not enum_values:
                    enum_values = (
                        nested_enum if isinstance(nested_enum, list) else [nested_enum]
                    )
                if nested_param.long_name and not long_name:
                    long_name = nested_param.long_name
                if nested_param.units and not units:
                    units = nested_param.units
            if (
                not enum_values
                and isinstance(default, str)
                and "::" in default
            ):
                enum_name = default.split("::", 1)[0].split()[-1]
                tokens = self._resolve_enum_tokens(enum_name, includes, class_map)
                if tokens:
                    enum_values = tokens
        elif macro == "ADD_PROPERTY":
            for argument in arguments[1:]:
                if argument.startswith('"') and argument.endswith('"'):
                    strings.append(argument.strip('"'))
            if strings:
                human_name = strings[0]
        definition = None
        if macro == "ADD_PROPERTY" and member:
            definition = self._resolve_property_definition(
                member, members.get(member), includes, class_map
            )

        return ParameterInfo(
            macro=macro,
            args=arguments,
            member=member,
            ctype=ctype,
            chain=chain_calls,
            default=default,
            strings=strings,
            range=range_info,
            name=human_name,
            enum=enum_values if enum_values else selector,
            long_name=long_name,
            units=units,
            definition=definition,
            hidden=hidden,
        )

    # ------------------------------------------------------------------
    def _resolve_property_definition(
        self,
        member: str,
        member_type: str | None,
        includes: Sequence[str],
        class_map: dict[str, ClassMetadata],
    ) -> list[dict[str, Any]] | None:
        """Resolve nested property definitions for ``ADD_PROPERTY`` macros.

        Args:
            member: Name of the member field in the owning class.
            member_type: C++ type of the member.
            includes: Include directives associated with the translation unit.
            class_map: Cache used to prevent duplicate parsing.

        Returns:
            Nested metadata or ``None`` when nothing could be resolved.
        """
        if not member_type:
            return None
        clean = (
            member_type.replace("*", "")
            .replace("&", "")
            .replace("const", "")
            .replace("class", "")
            .strip()
        )
        for include in includes:
            include_path = Path(include)
            if include_path.stem == clean and include_path.exists():
                extracted = self._inspect_file_recursive(include_path, class_map)
                return [entry.to_json() for entry in extracted]
        return None

    def _resolve_nested_parameter(
        self,
        pointer_type: str | None,
        attr_member: str,
        includes: Sequence[str],
        class_map: dict[str, ClassMetadata],
    ) -> ParameterInfo | None:
        if not pointer_type or not attr_member:
            return None
        target_class = _normalise_cpp_class_name(pointer_type)
        if not target_class:
            return None
        metadata = class_map.get(target_class)
        header_path: Path | None = None
        if metadata is None:
            for include in includes:
                include_path = Path(include)
                if include_path.stem != target_class or not include_path.exists():
                    continue
                header_path = include_path
                extracted = self._inspect_file_recursive(include_path, class_map)
                for entry in extracted:
                    class_map.setdefault(entry.class_name, entry)
                metadata = class_map.get(target_class)
                if metadata:
                    break
        if metadata is None:
            try_paths = list(FEBIO_ROOT.rglob(f"{target_class}.h"))
            for candidate in try_paths:
                header_path = candidate
                extracted = self._inspect_file_recursive(candidate, class_map)
                for entry in extracted:
                    class_map.setdefault(entry.class_name, entry)
                metadata = class_map.get(target_class)
                if metadata:
                    break
        if metadata is not None:
            for param in metadata.params:
                if param.member == attr_member:
                    return param

        header_ctype = None
        if header_path and header_path.exists():
            text = header_path.read_text(errors="ignore")
            pattern = re.compile(rf"([A-Za-z0-9_:<>]+)\s+{re.escape(attr_member)}\s*;")
            match = pattern.search(text)
            if match:
                header_ctype = match.group(1)

        try_paths_cpp = list(FEBIO_ROOT.rglob(f"{target_class}.cpp"))
        defaults_map: dict[str, Any] = {}
        for candidate in try_paths_cpp:
            text = candidate.read_text(errors="ignore")
            defaults_map.update(_extract_constructor_defaults(text, target_class))
        default_value = defaults_map.get(attr_member)
        if default_value is None and not header_ctype:
            return None
        return ParameterInfo(
            macro="ADD_PARAMETER",
            args=[],
            member=attr_member,
            ctype=header_ctype,
            chain=[],
            default=default_value,
            strings=[attr_member],
            range=None,
            name=attr_member,
            enum=None,
            long_name=None,
            units=None,
            definition=None,
        )

    def _resolve_enum_tokens(
        self, enum_name: str, includes: Sequence[str], class_map: dict[str, ClassMetadata]
    ) -> list[str] | None:
        for include in includes:
            include_path = Path(include)
            if include_path.exists():
                try:
                    tokens = _extract_enum_tokens_from_text(
                        enum_name, include_path.read_text(errors="ignore")
                    )
                except OSError:
                    tokens = None
                if tokens:
                    return tokens
        try_paths = list(FEBIO_ROOT.rglob(f"{enum_name}.h"))
        for candidate in try_paths:
            try:
                tokens = _extract_enum_tokens_from_text(
                    enum_name, candidate.read_text(errors="ignore")
                )
            except OSError:
                tokens = None
            if tokens:
                return tokens
        return None


def _dump_metadata(
    metadata: Sequence[ClassMetadata], output: Path | None, *, pretty: bool
) -> None:
    """Serialise metadata to stdout or JSON.

    Args:
        metadata: Extracted class metadata.
        output: Optional output path. When ``None`` the JSON is printed.
        pretty: Flag to enable indentation.
    """

    serialised = [entry.to_json() for entry in metadata]
    text = json.dumps(serialised, indent=2 if pretty else None)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    else:
        print(text)


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """

    parser = argparse.ArgumentParser(
        description="Extract FEBio metadata from header files"
    )
    parser.add_argument("headers", nargs="+", help="FEBio header files to inspect")
    parser.add_argument("-o", "--output", help="Optional file to store JSON output")
    parser.add_argument("--dir", help="Directory to store per-header JSON output")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output"
    )
    parser.add_argument(
        "--manifest-dir", help="Directory containing manifest JSON files to update"
    )
    parser.add_argument(
        "--category", help="Manifest category name when suggesting entries"
    )
    parser.add_argument("--xml-tag", help="Override XML tag stored in manifest entries")
    parser.add_argument(
        "--xml-section", help="Override XML section stored in manifest entries"
    )
    parser.add_argument(
        "--manifest-dry-run",
        action="store_true",
        help="Preview manifest updates without writing to disk",
    )
    parser.add_argument(
        "-I",
        "--include",
        dest="includes",
        action="append",
        default=[],
        help="Additional include directories forwarded to clang",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list for testing purposes.

    Returns:
        Exit status code.
    """

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.dir and args.output:
        parser.error("--dir and --output cannot be used together")
    if args.manifest_dir and not args.category:
        parser.error("--category is required when --manifest-dir is provided")

    extractor = FEBioExtractor(include_dirs=[Path(p) for p in args.includes])
    header_paths = [Path(header) for header in args.headers]

    target_dir = Path(args.dir) if args.dir else None
    if target_dir:
        target_dir.mkdir(parents=True, exist_ok=True)

    all_metadata: list[ClassMetadata] = []
    for header_path in header_paths:
        metadata = extractor.extract([header_path])
        if not metadata:
            print(f"[extractor] No FEBio classes discovered in {header_path}")
            continue
        all_metadata.extend(metadata)
        _maybe_process_manifest(metadata, header_path, args)
        if target_dir:
            output_path = target_dir / f"{header_path.stem}.json"
            _dump_metadata(metadata, output_path, pretty=True)

    if target_dir:
        return 0

    if not all_metadata:
        print("[extractor] No FEBio metadata produced.")
        return 0

    output_path = Path(args.output) if args.output else None
    _dump_metadata(all_metadata, output_path, pretty=args.pretty)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
