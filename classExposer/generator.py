"""Generate typed Python bindings for FEBio entities.

This module consumes the JSON metadata emitted by :mod:`codegen.extractor`
alongside a user-provided manifest to create high-level dataclasses capable of
serialising themselves to FEBio XML.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TYPE_MAP: dict[str, str] = {
    "int": "int",
    "bool": "bool",
    "double": "float",
    "float": "float",
    "FEParamDouble": "float",
    "std::string": "str",
}


VALUE_TYPE_MAP: dict[str, dict[str, Any]] = {
    "vec3d": {
        "python_name": "Vec3d",
        "fields": ("x", "y", "z"),
        "field_type": "float",
    },
}


CORE_HELPER_IDENTIFIERS: set[str] = {
    "Vec3d",
    "mat3d",
    "mat3ds",
    "MeshMappedValue",
    "FEParamDouble",
    "FEParamVec3",
    "FEParamMat3d",
    "FEParamMat3ds",
}


PARAM_FLAG_MAP = {
    "FE_PARAM_HIDDEN": 0x04,
}


ROOT_PACKAGE = "interFEBio"


_CATEGORY_INJECTIONS: dict[str, dict[str, Any]] = {
    "Core": {
        "imports": (
            f"from {ROOT_PACKAGE}.Core.Parameters import (\n"
            "    FEParamDouble,\n"
            "    FEParamVec3,\n"
            "    FEParamMat3d,\n"
            "    FEParamMat3ds,\n"
            "    MeshMappedValue,\n"
            "    Vec3d,\n"
            "    mat3d,\n"
            "    mat3ds,\n"
            ")\n\n"
        ),
        "exports": [
            "Vec3d",
            "mat3d",
            "mat3ds",
            "MeshMappedValue",
            "FEParamDouble",
            "FEParamVec3",
            "FEParamMat3d",
            "FEParamMat3ds",
        ],
        "stub_exclude": {
            "Vec3d",
            "mat3d",
            "mat3ds",
            "MeshMappedValue",
            "FEParamDouble",
            "FEParamVec3",
            "FEParamMat3d",
            "FEParamMat3ds",
        },
        "skip_value_types": True,
    },
    "BoundaryConditions": {
        "imports": (
            f"from {ROOT_PACKAGE}.Core.Parameters import Vec3d\n"
            f"from {ROOT_PACKAGE}.common.regions import NodeSetRef, SurfaceRef, coerce_nodeset, coerce_surface\n\n"
        ),
        "exports": [
            "Vec3d",
            "NodeSetRef",
            "SurfaceRef",
            "FENodalBC",
            "FEFixedBC",
            "FEPrescribedNodeSet",
            "FEPrescribedSurface",
        ],
        "stub_exclude": {
            "Vec3d",
            "FENodalBC",
            "FEFixedBC",
            "FEPrescribedNodeSet",
            "FEPrescribedSurface",
        },
        "skip_value_types": True,
        "prefix": (
            "@dataclass(kw_only=True)\n"
            "class FENodalBC(FEBoundaryCondition):\n"
            "    node_set: Optional[NodeSetRef] = field(\n"
            "        default=None, metadata={'fe_name': 'node_set', 'is_property': True}\n"
            "    )\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if self.node_set is not None:\n"
            "            self.node_set = coerce_nodeset(self.node_set)\n\n"
            "class FEFixedBC(FENodalBC):\n"
            "    pass\n\n"
            "class FEPrescribedNodeSet(FENodalBC):\n"
            "    pass\n\n"
            "@dataclass(kw_only=True)\n"
            "class FEPrescribedSurface(FEBoundaryCondition):\n"
            "    surface: Optional[SurfaceRef] = field(\n"
            "        default=None, metadata={'fe_name': 'surface', 'is_property': True}\n"
            "    )\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if self.surface is not None:\n"
            "            self.surface = coerce_surface(self.surface)\n\n"
            "\n"
        ),
    },
    "Loads": {
        "imports": (
            f"from {ROOT_PACKAGE}.Core.Parameters import Vec3d, FEParamVec3\n"
            f"from {ROOT_PACKAGE}.common.regions import (\n"
            "    NodeSetRef,\n"
            "    SurfaceRef,\n"
            "    ElementSetRef,\n"
            "    coerce_nodeset,\n"
            "    coerce_surface,\n"
            "    coerce_elementset,\n"
            ")\n\n"
        ),
        "exports": [
            "Vec3d",
            "NodeSetRef",
            "SurfaceRef",
            "ElementSetRef",
            "FENodalLoad",
            "FESurfaceLoad",
            "FEBodyLoad",
            "FEBodyForce",
        ],
        "stub_exclude": {
            "Vec3d",
            "FENodalLoad",
            "FESurfaceLoad",
            "FEBodyLoad",
            "FEBodyForce",
        },
        "skip_value_types": True,
        "prefix": (
            "class FEModelLoad(FEBioEntity):\n"
            "    pass\n\n"
            "@dataclass(kw_only=True)\n"
            "class FENodalLoad(FEModelLoad):\n"
            "    node_set: Optional[NodeSetRef] = field(\n"
            "        default=None, metadata={'fe_name': 'node_set', 'is_property': True}\n"
            "    )\n"
            "    relative: Optional[bool] = field(default=None, metadata={'fe_name': 'relative'})\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if self.node_set is not None:\n"
            "            self.node_set = coerce_nodeset(self.node_set)\n\n"
            "@dataclass(kw_only=True)\n"
            "class FESurfaceLoad(FEModelLoad):\n"
            "    surface: Optional[SurfaceRef] = field(\n"
            "        default=None, metadata={'fe_name': 'surface', 'is_property': True}\n"
            "    )\n"
            "    relative: Optional[bool] = field(default=None, metadata={'fe_name': 'relative'})\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if self.surface is not None:\n"
            "            self.surface = coerce_surface(self.surface)\n\n"
            "@dataclass(kw_only=True)\n"
            "class FEBodyLoad(FEModelLoad):\n"
            "    element_set: Optional[ElementSetRef] = field(\n"
            "        default=None, metadata={'fe_name': 'domain', 'is_property': True}\n"
            "    )\n"
            "    relative: Optional[bool] = field(default=None, metadata={'fe_name': 'relative'})\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if self.element_set is not None:\n"
            "            self.element_set = coerce_elementset(self.element_set)\n\n"
            "class FEBodyForce(FEBodyLoad):\n"
            "    pass\n\n"
            "\n"
        ),
    },
    "Contact": {
        "imports": (
            f"from {ROOT_PACKAGE}.common.regions import SurfaceRef, coerce_surface\n\n"
        ),
        "exports": ["SurfaceRef", "FEContactInterface"],
        "stub_exclude": {"SurfaceRef", "FEContactInterface"},
        "skip_value_types": True,
        "prefix": (
            "@dataclass(kw_only=True)\n"
            "class FEContactInterface(FEBioEntity):\n"
            "    primary: Optional[SurfaceRef] = field(\n"
            "        default=None, metadata={'fe_name': 'primary', 'is_property': True}\n"
            "    )\n"
            "    secondary: Optional[SurfaceRef] = field(\n"
            "        default=None, metadata={'fe_name': 'secondary', 'is_property': True}\n"
            "    )\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if self.primary is not None:\n"
            "            self.primary = coerce_surface(self.primary)\n"
            "        if self.secondary is not None:\n"
            "            self.secondary = coerce_surface(self.secondary)\n\n"
            "\n"
        ),
    },
    "Constraints": {
        "imports": (
            f"from {ROOT_PACKAGE}.common.regions import SurfaceRef, coerce_surface\n\n"
        ),
        "exports": ["SurfaceRef", "FESurfaceConstraint", "FEPrescribedSurface"],
        "stub_exclude": {"SurfaceRef", "FESurfaceConstraint", "FEPrescribedSurface"},
        "skip_value_types": True,
        "prefix": (
            "@dataclass(kw_only=True)\n"
            "class FESurfaceConstraint(FEBioEntity):\n"
            "    surface: Optional[SurfaceRef] = field(\n"
            "        default=None, metadata={'fe_name': 'surface', 'is_property': True}\n"
            "    )\n\n"
            "    def __post_init__(self) -> None:\n"
            "        if self.surface is not None:\n"
            "            self.surface = coerce_surface(self.surface)\n\n"
            "@dataclass(kw_only=True)\n"
            "class FEPrescribedSurface(FESurfaceConstraint):\n"
            "    pass\n\n"
            "\n"
        ),
    },
}


CLASS_EXTENSIONS: dict[str, dict[str, Any]] = {}


_ARRAY_CTYPE_PATTERN = re.compile(
    r"(?P<base>[A-Za-z_][A-Za-z0-9_:]*)\s*\[\s*(?P<length>\d+)\s*\]"
)


def _normalise_cpp_class(ctype: str | None) -> str | None:
    """Return the canonical C++ class identifier extracted from ``ctype``."""

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
    identifier = tokens[-1]
    return identifier.split("::")[-1]


def _parse_array_ctype(ctype: str | None) -> tuple[str, int] | None:
    if not ctype:
        return None
    match = _ARRAY_CTYPE_PATTERN.fullmatch(ctype.strip())
    if not match:
        return None
    return match.group("base"), int(match.group("length"))


def _format_array_type(base: str, length: int) -> str:
    if length <= 0:
        return "tuple[]"
    repeated = ", ".join(base for _ in range(length))
    return f"tuple[{repeated}]"


def _extract_flag_bits(tokens: Sequence[Any]) -> int:
    total = 0
    for token in tokens:
        for part in str(token).split("|"):
            entry = part.strip()
            if not entry:
                continue
            try:
                total |= int(entry, 0)
                continue
            except ValueError:
                pass
            name = entry.split("::")[-1]
            bit = PARAM_FLAG_MAP.get(name)
            if bit is not None:
                total |= bit
    return total


def _has_hidden_flag(param: Mapping[str, Any]) -> bool:
    for call in param.get("chain", []):
        if call.get("func") != "SetFlags":
            continue
        if _extract_flag_bits(call.get("args", [])) & PARAM_FLAG_MAP.get(
            "FE_PARAM_HIDDEN", 0
        ):
            return True
    return False


def _format_metadata_default(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        parts = [str(item) for item in value if item is not None]
        return ", ".join(parts)
    return value


def _infer_numeric_python_type(
    range_spec: dict[str, Any] | None,
    raw_default: Any,
    coerced_default: Any,
) -> str | None:
    candidates: list[Any] = []

    def _collect(value: Any) -> None:
        if isinstance(value, (int, float)):
            candidates.append(value)
        elif isinstance(value, str):
            value = value.strip()
            try:
                if "." in value or "e" in value.lower():
                    candidates.append(float(value))
                else:
                    candidates.append(int(value))
            except ValueError:
                pass

    if range_spec and isinstance(range_spec, dict):
        for key in (
            "min",
            "max",
            "not_equal",
            "min_expr",
            "max_expr",
            "not_equal_expr",
        ):
            _collect(range_spec.get(key))
    _collect(raw_default)
    if coerced_default is not _DEFAULT_SENTINEL:
        _collect(coerced_default)

    if not candidates:
        return None
    if any(
        isinstance(value, float) and not float(value).is_integer()
        for value in candidates
    ):
        return "float"
    if any(isinstance(value, float) for value in candidates):
        return "float"
    return "int"


def _map_primitive_type(ctype: str | None) -> str | None:
    """Return primitive Python type match for ``ctype`` when available."""

    if not ctype:
        return None
    for key, python_type in TYPE_MAP.items():
        if key in ctype:
            return python_type
    return None


def _normalise_identifier(raw: str | None, fallback: str) -> str:
    """Produce a valid Python identifier from FEBio metadata.

    Args:
        raw: Preferred identifier extracted from the metadata.
        fallback: Safe fallback name when ``raw`` is missing or invalid.

    Returns:
        Sanitised snake_case identifier.
    """

    target = raw or fallback
    if target.startswith("m_"):
        target = target[2:]
    target = target.replace("::", "_")
    target = re.sub(r"[^0-9a-zA-Z_]", "_", target)
    target = re.sub(r"_{2,}", "_", target)
    target = target.strip("_")
    target = target or fallback
    if target[0].isdigit():
        target = f"param_{target}"
    return target.lower()


def _coerce_python_default(value: Any) -> Any:
    """Return a safe Python literal for ``value`` or ``_DEFAULT_SENTINEL``."""

    if value is None:
        return _DEFAULT_SENTINEL
    if isinstance(value, (list, tuple)):
        coerced: list[Any] = []
        for item in value:
            if isinstance(item, (int, float, bool)) or item is None:
                coerced.append(item)
            elif isinstance(item, str):
                coerced.append(item)
            else:
                return _DEFAULT_SENTINEL
        return tuple(coerced)
    if isinstance(value, (int, float, bool)):
        return value
    if isinstance(value, str):
        if not value:
            return ""
        if re.search(r"[(){}=]", value):
            return _DEFAULT_SENTINEL
        return value
    return _DEFAULT_SENTINEL


def _extract_enum_values(param: Mapping[str, Any]) -> list[str] | None:
    """Return the list of enum tokens encoded via ``setEnums`` or metadata."""

    enum_meta = param.get("enum")
    if isinstance(enum_meta, list):
        filtered = [
            token
            for token in enum_meta
            if isinstance(token, str) and not token.startswith("$(")
        ]
        return filtered or None
    if isinstance(enum_meta, str):
        return None if enum_meta.startswith("$(") else [enum_meta]
    for call in param.get("chain", []):
        if call.get("func") != "setEnums":
            continue
        for raw_arg in call.get("args", []):
            text = raw_arg.strip()
            if text.startswith('"') and text.endswith('"') and len(text) >= 2:
                text = text[1:-1]
            try:
                decoded = bytes(text, encoding="utf-8").decode("unicode_escape")
            except Exception:
                decoded = text
            tokens = [token for token in decoded.split("\0") if token]
            tokens = [token for token in tokens if not token.startswith("$(")]
            if tokens:
                return tokens
    return None


def _pascal_case(identifier: str) -> str:
    """Convert a snake_case or mixed identifier to PascalCase."""

    parts = re.split(r"[^0-9a-zA-Z]+", identifier)
    cleaned = [part for part in parts if part]
    if not cleaned:
        return identifier.title()
    token = "".join(part[0].upper() + part[1:] for part in cleaned)
    if token and token[0].isdigit():
        token = f"Value{token}"
    return token or "Enum"


def _normalise_enum_member_name(token: str) -> str:
    """Return a valid Python identifier for an enum member from ``token``."""

    name = re.sub(r"[^0-9a-zA-Z_]+", "_", token).upper()
    name = re.sub(r"_{2,}", "_", name).strip("_") or "VALUE"
    if name[0].isdigit():
        name = f"VALUE_{name}"
    return name


def _build_enum_members(
    tokens: Sequence[str],
    base_type: str,
) -> list[EnumMember]:
    """Translate FEBio enum token strings into EnumMember objects."""

    members: list[EnumMember] = []
    next_value: int = 0
    for raw in tokens:
        token = raw
        value: Any
        explicit = False
        if "=" in raw:
            token, value_part = raw.split("=", 1)
            explicit = True
            value_part = value_part.strip()
            if base_type == "int":
                try:
                    value = int(value_part, 0)
                except ValueError:
                    value = value_part
            elif base_type == "float":
                try:
                    value = float(value_part)
                except ValueError:
                    value = value_part
            else:
                value = value_part
        else:
            if base_type == "int":
                value = next_value
            elif base_type == "float":
                value = float(next_value)
            else:
                value = token
        name = _normalise_enum_member_name(token)
        members.append(EnumMember(token=token, name=name, value=value))
        if base_type == "int" and not explicit:
            next_value = int(value) + 1 if isinstance(value, int) else next_value + 1
        elif base_type == "float" and not explicit:
            next_value += 1
    return members


def _match_enum_default(
    raw_default: Any,
    coerced_default: Any,
    members: Sequence[EnumMember],
) -> EnumMember | None:
    """Try to match ``raw_default`` or ``coerced_default`` to an enum member."""

    if not members:
        return None
    if coerced_default is not _DEFAULT_SENTINEL:
        for member in members:
            if member.value == coerced_default:
                return member
    if isinstance(raw_default, str):
        candidate = raw_default.strip()
        if candidate:
            variations = {
                candidate,
                candidate.upper(),
                candidate.lower(),
            }
            if candidate.upper().startswith("FE_"):
                variations.add(candidate[3:])
                variations.add(candidate.upper()[3:])
            if candidate.upper().startswith("FEBIO_"):
                variations.add(candidate[6:])
                variations.add(candidate.upper()[6:])
            variations.add(candidate.split("::")[-1])
            variations = {v.replace("::", "_") for v in variations if v}
            for member in members:
                if member.token in variations or member.name in variations:
                    return member
    return None


def _evaluate_numeric_expr(expr: str) -> Fraction | float | int | None:
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return None

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            operand = _eval(node.operand)
            return +operand if isinstance(node.op, ast.UAdd) else -operand
        if isinstance(node, ast.BinOp) and isinstance(
            node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)
        ):
            left = _eval(node.left)
            right = _eval(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise ZeroDivisionError
                return left / right
        raise ValueError

    try:
        result = _eval(node)
    except ValueError:
        return None
    except ZeroDivisionError:
        return None
    if isinstance(result, Fraction):
        if result.denominator == 1:
            return result.numerator
        return float(result)
    return result


def _format_range_value(value: Any) -> str:
    if isinstance(value, (int, float)):
        text = repr(value)
        if text.endswith(".0"):
            text = text[:-2]
        return text
    if isinstance(value, str):
        evaluated = _evaluate_numeric_expr(value)
        if evaluated is not None:
            return _format_range_value(evaluated)
    return str(value)


def _coerce_range_number(value: Any) -> float | int | None:
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str):
        evaluated = _evaluate_numeric_expr(value)
        if evaluated is None:
            return None
        if isinstance(evaluated, (int, float)):
            return evaluated
    return None


def _build_range_annotation_payload(spec: Any) -> str | None:
    if not isinstance(spec, dict):
        return None
    parts: list[str] = []
    min_value = _coerce_range_number(spec.get("min"))
    if min_value is not None:
        parts.append(f"min={min_value!r}")
        parts.append(f"min_inclusive={spec.get('min_inclusive', True)!r}")
    elif spec.get("min") is not None:
        parts.append(f"min_expr={spec['min']!r}")
        parts.append(f"min_inclusive={spec.get('min_inclusive', True)!r}")
    max_value = _coerce_range_number(spec.get("max"))
    if max_value is not None:
        parts.append(f"max={max_value!r}")
        parts.append(f"max_inclusive={spec.get('max_inclusive', True)!r}")
    elif spec.get("max") is not None:
        parts.append(f"max_expr={spec['max']!r}")
        parts.append(f"max_inclusive={spec.get('max_inclusive', True)!r}")
    not_equal = _coerce_range_number(spec.get("not_equal"))
    if not_equal is not None:
        parts.append(f"not_equal={not_equal!r}")
    elif spec.get("not_equal") is not None:
        parts.append(f"not_equal_expr={spec['not_equal']!r}")
    raw = spec.get("raw_args")
    if raw is not None:
        parts.append(f"raw={raw!r}")
    if not parts:
        return None
    args = ", ".join(parts)
    return f"RangeSpec({args})"


def _format_range_predicate(spec: Any) -> str | None:
    if not isinstance(spec, dict):
        return None
    min_val = spec.get("min")
    max_val = spec.get("max")
    not_equal = spec.get("not_equal")
    parts: list[str] = []
    if min_val is not None and max_val is not None:
        left = "<=" if spec.get("min_inclusive", True) else "<"
        right = "<=" if spec.get("max_inclusive", True) else "<"
        return f"{_format_range_value(min_val)} {left} value {right} {_format_range_value(max_val)}"
    if min_val is not None:
        op = ">=" if spec.get("min_inclusive", True) else ">"
        parts.append(f"value {op} {_format_range_value(min_val)}")
    if max_val is not None:
        op = "<=" if spec.get("max_inclusive", True) else "<"
        parts.append(f"value {op} {_format_range_value(max_val)}")
    if not_equal is not None:
        parts.append(f"value != {_format_range_value(not_equal)}")
    raw = spec.get("raw_args")
    if raw is not None and not parts:
        parts.append(f"range({raw})")
    return " and ".join(parts) if parts else None


_IDENTIFIER_REGEX = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _identify_core_helpers(type_expression: str) -> set[str]:
    helpers: set[str] = set()
    for token in _IDENTIFIER_REGEX.findall(type_expression):
        if token in CORE_HELPER_IDENTIFIERS:
            helpers.add(token)
    return helpers


_DEFAULT_SENTINEL = object()


@dataclass(slots=True)
class ParameterDefinition:
    """Represents a Python field derived from a FEBio parameter."""

    attr_name: str
    type_hint: str
    metadata: dict[str, Any]
    default: Any = _DEFAULT_SENTINEL
    default_expr: str | None = None
    allow_none: bool = False

    def type_expression(self) -> str:
        base_type = self.type_hint if self.type_hint else "Any"
        if self.allow_none:
            if base_type == "Any":
                return "Optional[Any]"
            if "|" in base_type:
                return f"{base_type} | None"
            return f"Optional[{base_type}]"
        return base_type

    def has_default(self) -> bool:
        """Return ``True`` when the field carries a default value."""

        return (
            self.default is not _DEFAULT_SENTINEL
            or self.default_expr is not None
            or self.allow_none
        )

    def render(self) -> str:
        """Render the parameter as a dataclass field definition."""

        meta_for_render = dict(self.metadata)
        if "default" in meta_for_render:
            meta_for_render["default"] = _format_metadata_default(
                meta_for_render["default"]
            )
        metadata_repr = repr(meta_for_render)
        type_expression = self.type_expression()
        if self.default_expr is not None:
            default_repr = self.default_expr
        elif self.default is _DEFAULT_SENTINEL:
            if self.allow_none:
                return (
                    f"    {self.attr_name}: {type_expression} = field("
                    f"default=None, metadata={metadata_repr})"
                )
            return (
                f"    {self.attr_name}: {type_expression} = field("
                f"metadata={metadata_repr})"
            )
        else:
            default_repr = repr(self.default)
        return (
            f"    {self.attr_name}: {type_expression} = field("
            f"default={default_repr}, metadata={metadata_repr})"
        )


@dataclass(slots=True)
class EnumMember:
    """Single enumeration entry definition."""

    token: str
    name: str
    value: Any


@dataclass(slots=True)
class EnumDefinition:
    """Container describing a Python Enum to be emitted."""

    name: str
    members: list[EnumMember]


@dataclass(slots=True)
class ModuleRenderResult:
    """Container holding the rendered module text and its exports."""

    text: str
    exports: list[str]


@dataclass(slots=True)
class ManifestItem:
    """Single entry describing how to expose a FEBio class to Python."""

    category: str
    class_name: str
    python_name: str
    xml_tag: str


class MetadataRepository:
    """Utility wrapper to query metadata by class name."""

    def __init__(self, entries: Iterable[dict[str, Any]]) -> None:
        self._entries = {entry["class"]: entry for entry in entries}

    def get(self, class_name: str) -> dict[str, Any]:
        """Return the metadata for ``class_name``.

        Args:
            class_name: Name of the FEBio class.

        Returns:
            Metadata dictionary for the class.

        Raises:
            KeyError: When the class is not present in the repository.
        """

        return self._entries[class_name]

    def has(self, class_name: str) -> bool:
        """Return whether metadata is available for ``class_name``."""

        return class_name in self._entries


class TypeResolver:
    """Resolve FEBio C++ types into Python annotations and collect stubs."""

    def __init__(self, manifest_items: Sequence[ManifestItem]) -> None:
        self._cpp_to_python: dict[str, str] = {
            item.class_name: item.python_name for item in manifest_items
        }
        self._class_to_manifest: dict[str, ManifestItem] = {
            item.class_name: item for item in manifest_items
        }
        self._python_to_category: dict[str, str] = {
            item.python_name: item.category for item in manifest_items
        }
        self._stub_names: set[str] = set()
        self._value_types: set[str] = set()
        self._dependencies: dict[str, set[str]] = {}

    def ensure_class(self, cpp_name: str | None) -> str:
        if not cpp_name:
            return "FEBioEntity"
        normalised = _normalise_cpp_class(cpp_name)
        if normalised and normalised in VALUE_TYPE_MAP:
            self._value_types.add(normalised)
            return VALUE_TYPE_MAP[normalised]["python_name"]
        python_name = self._cpp_to_python.get(cpp_name)
        if python_name:
            return python_name
        manifest = None
        if normalised and normalised in self._class_to_manifest:
            manifest = self._class_to_manifest[normalised]
        elif cpp_name in self._class_to_manifest:
            manifest = self._class_to_manifest[cpp_name]
        if manifest:
            python_name = manifest.python_name
            self._cpp_to_python[cpp_name] = python_name
            if normalised:
                self._cpp_to_python.setdefault(normalised, python_name)
            self._python_to_category.setdefault(python_name, manifest.category)
            return python_name
        python_name = (normalised or cpp_name).split("::")[-1]
        self._cpp_to_python[cpp_name] = python_name
        if python_name != "FEBioEntity":
            self._stub_names.add(python_name)
        return python_name

    def resolve_base(
        self, consumer_category: str | None, metadata: dict[str, Any]
    ) -> str:
        python_name = self.ensure_class(metadata.get("base"))
        self._record_dependency(consumer_category, python_name)
        return python_name

    def resolve_param_type(
        self, consumer_category: str | None, param: dict[str, Any]
    ) -> str:
        array_info = _parse_array_ctype(param.get("ctype"))
        if array_info:
            base_cpp, length = array_info
            primitive_base = _map_primitive_type(base_cpp)
            if primitive_base:
                base_python = primitive_base
            else:
                base_python = self.ensure_class(base_cpp)
                self._record_dependency(consumer_category, base_python)
            return _format_array_type(base_python, length)
        primitive = _map_primitive_type(param.get("ctype"))
        definitions = param.get("definition") or []
        candidates: list[str] = []
        if definitions:
            for entry in definitions:
                cpp_name = entry.get("class")
                if not cpp_name:
                    continue
                candidate = self.ensure_class(cpp_name)
                self._record_dependency(consumer_category, candidate)
                candidates.append(candidate)
        if candidates:
            unique = list(dict.fromkeys(candidates))
            return " | ".join(unique)
        if primitive:
            return primitive
        cpp_identifier = _normalise_cpp_class(param.get("ctype", ""))
        if cpp_identifier and cpp_identifier in VALUE_TYPE_MAP:
            self._value_types.add(cpp_identifier)
            return VALUE_TYPE_MAP[cpp_identifier]["python_name"]
        if cpp_identifier:
            python_name = self.ensure_class(cpp_identifier)
            self._record_dependency(consumer_category, python_name)
            return python_name
        return "Any"

    @property
    def stub_names(self) -> list[str]:
        return sorted(self._stub_names)

    @property
    def value_types(self) -> list[str]:
        return sorted(self._value_types)

    def _record_dependency(
        self, consumer_category: str | None, python_name: str
    ) -> None:
        if not consumer_category or not python_name:
            return
        dependency_category = self._python_to_category.get(python_name)
        if not dependency_category or dependency_category == consumer_category:
            return
        bucket = self._dependencies.setdefault(consumer_category, set())
        bucket.add(python_name)

    def dependencies_for(self, category: str) -> set[str]:
        return self._dependencies.get(category, set())

    def category_for(self, python_name: str) -> str | None:
        return self._python_to_category.get(python_name)


class ClassRenderer:
    """Convert a manifest item plus metadata into Python source code."""

    def __init__(
        self, manifest: ManifestItem, metadata: dict[str, Any], resolver: TypeResolver
    ) -> None:
        self._manifest = manifest
        self._metadata = metadata
        self._resolver = resolver
        self._enum_definitions: dict[str, EnumDefinition] = {}
        self._core_helpers: set[str] = set()

    def render(self) -> tuple[str, str, bool]:
        """Generate the Python dataclass definition.

        Returns:
            Tuple of source code and resolved base class name.
        """

        base_class = self._resolver.resolve_base(
            self._manifest.category, self._metadata
        )
        header = [
            "@dataclass(kw_only=True)",
            f"class {self._manifest.python_name}({base_class}):",
        ]
        fe_type = self._metadata.get("registration") or self._metadata["class"]

        field_definitions = self._parameters()
        enum_blocks: list[str] = []
        for enum_def in self._enum_definitions.values():
            enum_blocks.append(f"    class {enum_def.name}(Enum):")
            if enum_def.members:
                for member in enum_def.members:
                    enum_blocks.append(f"        {member.name} = {member.value!r}")
            else:
                enum_blocks.append("        pass")
            enum_blocks.append("")

        fields_src = [definition.render() for definition in field_definitions]
        class_extension = CLASS_EXTENSIONS.get(self._manifest.python_name)
        if class_extension and class_extension.get("extra_fields"):
            fields_src = class_extension["extra_fields"] + fields_src
        has_payload = bool(fields_src) or bool(enum_blocks)
        body_lines: list[str] = []
        if enum_blocks:
            body_lines.extend(enum_blocks)
        if enum_blocks and fields_src:
            if enum_blocks[-1] != "":
                body_lines.append("")
        if fields_src:
            body_lines.extend(fields_src)
        body_lines.append(f"    fe_class: str = field(init=False, default={fe_type!r})")
        body_lines.append(
            f"    xml_tag: str = field(init=False, default={self._manifest.xml_tag!r})"
        )
        if class_extension and class_extension.get("post_init"):
            body_lines.append("")
            body_lines.extend(class_extension["post_init"])

        return "\n".join(header + body_lines), base_class, has_payload

    def core_helpers(self) -> set[str]:
        return set(self._core_helpers)

    def _parameters(self) -> list[ParameterDefinition]:
        """Build the list of parameter field definitions.

        Returns:
            Collection of dataclass field descriptors.
        """

        parameters: list[ParameterDefinition] = []
        for param in self._metadata.get("params", []):
            if param.get("hidden") or _has_hidden_flag(param):
                continue
            macro = param.get("macro")
            fe_name_raw = param.get("name")
            if not fe_name_raw:
                strings = param.get("strings") or []
                if strings:
                    fe_name_raw = strings[0]
            if not fe_name_raw:
                fe_name_raw = param.get("member", "param")
            attr_name = _normalise_identifier(fe_name_raw, param.get("member", "param"))
            python_type = self._resolver.resolve_param_type(
                self._manifest.category, param
            )
            base_python_type = python_type
            fe_name = fe_name_raw or attr_name
            meta: dict[str, Any] = {"fe_name": fe_name}
            allow_none = False
            if macro == "ADD_PROPERTY":
                meta["is_property"] = True
                args = param.get("args", [])
                if any("optional" in str(arg).lower() for arg in args):
                    allow_none = True
                if any("preferred" in str(arg).lower() for arg in args):
                    allow_none = True
                if any(
                    call.get("func") == "SetDefaultType"
                    for call in param.get("chain", [])
                ):
                    allow_none = True
            if param.get("units"):
                meta["units"] = param["units"]
            range_spec = param.get("range")
            range_predicate = _format_range_predicate(range_spec)
            if range_spec is not None:
                meta["range"] = range_spec
            enum_values = _extract_enum_values(param)
            if enum_values:
                meta["enum"] = enum_values
            elif param.get("enum"):
                meta["enum"] = param["enum"]
            if param.get("long_name"):
                meta["long_name"] = param["long_name"]
            raw_default = param.get("default")
            coerced_default = _coerce_python_default(raw_default)
            default_expr: str | None = None
            if macro == "ADD_PARAMETER":
                for call in param.get("chain", []):
                    if call.get("func") == "SetFlags" and any(
                        "optional" in str(arg).lower() for arg in call.get("args", [])
                    ):
                        allow_none = True
                        break
            if allow_none:
                meta["optional"] = True

            if enum_values:
                enum_base_type = (
                    _map_primitive_type(param.get("ctype")) or base_python_type
                )
                if enum_base_type not in {"int", "float", "bool", "str"}:
                    enum_base_type = "str"
                enum_class_name = _pascal_case(attr_name) or "Enum"
                enum_members = _build_enum_members(enum_values, enum_base_type)
                if enum_class_name not in self._enum_definitions:
                    self._enum_definitions[enum_class_name] = EnumDefinition(
                        name=enum_class_name, members=enum_members
                    )
                meta["enum_class"] = enum_class_name
                if base_python_type == "Any":
                    python_type = enum_class_name
                else:
                    python_type = f"{enum_class_name} | {base_python_type}"
                matched = _match_enum_default(
                    raw_default, coerced_default, enum_members
                )
                if matched is not None:
                    coerced_default = matched.value
                    default_expr = f"{enum_class_name}.{matched.name}"
                    meta["default"] = _format_metadata_default(matched.value)
                elif raw_default is not None:
                    meta["default"] = _format_metadata_default(raw_default)
                elif coerced_default is not _DEFAULT_SENTINEL:
                    meta["default"] = _format_metadata_default(coerced_default)
                elif not allow_none and enum_members:
                    coerced_default = enum_members[0].value
                    default_expr = f"{enum_class_name}.{enum_members[0].name}"
                    meta["default"] = _format_metadata_default(enum_members[0].value)
            else:
                if raw_default is not None:
                    meta["default"] = _format_metadata_default(raw_default)
                elif coerced_default is not _DEFAULT_SENTINEL:
                    meta["default"] = _format_metadata_default(coerced_default)

            if python_type == "Any":
                inferred = _infer_numeric_python_type(
                    range_spec, raw_default, coerced_default
                )
                if inferred:
                    python_type = inferred
            if python_type in {"int", "float"} and isinstance(coerced_default, str):
                python_type = "str"
            if python_type in {"int", "float"} and isinstance(raw_default, str):
                python_type = "str"

            annotation_payload = _build_range_annotation_payload(range_spec)
            if annotation_payload and python_type in {"int", "float"}:
                python_type = f"Annotated[{python_type}, {annotation_payload!r}]"

            for helper in _identify_core_helpers(python_type):
                self._core_helpers.add(helper)

            parameters.append(
                ParameterDefinition(
                    attr_name=attr_name,
                    type_hint=python_type,
                    metadata=meta,
                    default=coerced_default,
                    default_expr=default_expr,
                    allow_none=allow_none,
                )
            )
        required: list[ParameterDefinition] = []
        optional_defs: list[ParameterDefinition] = []
        for definition in parameters:
            if definition.has_default():
                optional_defs.append(definition)
            else:
                required.append(definition)
        return required + optional_defs


def _build_module_header(
    *,
    uses_enum: bool,
    uses_any: bool,
    uses_optional: bool,
    uses_annotated: bool,
    uses_range_spec: bool,
) -> str:
    lines = [
        "# ---------------------------------------------------------------------------",
        "# WARNING: This file was auto-generated by classExposer.generator.",
        "# It is brittle by design—manual edits are likely to be lost on regeneration.",
        "# Update the metadata or manifests instead, then re-run the generator.",
        "# ---------------------------------------------------------------------------",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass, field",
    ]

    typing_parts: list[str] = []
    if uses_any:
        typing_parts.append("Any")
    if uses_optional:
        typing_parts.append("Optional")
    if typing_parts:
        if len(typing_parts) == 1:
            lines.append(f"from typing import {typing_parts[0]}")
        else:
            lines.append(f"from typing import {', '.join(typing_parts)}")

    if uses_enum:
        lines.append("from enum import Enum")

    common_parts: list[str] = []
    if uses_annotated:
        common_parts.append("Annotated")
    common_parts.append("FEBioEntity")
    if uses_range_spec:
        common_parts.append("RangeSpec")
    lines.append(f"from {ROOT_PACKAGE}.common.base import {', '.join(common_parts)}")
    return "\n".join(lines) + "\n\n"


def _render_module(
    manifest_items: Sequence[ManifestItem],
    metadata_repo: MetadataRepository,
    *,
    resolver: TypeResolver,
    category: str | None,
) -> ModuleRenderResult:
    """Compose the complete Python module.

    Args:
        manifest_items: Ordered list of manifest entries.
        metadata_repo: Repository with FEBio metadata.

    Returns:
        Full Python source representing the bindings module.
    """

    categories = (
        {category}
        if category
        else {item.category for item in manifest_items if item.category}
    )
    extra_imports, extra_exports, stub_exclude, skip_value_types, extra_prefix = (
        _collect_category_injections(categories)
    )

    class_entries: list[dict[str, Any]] = []
    used_bases: set[str] = set()
    core_helpers: set[str] = set()
    kept_items: list[ManifestItem] = []
    names_in_module: set[str] = set()
    for item in manifest_items:
        try:
            metadata = metadata_repo.get(item.class_name)
        except KeyError as exc:
            raise KeyError(
                f"Manifest references class '{item.class_name}' but it was not found in the metadata"
            ) from exc
        renderer = ClassRenderer(item, metadata, resolver)
        rendered, base_class, has_payload = renderer.render()
        if not has_payload:
            continue
        names_in_module.add(item.python_name)
        class_entries.append(
            {"name": item.python_name, "base": base_class, "source": rendered}
        )
        kept_items.append(item)
        if base_class:
            used_bases.add(base_class)
        core_helpers.update(renderer.core_helpers())

    for entry in class_entries:
        if entry["base"] not in names_in_module:
            entry["base"] = None
    ordered_entries = _order_class_entries(class_entries)
    class_definitions = [entry["source"] for entry in ordered_entries]
    ordered_names = [entry["name"] for entry in ordered_entries]
    order_index = {name: idx for idx, name in enumerate(ordered_names)}
    kept_items.sort(
        key=lambda item: order_index.get(item.python_name, len(order_index))
    )
    class_blob = "\n\n".join(class_definitions)

    used_value_type_keys: list[str] = []
    if not skip_value_types:
        for key in resolver.value_types:
            spec = VALUE_TYPE_MAP.get(key)
            if not spec:
                continue
            python_name = spec["python_name"]
            if python_name in class_blob:
                used_value_type_keys.append(key)

    value_type_block = (
        "" if skip_value_types else _render_value_types(used_value_type_keys)
    )
    stub_block = _render_stub_classes(
        resolver.stub_names, ignore=stub_exclude, used_bases=used_bases
    )
    import_block = _render_dependency_imports(resolver, category)

    seen_exports: set[str] = set()
    export_names: list[str] = []

    def _append_export(name: str) -> None:
        if name and name not in seen_exports:
            export_names.append(name)
            seen_exports.add(name)

    for item in kept_items:
        _append_export(item.python_name)
    exports = ",\n    ".join(f"'{name}'" for name in export_names)
    exports_block = (
        f"__all__ = [\n    {exports},\n]\n\n" if export_names else "__all__ = []\n\n"
    )

    body_parts: list[str] = []
    if import_block:
        body_parts.append(import_block)
    if extra_imports:
        body_parts.append(extra_imports)
    existing_helpers: set[str] = set()
    core_param_prefix = f"from {ROOT_PACKAGE}.Core.Parameters import"
    for block in filter(None, [import_block, extra_imports]):
        for line in block.splitlines():
            line = line.strip()
            if line.startswith(core_param_prefix):
                after = line.split("import", 1)[1]
                for token in after.split(","):
                    name = token.strip()
                    if name:
                        existing_helpers.add(name)
    pending_helpers = sorted(
        helper for helper in core_helpers if helper not in existing_helpers
    )
    if pending_helpers:
        body_parts.append(
            f"from {ROOT_PACKAGE}.Core.Parameters import {', '.join(pending_helpers)}\n"
        )
    if value_type_block:
        body_parts.append(value_type_block)
    if stub_block:
        body_parts.append(stub_block)
    if extra_prefix:
        body_parts.append(extra_prefix)
    body_parts.append(exports_block)
    if class_definitions:
        body_parts.append("\n\n".join(class_definitions))
    body_parts.append("")

    module_body = "\n".join(body_parts).replace("\n\n\n", "\n\n")

    uses_enum = bool(re.search(r"\bEnum\b", module_body))
    uses_any = bool(re.search(r"\bAny\b", module_body))
    uses_optional = "Optional[" in module_body
    uses_annotated = "Annotated[" in module_body
    uses_range_spec = "RangeSpec" in module_body

    header = _build_module_header(
        uses_enum=uses_enum,
        uses_any=uses_any,
        uses_optional=uses_optional,
        uses_annotated=uses_annotated,
        uses_range_spec=uses_range_spec,
    )

    full_module = (header + module_body).replace("\n\n\n", "\n\n")
    return ModuleRenderResult(text=full_module, exports=export_names)


def _collect_category_injections(
    categories: set[str],
) -> tuple[str, list[str], set[str], bool, str]:
    imports: list[str] = []
    exports: list[str] = []
    stub_exclude: set[str] = set()
    skip_value_types = False
    prefixes: list[str] = []
    for category in sorted(categories):
        injections = _CATEGORY_INJECTIONS.get(category)
        if not injections:
            continue
        block = injections.get("imports")
        if block:
            imports.append(block)
        exports.extend(injections.get("exports", []))
        stub_exclude.update(injections.get("stub_exclude", set()))
        if injections.get("skip_value_types"):
            skip_value_types = True
        prefix_block = injections.get("prefix")
        if prefix_block:
            prefixes.append(prefix_block)
    import_block = "\n".join(block for block in imports if block.strip())
    prefix_block = "\n".join(block for block in prefixes if block.strip())
    return import_block, exports, stub_exclude, skip_value_types, prefix_block


def _load_manifest(path: Path) -> dict[str, list[ManifestItem]]:
    """Load the generation manifest from a file or directory."""

    manifest_map: dict[str, list[ManifestItem]] = {}
    if path.is_dir():
        for child in sorted(path.glob("*.json")):
            _merge_manifest_map(manifest_map, _load_manifest_file(child))
    else:
        _merge_manifest_map(manifest_map, _load_manifest_file(path))
    return manifest_map


def _merge_manifest_map(
    manifest_map: dict[str, list[ManifestItem]],
    additions: dict[str, list[ManifestItem]],
) -> None:
    for category, items in additions.items():
        manifest_map.setdefault(category, []).extend(items)


def _load_manifest_file(path: Path) -> dict[str, list[ManifestItem]]:
    raw = json.loads(path.read_text())
    manifest_map: dict[str, list[ManifestItem]] = {}
    for category, entries in raw.items():
        bucket = manifest_map.setdefault(category, [])
        for entry in entries:
            bucket.append(
                ManifestItem(
                    category=category,
                    class_name=entry["class"],
                    python_name=entry.get("python_name", entry["class"]),
                    xml_tag=entry.get("xml_tag", ""),
                )
            )
    return manifest_map


def _render_stub_classes(
    stub_names: Sequence[str],
    *,
    ignore: set[str] | None = None,
    used_bases: set[str] | None = None,
) -> str:
    """Render placeholder base classes required for typing."""

    ignore_set = {"FEBioEntity", ""}
    if ignore:
        ignore_set.update(ignore)
    used = used_bases or set()
    filtered = [
        name
        for name in stub_names
        if name not in ignore_set and (not used or name in used)
    ]
    if not filtered:
        return ""
    lines: list[str] = []
    for name in filtered:
        lines.append(f"class {name}(FEBioEntity):")
        lines.append("    pass")
        lines.append("")
    return "\n".join(lines)


def _render_value_types(value_types: Sequence[str]) -> str:
    """Render simple value-type helpers like vec3d."""

    if not value_types:
        return ""
    blocks: list[str] = []
    for key in value_types:
        spec = VALUE_TYPE_MAP.get(key)
        if not spec:
            continue
        cls_name = spec["python_name"]
        field_type = spec.get("field_type", "float")
        default = spec.get("default", 0.0)
        field_names = spec.get("fields", ())
        fields_block = (
            "\n".join(
                f"    {field_name}: {field_type} = {default}"
                for field_name in field_names
            )
            or "    pass"
        )
        format_expr = ", ".join(f"{{self.{name}}}" for name in field_names) or "{self}"
        blocks.append(
            "@dataclass(kw_only=True)\n"
            f"class {cls_name}:\n"
            f"{fields_block}\n\n"
            "    def __str__(self) -> str:\n"
            f'        return f"{format_expr}"\n'
        )
        blocks.append("")
    return "\n".join(blocks)


def _order_class_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not entries:
        return []
    name_to_entry = {entry["name"]: entry for entry in entries}
    graph: dict[str, list[str]] = {name: [] for name in name_to_entry}
    indegree: dict[str, int] = {name: 0 for name in name_to_entry}

    for entry in entries:
        base = entry.get("base")
        if base in name_to_entry:
            graph.setdefault(base, []).append(entry["name"])
            indegree[entry["name"]] = indegree.get(entry["name"], 0) + 1
        graph.setdefault(entry["name"], [])
        indegree.setdefault(entry["name"], indegree.get(entry["name"], 0))

    queue = deque([name for name, deg in indegree.items() if deg == 0])
    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()

    while queue:
        name = queue.popleft()
        if name in name_to_entry and name not in seen:
            ordered.append(name_to_entry[name])
        seen.add(name)
        for child in graph.get(name, []):
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    for entry in entries:
        if entry["name"] not in seen:
            ordered.append(entry)
    return ordered


def _render_dependency_imports(resolver: TypeResolver, category: str | None) -> str:
    if not category:
        return ""
    dependencies = sorted(resolver.dependencies_for(category))
    if not dependencies:
        return ""
    lines: list[str] = []
    for python_name in dependencies:
        dependency_category = resolver.category_for(python_name)
        if not dependency_category or dependency_category == category:
            continue
        safe = _sanitise_category(dependency_category)
        module_path = f"{ROOT_PACKAGE}.{safe}.Generated_{safe}_automatic"
        lines.append(f"from {module_path} import {python_name}")
    return "\n".join(lines) + ("\n" if lines else "")


def _sanitise_category(category: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z]+", "_", category).strip("_")
    return token or "Category"


def _category_module_path(output_dir: Path, category: str) -> Path:
    safe = _sanitise_category(category)
    filename = f"Generated_{safe}_automatic.py"
    return output_dir / safe / filename


def _per_class_module_path(output_dir: Path, category: str, python_name: str) -> Path:
    safe_category = _sanitise_category(category)
    safe_class = _sanitise_category(python_name)
    filename = f"{safe_class}_automatic.py"
    return output_dir / safe_category / filename


def _write_package_init(
    output_dir: Path, manifest_map: dict[str, list[ManifestItem]]
) -> None:
    lines = ["# Auto-generated by classExposer.generator", "__all__ = ["]
    for category in sorted(manifest_map):
        safe = _sanitise_category(category)
        lines.append(f"    '{safe}',")
    lines.append("]\n")
    for category in sorted(manifest_map):
        safe = _sanitise_category(category)
        lines.append(f"from .{safe} import *  # noqa: F401,F403")
    module_path = output_dir / "__init__.py"
    module_path.parent.mkdir(parents=True, exist_ok=True)
    module_path.write_text("\n".join(lines) + "\n")


def _load_metadata_entries(metadata_path: Path) -> list[dict[str, Any]]:
    """Collect metadata entries from a JSON file or directory of JSON files."""

    if metadata_path.is_file():
        return json.loads(metadata_path.read_text())

    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata path '{metadata_path}' does not exist")

    entries: list[dict[str, Any]] = []
    for json_file in sorted(metadata_path.rglob("*.json")):
        if json_file.name == "manifest.json":
            continue
        if any(part.startswith("manifest") for part in json_file.parts):
            continue
        data = json.loads(json_file.read_text())
        if isinstance(data, list):
            entries.extend(data)
    return entries


def generate(
    manifest_path: Path,
    metadata_path: Path,
    output_path: Path | None,
    output_dir: Path | None,
    per_class_output_dir: Path | None,
    allow_missing: bool,
) -> None:
    """Generate the Python bindings module(s)."""

    manifest_map = _load_manifest(manifest_path)
    if not manifest_map:
        raise ValueError(f"Manifest at {manifest_path} yielded no entries")

    metadata_repo = MetadataRepository(_load_metadata_entries(metadata_path))

    all_manifest_items = [item for items in manifest_map.values() for item in items]
    resolver = TypeResolver(all_manifest_items)

    effective_manifest_map: dict[str, list[ManifestItem]] = {}
    missing_manifest_items: list[ManifestItem] = []

    for category, items in manifest_map.items():
        present_items: list[ManifestItem] = []
        for item in items:
            if metadata_repo.has(item.class_name):
                present_items.append(item)
            else:
                missing_manifest_items.append(item)
        if present_items:
            effective_manifest_map[category] = present_items

    if not allow_missing and missing_manifest_items:
        missing_classes = ", ".join(sorted({item.class_name for item in missing_manifest_items}))
        raise KeyError(
            "Metadata missing for manifest entries: "
            f"{missing_classes}"
        )

    if allow_missing and missing_manifest_items:
        skipped = ", ".join(sorted({item.class_name for item in missing_manifest_items}))
        print(
            f"[generator] Skipping manifest entries without metadata: {skipped}",
            flush=True,
        )

    if not effective_manifest_map:
        raise ValueError("No manifest entries have matching metadata to generate")

    category_items = [item for items in effective_manifest_map.values() for item in items]

    formatted_targets: set[Path] = set()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        module_result = _render_module(
            category_items, metadata_repo, resolver=resolver, category=None
        )
        output_path.write_text(module_result.text)
        formatted_targets.add(output_path)

    if output_dir:
        for category, items in effective_manifest_map.items():
            module_result = _render_module(
                items, metadata_repo, resolver=resolver, category=category
            )
            module_path = _category_module_path(output_dir, category)
            module_path.parent.mkdir(parents=True, exist_ok=True)
            module_path.write_text(module_result.text)
            formatted_targets.add(module_path)

    if per_class_output_dir:
        for category, items in effective_manifest_map.items():
            category_dir = per_class_output_dir / _sanitise_category(category)
            category_dir.mkdir(parents=True, exist_ok=True)
            init_path = category_dir / "__init__.py"
            if not init_path.exists():
                init_path.write_text(
                    "# Auto-generated by classExposer.generator\n__all__ = []\n"
                )
                formatted_targets.add(init_path)
            for item in items:
                module_result = _render_module(
                    [item], metadata_repo, resolver=resolver, category=category
                )
                if not module_result.exports:
                    continue
                module_path = _per_class_module_path(
                    per_class_output_dir, category, item.python_name
                )
                module_path.parent.mkdir(parents=True, exist_ok=True)
                module_path.write_text(module_result.text)
                formatted_targets.add(module_path)

    if not output_path and not output_dir and not per_class_output_dir:
        raise ValueError(
            "At least one of --output, --output-dir, or --per-class-output-dir must be provided"
        )

    _run_ruff(sorted(formatted_targets))


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser for the generator.

    Returns:
        Configured :class:`argparse.ArgumentParser` instance.
    """

    parser = argparse.ArgumentParser(
        description="Generate Python FEBio stubs from metadata"
    )
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to manifest.json or directory of manifests",
    )
    parser.add_argument(
        "--metadata",
        required=True,
        help="Path to metadata.json or directory containing per-class JSON files",
    )
    parser.add_argument("--output", help="Path to write a single consolidated module")
    parser.add_argument(
        "--output-dir",
        help="Directory where category-specific packages will be generated",
    )
    parser.add_argument(
        "--per-class-output-dir",
        help="Directory where individual per-class modules will be generated",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Skip manifest entries that do not have metadata instead of failing",
    )
    return parser


def _run_ruff(paths: Sequence[Path]) -> None:
    if not paths:
        return
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    cmd = [
        "ruff",
        "check",
        "--select",
        "I,F401",
        "--fix",
        *[str(path) for path in existing],
    ]
    try:
        subprocess.run(cmd, check=False)
    except FileNotFoundError:
        print("[generator] Ruff not found, skipping linting", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Args:
        argv: Optional argument list for testing purposes.

    Returns:
        Exit status code.
    """

    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not args.output and not args.output_dir and not args.per_class_output_dir:
        parser.error(
            "At least one of --output, --output-dir, or --per-class-output-dir must be provided"
        )
    generate(
        Path(args.manifest),
        Path(args.metadata),
        Path(args.output) if args.output else None,
        Path(args.output_dir) if args.output_dir else None,
        Path(args.per_class_output_dir) if args.per_class_output_dir else None,
        args.allow_missing,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
