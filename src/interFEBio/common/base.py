"""Shared FEBio binding helpers used across generated modules."""

from __future__ import annotations

from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Optional

import xml.etree.ElementTree as ET

try:  # pragma: no cover - typing fallback
    from typing import Annotated  # type: ignore
except ImportError:  # pragma: no cover
    from typing_extensions import Annotated  # type: ignore

__all__ = [
    "Annotated",
    "FEBioEntity",
    "RangeSpec",
    "indent_xml",
    "describe_range",
    "validate_range",
]


@dataclass(slots=True)
class RangeSpec:
    """Structured range descriptor attached to Annotated metadata."""

    min: float | None = None
    min_inclusive: bool = True
    max: float | None = None
    max_inclusive: bool = True
    not_equal: float | None = None
    min_expr: str | None = None
    max_expr: str | None = None
    not_equal_expr: str | None = None
    raw: str | None = None

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.min is not None:
            data["min"] = self.min
            data["min_inclusive"] = self.min_inclusive
        elif self.min_expr is not None:
            data["min_expr"] = self.min_expr
            data["min_inclusive"] = self.min_inclusive
        if self.max is not None:
            data["max"] = self.max
            data["max_inclusive"] = self.max_inclusive
        elif self.max_expr is not None:
            data["max_expr"] = self.max_expr
            data["max_inclusive"] = self.max_inclusive
        if self.not_equal is not None:
            data["not_equal"] = self.not_equal
        elif self.not_equal_expr is not None:
            data["not_equal_expr"] = self.not_equal_expr
        if self.raw is not None:
            data["raw"] = self.raw
        return data


def indent_xml(element: ET.Element, level: int = 0) -> None:
    """Recursively indent an XML tree for pretty-printing."""

    children = list(element)
    if children:
        if not element.text or not element.text.strip():
            element.text = "\n" + "  " * (level + 1)
        for child in children:
            indent_xml(child, level + 1)
            if not child.tail or not child.tail.strip():
                child.tail = "\n" + "  " * (level + 1)
        if not children[-1].tail or not children[-1].tail.strip():
            children[-1].tail = "\n" + "  " * level
    elif level and (not element.tail or not element.tail.strip()):
        element.tail = "\n" + "  " * level


def _coerce_numeric(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def describe_range(spec: Any) -> str:
    """Return a human-readable range description for metadata."""

    if isinstance(spec, RangeSpec):
        spec_dict = spec.as_dict()
    elif isinstance(spec, dict):
        spec_dict = spec
    else:
        return "specified range"
    min_val = spec_dict.get("min", spec_dict.get("min_expr"))
    max_val = spec_dict.get("max", spec_dict.get("max_expr"))
    not_equal = spec_dict.get("not_equal", spec_dict.get("not_equal_expr"))
    parts: list[str] = []
    if min_val is not None and max_val is not None:
        left = "<=" if spec_dict.get("min_inclusive", True) else "<"
        right = "<=" if spec_dict.get("max_inclusive", True) else "<"
        parts.append(f"{min_val} {left} value {right} {max_val}")
    else:
        if min_val is not None:
            op = ">=" if spec_dict.get("min_inclusive", True) else ">"
            parts.append(f"value {op} {min_val}")
        if max_val is not None:
            op = "<=" if spec_dict.get("max_inclusive", True) else "<"
            parts.append(f"value {op} {max_val}")
    if not_equal is not None:
        parts.append(f"value != {not_equal}")
    raw_args = spec_dict.get("raw") or spec_dict.get("raw_args")
    if raw_args and not parts:
        parts.append(f"range({raw_args})")
    return " and ".join(parts) if parts else "specified range"


def validate_range(owner: str, field_name: str, value: Any, spec: Any) -> None:
    """Validate ``value`` against ``spec`` and raise ``ValueError`` on failure."""

    if spec is None or isinstance(spec, str) or value is None:
        return
    if isinstance(value, Enum):
        value = value.value
    numeric = _coerce_numeric(value)
    if numeric is None:
        return
    if isinstance(spec, RangeSpec):
        spec_dict = spec.as_dict()
    elif isinstance(spec, dict):
        spec_dict = spec
    else:
        return
    message = describe_range(spec_dict)
    min_val = _coerce_numeric(spec_dict.get("min", spec_dict.get("min_expr")))
    if min_val is not None:
        if spec_dict.get("min_inclusive", True):
            if numeric < min_val:
                raise ValueError(f"{owner}.{field_name} must satisfy {message} (got {value!r})")
        else:
            if numeric <= min_val:
                raise ValueError(f"{owner}.{field_name} must satisfy {message} (got {value!r})")
    max_val = _coerce_numeric(spec_dict.get("max", spec_dict.get("max_expr")))
    if max_val is not None:
        if spec_dict.get("max_inclusive", True):
            if numeric > max_val:
                raise ValueError(f"{owner}.{field_name} must satisfy {message} (got {value!r})")
        else:
            if numeric >= max_val:
                raise ValueError(f"{owner}.{field_name} must satisfy {message} (got {value!r})")
    not_equal = spec_dict.get("not_equal", spec_dict.get("not_equal_expr"))
    if not_equal is not None:
        neq_numeric = _coerce_numeric(not_equal)
        if neq_numeric is not None:
            if numeric == neq_numeric:
                raise ValueError(f"{owner}.{field_name} must satisfy {message} (got {value!r})")
        elif value == not_equal:
            raise ValueError(f"{owner}.{field_name} must satisfy {message} (got {value!r})")


class FEBioEntity:
    """Base class for generated FEBio bindings."""

    fe_class: str = ""
    xml_tag: str = ""

    def to_xml_element(self, name: str | None = None, *, tag_override: str | None = None) -> ET.Element:
        tag = tag_override or self.xml_tag or "bc"
        element = ET.Element(tag)
        element.set("type", self.fe_class)
        if name:
            element.set("name", name)
        for field_info in fields(self):
            if field_info.name in {"fe_class", "xml_tag"}:
                continue
            value = getattr(self, field_info.name)
            if value is None:
                continue
            metadata = field_info.metadata or {}
            param_name = metadata.get("fe_name", field_info.name)
            if metadata.get("is_property"):
                self._append_property_element(element, param_name, value)
                continue
            param_element = ET.SubElement(element, param_name)
            if isinstance(value, Enum):
                value = value.value
            param_element.text = str(value)
        return element

    def _append_property_element(self, parent: ET.Element, tag: str, value: Any) -> None:
        if isinstance(value, FEBioEntity):
            child = value.to_xml_element(tag_override=tag)
            parent.append(child)
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                if item is None:
                    continue
                self._append_property_element(parent, tag, item)
            return
        if isinstance(value, Enum):
            value = value.value
        prop_element = ET.SubElement(parent, tag)
        prop_element.text = str(value)

    def __setattr__(self, name: str, value: Any) -> None:
        field_info = getattr(self, "__dataclass_fields__", {}).get(name)
        if field_info:
            metadata = field_info.metadata or {}
            range_spec = metadata.get("range")
            if range_spec is not None:
                validate_range(self.__class__.__name__, name, value, range_spec)
        object.__setattr__(self, name, value)

    def to_xml_string(self, name: str | None = None) -> str:
        element = self.to_xml_element(name=name)
        indent_xml(element)
        return ET.tostring(element, encoding="unicode")
