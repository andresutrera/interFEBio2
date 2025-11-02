"""Simplified FEBio template helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, cast
import xml.etree.ElementTree as ET

Number = float


@dataclass
class BuildContext:
    """
    Formatting options applied while writing FEBio templates.

    Parameters
    ----------
    fmt
        ``printf``-style format string used for numeric values.
    namespaces
        Optional mapping passed to :func:`xml.etree.ElementTree.findall`.
    """

    fmt: str = "%.20e"
    namespaces: Dict[str, str] = field(default_factory=dict)

    def format_value(self, value: Number) -> str:
        return self.fmt % float(value)


@dataclass
class ParameterBinding:
    """
    Map a single theta value onto an XML element's text content.

    The XPath is evaluated with :func:`ElementTree.findall`. All matching nodes
    are updated; set ``required`` to ``False`` to ignore missing nodes.
    """

    theta_name: str
    xpath: str
    required: bool = True

    def apply(
        self,
        root: ET.Element,
        theta: Mapping[str, Number],
        ctx: BuildContext,
    ) -> None:
        if self.theta_name not in theta:
            raise KeyError(f"theta value '{self.theta_name}' not provided")

        nodes = root.findall(self.xpath, namespaces=ctx.namespaces or None)
        if not nodes:
            if self.required:
                raise ValueError(f"No nodes matched XPath: {self.xpath}")
            return

        text = ctx.format_value(theta[self.theta_name])
        for node in nodes:
            node.text = text


@dataclass
class FebTemplate:
    """FEBio template plus bindings applied during rendering."""

    template_path: Path | str
    bindings: list[ParameterBinding] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.template_path = Path(self.template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")
        # Normalise bindings to a list for repeated iteration
        self.bindings = list(self.bindings)

    def add_binding(self, binding: ParameterBinding) -> None:
        self.bindings.append(binding)

    def render(
        self,
        theta: Mapping[str, Number],
        ctx: BuildContext | None = None,
    ) -> ET.ElementTree:
        ctx = ctx or BuildContext()
        tree = cast(ET.ElementTree, ET.parse(self.template_path))
        root = cast(ET.Element, tree.getroot())
        for binding in self.bindings:
            binding.apply(root, theta, ctx)
        return tree

    def write(
        self,
        theta: Mapping[str, Number],
        out_path: Path | str,
        ctx: BuildContext | None = None,
    ) -> Path:
        tree = self.render(theta, ctx)
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_file, encoding="ISO-8859-1", xml_declaration=True)
        return out_file


@dataclass
class FebBuilder:
    """Render FEBio input files under a dedicated case subfolder."""

    template: FebTemplate
    subfolder: str

    def build(
        self,
        theta: Mapping[str, Number],
        out_root: Path | str,
        out_name: str | None = None,
        ctx: BuildContext | None = None,
    ) -> Path:
        ctx = ctx or BuildContext()
        sim_dir = Path(out_root) / self.subfolder
        sim_dir.mkdir(parents=True, exist_ok=True)
        base_name = out_name or Path(self.template.template_path).name
        feb_path = sim_dir / base_name
        return self.template.write(theta, feb_path, ctx)


__all__ = [
    "BuildContext",
    "ParameterBinding",
    "FebTemplate",
    "FebBuilder",
]
