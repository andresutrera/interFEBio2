# interFEBio/Optimize/feb_bindings.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, Tuple
import os
import xml.etree.ElementTree as ET
import numpy as np

Number = float


# ---------- context ----------
@dataclass(frozen=True)
class BuildContext:
    iter_id: int
    case_name: str
    fmt: str = "%.20e"
    namespaces: Dict[str, str] = field(default_factory=dict)


# ---------- helpers ----------
def _set_text(node: ET.Element, val: Number, fmt: str) -> None:
    node.text = fmt % float(val)


def _findall(root: ET.Element, xpath: str, ns: Dict[str, str]) -> List[ET.Element]:
    return list(root.findall(xpath, namespaces=ns or None))


# ---------- base binding ----------
class FebBinding:
    def apply(
        self, root: ET.Element, theta: Dict[str, Number], ctx: BuildContext
    ) -> None:
        raise NotImplementedError


# ---------- bindings ----------
@dataclass
class ScalarXPathBinding(FebBinding):
    """Map one theta param to node text at XPath."""

    theta_name: str
    xpath: str  # e.g. ".//Step/step[@id='2']/Control/solver/max_ups"
    required: bool = True

    def apply(
        self, root: ET.Element, theta: Dict[str, Number], ctx: BuildContext
    ) -> None:
        val = theta[self.theta_name]
        nodes = _findall(root, self.xpath, ctx.namespaces)
        if self.required and not nodes:
            raise ValueError(f"No nodes for xpath: {self.xpath}")
        for n in nodes:
            _set_text(n, val, ctx.fmt)


@dataclass
class AttrXPathBinding(FebBinding):
    """Set attribute value on matched nodes."""

    theta_name: str
    xpath: str  # nodes to modify
    attr: str  # attribute name
    required: bool = True

    def apply(
        self, root: ET.Element, theta: Dict[str, Number], ctx: BuildContext
    ) -> None:
        val = theta[self.theta_name]
        nodes = _findall(root, self.xpath, ctx.namespaces)
        if self.required and not nodes:
            raise ValueError(f"No nodes for xpath: {self.xpath}")
        sval = ctx.fmt % float(val)
        for n in nodes:
            n.set(self.attr, sval)


@dataclass
class MaterialParamBinding(FebBinding):
    """
    Set a child tag inside <material> selected by id or name.
    Example: tag_name='E_phi', selector=('id','1') -> sets <material id="1">/<E_phi>.
    """

    theta_name: str
    tag_name: str
    selector: Tuple[Literal["id", "name"], str]

    def apply(
        self, root: ET.Element, theta: Dict[str, Number], ctx: BuildContext
    ) -> None:
        key, val = self.selector
        xp = f".//Material/material[@{key}='{val}']/{self.tag_name}"
        nodes = _findall(root, xp, ctx.namespaces)
        if not nodes:
            raise ValueError(f"MaterialParamBinding not found: {xp}")
        for n in nodes:
            _set_text(n, theta[self.theta_name], ctx.fmt)


@dataclass
class CallbackBinding(FebBinding):
    """
    Arbitrary edit: fn(root, theta, ctx) -> None
    Use for vectors, load curves, creating nodes, etc.
    """

    fn: Callable[[ET.Element, Dict[str, Number], BuildContext], None]

    def apply(
        self, root: ET.Element, theta: Dict[str, Number], ctx: BuildContext
    ) -> None:
        self.fn(root, theta, ctx)


# ---------- template and builder ----------
@dataclass
class FebTemplate:
    template_path: Path | str
    bindings: List[FebBinding] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.template_path = Path(self.template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")

    def add_binding(self, b: FebBinding) -> None:
        self.bindings.append(b)

    def render(self, theta: Dict[str, Number], ctx: BuildContext) -> ET.ElementTree:
        tree = ET.parse(str(self.template_path))
        root = tree.getroot()
        for b in self.bindings:
            b.apply(root, theta, ctx)
        return tree

    def write(self, theta: Dict[str, Number], out_path: str, ctx: BuildContext) -> None:
        tree = self.render(theta, ctx)
        out_file = Path(out_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        tree.write(out_file, encoding="ISO-8859-1", xml_declaration=True)


@dataclass
class FebBuilder:
    template: FebTemplate
    subfolder: str

    def build(
        self,
        theta: Dict[str, Number],
        out_root: str,
        out_name: Optional[str] = None,
        ctx: Optional[BuildContext] = None,
    ) -> Tuple[str, str]:
        if ctx is None:
            ctx = BuildContext(iter_id=0, case_name="default")
        sim_dir = os.path.join(out_root, self.subfolder)
        os.makedirs(sim_dir, exist_ok=True)
        base = out_name or os.path.basename(self.template.template_path.name)
        feb_path = os.path.join(sim_dir, base)
        self.template.write(theta, feb_path, ctx)
        xplt = os.path.splitext(base)[0] + ".xplt"
        return feb_path, os.path.join(sim_dir, xplt)
