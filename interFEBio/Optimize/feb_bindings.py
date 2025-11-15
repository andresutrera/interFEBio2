"""Simplified FEBio template helpers."""

from __future__ import annotations

import ast
import operator
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, cast

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
        """Format a scalar value according to the ``fmt`` pattern."""
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
        """Insert formatted values into all nodes matched by ``xpath``."""
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


class _ExpressionEvaluator:
    """Evaluate a restricted Python expression against mapping values."""

    _BIN_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
    }
    _UNARY_OPS = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def __init__(self, expression: str):
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:  # pragma: no cover - exercised in tests
            raise ValueError(f"Invalid expression '{expression}': {exc}") from exc
        self._tree = tree

    def evaluate(self, theta: Mapping[str, Number]) -> Number:
        return float(self._eval(self._tree.body, theta))

    def _eval(self, node: ast.AST, theta: Mapping[str, Number]) -> Number:
        if isinstance(node, ast.Expression):
            return self._eval(node.body, theta)
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self._BIN_OPS:
                raise ValueError(f"Unsupported operator in expression: {op_type.__name__}")
            left = self._eval(node.left, theta)
            right = self._eval(node.right, theta)
            return float(self._BIN_OPS[op_type](left, right))
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self._UNARY_OPS:
                raise ValueError(f"Unsupported operator in expression: {op_type.__name__}")
            return float(self._UNARY_OPS[op_type](self._eval(node.operand, theta)))
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are permitted in expressions")
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id not in theta:
                raise KeyError(f"theta value '{node.id}' not provided")
            return float(theta[node.id])
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")


@dataclass
class EvaluationBinding:
    """Bind an XPath to a value derived from multiple θ parameters.

    Parameters
    ----------
    xpath
        XPath targeting the XML node(s) to update.
    value
        Expression or callable producing a numeric value from ``theta``.
    required
        Same semantics as :class:`ParameterBinding`.
    """

    xpath: str
    value: str | Callable[[Mapping[str, Number]], Number]
    required: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.value, str):
            self._expr = _ExpressionEvaluator(self.value)
            self._callable: Callable[[Mapping[str, Number]], Number] | None = None
        elif callable(self.value):
            self._callable = self.value
            self._expr = None
        else:  # pragma: no cover - defensive
            raise TypeError("value must be a string expression or callable")

    def _resolve_value(self, theta: Mapping[str, Number]) -> Number:
        if self._expr is not None:
            return self._expr.evaluate(theta)
        if self._callable is not None:
            return float(self._callable(theta))
        raise RuntimeError("No evaluator available")

    def apply(
        self,
        root: ET.Element,
        theta: Mapping[str, Number],
        ctx: BuildContext,
    ) -> None:
        nodes = root.findall(self.xpath, namespaces=ctx.namespaces or None)
        if not nodes:
            if self.required:
                raise ValueError(f"No nodes matched XPath: {self.xpath}")
            return

        text = ctx.format_value(self._resolve_value(theta))
        for node in nodes:
            node.text = text


Binding = ParameterBinding | EvaluationBinding


@dataclass
class FebTemplate:
    """FEBio template plus bindings applied during rendering."""

    template_path: Path | str
    bindings: list[Binding] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.template_path = Path(self.template_path)
        if not self.template_path.exists():
            raise FileNotFoundError(f"Template not found: {self.template_path}")
        # Normalise bindings to a list for repeated iteration
        self.bindings = list(self.bindings)

    def add_binding(self, binding: Binding) -> None:
        """Register an additional binding for this template."""
        self.bindings.append(binding)

    def render(
        self,
        theta: Mapping[str, Number],
        ctx: BuildContext | None = None,
    ) -> ET.ElementTree:
        """Render the template into an XML tree using the supplied parameters."""
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
        """Render and write the template to ``out_path``."""
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
        """Render the template into the case subfolder and return the FEB path."""
        ctx = ctx or BuildContext()
        sim_dir = Path(out_root) / self.subfolder
        sim_dir.mkdir(parents=True, exist_ok=True)
        base_name = out_name or Path(self.template.template_path).name
        feb_path = sim_dir / base_name
        return self.template.write(theta, feb_path, ctx)


__all__ = [
    "BuildContext",
    "ParameterBinding",
    "EvaluationBinding",
    "FebTemplate",
    "FebBuilder",
]
