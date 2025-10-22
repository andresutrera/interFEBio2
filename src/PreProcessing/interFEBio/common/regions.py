"""Mesh entity references used by boundary conditions and loads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def _coerce_sequence(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


@dataclass(slots=True)
class NodeSetRef:
    """Reference to a FEBio node set by name or explicit node ids."""

    name: str | None = None
    node_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.name is None and not self.node_ids:
            raise ValueError("NodeSetRef requires a name or node ids")
        if self.node_ids:
            self.node_ids = _coerce_sequence(self.node_ids)

    def __str__(self) -> str:
        if self.name:
            return self.name
        return ", ".join(str(node) for node in self.node_ids or ())


@dataclass(slots=True)
class SurfaceRef:
    """Reference to a FEBio surface by name or explicit facet ids."""

    name: str | None = None
    facets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.name is None and not self.facets:
            raise ValueError("SurfaceRef requires a name or facet ids")
        if self.facets:
            self.facets = _coerce_sequence(self.facets)

    def __str__(self) -> str:
        if self.name:
            return self.name
        return ", ".join(str(fid) for fid in self.facets or ())


@dataclass(slots=True)
class ElementSetRef:
    """Reference to a FEBio element set."""

    name: str | None = None
    element_ids: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.name is None and not self.element_ids:
            raise ValueError("ElementSetRef requires a name or element ids")
        if self.element_ids:
            self.element_ids = _coerce_sequence(self.element_ids)

    def __str__(self) -> str:
        if self.name:
            return self.name
        return ", ".join(str(eid) for eid in self.element_ids or ())


def coerce_nodeset(target: NodeSetRef | str | Sequence[int] | None) -> NodeSetRef | None:
    if target is None:
        return None
    if isinstance(target, NodeSetRef):
        return target
    if isinstance(target, str):
        return NodeSetRef(name=target)
    if isinstance(target, Sequence):
        return NodeSetRef(node_ids=_coerce_sequence(target))
    raise TypeError(f"Unsupported node set reference type: {type(target)!r}")


def coerce_surface(target: SurfaceRef | str | Sequence[int] | None) -> SurfaceRef | None:
    if target is None:
        return None
    if isinstance(target, SurfaceRef):
        return target
    if isinstance(target, str):
        return SurfaceRef(name=target)
    if isinstance(target, Sequence):
        return SurfaceRef(facets=_coerce_sequence(target))
    raise TypeError(f"Unsupported surface reference type: {type(target)!r}")


def coerce_elementset(target: ElementSetRef | str | Sequence[int] | None) -> ElementSetRef | None:
    if target is None:
        return None
    if isinstance(target, ElementSetRef):
        return target
    if isinstance(target, str):
        return ElementSetRef(name=target)
    if isinstance(target, Sequence):
        return ElementSetRef(element_ids=_coerce_sequence(target))
    raise TypeError(f"Unsupported element set reference type: {type(target)!r}")


__all__ = [
    'NodeSetRef',
    'SurfaceRef',
    'ElementSetRef',
    'coerce_nodeset',
    'coerce_surface',
    'coerce_elementset',
]
