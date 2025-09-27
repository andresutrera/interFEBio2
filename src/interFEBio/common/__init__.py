"""Shared helper types for FEBio bindings."""

from .regions import (
    NodeSetRef,
    SurfaceRef,
    ElementSetRef,
    coerce_nodeset,
    coerce_surface,
    coerce_elementset,
)

__all__ = [
    'NodeSetRef',
    'SurfaceRef',
    'ElementSetRef',
    'coerce_nodeset',
    'coerce_surface',
    'coerce_elementset',
]
