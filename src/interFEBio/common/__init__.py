"""Shared helper types for FEBio bindings."""

from .base import Annotated, FEBioEntity, describe_range, indent_xml, validate_range
from .regions import (
    NodeSetRef,
    SurfaceRef,
    ElementSetRef,
    coerce_nodeset,
    coerce_surface,
    coerce_elementset,
)

__all__ = [
    'Annotated',
    'FEBioEntity',
    'describe_range',
    'indent_xml',
    'validate_range',
    'NodeSetRef',
    'SurfaceRef',
    'ElementSetRef',
    'coerce_nodeset',
    'coerce_surface',
    'coerce_elementset',
]
