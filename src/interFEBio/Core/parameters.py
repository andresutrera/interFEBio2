"""FEBio model parameter helpers.

These classes mimic the behaviour of FEBio's ``FEModelParam`` hierarchy
enough for the generated Python bindings to work with constant values while
leaving room for future mesh-mapped implementations.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Optional, TypeVar

from .value_types import Vec3d, mat3d, mat3ds

T = TypeVar("T")


@dataclass(slots=True)
class MeshMappedValue:
    """Placeholder for a mesh-dependent parameter definition.

    FEBio supports feeding parameters from data fields defined on a mesh. The
    Python bindings do not yet provide an implementation for exporting those
    datasets, but carrying the metadata around makes it straightforward to plug
    in once the data-path is decided.
    """

    dataset: str
    source: Optional[Path] = None

    @classmethod
    def from_table(cls, dataset: str, source: Path) -> "MeshMappedValue":
        """Register a data table to be hooked up later."""

        return cls(dataset=dataset, source=source)

    def reference(self) -> str:
        """Return a human-readable placeholder for XML output."""

        return f"@mesh:{self.dataset}"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.reference()


@dataclass(slots=True)
class ModelParameter(Generic[T]):
    """Base helper for FEBio parameters supporting const or mapped values."""

    constant: Optional[T] = None
    mapping: MeshMappedValue | str | None = None
    scale: float = 1.0
    domain: Optional[str] = None

    def __post_init__(self) -> None:
        if isinstance(self.mapping, str):
            self.mapping = MeshMappedValue(self.mapping)
        if self.constant is not None and self.mapping is not None:
            raise ValueError("ModelParameter cannot be initialised with both constant and mapping")
        if self.constant is not None:
            self.constant = self._coerce_constant(self.constant)

    def set_constant(self, value: T) -> None:
        self.constant = self._coerce_constant(value)
        self.mapping = None

    def use_mapping(self, mapping: MeshMappedValue | str) -> None:
        if isinstance(mapping, str):
            mapping = MeshMappedValue(mapping)
        self.mapping = mapping
        self.constant = None

    def is_constant(self) -> bool:
        return self.mapping is None and self.constant is not None

    def is_mapped(self) -> bool:
        return self.mapping is not None

    def xml_value(self) -> str:
        if self.mapping is not None:
            return self.mapping.reference()
        if self.constant is None:
            return ""
        return str(self.constant)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.xml_value()

    def _coerce_constant(self, value: T) -> T:
        return value


class FEParamDouble(ModelParameter[float]):
    """Scalar FEBio parameter supporting const or mapped values."""

    def _coerce_constant(self, value: float | int | str) -> float:
        if isinstance(value, str):
            return float(value)
        return float(value)


class FEParamVec3(ModelParameter[Vec3d]):
    """Vector-valued FEBio parameter."""

    def _coerce_constant(self, value: Vec3d | IterableABC[float]) -> Vec3d:
        if isinstance(value, Vec3d):
            return value
        return Vec3d.from_iterable(value)


class FEParamMat3d(ModelParameter[mat3d]):
    """3x3 matrix FEBio parameter."""

    def _coerce_constant(
        self,
        value: mat3d
        | mat3ds
        | IterableABC[IterableABC[float]]
        | IterableABC[float],
    ) -> mat3d:
        if isinstance(value, mat3d):
            return value
        if isinstance(value, mat3ds):
            return value.to_mat3d()
        if isinstance(value, IterableABC):
            return _coerce_mat3d(value)
        raise TypeError("Unsupported value for FEParamMat3d")


class FEParamMat3ds(ModelParameter[mat3ds]):
    """Symmetric 3x3 matrix FEBio parameter."""

    def _coerce_constant(self, value: mat3ds | mat3d | IterableABC[float]) -> mat3ds:
        if isinstance(value, mat3ds):
            return value
        if isinstance(value, mat3d):
            rows = value.rows()
            return mat3ds(
                rows[0][0],
                rows[1][1],
                rows[2][2],
                rows[0][1],
                rows[1][2],
                rows[0][2],
            )
        if isinstance(value, IterableABC):
            data = tuple(value)
            if len(data) != 6:
                raise ValueError("Iterable must provide six values for mat3ds")
            return mat3ds(*[float(component) for component in data])
        raise TypeError("Unsupported value for FEParamMat3ds")


def _coerce_mat3d(value: IterableABC[IterableABC[float]] | IterableABC[float]) -> mat3d:
    """Convert nested or flat iterables into a ``mat3d`` instance."""

    try:
        rows = list(value)
    except TypeError as exc:  # pragma: no cover - defensive
        raise TypeError("mat3d parameter expects an iterable") from exc
    if not rows:
        raise ValueError("Cannot build mat3d from an empty iterable")
    if isinstance(rows[0], IterableABC):
        return mat3d.from_rows(rows)  # type: ignore[arg-type]
    if len(rows) != 9:
        raise ValueError("Flat iterable must contain nine values for mat3d")
    floats = [float(component) for component in rows]
    return mat3d(*floats)


__all__ = [
    "MeshMappedValue",
    "ModelParameter",
    "FEParamDouble",
    "FEParamVec3",
    "FEParamMat3d",
    "FEParamMat3ds",
]
