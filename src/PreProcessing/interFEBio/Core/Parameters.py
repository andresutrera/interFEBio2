"""FEBio parameter helpers and basic math value types.

This module consolidates helper dataclasses that emulate common FEBio math
types (``vec3d``, ``mat3d`` and ``mat3ds``) together with the parameter wrappers
used throughout the generated bindings.
"""

from __future__ import annotations

from collections.abc import Iterable as IterableABC
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Iterable, Optional, Tuple, TypeVar


@dataclass(slots=True)
class Vec3d:
    """Lightweight three-component vector."""

    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @classmethod
    def from_iterable(cls, values: Iterable[float]) -> "Vec3d":
        x, y, z = _coerce_triplet(values)
        return cls(x, y, z)

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.x}, {self.y}, {self.z}"


@dataclass(slots=True)
class mat3d:
    """Row-major representation of a 3x3 matrix."""

    m11: float = 0.0
    m12: float = 0.0
    m13: float = 0.0
    m21: float = 0.0
    m22: float = 0.0
    m23: float = 0.0
    m31: float = 0.0
    m32: float = 0.0
    m33: float = 0.0

    @classmethod
    def from_rows(cls, rows: Iterable[Iterable[float]]) -> "mat3d":
        row_data = [tuple(row) for row in rows]
        if len(row_data) != 3 or any(len(row) != 3 for row in row_data):
            raise ValueError("mat3d.from_rows expects three rows with three values each")
        flat = [value for row in row_data for value in row]
        return cls(*flat)

    def rows(self) -> tuple[tuple[float, float, float], ...]:
        return (
            (self.m11, self.m12, self.m13),
            (self.m21, self.m22, self.m23),
            (self.m31, self.m32, self.m33),
        )

    def to_list(self) -> list[float]:
        return [value for row in self.rows() for value in row]

    def __str__(self) -> str:  # pragma: no cover - trivial
        return ", ".join(str(value) for value in self.to_list())


@dataclass(slots=True)
class mat3ds:
    """Symmetric 3x3 matrix stored in compact form."""

    xx: float = 0.0
    yy: float = 0.0
    zz: float = 0.0
    xy: float = 0.0
    yz: float = 0.0
    xz: float = 0.0

    def to_mat3d(self) -> mat3d:
        return mat3d(
            self.xx,
            self.xy,
            self.xz,
            self.xy,
            self.yy,
            self.yz,
            self.xz,
            self.yz,
            self.zz,
        )

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.xx}, {self.yy}, {self.zz}, {self.xy}, {self.yz}, {self.xz}"


def _coerce_triplet(values: Iterable[float]) -> tuple[float, float, float]:
    triplet = tuple(values)
    if len(triplet) != 3:
        raise ValueError("Expected three values to build a Vec3d")
    return tuple(float(component) for component in triplet)


T = TypeVar("T")


@dataclass(slots=True)
class MeshMappedValue:
    """Placeholder for a mesh-dependent parameter definition."""

    dataset: str
    source: Optional[Path] = None

    @classmethod
    def from_table(cls, dataset: str, source: Path) -> "MeshMappedValue":
        return cls(dataset=dataset, source=source)

    def reference(self) -> str:
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
        value: mat3d | mat3ds | IterableABC[IterableABC[float]] | IterableABC[float],
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
    "Vec3d",
    "mat3d",
    "mat3ds",
    "MeshMappedValue",
    "ModelParameter",
    "FEParamDouble",
    "FEParamVec3",
    "FEParamMat3d",
    "FEParamMat3ds",
]
