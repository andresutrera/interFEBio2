"""Value-type helpers used across FEBio bindings.

These classes mirror common FEBio structures (``vec3d``, ``mat3d`` and
``mat3ds``) and take care of converting to the textual representations that the
XML serialiser expects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple


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

    def __str__(self) -> str:
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

    def __str__(self) -> str:
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
            self.xx, self.xy, self.xz,
            self.xy, self.yy, self.yz,
            self.xz, self.yz, self.zz,
        )

    def __str__(self) -> str:
        return f"{self.xx}, {self.yy}, {self.zz}, {self.xy}, {self.yz}, {self.xz}"


def _coerce_triplet(values: Iterable[float]) -> tuple[float, float, float]:
    triplet = tuple(values)
    if len(triplet) != 3:
        raise ValueError("Expected three values to build a Vec3d")
    return tuple(float(component) for component in triplet)


__all__ = ["Vec3d", "mat3d", "mat3ds"]
