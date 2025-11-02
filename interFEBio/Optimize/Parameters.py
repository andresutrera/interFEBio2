"""Parameter reparameterisation utilities for optimisation workflows."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Mapping, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


@dataclass(frozen=True)
class Parameter:
    """
    Scalar optimisation parameter metadata.

    Attributes:
        name: Identifier used in θ-space mappings and FEB templates.
        theta0: Initial value in θ-space.
        vary: Flag indicating whether this parameter is optimised.
        bounds: Optional lower/upper limits in θ-space.
    """

    name: str
    theta0: float
    vary: bool = True
    bounds: tuple[float | None, float | None] = (None, None)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Parameter name may not be empty.")
        if not math.isfinite(self.theta0):
            raise ValueError("theta0 must be finite.")
        if self.bounds is None:
            object.__setattr__(self, "bounds", (None, None))


class ParameterSpace:
    """
    Mapping between φ (optimiser space) and θ (physical parameters).

    Each parameter θ_i is related to its optimisation counterpart φ_i via::

        θ_i = θ0_i * ξ**φ_i

    The exponential reparameterisation keeps θ positive while allowing unconstrained
    optimisation in φ-space. Parameters can be supplied either through the legacy
    constructor arguments or incrementally via :meth:`add_parameter`.
    """

    def __init__(
        self,
        names: Sequence[str] | None = None,
        theta0: Mapping[str, float] | None = None,
        *,
        xi: float = 10.0,
        vary: Mapping[str, bool] | None = None,
        theta_bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
        parameters: Iterable[Parameter] | None = None,
    ):
        if xi <= 0.0:
            raise ValueError("xi must be > 0.")
        if xi == 1.0:
            raise ValueError("xi == 1 makes dθ/dφ = 0.")

        self.xi = float(xi)
        self._ln_xi = math.log(self.xi)

        self._parameters: List[Parameter] = []
        self._theta0_map: Dict[str, float] = {}
        self._vary_map: Dict[str, bool] = {}
        self._bounds_map: Dict[str, tuple[float | None, float | None]] = {}

        if parameters is not None:
            for spec in parameters:
                self._append_parameter(spec, rebuild=False)

        if names is not None or theta0 is not None:
            if names is None or theta0 is None:
                raise ValueError("names and theta0 must be provided together.")
            for name in names:
                if name not in theta0:
                    raise ValueError(f"Missing theta0 entry for parameter '{name}'.")
                spec = Parameter(
                    name=name,
                    theta0=float(theta0[name]),
                    vary=True if vary is None else bool(vary.get(name, True)),
                    bounds=(
                        theta_bounds.get(name, (None, None))
                        if theta_bounds is not None
                        else (None, None)
                    ),
                )
                self._append_parameter(spec, rebuild=False)

        self._rebuild_cache()

    # ---------- parameter management ----------
    def add_parameter(
        self,
        parameter: Parameter | None = None,
        *,
        name: str | None = None,
        theta0: float | None = None,
        vary: bool = True,
        bounds: tuple[float | None, float | None] | None = None,
    ) -> Parameter:
        """Register a new optimisation parameter.

        Parameters can be supplied either as a :class:`Parameter` instance or through
        the keyword arguments ``name``/``theta0``/``vary``/``bounds``.
        """
        if parameter is None:
            if name is None or theta0 is None:
                raise ValueError("Provide a Parameter instance or name/theta0 pair.")
            parameter = Parameter(
                name=name,
                theta0=float(theta0),
                vary=bool(vary),
                bounds=bounds if bounds is not None else (None, None),
            )
        elif not isinstance(parameter, Parameter):
            raise TypeError("parameter must be a Parameter instance.")

        self._append_parameter(parameter, rebuild=True)
        return parameter

    def parameters(self) -> List[Parameter]:
        """Return a copy of the registered parameter specifications."""
        return list(self._parameters)

    # ---------- mapping ----------
    def theta_from_phi(self, phi_vec: Array) -> Array:
        """Map φ values to θ-space."""
        phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
        return cast(Array, self._theta0_vec * np.power(self.xi, phi_vec))

    def dtheta_dphi(self, phi_vec: Array) -> Array:
        """Return ∂θ/∂φ for the provided φ vector."""
        theta = self.theta_from_phi(phi_vec)
        return cast(Array, theta * self._ln_xi)

    def phi_from_theta(self, theta_vec: Array) -> Array:
        """Map θ values back into φ-space."""
        theta_vec = cast(Array, np.asarray(theta_vec, dtype=float))
        ratio = theta_vec / self._theta0_vec
        if np.any(ratio <= 0.0):
            raise ValueError("theta must be > 0 elementwise to invert mapping.")
        return cast(Array, np.log(ratio) / self._ln_xi)

    # ---------- bounds ----------
    def phi_bounds(self) -> tuple[Array, Array] | None:
        """Transform θ-space bounds into φ-space bounds."""
        if self._th_lo is None and self._th_hi is None:
            return None
        lo = cast(Array, np.full(len(self._names), -np.inf, dtype=float))
        hi = cast(Array, np.full(len(self._names), +np.inf, dtype=float))
        if self._th_lo is not None:
            mask = np.isfinite(self._th_lo)
            lo[mask] = self.phi_from_theta(self._th_lo)[mask]
        if self._th_hi is not None:
            mask = np.isfinite(self._th_hi)
            hi[mask] = self.phi_from_theta(self._th_hi)[mask]
        return lo, hi

    def clamp_theta(self, theta_vec: Array) -> Array:
        """Clamp θ values according to stored bounds."""
        if self._th_lo is None and self._th_hi is None:
            return theta_vec
        out = cast(Array, np.asarray(theta_vec, dtype=float).copy())
        if self._th_lo is not None:
            out = np.maximum(out, self._th_lo)
        if self._th_hi is not None:
            out = np.minimum(out, self._th_hi)
        return cast(Array, out)

    # ---------- masks and packing ----------
    def active_mask(self) -> BoolArray:
        """Return a boolean mask describing which parameters vary."""
        return cast(BoolArray, self._vary_vec.copy())

    def pack_dict(self, d: Mapping[str, float]) -> Array:
        """Pack a parameter dictionary into a vector ordered by ``names``."""
        return cast(Array, np.asarray([float(d[k]) for k in self._names], dtype=float))

    def unpack_vec(self, v: Sequence[float]) -> Dict[str, float]:
        """Convert a vector into a parameter dictionary."""
        return {k: float(x) for k, x in zip(self._names, np.asarray(v, dtype=float))}

    # ---------- Jacobian / gradient transforms ----------
    # J_theta: (m, n) = ∂r/∂theta ; returns ∂r/∂phi
    def J_theta_to_phi(self, J_theta: Array, phi_vec: Array) -> Array:
        scale = self.dtheta_dphi(phi_vec)  # (n,)
        return cast(Array, J_theta * scale[np.newaxis, :])

    # g_theta: (n,) = ∂f/∂theta ; returns ∂f/∂phi
    def grad_theta_to_phi(self, g_theta: Array, phi_vec: Array) -> Array:
        return cast(Array, g_theta * self.dtheta_dphi(phi_vec))

    # ---------- wrappers for scipy ----------
    def wrap_residual(
        self, residual_theta: Callable[[Array], Array]
    ) -> Callable[[Array], Array]:
        """Produce a residual function defined in φ-space."""

        def fun(phi_vec: Array) -> Array:
            theta = self.theta_from_phi(phi_vec)
            theta = self.clamp_theta(theta)
            return residual_theta(theta)

        return fun

    def wrap_jacobian(
        self,
        residual_theta: Callable[[Array], Array],
        jac_theta: Callable[[Array], Array] | None = None,
    ) -> Callable[[Array], Array]:
        """Produce a Jacobian function defined in φ-space."""
        if jac_theta is None:
            # user can provide their own FD in phi-space; we keep interface explicit
            raise ValueError(
                "jac_theta not provided; supply your own jacobian in phi-space."
            )

        def jac(phi_vec: Array) -> Array:
            theta = self.theta_from_phi(phi_vec)
            theta = self.clamp_theta(theta)
            Jt = jac_theta(theta)  # (m, n) wrt theta
            return self.J_theta_to_phi(Jt, phi_vec)

        return jac

    # ---------- compatibility helpers ----------
    @property
    def names(self) -> List[str]:
        return list(self._names)

    @property
    def theta0(self) -> Dict[str, float]:
        return dict(self._theta0_map)

    @property
    def theta_bounds(self) -> Dict[str, tuple[float | None, float | None]]:
        """Return θ-space bounds keyed by parameter name."""
        return dict(self._bounds_map)

    @property
    def vary(self) -> Dict[str, bool]:
        return dict(self._vary_map)

    # ---------- internal helpers ----------
    def _append_parameter(self, spec: Parameter, rebuild: bool) -> None:
        name = spec.name
        if name in self._theta0_map:
            raise ValueError(f"Parameter '{name}' already registered.")

        self._parameters.append(spec)
        self._theta0_map[name] = float(spec.theta0)
        self._vary_map[name] = bool(spec.vary)
        bounds = spec.bounds if spec.bounds is not None else (None, None)
        if len(bounds) != 2:
            raise ValueError("bounds must be a (low, high) tuple.")
        self._bounds_map[name] = tuple(bounds)  # type: ignore[assignment]

        if rebuild:
            self._rebuild_cache()

    def _rebuild_cache(self) -> None:
        if not self._parameters:
            self._names = []
            self._theta0_vec = cast(Array, np.asarray([], dtype=float))
            self._vary_vec = cast(BoolArray, np.asarray([], dtype=bool))
            self._th_lo = None
            self._th_hi = None
            return

        self._names = [p.name for p in self._parameters]
        self._theta0_vec = cast(
            Array,
            np.asarray([float(p.theta0) for p in self._parameters], dtype=float),
        )
        self._vary_vec = cast(
            BoolArray,
            np.asarray([bool(p.vary) for p in self._parameters], dtype=bool),
        )

        lo_vals = []
        hi_vals = []
        any_lo = False
        any_hi = False
        for p in self._parameters:
            lo, hi = p.bounds if p.bounds is not None else (None, None)
            lo_vals.append(-np.inf if lo is None else float(lo))
            hi_vals.append(+np.inf if hi is None else float(hi))
            any_lo = any_lo or (lo is not None)
            any_hi = any_hi or (hi is not None)

        self._th_lo = cast(Array, np.asarray(lo_vals, dtype=float)) if any_lo else None
        self._th_hi = cast(Array, np.asarray(hi_vals, dtype=float)) if any_hi else None


__all__ = ["Parameter", "ParameterSpace"]
