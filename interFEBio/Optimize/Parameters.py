# interFEBio/Optimize/parameters.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple
import math
import numpy as np

Array = np.ndarray


@dataclass
class ParameterSpace:
    """
    Single reparam across all params:
      theta_i = theta0_i * xi**phi_i
      dtheta_i/dphi_i = theta_i * ln(xi)

    Optimizer works in phi-space. FEM uses theta-space.
    """

    names: List[str]
    theta0: Dict[str, float]  # initial thetas per name
    xi: float = 10.0  # shared base for all parameters
    vary: Optional[Dict[str, bool]] = None
    theta_bounds: Optional[Dict[str, Tuple[Optional[float], Optional[float]]]] = (
        None  # bounds in theta-space
    )

    def __post_init__(self):
        if self.xi <= 0.0:
            raise ValueError("xi must be > 0.")
        if self.xi == 1.0:
            raise ValueError("xi == 1 makes dθ/dφ = 0.")
        # arrays in declared order
        self._theta0_vec = np.asarray(
            [float(self.theta0[k]) for k in self.names], dtype=float
        )
        self._vary_vec = np.asarray(
            [
                (True if self.vary is None else bool(self.vary.get(k, True)))
                for k in self.names
            ],
            dtype=bool,
        )
        # bounds in theta-space to arrays
        if self.theta_bounds is None:
            self._th_lo = None
            self._th_hi = None
        else:
            lo = []
            hi = []
            for k in self.names:
                b = self.theta_bounds.get(k, (None, None))
                lo.append(-np.inf if b[0] is None else float(b[0]))
                hi.append(+np.inf if b[1] is None else float(b[1]))
            self._th_lo = np.asarray(lo, dtype=float)
            self._th_hi = np.asarray(hi, dtype=float)

        self._ln_xi = math.log(self.xi)

    # ---------- mapping ----------
    def theta_from_phi(self, phi_vec: Array) -> Array:
        phi_vec = np.asarray(phi_vec, dtype=float)
        return self._theta0_vec * np.power(self.xi, phi_vec)

    def dtheta_dphi(self, phi_vec: Array) -> Array:
        theta = self.theta_from_phi(phi_vec)
        return theta * self._ln_xi

    def phi_from_theta(self, theta_vec: Array) -> Array:
        theta_vec = np.asarray(theta_vec, dtype=float)
        ratio = theta_vec / self._theta0_vec
        if np.any(ratio <= 0.0):
            raise ValueError("theta must be > 0 elementwise to invert mapping.")
        return np.log(ratio) / self._ln_xi

    # ---------- bounds ----------
    def phi_bounds(self) -> Optional[Tuple[Array, Array]]:
        """Transform theta-bounds to phi-bounds for scipy."""
        if self._th_lo is None and self._th_hi is None:
            return None
        lo = np.full(len(self.names), -np.inf, dtype=float)
        hi = np.full(len(self.names), +np.inf, dtype=float)
        if self._th_lo is not None:
            mask = np.isfinite(self._th_lo)
            lo[mask] = self.phi_from_theta(self._th_lo)[mask]
        if self._th_hi is not None:
            mask = np.isfinite(self._th_hi)
            hi[mask] = self.phi_from_theta(self._th_hi)[mask]
        return lo, hi

    def clamp_theta(self, theta_vec: Array) -> Array:
        if self._th_lo is None and self._th_hi is None:
            return theta_vec
        out = np.asarray(theta_vec, dtype=float).copy()
        if self._th_lo is not None:
            out = np.maximum(out, self._th_lo)
        if self._th_hi is not None:
            out = np.minimum(out, self._th_hi)
        return out

    # ---------- masks and packing ----------
    def active_mask(self) -> Array:
        return self._vary_vec.copy()

    def pack_dict(self, d: Dict[str, float]) -> Array:
        return np.asarray([float(d[k]) for k in self.names], dtype=float)

    def unpack_vec(self, v: Sequence[float]) -> Dict[str, float]:
        return {k: float(x) for k, x in zip(self.names, np.asarray(v, dtype=float))}

    # ---------- Jacobian / gradient transforms ----------
    # J_theta: (m, n) = ∂r/∂theta ; returns ∂r/∂phi
    def J_theta_to_phi(self, J_theta: Array, phi_vec: Array) -> Array:
        scale = self.dtheta_dphi(phi_vec)  # (n,)
        return J_theta * scale[np.newaxis, :]

    # g_theta: (n,) = ∂f/∂theta ; returns ∂f/∂phi
    def grad_theta_to_phi(self, g_theta: Array, phi_vec: Array) -> Array:
        return g_theta * self.dtheta_dphi(phi_vec)

    # ---------- wrappers for scipy ----------
    def wrap_residual(
        self, residual_theta: Callable[[Array], Array]
    ) -> Callable[[Array], Array]:
        """phi -> r(phi) = r(theta(phi))"""

        def fun(phi_vec: Array) -> Array:
            theta = self.theta_from_phi(phi_vec)
            theta = self.clamp_theta(theta)
            return residual_theta(theta)

        return fun

    def wrap_jacobian(
        self,
        residual_theta: Callable[[Array], Array],
        jac_theta: Optional[Callable[[Array], Array]] = None,
    ) -> Callable[[Array], Array]:
        """
        If jac_theta is provided: return analytic J_phi.
        Else: return directional transform using finite differences on phi not provided here.
        """
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
