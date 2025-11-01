"""Finite-difference Jacobian helper used by the optimisation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np

Array = np.ndarray


@dataclass
class JacobianComputer:
    """
    Simple forward-difference Jacobian generator.

    Parameters
    ----------
    perturbation
        Step size applied to each optimisation parameter (in φ-space).
    parallel
        Present for compatibility; the :class:`Engine` manages concurrency via
        the runner job pool, so this flag is informational only.
    """

    perturbation: float = 1e-6
    parallel: bool = True

    def compute(
        self,
        phi0: Array,
        theta_fn: Callable[[Array], Array],
        residual_fn: Callable[[Array, Optional[str]], Array],
        *,
        label_fn: Optional[Callable[[int], Optional[str]]] = None,
        base_residual: Optional[Array] = None,
    ) -> Tuple[Array, Array]:
        phi0 = np.asarray(phi0, dtype=float)
        theta0 = theta_fn(phi0)
        base_label = label_fn(-1) if label_fn is not None else None

        if base_residual is None:
            r0 = np.asarray(residual_fn(theta0, base_label), dtype=float)
        else:
            r0 = np.asarray(base_residual, dtype=float)

        n = len(phi0)
        J = np.zeros((r0.size, n), dtype=float)

        for i in range(n):
            phi = phi0.copy()
            phi[i] += self.perturbation
            theta = theta_fn(phi)
            lbl = label_fn(i) if label_fn is not None else None
            ri = np.asarray(residual_fn(theta, lbl), dtype=float)
            J[:, i] = (ri - r0) / self.perturbation

        return r0, J


__all__ = ["JacobianComputer"]

