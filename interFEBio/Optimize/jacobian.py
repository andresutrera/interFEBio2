"""Finite-difference Jacobian helper used by the optimisation engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast
import inspect

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


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
        residual_fn: Callable[[Array, str | None], Array],
        *,
        label_fn: Callable[[int], str | None] | None = None,
        base_residual: Array | None = None,
    ) -> tuple[Array, Array]:
        """Compute residuals and a forward-difference Jacobian."""
        phi0 = cast(Array, np.asarray(phi0, dtype=float))
        theta0 = theta_fn(phi0)
        base_label = label_fn(-1) if label_fn is not None else None

        accepts_label = self._residual_accepts_label(residual_fn)

        if base_residual is None:
            r0 = cast(
                Array,
                np.asarray(
                    self._call_residual(
                        residual_fn,
                        theta0,
                        base_label,
                        accepts_label,
                    ),
                    dtype=float,
                ),
            )
        else:
            r0 = cast(Array, np.asarray(base_residual, dtype=float))

        n = len(phi0)
        J = cast(Array, np.zeros((r0.size, n), dtype=float))

        for i in range(n):
            phi = phi0.copy()
            phi[i] += self.perturbation
            theta = theta_fn(phi)
            lbl = label_fn(i) if label_fn is not None else None
            ri = cast(
                Array,
                np.asarray(
                    self._call_residual(residual_fn, theta, lbl, accepts_label),
                    dtype=float,
                ),
            )
            J[:, i] = (ri - r0) / self.perturbation

        return r0, J

    @staticmethod
    def _residual_accepts_label(residual_fn: Callable[..., Array]) -> bool:
        """Return True when the residual callable accepts a label argument."""
        sig = inspect.signature(residual_fn)
        positional = [
            p
            for p in sig.parameters.values()
            if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)
        ]
        if len(positional) >= 2:
            return True
        for p in sig.parameters.values():
            if p.kind == p.VAR_POSITIONAL:
                return True
        return False

    @staticmethod
    def _call_residual(
        residual_fn: Callable[..., Array],
        theta: Array,
        label: str | None,
        accepts_label: bool,
    ) -> Array:
        """Dispatch to the residual callable, passing the label when required."""
        if accepts_label:
            return residual_fn(theta, label)
        return residual_fn(theta)


__all__ = ["JacobianComputer"]
