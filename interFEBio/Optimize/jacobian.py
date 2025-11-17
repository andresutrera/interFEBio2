"""Finite-difference Jacobian helper used by the optimisation engine."""

from __future__ import annotations

import inspect
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Callable, cast

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
        When ``True`` the evaluator distributes column perturbations across a
        worker pool so FEBio simulations can run concurrently.
    max_workers
        Optional override for the number of worker threads used for parallel
        evaluation. Defaults to ``os.cpu_count()`` when unset.
    """

    perturbation: float = 1e-6
    parallel: bool = True
    max_workers: int | None = None

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

        worker_count = self._worker_count(n)
        if worker_count > 1:
            with ThreadPoolExecutor(max_workers=worker_count) as pool:
                futures = {
                    pool.submit(
                        self._evaluate_column,
                        i,
                        phi0,
                        theta_fn,
                        residual_fn,
                        accepts_label,
                        label_fn(i) if label_fn is not None else None,
                    ): i
                    for i in range(n)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    ri = future.result()
                    J[:, idx] = (ri - r0) / self.perturbation
        else:
            for i in range(n):
                ri = self._evaluate_column(
                    i,
                    phi0,
                    theta_fn,
                    residual_fn,
                    accepts_label,
                    label_fn(i) if label_fn is not None else None,
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

    def _worker_count(self, n: int) -> int:
        if not self.parallel:
            return 1
        if self.max_workers is not None:
            return max(1, min(int(self.max_workers), n))
        cpu_count = os.cpu_count() or 1
        return max(1, min(cpu_count, n))

    def _evaluate_column(
        self,
        index: int,
        phi0: Array,
        theta_fn: Callable[[Array], Array],
        residual_fn: Callable[[Array, str | None], Array],
        accepts_label: bool,
        label: str | None,
    ) -> Array:
        phi = phi0.copy()
        phi[index] += self.perturbation
        theta = theta_fn(phi)
        return cast(
            Array,
            np.asarray(
                self._call_residual(residual_fn, theta, label, accepts_label),
                dtype=float,
            ),
        )


__all__ = ["JacobianComputer"]
