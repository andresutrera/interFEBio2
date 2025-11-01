from __future__ import annotations

import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

from .alignment import ResidualAssembler
from .feb_bindings import FebTemplate, BuildContext
from .runners import LocalParallelRunner

Array = np.ndarray


class FebBuilderCallable:
    def __call__(self, theta: Mapping[str, float], out_dir: Path) -> Tuple[Path, Path]:
        raise NotImplementedError


class ResidualEvaluator:
    def __call__(self, theta: Mapping[str, float]) -> Array:
        raise NotImplementedError


@dataclass
class JacobianComputer:
    perturbation: float
    max_workers: int = 4

    def compute(
        self,
        phi0: Array,
        theta_fn: Callable[[Array], Array],
        residual_fn: Callable[..., Array],
        label_fn: Optional[Callable[[int], Optional[str]]] = None,
    ) -> Tuple[Array, Array]:
        phi0 = np.asarray(phi0, dtype=float)
        theta0 = theta_fn(phi0)
        label0 = label_fn(-1) if label_fn is not None else None
        r0 = self._call_residual(residual_fn, theta0, label0)
        n = len(phi0)
        J = np.zeros((r0.size, n), dtype=float)

        def evaluate(i: int) -> Tuple[int, Array]:
            phi = phi0.copy()
            phi[i] += self.perturbation
            theta = theta_fn(phi)
            lbl = label_fn(i) if label_fn is not None else None
            r = self._call_residual(residual_fn, theta, lbl)
            diff = (r - r0) / self.perturbation
            return i, diff

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = [ex.submit(evaluate, i) for i in range(n)]
            for fut in concurrent.futures.as_completed(futures):
                idx, column = fut.result()
                J[:, idx] = column

        return r0, J

    @staticmethod
    def _call_residual(
        residual_fn: Callable[..., Array],
        theta: Array,
        label: Optional[str],
    ) -> Array:
        try:
            if label is not None:
                return np.asarray(residual_fn(theta, label), dtype=float)
        except TypeError:
            # residual_fn does not accept label; fall back to positional-only call
            pass
        return np.asarray(residual_fn(theta), dtype=float)
