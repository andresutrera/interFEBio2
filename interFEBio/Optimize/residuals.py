"""Residual assembly utilities for comparing experiments and simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .alignment import Aligner, EvaluationGrid


WeightFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class ResidualAssembler:
    """
    Align simulated results to experimental data and compute residual vectors.

    Parameters
    ----------
    grid
        Policy for choosing the evaluation grid shared between experiments and simulations.
    aligner
        Interpolation helper used to project data onto the chosen grid.
    weight_fn
        Optional callable that produces weights for the residual vector given the grid.
    """

    grid: EvaluationGrid
    aligner: Aligner = field(default_factory=Aligner)
    weight_fn: Optional[WeightFunction] = None

    def __post_init__(self) -> None:
        if self.aligner is None:
            self.aligner = Aligner()

    def assemble(
        self,
        experiments: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        simulations: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, Dict[str, slice]]:
        residuals: List[np.ndarray] = []
        slices: Dict[str, slice] = {}
        offset = 0
        for name, (x_exp, y_exp, weight) in experiments.items():
            sim = simulations.get(name)
            if sim is None:
                continue

            x_sim, y_sim = sim
            target = self.grid.select_grid(x_exp, x_sim)
            y_exp_interp = self.aligner.map(x_exp, y_exp, target)
            y_sim_interp = self.aligner.map(x_sim, y_sim, target)
            res = y_sim_interp - y_exp_interp

            if weight is not None:
                if weight.shape != res.shape:
                    interpolated = np.interp(target, x_exp, weight, left=1.0, right=1.0)
                    res = res * interpolated
                else:
                    res = res * weight
            elif self.weight_fn is not None:
                weights = self.weight_fn(target)
                res = res * weights

            residuals.append(res)
            slices[name] = slice(offset, offset + res.size)
            offset += res.size

        if not residuals:
            return np.array([], dtype=float), {}
        return np.concatenate(residuals), slices


__all__ = ["ResidualAssembler"]
