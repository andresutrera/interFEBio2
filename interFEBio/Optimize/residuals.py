"""Residual assembly utilities for comparing experiments and simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, cast

import numpy as np
from numpy.typing import NDArray

from .alignment import Aligner, EvaluationGrid


Array = NDArray[np.float64]
WeightFunction = Callable[[Array], Array]


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
    weight_fn: WeightFunction | None = None

    def __post_init__(self) -> None:
        """Ensure an aligner instance is available."""
        if self.aligner is None:
            self.aligner = Aligner()

    def assemble(
        self,
        experiments: Dict[str, tuple[Array, Array, Array | None]],
        simulations: Dict[str, tuple[Array, Array]],
    ) -> tuple[Array, Dict[str, slice]]:
        """Return concatenated residuals and slice metadata."""
        residuals, slices, _ = self.assemble_with_details(experiments, simulations)
        return residuals, slices

    def assemble_with_details(
        self,
        experiments: Dict[str, tuple[Array, Array, Array | None]],
        simulations: Dict[str, tuple[Array, Array]],
    ) -> tuple[Array, Dict[str, slice], Dict[str, Dict[str, Array | None]]]:
        """Return residuals together with per-experiment alignment details."""
        residuals: List[Array] = []
        slices: Dict[str, slice] = {}
        details: Dict[str, Dict[str, Array | None]] = {}
        offset = 0
        for name, (x_exp, y_exp, weight) in experiments.items():
            sim = simulations.get(name)
            if sim is None:
                continue

            x_sim, y_sim = sim
            target = self.grid.select_grid(x_exp, x_sim)
            y_exp_interp = self.aligner.map(x_exp, y_exp, target)
            y_sim_interp = self.aligner.map(x_sim, y_sim, target)
            delta = y_sim_interp - y_exp_interp
            res = delta.copy()
            weights_applied: Array | None = None

            if weight is not None:
                if weight.shape != res.shape:
                    interpolated = np.interp(target, x_exp, weight, left=1.0, right=1.0)
                    weights_applied = cast(Array, np.asarray(interpolated, dtype=float))
                else:
                    weights_applied = cast(Array, np.asarray(weight, dtype=float))
            elif self.weight_fn is not None:
                weights_applied = self.weight_fn(target)

            if weights_applied is not None:
                res = res * weights_applied

            residuals.append(cast(Array, res))
            slices[name] = slice(offset, offset + res.size)
            details[name] = {
                "grid": target,
                "y_exp": y_exp_interp,
                "y_sim": y_sim_interp,
                "residual": delta,
                "weights": weights_applied,
            }
            offset += res.size

        if not residuals:
            empty = cast(Array, np.array([], dtype=float))
            return empty, {}, {}
        return cast(Array, np.concatenate(residuals)), slices, details


__all__ = ["ResidualAssembler"]
