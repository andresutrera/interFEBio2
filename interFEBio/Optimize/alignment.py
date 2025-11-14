"""Utilities for selecting evaluation grids and aligning simulation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


GridPolicy = Literal["exp_to_sim", "sim_to_exp", "fixed_user"]


@dataclass
class EvaluationGrid:
    """Policy-driven evaluation grid selection.

    Attributes:
        policy: Grid policy that controls how experimental and simulation domains are matched.
        common_grid: Shared grid used when the policy is ``"fixed_user"``.
    """

    policy: GridPolicy
    common_grid: Array | None = None

    def select_grid(self, x_exp: Array, x_sim: Array) -> Array:
        """Choose the evaluation grid for an experiment/simulation pair.

        Args:
            x_exp: Experimental abscissa samples.
            x_sim: Simulation abscissa samples.

        Returns:
            Grid where residuals should be evaluated.
        """
        if self.policy == "exp_to_sim":
            return cast(Array, np.asarray(x_sim, dtype=float))
        if self.policy == "sim_to_exp":
            return cast(Array, np.asarray(x_exp, dtype=float))
        if self.policy == "fixed_user":
            if self.common_grid is None:
                raise ValueError("common_grid must be provided for fixed_user policy")
            return cast(Array, np.asarray(self.common_grid, dtype=float))
        raise ValueError(f"Unknown grid policy: {self.policy}")


class Aligner:
    """Interpolation helper that projects data onto a target grid."""

    def __init__(self, kind: str = "linear", fill_value: float = 0.0):
        """Configure interpolation behaviour.

        Args:
            kind: Interpolation scheme, ``"linear"`` or ``"nearest"``.
            fill_value: Value used outside the known domain.
        """
        self.kind = kind
        self.fill_value = fill_value

    def map(self, x_src: Array, y_src: Array, x_tgt: Array) -> Array:
        """Interpolate source data onto a target grid.

        Args:
            x_src: Source abscissa samples.
            y_src: Source ordinate samples.
            x_tgt: Target grid where values are required.

        Returns:
            Interpolated ordinate values on ``x_tgt``.
        """
        x_src = cast(Array, np.asarray(x_src, dtype=float).reshape(-1))
        y_src = cast(Array, np.asarray(y_src, dtype=float).reshape(-1))
        x_tgt = cast(Array, np.asarray(x_tgt, dtype=float).reshape(-1))
        if x_src.size == 0:
            return cast(Array, np.full_like(x_tgt, self.fill_value))
        if x_src.shape != y_src.shape:
            raise ValueError(
                f"x and y must have the same length; got {x_src.shape} and {y_src.shape}"
            )
        if self.kind == "nearest":
            indices = np.searchsorted(x_src, x_tgt, side="left")
            indices = np.clip(indices, 0, len(x_src) - 1)
            return cast(Array, y_src[indices])
        return cast(
            Array,
            np.interp(
                x_tgt,
                x_src,
                y_src,
                left=self.fill_value,
                right=self.fill_value,
            ),
        )
