"""Utilities for selecting evaluation grids and aligning simulation data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from scipy.interpolate import interp1d
from numpy.typing import NDArray

Array = NDArray[np.float64]

GridPolicy = Literal["exp_to_sim", "fixed_user"]


@dataclass
class EvaluationGrid:
    """Policy-driven evaluation grid selection.

    ``exp_to_sim`` preserves the experimental sampling and ensures simulations
    are compared on that grid. ``fixed_user`` enforces a custom user-provided
    grid, enabling shared abscissa among multiple experiments.
    """

    policy: GridPolicy
    common_grid: Array | None = None

    def select_grid(self, x_exp: Array, x_sim: Array) -> Array:
        """Return the abscissa where residuals should be evaluated."""
        if self.policy == "exp_to_sim":
            return cast(Array, np.asarray(x_exp, dtype=float))
        if self.policy == "fixed_user":
            if self.common_grid is None:
                raise ValueError("common_grid must be provided for fixed_user policy")
            return cast(Array, np.asarray(self.common_grid, dtype=float))
        raise ValueError(f"Unknown grid policy: {self.policy}")


class Aligner:
    """Thin wrapper above :func:`scipy.interpolate.interp1d` style API."""

    def map(self, x_src: Array, y_src: Array, x_tgt: Array) -> Array:
        """Project the ``(x_src, y_src)`` samples onto ``x_tgt``."""
        x_src = cast(Array, np.asarray(x_src, dtype=float).reshape(-1))
        y_src = cast(Array, np.asarray(y_src, dtype=float).reshape(-1))
        x_tgt = cast(Array, np.asarray(x_tgt, dtype=float).reshape(-1))
        if x_src.size == 0:
            return cast(Array, np.zeros_like(x_tgt))
        if x_src.shape != y_src.shape:
            raise ValueError(
                f"x and y must have the same length; got {x_src.shape} and {y_src.shape}"
            )
        if x_src.size == 1:
            return cast(Array, np.full_like(x_tgt, y_src[0]))
        order = np.argsort(x_src)
        x_src = x_src[order]
        y_src = y_src[order]
        interpolator = interp1d(
            x_src,
            y_src,
            kind="linear",
            fill_value="extrapolate",
            bounds_error=False,
        )
        return cast(Array, interpolator(x_tgt))
