from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


GridPolicy = Literal["exp_to_sim", "sim_to_exp", "fixed_user"]


@dataclass
class EvaluationGrid:
    policy: GridPolicy
    common_grid: Array | None = None

    def select_grid(self, x_exp: Array, x_sim: Array) -> Array:
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
    def __init__(self, kind: str = "linear", fill_value: float = 0.0):
        self.kind = kind
        self.fill_value = fill_value

    def map(self, x_src: Array, y_src: Array, x_tgt: Array) -> Array:
        x_src = cast(Array, np.asarray(x_src, dtype=float))
        y_src = cast(Array, np.asarray(y_src, dtype=float))
        x_tgt = cast(Array, np.asarray(x_tgt, dtype=float))
        if x_src.size == 0:
            return cast(Array, np.full_like(x_tgt, self.fill_value))
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
