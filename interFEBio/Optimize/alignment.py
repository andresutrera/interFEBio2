from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Literal, Optional, Tuple

import numpy as np


GridPolicy = Literal["exp_to_sim", "sim_to_exp", "fixed_common", "fixed_user"]


@dataclass
class EvaluationGrid:
    policy: GridPolicy
    common_grid: Optional[np.ndarray] = None

    def select_grid(self, x_exp: np.ndarray, x_sim: np.ndarray) -> np.ndarray:
        if self.policy == "exp_to_sim":
            return np.asarray(x_sim, dtype=float)
        if self.policy == "sim_to_exp":
            return np.asarray(x_exp, dtype=float)
        if self.policy == "fixed_common":
            if self.common_grid is None:
                raise ValueError("common_grid must be provided for fixed_common policy")
            return np.asarray(self.common_grid, dtype=float)
        if self.policy == "fixed_user":
            if self.common_grid is None:
                raise ValueError("common_grid must be provided for fixed_user policy")
            return np.asarray(self.common_grid, dtype=float)
        raise ValueError(f"Unknown grid policy: {self.policy}")


class Aligner:
    def __init__(self, kind: str = "linear", fill_value: float = 0.0):
        self.kind = kind
        self.fill_value = fill_value

    def map(self, x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
        x_src = np.asarray(x_src, dtype=float)
        y_src = np.asarray(y_src, dtype=float)
        x_tgt = np.asarray(x_tgt, dtype=float)
        if x_src.size == 0:
            return np.full_like(x_tgt, self.fill_value)
        if self.kind == "nearest":
            indices = np.searchsorted(x_src, x_tgt, side="left")
            indices = np.clip(indices, 0, len(x_src) - 1)
            return y_src[indices]
        return np.interp(x_tgt, x_src, y_src, left=self.fill_value, right=self.fill_value)


class ResidualAssembler:
    def __init__(
        self,
        grid: EvaluationGrid,
        aligner: Optional[Aligner] = None,
        weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ):
        self.grid = grid
        self.aligner = aligner or Aligner()
        self.weight_fn = weight_fn

    def assemble(
        self,
        experiments: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]],
        simulations: Dict[str, Tuple[np.ndarray, np.ndarray]],
    ) -> Tuple[np.ndarray, Dict[str, slice]]:
        residuals: List[np.ndarray] = []
        slices: Dict[str, slice] = {}
        offset = 0
        for name, (x_exp, y_exp, weight) in experiments.items():
            if name not in simulations:
                continue
            x_sim, y_sim = simulations[name]
            target = self.grid.select_grid(x_exp, x_sim)
            y_exp_interp = self.aligner.map(x_exp, y_exp, target)
            y_sim_interp = self.aligner.map(x_sim, y_sim, target)
            res = y_sim_interp - y_exp_interp
            if weight is not None:
                if weight.shape != res.shape:
                    res = res * np.interp(target, x_exp, weight, left=1.0, right=1.0)
                else:
                    res = res * weight
            elif self.weight_fn is not None:
                weights = self.weight_fn(target)
                res = res * weights
            residuals.append(res)
            slices[name] = slice(offset, offset + len(res))
            offset += len(res)
        if not residuals:
            return np.array([], dtype=float), {}
        return np.concatenate(residuals), slices

