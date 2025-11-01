"""Experimental data helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

ArrayLike = Sequence[float]
WeightFunction = Callable[[ArrayLike], np.ndarray]


@dataclass
class ExperimentSeries:
    """Simple container for experimental x/y data and optional weights."""

    x: np.ndarray
    y: np.ndarray
    weight: Optional[WeightFunction] = None

    def weighted(self) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        if self.weight is None:
            return self.x, self.y, None
        weights = self.weight(self.x)
        return self.x, self.y, np.asarray(weights, dtype=float)


__all__ = ["ExperimentSeries", "WeightFunction"]

