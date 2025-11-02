"""Experimental data helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence, Tuple

import numpy as np

WeightFunction = Callable[[np.ndarray], np.ndarray]


@dataclass
class ExperimentSeries:
    """Simple container for experimental x/y data and optional weights."""

    x: np.ndarray
    y: np.ndarray
    weight: WeightFunction | None = None

    def weighted(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self.weight is None:
            return self.x, self.y, None
        weights = self.weight(self.x)
        return self.x, self.y, np.asarray(weights, dtype=float)


__all__ = ["ExperimentSeries", "WeightFunction"]
