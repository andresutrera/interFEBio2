"""Adapters that load simulation results."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np


@dataclass
class SimulationAdapter:
    """Thin wrapper around a callable that reads FEBio outputs."""

    reader: Callable[[Path], Tuple[np.ndarray, np.ndarray]]

    def read(self, xplt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        return self.reader(Path(xplt_path))


__all__ = ["SimulationAdapter"]

