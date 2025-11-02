"""Simulation adapters for reading FEBio output artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Tuple

import numpy as np


@dataclass
class SimulationAdapter:
    """Callable-based adapter that extracts data from FEBio artefacts.

    Attributes:
        reader: Callable invoked with the path to the `.xplt` file.
    """

    reader: Callable[[Path], Tuple[np.ndarray, np.ndarray]]

    def read(self, xplt_path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Return the simulated series for a given `.xplt` file.

        Args:
            xplt_path: Path to the simulation output file.

        Returns:
            Tuple with the simulated abscissa and ordinate arrays.
        """
        return self.reader(Path(xplt_path))


__all__ = ["SimulationAdapter"]
