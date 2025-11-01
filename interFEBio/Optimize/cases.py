"""Lightweight helpers for preparing FEBio simulation cases."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

import numpy as np

from .adapters import SimulationAdapter
from .experiments import ExperimentSeries
from .feb_bindings import BuildContext, FebBuilder, FebTemplate


@dataclass
class SimulationCase:
    template: FebTemplate
    subfolder: str
    experiments: Mapping[str, ExperimentSeries]
    adapters: Mapping[str, SimulationAdapter]
    _builder: FebBuilder = field(init=False, repr=False)

    def __post_init__(self) -> None:
        missing = set(self.experiments.keys()) - set(self.adapters.keys())
        if missing:
            raise ValueError(f"Missing adapters for experiments: {sorted(missing)}")
        self._builder = FebBuilder(self.template, subfolder=self.subfolder)

    def prepare(
        self,
        theta: Mapping[str, float],
        out_root: Path,
        ctx: Optional[BuildContext] = None,
        out_name: Optional[str] = None,
    ) -> Path:
        ctx = ctx or BuildContext()
        target_name = out_name or f"{self.subfolder}.feb"
        return self._builder.build(
            theta=dict(theta),
            out_root=str(out_root),
            ctx=ctx,
            out_name=target_name,
        )

    def collect(self, feb_path: Path) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        xplt_path = feb_path.with_suffix(".xplt")
        results: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, adapter in self.adapters.items():
            results[name] = adapter.read(xplt_path)
        return results
