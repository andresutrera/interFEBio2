"""Case definitions for FEBio optimisation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np

from .adapters import SimulationAdapter
from .experiments import ExperimentSeries
from .feb_bindings import BuildContext, FebBuilder, FebTemplate
from .options import GridPolicyOptions


@dataclass
class SimulationCase:
    """Container describing how to generate and collect a FEBio simulation.

    Attributes:
        template: FEBio template and parameter bindings for this case.
        subfolder: Directory name used when generating simulation files.
        experiments: Experimental series that should be compared against simulations.
        adapters: Mapping of experiment identifiers to adapters that read simulation data.
        omp_threads: Number of OpenMP threads to advertise to FEBio processes.
        grids: Optional mapping configuring the evaluation grid policy per
            experiment.
    """

    template: FebTemplate
    subfolder: str
    experiments: Mapping[str, ExperimentSeries]
    adapters: Mapping[str, SimulationAdapter]
    omp_threads: int | None = None
    grids: Mapping[str, Any] | None = None
    _builder: FebBuilder = field(init=False, repr=False)
    _grid_policies: Dict[str, GridPolicyOptions] = field(
        init=False, repr=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        """Initialise helper objects and validate experiment coverage."""
        missing = set(self.experiments.keys()) - set(self.adapters.keys())
        if missing:
            raise ValueError(f"Missing adapters for experiments: {sorted(missing)}")
        if self.omp_threads is not None:
            threads = int(self.omp_threads)
            if threads <= 0:
                raise ValueError("omp_threads must be a positive integer.")
            self.omp_threads = threads
        self._builder = FebBuilder(self.template, subfolder=self.subfolder)
        raw_configs = dict(self.grids or {})
        for name in self.experiments.keys():
            self._grid_policies[name] = self._coerce_grid_options(raw_configs.get(name))

    @staticmethod
    def _coerce_grid_options(value: Any) -> GridPolicyOptions:
        if isinstance(value, GridPolicyOptions):
            return value
        if isinstance(value, str):
            return GridPolicyOptions(policy=value)
        if isinstance(value, Mapping):
            return GridPolicyOptions(**value)
        return GridPolicyOptions()

    def prepare(
        self,
        theta: Mapping[str, float],
        out_root: Path,
        ctx: BuildContext | None = None,
        out_name: str | None = None,
    ) -> Path:
        """Render a FEB file populated with the provided parameters.

        Args:
            theta: Mapping of parameter names to θ-space values.
            out_root: Directory where generated files should be stored.
            ctx: Optional FEB builder context with formatting preferences.
            out_name: Name of the generated FEB file.

        Returns:
            Absolute path to the generated FEB file.
        """
        ctx = ctx or BuildContext()
        target_name = out_name or f"{self.subfolder}.feb"
        return self._builder.build(
            theta=dict(theta),
            out_root=str(out_root),
            ctx=ctx,
            out_name=target_name,
        )

    def collect(self, feb_path: Path) -> Dict[str, tuple[np.ndarray, np.ndarray]]:
        """Read back simulation data produced by FEBio.

        Args:
            feb_path: Path to the FEB file used for the simulation run.

        Returns:
            Mapping from experiment identifier to simulated x/y arrays.
        """
        xplt_path = feb_path.with_suffix(".xplt")
        results: Dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, adapter in self.adapters.items():
            results[name] = adapter.read(xplt_path)
        return results

    def environment(self) -> Dict[str, str]:
        """Return environment overrides for this simulation.

        Returns:
            Mapping with per-case environment definitions.
        """
        env: Dict[str, str] = {}
        if self.omp_threads is not None:
            env["OMP_NUM_THREADS"] = str(self.omp_threads)
        return env

    def grid_policy(self, experiment: str) -> GridPolicyOptions:
        """Return the configured grid policy for a given experiment."""
        return self._grid_policies.get(experiment, GridPolicyOptions())
