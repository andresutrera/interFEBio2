"""Case definitions for FEBio optimisation workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from .adapters import SimulationAdapter
from .alignment import Aligner, EvaluationGrid
from .experiments import ExperimentSeries
from .feb_bindings import BuildContext, FebBuilder, FebTemplate
from .options import GridPolicyOptions
from .residuals import ResidualAssembler
from .runners import RunHandle, Runner

Array = NDArray[np.float64]


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
            return GridPolicyOptions(
                policy=cast(Literal["exp_to_sim", "fixed_user"], value)
            )
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


@dataclass(slots=True)
class CaseJob:
    case: SimulationCase
    feb_path: Path
    handle: RunHandle


@dataclass(slots=True)
class EvaluationResult:
    residual: Array
    metrics: Dict[str, Any]
    series: Dict[str, Dict[str, Any]]


class CaseEvaluator:
    """Launch FEBio simulations and assemble residuals for all configured cases."""

    def __init__(self, cases: Sequence[SimulationCase], runner: Runner, logger) -> None:
        self.runner = runner
        self.logger = logger
        self.aligner = Aligner()
        self.residual_assembler = ResidualAssembler(
            grid=EvaluationGrid(policy="exp_to_sim"), aligner=self.aligner
        )
        self._cases = list(cases)
        self._prepared_experiments: Dict[
            int, Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]
        ] = {}
        for case in self._cases:
            prepared: Dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]] = {}
            for name, series in case.experiments.items():
                x_exp, y_exp, weight = series.weighted()
                x_arr = cast(np.ndarray, np.asarray(x_exp, dtype=float))
                y_arr = cast(np.ndarray, np.asarray(y_exp, dtype=float))
                w_arr = (
                    None
                    if weight is None
                    else cast(np.ndarray, np.asarray(weight, dtype=float))
                )
                prepared[name] = (x_arr, y_arr, w_arr)
            self._prepared_experiments[id(case)] = prepared

    def describe_cases(self) -> List[Mapping[str, Any]]:
        descriptors: List[Mapping[str, Any]] = []
        for case in self._cases:
            experiments = sorted(case.experiments.keys())
            entry: Dict[str, Any] = {
                "subfolder": case.subfolder,
                "experiments": experiments,
            }
            if case.omp_threads is not None:
                entry["omp_threads"] = int(case.omp_threads)
            descriptors.append(entry)
        return descriptors

    def evaluate(
        self,
        theta: Mapping[str, float],
        iter_dir: Path,
        *,
        label: str | None = None,
        track_series: bool = True,
    ) -> EvaluationResult:
        jobs = self.launch_jobs(theta, iter_dir, label)
        return self.finalize_jobs(jobs, track_series=track_series)

    def launch_jobs(
        self,
        theta: Mapping[str, float],
        iter_dir: Path,
        label: str | None,
    ) -> List[CaseJob]:
        theta_values = {k: float(v) for k, v in theta.items()}
        iter_dir.mkdir(parents=True, exist_ok=True)
        jobs: List[CaseJob] = []
        for case in self._cases:
            base_name = f"{case.subfolder}{label}" if label else case.subfolder
            feb_path = case.prepare(theta_values, iter_dir, out_name=f"{base_name}.feb")
            handle = self.runner.run(
                feb_path.parent, feb_path.name, env=case.environment() or None
            )
            jobs.append(CaseJob(case=case, feb_path=feb_path, handle=handle))
        return jobs

    def finalize_jobs(
        self,
        jobs: Sequence[CaseJob],
        *,
        track_series: bool,
    ) -> EvaluationResult:
        residuals: List[Array] = []
        r_squared: Dict[str, float] = {}
        all_exp: List[Array] = []
        all_sim: List[Array] = []
        series_latest: Dict[str, Dict[str, Any]] = {}

        for job in jobs:
            try:
                result = job.handle.wait()
            except KeyboardInterrupt:
                self.runner.shutdown()
                raise
            if result.exit_code != 0:
                raise RuntimeError(
                    f"FEBio exited with code {result.exit_code} for case '{job.case.subfolder}'"
                )
            simulations = job.case.collect(job.feb_path)
            overrides = self._build_target_grids(job.case, simulations)
            residual, _, details = self.residual_assembler.assemble_with_details(
                self._prepared_experiments[id(job.case)],
                simulations,
                target_grids=overrides or None,
            )
            if residual.size:
                residuals.append(cast(Array, residual))
            for exp_name, info in details.items():
                y_exp = cast(Array, np.asarray(info["y_exp"], dtype=float))
                y_sim = cast(Array, np.asarray(info["y_sim"], dtype=float))
                if y_exp.size == 0 or y_sim.size == 0:
                    continue
                all_exp.append(y_exp)
                all_sim.append(y_sim)
                diff = y_exp - y_sim
                ss_res = float(np.dot(diff, diff))
                mean_exp = float(np.mean(y_exp))
                centered = y_exp - mean_exp
                ss_tot = float(np.dot(centered, centered))
                if ss_tot > 0.0:
                    r2 = 1.0 - ss_res / ss_tot
                else:
                    r2 = 1.0 if ss_res == 0.0 else float("nan")
                key = f"{job.case.subfolder}/{exp_name}"
                r_squared[key] = r2
                if not track_series:
                    continue
                try:
                    grid_vals = info.get("grid")
                    y_exp_vals = info.get("y_exp")
                    y_sim_vals = info.get("y_sim")
                    if (
                        grid_vals is not None
                        and y_exp_vals is not None
                        and y_sim_vals is not None
                    ):
                        series_latest[key] = {
                            "x": _to_list(grid_vals),
                            "y_exp": _to_list(y_exp_vals),
                            "y_sim": _to_list(y_sim_vals),
                        }
                except Exception:
                    continue

        nrmse = float("nan")
        if all_exp and all_sim:
            exp_concat = np.concatenate(all_exp)
            sim_concat = np.concatenate(all_sim)
            if exp_concat.size and sim_concat.size:
                mse = float(np.mean((sim_concat - exp_concat) ** 2))
                rmse = float(np.sqrt(mse))
                data_range = float(exp_concat.max() - exp_concat.min())
                if data_range > 0.0:
                    nrmse = rmse / data_range
                else:
                    nrmse = 0.0 if rmse == 0.0 else float("nan")

        metrics = {"nrmse": nrmse, "r_squared": r_squared}
        if not residuals:
            residual_array = cast(Array, np.array([], dtype=float))
        else:
            residual_array = cast(Array, np.concatenate(residuals))
        return EvaluationResult(
            residual=residual_array,
            metrics=metrics,
            series=series_latest if track_series else {},
        )

    def _build_target_grids(
        self,
        case: SimulationCase,
        simulations: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> Dict[str, Array]:
        grids: Dict[str, Array] = {}
        prepared = self._prepared_experiments.get(id(case), {})
        for exp_name in prepared.keys():
            exp_entry = prepared.get(exp_name)
            sim_entry = simulations.get(exp_name)
            if exp_entry is None or sim_entry is None:
                continue
            policy = case.grid_policy(exp_name)
            grids[exp_name] = self._resolve_grid(
                policy,
                exp_entry[0],
                sim_entry[0],
                f"{case.subfolder}/{exp_name}",
                exp_name,
            )
        return grids

    def _resolve_grid(
        self,
        policy: GridPolicyOptions,
        x_exp: Array,
        x_sim: Array,
        exp_label: str,
        adapter_label: str,
    ) -> Array:
        if policy.policy == "exp_to_sim":
            return cast(Array, np.asarray(x_exp, dtype=float))
        if policy.policy == "fixed_user":
            if policy.values is None:
                raise ValueError("fixed_user grid policy requires explicit values.")
            grid = cast(Array, np.asarray(list(policy.values), dtype=float).reshape(-1))
            self._check_grid_bounds(exp_label, adapter_label, "experiment", grid, x_exp)
            self._check_grid_bounds(exp_label, adapter_label, "simulation", grid, x_sim)
            return grid
        raise ValueError(f"Unsupported grid policy: {policy.policy}")

    def _check_grid_bounds(
        self,
        exp_label: str,
        adapter_label: str,
        source_label: str,
        grid: Array,
        samples: Array,
        tol: float = 1e-6,
    ) -> None:
        if grid.size == 0 or samples.size == 0:
            return
        g_min = float(np.min(grid))
        g_max = float(np.max(grid))
        s_min = float(np.min(samples))
        s_max = float(np.max(samples))
        if g_min < s_min - tol or g_max > s_max + tol:
            message = (
                f"\n[{exp_label} | {adapter_label}] fixed_user grid span "
                f"[{g_min:.6f}, {g_max:.6f}] extends beyond {source_label} x-range "
                f"[{s_min:.6f}, {s_max:.6f}]; values will be extrapolated"
            )
            self.logger.warning(message)


def _to_list(data: Any) -> List[float]:
    array = np.asarray(data, dtype=float).reshape(-1)
    return cast(List[float], array.tolist())


__all__ = [
    "SimulationCase",
    "CaseEvaluator",
    "CaseJob",
    "EvaluationResult",
]
