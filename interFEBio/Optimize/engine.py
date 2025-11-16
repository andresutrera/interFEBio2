"""Unified optimisation engine for FEBio parameter fitting."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    cast,
)

import numpy as np
from numpy.typing import NDArray
from prettytable import PrettyTable

from ..Log import Log
from ..monitoring.client import MonitorConfig, OptimizationMonitorClient
from .alignment import Aligner, EvaluationGrid, GridPolicy
from .cases import SimulationCase
from .jacobian import JacobianComputer
from .optimizers import (
    BoundsLike,
    OptimizerAdapter,
    ScipyLeastSquaresAdapter,
    ScipyMinimizeAdapter,
)
from .options import EngineOptions
from .Parameters import ParameterSpace
from .residuals import ResidualAssembler
from .runners import LocalParallelRunner, LocalSerialRunner, RunHandle, Runner
from .Storage import StorageManager

Array = NDArray[np.float64]


def _to_list(data: Any) -> List[float]:
    array = np.asarray(data, dtype=float).reshape(-1)
    return cast(List[float], array.tolist())


@dataclass
class OptimizeResult:
    phi: Array
    theta: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class _CaseState:
    case: SimulationCase
    experiments: Dict[str, tuple[Array, Array, Array | None]]
    name: str


class Engine:
    """Coordinate FEBio simulations and optimisation loops.

    The engine transforms φ-space parameters into θ-space values, renders FEBio input
    files, launches the simulations, and assembles residuals for the optimiser. It also
    keeps track of iteration artefacts and performs optional cleanup when the run ends.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        cases: Sequence[SimulationCase],
        *,
        options: EngineOptions | None = None,
    ) -> None:
        """Initialise the optimisation engine using structured configuration."""
        if not cases:
            raise ValueError("At least one SimulationCase is required.")

        self.options = options = options or EngineOptions()
        self.parameter_space = parameter_space
        self._param_names = list(self.parameter_space.names)
        grid_policy = options.grid.policy
        grid_values = options.grid.values
        jacobian_opts = options.jacobian
        cleanup_opts = options.cleanup
        runner_opts = options.runner
        storage_opts = options.storage
        monitor_opts = options.monitor
        optimizer_opts = options.optimizer
        self._reparam_enabled = bool(optimizer_opts.reparametrize)

        self.jacobian: JacobianComputer | None = (
            JacobianComputer(
                perturbation=float(jacobian_opts.perturbation),
                parallel=bool(jacobian_opts.parallel),
            )
            if jacobian_opts.enabled
            else None
        )
        storage_mode = storage_opts.mode.lower()
        self.storage_mode = storage_mode

        cleanup_previous = bool(cleanup_opts.remove_previous)
        cleanup_mode = cleanup_opts.mode.lower()
        if storage_mode == "tmp":
            cleanup_previous = True
            cleanup_mode = "all"
        if cleanup_mode not in {"none", "retain_best", "all"}:
            raise ValueError(
                "cleanup_mode must be one of: 'none', 'retain_best', 'all'"
            )
        self.cleanup_previous = bool(cleanup_previous)
        self.cleanup_mode = cleanup_mode
        optimizer_settings = dict(optimizer_opts.settings or {})
        self.optimizer_adapter = self._build_optimizer(
            optimizer_opts.name, optimizer_settings
        )
        self.grid = EvaluationGrid(
            policy=grid_policy,
            common_grid=None
            if grid_values is None
            else cast(Array, np.asarray(grid_values, dtype=float)),
        )
        self.aligner = Aligner()
        self.residual_assembler = ResidualAssembler(
            grid=self.grid, aligner=self.aligner
        )

        if storage_mode not in {"disk", "tmp"}:
            raise ValueError("storage_mode must be 'disk' or 'tmp'")
        storage_parent = (
            Path(storage_opts.root).expanduser()
            if storage_opts.root is not None
            else None
        )
        self.storage = StorageManager(
            parent=storage_parent,
            use_tmp=(storage_mode == "tmp"),
        )
        self.workdir = self.storage.resolve()
        if storage_mode == "tmp":
            if storage_opts.root is not None:
                persist_root = Path(storage_opts.root).expanduser().resolve()
            else:
                persist_root = Path.cwd() / self.workdir.name
            persist_root.mkdir(parents=True, exist_ok=True)
            self.persist_root = persist_root
        else:
            self.persist_root = self.workdir
        self._log_file = self._resolve_log_file(
            storage_opts.log_file, storage_opts.root
        )
        log_instance = Log(log_file=self._log_file)
        self._logger = log_instance.logger
        self._initMsg()

        runner_jobs = int(max(1, runner_opts.jobs))
        runner_command = tuple(runner_opts.command or ("febio4", "-i"))
        runner_env = dict(runner_opts.env) if runner_opts.env is not None else None
        self._runner_jobs = runner_jobs
        if runner_jobs <= 1:
            self.runner: Runner = LocalSerialRunner(
                command=runner_command, env=runner_env
            )
        else:
            self.runner = LocalParallelRunner(
                n_jobs=self._runner_jobs,
                command=runner_command,
                env=runner_env,
            )

        self._cases: List[_CaseState] = []
        self._case_grid_cache: Dict[str, Dict[str, Array]] = {}

        for case in cases:
            experiments: Dict[str, tuple[Array, Array, Array | None]] = {}
            grid_overrides: Dict[str, Array | None] = {}
            for name, series in case.experiments.items():
                x, y, weight = series.weighted()
                x_arr = cast(Array, np.asarray(x, dtype=float))
                y_arr = cast(Array, np.asarray(y, dtype=float))
                w_arr = (
                    None
                    if weight is None
                    else cast(Array, np.asarray(weight, dtype=float))
                )
                experiments[name] = (x_arr, y_arr, w_arr)
                grid_val = series.evaluation_grid()
                if grid_val is not None:
                    grid_overrides[name] = cast(
                        Array, np.asarray(grid_val, dtype=float).reshape(-1)
                    )
                else:
                    grid_overrides[name] = None
            case_name = self._resolve_case_name(case)
            self._cases.append(
                _CaseState(
                    case=case,
                    experiments=experiments,
                    name=case_name,
                )
            )
            self._case_grid_cache[case_name] = {
                exp_name: grid
                for exp_name, grid in grid_overrides.items()
                if grid is not None
            }

        self._progress_index = 0
        self._eval_index = 0
        self._iter_dirs: List[Path] = []
        self._last_phi: Array | None = None
        self._last_theta_vec: Array | None = None
        self._last_residual: Array | None = None
        self._last_iter_dir: Path | None = None
        self._last_metrics: Dict[str, Any] = {}
        self._monitor_enabled = bool(monitor_opts.enabled)
        self._monitor_socket = (
            Path(monitor_opts.socket).expanduser() if monitor_opts.socket else None
        )
        self._monitor_label = monitor_opts.label
        self._monitor_client: OptimizationMonitorClient | None = None
        self._series_latest: Dict[str, Dict[str, Any]] = {}
        self._cached_jac_phi: Array | None = None
        self._cached_jacobian: Array | None = None
        self._pending_initial_log = False
        self._log_progress = True

    # ------------------------------------------------------------------ public API
    def run(
        self,
        *,
        phi0: Sequence[float] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        verbose: bool = True,
        callbacks: Iterable[Callable[[Array, float], None]] | None = None,
    ) -> OptimizeResult:
        """Execute the optimisation and return the fitted parameters.

        Args:
            phi0: Initial guess in φ-space. Defaults to zeros if omitted.
            bounds: Bounds in φ-space supplied to the optimiser.
            verbose: Enable progress logging and tabular summaries.
            callbacks: Extra callbacks invoked every iteration with φ and cost.

        Returns:
            Optimisation result containing φ, θ, and optimiser metadata.
        """

        interrupted = False
        monitor_client = self._prepare_monitor_client()
        self._monitor_client = monitor_client
        try:
            self._progress_index = 0
            self._eval_index = 0
            self._iter_dirs = []
            self._last_phi = None
            self._last_theta_vec = None
            self._last_residual = None
            self._last_iter_dir = None
            self._last_metrics = {}
            self._pending_initial_log = True
            self._log_progress = bool(verbose)

            if phi0 is not None:
                phi0_vec = cast(Array, np.asarray(phi0, dtype=float))
            else:
                if self._reparam_enabled:
                    phi0_vec = cast(
                        Array,
                        np.zeros(len(self.parameter_space.names), dtype=float),
                    )
                else:
                    theta0_vec = self.parameter_space.pack_dict(
                        self.parameter_space.theta0
                    )
                    phi0_vec = cast(Array, np.asarray(theta0_vec, dtype=float))
            bounds_input: BoundsLike = bounds
            if bounds_input is None:
                bounds_input = (
                    self.parameter_space.phi_bounds()
                    if self._reparam_enabled
                    else self.parameter_space.theta_bounds_array()
                )
            if monitor_client is not None:
                try:
                    self._emit_monitor_start(monitor_client, phi0_vec, bounds_input)
                except Exception:
                    self._logger.exception("Failed to emit monitor run start event.")

            callback_list: List[Callable[[Array, float], None]] = list(callbacks or [])
            if verbose:
                callback_list.append(self._progress_printer())

            def residual_phi(phi_vec: Array) -> Array:
                phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
                residual = self._evaluate_residual(phi_vec)
                if self._pending_initial_log:
                    initial_cost = 0.5 * float(np.dot(residual, residual))
                    self._record_iteration_progress(
                        phi_vec,
                        initial_cost,
                        log_output=self._log_progress,
                    )
                    self._pending_initial_log = False
                return residual

            jac_fn: Callable[[Array], Array] | None = None
            if self.jacobian is not None:
                jac_fn = self._build_jacobian_wrapper()

            phi_opt, meta = self.optimizer_adapter.minimize(
                residual_phi,
                jac_fn,
                phi0_vec,
                bounds_input,
                callback_list,
            )

            phi_opt_array = cast(Array, np.asarray(phi_opt, dtype=float))
            theta_opt_vec = self._phi_to_theta(phi_opt_array)
            theta_opt = self.parameter_space.unpack_vec(theta_opt_vec.tolist())
            self._log_final_summary(phi_opt_array, theta_opt)
            if monitor_client is not None:
                try:
                    summary = self._build_run_summary(theta_opt, meta)
                    monitor_client.run_completed(summary=summary)
                except Exception:
                    self._logger.exception("Failed to emit monitor completion event.")
            try:
                self._write_series_outputs()
            except Exception:
                self._logger.exception("Failed to write series outputs.")
            return OptimizeResult(
                phi=phi_opt_array,
                theta=theta_opt,
                metadata=meta,
            )
        except KeyboardInterrupt:
            interrupted = True
            if monitor_client is not None:
                try:
                    monitor_client.run_failed(reason="interrupted")
                except Exception:
                    self._logger.exception("Failed to emit monitor interruption event.")
            raise
        except Exception as exc:
            if monitor_client is not None:
                try:
                    monitor_client.run_failed(reason=f"{exc.__class__.__name__}: {exc}")
                except Exception:
                    self._logger.exception("Failed to emit monitor failure event.")
            raise
        finally:
            try:
                self._final_cleanup(interrupted=interrupted)
            finally:
                self._monitor_client = None
                self.close()

    def close(self) -> None:
        """Shut down background workers and release runner resources."""
        try:
            self.runner.shutdown()
        except Exception:
            pass
        self._monitor_client = None

    # ------------------------------------------------------------------ monitoring helpers
    def _prepare_monitor_client(self) -> OptimizationMonitorClient | None:
        if not self._monitor_enabled:
            return None
        try:
            config = MonitorConfig(
                socket_path=self._monitor_socket,
                label=self._monitor_label or self._default_monitor_label(),
            )
            return OptimizationMonitorClient(config)
        except Exception:
            self._logger.exception(
                "Monitor initialisation failed; disabling monitoring."
            )
            return None

    def _default_monitor_label(self) -> str:
        for candidate in (self.persist_root.name, self.workdir.name):
            if candidate:
                return candidate
        return datetime.now().strftime("run-%Y%m%d%H%M%S")

    def _emit_monitor_start(
        self,
        monitor: OptimizationMonitorClient,
        phi0_vec: Array,
        bounds: tuple[Array, Array] | Sequence[tuple[float, float]] | None,
    ) -> None:
        theta0_vec = self._phi_to_theta(phi0_vec)
        parameters: Dict[str, Any] = {
            "names": list(self.parameter_space.names),
            "phi0": [float(x) for x in np.asarray(phi0_vec, dtype=float)],
            "theta0": [float(x) for x in np.asarray(theta0_vec, dtype=float)],
        }
        bounds_payload = self._serialise_bounds(bounds)
        if bounds_payload is not None:
            parameters["bounds"] = bounds_payload
        case_list = self._case_descriptors()
        cases: List[Mapping[str, Any]] = list(case_list)
        optimizer_meta = {"adapter": type(self.optimizer_adapter).__name__}
        meta = {
            "storage_root": str(self.persist_root),
            "runner_jobs": getattr(self, "_runner_jobs", None),
        }
        monitor.run_started(
            parameters=parameters,
            cases=cases,
            optimizer=optimizer_meta,
            meta={k: v for k, v in meta.items() if v is not None},
        )

    def _case_descriptors(self) -> List[Mapping[str, Any]]:
        descriptors: List[Mapping[str, Any]] = []
        for state in self._cases:
            experiments = sorted(state.case.experiments.keys())
            entry: Dict[str, Any] = {
                "subfolder": state.case.subfolder,
                "experiments": experiments,
            }
            omp_threads = state.case.omp_threads
            if omp_threads is not None:
                entry["omp_threads"] = int(omp_threads)
            descriptors.append(entry)
        return descriptors

    def _resolve_case_name(self, case: SimulationCase) -> str:
        """Return a stable directory name for a case."""
        name = (case.subfolder or "").strip()
        if name:
            return name
        template = getattr(case, "template", None)
        path = getattr(template, "template_path", None) if template else None
        if path:
            stem = Path(path).stem
            if stem:
                return stem
        return "case"

    def _serialise_bounds(
        self, bounds: tuple[Array, Array] | Sequence[tuple[float, float]] | None
    ) -> List[tuple[float, float]] | None:
        if bounds is None:
            return None
        if isinstance(bounds, tuple) and len(bounds) == 2:
            lo_arr, hi_arr = bounds
            lo_vec = np.asarray(lo_arr, dtype=float).reshape(-1)
            hi_vec = np.asarray(hi_arr, dtype=float).reshape(-1)
            return list(zip(lo_vec.tolist(), hi_vec.tolist()))
        serialised: List[tuple[float, float]] = []
        seq_bounds = cast(Sequence[tuple[float, float]], bounds)
        for pair in seq_bounds:
            try:
                lo, hi = pair
            except Exception:
                continue
            serialised.append((float(lo), float(hi)))
        return serialised

    def _sanitize_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        nrmse = metrics.get("nrmse")
        if isinstance(nrmse, (int, float)) and np.isfinite(nrmse):
            clean["nrmse"] = float(nrmse)
        r_squared = metrics.get("r_squared")
        if isinstance(r_squared, Mapping):
            clean["r_squared"] = {
                key: (
                    float(value)
                    if isinstance(value, (int, float)) and np.isfinite(value)
                    else None
                )
                for key, value in r_squared.items()
            }
        return clean

    def _simplify_meta(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        return {str(key): self._simplify_value(value) for key, value in data.items()}

    def _simplify_value(self, value: Any) -> Any:
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            limited_items = list(value.items())[:16]
            return {str(k): self._simplify_value(v) for k, v in limited_items}
        if isinstance(value, (list, tuple, set)):
            limited = list(value)[:16]
            return [self._simplify_value(item) for item in limited]
        if hasattr(value, "tolist"):
            arr = np.asarray(value)
            if arr.ndim == 0:
                scalar = arr.item()
                return (
                    float(scalar)
                    if isinstance(scalar, (int, float))
                    else self._simplify_value(scalar)
                )
            flat = arr.reshape(-1)
            limited = flat[:16].tolist()
            return [float(x) for x in limited]
        return str(value)

    def _build_run_summary(
        self,
        theta_opt: Mapping[str, float],
        optimizer_meta: Mapping[str, Any],
    ) -> Dict[str, Any]:
        summary: Dict[str, Any] = {
            "theta_opt": {name: float(val) for name, val in theta_opt.items()}
        }
        metrics = self._sanitize_metrics(self._last_metrics or {})
        if metrics:
            summary.update(metrics)
        final_meta = self._simplify_meta(dict(optimizer_meta or {}))
        if final_meta:
            summary["optimizer"] = final_meta
        return summary

    def _write_series_outputs(self) -> None:
        if not self._series_latest:
            return
        target_dir = self._log_file.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        for key, payload in self._series_latest.items():
            x_vals = payload.get("x")
            exp_vals = payload.get("y_exp")
            sim_vals = payload.get("y_sim")
            if not (x_vals and exp_vals and sim_vals):
                continue
            length = min(len(x_vals), len(exp_vals), len(sim_vals))
            if length == 0:
                continue
            safe_name = key.replace("/", "_")
            path = target_dir / f"{safe_name}_series.txt"
            lines = [f"# {key}\n", "# x y_exp y_sim\n"]
            for idx in range(length):
                lines.append(
                    f"{x_vals[idx]:.10g} {exp_vals[idx]:.10g} {sim_vals[idx]:.10g}\n"
                )
            path.write_text("".join(lines), encoding="utf-8")

    # ------------------------------------------------------------------ internals
    def _build_optimizer(
        self,
        name: str,
        options: Dict[str, Any] | None,
    ) -> OptimizerAdapter:
        """Create an optimiser adapter based on the supplied selector."""
        opts = dict(options or {})
        key = name.lower()
        if key == "least_squares":
            return ScipyLeastSquaresAdapter(**opts)
        if key == "minimize":
            method = opts.pop("method", "L-BFGS-B")
            return ScipyMinimizeAdapter(method=method, **opts)
        raise ValueError(f"Unsupported optimizer: {name}")

    def _resolve_log_file(
        self,
        log_file: str | Path | None,
        storage_root: str | Path | None,
    ) -> Path:
        """Determine the log file path and prepare parent directories."""
        path: Path | None = None
        if log_file is not None:
            candidate = str(log_file).strip()
            if candidate:
                path = Path(candidate).expanduser()

        if path is None:
            base: Path | None = None
            if storage_root is not None:
                storage_str = str(storage_root).strip()
                if storage_str:
                    base = Path(storage_str).expanduser()
            if base is not None:
                base.mkdir(parents=True, exist_ok=True)
                path = base / "optimization.log"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path.cwd() / f"optimization_{timestamp}.log"

        resolved = path.expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved

    def _progress_printer(self) -> Callable[[Array, float], None]:
        """Return a callback that logs iteration summaries."""
        def callback(phi_vec: Array, cost: float) -> None:
            self._record_iteration_progress(phi_vec, cost, log_output=True)

        return callback

    def _record_iteration_progress(
        self,
        phi_vec: Array,
        cost: float,
        *,
        log_output: bool,
    ) -> None:
        """Log and emit iteration metrics for the provided φ vector."""
        theta_vec = self._phi_to_theta(phi_vec)
        theta_dict: Dict[str, float] = {}
        for name, phi_value, theta_value in zip(
            self.parameter_space.names, phi_vec, theta_vec, strict=True
        ):
            theta_float = float(theta_value)
            theta_dict[name] = theta_float

        metrics = self._last_metrics or {}
        r_squared = metrics.get("r_squared", {})
        nrmse_val = metrics.get("nrmse")
        if log_output:
            table = PrettyTable()
            if self._reparam_enabled:
                table.field_names = ["parameter", "phi", "theta"]
            else:
                table.field_names = ["parameter", "theta"]
            for name, phi_value, theta_value in zip(
                self.parameter_space.names, phi_vec, theta_vec, strict=True
            ):
                theta_float = float(theta_value)
                if self._reparam_enabled:
                    table.add_row(
                        [
                            name,
                            f"{float(phi_value):.6e}",
                            f"{theta_float:.6e}",
                        ]
                    )
                else:
                    table.add_row(
                        [
                            name,
                            f"{theta_float:.6e}",
                        ]
                    )
            if nrmse_val is None or not np.isfinite(nrmse_val):
                nrmse_display = "nan"
            else:
                nrmse_display = f"{float(nrmse_val):.6e}"
            rsq_table = PrettyTable()
            rsq_table.field_names = ["case/experiment", "R^2"]
            if r_squared:
                for key in sorted(r_squared):
                    value = r_squared[key]
                    if value is None or not np.isfinite(value):
                        rsq_table.add_row([key, "nan"])
                    else:
                        rsq_table.add_row([key, f"{float(value):.6f}"])
            else:
                rsq_table.add_row(["-", "-"])
            self._logger.info(
                "\n[iter {iter:03d}] cost={cost:.6e} nrmse={nrmse}\n{param_table}\n{rsq_table}",
                iter=self._progress_index,
                cost=cost,
                nrmse=nrmse_display,
                param_table=table.get_string(),
                rsq_table=rsq_table.get_string(),
            )

        if self._monitor_client is not None:
            try:
                payload_metrics = self._sanitize_metrics(metrics)
                self._monitor_client.record_iteration(
                    index=self._progress_index,
                    cost=float(cost),
                    theta=theta_dict,
                    metrics=payload_metrics,
                    series=self._series_latest,
                )
            except Exception:
                self._logger.exception("Failed to emit monitor iteration event.")
        self._progress_index += 1
        self._cleanup_previous_iterations()

    def _jacobian_label_fn(self) -> Callable[[int], str | None]:
        names = self._param_names

        def label_fn(idx: int) -> str | None:
            if idx < 0:
                return "_base"
            if idx < len(names):
                return f"_{names[idx]}"
            return f"_col_{idx}"

        return label_fn

    def _build_jacobian_wrapper(self) -> Callable[[Array], Array]:
        """Wrap the Jacobian evaluator so it stays in sync with engine state."""
        label_fn = self._jacobian_label_fn()

        def jacobian(phi_vec: Array) -> Array:
            phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
            if self._last_phi is None or not np.allclose(phi_vec, self._last_phi):
                self._evaluate_residual(phi_vec)

            iter_dir = self._last_iter_dir
            if iter_dir is None:
                iter_dir = self._next_iter_dir()
            base_residual = self._last_residual

            assert self.jacobian is not None
            if self.jacobian.parallel:
                if (
                    self._cached_jac_phi is not None
                    and self._cached_jacobian is not None
                    and np.array_equal(phi_vec, self._cached_jac_phi)
                ):
                    return cast(Array, self._cached_jacobian.copy())
                if base_residual is None:
                    raise RuntimeError("Base residual not available for Jacobian.")
                jobs = self._schedule_jacobian_jobs(phi_vec, iter_dir, label_fn)
                J = self._finalize_jacobian_jobs(jobs, base_residual)
                self._cached_jac_phi = phi_vec.copy()
                self._cached_jacobian = J.copy()
                return J

            def residual_with_label(theta_vec: Array, lbl: str | None) -> Array:
                return self._residual_theta_vec(
                    theta_vec,
                    label=lbl,
                    iter_dir=iter_dir,
                )

            _, J = self.jacobian.compute(
                phi_vec,
                self._theta_vec,
                residual_with_label,
                label_fn=label_fn,
                base_residual=base_residual,
            )
            return J

        return jacobian

    def _theta_vec(self, phi_vec: Array) -> Array:
        """Map φ values to constrained θ values using the parameter space."""
        return self._phi_to_theta(phi_vec)

    def _phi_to_theta(self, phi_vec: Array) -> Array:
        """Convert optimiser coordinates into θ-space, applying bounds."""
        phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
        if self._reparam_enabled:
            theta_vec = self.parameter_space.theta_from_phi(phi_vec)
        else:
            theta_vec = phi_vec
        theta_vec = self.parameter_space.clamp_theta(theta_vec)
        return cast(Array, np.asarray(theta_vec, dtype=float))

    def _residual_theta_vec(
        self,
        theta_vec: Array,
        label: str | None = None,
        iter_dir: Path | None = None,
    ) -> Array:
        """Evaluate residuals for a θ vector, allocating an iteration directory."""
        theta_vec = cast(Array, np.asarray(theta_vec, dtype=float))
        theta_vec = self.parameter_space.clamp_theta(theta_vec)
        theta_dict = self.parameter_space.unpack_vec(theta_vec.tolist())
        target_dir = iter_dir or self._next_iter_dir()
        return self._execute_cases(theta_dict, target_dir, label)

    def _schedule_jacobian_jobs(
        self,
        phi_vec: Array,
        iter_dir: Path,
        label_fn: Callable[[int], str | None],
    ) -> Dict[int, List[tuple[_CaseState, Path, RunHandle, str]]]:
        """Launch Jacobian perturbation simulations without waiting."""
        assert self.jacobian is not None
        perturb = float(self.jacobian.perturbation)
        jobs_by_index: Dict[int, List[tuple[_CaseState, Path, RunHandle, str]]] = {}
        # self._logger.info(
        #     "Scheduling %d Jacobian columns under %s",
        #     len(self._param_names),
        #     iter_dir,
        # )
        for idx, name in enumerate(self._param_names):
            phi = phi_vec.copy()
            phi[idx] += perturb
            theta_vec = self._theta_vec(phi)
            theta_dict = self.parameter_space.unpack_vec(theta_vec.tolist())
            label = label_fn(idx)
            jobs = self._launch_case_jobs(
                theta_dict,
                iter_dir,
                label,
            )
            jobs_by_index[idx] = jobs
        return jobs_by_index

    def _finalize_jacobian_jobs(
        self,
        jobs_by_index: Dict[int, List[tuple[_CaseState, Path, RunHandle, str]]],
        base_residual: Array,
    ) -> Array:
        """Wait for perturbation simulations and assemble the Jacobian."""
        assert self.jacobian is not None
        perturb = float(self.jacobian.perturbation)
        J = cast(
            Array, np.zeros((base_residual.size, len(self._param_names)), dtype=float)
        )
        for idx, jobs in jobs_by_index.items():
            residual = self._finalize_case_jobs(
                jobs,
                track_series=False,
                grid_overrides=self._case_grid_cache,
            )
            J[:, idx] = (residual - base_residual) / perturb
        return J

    def _next_iter_dir(self) -> Path:
        """Return the directory where the next evaluation writes artefacts."""
        self._eval_index += 1
        iter_dir = self.workdir / f"eval{self._eval_index}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        return iter_dir

    def _evaluate_residual(self, phi_vec: Array) -> Array:
        """Evaluate residuals for the given φ vector, reusing cached results when possible."""
        phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
        if (
            self._last_phi is not None
            and self._last_residual is not None
            and np.array_equal(phi_vec, self._last_phi)
        ):
            last_residual = self._last_residual
            return cast(Array, last_residual.copy())
        theta_vec = self._theta_vec(phi_vec)
        theta_dict = self.parameter_space.unpack_vec(theta_vec.tolist())
        iter_dir = self._next_iter_dir()
        self._iter_dirs.append(iter_dir)
        label = "_base" if self.jacobian is not None else ""
        self._series_latest = {}
        base_jobs = self._launch_case_jobs(theta_dict, iter_dir, label)
        residual = self._finalize_case_jobs(
            base_jobs,
            track_series=True,
            grid_overrides=self._case_grid_cache,
            update_grid_cache=True,
        )
        self._last_phi = phi_vec.copy()
        self._last_theta_vec = theta_vec
        self._last_residual = residual.copy()
        self._last_iter_dir = iter_dir
        self._cached_jac_phi = None
        self._cached_jacobian = None
        self._cleanup_previous_iterations()
        return residual

    def _launch_case_jobs(
        self,
        theta: Mapping[str, float],
        iter_dir: Path,
        label: str | None,
    ) -> List[tuple[_CaseState, Path, RunHandle, str]]:
        """Prepare FEB files and launch simulations without waiting for completion."""
        theta_values = {k: float(v) for k, v in theta.items()}
        base_dir = iter_dir
        base_dir.mkdir(parents=True, exist_ok=True)
        jobs: List[tuple[_CaseState, Path, RunHandle, str]] = []
        for state in self._cases:
            case_name = state.name
            if label:
                base_name = f"{case_name}{label}"
            else:
                base_name = case_name
            feb_path = state.case.prepare(
                dict(theta_values),
                base_dir,
                out_name=f"{base_name}.feb",
            )
            case_env = state.case.environment()
            env_override = case_env if case_env else None
            handle = self.runner.run(
                feb_path.parent,
                feb_path.name,
                env=env_override,
            )
            # sim_label = label or "_base"
            # self._logger.info(
            #     f"Launching simulation '{feb_path.name}' (label={sim_label}) in {feb_path.parent}"
            # )
            jobs.append((state, feb_path, handle, case_name))
        return jobs

    def _finalize_case_jobs(
        self,
        jobs: List[tuple[_CaseState, Path, RunHandle, str]],
        *,
        track_series: bool,
        grid_overrides: Dict[str, Dict[str, Array]] | None = None,
        update_grid_cache: bool = False,
    ) -> Array:
        """Wait for running simulations and assemble residuals."""
        residuals: List[Array] = []
        r_squared: Dict[str, float] = {}
        all_exp: List[Array] = []
        all_sim: List[Array] = []
        series_latest: Dict[str, Dict[str, Any]] = {}

        for state, feb_path, handle, case_name in jobs:
            try:
                result = handle.wait()
            except KeyboardInterrupt:
                self.runner.shutdown()
                raise
            if result.exit_code != 0:
                raise RuntimeError(
                    f"FEBio exited with code {result.exit_code} for case '{case_name}'"
                )
            simulations = state.case.collect(feb_path)
            overrides = None
            if grid_overrides is not None:
                overrides = grid_overrides.get(case_name)
            residual, _, details = self.residual_assembler.assemble_with_details(
                state.experiments,
                simulations,
                target_grids=overrides,
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
                key = f"{case_name}/{exp_name}"
                r_squared[key] = r2
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
                    if update_grid_cache and grid_vals is not None:
                        cache = self._case_grid_cache.setdefault(case_name, {})
                        cache[exp_name] = cast(
                            Array, np.asarray(grid_vals, dtype=float).reshape(-1)
                        )
                except Exception:
                    pass

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
        if track_series:
            self._series_latest = series_latest
            self._last_metrics = metrics

        if not residuals:
            return cast(Array, np.array([], dtype=float))
        return cast(Array, np.concatenate(residuals))

    def _execute_cases(
        self,
        theta: Mapping[str, float],
        iter_dir: Path,
        label: str | None,
    ) -> Array:
        """Render FEB files, launch simulations, and assemble residuals."""
        jobs = self._launch_case_jobs(theta, iter_dir, label)
        return self._finalize_case_jobs(
            jobs,
            track_series=True,
            grid_overrides=self._case_grid_cache,
            update_grid_cache=True,
        )

    def _cleanup_previous_iterations(self) -> None:
        """Remove artefacts from older iterations, keeping current and best runs."""
        if not self.cleanup_previous:
            return
        keep: Set[Path] = set()
        if self._last_iter_dir is not None:
            keep.add(self._last_iter_dir.resolve())
        retained: List[Path] = []
        for dir_path in self._iter_dirs:
            resolved = dir_path.resolve()
            if resolved in keep:
                retained.append(dir_path)
            else:
                self.storage.cleanup_path(dir_path)
        self._iter_dirs = retained

    def _final_cleanup(self, interrupted: bool = False) -> None:
        """Apply final cleanup according to policy and storage mode."""
        if self.storage_mode == "tmp":
            self._persist_best()
            shutil.rmtree(self.workdir, ignore_errors=True)
            self._iter_dirs = []
            return

        mode = self.cleanup_mode
        if mode == "none":
            return
        keep_paths: Set[Path] = set()
        if mode == "retain_best":
            candidate = self._last_iter_dir
            if candidate is None:
                return
            keep_paths.add(candidate.resolve())
        self.storage.cleanup_all(keep_paths if keep_paths else None)
        if keep_paths:
            keep_resolved = {p.resolve() for p in keep_paths}
            self._iter_dirs = [
                p for p in self._iter_dirs if p.resolve() in keep_resolved
            ]
        else:
            self._iter_dirs = []

    def _persist_best(self) -> None:
        """Copy the last iteration directory to the persistent root."""
        best_dir = self._last_iter_dir
        if best_dir is None:
            return
        best_dir = best_dir.resolve()
        if not best_dir.exists():
            return
        dest = self.persist_root / best_dir.name
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(best_dir, dest, dirs_exist_ok=True)

    def _initMsg(self):
        """Emit a small banner when the engine is constructed."""
        banner_lines = [
            " _       _            _____ _____ ____  _      ",
            "(_)_ __ | |_ ___ _ __|  ___| ____| __ )(_) ___  ",
            "| | '_ \| __/ _ \ '__| |_  |  _| |  _ \| |/ _ \ ",
            "| | | | | ||  __/ |  |  _| | |___| |_) | | (_) |",
            "|_|_| |_|\__\___|_|  |_|   |_____|____/|_|\___/ ",
            "                                                ",
        ]
        self._logger.info("\n" + "\n".join(banner_lines))

    def _log_final_summary(
        self,
        phi_vec: Array,
        theta: Mapping[str, float],
    ) -> None:
        """Log a final summary table when optimisation completes."""
        table = PrettyTable()
        table.field_names = ["parameter", "phi", "theta"]
        phi_values = np.asarray(phi_vec, dtype=float).reshape(-1)
        theta_values = {k: float(v) for k, v in theta.items()}
        for name, phi_value in zip(self.parameter_space.names, phi_values, strict=True):
            theta_value = theta_values.get(name, float("nan"))
            if not np.isfinite(theta_value):
                theta_display = "nan"
            else:
                theta_display = f"{theta_value:+.6e}"
            phi_display = f"{float(phi_value):+.6e}"
            table.add_row([name, phi_display, theta_display])

        metrics = self._last_metrics or {}
        nrmse_val = metrics.get("nrmse")
        if nrmse_val is None or not np.isfinite(nrmse_val):
            nrmse_display = "nan"
        else:
            nrmse_display = f"{float(nrmse_val):.6e}"

        self._logger.info(
            "\nOptimization complete nrmse={nrmse}\n{table}",
            nrmse=nrmse_display,
            table=table.get_string(),
        )


__all__ = ["Engine", "OptimizeResult"]
