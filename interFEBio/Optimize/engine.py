"""Unified optimisation engine for FEBio parameter fitting."""

from __future__ import annotations

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
    cast,
)
import shutil

import numpy as np
from numpy.typing import NDArray
from prettytable import PrettyTable

from ..Log import Log
from .Parameters import ParameterSpace
from .Storage import StorageManager
from .alignment import EvaluationGrid, Aligner, GridPolicy
from .cases import SimulationCase
from .jacobian import JacobianComputer
from .optimizers import (
    OptimizerAdapter,
    ScipyLeastSquaresAdapter,
    ScipyMinimizeAdapter,
)
from .residuals import ResidualAssembler
from .runners import LocalParallelRunner, LocalSerialRunner, RunHandle, Runner


Array = NDArray[np.float64]


@dataclass
class OptimizeResult:
    phi: Array
    theta: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class _CaseState:
    case: SimulationCase
    experiments: Dict[str, tuple[Array, Array, Array | None]]


class Engine:
    """
    High-level optimisation driver combining FEB building, execution, and fitting.

    Parameters
    ----------
    parameter_space
        Mapping between optimisation parameters (φ) and physical parameters (θ).
    cases
        Collection of simulation cases to run for each residual evaluation.
    grid_policy
        Policy string understood by :class:`EvaluationGrid`.
    grid_values
        Optional grid values for ``fixed_user`` policies.
    use_jacobian
        Set to ``True`` to compute forward-difference Jacobians each iteration.
    jacobian_perturbation
        Step size applied to each optimisation parameter when estimating the Jacobian.
    jacobian_parallel
        Informational flag indicating whether Jacobian evaluations should be scheduled in
        parallel; the engine still respects the runner job pool ``runner_jobs``.
    cleanup_previous
        When ``True``, delete artifacts from older iterations as new ones are produced,
        retaining the best (if known) and the most recent iteration.
    cleanup_mode
        Final cleanup behaviour when optimisation ends (or is interrupted): ``"none"``
        keeps all outputs, ``"retain_best"`` preserves only the best iteration, and
        ``"all"`` removes everything under the storage directory.
    optimizer
        String selector for the optimiser; ``\"least_squares\"`` and ``\"minimize\"`` supported.
    optimizer_options
        Keyword arguments forwarded to the optimiser adapter.
    runner_jobs
        Number of parallel FEBio jobs to launch per residual evaluation.
    runner_command
        Command tuple used to invoke FEBio (default ``(\"febio4\", \"-i\")``).
    runner_env
        Optional environment overrides for launched processes.
    storage_mode
        Either ``\"disk\"`` or ``\"tmp\"``; controls placement of generated FEBio files.
    storage_root
        Optional root directory. When ``storage_mode='disk'`` this path is used directly;
        when ``'tmp'`` it overrides the default ``/tmp`` base directory.
    log_file
        Optional log file path. When omitted, defaults to ``storage_root/optimization.log`` if
        ``storage_root`` is provided, otherwise a timestamped file in the current working directory.
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        cases: Sequence[SimulationCase],
        grid_policy: GridPolicy = "sim_to_exp",
        grid_values: Sequence[float] | None = None,
        *,
        use_jacobian: bool = False,
        jacobian_perturbation: float = 1e-6,
        jacobian_parallel: bool = True,
        cleanup_previous: bool = False,
        cleanup_mode: str = "none",
        optimizer: str = "least_squares",
        optimizer_options: Dict[str, Any] | None = None,
        runner_jobs: int = 1,
        runner_command: Sequence[str] | None = None,
        runner_env: Dict[str, str] | None = None,
        storage_mode: str = "disk",
        storage_root: str | Path | None = None,
        log_file: str | Path | None = None,
    ) -> None:
        if not cases:
            raise ValueError("At least one SimulationCase is required.")

        self.parameter_space = parameter_space
        self.jacobian: JacobianComputer | None = (
            JacobianComputer(
                perturbation=float(jacobian_perturbation),
                parallel=bool(jacobian_parallel),
            )
            if use_jacobian
            else None
        )
        self.storage_mode = storage_mode
        if storage_mode == "tmp":
            cleanup_previous = True
            cleanup_mode = "all"
        cleanup_mode = cleanup_mode.lower()
        if cleanup_mode not in {"none", "retain_best", "all"}:
            raise ValueError(
                "cleanup_mode must be one of: 'none', 'retain_best', 'all'"
            )
        self.cleanup_previous = bool(cleanup_previous)
        self.cleanup_mode = cleanup_mode
        self.optimizer_adapter = self._build_optimizer(optimizer, optimizer_options)
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

        storage_mode = storage_mode.lower()
        if storage_mode not in {"disk", "tmp"}:
            raise ValueError("storage_mode must be 'disk' or 'tmp'")
        storage_parent = (
            Path(storage_root).expanduser() if storage_root is not None else None
        )
        self.storage = StorageManager(
            parent=storage_parent,
            use_tmp=(storage_mode == "tmp"),
        )
        self.workdir = self.storage.resolve()
        if storage_mode == "tmp":
            if storage_root is not None:
                persist_root = Path(storage_root).expanduser().resolve()
            else:
                persist_root = Path.cwd() / self.workdir.name
            persist_root.mkdir(parents=True, exist_ok=True)
            self.persist_root = persist_root
        else:
            self.persist_root = self.workdir
        self._log_file = self._resolve_log_file(log_file, storage_root)
        log_instance = Log(log_file=self._log_file)
        self._logger = log_instance.logger
        self._initMsg()

        runner_command = tuple(runner_command or ("febio4", "-i"))
        if runner_jobs <= 1:
            self.runner: Runner = LocalSerialRunner(
                command=runner_command, env=runner_env
            )
        else:
            self.runner = LocalParallelRunner(
                n_jobs=runner_jobs,
                command=runner_command,
                env=runner_env,
            )

        self._cases: List[_CaseState] = []
        for case in cases:
            experiments: Dict[str, tuple[Array, Array, Array | None]] = {}
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
            self._cases.append(_CaseState(case=case, experiments=experiments))

        self._progress_index = 0
        self._eval_index = 0
        self._iter_dirs: List[Path] = []
        self._best_iter_dir: Path | None = None
        self._best_cost: float | None = None
        self._last_phi: Array | None = None
        self._last_theta_vec: Array | None = None
        self._last_residual: Array | None = None
        self._last_iter_dir: Path | None = None
        self._last_metrics: Dict[str, Any] = {}

    # ------------------------------------------------------------------ public API
    def run(
        self,
        *,
        phi0: Sequence[float] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        verbose: bool = True,
        callbacks: Iterable[Callable[[Array, float], None]] | None = None,
    ) -> OptimizeResult:
        """Execute the optimisation and return the fitted parameters."""

        interrupted = False
        try:
            self._progress_index = 0
            self._eval_index = 0
            self._iter_dirs = []
            self._best_iter_dir = None
            self._best_cost = None
            self._last_phi = None
            self._last_theta_vec = None
            self._last_residual = None
            self._last_iter_dir = None
            self._last_metrics = {}

            phi0_vec = cast(
                Array,
                np.asarray(phi0, dtype=float)
                if phi0 is not None
                else np.zeros(len(self.parameter_space.names), dtype=float),
            )
            bounds_input = (
                bounds if bounds is not None else self.parameter_space.phi_bounds()
            )

            callback_list: List[Callable[[Array, float], None]] = list(callbacks or [])
            if verbose:
                callback_list.append(self._progress_printer())

            def residual_phi(phi_vec: Array) -> Array:
                phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
                return self._evaluate_residual(phi_vec)

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
            theta_opt_vec = self.parameter_space.theta_from_phi(phi_opt_array)
            theta_opt_vec = self.parameter_space.clamp_theta(theta_opt_vec)
            theta_opt = self.parameter_space.unpack_vec(theta_opt_vec.tolist())
            self._log_final_summary(phi_opt_array, theta_opt)
            return OptimizeResult(
                phi=phi_opt_array,
                theta=theta_opt,
                metadata=meta,
            )
        except KeyboardInterrupt:
            interrupted = True
            raise
        finally:
            try:
                self._final_cleanup(interrupted=interrupted)
            finally:
                self.close()

    def close(self) -> None:
        """Shut down any background workers."""
        try:
            self.runner.shutdown()
        except Exception:
            pass

    # ------------------------------------------------------------------ internals
    def _build_optimizer(
        self,
        name: str,
        options: Dict[str, Any] | None,
    ) -> OptimizerAdapter:
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
        logger = self._logger

        def callback(phi_vec: Array, cost: float) -> None:
            theta_vec = self.parameter_space.theta_from_phi(phi_vec)
            table = PrettyTable()
            table.field_names = ["parameter", "phi", "theta"]
            for name, phi_value, theta_value in zip(
                self.parameter_space.names, phi_vec, theta_vec, strict=True
            ):
                table.add_row(
                    [
                        name,
                        f"{float(phi_value):.6e}",
                        f"{float(theta_value):.6e}",
                    ]
                )
            metrics = self._last_metrics or {}
            nrmse_val = metrics.get("nrmse")
            r_squared = metrics.get("r_squared", {})
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
            param_table = table.get_string()
            rsq_table_str = rsq_table.get_string()
            logger.info(
                "\n[iter {iter:03d}] cost={cost:.6e} nrmse={nrmse}\n{param_table}\n{rsq_table}",
                iter=self._progress_index,
                cost=cost,
                nrmse=nrmse_display,
                param_table=param_table,
                rsq_table=rsq_table_str,
            )
            self._progress_index += 1
            if self._last_iter_dir is not None:
                if self._best_cost is None or cost <= self._best_cost:
                    self._best_cost = cost
                    self._best_iter_dir = self._last_iter_dir
            self._cleanup_previous_iterations()

        return callback

    def _build_jacobian_wrapper(self) -> Callable[[Array], Array]:
        names = list(self.parameter_space.names)

        def label_fn(idx: int) -> str | None:
            if idx < 0:
                return "_base"
            if idx < len(names):
                return f"_{names[idx]}"
            return f"_col_{idx}"

        def jacobian(phi_vec: Array) -> Array:
            phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
            if self._last_phi is None or not np.allclose(phi_vec, self._last_phi):
                self._evaluate_residual(phi_vec)

            iter_dir = self._last_iter_dir
            if iter_dir is None:
                iter_dir = self._next_iter_dir()
            base_residual = self._last_residual

            def residual_with_label(theta_vec: Array, lbl: str | None) -> Array:
                return self._residual_theta_vec(
                    theta_vec,
                    label=lbl,
                    iter_dir=iter_dir,
                )

            assert self.jacobian is not None
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
        phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
        theta_vec = self.parameter_space.theta_from_phi(phi_vec)
        theta_vec = self.parameter_space.clamp_theta(theta_vec)
        return cast(Array, np.asarray(theta_vec, dtype=float))

    def _residual_theta_vec(
        self,
        theta_vec: Array,
        label: str | None = None,
        iter_dir: Path | None = None,
    ) -> Array:
        theta_vec = cast(Array, np.asarray(theta_vec, dtype=float))
        theta_vec = self.parameter_space.clamp_theta(theta_vec)
        theta_dict = self.parameter_space.unpack_vec(theta_vec.tolist())
        target_dir = iter_dir or self._next_iter_dir()
        return self._execute_cases(theta_dict, target_dir, label)

    def _next_iter_dir(self) -> Path:
        self._eval_index += 1
        iter_dir = self.workdir / f"eval{self._eval_index}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        return iter_dir

    def _evaluate_residual(self, phi_vec: Array) -> Array:
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
        residual = self._execute_cases(theta_dict, iter_dir, label)
        self._last_phi = phi_vec.copy()
        self._last_theta_vec = theta_vec
        self._last_residual = residual.copy()
        self._last_iter_dir = iter_dir
        self._cleanup_previous_iterations()
        return residual

    def _execute_cases(
        self,
        theta: Mapping[str, float],
        iter_dir: Path,
        label: str | None,
    ) -> Array:
        theta_values = {k: float(v) for k, v in theta.items()}
        jobs: List[tuple[_CaseState, Path, RunHandle, str]] = []
        for state in self._cases:
            case_name = state.case.subfolder
            if not case_name:
                template_path = getattr(
                    getattr(state.case, "template", None), "template_path", None
                )
                case_name = Path(template_path).stem if template_path else "case"
            case_dir = iter_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            if label:
                base_name = f"{case_name}{label}"
            else:
                base_name = case_name
            feb_path = state.case.prepare(
                dict(theta_values),
                case_dir,
                out_name=f"{base_name}.feb",
            )
            case_env = state.case.environment()
            env_override = case_env if case_env else None
            handle = self.runner.run(
                feb_path.parent,
                feb_path.name,
                env=env_override,
            )
            jobs.append((state, feb_path, handle, case_name))

        residuals: List[Array] = []
        r_squared: Dict[str, float] = {}
        all_exp: List[Array] = []
        all_sim: List[Array] = []

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
            residual, _, details = self.residual_assembler.assemble_with_details(
                state.experiments, simulations
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

        self._last_metrics = {"nrmse": nrmse, "r_squared": r_squared}

        if not residuals:
            return cast(Array, np.array([], dtype=float))
        return cast(Array, np.concatenate(residuals))

    def _cleanup_previous_iterations(self) -> None:
        if not self.cleanup_previous:
            return
        keep: Set[Path] = set()
        if self._last_iter_dir is not None:
            keep.add(self._last_iter_dir.resolve())
        if self._best_iter_dir is not None:
            keep.add(self._best_iter_dir.resolve())
        retained: List[Path] = []
        for dir_path in self._iter_dirs:
            resolved = dir_path.resolve()
            if resolved in keep:
                retained.append(dir_path)
            else:
                self.storage.cleanup_path(dir_path)
        self._iter_dirs = retained

    def _final_cleanup(self, interrupted: bool = False) -> None:
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
            candidate = self._best_iter_dir or self._last_iter_dir
            if candidate is not None:
                keep_paths.add(candidate.resolve())
            else:
                return
        self.storage.cleanup_all(keep_paths if keep_paths else None)
        if keep_paths:
            keep_resolved = {p.resolve() for p in keep_paths}
            self._iter_dirs = [
                p for p in self._iter_dirs if p.resolve() in keep_resolved
            ]
        else:
            self._iter_dirs = []

    def _persist_best(self) -> None:
        best_dir = self._best_iter_dir or self._last_iter_dir
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
