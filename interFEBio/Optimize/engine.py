"""Unified optimisation engine for FEBio parameter fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Sequence, cast

import numpy as np
from numpy.typing import NDArray

from ..Log import Log
from .cases import CaseEvaluator, CaseJob, SimulationCase
from .engine_support import RunReporter
from .jacobian import JacobianComputer
from .optimizers import BoundsLike, OptimizerAdapter
from .options import EngineOptions
from .Parameters import ParameterMapper, ParameterSpace
from .runners import LocalParallelRunner, LocalSerialRunner, Runner
from .Storage import StorageWorkspace

Array = NDArray[np.float64]


@dataclass
class OptimizeResult:
    phi: Array
    theta: dict[str, float]
    metadata: dict[str, Any]


class Engine:
    """Coordinate FEBio simulations and optimisation loops."""

    def __init__(
        self,
        parameter_space: ParameterSpace,
        cases: Sequence[SimulationCase],
        *,
        options: EngineOptions | None = None,
    ) -> None:
        if not cases:
            raise ValueError("At least one SimulationCase is required.")

        self.options = options = options or EngineOptions()
        self.parameter_space = parameter_space
        self._param_names = list(self.parameter_space.names)
        optimizer_opts = options.optimizer
        jacobian_opts = options.jacobian
        runner_opts = options.runner
        storage_opts = options.storage
        cleanup_opts = options.cleanup
        monitor_opts = options.monitor
        self._reparam_enabled = bool(optimizer_opts.reparametrize)

        self.workspace = StorageWorkspace(storage_opts, cleanup_opts)
        self.workdir = self.workspace.workdir
        self.persist_root = self.workspace.persist_root
        log_instance = Log(log_file=self.workspace.log_file)
        self._logger = log_instance.logger

        self.jacobian: JacobianComputer | None = (
            JacobianComputer(
                perturbation=float(jacobian_opts.perturbation),
                parallel=bool(jacobian_opts.parallel),
            )
            if jacobian_opts.enabled
            else None
        )

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

        self.case_evaluator = CaseEvaluator(cases, self.runner, self._logger)
        self.parameter_mapper = ParameterMapper(parameter_space, self._reparam_enabled)
        self.reporter = RunReporter(
            logger=self._logger,
            parameter_space=self.parameter_space,
            case_descriptions=self.case_evaluator.describe_cases(),
            monitor_opts=monitor_opts,
            workspace=self.workspace,
            reparam_enabled=self._reparam_enabled,
        )
        self.reporter.log_banner()
        self.optimizer_adapter = OptimizerAdapter.build(
            optimizer_opts.name, optimizer_opts.settings
        )

        self.reporter.log_configuration(
            options=options,
            runner_command=runner_command,
            runner_env=runner_env,
            optimizer_adapter=type(self.optimizer_adapter).__name__,
        )

        self._progress_index = 0
        self._pending_initial_log = False
        self._log_progress = True
        self._last_phi: Array | None = None
        self._last_theta_vec: Array | None = None
        self._last_residual: Array | None = None
        self._last_iter_dir: Path | None = None
        self._last_metrics: dict[str, Any] = {}
        self._series_latest: dict[str, dict[str, Any]] = {}
        self._cached_jac_phi: Array | None = None
        self._cached_jacobian: Array | None = None

    def run(
        self,
        *,
        phi0: Sequence[float] | None = None,
        bounds: Sequence[tuple[float, float]] | None = None,
        verbose: bool = True,
        callbacks: Iterable[Callable[[Array, float], None]] | None = None,
    ) -> OptimizeResult:
        monitor_client = self.reporter.ensure_monitor()
        try:
            self._progress_index = 0
            self._pending_initial_log = True
            self._log_progress = bool(verbose)
            self._last_phi = None
            self._last_theta_vec = None
            self._last_residual = None
            self._last_iter_dir = None
            self._last_metrics = {}
            self._series_latest = {}
            self._cached_jac_phi = None
            self._cached_jacobian = None

            phi0_vec = self.parameter_mapper.initial_phi(phi0)
            bounds_input: BoundsLike = self.parameter_mapper.bounds(bounds)
            if monitor_client is not None:
                theta0_vec = self.parameter_mapper.phi_to_theta(phi0_vec)
                self.reporter.notify_run_started(
                    phi0_vec,
                    theta0_vec,
                    bounds_input,
                    type(self.optimizer_adapter).__name__,
                    getattr(self, "_runner_jobs", None),
                )

            def residual_phi(phi_vec: Array) -> Array:
                residual = self._evaluate_residual(phi_vec)
                if self._pending_initial_log:
                    initial_cost = 0.5 * float(np.dot(residual, residual))
                    self._record_iteration(
                        phi_vec, initial_cost, log_output=self._log_progress
                    )
                    self._pending_initial_log = False
                return residual

            callback_list: List[Callable[[Array, float], None]] = list(callbacks or [])
            if verbose:
                callback_list.append(
                    lambda phi_vec, cost: self._record_iteration(
                        phi_vec, cost, log_output=True
                    )
                )

            jacobian_fn = None
            if self.jacobian is not None:
                jacobian_fn = self._build_jacobian_wrapper()

            phi_opt, meta = self.optimizer_adapter.minimize(
                residual_phi,
                jacobian_fn,
                phi0_vec,
                bounds_input,
                callback_list,
            )

            phi_opt_array = cast(Array, np.asarray(phi_opt, dtype=float))
            theta_opt_vec = self.parameter_mapper.phi_to_theta(phi_opt_array)
            theta_opt = self.parameter_space.unpack_vec(theta_opt_vec.tolist())
            self.reporter.log_final_summary(
                phi_opt_array, theta_opt, self._last_metrics or {}
            )
            self.reporter.notify_completed(theta_opt, meta, self._last_metrics or {})
            try:
                self.workspace.write_series(self._series_latest)
            except Exception:
                self._logger.exception("Failed to write series outputs.")
            return OptimizeResult(phi=phi_opt_array, theta=theta_opt, metadata=meta)
        except KeyboardInterrupt:
            self.reporter.notify_failed("interrupted")
            raise
        except Exception as exc:
            self.reporter.notify_failed(f"{exc.__class__.__name__}: {exc}")
            raise
        finally:
            try:
                self.workspace.final_cleanup(self._last_iter_dir)
            finally:
                self.reporter.close()
                self.close()

    def _record_iteration(
        self, phi_vec: Array, cost: float, *, log_output: bool
    ) -> None:
        theta_vec = self._last_theta_vec
        if theta_vec is None:
            theta_vec = self.parameter_mapper.phi_to_theta(phi_vec)
        metrics = self._last_metrics or {}
        series = self._series_latest
        self.reporter.record_iteration(
            index=self._progress_index,
            phi_vec=phi_vec,
            theta_vec=theta_vec,
            cost=cost,
            metrics=metrics,
            series=series,
            log_output=log_output,
        )
        self._progress_index += 1
        self.workspace.prune_old_iterations(self._last_iter_dir)

    def _evaluate_residual(self, phi_vec: Array) -> Array:
        phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
        if (
            self._last_phi is not None
            and self._last_residual is not None
            and np.array_equal(phi_vec, self._last_phi)
        ):
            return cast(Array, self._last_residual.copy())
        theta_vec = self.parameter_mapper.phi_to_theta(phi_vec)
        theta_dict = self.parameter_space.unpack_vec(theta_vec.tolist())
        iter_dir = self.workspace.next_iter_dir()
        result = self.case_evaluator.evaluate(
            theta_dict, iter_dir, label="_base", track_series=True
        )
        self._last_phi = phi_vec.copy()
        self._last_theta_vec = theta_vec
        self._last_residual = result.residual.copy()
        self._last_iter_dir = iter_dir
        self._last_metrics = result.metrics
        self._series_latest = result.series
        self._cached_jac_phi = None
        self._cached_jacobian = None
        return result.residual

    def _build_jacobian_wrapper(self) -> Callable[[Array], Array]:
        assert self.jacobian is not None
        label_fn = self._jacobian_label_fn()

        def jacobian(phi_vec: Array) -> Array:
            phi_vec = cast(Array, np.asarray(phi_vec, dtype=float))
            if self._last_phi is None or not np.array_equal(phi_vec, self._last_phi):
                self._evaluate_residual(phi_vec)
            iter_dir = self._last_iter_dir
            if iter_dir is None:
                iter_dir = self.workspace.next_iter_dir()
            base_residual = self._last_residual
            if base_residual is None:
                raise RuntimeError(
                    "Residuals must be evaluated before computing the Jacobian."
                )
            if self.jacobian.parallel:
                if (
                    self._cached_jac_phi is not None
                    and self._cached_jacobian is not None
                    and np.array_equal(phi_vec, self._cached_jac_phi)
                ):
                    return cast(Array, self._cached_jacobian.copy())
                jobs = self._schedule_jacobian_jobs(phi_vec, iter_dir, label_fn)
                J = self._finalize_jacobian_jobs(jobs, base_residual)
                self._cached_jac_phi = phi_vec.copy()
                self._cached_jacobian = J.copy()
                return J

            def residual_with_label(theta_vec: Array, lbl: str | None) -> Array:
                return self._residual_for_theta(
                    theta_vec, iter_dir, lbl, track_series=False
                )

            _, J = self.jacobian.compute(
                phi_vec,
                self.parameter_mapper.phi_to_theta,
                residual_with_label,
                label_fn=label_fn,
                base_residual=base_residual,
            )
            return J

        return jacobian

    def _schedule_jacobian_jobs(
        self,
        phi_vec: Array,
        iter_dir: Path,
        label_fn: Callable[[int], str | None],
    ) -> dict[int, List["CaseJob"]]:
        assert self.jacobian is not None
        jobs_by_index: dict[int, List["CaseJob"]] = {}
        for idx in range(len(self._param_names)):
            phi = phi_vec.copy()
            phi[idx] += float(self.jacobian.perturbation)
            theta_vec = self.parameter_mapper.phi_to_theta(phi)
            theta_dict = self.parameter_space.unpack_vec(theta_vec.tolist())
            label = label_fn(idx)
            jobs_by_index[idx] = self.case_evaluator.launch_jobs(
                theta_dict, iter_dir, label
            )
        return jobs_by_index

    def _finalize_jacobian_jobs(
        self,
        jobs_by_index: dict[int, List["CaseJob"]],
        base_residual: Array,
    ) -> Array:
        assert self.jacobian is not None
        J = cast(
            Array, np.zeros((base_residual.size, len(self._param_names)), dtype=float)
        )
        for idx, jobs in jobs_by_index.items():
            result = self.case_evaluator.finalize_jobs(jobs, track_series=False)
            residual = result.residual
            if residual.shape != base_residual.shape:
                raise RuntimeError("Residual size mismatch while forming the Jacobian.")
            J[:, idx] = (residual - base_residual) / float(self.jacobian.perturbation)
        return J

    def _residual_for_theta(
        self,
        theta_vec: Array,
        iter_dir: Path,
        label: str | None,
        *,
        track_series: bool,
    ) -> Array:
        theta_vec = cast(Array, np.asarray(theta_vec, dtype=float))
        theta_vec = self.parameter_space.clamp_theta(theta_vec)
        theta_dict = self.parameter_space.unpack_vec(theta_vec.tolist())
        result = self.case_evaluator.evaluate(
            theta_dict, iter_dir, label=label, track_series=track_series
        )
        return result.residual

    def _jacobian_label_fn(self) -> Callable[[int], str | None]:
        names = self._param_names

        def label(idx: int) -> str | None:
            if idx < 0:
                return "_base"
            if idx < len(names):
                return f"_{names[idx]}"
            return f"_col_{idx}"

        return label

    def close(self) -> None:
        try:
            self.runner.shutdown()
        except Exception:
            pass
