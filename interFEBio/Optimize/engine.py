"""Unified optimisation engine for FEBio parameter fitting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple
import shutil

import numpy as np

from .Parameters import ParameterSpace
from .Storage import StorageManager
from .alignment import EvaluationGrid, Aligner
from .cases import SimulationCase
from .jacobian import JacobianComputer
from .optimizers import (
    OptimizerAdapter,
    ScipyLeastSquaresAdapter,
    ScipyMinimizeAdapter,
)
from .residuals import ResidualAssembler
from .runners import LocalParallelRunner, LocalSerialRunner, RunHandle, Runner


Array = np.ndarray


@dataclass
class OptimizeResult:
    phi: Array
    theta: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class _CaseState:
    case: SimulationCase
    experiments: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]


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
    """

    def __init__(
        self,
        parameter_space: ParameterSpace,
        cases: Sequence[SimulationCase],
        grid_policy: str = "sim_to_exp",
        grid_values: Optional[Sequence[float]] = None,
        *,
        use_jacobian: bool = False,
        jacobian_perturbation: float = 1e-6,
        jacobian_parallel: bool = True,
        cleanup_previous: bool = False,
        cleanup_mode: str = "none",
        optimizer: str = "least_squares",
        optimizer_options: Optional[Dict[str, Any]] = None,
        runner_jobs: int = 1,
        runner_command: Optional[Sequence[str]] = None,
        runner_env: Optional[Dict[str, str]] = None,
        storage_mode: str = "disk",
        storage_root: Optional[str | Path] = None,
    ):
        if not cases:
            raise ValueError("At least one SimulationCase is required.")

        self.parameter_space = parameter_space
        self.jacobian = (
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
            raise ValueError("cleanup_mode must be one of: 'none', 'retain_best', 'all'")
        self.cleanup_previous = bool(cleanup_previous)
        self.cleanup_mode = cleanup_mode
        self.optimizer_adapter = self._build_optimizer(optimizer, optimizer_options)
        self.grid = EvaluationGrid(
            policy=grid_policy,
            common_grid=None if grid_values is None else np.asarray(grid_values, dtype=float),
        )
        self.aligner = Aligner()
        self.residual_assembler = ResidualAssembler(grid=self.grid, aligner=self.aligner)

        storage_mode = storage_mode.lower()
        if storage_mode not in {"disk", "tmp"}:
            raise ValueError("storage_mode must be 'disk' or 'tmp'")
        storage_parent = Path(storage_root).expanduser() if storage_root else None
        self.storage = StorageManager(
            parent=storage_parent,
            use_tmp=(storage_mode == "tmp"),
        )
        self.workdir = self.storage.resolve()
        if storage_mode == "tmp":
            if storage_root:
                persist_root = Path(storage_root).expanduser().resolve()
            else:
                persist_root = Path.cwd() / self.workdir.name
            persist_root.mkdir(parents=True, exist_ok=True)
            self.persist_root = persist_root
        else:
            self.persist_root = self.workdir

        runner_command = tuple(runner_command or ("febio4", "-i"))
        if runner_jobs <= 1:
            self.runner: Runner = LocalSerialRunner(command=runner_command, env=runner_env)
        else:
            self.runner = LocalParallelRunner(
                n_jobs=runner_jobs,
                command=runner_command,
                env=runner_env,
            )

        self._cases: List[_CaseState] = []
        for case in cases:
            experiments: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
            for name, series in case.experiments.items():
                x, y, weight = series.weighted()
                experiments[name] = (
                    np.asarray(x, dtype=float),
                    np.asarray(y, dtype=float),
                    None if weight is None else np.asarray(weight, dtype=float),
                )
            self._cases.append(_CaseState(case=case, experiments=experiments))

        self._progress_index = 0
        self._eval_index = 0
        self._iter_dirs: List[Path] = []
        self._best_iter_dir: Optional[Path] = None
        self._best_cost: Optional[float] = None
        self._last_phi: Optional[np.ndarray] = None
        self._last_theta_vec: Optional[np.ndarray] = None
        self._last_residual: Optional[np.ndarray] = None
        self._last_iter_dir: Optional[Path] = None

    # ------------------------------------------------------------------ public API
    def run(
        self,
        *,
        phi0: Optional[Sequence[float]] = None,
        bounds: Optional[Sequence[Tuple[float, float]]] = None,
        verbose: bool = True,
        callbacks: Optional[Iterable[Callable[[Array, float], None]]] = None,
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

            phi0_vec = (
                np.asarray(phi0, dtype=float)
                if phi0 is not None
                else np.zeros(len(self.parameter_space.names), dtype=float)
            )
            bounds = bounds or self.parameter_space.phi_bounds()

            callback_list: List[Callable[[Array, float], None]] = list(callbacks or [])
            if verbose:
                callback_list.append(self._progress_printer())

            def residual_phi(phi_vec: Array) -> Array:
                return self._evaluate_residual(np.asarray(phi_vec, dtype=float))

            jac_fn = None
            if self.jacobian is not None:
                jac_fn = self._build_jacobian_wrapper()

            phi_opt, meta = self.optimizer_adapter.minimize(
                residual_phi,
                jac_fn,
                phi0_vec,
                bounds,
                callback_list,
            )

            theta_opt_vec = self.parameter_space.theta_from_phi(phi_opt)
            theta_opt_vec = self.parameter_space.clamp_theta(theta_opt_vec)
            theta_opt = self.parameter_space.unpack_vec(theta_opt_vec)
            return OptimizeResult(
                phi=np.asarray(phi_opt, dtype=float),
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
        options: Optional[Dict[str, Any]],
    ) -> OptimizerAdapter:
        opts = dict(options or {})
        key = name.lower()
        if key == "least_squares":
            return ScipyLeastSquaresAdapter(**opts)
        if key == "minimize":
            method = opts.pop("method", "L-BFGS-B")
            return ScipyMinimizeAdapter(method=method, **opts)
        raise ValueError(f"Unsupported optimizer: {name}")

    def _progress_printer(self) -> Callable[[Array, float], None]:
        def callback(phi_vec: Array, cost: float) -> None:
            theta_vec = self.parameter_space.theta_from_phi(phi_vec)
            theta = self.parameter_space.unpack_vec(theta_vec)
            pretty_theta = ", ".join(f"{k}={v:+.6e}" for k, v in theta.items())
            print(
                f"[iter {self._progress_index:03d}] cost={cost:.6e} theta={{ {pretty_theta} }}",
                flush=True,
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

        def label_fn(idx: int) -> Optional[str]:
            if idx < 0:
                return "_base"
            if idx < len(names):
                return f"_{names[idx]}"
            return f"_col_{idx}"

        def jacobian(phi_vec: Array) -> Array:
            phi_vec = np.asarray(phi_vec, dtype=float)
            if self._last_phi is None or not np.allclose(phi_vec, self._last_phi):
                self._evaluate_residual(phi_vec)

            iter_dir = self._last_iter_dir
            if iter_dir is None:
                iter_dir = self._next_iter_dir()
            base_residual = self._last_residual

            def residual_with_label(theta_vec: Array, lbl: Optional[str]) -> Array:
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
        phi_vec = np.asarray(phi_vec, dtype=float)
        theta_vec = self.parameter_space.theta_from_phi(phi_vec)
        theta_vec = self.parameter_space.clamp_theta(theta_vec)
        return np.asarray(theta_vec, dtype=float)

    def _residual_theta_vec(
        self,
        theta_vec: Array,
        label: Optional[str] = None,
        iter_dir: Optional[Path] = None,
    ) -> Array:
        theta_vec = np.asarray(theta_vec, dtype=float)
        theta_vec = self.parameter_space.clamp_theta(theta_vec)
        theta_dict = self.parameter_space.unpack_vec(theta_vec)
        target_dir = iter_dir or self._next_iter_dir()
        return self._execute_cases(theta_dict, target_dir, label)

    def _next_iter_dir(self) -> Path:
        self._eval_index += 1
        iter_dir = self.workdir / f"eval{self._eval_index}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        return iter_dir

    def _evaluate_residual(self, phi_vec: Array) -> Array:
        phi_vec = np.asarray(phi_vec, dtype=float)
        if (
            self._last_phi is not None
            and self._last_residual is not None
            and np.array_equal(phi_vec, self._last_phi)
        ):
            return self._last_residual.copy()
        theta_vec = self._theta_vec(phi_vec)
        theta_dict = self.parameter_space.unpack_vec(theta_vec)
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
        label: Optional[str],
    ) -> Array:
        theta_values = {k: float(v) for k, v in theta.items()}
        jobs: List[Tuple[_CaseState, Path, RunHandle]] = []
        for state in self._cases:
            case_name = state.case.subfolder
            if not case_name:
                template_path = getattr(getattr(state.case, "template", None), "template_path", None)
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
            handle = self.runner.run(feb_path.parent, feb_path.name)
            jobs.append((state, feb_path, handle))

        residuals: List[np.ndarray] = []
        for state, feb_path, handle in jobs:
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
            residual, _ = self.residual_assembler.assemble(state.experiments, simulations)
            if residual.size:
                residuals.append(residual)

        if not residuals:
            return np.array([], dtype=float)
        return np.concatenate(residuals)

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
            self._iter_dirs = [p for p in self._iter_dirs if p.resolve() in keep_resolved]
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


__all__ = ["Engine", "OptimizeResult"]
