from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Literal,
)

import numpy as np

from .Parameters import ParameterSpace
from .alignment import Aligner, EvaluationGrid, ResidualAssembler
from .cases import ExperimentSeries, SimulationAdapter, SimulationCase
from .engine import CasePlan, OptimizationConfig, OptimizationEngine
from .feb_bindings import FebBinding, FebTemplate
from .jacobian import JacobianComputer
from .optimizers import OptimizerAdapter, ScipyLeastSquaresAdapter, ScipyMinimizeAdapter
from .runners import LocalParallelRunner, LocalSerialRunner, Runner

Array = np.ndarray


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class StorageConfig:
    """Filesystem placement for intermediate FEBio runs."""

    root: str | Path | None = None
    kind: Literal["disk", "tmpfs"] = "disk"
    label: str = "optimize_runs"
    create: bool = True

    def resolve(self) -> Path:
        if self.kind == "tmpfs":
            return self._resolve_tmpfs()
        return self._resolve_disk()

    def _resolve_disk(self) -> Path:
        path = Path(self.root) if self.root is not None else Path.cwd() / self.label
        if self.create:
            path.mkdir(parents=True, exist_ok=True)
        return path

    def _resolve_tmpfs(self) -> Path:
        if self.root is not None:
            path = Path(self.root)
            if self.create:
                path.mkdir(parents=True, exist_ok=True)
            return path

        candidates = [
            Path("/mnt/interfebio_ram"),
            Path("/dev/shm/interFEBio"),
            Path("/tmp/interFEBio_ram"),
        ]
        last_error: Optional[Exception] = None
        for base in candidates:
            path = base / self.label
            if not self.create:
                return path
            try:
                path.mkdir(parents=True, exist_ok=True)
                return path
            except OSError as exc:
                last_error = exc
        raise RuntimeError(
            "Could not create tmpfs storage directory; specify StorageConfig.root."
        ) from last_error


@dataclass
class ParameterConfig:
    names: Sequence[str]
    theta0: Mapping[str, float]
    xi: float = 10.0
    vary: Optional[Mapping[str, bool]] = None
    theta_bounds: Optional[Mapping[str, Tuple[Optional[float], Optional[float]]]] = None

    def build(self) -> ParameterSpace:
        vary = dict(self.vary) if self.vary is not None else None
        theta_bounds = (
            {k: tuple(v) for k, v in self.theta_bounds.items()}
            if self.theta_bounds is not None
            else None
        )
        return ParameterSpace(
            names=list(self.names),
            theta0=dict(self.theta0),
            xi=float(self.xi),
            vary=vary,
            theta_bounds=theta_bounds,
        )


@dataclass
class CaseConfig:
    name: str
    template_path: str | Path
    bindings: Sequence[FebBinding] = field(default_factory=list)
    experiments: Mapping[str, ExperimentSeries] = field(default_factory=dict)
    readers: Mapping[str, Callable[[Path], Tuple[np.ndarray, np.ndarray]]] = field(
        default_factory=dict
    )
    subfolder: Optional[str] = None
    grid_policy: str = "sim_to_exp"
    common_grid: Optional[np.ndarray] = None
    align_kind: str = "linear"
    align_fill_value: float = 0.0
    weight_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None

    def build(self) -> CasePlan:
        if set(self.experiments.keys()) != set(self.readers.keys()):
            missing_readers = set(self.experiments.keys()) - set(self.readers.keys())
            missing_experiments = set(self.readers.keys()) - set(
                self.experiments.keys()
            )
            raise ValueError(
                f"Case '{self.name}' has mismatched experiments and readers. "
                f"Missing readers: {sorted(missing_readers)}. "
                f"Missing experiments: {sorted(missing_experiments)}."
            )

        template = FebTemplate(
            self.template_path,
            bindings=list(self.bindings),
        )
        adapters = {
            name: SimulationAdapter(reader) for name, reader in self.readers.items()
        }
        case = SimulationCase(
            template=template,
            subfolder=self.subfolder or self.name,
            experiments=self.experiments,
            adapters=adapters,
        )
        assembler = ResidualAssembler(
            grid=EvaluationGrid(
                policy=self.grid_policy,
                common_grid=self.common_grid,
            ),
            aligner=Aligner(kind=self.align_kind, fill_value=self.align_fill_value),
            weight_fn=self.weight_fn,
        )
        return CasePlan(name=self.name, case=case, assembler=assembler)


@dataclass
class RunnerConfig:
    command: Sequence[str] = field(default_factory=lambda: ("febio4", "-i"))
    parallel_jobs: int = 1
    env: Optional[Mapping[str, str]] = None

    def build(self) -> Runner:
        env = dict(self.env) if self.env is not None else None
        if self.parallel_jobs > 1:
            return LocalParallelRunner(
                n_jobs=int(self.parallel_jobs),
                command=list(self.command),
                env=env,
            )
        return LocalSerialRunner(
            command=list(self.command),
            env=env,
        )


@dataclass
class OptimizerConfig:
    kind: str = "least_squares"
    method: Optional[str] = None
    options: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> OptimizerAdapter:
        if self.kind == "least_squares":
            return ScipyLeastSquaresAdapter(**self.options)
        if self.kind == "minimize":
            method = self.method or "L-BFGS-B"
            return ScipyMinimizeAdapter(method=method, **self.options)
        raise ValueError(f"Unsupported optimizer kind: {self.kind}")


@dataclass
class JacobianConfig:
    enabled: bool = False
    perturbation: float = 1e-6
    max_workers: int = 4

    def build(self) -> Optional[JacobianComputer]:
        if not self.enabled:
            return None
        return JacobianComputer(
            perturbation=float(self.perturbation),
            max_workers=int(self.max_workers),
        )


@dataclass
class OptimizeConfig:
    parameters: ParameterConfig
    cases: Sequence[CaseConfig]
    storage: StorageConfig
    runner: RunnerConfig = field(default_factory=RunnerConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    jacobian: Optional[JacobianConfig] = None
    phi0: Optional[Sequence[float]] = None
    bounds: Optional[Sequence[Tuple[float, float]]] = None
    callbacks: Iterable[Callable[[np.ndarray, float], None]] = field(
        default_factory=list
    )
    log_progress: bool = True


@dataclass
class OptimizeResult:
    phi: np.ndarray
    theta: Dict[str, float]
    metadata: Dict[str, Any]


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def _default_progress_callback(
    param_space: ParameterSpace,
) -> Callable[[np.ndarray, float], None]:
    counter = {"iter": 0}

    def format_phi(phi_vec: np.ndarray) -> str:
        return "[" + ", ".join(f"{val:+.4e}" for val in phi_vec) + "]"

    def format_theta(theta_vals: Dict[str, float]) -> str:
        return ", ".join(f"{name}={val:+.6e}" for name, val in theta_vals.items())

    def cb(phi: np.ndarray, cost: float) -> None:
        phi = np.asarray(phi, dtype=float)
        theta_vec = param_space.theta_from_phi(phi)
        theta_dict = param_space.unpack_vec(theta_vec)
        print(
            f"[iter {counter['iter']:03d}] "
            f"cost={float(cost):.6e} "
            f"phi={format_phi(phi)} "
            f"theta={{ {format_theta(theta_dict)} }}",
            flush=True,
        )
        counter["iter"] += 1

    return cb


def run_optimization(cfg: OptimizeConfig) -> OptimizeResult:
    param_space = cfg.parameters.build()
    workdir = cfg.storage.resolve()
    runner = cfg.runner.build()
    optimizer = cfg.optimizer.build()
    jac = cfg.jacobian.build() if cfg.jacobian is not None else None

    case_plans = [case_cfg.build() for case_cfg in cfg.cases]

    engine = OptimizationEngine(
        param_space=param_space,
        cases=case_plans,
        runner=runner,
        optimizer=optimizer,
        workdir=workdir,
        jacobian=jac,
    )

    try:
        phi0 = (
            np.asarray(cfg.phi0, dtype=float)
            if cfg.phi0 is not None
            else np.zeros(len(param_space.names), dtype=float)
        )
        bounds = cfg.bounds or param_space.phi_bounds()
        callbacks: list[Callable[[np.ndarray, float], None]] = []
        if cfg.log_progress:
            callbacks.append(_default_progress_callback(param_space))
        callbacks.extend(cfg.callbacks)

        run_cfg = OptimizationConfig(
            phi0=phi0,
            bounds=bounds,
            callbacks=callbacks,
        )
        phi_opt, meta = engine.run(run_cfg)
    finally:
        engine.close()

    theta_opt = param_space.unpack_vec(param_space.theta_from_phi(phi_opt))
    return OptimizeResult(
        phi=np.asarray(phi_opt, dtype=float),
        theta=theta_opt,
        metadata=meta,
    )


__all__ = [
    "OptimizeConfig",
    "OptimizeResult",
    "ParameterConfig",
    "CaseConfig",
    "RunnerConfig",
    "OptimizerConfig",
    "JacobianConfig",
    "StorageConfig",
    "run_optimization",
]
