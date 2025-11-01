from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple, List
import threading

import numpy as np

from .Parameters import ParameterSpace
from .alignment import ResidualAssembler
from .cases import SimulationCase, TaskContext
from .jacobian import JacobianComputer
from .optimizers import OptimizerAdapter
from .feb_bindings import BuildContext
from .runners import Runner

Array = np.ndarray


@dataclass
class OptimizationConfig:
    phi0: Array
    bounds: Optional[Sequence[Tuple[float, float]]] = None
    callbacks: Iterable[Callable[[Array, float], None]] = field(default_factory=list)


@dataclass
class CasePlan:
    name: str
    case: SimulationCase
    assembler: ResidualAssembler


@dataclass
class _CaseState:
    plan: CasePlan
    experiments: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]]
    iter_counter: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass
class OptimizationEngine:
    param_space: ParameterSpace
    cases: Sequence[CasePlan]
    runner: Runner
    optimizer: OptimizerAdapter
    workdir: Path
    jacobian: Optional[JacobianComputer] = None

    def __post_init__(self) -> None:
        self.workdir = Path(self.workdir)
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._cases: List[_CaseState] = []
        for plan in self.cases:
            experiments: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]] = {}
            for name, series in plan.case.experiments.items():
                weight = None
                if series.weight is not None:
                    weight = np.asarray(series.weight(series.x), dtype=float)
                experiments[name] = (
                    np.asarray(series.x, dtype=float),
                    np.asarray(series.y, dtype=float),
                    weight,
                )
            self._cases.append(_CaseState(plan=plan, experiments=experiments))

    def close(self) -> None:
        try:
            self.runner.shutdown()
        except Exception:
            pass

    # --- internal helpers -------------------------------------------------
    def _theta_from_phi(self, phi: Array) -> Mapping[str, float]:
        theta_vec = self.param_space.theta_from_phi(phi)
        return self.param_space.unpack_vec(theta_vec)

    def _theta_vec(self, phi: Array) -> Array:
        theta = self.param_space.theta_from_phi(phi)
        return np.asarray(theta, dtype=float)

    def _residual_theta_vec(
        self, theta_vec: Array, label: Optional[str] = None
    ) -> Array:
        theta_dict = {
            name: float(val)
            for name, val in zip(self.param_space.names, theta_vec)
        }
        residual, _ = self._evaluate(theta_dict, label=label)
        return residual

    def _evaluate(
        self, theta: Mapping[str, float], label: Optional[str] = None
    ) -> Tuple[Array, Dict[str, slice]]:
        residual_blocks: List[np.ndarray] = []
        slices: Dict[str, slice] = {}
        offset = 0

        for state in self._cases:
            feb_path, xplt_path, task_ctx = self._prepare_case(
                state, theta, label=label
            )

            handle = self.runner.run(feb_path.parent, feb_path.name)
            result = handle.wait()
            if result.exit_code != 0:
                raise RuntimeError(f"FEBio exited with code {result.exit_code}")

            sim_results, _ = state.plan.case.collect(xplt_path, task_ctx)
            residual, case_slices = state.plan.assembler.assemble(
                state.experiments, sim_results
            )
            if residual.size == 0:
                continue
            residual_blocks.append(residual)
            for name, sl in case_slices.items():
                start_rel = 0 if sl.start is None else sl.start
                stop_rel = start_rel if sl.stop is None else sl.stop
                start = offset + start_rel
                stop = offset + stop_rel
                key = f"{state.plan.name}:{name}"
                slices[key] = slice(start, stop)
            offset += residual.size

        if not residual_blocks:
            return np.array([], dtype=float), {}
        return np.concatenate(residual_blocks), slices

    def _prepare_case(
        self,
        state: _CaseState,
        theta: Mapping[str, float],
        label: Optional[str] = None,
    ) -> Tuple[Path, Path, TaskContext]:
        with state.lock:
            iter_id = state.iter_counter
            state.iter_counter += 1

        suffix = f"{iter_id:05d}"
        folder = f"{label}_{suffix}" if label else f"iter_{suffix}"
        job_root = self.workdir / state.plan.case.subfolder / folder
        ctx = BuildContext(iter_id=iter_id, case_name=state.plan.case.subfolder)
        feb_path, xplt_path, task_ctx = state.plan.case.prepare(theta, job_root, ctx)
        return feb_path, xplt_path, task_ctx

    # --- public API -------------------------------------------------------
    def run(self, cfg: OptimizationConfig) -> Tuple[Array, Dict[str, object]]:
        phi0 = np.asarray(cfg.phi0, dtype=float)
        bounds = cfg.bounds or self.param_space.phi_bounds()

        def residual_phi(phi: Array) -> Array:
            theta = self._theta_from_phi(phi)
            residual, _ = self._evaluate(theta)
            return residual

        jac_phi = None
        if self.jacobian is not None:
            names = list(self.param_space.names)

            def _label(idx: int) -> Optional[str]:
                if idx < 0:
                    return "jac_base"
                if 0 <= idx < len(names):
                    return f"jac_{names[idx]}"
                return f"jac_col_{idx}"

            def jac_phi(phi: Array) -> Array:
                _, J = self.jacobian.compute(
                    phi,
                    self._theta_vec,
                    self._residual_theta_vec,
                    label_fn=_label,
                )
                return J

        phi_opt, meta = self.optimizer.minimize(
            residual_phi,
            jac_phi,
            phi0,
            bounds,
            cfg.callbacks,
        )

        return phi_opt, meta
