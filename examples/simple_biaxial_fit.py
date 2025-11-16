"""Minimal optimisation example for the simple biaxial benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from interFEBio.Optimize.adapters import SimulationAdapter
from interFEBio.Optimize.cases import SimulationCase
from interFEBio.Optimize.engine import Engine
from interFEBio.Optimize.experiments import ExperimentSeries
from interFEBio.Optimize.feb_bindings import FebTemplate, ParameterBinding
from interFEBio.Optimize.options import (
    CleanupOptions,
    EngineOptions,
    JacobianOptions,
    MonitorOptions,
    OptimizerOptions,
    RunnerOptions,
    StorageOptions,
    GridPolicyOptions,
)
from interFEBio.Optimize.Parameters import ParameterSpace
from interFEBio.XPLT import xplt

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "../tests/optimize/data.txt"
TEMPLATE_PATH = HERE / "../tests/optimize/simpleBiaxial.feb"
WORK_ROOT = HERE / "work_biaxial"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

FEBIO_COMMAND = ("febio4", "-i")
PARALLEL_JOBS = 2

FIXED_GRID = np.linspace(0.0, 2.0, 51)


def read_sigma_xx(xplt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract sigma_xx from element 10 across all time steps."""

    xp = xplt(str(xplt_path))
    xp.readAllStates()
    stress = xp.results["stress"].domain("volume")
    sigma_xx = stress.time(":").items(9).comp("xx")
    times = np.asarray(xp._time, dtype=float)
    return times, np.asarray(sigma_xx, dtype=float)


def build_case(exp_series: ExperimentSeries, name: str) -> SimulationCase:
    template = FebTemplate(
        TEMPLATE_PATH,
        bindings=[
            ParameterBinding(
                theta_name="G",
                xpath=".//Material/material[@id='1']/G",
            )
        ],
    )
    return SimulationCase(
        template=template,
        subfolder="",
        experiments={name: exp_series},
        adapters={name: SimulationAdapter(read_sigma_xx)},
        omp_threads=1,
        grids={
            name: GridPolicyOptions(policy="fixed_user", values=FIXED_GRID.tolist())
        },
    )


def main() -> None:
    exp_data = np.loadtxt(DATA_PATH)
    exp_series = ExperimentSeries(x=exp_data[:, 0], y=exp_data[:, 1])

    parameter_space = ParameterSpace(xi=2.0)
    parameter_space.add_parameter(name="G", theta0=0.5, bounds=(0, 10))
    case = build_case(exp_series, "biax1")
    case2 = build_case(exp_series, "biax2")

    options = EngineOptions(
        jacobian=JacobianOptions(enabled=True, perturbation=1e-4, parallel=True),
        cleanup=CleanupOptions(remove_previous=True, mode="retain_best"),
        runner=RunnerOptions(jobs=PARALLEL_JOBS, command=FEBIO_COMMAND),
        storage=StorageOptions(mode="tmp", root=WORK_ROOT),
        monitor=MonitorOptions(enabled=True),
        optimizer=OptimizerOptions(
            name="least_squares",
            settings={
                "ftol": 1e-6,
                "xtol": 1e-6,
            },
            reparametrize=True,
        ),
    )

    engine = Engine(
        parameter_space=parameter_space,
        cases=[case, case2],
        options=options,
    )

    result = engine.run()


if __name__ == "__main__":
    main()
