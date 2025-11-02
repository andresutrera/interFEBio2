"""Minimal optimisation example for the simple biaxial benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from interFEBio.Optimize.Parameters import ParameterSpace
from interFEBio.Optimize.adapters import SimulationAdapter
from interFEBio.Optimize.cases import SimulationCase
from interFEBio.Optimize.engine import Engine
from interFEBio.Optimize.experiments import ExperimentSeries
from interFEBio.Optimize.feb_bindings import FebTemplate, ParameterBinding
from interFEBio.XPLT import xplt


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "../tests/optimize/data.txt"
TEMPLATE_PATH = HERE / "../tests/optimize/simpleBiaxial.feb"
WORK_ROOT = HERE / "work_biaxial"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

FEBIO_COMMAND = ("febio4", "-i")
PARALLEL_JOBS = 2


def read_sigma_xx(xplt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract sigma_xx from element 10 across all time steps."""

    xp = xplt(str(xplt_path))
    xp.readAllStates()
    stress = xp.results["stress"].domain("volume")
    sigma_xx = stress.time(":").items(9).comp("xx")
    times = np.asarray(xp._time, dtype=float)
    return times, np.asarray(sigma_xx, dtype=float)


def build_case(exp_series: ExperimentSeries) -> SimulationCase:
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
        experiments={"biaxial": exp_series},
        adapters={"biaxial": SimulationAdapter(read_sigma_xx)},
    )


def main() -> None:
    exp_data = np.loadtxt(DATA_PATH)
    exp_series = ExperimentSeries(x=exp_data[:, 0], y=exp_data[:, 1])

    parameter_space = ParameterSpace(xi=2.0)
    parameter_space.add_parameter(name="G", theta0=0.02, bounds=(0.001, 0.08))
    case = build_case(exp_series)

    engine = Engine(
        parameter_space=parameter_space,
        cases=[case],
        grid_policy="sim_to_exp",
        use_jacobian=True,
        jacobian_perturbation=1e-6,
        optimizer="least_squares",
        optimizer_options={
            "ftol": 1e-3,
            "xtol": 1e-3,
        },
        cleanup_mode="retain_best",
        cleanup_previous=True,
        runner_jobs=PARALLEL_JOBS,
        runner_command=FEBIO_COMMAND,
        storage_mode="tmp",
        storage_root=WORK_ROOT,
        monitor=False,
    )

    result = engine.run()


if __name__ == "__main__":
    main()
