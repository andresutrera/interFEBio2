"""
Minimal optimization example for the simple biaxial benchmark.

The new ``run_optimization`` orchestrator only needs a configuration object
describing parameters, cases, runner, and optimizer. No environment variables
are used; adjust the config literals below to suit your machine.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from interFEBio.Optimize.cases import ExperimentSeries
from interFEBio.Optimize.feb_bindings import MaterialParamBinding
from interFEBio.Optimize.orchestrator import (
    CaseConfig,
    JacobianConfig,
    OptimizeConfig,
    ParameterConfig,
    RunnerConfig,
    StorageConfig,
    run_optimization,
)
from interFEBio.XPLT import xplt


HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "../tests/optimize/data.txt"
TEMPLATE_PATH = HERE / "../tests/optimize/simpleBiaxial.feb"
WORK_ROOT = HERE / "work_biaxial"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

FEBIO_COMMAND = ["febio4", "-i"]
PARALLEL_JOBS = 4


def read_sigma_xx(xplt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Extract sigma_xx from element 10 across all time steps."""

    xp = xplt(str(xplt_path))
    xp.readAllStates()
    stress = xp.results["stress"].domain("volume")
    sigma_xx = stress.time(":").items(9).comp("xx")
    times = np.asarray(xp._time, dtype=float)
    return times, np.asarray(sigma_xx, dtype=float)


def main() -> None:
    # Experimental curve (time, sigma_xx)
    exp_data = np.loadtxt(DATA_PATH)
    exp_series = ExperimentSeries(x=exp_data[:, 0], y=exp_data[:, 1])

    cfg = OptimizeConfig(
        parameters=ParameterConfig(
            names=["G"],
            theta0={"G": 0.5},
            xi=2.0,
        ),
        storage=StorageConfig(kind="tmpfs"),
        cases=[
            CaseConfig(
                name="biaxial",
                template_path=TEMPLATE_PATH,
                bindings=[
                    MaterialParamBinding(
                        theta_name="G",
                        tag_name="G",
                        selector=("id", "1"),
                    )
                ],
                experiments={"biaxial": exp_series},
                readers={"biaxial": read_sigma_xx},
                grid_policy="sim_to_exp",
            )
        ],
        runner=RunnerConfig(
            command=FEBIO_COMMAND,
            parallel_jobs=PARALLEL_JOBS,
        ),
    )

    result = run_optimization(cfg)
    print("Optimal phi:", result.phi)
    print("Optimal theta:", result.theta)


if __name__ == "__main__":
    main()
