"""Ring and longitudinal fit example using the optimisation engine."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping

import numpy as np
from interFEBio.Optimize.adapters import SimulationAdapter
from interFEBio.Optimize.cases import SimulationCase
from interFEBio.Optimize.engine import Engine
from interFEBio.Optimize.experiments import ExperimentSeries
from interFEBio.Optimize.feb_bindings import (
    BuildContext,
    EvaluationBinding,
    FebTemplate,
    ParameterBinding,
)
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
DATA_RING = HERE / "anillo_aorta_abdominal006Data_force.txt"
DATA_LONG = HERE / "abdominal_uniaxialData_stress.txt"
RING_TEMPLATE = HERE / "ring.feb"
LONG_TEMPLATE = HERE / "long.feb"
WORK_ROOT = HERE / "work_ring"
WORK_ROOT.mkdir(parents=True, exist_ok=True)

FEBIO_COMMAND = ("febio4", "-i")
GRID_POINTS = 200
RING_FIXED_GRID = np.linspace(0.0, 2.5, 100)
LONG_FIXED_GRID = np.linspace(1.0, 1.4, 100)

MaterialMapping = dict[str, str]


def read_ring_results(xplt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return displacement/force curves extracted from the ring simulation."""
    xp = xplt(str(xplt_path))
    xp.readAllStates()
    times = np.asarray(xp.results.times(), dtype=float)
    surf_node = int(xp.mesh.surfaces["contactPin"].faces[1, 0])
    disp_view = xp.results["displacement"]
    disp = (
        np.asarray(disp_view[:, surf_node, "y"], dtype=float).reshape(len(times)) * 2.0
    )
    force = np.nan_to_num(
        xp.results["contact force"].region("contactPin").time(":").comp("z") * -4.0
    )

    return disp, force


def read_longitudinal_results(xplt_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return stretch/stress curves from the longitudinal test simulation."""
    xp = xplt(str(xplt_path))
    xp.readAllStates()
    times = np.asarray(xp.results.times(), dtype=float)
    disp_view = xp.results["displacement"]
    lamb = np.asarray(disp_view[:, 1, "x"], dtype=float).reshape(len(times)) + 1.0
    stress_view = xp.results["stress"]
    domain_name = stress_view.domains()[0]
    stress = np.asarray(
        stress_view.domain(domain_name)[:, 0, "xx"], dtype=float
    ).reshape(len(times))
    return lamb, stress


def _load_experiment(
    path: Path, grid_override: np.ndarray | None = None
) -> ExperimentSeries:
    """Load and resample experimental data stored in a text file."""
    data = np.loadtxt(path, dtype=float)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected data shape in {path}")
    order = np.argsort(data[:, 0])
    x = data[order, 0]
    y = np.abs(data[order, 1])
    if grid_override is not None:
        grid = np.asarray(grid_override, dtype=float)
    else:
        grid = np.linspace(x[0], x[-1], GRID_POINTS)
    y_interp = np.interp(grid, x, y)
    grid_attr = grid if grid_override is not None else None
    return ExperimentSeries(x=grid, y=y_interp, grid=grid_attr)


def build_parameter_space() -> ParameterSpace:
    """Return the parameter space configured for the vessel material."""
    base_values = {
        "c": 0.02834531,
        "k1": 0.0566152,
        "k2": 1.32378791,
        "kappa": 0.21713678,
        "gamma": 62.4763641,
    }
    bounds = {
        "c": (1e-5, 80e-3),
        "k1": (1e-5, 500e-3),
        "k2": (1e-5, 4.0),
        "kappa": (0.05, 0.28),
        "gamma": (20.0, 75.0),
    }
    space = ParameterSpace(xi=2.0)
    for name, theta0 in base_values.items():
        lo, hi = bounds[name]
        space.add_parameter(name=name, theta0=theta0, bounds=(lo, hi))
    # space.add_parameter(
    #     name="k", theta0=500.0 * base_values["k1"], vary=False, bounds=(0.01, 500)
    # )
    return space


def _material_bindings_long() -> list[ParameterBinding | EvaluationBinding]:
    base = ".//Material/material[@id='1']"
    # The 'k' parameter remains constant during optimisation and is thus
    # derived from the varying k1 value via an evaluation binding.
    return [
        ParameterBinding(theta_name="c", xpath=f"{base}/c"),
        ParameterBinding(theta_name="k1", xpath=f"{base}/k1"),
        ParameterBinding(theta_name="k2", xpath=f"{base}/k2"),
        ParameterBinding(theta_name="kappa", xpath=f"{base}/kappa"),
        ParameterBinding(theta_name="gamma", xpath=f"{base}/gamma"),
        EvaluationBinding(xpath=f"{base}/k", value="300 * k1"),
    ]


def _material_bindings_ring() -> list[ParameterBinding | EvaluationBinding]:
    base = ".//Material/material[@id='1']"
    elastic = f"{base}//elastic"
    return [
        ParameterBinding(theta_name="c", xpath=f"{elastic}/c"),
        ParameterBinding(theta_name="k1", xpath=f"{elastic}/k1"),
        ParameterBinding(theta_name="k2", xpath=f"{elastic}/k2"),
        ParameterBinding(theta_name="kappa", xpath=f"{elastic}/kappa"),
        ParameterBinding(theta_name="gamma", xpath=f"{elastic}/gamma"),
        EvaluationBinding(xpath=f"{base}/k", value="300 * k1"),
        # Scale the final load-curve point with the current k1 value
        # EvaluationBinding(
        #     xpath=".//LoadData/load_controller[@id='3']/points/pt[4]",
        #     value="500 * k1",
        #     text_template="2,{value}",
        # ),
    ]


def build_ring_case(series: ExperimentSeries) -> SimulationCase:
    template = FebTemplate(RING_TEMPLATE, bindings=_material_bindings_ring())
    return SimulationCase(
        template=template,
        subfolder="ring",
        experiments={"ring": series},
        adapters={"ring": SimulationAdapter(read_ring_results)},
        omp_threads=8,
    )


def build_long_case(series: ExperimentSeries) -> SimulationCase:
    template = FebTemplate(LONG_TEMPLATE, bindings=_material_bindings_long())
    return SimulationCase(
        template=template,
        subfolder="longitudinal",
        experiments={"long": series},
        adapters={"long": SimulationAdapter(read_longitudinal_results)},
        omp_threads=1,
    )


def main() -> None:
    ring_series = _load_experiment(DATA_RING, grid_override=RING_FIXED_GRID)
    long_series = _load_experiment(DATA_LONG, grid_override=LONG_FIXED_GRID)
    parameter_space = build_parameter_space()
    cases = [
        build_long_case(long_series),
        build_ring_case(ring_series),
    ]  # build_ring_case(ring_series),
    options = EngineOptions(
        grid=GridPolicyOptions(policy="exp_to_sim"),
        jacobian=JacobianOptions(enabled=True, perturbation=1e-4, parallel=True),
        cleanup=CleanupOptions(remove_previous=False, mode="none"),
        runner=RunnerOptions(jobs=6, command=FEBIO_COMMAND),
        storage=StorageOptions(mode="tmp", root=WORK_ROOT),
        monitor=MonitorOptions(enabled=True),
        optimizer=OptimizerOptions(
            name="least_squares",
            settings={"ftol": 1e-2, "xtol": 1e-2},
            reparametrize=True,
        ),
    )
    engine = Engine(parameter_space=parameter_space, cases=cases, options=options)
    result = engine.run()


if __name__ == "__main__":
    main()
