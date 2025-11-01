import numpy as np
import pytest
from pathlib import Path

from interFEBio.Optimize.adapters import SimulationAdapter
from interFEBio.Optimize.cases import SimulationCase
from interFEBio.Optimize.experiments import ExperimentSeries
from interFEBio.Optimize.feb_bindings import FebTemplate, ParameterBinding


def _template_path() -> Path:
    return Path(__file__).with_name("simpleBiaxial.feb")


def test_simulation_case_prepare_and_collect(tmp_path: Path) -> None:
    template = FebTemplate(
        _template_path(),
        bindings=[
            ParameterBinding(theta_name="k", xpath=".//Material/material[@id='1']/k"),
            ParameterBinding(theta_name="G", xpath=".//Material/material[@id='1']/G"),
        ],
    )

    experiments = {
        "exp1": ExperimentSeries(x=np.array([0.0, 1.0]), y=np.array([0.0, 2.0]))
    }
    adapters = {
        "exp1": SimulationAdapter(lambda path: (np.array([0.0, 1.0]), np.array([1.0, 3.0])))
    }

    case = SimulationCase(
        template=template,
        subfolder="caseA",
        experiments=experiments,
        adapters=adapters,
    )

    theta = {"k": 12.0, "G": 8.0}
    feb_path = case.prepare(theta, tmp_path)
    assert feb_path.exists()

    xplt_path = feb_path.with_suffix(".xplt")
    xplt_path.write_text("", encoding="utf-8")

    results = case.collect(feb_path)
    assert "exp1" in results
    x, y = results["exp1"]
    np.testing.assert_array_equal(x, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(y, np.array([1.0, 3.0]))


def test_simulation_case_missing_adapter_raises() -> None:
    template = FebTemplate(_template_path())
    experiments = {"a": ExperimentSeries(x=np.array([0.0]), y=np.array([0.0]))}
    with pytest.raises(ValueError):
        SimulationCase(
            template=template,
            subfolder="case",
            experiments=experiments,
            adapters={},
        )
