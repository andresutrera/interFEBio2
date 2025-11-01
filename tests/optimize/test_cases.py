import numpy as np
import pytest
from pathlib import Path

from interFEBio.Optimize.cases import (
    ExperimentSeries,
    PreprocessTask,
    PostprocessTask,
    SimulationAdapter,
    SimulationCase,
    TaskContext,
    TaskPipeline,
)
from interFEBio.Optimize.feb_bindings import FebTemplate, MaterialParamBinding


def _template_path() -> Path:
    return Path(__file__).with_name("simpleBiaxial.feb")


def test_simulation_case_prepare_and_collect(tmp_path: Path) -> None:
    template = FebTemplate(
        _template_path(),
        bindings=[
            MaterialParamBinding(theta_name="k", tag_name="k", selector=("id", "1")),
            MaterialParamBinding(theta_name="G", tag_name="G", selector=("id", "1")),
        ],
    )

    experiments = {
        "exp1": ExperimentSeries(x=np.array([0.0, 1.0]), y=np.array([0.0, 2.0]))
    }
    adapters = {
        "exp1": SimulationAdapter(lambda path: (np.array([0.0, 1.0]), np.array([1.0, 3.0])))
    }

    pipeline = TaskPipeline()
    pipeline.add_pre(PreprocessTask(lambda ctx: ctx))
    pipeline.add_post(PostprocessTask(lambda ctx: ctx))

    case = SimulationCase(
        template=template,
        subfolder="caseA",
        experiments=experiments,
        adapters=adapters,
        tasks=pipeline,
    )

    theta = {"k": 12.0, "G": 8.0}
    feb_path, xplt_path, task_ctx = case.prepare(theta, tmp_path)
    assert feb_path.exists()
    xplt_path.write_text("", encoding="utf-8")
    assert task_ctx["theta"] == theta

    results, post_ctx = case.collect(xplt_path, task_ctx)
    assert "exp1" in results
    x, y = results["exp1"]
    np.testing.assert_array_equal(x, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(y, np.array([1.0, 3.0]))
    assert "results" in post_ctx


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
