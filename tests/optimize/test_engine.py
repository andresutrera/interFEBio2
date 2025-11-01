import numpy as np

from interFEBio.Optimize.Parameters import ParameterSpace
from interFEBio.Optimize.engine import Engine
from interFEBio.Optimize.experiments import ExperimentSeries


def test_engine_runs_with_mocked_residual(tmp_path):
    param_space = ParameterSpace(names=["a"], theta0={"a": 1.0}, xi=2.0)

    class DummyCase:
        subfolder = "case"
        experiments = {"exp": ExperimentSeries(x=np.array([0.0]), y=np.array([0.0]))}

        def prepare(self, theta, out_root, ctx=None, out_name=None):  # pragma: no cover - not used
            raise NotImplementedError

        def collect(self, feb_path):  # pragma: no cover - not used
            raise NotImplementedError

    engine = Engine(
        parameter_space=param_space,
        cases=[DummyCase()],
        grid_policy="sim_to_exp",
        runner_jobs=1,
        storage_mode="disk",
        storage_root=tmp_path,
    )

    engine._execute_cases = lambda theta, iter_dir, label: np.array([theta["a"] - 2.0])  # type: ignore[attr-defined]

    calls = []

    result = engine.run(
        phi0=[0.0],
        verbose=False,
        callbacks=[lambda phi, cost: calls.append((phi.copy(), cost))],
    )

    assert calls, "callback not invoked"
    np.testing.assert_allclose(result.phi, np.array([1.0]), atol=1e-6)
