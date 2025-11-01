from types import SimpleNamespace
from pathlib import Path

import numpy as np

from interFEBio.Optimize.Parameters import ParameterSpace
from interFEBio.Optimize.engine import CasePlan, OptimizationConfig, OptimizationEngine
from interFEBio.Optimize.optimizers import ScipyLeastSquaresAdapter


def test_optimization_engine(tmp_path):
    param_space = ParameterSpace(names=["a"], theta0={"a": 1.0}, xi=2.0)

    class DummyRunner:
        def run(self, job_dir, feb_name):
            class Handle:
                def wait(self, timeout=None):
                    class Result:
                        exit_code = 0
                    return Result()

            return Handle()

        def shutdown(self):
            pass

    dummy_series = SimpleNamespace(x=np.array([0.0]), y=np.array([0.0]), weight=None)
    dummy_case = SimpleNamespace(subfolder="case", experiments={"exp": dummy_series})
    assembler = SimpleNamespace()

    optimizer = ScipyLeastSquaresAdapter()
    engine = OptimizationEngine(
        param_space=param_space,
        cases=[CasePlan(name="case", case=dummy_case, assembler=assembler)],
        runner=DummyRunner(),
        optimizer=optimizer,
        workdir=tmp_path,
    )

    # Monkeypatch engine evaluation to avoid FEB runs.
    engine._evaluate = (
        lambda theta, label=None: (np.array([theta["a"] - 2.0]), {"exp": slice(0, 1)})
    )

    calls = []

    cfg = OptimizationConfig(
        phi0=np.array([0.0]),
        callbacks=[lambda phi, cost: calls.append((phi.copy(), cost))],
    )
    phi_opt, meta = engine.run(cfg)
    assert calls, "callback not invoked"
    np.testing.assert_allclose(phi_opt, np.array([1.0]), atol=1e-6)
    engine.close()
