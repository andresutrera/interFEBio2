import numpy as np

from interFEBio.Optimize.optimizers import (
    OptimizerAdapter,
    ScipyLeastSquaresAdapter,
    ScipyMinimizeAdapter,
)


def test_least_squares_adapter():
    def residual(x):
        return np.array([x[0] - 1.0])

    adapter = ScipyLeastSquaresAdapter()
    calls = []
    x_opt, meta = adapter.minimize(
        residual,
        None,
        np.array([0.0]),
        bounds=None,
        callbacks=[lambda x, c: calls.append((x.copy(), c))],
    )
    np.testing.assert_allclose(x_opt, np.array([1.0]), atol=1e-6)
    assert calls, "callback not invoked"


def test_minimize_adapter():
    def residual(x):
        return np.array([x[0] - 2.0])

    def jac(x):
        return np.array([[1.0]])

    adapter = ScipyMinimizeAdapter()
    calls = []
    x_opt, meta = adapter.minimize(
        residual,
        jac,
        np.array([0.0]),
        bounds=None,
        callbacks=[lambda x, c: calls.append((x.copy(), c))],
    )
    np.testing.assert_allclose(x_opt, np.array([2.0]), atol=1e-6)
    assert calls, "callback not invoked"
