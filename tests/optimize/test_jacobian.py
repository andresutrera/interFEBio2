import numpy as np

from interFEBio.Optimize.jacobian import JacobianComputer


def test_jacobian_computer():
    def theta_fn(phi):
        return phi * 2.0

    def residual_fn(theta):
        return theta**2

    jac = JacobianComputer(perturbation=1e-6)
    phi0 = np.array([1.0, 2.0])
    r0, J = jac.compute(phi0, theta_fn, residual_fn)
    np.testing.assert_allclose(r0, np.array([4.0, 16.0]))
    # derivative dr/dphi = 8*phi (since theta=2phi, r=theta^2=4phi^2)
    expected_J = np.array([[8.0, 0.0], [0.0, 16.0]])
    np.testing.assert_allclose(J, expected_J, atol=1e-2)
