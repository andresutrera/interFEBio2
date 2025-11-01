import numpy as np

from interFEBio.Optimize.alignment import Aligner, EvaluationGrid, ResidualAssembler


def test_aligner_linear():
    aligner = Aligner()
    x_src = np.array([0.0, 1.0, 2.0])
    y_src = np.array([0.0, 2.0, 4.0])
    x_tgt = np.array([0.5, 1.5])
    out = aligner.map(x_src, y_src, x_tgt)
    np.testing.assert_allclose(out, np.array([1.0, 3.0]))


def test_residual_assembler():
    grid = EvaluationGrid(policy="exp_to_sim")
    assembler = ResidualAssembler(grid)
    experiments = {
        "exp1": (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.0, 2.0]),
            None,
        )
    }
    simulations = {
        "exp1": (
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 1.5, 3.0]),
        )
    }
    residual, slices = assembler.assemble(experiments, simulations)
    np.testing.assert_allclose(residual, np.array([0.0, 0.5, 1.0]))
    assert "exp1" in slices
