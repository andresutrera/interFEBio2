import numpy as np

from interFEBio.Optimize.alignment import Aligner, EvaluationGrid
from interFEBio.Optimize.residuals import ResidualAssembler


def test_aligner_linear():
    aligner = Aligner()
    x_src = np.array([0.0, 1.0, 2.0])
    y_src = np.array([0.0, 2.0, 4.0])
    x_tgt = np.array([0.5, 1.5])
    out = aligner.map(x_src, y_src, x_tgt)
    np.testing.assert_allclose(out, np.array([1.0, 3.0]))


def test_aligner_handles_column_vectors():
    aligner = Aligner()
    x_src = np.array([[0.0], [1.0], [2.0]])
    y_src = np.array([[0.0], [2.0], [4.0]])
    x_tgt = np.array([[0.5], [1.5]])
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


def test_exp_to_sim_preserves_experimental_grid():
    grid = EvaluationGrid(policy="exp_to_sim")
    assembler = ResidualAssembler(grid)
    experiments = {
        "exp": (
            np.array([0.0, 0.5, 1.0]),
            np.array([0.0, 0.5, 1.0]),
            None,
        )
    }
    simulations = {
        "exp": (
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.0, 1.0, 2.0, 3.0]),
        )
    }
    residuals, _, details = assembler.assemble_with_details(experiments, simulations)
    np.testing.assert_allclose(details["exp"]["grid"], np.array([0.0, 0.5, 1.0]))
    assert residuals.shape[0] == 3


def test_exp_to_sim_extends_when_sim_grid_longer():
    grid = EvaluationGrid(policy="exp_to_sim")
    assembler = ResidualAssembler(grid)
    experiments = {
        "exp": (
            np.array([0.0, 0.5, 1.0, 1.5]),
            np.array([0.0, 0.5, 1.0, 1.5]),
            None,
        )
    }
    simulations = {
        "exp": (
            np.array([0.0, 0.75, 1.5]),
            np.array([0.0, 1.0, 2.0]),
        )
    }
    residuals, _, details = assembler.assemble_with_details(experiments, simulations)
    np.testing.assert_allclose(details["exp"]["grid"], np.array([0.0, 0.5, 1.0, 1.5]))
    assert residuals.shape[0] == 4


def test_residual_assembler_target_override():
    grid = EvaluationGrid(policy="exp_to_sim")
    assembler = ResidualAssembler(grid)
    experiments = {
        "exp": (
            np.array([0.0, 0.4, 0.8, 1.2]),
            np.array([0.0, 0.2, 0.4, 0.6]),
            None,
        )
    }
    simulations = {
        "exp": (
            np.array([0.0, 0.25, 0.5]),
            np.array([0.0, 0.5, 1.0]),
        )
    }
    target = np.linspace(0.0, 1.2, 5)
    residuals, _, details = assembler.assemble_with_details(
        experiments,
        simulations,
        target_grids={"exp": target},
    )
    np.testing.assert_allclose(details["exp"]["grid"], target)
    assert residuals.shape[0] == target.size


def test_residual_assembler_override_skips_clipping():
    grid = EvaluationGrid(policy="exp_to_sim")
    assembler = ResidualAssembler(grid)
    experiments = {
        "exp": (
            np.linspace(0.0, 2.0, 5),
            np.linspace(0.0, 1.0, 5),
            None,
        )
    }
    simulations = {
        "exp": (
            np.array([0.0, 0.2, 0.4]),
            np.array([0.0, 0.4, 0.8]),
        )
    }
    target = np.linspace(0.0, 2.0, 8)
    residuals, _, details = assembler.assemble_with_details(
        experiments,
        simulations,
        target_grids={"exp": target},
    )
    np.testing.assert_allclose(details["exp"]["grid"], target)
    assert residuals.shape[0] == target.size
