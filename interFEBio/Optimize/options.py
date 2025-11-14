"""Configuration dataclasses for the optimisation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .alignment import GridPolicy

CleanupMode = Literal["none", "retain_best", "all"]
StorageMode = Literal["disk", "tmp"]
OptimizerName = Literal["least_squares", "minimize"]


@dataclass(slots=True)
class GridPolicyOptions:
    """Configuration for how simulations align with experiments.

    Attributes:
        policy (GridPolicy): Strategy for choosing the evaluation grid. Using
            ``"exp_to_sim"`` maps simulations onto experimental abscissa
            samples, ``"sim_to_exp"`` does the inverse, and ``"fixed_user"``
            forces both to use ``values``.
        values (Sequence[float] | None): Explicit grid used when ``policy`` is
            ``"fixed_user"``. Supplying a denser grid can improve alignment
            fidelity at the cost of more interpolation.
    """

    policy: GridPolicy = "sim_to_exp"
    values: Sequence[float] | None = None


@dataclass(slots=True)
class JacobianOptions:
    """Jacobian evaluation controls.

    Attributes:
        enabled (bool): Enables finite-difference Jacobian evaluation. This
            improves convergence for gradient-based optimisers but increases
            the number of FEBio runs per iteration.
        perturbation (float): Perturbation magnitude applied per-parameter when
            computing derivatives. Larger values dampen noise, whereas smaller
            values capture local curvature more accurately.
        parallel (bool): Run Jacobian perturbations concurrently using the
            configured runner jobs. Keeping this ``True`` reduces the added
            turnaround time when multiple workers are available.
    """

    enabled: bool = False
    perturbation: float = 1e-6
    parallel: bool = True


@dataclass(slots=True)
class CleanupOptions:
    """Cleanup behaviour for working directories.

    Attributes:
        remove_previous (bool): If ``True``, delete artefacts from older
            iterations while the run is ongoing, keeping storage usage low at
            the cost of losing intermediate data.
        mode (CleanupMode): Policy applied when the engine finishes. ``"none"``
            preserves everything, ``"retain_best"`` keeps only the best
            iteration, and ``"all"`` wipes the storage root entirely.
    """

    remove_previous: bool = False
    mode: CleanupMode = "none"


@dataclass(slots=True)
class RunnerOptions:
    """Execution backend configuration.

    Attributes:
        jobs (int): Number of FEBio processes spawned in parallel. Increasing
            this speeds up throughput but consumes more CPU and memory.
        command (Sequence[str] | None): Exact executable and base arguments
            used to launch FEBio. Adjust this to select a different binary or
            add wrapper flags.
        env (Mapping[str, str] | None): Additional environment variables passed
            to each run, enabling fine-tuned FEBio behaviour per project.
    """

    jobs: int = 1
    command: Sequence[str] | None = None
    env: Mapping[str, str] | None = None


@dataclass(slots=True)
class StorageOptions:
    """Storage and logging configuration.

    Attributes:
        mode (StorageMode): Selects the storage backend. ``"disk"`` keeps artefacts
            under ``root`` permanently, while ``"tmp"`` leverages ``"/tmp"`` for faster
            IO and automatic cleanup.
        root (str | Path | None): Parent directory for generated artefacts.
            Supplying a path ensures deterministic locations and, when paired
            with ``"tmp"``, defines where the best iteration is persisted.
        log_file (str | Path | None): Path to the optimisation log. Specifying
            a location isolates engine logs from FEBio logs and simplifies
            collecting diagnostics.
    """

    mode: StorageMode = "disk"
    root: str | Path | None = None
    log_file: str | Path | None = None


@dataclass(slots=True)
class MonitorOptions:
    """Monitoring client configuration.

    Attributes:
        enabled (bool): Master switch for telemetry. Disable to avoid any IPC
            overhead when monitoring is unnecessary.
        socket (str | Path | None): UNIX socket path used to communicate with
            monitoring services. Override when the default auto-detected socket
            is not available.
        label (str | None): Descriptive name published to the monitor, helping
            differentiate runs when multiple optimisations are observed
            simultaneously.
    """

    enabled: bool = True
    socket: str | Path | None = None
    label: str | None = None


@dataclass(slots=True)
class OptimizerOptions:
    """Optimizer adapter selection and advanced settings.

    Attributes:
        name (OptimizerName): Optimiser interface used by the engine.
            ``"least_squares"`` hooks into ``scipy.optimize.least_squares``
            while ``"minimize"`` leverages ``scipy.optimize.minimize``.
        settings (Mapping[str, Any]): Extra keyword arguments forwarded to the
            SciPy call. Use this to set tolerances, iteration limits, or
            algorithm-specific knobs.
        reparametrize (bool): When ``True`` the engine optimises in φ-space
            using the exponential mapping controlled by :class:`ParameterSpace`.
            Setting this to ``False`` switches the optimiser to operate directly
            on θ values, effectively ignoring the ``xi`` reparameterisation.
    """

    name: OptimizerName = "least_squares"
    settings: Mapping[str, Any] = field(default_factory=dict)
    reparametrize: bool = True


@dataclass(slots=True)
class EngineOptions:
    """Aggregate container for optimisation engine configuration.

    Attributes:
        grid (GridPolicyOptions): Controls how experimental and simulated data
            align on a shared grid.
        jacobian (JacobianOptions): Determines whether and how finite-difference
            Jacobians are evaluated.
        cleanup (CleanupOptions): Governs interim and final cleanup of working
            directories.
        runner (RunnerOptions): Selects the execution backend configuration for
            FEBio commands.
        storage (StorageOptions): Defines where artefacts and logs are written.
        monitor (MonitorOptions): Configures emission of telemetry to external
            monitoring tools.
        optimizer (OptimizerOptions): Picks and tunes the optimisation backend
            responsible for driving parameter updates.
    """

    grid: GridPolicyOptions = field(default_factory=GridPolicyOptions)
    jacobian: JacobianOptions = field(default_factory=JacobianOptions)
    cleanup: CleanupOptions = field(default_factory=CleanupOptions)
    runner: RunnerOptions = field(default_factory=RunnerOptions)
    storage: StorageOptions = field(default_factory=StorageOptions)
    monitor: MonitorOptions = field(default_factory=MonitorOptions)
    optimizer: OptimizerOptions = field(default_factory=OptimizerOptions)


__all__ = [
    "CleanupOptions",
    "EngineOptions",
    "GridPolicyOptions",
    "JacobianOptions",
    "MonitorOptions",
    "OptimizerOptions",
    "RunnerOptions",
    "StorageOptions",
]
