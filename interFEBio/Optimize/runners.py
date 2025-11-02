"""Lightweight helpers for launching local FEBio runs."""

from __future__ import annotations

import atexit
import os
import subprocess
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Sequence
from contextlib import suppress
import threading


@dataclass
class RunResult:
    """Summary information about a finished simulation command."""

    exit_code: int
    started_at: float
    ended_at: float
    log_path: Path
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.ended_at - self.started_at


@dataclass
class RunHandle:
    """Future-like wrapper returned by runner implementations."""

    _future: Future[RunResult]

    def wait(self, timeout: float | None = None) -> RunResult:
        return self._future.result(timeout)

    def done(self) -> bool:  # pragma: no cover - passthrough convenience
        return self._future.done()

    def cancel(self) -> bool:  # pragma: no cover - passthrough convenience
        return self._future.cancel()

    def result(self) -> RunResult:  # pragma: no cover - passthrough convenience
        return self._future.result()

    def __await__(self):  # pragma: no cover - passthrough convenience
        return self._future.__await__()


class Runner:
    """Minimal interface expected by the optimisation engine."""

    def run(
        self,
        job_dir: str | Path,
        feb_name: str | Path,
        *,
        env: dict[str, str] | None = None,
    ) -> RunHandle:
        raise NotImplementedError

    def shutdown(self) -> None:  # pragma: no cover - simple default
        return None


class _BaseLocalRunner(Runner):
    """Execute FEBio commands locally, optionally in parallel."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        max_workers: int = 1,
        env: dict[str, str] | None = None,
    ):
        self.command = tuple(command or ("febio4", "-i"))
        if not self.command:
            raise ValueError("command may not be empty.")
        self.env = dict(env) if env else None
        self._executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)))
        self._active: set[subprocess.Popen] = set()
        self._active_lock = threading.Lock()
        atexit.register(self.shutdown)

    def run(
        self,
        job_dir: str | Path,
        feb_name: str | Path,
        *,
        env: dict[str, str] | None = None,
    ) -> RunHandle:
        job_path = Path(job_dir)
        feb_path = Path(feb_name)
        future = self._executor.submit(self._run_once, job_path, feb_path, env)
        return RunHandle(future)

    def shutdown(self) -> None:
        with self._active_lock:
            procs = list(self._active)
        for proc in procs:
            with suppress(Exception):
                proc.terminate()
        for proc in procs:
            with suppress(Exception):
                proc.wait(timeout=5)
        self._executor.shutdown(wait=False)

    # ---- internal helpers -------------------------------------------------
    def _run_once(
        self,
        job_path: Path,
        feb_name: Path,
        env: dict[str, str] | None = None,
    ) -> RunResult:
        job_path.mkdir(parents=True, exist_ok=True)
        feb_path = feb_name if feb_name.is_absolute() else job_path / feb_name
        if not feb_path.exists():
            raise FileNotFoundError(f"FEB file not found: {feb_path}")

        log_path = feb_path.with_suffix(".log")
        cmd = [*self.command, str(feb_path)]
        started = time.time()
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.Popen(
                cmd,
                cwd=str(job_path),
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=self._merged_env(env),
                start_new_session=True,
            )
            with self._active_lock:
                self._active.add(proc)
            try:
                proc.wait()
                returncode = proc.returncode
            finally:
                with self._active_lock:
                    self._active.discard(proc)
        ended = time.time()
        result = RunResult(
            exit_code=returncode,
            started_at=started,
            ended_at=ended,
            log_path=log_path,
            metadata={"cmd": cmd},
        )
        return result

    def _merged_env(
        self,
        override: dict[str, str] | None = None,
    ) -> dict[str, str] | None:
        if self.env is None and not override:
            return None
        merged = os.environ.copy()
        if self.env:
            merged.update(self.env)
        if override:
            merged.update(override)
        return merged


class LocalSerialRunner(_BaseLocalRunner):
    """Execute simulations sequentially on the local machine."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        env: dict[str, str] | None = None,
    ):
        super().__init__(command, max_workers=1, env=env)


class LocalParallelRunner(_BaseLocalRunner):
    """Execute simulations concurrently using a thread pool."""

    def __init__(
        self,
        n_jobs: int,
        command: Sequence[str] | None = None,
        *,
        env: dict[str, str] | None = None,
    ):
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1.")
        super().__init__(command, max_workers=n_jobs, env=env)
        self.n_jobs = int(n_jobs)


__all__ = [
    "RunHandle",
    "RunResult",
    "Runner",
    "LocalSerialRunner",
    "LocalParallelRunner",
]
