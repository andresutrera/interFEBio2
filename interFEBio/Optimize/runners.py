from __future__ import annotations

import atexit
import os
import shlex
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, TextIO, Tuple, cast


DEFAULT_TEE = object()


@dataclass
class RunResult:
    """Metadata captured for a finished FEBio job."""

    exit_code: int
    started_at: float
    ended_at: float
    log_path: Path
    job_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.ended_at - self.started_at


@dataclass
class RunHandle:
    """Future-like handle returned by Runner.run."""

    _future: Future[RunResult]

    def wait(self, timeout: Optional[float] = None) -> RunResult:
        return self._future.result(timeout)

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        return self._future.cancel()

    def result(self) -> RunResult:
        return self._future.result()

    def __await__(self):
        return self._future.__await__()


class Runner:
    """Common interface for launching FEBio simulations."""

    def run(self, job_dir: str | Path, feb_name: str | Path) -> RunHandle:
        raise NotImplementedError

    def shutdown(self) -> None:
        """Allow subclasses to stop background resources."""
        return None


class _BaseLocalRunner(Runner):
    """Shared logic for local launchers."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        max_workers: int = 1,
        env: Optional[Dict[str, str]] = None,
        tee_stream: Optional[TextIO] | object = DEFAULT_TEE,
    ):
        self.command = tuple(command or ("febio", "-i"))
        if not self.command:
            raise ValueError("command may not be empty.")
        self.max_workers = max(1, int(max_workers))
        self.env = dict(env) if env else None
        if tee_stream is DEFAULT_TEE:
            self._tee_stream: Optional[TextIO] = None
        else:
            self._tee_stream = cast(Optional[TextIO], tee_stream)
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self._executor_lock = threading.Lock()
        atexit.register(self.shutdown)

    def run(self, job_dir: str | Path, feb_name: str | Path) -> RunHandle:
        job_path = Path(job_dir)
        feb_path = Path(feb_name)
        future = self._executor.submit(self._run_once, job_path, feb_path)
        return RunHandle(future)

    def shutdown(self) -> None:
        with self._executor_lock:
            self._executor.shutdown(wait=False)

    # ----- helpers -----
    def _build_command(self, feb_path: Path) -> Sequence[str]:
        return [*self.command, str(feb_path)]

    def _merged_env(self) -> Optional[Dict[str, str]]:
        if self.env is None:
            return None
        merged = os.environ.copy()
        merged.update(self.env)
        return merged

    def _run_once(self, job_path: Path, feb_path: Path) -> RunResult:
        job_path.mkdir(parents=True, exist_ok=True)
        if not feb_path.is_absolute():
            feb_path = job_path / feb_path
        if not feb_path.exists():
            raise FileNotFoundError(f"FEB file not found: {feb_path}")

        log_path = feb_path.with_suffix(".log")
        cmd = list(self._build_command(feb_path))

        started = time.time()
        proc = subprocess.Popen(
            cmd,
            cwd=str(job_path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=self._merged_env(),
        )
        log_file: Optional[TextIO] = None
        try:
            if proc.stdout is None:
                proc.wait()
            else:
                for _ in proc.stdout:
                    pass
                proc.wait()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
        ended = time.time()
        exit_code = proc.returncode if proc.returncode is not None else -1
        default_log = feb_path.with_suffix(".log")
        meta = {
            "cmd": cmd,
            "log_path": str(log_path),
            "default_log": str(default_log),
        }
        return RunResult(
            exit_code=exit_code,
            started_at=started,
            ended_at=ended,
            log_path=log_path,
            metadata=meta,
        )

    def _tee(self, text: str) -> None:
        if self._tee_stream is None:
            return
        try:
            print(text, end="", file=self._tee_stream, flush=True)
        except Exception:
            # Avoid crashing runners if the stream is closed.
            self._tee_stream = None


class LocalSerialRunner(_BaseLocalRunner):
    """Serial FEBio execution with stdout teeing."""

    def __init__(
        self,
        command: Sequence[str] | None = None,
        *,
        env: Optional[Dict[str, str]] = None,
        tee_stream: Optional[TextIO] | object = DEFAULT_TEE,
    ):
        super().__init__(
            command,
            max_workers=1,
            env=env,
            tee_stream=tee_stream,
        )


class LocalParallelRunner(_BaseLocalRunner):
    """Thread-pooled FEBio launcher limited by n_jobs."""

    def __init__(
        self,
        n_jobs: int,
        command: Sequence[str] | None = None,
        *,
        env: Optional[Dict[str, str]] = None,
        tee_stream: Optional[TextIO] | object = DEFAULT_TEE,
    ):
        if n_jobs < 1:
            raise ValueError("n_jobs must be >= 1.")
        super().__init__(
            command,
            max_workers=n_jobs,
            env=env,
            tee_stream=tee_stream,
        )
        self.n_jobs = n_jobs


class SlurmRunner(Runner):
    """Submit FEBio runs through SLURM via sbatch."""

    _PENDING_STATES = {
        "PENDING",
        "CONFIGURING",
        "RUNNING",
        "COMPLETING",
        "STAGE_OUT",
        "SUSPENDED",
    }

    def __init__(
        self,
        sbatch_opts: Iterable[str] | None = None,
        *,
        poll_interval: float = 10.0,
        command: Sequence[str] | None = None,
        env: Optional[Dict[str, str]] = None,
        tee_stream: Optional[TextIO] = None,
    ):
        self.command = tuple(command or ("febio", "-i"))
        if not self.command:
            raise ValueError("command may not be empty.")
        self.sbatch_opts = tuple(sbatch_opts or ())
        self.poll_interval = max(0.5, float(poll_interval))
        self.env = dict(env) if env else None
        self._tee_stream = tee_stream if tee_stream is not None else sys.stdout
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._executor_lock = threading.Lock()
        atexit.register(self.shutdown)

    def run(self, job_dir: str | Path, feb_name: str | Path) -> RunHandle:
        job_path = Path(job_dir)
        feb_path = Path(feb_name)
        future = self._executor.submit(self._submit_and_monitor, job_path, feb_path)
        return RunHandle(future)

    def shutdown(self) -> None:
        with self._executor_lock:
            self._executor.shutdown(wait=False)

    # ----- core workflow -----
    def _submit_and_monitor(self, job_path: Path, feb_path: Path) -> RunResult:
        job_path.mkdir(parents=True, exist_ok=True)
        if not feb_path.is_absolute():
            feb_path = job_path / feb_path
        if not feb_path.exists():
            raise FileNotFoundError(f"FEB file not found: {feb_path}")

        log_path = job_path / "stdout.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        # Touch the log so tailing works before content arrives.
        log_path.touch(exist_ok=True)

        cmd = [*self.command, str(feb_path)]
        script_path = job_path / f"slurm_runner_{uuid.uuid4().hex}.sh"
        script = "\n".join(
            [
                "#!/bin/bash",
                "set -euo pipefail",
                "",
                " ".join(shlex.quote(part) for part in cmd),
                "",
            ]
        )
        script_path.write_text(script, encoding="utf-8")
        script_path.chmod(0o750)

        sbatch = [
            "sbatch",
            "--parsable",
            "--chdir",
            str(job_path),
            "--output",
            str(log_path),
            "--error",
            str(log_path),
            *self.sbatch_opts,
            str(script_path),
        ]
        started = time.time()
        try:
            proc = subprocess.run(
                sbatch,
                capture_output=True,
                text=True,
                check=True,
                env=self._merged_env(),
            )
        except FileNotFoundError as exc:
            raise RuntimeError("sbatch command not found in PATH.") from exc
        finally:
            # Script is no longer needed once submitted.
            with suppress(FileNotFoundError):
                script_path.unlink()

        job_id = proc.stdout.strip().split(";")[0]
        if not job_id:
            raise RuntimeError(f"Failed to parse sbatch output: {proc.stdout!r}")

        exit_code = None
        state = None
        log_offset = 0

        while True:
            log_offset = self._tail_log(log_path, log_offset)
            state, exit_candidate = self._query_state(job_id)
            if state is None:
                time.sleep(self.poll_interval)
                continue
            if state in self._PENDING_STATES:
                exit_code = exit_candidate
                time.sleep(self.poll_interval)
                continue
            exit_code = exit_candidate if exit_candidate is not None else 1
            break

        # Final flush of remaining log content.
        self._tail_log(log_path, log_offset)
        ended = time.time()

        meta = {
            "cmd": cmd,
            "sbatch_cmd": sbatch,
            "final_state": state,
        }
        return RunResult(
            exit_code=exit_code,
            started_at=started,
            ended_at=ended,
            log_path=log_path,
            job_id=job_id,
            metadata=meta,
        )

    # ----- helpers -----
    def _merged_env(self) -> Optional[Dict[str, str]]:
        if self.env is None:
            return None
        merged = os.environ.copy()
        merged.update(self.env)
        return merged

    def _tail_log(self, log_path: Path, offset: int) -> int:
        if not log_path.exists():
            return offset
        try:
            with open(log_path, "r", encoding="utf-8", errors="replace") as log_file:
                log_file.seek(offset)
                chunk = log_file.read()
                if chunk:
                    self._tee(chunk)
                return log_file.tell()
        except OSError:
            return offset

    def _tee(self, text: str) -> None:
        if self._tee_stream is None:
            return
        try:
            print(text, end="", file=self._tee_stream, flush=True)
        except Exception:
            self._tee_stream = None

    def _query_state(self, job_id: str) -> Tuple[Optional[str], Optional[int]]:
        try:
            proc = subprocess.run(
                ["scontrol", "show", "job", job_id],
                capture_output=True,
                text=True,
                check=True,
                env=self._merged_env(),
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            return None, None
        if not proc.stdout.strip():
            return None, None
        return self._parse_scontrol(proc.stdout)

    @staticmethod
    def _parse_scontrol(payload: str) -> Tuple[Optional[str], Optional[int]]:
        state = None
        exit_code = None
        tokens = payload.replace("\n", " ").split()
        for token in tokens:
            if token.startswith("JobState="):
                state = token.split("=", 1)[1]
                if state and "," in state:
                    state = state.split(",", 1)[0]
            elif token.startswith("ExitCode="):
                val = token.split("=", 1)[1]
                primary = val.split(":", 1)[0]
                with suppress(ValueError):
                    exit_code = int(primary)
        return state, exit_code


__all__ = [
    "RunHandle",
    "RunResult",
    "Runner",
    "LocalSerialRunner",
    "LocalParallelRunner",
    "SlurmRunner",
]
