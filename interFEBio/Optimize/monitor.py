"""
Monitoring helpers for FEBio optimization jobs.

The classes here provide lightweight, thread-safe building blocks for
tracking job status, tailing logs, and aggregating progress updates that a UI
or CLI can consume without interfering with long-running simulations.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO

from interFEBio.monitoring.events import EventEmitter, NullEventEmitter


@dataclass
class JobView:
    """Snapshot of a job's status for monitoring purposes."""

    key: str
    name: str
    job_dir: Path
    log_path: Path
    status: str = "pending"
    last_log_line: str = ""
    exit_code: Optional[int] = None
    updated_at: float = field(default_factory=time.time)
    meta: Dict[str, object] = field(default_factory=dict)

    def clone(self) -> "JobView":
        """Return a shallow clone that can be safely mutated by callers."""
        return replace(
            self,
            job_dir=Path(self.job_dir),
            log_path=Path(self.log_path),
            meta=dict(self.meta),
        )


class MonitorFSView:
    """
    Thread-safe registry for job views.

    Production code can populate this from StorageManager manifests, while
    lightweight runners may register ad-hoc jobs.
    """

    def __init__(self):
        self._jobs: Dict[str, JobView] = {}
        self._lock = threading.Lock()

    # --- registration & snapshots ---
    def register(self, job: JobView) -> None:
        """Register a single job view."""
        with self._lock:
            self._jobs[job.key] = job

    def register_many(self, jobs: Iterable[JobView]) -> None:
        """Register multiple job views in one operation."""
        with self._lock:
            for job in jobs:
                self._jobs[job.key] = job

    def get(self, key: str) -> Optional[JobView]:
        """Return a cloned job view by key, or ``None`` if not registered."""
        with self._lock:
            job = self._jobs.get(key)
            return job.clone() if job else None

    def snapshot(self) -> List[JobView]:
        """Return a cloned snapshot of all registered job views."""
        with self._lock:
            return [job.clone() for job in self._jobs.values()]

    # --- updates ---
    def set_status(
        self, key: str, status: str, exit_code: Optional[int] = None
    ) -> None:
        """Update status and optional exit code for the given job."""
        with self._lock:
            job = self._jobs.get(key)
            if not job:
                return
            job.status = status
            if exit_code is not None:
                job.exit_code = exit_code
            job.updated_at = time.time()

    def record_log_metadata(self, key: str, meta: Dict[str, object]) -> None:
        """Merge log-derived metadata into the stored job view."""
        with self._lock:
            job = self._jobs.get(key)
            if not job:
                return
            for k, value in meta.items():
                if k in {"time"} and isinstance(value, (int, float)):
                    existing = job.meta.get(k)
                    if not isinstance(existing, (int, float)) or value >= existing:
                        job.meta[k] = float(value)
                elif k in {"step", "substep", "substeps_total"} and isinstance(
                    value, (int, float)
                ):
                    existing = job.meta.get(k)
                    val_int = int(value)
                    if not isinstance(existing, (int, float)) or val_int >= int(
                        existing
                    ):
                        job.meta[k] = val_int
                else:
                    job.meta[k] = value
            # Derived values
            if "time" not in job.meta:
                step = job.meta.get("step")
                step_size = job.meta.get("step_size")
                if isinstance(step, (int, float)) and isinstance(
                    step_size, (int, float)
                ):
                    job.meta["time"] = float(step) * float(step_size)
            job.updated_at = time.time()


_STEP_RE = re.compile(r"\b(?:step|STEP)\s*(?:=|:)?\s*(\d+)")
_SUB_RE = re.compile(r"\b(?:substep|SUBSTEP)\s*(?:=|:)?\s*(\d+)(?:\s*/\s*(\d+))?")
_TIME_RE = re.compile(r"\b(?:time|TIME)\s*(?:=|:)?\s*([0-9.+\-Ee]+)")
_TIME_STEPS_RE = re.compile(r"time_steps[^:]*:\s*(\d+)", re.IGNORECASE)
_STEP_SIZE_RE = re.compile(r"step_size[^:]*:\s*([0-9.+\-Ee]+)", re.IGNORECASE)


def _extract_progress(line: str) -> Dict[str, object]:
    """
    Extract basic progress indicators from a FEBio log line.

    Recognises tokens like "Step 4", "substep 3/10", and "time = 0.123".
    """

    info: Dict[str, object] = {}
    if match := _STEP_RE.search(line):
        info["step"] = int(match.group(1))
    if match := _SUB_RE.search(line):
        info["substep"] = int(match.group(1))
        if match.group(2):
            info["substeps_total"] = int(match.group(2))
    if match := _TIME_RE.search(line):
        token = match.group(1).strip()
        if token and any(ch.isdigit() for ch in token):
            try:
                info["time"] = float(token)
            except ValueError:
                info["time"] = token
    return info


def _extract_config(line: str) -> Dict[str, object]:
    """Extract static configuration hints from header log lines."""
    info: Dict[str, object] = {}
    if match := _TIME_STEPS_RE.search(line):
        try:
            info["time_steps"] = int(match.group(1))
        except ValueError:
            pass
    if match := _STEP_SIZE_RE.search(line):
        token = match.group(1).strip()
        if token and any(ch.isdigit() for ch in token):
            try:
                info["step_size"] = float(token)
            except ValueError:
                info["step_size"] = token
    return info


class LogTailer:
    """Incremental log reader used by the monitoring utilities."""

    def __init__(
        self,
        path: Path,
        *,
        encoding: str = "utf-8",
        chunk_size: int = 1 << 14,
    ):
        """Initialise the tailer with the target log path."""
        self.path = Path(path)
        self.encoding = encoding
        self.chunk_size = chunk_size
        self._offset = 0
        self._buffer = ""
        self._inode = None
        self._lock = threading.Lock()

    def reset(self) -> None:
        """Reset internal offsets so polling restarts from the beginning."""
        with self._lock:
            self._offset = 0
            self._buffer = ""
            self._inode = None

    def poll(self) -> List[str]:
        """
        Return newly appended complete lines since the last poll.
        Incomplete trailing fragments are buffered until the next call.
        """
        with self._lock:
            try:
                stat = self.path.stat()
            except FileNotFoundError:
                self._offset = 0
                self._buffer = ""
                self._inode = None
                return []

            inode = getattr(stat, "st_ino", None)
            if inode is not None and inode != self._inode:
                # Log was rotated or replaced.
                self._offset = 0
                self._buffer = ""
                self._inode = inode
            elif stat.st_size < self._offset:
                # Truncated.
                self._offset = 0
                self._buffer = ""

            if stat.st_size == self._offset:
                return []

            with self.path.open("r", encoding=self.encoding, errors="replace") as fh:
                fh.seek(self._offset)
                data = fh.read(self.chunk_size)
                if not data:
                    return []
                self._offset = fh.tell()

            text = self._buffer + data
            if text.endswith("\n"):
                self._buffer = ""
                lines = text.splitlines()
            else:
                parts = text.splitlines()
                if parts:
                    self._buffer = parts[-1]
                    lines = parts[:-1]
                else:
                    self._buffer = text
                    lines = []
            return lines


class XpltProbe:
    """Lightweight state tracker for ``.xplt`` files."""

    def __init__(self, path: Path):
        """Initialise the probe for the supplied file path."""
        self.path = Path(path)
        self._last_size = -1
        self._last_mtime = -1.0

    def poll(self) -> Optional[Dict[str, float]]:
        """Return file metadata when the file changes, otherwise ``None``."""
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            self._last_size = -1
            self._last_mtime = -1.0
            return None
        if stat.st_size == self._last_size and stat.st_mtime == self._last_mtime:
            return None
        self._last_size = stat.st_size
        self._last_mtime = stat.st_mtime
        return {"size": float(stat.st_size), "mtime": float(stat.st_mtime)}


class ProgressAggregator:
    """Background worker that tails logs and aggregates job status."""

    def __init__(
        self,
        view: MonitorFSView,
        *,
        poll_interval: float = 1.0,
        stream: Optional[TextIO] = None,
        line_char_limit: int = 96,
        event_emitter: Optional[EventEmitter] = None,
    ):
        self.view = view
        self.poll_interval = max(0.1, float(poll_interval))
        self.stream = stream
        self.line_char_limit = max(32, int(line_char_limit))
        self._tailers: Dict[str, LogTailer] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._render_lock = threading.Lock()
        self._pending_render = False
        self._last_render = ""
        self._emitter = event_emitter or NullEventEmitter()

    def register_jobs(self, keys: Iterable[str]) -> None:
        """Attach log tailers for the supplied job keys."""
        for key in keys:
            job = self.view.get(key)
            if not job:
                continue
            self._tailers[key] = LogTailer(job.log_path)
        # Force an early render with initial state when running.
        self._mark_dirty()

    def start(self) -> None:
        """Start the background polling thread if it is not already running."""
        if self._thread is not None:
            return
        if not self._tailers:
            # Make sure tailers are ready even if register_jobs wasn't called.
            self.register_jobs(job.key for job in self.view.snapshot())
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="ProgressAggregator")
        self._thread.daemon = True
        self._thread.start()

    def stop(self, force_render: bool = True) -> None:
        """Stop the background thread and optionally emit a final snapshot."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=self.poll_interval * 2.0)
        self._thread = None
        # One more pass to capture log lines written after the worker stopped.
        dirty = self._poll_tailers()
        self._backfill_progress()
        if force_render and self.stream is not None:
            # Emit final snapshot with latest statuses.
            if dirty:
                self._mark_dirty()
            self._render(force=True)

    def update_status(
        self, key: str, status: str, exit_code: Optional[int] = None
    ) -> None:
        """Forward status changes to the view and emit monitoring events."""
        self.view.set_status(key, status, exit_code)
        self._mark_dirty()
        self._emitter.emit(key, "status", {"status": status, "exit_code": exit_code})

    def register_tail(self, key: str) -> None:
        """Attach a tailer for a job that started after initial registration."""
        job = self.view.get(key)
        if not job:
            return
        self._tailers[key] = LogTailer(job.log_path)
        self._mark_dirty()

    # ---- internals ----
    def _run(self) -> None:
        # Initial render.
        self._render(force=True)
        while not self._stop_event.wait(self.poll_interval):
            dirty = self._poll_tailers()
            if dirty or self._consume_pending_render():
                self._render()
        # Final poll to capture any remaining data.
        dirty = self._poll_tailers()
        if dirty or self._consume_pending_render():
            self._render()

    def _poll_tailers(self) -> bool:
        dirty = False
        for key, tailer in list(self._tailers.items()):
            try:
                lines = tailer.poll()
            except Exception:
                continue
            if not lines:
                continue
            job_snapshot = self.view.get(key)
            if job_snapshot and job_snapshot.status not in {
                "running",
                "finished",
                "failed",
            }:
                self.view.set_status(key, "running")
                self._emitter.emit(key, "status", {"status": "running"})
            for raw in lines:
                clean = raw.strip()
                if not clean:
                    continue
                meta_updates: Dict[str, object] = {}
                config = _extract_config(clean)
                if config:
                    meta_updates.update(config)
                progress = _extract_progress(clean)
                if progress:
                    meta_updates.update(progress)
                if meta_updates:
                    self.view.record_log_metadata(key, meta_updates)
                    dirty = True
            job_snapshot = self.view.get(key)
            if job_snapshot:
                total_steps = job_snapshot.meta.get("time_steps")
                step_size = job_snapshot.meta.get("step_size")
                current_step = job_snapshot.meta.get("step")
                if (
                    isinstance(total_steps, (int, float))
                    and isinstance(current_step, (int, float))
                    and int(current_step) >= int(total_steps)
                ):
                    if isinstance(step_size, (int, float)):
                        final_time = float(step_size) * float(total_steps)
                        self.view.record_log_metadata(key, {"time": final_time})
                    if job_snapshot.status != "finished":
                        self.view.set_status(key, "finished")
                        self._mark_dirty()
                        self._emitter.emit(
                            key,
                            "status",
                            {
                                "status": "finished",
                                "time": job_snapshot.meta.get("time"),
                            },
                        )
        return dirty

    def _backfill_progress(self) -> None:
        for key, _tailer in self._tailers.items():
            job = self.view.get(key)
            if not job:
                continue
            path = Path(job.log_path)
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except FileNotFoundError:
                continue
            summary: Dict[str, object] = {}
            for line in text.splitlines():
                clean = line.strip()
                if not clean:
                    continue
                config = _extract_config(clean)
                if config:
                    summary.update(config)
                progress = _extract_progress(clean)
                if progress:
                    summary.update(progress)
            if summary:
                self.view.record_log_metadata(key, summary)
                job_after = self.view.get(key)
                if job_after:
                    total_steps = job_after.meta.get("time_steps")
                    step_size = job_after.meta.get("step_size")
                    current_step = job_after.meta.get("step")
                    if (
                        isinstance(total_steps, (int, float))
                        and isinstance(current_step, (int, float))
                        and int(current_step) >= int(total_steps)
                    ):
                        if isinstance(step_size, (int, float)):
                            final_time = float(step_size) * float(total_steps)
                            self.view.record_log_metadata(key, {"time": final_time})
                        if job_after.status != "finished":
                            self.view.set_status(key, "finished")
                            self._emitter.emit(
                                key,
                                "status",
                                {
                                    "status": "finished",
                                    "time": job_after.meta.get("time"),
                                },
                            )

    def _mark_dirty(self) -> None:
        with self._render_lock:
            self._pending_render = True

    def _consume_pending_render(self) -> bool:
        with self._render_lock:
            flag = self._pending_render
            self._pending_render = False
            return flag

    def _render(self, force: bool = False) -> None:
        if self.stream is None:
            return
        snapshot = sorted(self.view.snapshot(), key=lambda j: j.key)
        lines = []
        for job in snapshot:
            exit_fragment = "" if job.exit_code is None else f" exit={job.exit_code}"
            timestamp = time.strftime("%H:%M:%S", time.localtime(job.updated_at))
            progress_tokens: List[str] = []
            if (t := job.meta.get("time")) is not None:
                if isinstance(t, float):
                    progress_tokens.append(f"t={t:.5g}")
                else:
                    progress_tokens.append(f"t={t}")
            if (step := job.meta.get("step")) is not None:
                progress_tokens.append(f"step {step}")
            if (sub := job.meta.get("substep")) is not None:
                total = job.meta.get("substeps_total")
                if total:
                    progress_tokens.append(f"sub {sub}/{total}")
                else:
                    progress_tokens.append(f"sub {sub}")
            progress_str = " ".join(progress_tokens) or "-"
            lines.append(
                f"{timestamp} {job.key:<12} {job.status:<10}{exit_fragment} "
                f"| {progress_str:<18}"
            )
        output = "\n".join(lines)
        if not output:
            return
        if force or output != self._last_render:
            print(output, file=self.stream, flush=True)
            self._last_render = output


__all__ = [
    "JobView",
    "MonitorFSView",
    "LogTailer",
    "XpltProbe",
    "ProgressAggregator",
]
