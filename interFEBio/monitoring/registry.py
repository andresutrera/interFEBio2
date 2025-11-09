"""Manage the JSONL-backed registry of optimization runs."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Dict, List, Optional

from .runstate import OptimizationRun


class ActiveRunDeletionError(RuntimeError):
    """Raised when attempting to delete an optimisation run that is still active."""

    def __init__(self, run_id: str):
        super().__init__(f"Run '{run_id}' is still running and cannot be deleted.")
        self.run_id = run_id


class RunRegistry:
    """Thread-safe registry storing optimization run snapshots backed by JSONL."""

    def __init__(self, db_path: Path, *, max_history: Optional[int] = None):
        """Initialize the registry with the backing database path."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._runs: Dict[str, OptimizationRun] = {}
        self._max_history = max_history
        self._load()

    def list_runs(self) -> List[OptimizationRun]:
        """Return all tracked optimization runs."""
        with self._lock:
            return [run for run in self._runs.values()]

    def get_run(self, run_id: str) -> Optional[OptimizationRun]:
        """Return the run snapshot for the given identifier."""
        with self._lock:
            return self._runs.get(run_id)

    def apply_event(
        self, run_id: str, event: str, payload: Dict[str, object], ts: float
    ) -> None:
        """Incorporate an incoming event into the relevant run."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                run = OptimizationRun(run_id=run_id, label=run_id)
                self._runs[run_id] = run
            run.apply_event(event, payload, ts)
            if self._max_history is not None and len(run.iterations) > self._max_history:
                # Drop oldest entries to keep storage bounded.
                drop = len(run.iterations) - self._max_history
                del run.iterations[0:drop]
            self._sync_locked()

    def refresh(self) -> None:
        """Reload the registry contents from disk."""
        self._load()

    def delete_run(self, run_id: str) -> bool:
        """Remove a finished run if it is safe to delete."""
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                return False
            if run.status in {"created", "running"}:
                raise ActiveRunDeletionError(run_id)
            del self._runs[run_id]
            self._sync_locked()
            return True
        return False

    def clear(self) -> List[str]:
        """Drop all deletable runs while listing those that remain."""
        with self._lock:
            protected: List[str] = []
            removable = []
            for run_id, run in self._runs.items():
                if run.status in {"created", "running"}:
                    protected.append(run_id)
                else:
                    removable.append(run_id)
            if not removable:
                return protected
            for run_id in removable:
                del self._runs[run_id]
            self._sync_locked()
            return protected

    def _load(self) -> None:
        """Load registry entries from the JSONL file on disk."""
        if not self.db_path.exists():
            return
        try:
            with self.db_path.open("r", encoding="utf-8") as fh:
                runs: Dict[str, OptimizationRun] = {}
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    run = OptimizationRun.from_dict(payload)
                    runs[run.run_id] = run
        except OSError:
            return
        with self._lock:
            self._runs = runs

    def _sync_locked(self) -> None:
        """Persist the current cache to disk while holding the lock."""
        tmp_path = self.db_path.with_suffix(".tmp")
        try:
            with tmp_path.open("w", encoding="utf-8") as fh:
                for run in self._runs.values():
                    fh.write(json.dumps(run.to_dict()) + "\n")
                    fh.flush()
        except OSError:
            return
        try:
            tmp_path.replace(self.db_path)
        except OSError:
            tmp_path.unlink(missing_ok=True)


__all__ = ["RunRegistry", "ActiveRunDeletionError"]
