"""Helpers to resolve default directories used by the monitoring components."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _ensure_dir(path: Path) -> Path:
    """Create the directory tree if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def _first_writable(candidates: list[Path]) -> Optional[Path]:
    """Return the first path that can be created for writing."""
    for candidate in candidates:
        try:
            candidate = candidate.expanduser()
            candidate.mkdir(parents=True, exist_ok=True)
        except OSError:
            continue
        return candidate
    return None


def default_runtime_dir() -> Path:
    """Resolve the directory used for runtime sockets and locks."""
    env = os.environ.get("INTERFEBIO_MONITOR_RUNTIME")
    if env:
        return _ensure_dir(Path(env).expanduser())
    candidates = []
    if os.name == "posix":
        candidates.append(Path("/run/interfebio-monitor"))
    candidates.append(Path.home() / ".cache" / "interfebio-monitor")
    candidates.append(Path(".") / ".interfebio-runtime")
    runtime = _first_writable(candidates)
    if runtime is None:
        raise RuntimeError("Could not determine writable runtime directory for monitor.")
    return runtime


def default_data_dir() -> Path:
    """Determine the directory for persistent monitoring data."""
    env = os.environ.get("INTERFEBIO_MONITOR_DATA")
    if env:
        return _ensure_dir(Path(env).expanduser())
    candidates = []
    if os.name == "posix":
        candidates.append(Path("/var/lib/interfebio-monitor"))
    candidates.append(Path.home() / ".local" / "share" / "interfebio-monitor")
    candidates.append(Path(".") / ".interfebio-data")
    data_dir = _first_writable(candidates)
    if data_dir is None:
        raise RuntimeError("Could not determine writable data directory for monitor.")
    return data_dir


def default_socket_path() -> Path:
    """Return the default UNIX socket path for the monitor."""
    return default_runtime_dir() / "monitor.sock"


def default_registry_path() -> Path:
    """Return the default path where completed runs are listed."""
    return default_data_dir() / "runs.jsonl"


__all__ = [
    "default_data_dir",
    "default_registry_path",
    "default_runtime_dir",
    "default_socket_path",
]
