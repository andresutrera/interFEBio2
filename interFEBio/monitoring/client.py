from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .events import EventEmitter, NullEventEmitter, create_event_emitter
from .paths import default_socket_path


def _slug(value: str) -> str:
    return "".join(ch for ch in value if ch.isalnum() or ch in ("-", "_")).strip("_") or "run"


def generate_run_id(label: str | None = None) -> str:
    prefix = _slug(label or "run")
    suffix = uuid.uuid4().hex[:12]
    return f"{prefix}-{suffix}"


@dataclass
class MonitorConfig:
    socket_path: Optional[Path] = None
    run_id: Optional[str] = None
    label: Optional[str] = None


class OptimizationMonitorClient:
    """
    Helper that emits optimisation lifecycle events to the monitor service.
    """

    def __init__(self, config: MonitorConfig | None = None):
        if config is None:
            config = MonitorConfig()
        socket_path = config.socket_path or default_socket_path()
        emitter = create_event_emitter(socket_path)
        self._emitter: EventEmitter = emitter
        self.run_id = config.run_id or generate_run_id(config.label)
        self.label = config.label or self.run_id
        self._started = False

    def run_started(
        self,
        *,
        parameters: Mapping[str, Any] | None = None,
        cases: list[Mapping[str, Any]] | None = None,
        optimizer: Mapping[str, Any] | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "label": self.label,
            "parameters": dict(parameters or {}),
            "meta": dict(meta or {}),
        }
        if cases:
            payload["cases"] = list(cases)
        if optimizer:
            payload["optimizer"] = dict(optimizer)
        self._emit("run_started", payload)
        self._started = True

    def record_iteration(
        self,
        *,
        index: int,
        cost: float,
        theta: Mapping[str, float],
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "index": int(index),
            "cost": float(cost),
            "theta": {name: float(val) for name, val in theta.items()},
            "metrics": dict(metrics or {}),
            "timestamp": time.time(),
        }
        self._emit("iteration", payload)

    def run_completed(
        self,
        *,
        best_cost: Optional[float] = None,
        summary: Mapping[str, Any] | None = None,
        exit_code: Optional[int] = None,
    ) -> None:
        payload: Dict[str, Any] = {}
        if best_cost is not None:
            payload["best_cost"] = float(best_cost)
        if summary:
            payload["summary"] = dict(summary)
        if exit_code is not None:
            payload["exit_code"] = exit_code
        self._emit("run_completed", payload)

    def run_failed(self, *, reason: str, exit_code: Optional[int] = None) -> None:
        payload: Dict[str, Any] = {"reason": reason}
        if exit_code is not None:
            payload["exit_code"] = exit_code
        self._emit("run_failed", payload)

    def emit_meta(self, meta: Mapping[str, Any]) -> None:
        self._emit("meta", dict(meta))

    def _emit(self, event: str, payload: Dict[str, Any]) -> None:
        try:
            self._emitter.emit(self.run_id, event, payload)
        except Exception:
            if isinstance(self._emitter, NullEventEmitter):
                return
            raise


__all__ = ["OptimizationMonitorClient", "MonitorConfig", "generate_run_id"]
