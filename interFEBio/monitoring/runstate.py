"""Represent optimization runs and iteration histories for the monitor."""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class IterationRecord:
    """Record metrics for a single optimizer iteration."""
    index: int
    cost: float
    theta: Dict[str, float]
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    series: Optional[Dict[str, Dict[str, List[float]]]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the iteration record into JSON-friendly form."""
        payload = asdict(self)
        payload["timestamp"] = float(self.timestamp)
        return payload


@dataclass
class OptimizationRun:
    """Track the lifecycle and metadata of an optimization run."""
    run_id: str
    label: str
    status: str = "created"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parameters: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    iterations: List[IterationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the run along with its metadata."""
        return {
            "run_id": self.run_id,
            "label": self.label,
            "status": self.status,
            "created_at": float(self.created_at),
            "updated_at": float(self.updated_at),
            "parameters": self.parameters,
            "meta": self.meta,
            "iterations": [it.to_dict() for it in self.iterations],
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OptimizationRun":
        """Rebuild a run snapshot from the persisted dictionary."""
        iterations = [
            IterationRecord(
                index=int(item.get("index", 0)),
                cost=float(item.get("cost", 0.0)),
                theta=dict(item.get("theta", {})),
                metrics=dict(item.get("metrics", {})),
                timestamp=float(item.get("timestamp", time.time())),
                series=_sanitize_series_payload(item.get("series")),
            )
            for item in payload.get("iterations", [])
        ]
        return cls(
            run_id=str(payload["run_id"]),
            label=str(payload.get("label", payload["run_id"])),
            status=str(payload.get("status", "created")),
            created_at=float(payload.get("created_at", time.time())),
            updated_at=float(payload.get("updated_at", time.time())),
            parameters=dict(payload.get("parameters", {})),
            meta=dict(payload.get("meta", {})),
            iterations=iterations,
        )

    def apply_event(self, event: str, payload: Dict[str, Any], ts: float) -> None:
        """Update the run metadata based on the incoming event."""
        self.updated_at = float(ts)
        if event == "run_started":
            self.status = "running"
            self.parameters.update(payload.get("parameters", {}))
            label = payload.get("label")
            if isinstance(label, str) and label:
                self.label = label
            cases = payload.get("cases")
            if cases:
                self.meta.setdefault("cases", cases)
            optimizer = payload.get("optimizer")
            if optimizer:
                self.meta["optimizer"] = optimizer
            self.meta.update(payload.get("meta", {}))
        elif event == "iteration":
            try:
                cost = float(payload["cost"])
            except (KeyError, TypeError, ValueError):
                return
            index = int(payload.get("index", len(self.iterations)))
            theta = payload.get("theta", {})
            metrics_payload = payload.get("metrics", {}) or {}
            metrics_clean: Dict[str, Any] = {}
            for key, value in metrics_payload.items():
                if _is_number(value) or isinstance(value, str):
                    metrics_clean[key] = value
                elif isinstance(value, dict):
                    metrics_clean[key] = {
                        str(subkey): (
                            float(subval)
                            if _is_number(subval)
                            else (None if subval is None else str(subval))
                        )
                        for subkey, subval in value.items()
                    }
                elif isinstance(value, (list, tuple)):
                    try:
                        metrics_clean[key] = [float(v) for v in value]
                    except Exception:
                        metrics_clean[key] = [str(v) for v in value]
            if metrics_clean:
                metrics = metrics_clean
            else:
                metrics = {}
            series_clean = _sanitize_series_payload(payload.get("series"))
            record = IterationRecord(
                index=index,
                cost=cost,
                theta={k: float(v) for k, v in theta.items() if _is_number(v)},
                metrics=metrics,
                timestamp=float(payload.get("timestamp", ts)),
                series=series_clean or None,
            )
            self.iterations.append(record)
            best = self.meta.get("best_cost")
            if best is None or cost <= best:
                self.meta["best_cost"] = cost
            self.meta["last_cost"] = cost
            r_sq = metrics.get("r_squared")
            if isinstance(r_sq, dict):
                dest = self.meta.setdefault("r_squared", {})
                if isinstance(dest, dict):
                    for name, value in r_sq.items():
                        dest[str(name)] = value
        elif event == "run_completed":
            self.status = "finished"
            summary = payload.get("summary")
            if isinstance(summary, dict):
                self.meta["summary"] = summary
            best_cost = payload.get("best_cost")
            if best_cost is not None and _is_number(best_cost):
                self.meta["best_cost"] = float(best_cost)
            exit_code = payload.get("exit_code")
            if exit_code is not None:
                self.meta["exit_code"] = exit_code
        elif event == "run_failed":
            self.status = "failed"
            reason = payload.get("reason")
            if reason:
                self.meta["failure_reason"] = str(reason)
            exit_code = payload.get("exit_code")
            if exit_code is not None:
                self.meta["exit_code"] = exit_code
        elif event == "meta":
            self.meta.update(payload)
        else:
            events = self.meta.setdefault("events", [])
            events.append({"event": event, "payload": payload, "timestamp": ts})


def _is_number(value: Any) -> bool:
    """Quick check for primitive numeric values."""
    return isinstance(value, (int, float))


def _sanitize_series_payload(raw: Any) -> Dict[str, Dict[str, List[float]]]:
    """Normalize any series payload into numeric sequences."""
    cleaned: Dict[str, Dict[str, List[float]]] = {}
    if not isinstance(raw, dict):
        return cleaned
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        entry: Dict[str, List[float]] = {}
        for field in ("x", "y_exp", "y_sim"):
            arr = value.get(field)
            if arr is None:
                continue
            try:
                entry[field] = [float(v) for v in arr]
            except Exception:
                continue
        if entry:
            cleaned[str(key)] = entry
    return cleaned


__all__ = ["IterationRecord", "OptimizationRun"]
