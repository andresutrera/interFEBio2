from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List


@dataclass
class IterationRecord:
    index: int
    cost: float
    theta: Dict[str, float]
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = float(self.timestamp)
        return payload


@dataclass
class OptimizationRun:
    run_id: str
    label: str
    status: str = "created"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    parameters: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)
    iterations: List[IterationRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
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
        iterations = [
            IterationRecord(
                index=int(item.get("index", 0)),
                cost=float(item.get("cost", 0.0)),
                theta=dict(item.get("theta", {})),
                metrics=dict(item.get("metrics", {})),
                timestamp=float(item.get("timestamp", time.time())),
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
            if metrics_clean:
                metrics = metrics_clean
            else:
                metrics = {}
            record = IterationRecord(
                index=index,
                cost=cost,
                theta={k: float(v) for k, v in theta.items() if _is_number(v)},
                metrics=metrics,
                timestamp=float(payload.get("timestamp", ts)),
            )
            self.iterations.append(record)
            best = self.meta.get("best_cost")
            if best is None or cost <= best:
                self.meta["best_cost"] = cost
            self.meta["last_cost"] = cost
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
    return isinstance(value, (int, float))


__all__ = ["IterationRecord", "OptimizationRun"]
