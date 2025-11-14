from __future__ import annotations

from typing import List, Tuple

from interFEBio.monitoring.client import MonitorConfig, OptimizationMonitorClient
from interFEBio.monitoring.events import EventEmitter


class Collector(EventEmitter):
    def __init__(self) -> None:
        self.records: List[Tuple[str, str, dict]] = []

    def emit(self, job_id: str, event: str, payload: dict) -> None:
        self.records.append((job_id, event, payload))


def test_monitor_client_emits_expected_payloads(tmp_path):
    client = OptimizationMonitorClient(
        MonitorConfig(socket_path=tmp_path / "sock", label="My Run")
    )
    collector = Collector()
    client._emitter = collector  # type: ignore[attr-defined]

    client.run_started(
        parameters={"names": ["a"], "phi0": [0.0]},
        cases=[{"subfolder": "case", "experiments": ["exp"]}],
        optimizer={"adapter": "Dummy"},
    )
    client.record_iteration(
        index=0,
        cost=2.5,
        theta={"a": 1.5},
        metrics={"nrmse": 0.2},
    )
    client.run_completed(summary={"nrmse": 0.2})

    assert len(collector.records) == 3
    run_id = collector.records[0][0]
    assert run_id.startswith("MyRun-")
    assert collector.records[0][1] == "run_started"
    assert collector.records[1][1] == "iteration"
    assert collector.records[1][2]["theta"]["a"] == 1.5
    assert collector.records[2][1] == "run_completed"
    assert collector.records[2][2]["summary"]["nrmse"] == 0.2
