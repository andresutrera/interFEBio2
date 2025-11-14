from __future__ import annotations

import time

import pytest

from interFEBio.monitoring.registry import ActiveRunDeletionError, RunRegistry


def test_run_registry_persists_events(tmp_path):
    db_path = tmp_path / "runs.jsonl"
    registry = RunRegistry(db_path)
    ts = time.time()
    registry.apply_event(
        "demo-run",
        "run_started",
        {
            "label": "Demo Run",
            "parameters": {"names": ["k"], "phi0": [0.0]},
            "optimizer": {"adapter": "Test"},
        },
        ts,
    )
    registry.apply_event(
        "demo-run",
        "iteration",
        {
            "index": 0,
            "cost": 1.23,
            "theta": {"k": 3.14},
            "metrics": {"nrmse": 0.1, "r_squared": {"case/exp": 0.95}},
        },
        ts + 1,
    )
    registry.apply_event(
        "demo-run",
        "run_completed",
        {"summary": {"nrmse": 0.1}},
        ts + 2,
    )

    registry2 = RunRegistry(db_path)
    run = registry2.get_run("demo-run")
    assert run is not None
    assert run.label == "Demo Run"
    assert run.status == "finished"
    assert run.meta["last_cost"] == 1.23
    assert len(run.iterations) == 1
    iteration = run.iterations[0]
    assert iteration.index == 0
    assert iteration.cost == 1.23
    assert iteration.theta["k"] == 3.14
    assert iteration.metrics["nrmse"] == 0.1
    assert iteration.metrics["r_squared"]["case/exp"] == 0.95


def test_run_registry_delete_and_clear(tmp_path):
    db_path = tmp_path / "runs.jsonl"
    registry = RunRegistry(db_path)
    registry.apply_event("run-a", "run_started", {}, time.time())
    registry.apply_event("run-a", "run_completed", {}, time.time())
    registry.apply_event("run-b", "run_started", {}, time.time())
    registry.apply_event("run-b", "run_failed", {}, time.time())
    assert len(registry.list_runs()) == 2
    assert registry.delete_run("run-a") is True
    assert registry.get_run("run-a") is None
    assert len(registry.list_runs()) == 1
    protected = registry.clear()
    assert protected == []
    assert registry.list_runs() == []


def test_run_registry_blocks_active_deletion(tmp_path):
    db_path = tmp_path / "runs.jsonl"
    registry = RunRegistry(db_path)
    registry.apply_event("run-active", "run_started", {}, time.time())
    with pytest.raises(ActiveRunDeletionError):
        registry.delete_run("run-active")
    protected = registry.clear()
    assert protected == ["run-active"]
    assert registry.get_run("run-active") is not None
    assert registry.delete_run("run-active", force=True) is True
    assert registry.get_run("run-active") is None


def test_run_registry_clear_force(tmp_path):
    db_path = tmp_path / "runs.jsonl"
    registry = RunRegistry(db_path)
    registry.apply_event("run-active", "run_started", {}, time.time())
    registry.apply_event("run-finished", "run_started", {}, time.time())
    registry.apply_event("run-finished", "run_completed", {}, time.time())
    protected = registry.clear()
    assert protected == ["run-active"]
    registry.apply_event("run-active-2", "run_started", {}, time.time())
    remaining = {run.run_id for run in registry.list_runs()}
    assert remaining == {"run-active", "run-active-2"}
    protected = registry.clear(force=True)
    assert protected == []
    assert registry.list_runs() == []
