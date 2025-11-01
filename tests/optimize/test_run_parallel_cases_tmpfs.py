import json
import time
from concurrent.futures import Future
from pathlib import Path

from interFEBio.Optimize.runners import RunHandle, RunResult


def test_run_parallel_cases_with_tmpfs_backend(monkeypatch, tmp_path: Path) -> None:
    from tests.optimize import run_parallel_cases as rpc

    log_records = []

    class FakeRunner:
        def __init__(self, *args, **kwargs):
            pass

        def run(self, job_dir: str | Path, feb_name: str | Path) -> RunHandle:
            job_dir = Path(job_dir)
            feb_name = Path(feb_name)
            feb_path = job_dir / feb_name
            assert feb_path.exists()
            log_path = feb_path.with_suffix(".log")
            log_path.write_text(
                "CONTROL DATA\n"
                "\ttime_steps ......................................... : 5\n"
                "\tstep_size .......................................... : 0.1\n"
                "Step 1\n"
                "time = 0.10\n"
                "Step 5\n"
                "time = 0.50\n",
                encoding="utf-8",
            )
            now = time.time()
            result = RunResult(
                exit_code=0,
                started_at=now,
                ended_at=now + 0.01,
                log_path=log_path,
                metadata={"cmd": ["fake"]},
            )
            future = Future()
            future.set_result(result)
            handle = RunHandle(future)
            log_records.append(log_path)
            return handle

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(rpc, "LocalParallelRunner", FakeRunner)

    class CollectingEmitter:
        def __init__(self) -> None:
            self.events: list[tuple[str, str, dict]] = []

        def emit(self, job_id: str, event: str, payload: dict) -> None:
            self.events.append((job_id, event, payload))

    collector = CollectingEmitter()
    monkeypatch.setattr(rpc, "create_event_emitter", lambda path: collector)

    output_dir = tmp_path / "summary"
    args = [
        "--jobs",
        "2",
        "--output",
        str(output_dir),
        "--storage-backend",
        "tmpfs",
        "--command",
        "fake",
        "--no-monitor",
        "--event-socket",
        str(tmp_path / "sock"),
    ]

    rc = rpc.main(args)
    assert rc == 0
    assert len(log_records) == 3

    summary_path = output_dir / "case_summary.json"
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert len(data) == 3
    for entry in data:
        log_path = Path(entry["stdout_log"])
        assert log_path.exists()
        text = log_path.read_text(encoding="utf-8")
        assert "Step 5" in text

    assert collector.events, "expected events to be emitted"
    statuses = [payload["status"] for _, event, payload in collector.events if event == "status"]
    assert any(status == "queued" for status in statuses)
    assert any(status == "finished" for status in statuses)
