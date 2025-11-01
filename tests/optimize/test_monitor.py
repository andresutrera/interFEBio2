import io
import time
from pathlib import Path

from interFEBio.Optimize.monitor import (
    JobView,
    LogTailer,
    MonitorFSView,
    ProgressAggregator,
)


def test_log_tailer_incremental(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    tailer = LogTailer(log_path)

    # No file yet.
    assert tailer.poll() == []

    log_path.write_text("line1\nline2\n", encoding="utf-8")
    assert tailer.poll() == ["line1", "line2"]

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("line3\npartial")
    assert tailer.poll() == ["line3"]

    with log_path.open("a", encoding="utf-8") as fh:
        fh.write("4\n")
    assert tailer.poll() == ["partial4"]


def test_progress_aggregator_tracks_status(tmp_path: Path) -> None:
    log_path = tmp_path / "job.log"
    view = MonitorFSView()
    job = JobView(
        key="case_0",
        name="case_0",
        job_dir=tmp_path,
        log_path=log_path,
    )
    view.register(job)

    stream = io.StringIO()
    agg = ProgressAggregator(view, poll_interval=0.05, stream=stream)
    agg.register_jobs(["case_0"])
    agg.start()

    agg.update_status("case_0", "running")
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write("simulation step 1\n")

    time.sleep(0.15)

    agg.update_status("case_0", "finished", exit_code=0)
    agg.stop()

    output = stream.getvalue()
    assert "case_0" in output
    assert "running" in output
    assert "finished" in output
    assert "step 1" in output
