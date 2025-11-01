import io
from pathlib import Path

from interFEBio.Optimize.runners import (
    LocalParallelRunner,
    LocalSerialRunner,
    RunHandle,
    SlurmRunner,
)


def _make_fake_febio(tmp_path: Path) -> Path:
    script = tmp_path / "fake_febio.sh"
    script.write_text(
        "#!/bin/bash\n"
        "echo \"Starting $1\"\n"
        "sleep 0.05\n"
        "echo \"Finished $1\"\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def _create_job(tmp_path: Path, name: str) -> tuple[Path, Path]:
    job_dir = tmp_path / name
    job_dir.mkdir()
    feb_path = job_dir / f"{name}.feb"
    feb_path.write_text("<febio_spec/>", encoding="utf-8")
    return job_dir, feb_path


def test_local_serial_runner_logs_output(tmp_path: Path):
    fake_cmd = _make_fake_febio(tmp_path)
    capture = io.StringIO()
    runner = LocalSerialRunner(command=[str(fake_cmd)], tee_stream=capture)

    job_dir, feb_path = _create_job(tmp_path, "job_serial")

    handle = runner.run(job_dir, feb_path)
    assert isinstance(handle, RunHandle)
    result = handle.wait(timeout=5)

    assert result.exit_code == 0
    assert result.log_path == Path(job_dir) / "job_serial.log"
    # Runner no longer tees to stdout by default.
    runner.shutdown()


def test_local_parallel_runner_runs_multiple_jobs(tmp_path: Path):
    fake_cmd = _make_fake_febio(tmp_path)
    runner = LocalParallelRunner(
        n_jobs=2, command=[str(fake_cmd)], tee_stream=io.StringIO()
    )

    handles = []
    for idx in range(3):
        job_dir, feb_path = _create_job(tmp_path, f"job_parallel_{idx}")
        handles.append(runner.run(job_dir, feb_path))

    results = [h.wait(timeout=5) for h in handles]
    assert all(r.exit_code == 0 for r in results)
    for idx, res in enumerate(results):
        assert res.log_path == Path(tmp_path) / f"job_parallel_{idx}" / f"job_parallel_{idx}.log"
    runner.shutdown()


def test_slurm_runner_parse_scontrol_output():
    payload = (
        "JobId=1234 JobName=test JobState=COMPLETED Reason=None "
        "ExitCode=0:0 Time=00:01:00"
    )
    state, exit_code = SlurmRunner._parse_scontrol(payload)
    assert state == "COMPLETED"
    assert exit_code == 0

    payload_failed = (
        "JobId=2 JobState=FAILED Reason=NonZeroExitCode ExitCode=2:0"
    )
    state, exit_code = SlurmRunner._parse_scontrol(payload_failed)
    assert state == "FAILED"
    assert exit_code == 2
