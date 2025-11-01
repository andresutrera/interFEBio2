from pathlib import Path

from interFEBio.Optimize.runners import LocalParallelRunner, LocalSerialRunner, RunHandle


def _make_fake_febio(tmp_path: Path) -> Path:
    script = tmp_path / "fake_febio.sh"
    script.write_text(
        "#!/bin/bash\n"
        "echo \"Starting $1\"\n"
        "sleep 0.01\n"
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


def test_local_serial_runner_writes_log(tmp_path: Path) -> None:
    fake_cmd = _make_fake_febio(tmp_path)
    runner = LocalSerialRunner(command=[str(fake_cmd)])

    job_dir, feb_path = _create_job(tmp_path, "serial_case")
    handle = runner.run(job_dir, feb_path)
    assert isinstance(handle, RunHandle)

    result = handle.wait(timeout=5)
    assert result.exit_code == 0
    assert result.log_path == job_dir / "serial_case.log"
    assert result.log_path.read_text(encoding="utf-8")

    runner.shutdown()


def test_local_parallel_runner_handles_multiple_jobs(tmp_path: Path) -> None:
    fake_cmd = _make_fake_febio(tmp_path)
    runner = LocalParallelRunner(n_jobs=2, command=[str(fake_cmd)])

    handles = []
    for idx in range(3):
        job_dir, feb_path = _create_job(tmp_path, f"parallel_case_{idx}")
        handles.append(runner.run(job_dir, feb_path))

    results = [handle.wait(timeout=5) for handle in handles]
    assert all(res.exit_code == 0 for res in results)
    for idx, res in enumerate(results):
        expected = tmp_path / f"parallel_case_{idx}" / f"parallel_case_{idx}.log"
        assert res.log_path == expected
        assert expected.read_text(encoding="utf-8")

    runner.shutdown()

