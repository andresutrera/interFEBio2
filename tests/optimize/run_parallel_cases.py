#!/usr/bin/env python3
"""
Drive FEBio builds and launches for three parameter sets in parallel.

Usage:
    python tests/optimize/run_parallel_cases.py \
        --output ./out/runs --jobs 2 --command febio4 -i

The script demonstrates how `ParameterSpace`, `FebTemplate`, `FebBuilder`,
`MonitorFSView`, and `LocalParallelRunner` cooperate: we transform three φ
vectors into θ-space, render dedicated FEB files under the output directory,
and launch them concurrently while a lightweight monitor tails their logs.
Use `--no-monitor` to suppress the live table or `--no-quiet` to mirror FEBio
stdout in real time.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np

from interFEBio.Optimize.Parameters import ParameterSpace
from interFEBio.Optimize.feb_bindings import (
    BuildContext,
    FebBuilder,
    FebTemplate,
    MaterialParamBinding,
)
from interFEBio.Optimize.monitor import JobView, MonitorFSView, ProgressAggregator
from interFEBio.Optimize.Storage import StorageManager, TmpfsBackend
from interFEBio.Optimize.runners import LocalParallelRunner, RunHandle, RunResult
from interFEBio.Optimize.webui import MonitorWebUIServer
from interFEBio.monitoring.events import EventEmitter, create_event_emitter


@dataclass
class Case:
    idx: int
    key: str
    phi: np.ndarray
    theta: dict[str, float]
    job_dir: Path
    feb_path: Path
    stdout_log: Path
    storage_case: Optional[str]
    storage_tag: Optional[str]
    handle: RunHandle


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run simple FEBio jobs in parallel using LocalParallelRunner."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("opt_runs"),
        help="Directory where cases will be written (default: ./opt_runs).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=2,
        help="Number of parallel jobs to run (default: 2).",
    )
    parser.add_argument(
        "--command",
        nargs="+",
        default=None,
        help=(
            "Command prefix for FEBio (default: uses FEBIO_CMD env var or `febio4 -i`). "
            "Example: --command /path/to/febio -i"
        ),
    )
    parser.add_argument(
        "--quiet",
        dest="quiet",
        action="store_true",
        default=True,
        help="Suppress console output from this script (default).",
    )
    parser.add_argument(
        "--no-quiet",
        dest="quiet",
        action="store_false",
        help="Enable live stdout tee and summary printing.",
    )
    parser.add_argument(
        "--monitor",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show live progress summary (default: enabled).",
    )
    parser.add_argument(
        "--monitor-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for the progress summary (default: 1.0).",
    )
    parser.add_argument(
        "--storage-backend",
        choices=["none", "tmpfs"],
        default="none",
        help="Optional storage backend to manage outputs (default: none).",
    )
    parser.add_argument(
        "--tmpfs-base",
        type=Path,
        default=None,
        help="Base directory for TmpfsBackend when --storage-backend=tmpfs.",
    )
    parser.add_argument(
        "--webui-port",
        type=int,
        default=None,
        help="Start a simple web UI on this port while monitoring is active.",
    )
    parser.add_argument(
        "--webui-refresh",
        type=float,
        default=1.0,
        help="Refresh interval (seconds) for the web UI (default: 1.0).",
    )
    parser.add_argument(
        "--event-socket",
        type=Path,
        default=None,
        help="Unix domain socket for monitor event emission.",
    )
    return parser.parse_args(argv)


def build_cases(
    output_dir: Path,
    template_path: Path,
    parameter_space: ParameterSpace,
    phi_vectors: Iterable[np.ndarray],
) -> List[dict]:
    template = FebTemplate(
        template_path,
        bindings=[
            MaterialParamBinding(theta_name="k", tag_name="k", selector=("id", "1")),
            MaterialParamBinding(theta_name="G", tag_name="G", selector=("id", "1")),
        ],
    )
    builder = FebBuilder(template=template, subfolder="run")

    cases: List[dict] = []
    for idx, phi in enumerate(phi_vectors):
        theta_vec = parameter_space.theta_from_phi(phi)
        theta_dict = {
            name: float(val) for name, val in zip(parameter_space.names, theta_vec)
        }
        case_dir = output_dir / f"case_{idx}"
        ctx = BuildContext(iter_id=idx, case_name=f"case{idx}", fmt="%.6f")
        feb_path_str, _ = builder.build(
            theta=theta_dict,
            out_root=str(case_dir),
            out_name=f"simpleBiaxial_{idx}.feb",
            ctx=ctx,
        )
        feb_path = Path(feb_path_str).resolve()
        stdout_log = feb_path.with_suffix(".log")
        try:
            stdout_log.unlink()
        except FileNotFoundError:
            pass
        cases.append(
            {
                "idx": idx,
                "key": f"case_{idx}",
                "phi": phi.tolist(),
                "theta": theta_dict,
                "job_dir": feb_path.parent,
                "feb_path": feb_path,
                "stdout_log": stdout_log,
                "storage_case": None,
                "storage_tag": None,
            }
        )
    return cases


def build_cases_storage(
    storage: StorageManager,
    iter_id: int,
    template_path: Path,
    parameter_space: ParameterSpace,
    phi_vectors: Sequence[np.ndarray],
) -> List[dict]:
    template = FebTemplate(
        template_path,
        bindings=[
            MaterialParamBinding(theta_name="k", tag_name="k", selector=("id", "1")),
            MaterialParamBinding(theta_name="G", tag_name="G", selector=("id", "1")),
        ],
    )
    builder = FebBuilder(template=template, subfolder="")

    cases: List[dict] = []
    for idx, phi in enumerate(phi_vectors):
        theta_vec = parameter_space.theta_from_phi(phi)
        theta_dict = {
            name: float(val) for name, val in zip(parameter_space.names, theta_vec)
        }
        case_name = f"case_{idx}"
        tag = "run"
        job_dir = storage.job_dir(iter_id, case_name, tag)
        ctx = BuildContext(iter_id=iter_id, case_name=case_name, fmt="%.6f")
        feb_path_str, _ = builder.build(
            theta=theta_dict,
            out_root=str(job_dir),
            out_name=f"{case_name}.feb",
            ctx=ctx,
        )
        feb_path = Path(feb_path_str).resolve()
        stdout_log = feb_path.with_suffix(".log")
        try:
            stdout_log.unlink()
        except FileNotFoundError:
            pass
        storage.begin_job(
            iter_id,
            case_name,
            tag,
            meta={"phi": phi.tolist(), "theta": theta_dict},
        )
        cases.append(
            {
                "idx": idx,
                "key": case_name,
                "phi": phi.tolist(),
                "theta": theta_dict,
                "job_dir": feb_path.parent,
                "feb_path": feb_path,
                "stdout_log": stdout_log,
                "storage_case": case_name,
                "storage_tag": tag,
            }
        )
    return cases


def discover_command(command: Sequence[str] | None) -> List[str]:
    if command:
        return list(command)
    env_cmd = os.environ.get("FEBIO_CMD")
    if env_cmd:
        return env_cmd.split()
    return ["febio4", "-i"]


def update_status(
    view: MonitorFSView,
    aggregator: Optional[ProgressAggregator],
    emitter: EventEmitter,
    key: str,
    status: str,
    exit_code: Optional[int] = None,
    extra: Optional[dict] = None,
) -> None:
    if aggregator:
        aggregator.update_status(key, status, exit_code)
    else:
        view.set_status(key, status, exit_code)
        payload = {"status": status}
        if exit_code is not None:
            payload["exit_code"] = exit_code
        emitter.emit(key, "status", payload)
    if extra:
        view.record_log_metadata(key, extra)
        emitter.emit(key, "meta", extra)


def launch_cases(
    cases: List[dict],
    runner: LocalParallelRunner,
    view: MonitorFSView,
    aggregator: Optional[ProgressAggregator],
    emitter: EventEmitter,
) -> List[Case]:
    running: List[Case] = []
    for data in cases:
        feb_path = data["feb_path"]
        job_dir = feb_path.parent
        update_status(view, aggregator, emitter, data["key"], "queued")
        handle = runner.run(job_dir, feb_path.name)
        update_status(view, aggregator, emitter, data["key"], "running")
        case = Case(
            idx=data["idx"],
            key=data["key"],
            phi=np.asarray(data["phi"], dtype=float),
            theta=data["theta"],
            job_dir=job_dir,
            feb_path=feb_path,
            stdout_log=data["stdout_log"],
            storage_case=data.get("storage_case"),
            storage_tag=data.get("storage_tag"),
            handle=handle,
        )
        running.append(case)
    return running


def collect_results(running: List[Case]) -> List[tuple[Case, RunResult]]:
    results: List[tuple[Case, RunResult]] = []
    for case in running:
        result = case.handle.wait()
        results.append((case, result))
    return results


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    emitter = create_event_emitter(args.event_socket)

    template_path = (
        Path(__file__).resolve().parents[2] / "tests" / "optimize" / "simpleBiaxial.feb"
    )
    if not template_path.exists():
        raise FileNotFoundError(f"simpleBiaxial.feb not found at {template_path}")

    parameter_space = ParameterSpace(
        names=["k", "G"],
        theta0={"k": 10.0, "G": 10.0},
        xi=2.0,
    )
    phi_vectors = [
        np.array([0.0, 0.0]),
        np.array([0.5, -0.25]),
        np.array([-0.75, 0.25]),
    ]

    use_storage = args.storage_backend != "none"
    storage: Optional[StorageManager] = None
    iter_id = 0
    case_names = [f"case_{idx}" for idx in range(len(phi_vectors))]

    if use_storage:
        if args.storage_backend == "tmpfs":
            backend = TmpfsBackend(base=args.tmpfs_base)
        else:
            raise ValueError(f"Unsupported storage backend: {args.storage_backend}")
        project_name = args.output.name or "run_parallel"
        storage = StorageManager(backend=backend, project=project_name)
        zero_phi = np.zeros(len(parameter_space.names))
        theta0 = parameter_space.theta_from_phi(zero_phi)
        storage.begin_iter(
            iter_id,
            phi={name: 0.0 for name in parameter_space.names},
            theta={
                name: float(val) for name, val in zip(parameter_space.names, theta0)
            },
            cases=case_names,
        )
        cases = build_cases_storage(
            storage,
            iter_id,
            template_path,
            parameter_space,
            phi_vectors,
        )
    else:
        cases = build_cases(
            output_dir,
            template_path,
            parameter_space,
            phi_vectors,
        )

    view = MonitorFSView()
    job_views = [
        JobView(
            key=case["key"],
            name=f"case{case['idx']}",
            job_dir=case["job_dir"],
            log_path=case["stdout_log"],
        )
        for case in cases
    ]
    view.register_many(job_views)

    summary_path = output_dir / "case_summary.json"
    summary_payload = [
        {
            "idx": case["idx"],
            "key": case["key"],
            "phi": case["phi"],
            "theta": case["theta"],
            "job_dir": str(case["job_dir"]),
            "feb_path": str(case["feb_path"]),
            "stdout_log": str(case["stdout_log"]),
            "default_log": str(Path(case["feb_path"]).with_suffix(".log")),
        }
        for case in cases
    ]
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    command = discover_command(args.command)
    if not args.quiet:
        print(f"[runner] Using command: {' '.join(command)}")
        print(f"[runner] Writing cases under: {output_dir}")

    tee_stream = None if args.quiet else sys.stdout
    runner = LocalParallelRunner(
        n_jobs=args.jobs,
        command=command,
        tee_stream=tee_stream,
    )
    results: List[tuple[Case, RunResult]] = []
    aggregator: Optional[ProgressAggregator] = None
    web_server: Optional[MonitorWebUIServer] = None
    try:
        if args.monitor:
            aggregator = ProgressAggregator(
                view,
                poll_interval=args.monitor_interval,
                stream=(None if args.quiet else sys.stdout),
                event_emitter=emitter,
            )
            aggregator.register_jobs(job.key for job in job_views)
            aggregator.start()
            if args.webui_port is not None:
                web_server = MonitorWebUIServer(
                    view,
                    port=args.webui_port,
                    refresh_interval=args.webui_refresh,
                )
                web_server.start()
                if not args.quiet:
                    print(
                        f"[webui] Serving at http://127.0.0.1:{args.webui_port}",
                        flush=True,
                    )
        elif args.webui_port is not None:
            web_server = MonitorWebUIServer(
                view,
                port=args.webui_port,
                refresh_interval=args.webui_refresh,
            )
            web_server.start()
            if not args.quiet:
                print(
                    f"[webui] Serving at http://127.0.0.1:{args.webui_port}",
                    flush=True,
                )

        running = launch_cases(cases, runner, view, aggregator, emitter)
        results = collect_results(running)

        for case, result in results:
            final_status = "finished" if result.exit_code == 0 else "failed"
            if storage and case.storage_case and case.storage_tag:
                storage.register_artifact(
                    iter_id,
                    case.storage_case,
                    case.storage_tag,
                    case.feb_path,
                    kind="feb",
                )
                if case.stdout_log.exists():
                    storage.register_artifact(
                        iter_id,
                        case.storage_case,
                        case.storage_tag,
                        case.stdout_log,
                        kind="log",
                    )
                storage.end_job(
                    iter_id,
                    case.storage_case,
                    case.storage_tag,
                    status="done" if result.exit_code == 0 else "failed",
                )
            update_status(
                view,
                aggregator,
                emitter,
                case.key,
                final_status,
                result.exit_code,
                extra={"ended_at": result.ended_at},
            )
    finally:
        if web_server:
            web_server.stop()
        if aggregator:
            aggregator.stop(force_render=True)
        runner.shutdown()

    if storage is not None:
        storage.end_iter(iter_id, summary={"cases": len(results)}, is_best=False)

    if not args.quiet:
        for case, result in results:
            status = "OK" if result.exit_code == 0 else f"ERR({result.exit_code})"
            print(
                f"[result] case={case.idx} status={status} "
                f"log={result.log_path} duration={result.duration:.2f}s"
            )

    failures = [r for _, r in results if r.exit_code != 0]
    if failures:
        if not args.quiet:
            print(f"[runner] {len(failures)} job(s) failed.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
