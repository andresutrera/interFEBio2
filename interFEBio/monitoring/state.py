from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional


@dataclass
class ArtifactInfo:
    kind: str
    path: Path
    size: int


@dataclass
class JobInfo:
    project: str
    iter_id: int
    case: str
    tag: str
    status: str
    started_at: float
    ended_at: float
    placement_root: Path
    artifacts: List[ArtifactInfo] = field(default_factory=list)
    summary: Dict[str, float] = field(default_factory=dict)
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def job_id(self) -> str:
        return f"{self.project}/iter{self.iter_id}/{self.case}/{self.tag}"


class StorageInventory:
    """
    Scans storage roots produced by :class:`StorageManager` and builds a view of jobs.
    """

    def __init__(self, roots: Iterable[Path], *, poll_interval: float = 5.0):
        self.roots = [Path(r).resolve() for r in roots]
        self.poll_interval = max(1.0, float(poll_interval))
        self._lock = threading.Lock()
        self._jobs: Dict[str, JobInfo] = {}
        self._last_scan = 0.0

    def refresh(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_scan < self.poll_interval:
            return
        jobs: Dict[str, JobInfo] = {}
        for root in self.roots:
            project = root.name
            iters_dir = root / "iters"
            if not iters_dir.exists():
                continue
            for iter_dir in iters_dir.glob("iter*"):
                if not iter_dir.is_dir():
                    continue
                try:
                    iter_id = int(iter_dir.name.replace("iter", ""))
                except ValueError:
                    continue
                iter_manifest = iter_dir / "iter_manifest.json"
                summary: Dict[str, float] = {}
                if iter_manifest.exists():
                    try:
                        summary = json.loads(iter_manifest.read_text(encoding="utf-8")).get(
                            "summary", {}
                        )
                    except Exception:
                        summary = {}
                for case_dir in iter_dir.glob("*"):
                    if not case_dir.is_dir():
                        continue
                    case_name = case_dir.name
                    for tag_dir in case_dir.glob("*"):
                        if not tag_dir.is_dir():
                            continue
                        tag_name = tag_dir.name
                        manifest_path = tag_dir / "manifest.json"
                        if not manifest_path.exists():
                            continue
                        try:
                            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        except Exception:
                            continue
                        placement_root = Path(
                            manifest.get("meta", {}).get("placement_root", tag_dir)
                        )
                        job = JobInfo(
                            project=project,
                            iter_id=iter_id,
                            case=case_name,
                            tag=tag_name,
                            status=manifest.get("status", "unknown"),
                            started_at=float(manifest.get("started_at", 0.0)),
                            ended_at=float(manifest.get("ended_at", 0.0)),
                            placement_root=placement_root,
                            summary=summary,
                        )
                        artifacts = []
                        for art in manifest.get("artifacts", []):
                            artifacts.append(
                                ArtifactInfo(
                                    kind=art.get("kind", "aux"),
                                    path=placement_root / art.get("relpath", ""),
                                    size=int(art.get("bytes", 0)),
                                )
                            )
                        job.artifacts = artifacts
                        jobs[job.job_id] = job
        with self._lock:
            self._jobs = jobs
            self._last_scan = now

    def list_jobs(self) -> List[JobInfo]:
        with self._lock:
            return list(self._jobs.values())

    def get_job(self, job_id: str) -> Optional[JobInfo]:
        with self._lock:
            return self._jobs.get(job_id)

    # --- event ingestion ---
    def apply_event(
        self, job_id: str, event: str, payload: Dict[str, object], ts: float
    ) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                job = self._create_job_from_id(job_id)
                self._jobs[job_id] = job
            if event == "status":
                status = payload.get("status")
                if isinstance(status, str):
                    job.status = status
                    if status == "running" and job.started_at <= 0:
                        job.started_at = ts
                    if status in {"finished", "failed"}:
                        job.ended_at = ts
                exit_code = payload.get("exit_code")
                if isinstance(exit_code, (int, float)):
                    job.meta["exit_code"] = exit_code
            elif event == "meta":
                job.meta.update(payload)
                for key, value in payload.items():
                    if isinstance(value, (int, float)):
                        job.summary[key] = float(value)
            else:
                events = job.meta.get("events")
                if not isinstance(events, list):
                    events = []
                    job.meta["events"] = events
                events.append({"event": event, "data": payload})
            self._last_scan = ts

    def _create_job_from_id(self, job_id: str) -> JobInfo:
        parts = job_id.split("/")
        project = parts[0] if parts else "default"
        iter_part = parts[1] if len(parts) > 1 else "iter0"
        case = parts[2] if len(parts) > 2 else "case"
        tag = parts[3] if len(parts) > 3 else "run"
        try:
            iter_id = int(iter_part.replace("iter", ""))
        except ValueError:
            iter_id = 0
        return JobInfo(
            project=project,
            iter_id=iter_id,
            case=case,
            tag=tag,
            status="pending",
            started_at=0.0,
            ended_at=0.0,
            placement_root=Path("."),
        )
