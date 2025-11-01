# interFEBio/Optimize/storage.py
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Tuple, Literal, Any
from pathlib import Path
import os, io, json, time, shutil, hashlib, tempfile

ArtifactKind = Literal["feb", "xplt", "log", "aux", "metrics"]
JobStatus = Literal["queued", "running", "done", "failed", "skipped", "cached"]

# ---------------- backends ----------------


class Backend:
    def root(self) -> Path:
        raise NotImplementedError

    def ensure(self, rel: Path) -> Path:
        raise NotImplementedError

    def exists(self, rel: Path) -> bool:
        raise NotImplementedError

    def rm_tree(self, rel: Path) -> None:
        raise NotImplementedError

    def usable_bytes(self) -> int:
        raise NotImplementedError


class TmpfsBackend(Backend):
    def __init__(self, base: Path | None = None):
        # Prefer a dedicated mount; else /dev/shm; else /tmp
        base = base or Path("/mnt/interfebio_ram")
        if not base.exists():
            base = Path("/dev/shm/interFEBio")
        if not base.exists():
            base = Path("/tmp/interFEBio_ram")
        self._base = base
        self._base.mkdir(parents=True, exist_ok=True)

    def root(self) -> Path:
        return self._base

    def ensure(self, rel: Path) -> Path:
        p = self._base / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, rel: Path) -> bool:
        return (self._base / rel).exists()

    def rm_tree(self, rel: Path) -> None:
        shutil.rmtree(self._base / rel, ignore_errors=True)

    def usable_bytes(self) -> int:
        try:
            st = os.statvfs(self._base)
            return int(st.f_bavail * st.f_frsize)
        except Exception:
            return 0


class DiskBackend(Backend):
    def __init__(self, base: Path):
        self._base = base
        self._base.mkdir(parents=True, exist_ok=True)

    def root(self) -> Path:
        return self._base

    def ensure(self, rel: Path) -> Path:
        p = self._base / rel
        p.mkdir(parents=True, exist_ok=True)
        return p

    def exists(self, rel: Path) -> bool:
        return (self._base / rel).exists()

    def rm_tree(self, rel: Path) -> None:
        shutil.rmtree(self._base / rel, ignore_errors=True)

    def usable_bytes(self) -> int:
        try:
            du = shutil.disk_usage(self._base)
            return int(du.free)
        except Exception:
            return 0


class HybridBackend(Backend):
    """
    RAM first. If estimate exceeds quota or RAM free is low, place on disk.
    """

    def __init__(
        self,
        ram: TmpfsBackend,
        disk: DiskBackend,
        quota_bytes: int = 1 << 29,
        min_free_ram: int = 1 << 28,
    ):
        self.ram = ram
        self.disk = disk
        self.quota = int(quota_bytes)
        self.min_free_ram = int(min_free_ram)
        self._base = Path("/")  # unused

    def root(self) -> Path:  # not used directly
        return Path("/")

    def choose_base(self, estimate_bytes: int | None) -> Backend:
        if estimate_bytes is None:
            # prefer RAM if there is headroom
            return (
                self.ram if self.ram.usable_bytes() > self.min_free_ram else self.disk
            )
        if estimate_bytes > self.quota:
            return self.disk
        if self.ram.usable_bytes() - estimate_bytes < self.min_free_ram:
            return self.disk
        return self.ram

    # For the interface, rel is full project-relative path.
    def ensure(self, rel: Path, estimate_bytes: int | None = None) -> Path:
        base = self.choose_base(estimate_bytes)
        return base.ensure(rel)

    def exists(self, rel: Path) -> bool:
        return self.ram.exists(rel) or self.disk.exists(rel)

    def rm_tree(self, rel: Path) -> None:
        if self.ram.exists(rel):
            self.ram.rm_tree(rel)
        if self.disk.exists(rel):
            self.disk.rm_tree(rel)

    def usable_bytes(self) -> int:
        # report combined free
        return self.ram.usable_bytes() + self.disk.usable_bytes()


# ---------------- policies ----------------


@dataclass
class RetentionPolicy:
    keep_last: int = 5
    keep_best: bool = True
    max_bytes: Optional[int] = None
    max_age_days: Optional[int] = None


# ---------------- manifests ----------------


@dataclass
class Artifact:
    kind: ArtifactKind
    relpath: str
    bytes: int


@dataclass
class JobManifest:
    case: str
    tag: str
    status: JobStatus = "queued"
    started_at: float = 0.0
    ended_at: float = 0.0
    artifacts: List[Artifact] = field(default_factory=list)
    meta: Dict[str, Any] = field(
        default_factory=dict
    )  # cmd, env, hashes, placement_root

    def to_json(self) -> Dict[str, Any]:
        d = asdict(self)
        d["artifacts"] = [asdict(a) for a in self.artifacts]
        return d

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "JobManifest":
        jm = JobManifest(
            case=d["case"],
            tag=d["tag"],
            status=d.get("status", "queued"),
            started_at=d.get("started_at", 0.0),
            ended_at=d.get("ended_at", 0.0),
            meta=d.get("meta", {}),
        )
        for a in d.get("artifacts", []):
            jm.artifacts.append(
                Artifact(kind=a["kind"], relpath=a["relpath"], bytes=a["bytes"])
            )
        return jm


@dataclass
class IterManifest:
    iter_id: int
    created_at: float
    phi: Dict[str, float]
    theta: Dict[str, float]
    cases: List[str]
    jobs: Dict[str, JobManifest] = field(default_factory=dict)  # key: case/tag
    is_best: bool = False
    summary: Dict[str, float] = field(default_factory=dict)  # e.g., rmse, rnorm, wall

    def to_json(self) -> Dict[str, Any]:
        return {
            "iter_id": self.iter_id,
            "created_at": self.created_at,
            "phi": self.phi,
            "theta": self.theta,
            "cases": self.cases,
            "jobs": {k: v.to_json() for k, v in self.jobs.items()},
            "is_best": self.is_best,
            "summary": self.summary,
        }

    @staticmethod
    def from_json(d: Dict[str, Any]) -> "IterManifest":
        im = IterManifest(
            iter_id=d["iter_id"],
            created_at=d.get("created_at", time.time()),
            phi=d.get("phi", {}),
            theta=d.get("theta", {}),
            cases=d.get("cases", []),
            is_best=d.get("is_best", False),
            summary=d.get("summary", {}),
        )
        for k, v in d.get("jobs", {}).items():
            im.jobs[k] = JobManifest.from_json(v)
        return im


# ---------------- storage manager ----------------


class StorageManager:
    """
    Project-aware paths, hybrid placement, atomic writes, monitoring aides.
    """

    def __init__(
        self,
        backend: Backend,
        project: str = "interFEBio",
        retention: RetentionPolicy | None = None,
    ):
        self.backend = backend
        self.project = project
        self.retention = retention or RetentionPolicy()
        self._job_roots: Dict[
            Tuple[int, str, str], Path
        ] = {}  # (iter, case, tag) -> abs path

    # ----- paths -----
    def _rel_iter(self, it: int) -> Path:
        return Path(self.project) / "iters" / f"iter{it}"

    def _rel_job(self, it: int, case: str, tag: str) -> Path:
        return self._rel_iter(it) / case / tag

    def iter_dir(self, it: int, estimate_bytes: int | None = None) -> Path:
        return self._ensure(self._rel_iter(it), estimate_bytes)

    def job_dir(
        self, it: int, case: str, tag: str, estimate_bytes: int | None = None
    ) -> Path:
        p = self._ensure(self._rel_job(it, case, tag), estimate_bytes)
        self._job_roots[(it, case, tag)] = p
        return p

    def artifact_path(self, it: int, case: str, tag: str, filename: str) -> Path:
        root = self._job_root(it, case, tag)
        return root / filename

    def stdout_path(self, it: int, case: str, tag: str) -> Path:
        return self.artifact_path(it, case, tag, "stdout.log")

    def progress_path(self, it: int, case: str, tag: str) -> Path:
        return self.artifact_path(it, case, tag, "progress.json")

    def events_path(self, it: int, case: str, tag: str) -> Path:
        return self.artifact_path(it, case, tag, "events.jsonl")

    def xplt_probe_path(self, it: int, case: str, tag: str) -> Path:
        return self.artifact_path(it, case, tag, "xplt_probe.json")

    def job_manifest_path(self, it: int, case: str, tag: str) -> Path:
        return self.artifact_path(it, case, tag, "manifest.json")

    def iter_manifest_path(self, it: int) -> Path:
        return self.iter_dir(it) / "iter_manifest.json"

    # ----- lifecycle -----
    def begin_iter(
        self, it: int, phi: Dict[str, float], theta: Dict[str, float], cases: List[str]
    ) -> None:
        self.iter_dir(it)
        m = IterManifest(
            iter_id=it,
            created_at=time.time(),
            phi=dict(phi),
            theta=dict(theta),
            cases=list(cases),
        )
        self._atomic_json(self.iter_manifest_path(it), m.to_json())

    def end_iter(self, it: int, summary: Dict[str, float], is_best: bool) -> None:
        m = self.load_iter_manifest(it)
        m.summary = dict(summary)
        m.is_best = bool(is_best)
        self._atomic_json(self.iter_manifest_path(it), m.to_json())

    def begin_job(
        self,
        it: int,
        case: str,
        tag: str,
        meta: Dict[str, Any] | None = None,
        estimate_bytes: int | None = None,
    ) -> None:
        root = self.job_dir(it, case, tag, estimate_bytes)
        jm = JobManifest(
            case=case,
            tag=tag,
            status="running",
            started_at=time.time(),
            meta=meta or {},
        )
        jm.meta["placement_root"] = str(root)
        self._atomic_json(self.job_manifest_path(it, case, tag), jm.to_json())

    def end_job(self, it: int, case: str, tag: str, status: JobStatus) -> None:
        jm = self._load_job_manifest(it, case, tag)
        jm.status = status
        jm.ended_at = time.time()
        self._atomic_json(self.job_manifest_path(it, case, tag), jm.to_json())

    # ----- artifacts & monitoring -----
    def register_artifact(
        self, it: int, case: str, tag: str, path: Path, kind: ArtifactKind
    ) -> None:
        jm = self._load_job_manifest(it, case, tag)
        rel = path.relative_to(self._job_root(it, case, tag))
        size = path.stat().st_size if path.exists() else 0
        jm.artifacts.append(Artifact(kind=kind, relpath=str(rel), bytes=size))
        self._atomic_json(self.job_manifest_path(it, case, tag), jm.to_json())

    def append_event(self, it: int, case: str, tag: str, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault("ts", time.time())
        p = self.events_path(it, case, tag)
        self._append_jsonl(p, event)

    def write_progress(
        self, it: int, case: str, tag: str, progress: Dict[str, Any]
    ) -> None:
        progress = dict(progress)
        progress.setdefault("ts", time.time())
        self._atomic_json(self.progress_path(it, case, tag), progress)

    def write_probe(self, it: int, case: str, tag: str, probe: Dict[str, Any]) -> None:
        probe = dict(probe)
        probe.setdefault("ts", time.time())
        self._atomic_json(self.xplt_probe_path(it, case, tag), probe)

    # ----- manifests I/O -----
    def load_iter_manifest(self, it: int) -> IterManifest:
        with open(self.iter_manifest_path(it), "r", encoding="utf-8") as f:
            return IterManifest.from_json(json.load(f))

    def _load_job_manifest(self, it: int, case: str, tag: str) -> JobManifest:
        with open(self.job_manifest_path(it, case, tag), "r", encoding="utf-8") as f:
            jm = JobManifest.from_json(json.load(f))
        # cache physical root
        r = jm.meta.get("placement_root")
        if r:
            self._job_roots[(it, case, tag)] = Path(r)
        return jm

    # ----- cleanup / retention -----
    def enforce_retention(self) -> None:
        it_root = self._ensure(self._rel_iter(0)).parent  # project/iters root
        iters = sorted(
            [
                int(p.name.replace("iter", ""))
                for p in (it_root).iterdir()
                if p.is_dir() and p.name.startswith("iter")
            ]
        )
        if not iters:
            return
        keep = set(iters[-self.retention.keep_last :])
        if self.retention.keep_best:
            # read all manifests and keep best ones
            for i in iters:
                mp = it_root / f"iter{i}" / "iter_manifest.json"
                if mp.exists():
                    try:
                        with open(mp, "r", encoding="utf-8") as f:
                            jj = json.load(f)
                        if jj.get("is_best", False):
                            keep.add(i)
                    except Exception:
                        pass
        for i in iters:
            if i not in keep:
                self.prune_iter(i)

        if self.retention.max_bytes is not None:
            # crude size cap: delete oldest until under cap
            while self._project_bytes() > self.retention.max_bytes:
                survivors = sorted(
                    [
                        int(p.name.replace("iter", ""))
                        for p in it_root.iterdir()
                        if p.is_dir()
                        and p.name.startswith("iter")
                        and int(p.name.replace("iter", "")) in keep
                    ]
                )
                victims = sorted(
                    [
                        int(p.name.replace("iter", ""))
                        for p in it_root.iterdir()
                        if p.is_dir()
                        and p.name.startswith("iter")
                        and int(p.name.replace("iter", "")) not in keep
                    ]
                )
                if victims:
                    self.prune_iter(victims[0])
                    victims.pop(0)
                else:
                    break

    def prune_iter(self, it: int) -> None:
        rel = self._rel_iter(it)
        if isinstance(self.backend, HybridBackend):
            self.backend.rm_tree(rel)
        else:
            self.backend.rm_tree(rel)

    # ----- hashing -----
    @staticmethod
    def file_sha1(p: Path, chunk: int = 1 << 20) -> str:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()

    @staticmethod
    def dict_sha1(d: Dict[str, Any]) -> str:
        s = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(s).hexdigest()

    # ----- internals -----
    def _job_root(self, it: int, case: str, tag: str) -> Path:
        k = (it, case, tag)
        if k in self._job_roots:
            return self._job_roots[k]
        # fallback: read manifest
        jm = self._load_job_manifest(it, case, tag)
        r = jm.meta.get("placement_root")
        if not r:
            raise RuntimeError("job root not known")
        p = Path(r)
        self._job_roots[k] = p
        return p

    def _ensure(self, rel: Path, estimate_bytes: int | None = None) -> Path:
        if isinstance(self.backend, HybridBackend):
            return self.backend.ensure(rel, estimate_bytes)
        return self.backend.ensure(rel)

    def _atomic_json(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _append_jsonl(self, path: Path, event: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # atomic append via temp + rename is overkill; open append is fine for single-writer runner
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _project_bytes(self) -> int:
        base = self._ensure(self._rel_iter(0)).parent
        total = 0
        for p in base.rglob("*"):
            try:
                if p.is_file():
                    total += p.stat().st_size
            except Exception:
                pass
        return total
