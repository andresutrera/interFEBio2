"""Minimal storage helpers for FEBio optimization runs."""

from __future__ import annotations

import shutil
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from .options import CleanupOptions, StorageOptions

def _unescape_mount_token(token: str) -> str:
    """Decode escape sequences used in ``/proc/mounts`` entries."""
    return (
        token.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _filesystem_type(path: Path) -> str | None:
    """Return the filesystem type for ``path`` when it can be determined."""
    mounts = Path("/proc/mounts")
    if not mounts.exists():
        return None

    target = path
    try:
        target = path.resolve()
    except OSError:
        pass

    try:
        with mounts.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point = Path(_unescape_mount_token(parts[1]))
                try:
                    mount_point_resolved = mount_point.resolve()
                except OSError:
                    mount_point_resolved = mount_point
                if target == mount_point_resolved:
                    return parts[2]
    except OSError:
        return None
    return None


@dataclass
class StorageManager:
    """
    Resolve a working directory for FEBio simulations.

    Users can either provide a parent directory on disk or ask for a temporary
    directory under ``/tmp``. When ``/tmp`` is selected, the project folder is
    named after the current working directory.
    """

    parent: Path | None = None
    use_tmp: bool = False
    create: bool = True
    _root: Path | None = field(default=None, init=False, repr=False)

    def resolve(self) -> Path:
        """Return the directory where all simulation files should be generated."""
        if self._root is None:
            if self.use_tmp or self.parent is None:
                self._root = self._resolve_tmp()
            else:
                self._root = self._resolve_parent()
        return self._root

    def _resolve_parent(self) -> Path:
        if self.parent is None:
            raise ValueError("parent must be provided for disk storage.")
        root = self.parent.expanduser().resolve()
        if self.create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    def _resolve_tmp(self) -> Path:
        """Resolve a temporary storage directory under ``/tmp``."""
        tmp_root = Path("/tmp")
        fs_type = _filesystem_type(tmp_root)
        if fs_type is not None and fs_type.lower() != "tmpfs":
            warnings.warn(
                f"{tmp_root} is mounted as {fs_type}, not tmpfs; continuing anyway.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.parent is not None:
            label = Path(self.parent).name
        else:
            label = Path.cwd().name
        if not label:
            label = "interFEBio"
        root = tmp_root / label
        if self.create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    def cleanup_path(self, path: Path) -> None:
        """Remove a file or directory tree under the storage root."""
        target = Path(path)
        if not target.exists():
            return
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)

    def cleanup_all(self, keep: Iterable[Path] | None = None) -> None:
        """Remove all children under the storage root except entries in ``keep``."""
        root = self.resolve()
        keep_set = {Path(p).resolve() for p in (keep or [])}
        for child in root.iterdir():
            resolved = child.resolve()
            if keep_set and resolved in keep_set:
                continue
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)


class StorageWorkspace:
    """Manage working directories, persistent artefacts, and series exports."""

    def __init__(self, storage_opts: StorageOptions, cleanup_opts: CleanupOptions) -> None:
        storage_mode = storage_opts.mode.lower()
        if storage_mode not in {"disk", "tmp"}:
            raise ValueError("storage_mode must be 'disk' or 'tmp'")
        cleanup_mode = cleanup_opts.mode.lower()
        cleanup_previous = bool(cleanup_opts.remove_previous)
        if storage_mode == "tmp":
            cleanup_previous = True
            cleanup_mode = "all"
        if cleanup_mode not in {"none", "retain_best", "all"}:
            raise ValueError("cleanup_mode must be one of: 'none', 'retain_best', 'all'")

        parent = Path(storage_opts.root).expanduser() if storage_opts.root is not None else None
        self.storage = StorageManager(parent=parent, use_tmp=(storage_mode == "tmp"))
        self.storage_mode = storage_mode
        self.cleanup_mode = cleanup_mode
        self.cleanup_previous = cleanup_previous
        self.workdir = self.storage.resolve()
        if storage_mode == "tmp":
            if storage_opts.root is not None:
                persist_root = Path(storage_opts.root).expanduser().resolve()
            else:
                persist_root = Path.cwd() / self.workdir.name
            persist_root.mkdir(parents=True, exist_ok=True)
            self.persist_root = persist_root
        else:
            self.persist_root = self.workdir
        self.log_file = self._resolve_log_file(storage_opts.log_file, storage_opts.root)
        self._iter_dirs: list[Path] = []
        self._eval_index = 0

    def describe(self) -> list[str]:
        return [
            f"• mode: {self.storage_mode}",
            f"• workdir: {self.workdir}",
            f"• persist_root: {self.persist_root}",
            f"• cleanup_mode: {self.cleanup_mode}",
            f"• remove_previous: {self.cleanup_previous}",
        ]

    def next_iter_dir(self) -> Path:
        self._eval_index += 1
        iter_dir = self.workdir / f"eval{self._eval_index}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        self._iter_dirs.append(iter_dir)
        return iter_dir

    def prune_old_iterations(self, last_iter_dir: Path | None) -> None:
        if not self.cleanup_previous:
            return
        keep: set[Path] = set()
        if last_iter_dir is not None:
            keep.add(last_iter_dir.resolve())
        retained: list[Path] = []
        for dir_path in self._iter_dirs:
            resolved = dir_path.resolve()
            if resolved in keep:
                retained.append(dir_path)
            else:
                self.storage.cleanup_path(dir_path)
        self._iter_dirs = retained

    def final_cleanup(self, last_iter_dir: Path | None) -> None:
        if self.storage_mode == "tmp":
            self._persist_best(last_iter_dir)
            shutil.rmtree(self.workdir, ignore_errors=True)
            self._iter_dirs = []
            return

        if self.cleanup_mode == "none":
            return
        keep_paths: set[Path] = set()
        if self.cleanup_mode == "retain_best" and last_iter_dir is not None:
            keep_paths.add(last_iter_dir.resolve())
        self.storage.cleanup_all(keep_paths if keep_paths else None)
        if keep_paths:
            keep_resolved = {p.resolve() for p in keep_paths}
            self._iter_dirs = [path for path in self._iter_dirs if path.resolve() in keep_resolved]
        else:
            self._iter_dirs = []

    def write_series(self, series: Mapping[str, Mapping[str, Sequence[float]]]) -> None:
        if not series:
            return
        target_dir = self.log_file.parent
        target_dir.mkdir(parents=True, exist_ok=True)
        for key, payload in series.items():
            x_vals = payload.get("x")
            exp_vals = payload.get("y_exp")
            sim_vals = payload.get("y_sim")
            if not (x_vals and exp_vals and sim_vals):
                continue
            length = min(len(x_vals), len(exp_vals), len(sim_vals))
            if length == 0:
                continue
            safe_name = key.replace("/", "_")
            path = target_dir / f"{safe_name}_series.txt"
            lines = [f"# {key}\n", "# x y_exp y_sim\n"]
            for idx in range(length):
                lines.append(f"{x_vals[idx]:.10g} {exp_vals[idx]:.10g} {sim_vals[idx]:.10g}\n")
            path.write_text("".join(lines), encoding="utf-8")

    def _persist_best(self, last_iter_dir: Path | None) -> None:
        best_dir = last_iter_dir
        if best_dir is None:
            return
        best_dir = best_dir.resolve()
        if not best_dir.exists():
            return
        dest = self.persist_root / best_dir.name
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(best_dir, dest, dirs_exist_ok=True)

    @staticmethod
    def _resolve_log_file(log_file: str | Path | None, storage_root: str | Path | None) -> Path:
        path: Path | None = None
        if log_file is not None:
            candidate = str(log_file).strip()
            if candidate:
                path = Path(candidate).expanduser()
        if path is None:
            base: Path | None = None
            if storage_root is not None:
                storage_str = str(storage_root).strip()
                if storage_str:
                    base = Path(storage_str).expanduser()
            if base is not None:
                base.mkdir(parents=True, exist_ok=True)
                path = base / "optimization.log"
            else:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = Path.cwd() / f"optimization_{timestamp}.log"
        resolved = path.expanduser()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        return resolved


__all__ = ["StorageManager", "StorageWorkspace"]
