"""Minimal storage helpers for FEBio optimization runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional
import shutil
import warnings


def _unescape_mount_token(token: str) -> str:
    """Decode the escape sequences used in /proc/mounts."""
    return (
        token.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _filesystem_type(path: Path) -> Optional[str]:
    """
    Best-effort detection of the filesystem type at `path`.

    Returns ``None`` when the platform does not expose /proc/mounts.
    """
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

    parent: Optional[Path] = None
    use_tmp: bool = False
    create: bool = True
    _root: Optional[Path] = field(default=None, init=False, repr=False)

    def resolve(self) -> Path:
        """Return the directory where all simulation files should be generated."""
        if self._root is None:
            if self.use_tmp or self.parent is None:
                self._root = self._resolve_tmp()
            else:
                self._root = self._resolve_parent()
        return self._root

    def _resolve_parent(self) -> Path:
        root = Path(self.parent).expanduser()
        root = root.resolve()
        if self.create:
            root.mkdir(parents=True, exist_ok=True)
        return root

    def _resolve_tmp(self) -> Path:
        tmp_root = Path("/tmp")
        fs_type = _filesystem_type(tmp_root)
        if fs_type is not None and fs_type.lower() != "tmpfs":
            warnings.warn(
                f"{tmp_root} is mounted as {fs_type}, not tmpfs; continuing anyway.",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.parent:
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
        target = Path(path)
        if not target.exists():
            return
        if target.is_dir():
            shutil.rmtree(target, ignore_errors=True)
        else:
            target.unlink(missing_ok=True)

    def cleanup_all(self, keep: Optional[Iterable[Path]] = None) -> None:
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


__all__ = ["StorageManager"]
