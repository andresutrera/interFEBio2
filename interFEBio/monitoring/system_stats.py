"""System statistics helpers for the monitoring web UI."""

from __future__ import annotations

import os
import shutil
import time
from math import isfinite
from pathlib import Path
from typing import Any

try:  # pragma: no cover - optional dependency at runtime
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    psutil = None


class SystemStatsCollector:
    """Collect best-effort CPU, memory, and disk metrics."""

    def __init__(self, *, disk_limit: int = 4) -> None:
        self.disk_limit = max(1, disk_limit)
        self._process_enabled = psutil is not None

    @property
    def process_support(self) -> bool:
        """Return ``True`` when process inspection is available."""
        return self._process_enabled

    def collect(self) -> dict[str, Any]:
        """Return a snapshot of host CPU, memory, and disk usage."""
        memory = self._safe_memory_stats() or {}
        disks = self._safe_disk_stats()
        try:
            load_avg = os.getloadavg()
        except (AttributeError, OSError):
            load_avg = None
        snapshot: dict[str, Any] = {
            "timestamp": time.time(),
            "cpu_percent": self._safe_cpu_percent(),
            "cpu_count": os.cpu_count(),
            "memory": memory,
            "disks": disks,
        }
        if load_avg is not None:
            snapshot["load_avg"] = [self._clean_number(value) for value in load_avg]
        return snapshot

    def collect_processes(
        self,
        *,
        root: str | os.PathLike[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Return lightweight process info filtered to those under ``root``."""
        if not self._process_enabled:
            return []
        assert psutil is not None  # for type-checkers
        root_path = self._normalize_root(root)
        if root is not None and root_path is None:
            return []
        if root_path is None:
            return []
        entries: list[dict[str, Any]] = []
        try:
            attrs = [
                "pid",
                "name",
                "cmdline",
                "cwd",
                "create_time",
                "status",
            ]
            if root_path is not None:
                attrs.append("environ")
            iterator = psutil.process_iter(attrs)
        except Exception:
            return []
        for proc in iterator:
            info = getattr(proc, "info", {}) or {}
            try:
                if root_path and not self._process_matches_root(info, root_path):
                    continue
                entries.append(
                    self._summarize_process(info, fetch_env=root_path is not None)
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied, ProcessLookupError, psutil.ZombieProcess):
                continue
            except Exception:
                continue
        entries.sort(key=lambda item: (item.get("started_at") or 0.0, item["pid"]))
        return entries

    @staticmethod
    def _normalize_root(root: str | os.PathLike[str] | None) -> Path | None:
        if root is None:
            return None
        try:
            return Path(root).expanduser().resolve()
        except OSError:
            try:
                return Path(root).expanduser()
            except Exception:
                return None

    @staticmethod
    def _process_matches_root(info: dict[str, Any], root: Path) -> bool:
        cwd = info.get("cwd")
        if isinstance(cwd, str) and cwd:
            try:
                cwd_path = Path(cwd).resolve()
            except OSError:
                cwd_path = Path(cwd)
            if SystemStatsCollector._is_relative_to(cwd_path, root):
                return True
        cmdline = info.get("cmdline") or []
        root_str = str(root)
        for arg in cmdline:
            if isinstance(arg, str) and root_str in arg:
                return True
        return False

    @staticmethod
    def _is_relative_to(candidate: Path, base: Path) -> bool:
        try:
            candidate.relative_to(base)
            return True
        except ValueError:
            return False

    @staticmethod
    def _summarize_process(info: dict[str, Any], *, fetch_env: bool) -> dict[str, Any]:
        cmdline = [str(arg) for arg in info.get("cmdline") or []]
        started_at = info.get("create_time")
        try:
            started = float(started_at) if started_at is not None else None
        except (TypeError, ValueError):
            started = None
        pid_value = info.get("pid")
        try:
            pid = int(pid_value)
        except (TypeError, ValueError):
            pid = -1
        return {
            "pid": pid,
            "name": info.get("name") or "",
            "status": info.get("status"),
            "cmdline": cmdline,
            "cwd": info.get("cwd"),
            "started_at": started,
            "omp_threads": SystemStatsCollector._read_omp_env(info) if fetch_env else None,
        }

    @staticmethod
    def _read_omp_env(info: dict[str, Any]) -> int | None:
        environ = info.get("environ")
        if isinstance(environ, dict):
            value = environ.get("OMP_NUM_THREADS")
            if value is not None:
                try:
                    threads = int(value)
                    if threads > 0:
                        return threads
                except (TypeError, ValueError):
                    return None
        return None

    @staticmethod
    def _safe_cpu_percent() -> float | None:
        if psutil is not None:
            try:
                value = psutil.cpu_percent(interval=None)
                return SystemStatsCollector._clean_number(value)
            except Exception:  # pragma: no cover
                return None
        try:
            load1, _, _ = os.getloadavg()
            cores = os.cpu_count() or 1
            value = max(0.0, min(100.0, (load1 / cores) * 100.0))
            return SystemStatsCollector._clean_number(value)
        except (AttributeError, OSError):  # pragma: no cover - platform-specific
            return None

    @staticmethod
    def _fallback_memory_stats() -> dict[str, float] | None:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            phys_pages = os.sysconf("SC_PHYS_PAGES")
            avail_pages = os.sysconf("SC_AVPHYS_PAGES")
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            return None
        total = float(page_size) * float(phys_pages)
        available = float(page_size) * float(avail_pages)
        used = max(total - available, 0.0)
        percent = (used / total * 100.0) if total else 0.0
        return {
            "total": SystemStatsCollector._clean_number(total),
            "available": SystemStatsCollector._clean_number(available),
            "used": SystemStatsCollector._clean_number(used),
            "percent": SystemStatsCollector._clean_number(percent),
        }

    def _safe_memory_stats(self) -> dict[str, float] | None:
        if psutil is not None:
            try:
                stats = psutil.virtual_memory()
                return {
                    "total": self._clean_number(stats.total),
                    "available": self._clean_number(stats.available),
                    "used": self._clean_number(stats.used),
                    "percent": self._clean_number(stats.percent),
                }
            except Exception:  # pragma: no cover
                return None
        return self._fallback_memory_stats()

    def _safe_disk_stats(self) -> list[dict[str, Any]]:
        disks: list[dict[str, Any]] = []
        seen: set[str] = set()
        if psutil is not None:
            try:
                partitions = psutil.disk_partitions(all=False)
            except Exception:  # pragma: no cover
                partitions = []
            for part in partitions:
                mount = part.mountpoint or part.device or ""
                if not mount or mount in seen:
                    continue
                if mount.startswith("/snap/") or mount == "/snap":
                    continue
                device_name = part.device or ""
                if device_name.startswith("/dev/loop"):
                    continue
                try:
                    usage = psutil.disk_usage(part.mountpoint)
                except (PermissionError, FileNotFoundError, OSError):
                    continue
                if usage.total <= 0:
                    continue
                seen.add(mount)
                disks.append(
                    {
                        "mount": part.mountpoint,
                        "device": part.device,
                        "fstype": part.fstype,
                        "total": self._clean_number(usage.total),
                        "used": self._clean_number(usage.used),
                        "free": self._clean_number(usage.free),
                        "percent": self._clean_number(usage.percent),
                    }
                )
        if not disks:
            try:
                usage = shutil.disk_usage(Path("/") if os.name != "nt" else Path("C:/"))
            except (PermissionError, FileNotFoundError, OSError):
                usage = None
            if usage:
                disks.append(
                    {
                        "mount": "/" if os.name != "nt" else "C:/",
                        "device": None,
                        "fstype": None,
                        "total": self._clean_number(usage.total),
                        "used": self._clean_number(usage.used),
                        "free": self._clean_number(usage.free),
                        "percent": self._clean_number(
                            float(usage.used) / float(usage.total) * 100.0
                            if usage.total
                            else 0.0
                        ),
                    }
                )
        disks.sort(key=lambda item: item.get("total", 0.0), reverse=True)
        return disks[: self.disk_limit]

    @staticmethod
    def _clean_number(value: Any) -> float | None:
        """Return a JSON-safe float or ``None`` when the value is invalid."""
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if isfinite(number):
            return number
        return None


__all__ = ["SystemStatsCollector"]
