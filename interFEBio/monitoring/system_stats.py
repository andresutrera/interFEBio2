"""System statistics helpers for the monitoring web UI."""

from __future__ import annotations

import os
import shutil
import time
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
            snapshot["load_avg"] = list(load_avg)
        return snapshot

    @staticmethod
    def _safe_cpu_percent() -> float | None:
        if psutil is not None:
            try:
                return float(psutil.cpu_percent(interval=None))
            except Exception:  # pragma: no cover
                return None
        try:
            load1, _, _ = os.getloadavg()
            cores = os.cpu_count() or 1
            return max(0.0, min(100.0, (load1 / cores) * 100.0))
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
            "total": total,
            "available": available,
            "used": used,
            "percent": percent,
        }

    def _safe_memory_stats(self) -> dict[str, float] | None:
        if psutil is not None:
            try:
                stats = psutil.virtual_memory()
                return {
                    "total": float(stats.total),
                    "available": float(stats.available),
                    "used": float(stats.used),
                    "percent": float(stats.percent),
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
                        "total": float(usage.total),
                        "used": float(usage.used),
                        "free": float(usage.free),
                        "percent": float(usage.percent),
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
                        "total": float(usage.total),
                        "used": float(usage.used),
                        "free": float(usage.free),
                        "percent": float(usage.used) / float(usage.total) * 100.0
                        if usage.total
                        else 0.0,
                    }
                )
        disks.sort(key=lambda item: item.get("total", 0.0), reverse=True)
        return disks[: self.disk_limit]


__all__ = ["SystemStatsCollector"]
