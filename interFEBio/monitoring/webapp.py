"""FastAPI web UI for monitoring optimization runs."""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "FastAPI is required for the monitoring web app. "
        "Install interFEBio with the 'monitor' extra, e.g. "
        "`pip install interFEBio[monitor]`."
    ) from exc

from .events import EventSocketListener
from .paths import default_registry_path, default_socket_path
from .registry import ActiveRunDeletionError, RunRegistry
from .state import StorageInventory
from .system_stats import SystemStatsCollector
from .templates import MonitorPageTemplate

logger = logging.getLogger(__name__)

TEMPLATE = MonitorPageTemplate()


def render_home_page() -> str:
    """Render the static HTML shell for the monitor dashboard."""
    return TEMPLATE.render()


def _json_safe(value: Any) -> Any:
    """Recursively sanitise values so JSON encoding never sees NaN/Inf."""
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int,)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def create_app(
    registry: RunRegistry,
    *,
    inventory: Optional[StorageInventory] = None,
    event_socket: Optional[Path] = None,
    stats_collector: Optional[SystemStatsCollector] = None,
) -> FastAPI:
    """Build a FastAPI application exposing the monitor endpoints."""
    registry.refresh()
    system_stats = stats_collector or SystemStatsCollector()
    app = FastAPI(title="interFEBio Monitor", version="2.0.0")
    listener: Optional[EventSocketListener] = None

    @app.on_event("startup")
    async def _startup() -> None:
        """Start the event listener when the application boots."""
        nonlocal listener
        logger.info("Monitor web app starting up")
        if event_socket:
            listener = EventSocketListener(Path(event_socket), registry)
            listener.start()
            logger.info("Event socket listener active at %s", event_socket)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        """Shut down the event listener when the app stops."""
        if listener:
            listener.stop()

    @app.get("/api/runs")
    async def list_runs():
        """Return the latest list of runs."""
        registry.refresh()
        runs = registry.list_runs()
        payload = []
        for run in runs:
            meta = dict(run.meta)
            payload.append(
                {
                    "run_id": run.run_id,
                    "label": run.label,
                    "status": run.status,
                    "created_at": run.created_at,
                    "updated_at": run.updated_at,
                    "last_cost": meta.get("last_cost"),
                    "iteration_count": len(run.iterations),
                    "optimizer": meta.get("optimizer"),
                }
            )
        return _json_safe(payload)

    @app.get("/api/system/metrics")
    async def system_metrics():
        """Return current CPU, memory, and disk utilisation."""
        return system_stats.collect()

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
        """Return metadata for the selected run."""
        registry.refresh()
        run = registry.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        data = run.to_dict()
        if inventory:
            inventory.refresh()
            job = inventory.get_job(run_id)
            if job:
                data["artifacts"] = [
                    {"kind": art.kind, "path": str(art.path), "size": art.size}
                    for art in job.artifacts
                ]
        return _json_safe(data)

    @app.get("/api/runs/{run_id}/processes")
    async def run_processes(run_id: str):
        """Return OS processes likely associated with this run."""
        registry.refresh()
        run = registry.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        meta = run.meta if isinstance(run.meta, dict) else {}
        runtime_root = meta.get("runtime_root")
        storage_root = meta.get("storage_root")
        roots: list[str] = []
        if runtime_root is not None:
            roots.append(str(runtime_root))
        if storage_root is not None:
            storage_str = str(storage_root)
            if not roots or storage_str not in roots:
                roots.append(storage_str)
        root_used: str | None = None
        processes: list[dict[str, object]] = []
        for candidate in roots:
            root_used = candidate
            processes = system_stats.collect_processes(root=candidate)
            if processes:
                break
        return _json_safe({
            "run_id": run_id,
            "root": root_used,
            "processes": processes,
            "supported": system_stats.process_support,
        })

    @app.get("/api/runs/{run_id}/iterations")
    async def run_iterations(run_id: str):
        """Return the iteration history for the selected run."""
        registry.refresh()
        run = registry.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return _json_safe([record.to_dict() for record in run.iterations])

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str, force: bool = False):
        """Delete a single run, optionally forcing removal."""
        try:
            removed = registry.delete_run(run_id, force=force)
        except ActiveRunDeletionError:
            raise HTTPException(
                status_code=409,
                detail="Run is still active; stop the optimisation before deleting.",
            ) from None
        if not removed:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": "deleted", "run_id": run_id, "forced": force}

    @app.delete("/api/runs")
    async def delete_all_runs(force: bool = False):
        """Delete runs and optionally force removal of active entries."""
        registry.refresh()
        total_before = len(registry.list_runs())
        protected = registry.clear(force=force)
        total_after = len(registry.list_runs())
        removed = max(total_before - total_after, 0)
        return {
            "status": "cleared",
            "removed": removed,
            "protected": protected,
            "forced": force,
        }

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        """Render the dashboard HTML."""
        return HTMLResponse(render_home_page())

    return app


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for the monitoring web app."""
    parser = argparse.ArgumentParser(description="interFEBio monitoring service")
    parser.add_argument(
        "--registry",
        type=Path,
        default=default_registry_path(),
        help="Path to monitor registry file (default: %(default)s).",
    )
    parser.add_argument(
        "--event-socket",
        type=Path,
        default=default_socket_path(),
        help="Unix domain socket path for monitor events (default: %(default)s).",
    )
    parser.add_argument(
        "--storage-root",
        action="append",
        type=Path,
        help="Optional storage roots to scan for artifacts.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Directory scan interval in seconds for storage roots (default: 5).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="HTTP port to bind (default: 8765).",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    """Run the monitoring web app CLI using parsed options."""
    import uvicorn

    args = parse_args(list(argv) if argv is not None else None)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    registry = RunRegistry(args.registry)
    inventory = None
    if args.storage_root:
        inventory = StorageInventory(
            args.storage_root, poll_interval=args.poll_interval
        )
    app = create_app(
        registry,
        inventory=inventory,
        event_socket=args.event_socket,
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
