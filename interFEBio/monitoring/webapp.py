from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, Optional

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "FastAPI is required for the monitoring web app. "
        "Install interFEBio with the 'monitor' extra, e.g. "
        "`pip install interFEBio[monitor]`."
    ) from exc

from pathlib import Path

from .state import StorageInventory
from .events import EventSocketListener

logger = logging.getLogger(__name__)


def create_app(
    inventory: StorageInventory, *, event_socket: Optional[Path] = None
) -> FastAPI:
    inventory.refresh(force=True)
    app = FastAPI(title="interFEBio Monitor", version="1.0.0")
    listener: Optional[EventSocketListener] = None

    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal listener
        logger.info("Monitor web app starting up")
        if event_socket:
            listener = EventSocketListener(Path(event_socket), inventory)
            listener.start()
            logger.info("Event socket listener active at %s", event_socket)

    @app.get("/api/jobs")
    async def list_jobs():
        inventory.refresh()
        return [
            {
                "job_id": job.job_id,
                "project": job.project,
                "iter_id": job.iter_id,
                "case": job.case,
                "tag": job.tag,
                "status": job.status,
                "started_at": job.started_at,
                "ended_at": job.ended_at,
                "summary": job.summary,
                "meta": job.meta,
            }
            for job in inventory.list_jobs()
        ]

    @app.get("/api/jobs/{job_id}")
    async def job_detail(job_id: str):
        inventory.refresh()
        job = inventory.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return {
            "job_id": job.job_id,
            "project": job.project,
            "iter_id": job.iter_id,
            "case": job.case,
            "tag": job.tag,
            "status": job.status,
            "started_at": job.started_at,
            "ended_at": job.ended_at,
            "placement_root": str(job.placement_root),
            "summary": job.summary,
            "meta": job.meta,
            "artifacts": [
                {
                    "kind": art.kind,
                    "path": str(art.path),
                    "size": art.size,
                }
                for art in job.artifacts
            ],
        }

    @app.get("/", response_class=HTMLResponse)
    async def home():
        inventory.refresh()
        jobs = inventory.list_jobs()
        rows = "".join(
            f"<tr>"
            f"<td><a href='/jobs/{job.job_id}'>{job.job_id}</a></td>"
            f"<td>{job.status}</td>"
            f"<td>{job.summary}</td>"
            f"</tr>"
            for job in jobs
        ) or "<tr><td colspan='3'>No jobs</td></tr>"
        html = f"""
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>interFEBio Monitor</title>
    <style>
      body {{ font-family: Arial, sans-serif; margin: 1.5rem; }}
      table {{ border-collapse: collapse; width: 100%; }}
      th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
      th {{ background: #f1f1f1; }}
    </style>
  </head>
  <body>
    <h1>Optimization Jobs</h1>
    <table>
      <thead><tr><th>Job</th><th>Status</th><th>Summary</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>
  </body>
</html>
"""
        return HTMLResponse(html)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if listener:
            listener.stop()

    return app


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="interFEBio monitoring service")
    parser.add_argument(
        "--storage-root",
        action="append",
        type=Path,
        help="Storage root produced by StorageManager (can be passed multiple times).",
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
        "--poll-interval",
        type=float,
        default=5.0,
        help="Directory scan interval in seconds (default: 5).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Logging level (default: INFO).",
    )
    parser.add_argument(
        "--event-socket",
        type=Path,
        default=None,
        help="Unix domain socket path for job event ingestion.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> int:
    import uvicorn

    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not args.storage_root:
        raise SystemExit("No --storage-root provided.")
    inventory = StorageInventory(args.storage_root, poll_interval=args.poll_interval)
    app = create_app(inventory, event_socket=args.event_socket)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
