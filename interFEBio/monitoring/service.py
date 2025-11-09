"""CLI helpers for installing and running the monitoring service."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

from .paths import default_registry_path, default_socket_path


UNIT_TEMPLATE = """[Unit]
Description=interFEBio monitoring web service
After=network.target

[Service]
Type=simple
ExecStart={python} -m interFEBio.monitoring.service run --registry {registry} --socket {socket}
Restart=on-failure
RestartSec=2

[Install]
WantedBy=default.target
"""


def ensure_dependencies() -> None:
    """Ensure the optional monitoring stack is available."""
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional dependency check
        raise RuntimeError(
            "interfebio-monitor requires the 'monitor' extra. Install with "
            "pip install interFEBio[monitor]."
        ) from exc


def run_service(*, registry: Path | None = None, socket: Path | None = None) -> int:
    """Start the monitoring web service with the given overrides."""
    ensure_dependencies()
    from .webapp import main as web_main

    args = [
        "--registry",
        str(registry or default_registry_path()),
        "--event-socket",
        str(socket or default_socket_path()),
        "--host",
        "127.0.0.1",
        "--port",
        "8765",
    ]
    return web_main(args)


def install_service(*, user: bool = True, force: bool = False) -> None:
    """Install the systemd unit for the monitoring service."""
    ensure_dependencies()
    unit_content = UNIT_TEMPLATE.format(
        python=sys.executable,
        registry=default_registry_path(),
        socket=default_socket_path(),
    )
    if user:
        unit_dir = Path.home() / ".config/systemd/user"
    else:
        unit_dir = Path("/etc/systemd/system")
    unit_dir.mkdir(parents=True, exist_ok=True)
    unit_path = unit_dir / "interfebio-monitor.service"
    if unit_path.exists() and not force:
        raise FileExistsError(f"Unit already exists at {unit_path}. Use --force to overwrite.")
    unit_path.write_text(unit_content, encoding="utf-8")
    _systemctl(["daemon-reload"], user=user)
    _systemctl(["enable", "interfebio-monitor.service"], user=user)
    _systemctl(["start", "interfebio-monitor.service"], user=user)
    print(f"Systemd unit installed at {unit_path}")


def uninstall_service(*, user: bool = True) -> None:
    """Disable and remove the monitoring systemd unit."""
    unit_path = (
        Path.home() / ".config/systemd/user/interfebio-monitor.service"
        if user
        else Path("/etc/systemd/system/interfebio-monitor.service")
    )
    _systemctl(["disable", "interfebio-monitor.service"], user=user, check=False)
    _systemctl(["stop", "interfebio-monitor.service"], user=user, check=False)
    if unit_path.exists():
        unit_path.unlink()
    _systemctl(["daemon-reload"], user=user, check=False)
    print("interfebio-monitor service uninstalled")


def _systemctl(args: list[str], *, user: bool, check: bool = True) -> None:
    """Wrapper around systemctl invocation used by the install scripts."""
    executable = shutil.which("systemctl")
    if executable is None:
        raise RuntimeError("systemctl executable not found in PATH")
    cmd = [executable]
    if user:
        cmd.append("--user")
    cmd.extend(args)
    result = subprocess.run(cmd, check=check)
    if check and result.returncode != 0:
        raise RuntimeError(f"systemctl {' '.join(args)} failed with code {result.returncode}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    """Parse the CLI arguments for the monitoring helper."""
    parser = argparse.ArgumentParser(description="interFEBio monitoring service helper")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("run", help="Run the monitoring web service with default settings")

    install_parser = subparsers.add_parser("install", help="Install and enable the systemd unit")
    install_parser.add_argument("--system", action="store_true", help="Install system-wide (requires root)")
    install_parser.add_argument("--force", action="store_true", help="Overwrite existing unit")

    run_parser = subparsers.choices["run"]
    run_parser.add_argument("--registry", type=Path, default=None, help="Registry path override")
    run_parser.add_argument("--socket", type=Path, default=None, help="Event socket override")

    uninstall_parser = subparsers.add_parser("uninstall", help="Disable and remove the systemd unit")
    uninstall_parser.add_argument("--system", action="store_true", help="Remove system-wide unit")

    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    """Dispatch commands from the monitoring helper CLI."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.command == "run":
        return run_service(registry=args.registry, socket=args.socket)
    if args.command == "install":
        install_service(user=not args.system, force=args.force)
        return 0
    if args.command == "uninstall":
        uninstall_service(user=not args.system)
        return 0
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
