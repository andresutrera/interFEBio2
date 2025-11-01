from __future__ import annotations

import json
import os
import socket
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .state import StorageInventory


@dataclass
class EventEnvelope:
    job_id: str
    event: str
    payload: dict
    ts: float = time.time()
    schema: str = "interfebio.jobevent/v1"

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema": self.schema,
                "job_id": self.job_id,
                "event": self.event,
                "payload": self.payload,
                "timestamp": self.ts,
            }
        )


class EventEmitter:
    def emit(self, job_id: str, event: str, payload: dict) -> None:
        raise NotImplementedError


class NullEventEmitter(EventEmitter):
    def emit(self, job_id: str, event: str, payload: dict) -> None:
        return


class SocketEventEmitter(EventEmitter):
    def __init__(self, socket_path: Path):
        self.socket_path = Path(socket_path)

    def emit(self, job_id: str, event: str, payload: dict) -> None:
        envelope = EventEnvelope(job_id=job_id, event=event, payload=payload)
        message = envelope.to_json() + "\n"
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(0.2)
                sock.connect(str(self.socket_path))
                sock.sendall(message.encode("utf-8"))
        except OSError:
            # Monitor might be offline; we intentionally drop the event.
            pass


def create_event_emitter(socket_path: Optional[Path]) -> EventEmitter:
    if socket_path:
        return SocketEventEmitter(socket_path)
    return NullEventEmitter()


class EventSocketListener:
    def __init__(self, socket_path: Path, inventory: StorageInventory):
        self.socket_path = Path(socket_path)
        self.inventory = inventory
        self._server: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            return
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o660)
        server.listen()
        server.settimeout(1.0)
        self._server = server
        self._thread = threading.Thread(target=self._serve, name="EventSocketListener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._server is not None:
            try:
                self._server.close()
            except OSError:
                pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None
        if self.socket_path.exists():
            try:
                self.socket_path.unlink()
            except OSError:
                pass

    def _serve(self) -> None:
        assert self._server is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            threading.Thread(target=self._handle_conn, args=(conn,), daemon=True).start()

    def _handle_conn(self, conn: socket.socket) -> None:
        with conn:
            conn.settimeout(1.0)
            buffer = ""
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not chunk:
                    break
                buffer += chunk.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    job_id = msg.get("job_id")
                    event = msg.get("event")
                    payload = msg.get("payload", {})
                    ts = float(msg.get("timestamp", time.time()))
                    if not job_id or not event:
                        continue
                    if isinstance(payload, dict):
                        self.inventory.apply_event(job_id, event, payload, ts)


__all__ = [
    "EventEmitter",
    "NullEventEmitter",
    "SocketEventEmitter",
    "create_event_emitter",
    "EventSocketListener",
]
