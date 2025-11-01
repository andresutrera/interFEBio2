from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import List, Optional

from .monitor import MonitorFSView


class MonitorWebUIServer:
    """Minimal HTTP server presenting monitor status as JSON+HTML."""

    def __init__(
        self,
        view: MonitorFSView,
        *,
        host: str = "127.0.0.1",
        port: int = 8765,
        refresh_interval: float = 1.0,
    ):
        self.view = view
        self.host = host
        self.port = port
        self.refresh_interval = max(0.2, float(refresh_interval))
        self._server: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._server is not None:
            return
        handler_cls = self._make_handler()
        self._server = ThreadingHTTPServer((self.host, self.port), handler_cls)
        self._thread = threading.Thread(
            target=self._server.serve_forever, name="MonitorWebUI", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self._server = None
        self._thread = None

    def _make_handler(self) -> type[BaseHTTPRequestHandler]:
        view = self.view
        refresh_ms = int(self.refresh_interval * 1000)

        class Handler(BaseHTTPRequestHandler):
            def _snapshot(self) -> List[dict[str, object]]:
                data: List[dict[str, object]] = []
                for job in view.snapshot():
                    data.append(
                        {
                            "key": job.key,
                            "status": job.status,
                            "updated_at": job.updated_at,
                            "exit_code": job.exit_code,
                            "meta": job.meta,
                        }
                    )
                return data

            def do_GET(self) -> None:  # type: ignore[override]
                if self.path == "/status":
                    payload = json.dumps(self._snapshot()).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Cache-Control", "no-store")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                    return

                snapshot = self._snapshot()
                body = self._render_html(snapshot, refresh_ms).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def _render_html(
                self, snapshot: List[dict[str, object]], refresh_ms: int
            ) -> str:
                initial_json = json.dumps(snapshot)
                return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>interFEBio Monitor</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 1.5rem; background: #1f2330; color: #f5f5f5; }}
    h1 {{ margin-bottom: 0.5rem; }}
    table {{ border-collapse: collapse; width: 100%; background: #2b3245; border-radius: 8px; overflow: hidden; }}
    th, td {{ padding: 0.6rem 0.8rem; text-align: left; }}
    th {{ background: #3a435b; text-transform: uppercase; font-size: 0.75rem; letter-spacing: 0.05rem; }}
    tr:nth-child(even) {{ background: #242a3b; }}
    td.status.finished {{ color: #81d887; }}
    td.status.failed {{ color: #ff6b6b; }}
    td.status.running {{ color: #ffd166; }}
    td.status.queued {{ color: #9bb4ff; }}
    #updated {{ font-size: 0.8rem; margin-top: 0.75rem; opacity: 0.7; }}
    button {{ margin-right: 0.5rem; padding: 0.4rem 0.8rem; background: #3a7bd5; color: #fff; border: none; border-radius: 4px; cursor: pointer; }}
    button:hover {{ background: #336bb5; }}
  </style>
</head>
<body>
  <h1>Simulation Monitor</h1>
  <div>
    <button id="manualRefresh">Refresh</button>
    <button id="toggleAuto">Auto Refresh: <span id="autoLabel">ON</span></button>
  </div>
  <table>
    <thead>
      <tr><th>Job</th><th>Status</th><th>Time</th><th>Step</th><th>Exit</th></tr>
    </thead>
    <tbody id="jobRows"></tbody>
  </table>
  <div id="updated">Last update: <span id="lastUpdate">never</span></div>
  <script>
    const REFRESH_MS = {refresh_ms};
    const initialData = {initial_json};
    let auto = true;

    function escapeHtml(str) {{
      return str.replace(/[&<>\"']/g, c => ({{
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '\"': '&quot;',
        \"'\": '&#39;'
      }})[c] || c);
    }}

    function render(data) {{
      const body = document.getElementById('jobRows');
      if (!body) return;
      if (!data.length) {{
        body.innerHTML = "<tr><td colspan='5'>No jobs yet</td></tr>";
        return;
      }}
      body.innerHTML = data.map(job => {{
        const meta = job.meta || {{}};
        const timeVal = meta.time ?? '-';
        const stepVal = meta.step ?? '-';
        const exitCode = job.exit_code ?? '';
        const status = job.status || 'unknown';
        return `<tr>
          <td>${{escapeHtml(String(job.key ?? ''))}}</td>
          <td class="status ${{escapeHtml(status)}}">${{escapeHtml(status)}}</td>
          <td>${{escapeHtml(String(timeVal))}}</td>
          <td>${{escapeHtml(String(stepVal))}}</td>
          <td>${{escapeHtml(String(exitCode))}}</td>
        </tr>`;
      }}).join('');
      const stamp = document.getElementById('lastUpdate');
      if (stamp) stamp.textContent = new Date().toLocaleTimeString();
    }}

    async function fetchStatus() {{
      try {{
        const res = await fetch('/status', {{ cache: 'no-store' }});
        const data = await res.json();
        render(data);
      }} catch (err) {{
        console.error('Status refresh failed', err);
      }}
    }}

    document.addEventListener('DOMContentLoaded', () => {{
      document.getElementById('manualRefresh').addEventListener('click', fetchStatus);
      document.getElementById('toggleAuto').addEventListener('click', () => {{
        auto = !auto;
        document.getElementById('autoLabel').textContent = auto ? 'ON' : 'OFF';
      }});
      setInterval(() => {{ if (auto) fetchStatus(); }}, REFRESH_MS);
      render(initialData);
    }});
  </script>
</body>
</html>
""".format(
                    refresh_ms=refresh_ms, initial_json=initial_json
                )

            def log_message(self, format: str, *args: object) -> None:  # type: ignore[override]
                return

        return Handler


__all__ = ["MonitorWebUIServer"]
