"""FastAPI web UI for monitoring optimization runs."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import textwrap
import time
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

try:  # pragma: no cover - optional dependency at runtime
    import psutil  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    psutil = None

from .events import EventSocketListener
from .paths import default_registry_path, default_socket_path
from .registry import ActiveRunDeletionError, RunRegistry
from .state import StorageInventory

logger = logging.getLogger(__name__)

if psutil is not None:  # pragma: no cover - best effort initialisation
    try:
        psutil.cpu_percent(interval=None)
    except Exception:  # pragma: no cover - ignore sensor issues
        pass


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


def _safe_memory_stats() -> dict[str, float] | None:
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
    return _fallback_memory_stats()


def _safe_disk_stats() -> list[dict[str, Any]]:
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
    return disks[:4]


def _collect_system_metrics() -> dict[str, Any]:
    memory = _safe_memory_stats() or {}
    disks = _safe_disk_stats()
    try:
        load_avg = os.getloadavg()
    except (AttributeError, OSError):
        load_avg = None
    snapshot = {
        "timestamp": time.time(),
        "cpu_percent": _safe_cpu_percent(),
        "cpu_count": os.cpu_count(),
        "memory": memory,
        "disks": disks,
    }
    if load_avg is not None:
        snapshot["load_avg"] = list(load_avg)
    return snapshot


HOME_HEAD = textwrap.dedent(
    """\
    <head>
      <meta charset="utf-8">
      <title>interFEBio Monitor</title>
      <style>
        :root {
          color-scheme: dark;
          --bg-color: #0f1526;
          --text-color: #f5f5f5;
          --muted-text: rgba(245, 245, 245, 0.6);
          --panel-bg: #1d2538;
          --card-bg: #232c42;
          --table-bg: #232c42;
          --table-header-bg: #2f3954;
          --table-row-alt: #1c2438;
          --table-row-hover: #314162;
          --link-color: #65b7ff;
          --chart-bg: #10172a;
          --chart-axis: rgba(245, 245, 245, 0.75);
          --chart-grid: rgba(255, 255, 255, 0.08);
          --chart-zero: rgba(255, 255, 255, 0.18);
          --json-bg: #111727;
          --param-value: #9bd4ff;
          --ghost-button-bg: rgba(255, 255, 255, 0.08);
          --ghost-button-border: rgba(255, 255, 255, 0.2);
          --cpu-line: #9b8bff;
          --mem-line: #34d399;
          --disk-used: #fb7185;
          --disk-free: #2dd4bf;
        }
        body.light {
          color-scheme: light;
          --bg-color: #f3f4f7;
          --text-color: #1f2430;
          --muted-text: rgba(31, 36, 48, 0.65);
          --panel-bg: #ffffff;
          --card-bg: #f7f9fc;
          --table-bg: #ffffff;
          --table-header-bg: #edf1f7;
          --table-row-alt: #f4f6fb;
          --table-row-hover: #e1e8f3;
          --link-color: #2f5aa8;
          --chart-bg: #ffffff;
          --chart-axis: rgba(31, 36, 48, 0.78);
          --chart-grid: rgba(100, 116, 139, 0.2);
          --chart-zero: rgba(15, 23, 42, 0.2);
          --json-bg: #f5f7fb;
          --param-value: #1b4f91;
          --ghost-button-bg: rgba(38, 84, 164, 0.1);
          --ghost-button-border: rgba(38, 84, 164, 0.35);
          --cpu-line: #5b21b6;
          --mem-line: #0f766e;
          --disk-used: #f97316;
          --disk-free: #059669;
        }
        * { box-sizing: border-box; }
        body { font-family: "Inter", Arial, sans-serif; margin: 1.5rem; background: var(--bg-color); color: var(--text-color); transition: background 0.2s ease, color 0.2s ease; }
        a { color: var(--link-color); text-decoration: none; }
        a:hover { text-decoration: underline; }
        .layout { display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: flex-start; }
        .panel { background: var(--panel-bg); border-radius: 12px; box-shadow: 0 10px 24px rgba(10, 18, 46, 0.18); padding: 1rem 1.25rem; flex: 1 1 360px; min-width: 320px; transition: background 0.2s ease, color 0.2s ease; }
        .panel h2 { margin-top: 0; font-size: 1.1rem; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.8; }
        .panel-header { display: flex; justify-content: space-between; align-items: center; gap: 0.75rem; margin-bottom: 0.6rem; }
        .panel-header h2 { margin: 0; }
        .collapse-toggle { margin-right: 0; padding: 0.3rem 0.9rem; font-size: 0.78rem; }
        .panel.collapsed .collapsible-body { display: none; }
        .panel.collapsed { padding-bottom: 0.6rem; }
        .runs-actions { display: flex; justify-content: space-between; align-items: center; gap: 0.8rem; flex-wrap: wrap; margin-bottom: 0.6rem; }
        .runs-actions .action-buttons { display: flex; gap: 0.5rem; flex-wrap: wrap; }
        .selection-toolbar { display: none; align-items: center; justify-content: space-between; gap: 0.75rem; padding: 0.45rem 0.75rem; border-radius: 999px; background: rgba(255,255,255,0.06); margin-bottom: 0.5rem; }
        body.light .selection-toolbar { background: rgba(31,36,48,0.12); }
        #runsPanel.selecting .selection-toolbar { display: flex; }
        .selection-toolbar strong { font-size: 0.85rem; letter-spacing: 0.02rem; text-transform: uppercase; }
        .selection-toolbar-actions { display: flex; gap: 0.45rem; flex-wrap: wrap; }
        .button-danger { background: linear-gradient(135deg, #f87171, #ef4444); }
        .button-danger:hover { box-shadow: 0 8px 16px rgba(239, 68, 68, 0.35); }
        table { border-collapse: collapse; width: 100%; background: var(--table-bg); border-radius: 8px; overflow: hidden; }
        th, td { padding: 0.6rem 0.8rem; text-align: left; font-size: 0.9rem; }
        th { background: var(--table-header-bg); text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05rem; }
        tr:nth-child(even) { background: var(--table-row-alt); }
        tr:hover { background: var(--table-row-hover); }
        .select-col, .select-cell { width: 36px; text-align: center; }
        #runsPanel:not(.selecting) .select-col,
        #runsPanel:not(.selecting) .select-cell { display: none; }
        .row-select { width: 22px; height: 22px; border-radius: 8px; border: 1px solid var(--ghost-button-border); background: transparent; cursor: pointer; display: inline-flex; align-items: center; justify-content: center; transition: background 0.2s ease, border-color 0.2s ease; }
        .row-select::after { content: ''; width: 10px; height: 10px; border-radius: 4px; background: transparent; transition: background 0.2s ease; }
        .row-select:hover { border-color: #94a3b8; }
        .row-select.selected { border-color: #f87171; background: rgba(248, 113, 113, 0.15); }
        .row-select.selected::after { background: #f87171; }
        #runsPanel.selecting tr { cursor: pointer; }
        #runsPanel.selecting tr.selected { background: rgba(56, 189, 248, 0.18); }
        #runsPanel.selecting .deleteBtn { display: none; }
        .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0, 0, 0, 0); border: 0; }
        .status { font-weight: 600; text-transform: uppercase; letter-spacing: 0.03rem; }
        .status.finished { color: #81d887; }
        .status.failed { color: #ff6b6b; }
        .status.running { color: #ffd166; }
        .status.created { color: #9bb4ff; }
        button { margin-right: 0.5rem; padding: 0.45rem 0.9rem; background: linear-gradient(135deg, #3a7bd5, #6f8fdb); color: #fff; border: none; border-radius: 999px; cursor: pointer; font-weight: 600; transition: transform 0.12s ease, box-shadow 0.12s ease; }
        button:hover { transform: translateY(-1px); box-shadow: 0 8px 16px rgba(58, 123, 213, 0.35); }
        .table-actions button { margin-right: 0; }
        .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
        .summary-card { background: var(--card-bg); border-radius: 10px; padding: 0.75rem 0.9rem; }
        .summary-card dt { font-size: 0.75rem; text-transform: uppercase; opacity: 0.6; margin-bottom: 0.25rem; letter-spacing: 0.04rem; }
        .summary-card dd { margin: 0; font-size: 1rem; font-weight: 600; }
        .chart-wrapper { background: var(--card-bg); border-radius: 10px; padding: 0.75rem 1rem; }
        .chart-wrapper h3 { margin: 0 0 0.5rem 0; font-size: 0.95rem; letter-spacing: 0.03rem; text-transform: uppercase; opacity: 0.75; }
        .chart { width: 100%; height: 420px; display: block; border-radius: 6px; background: var(--chart-bg); }
        .chart-sm { height: 300px; }
        .panel.full-width { flex: 1 1 100%; }
        .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }
        .disk-pies { display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.6rem; }
        .disk-pie { background: var(--card-bg); border-radius: 10px; padding: 0.65rem 0.8rem; flex: 1 1 200px; min-width: 160px; display: flex; flex-direction: column; align-items: center; }
        .disk-pie h4 { margin: 0 0 0.35rem 0; font-size: 0.85rem; letter-spacing: 0.04rem; text-transform: uppercase; opacity: 0.75; }
        .disk-pie p { margin: 0.2rem 0 0 0; font-size: 0.78rem; color: var(--muted-text); }
        .disk-chart { height: 160px; width: 160px; margin: 0 auto; display: flex; align-items: center; justify-content: center; overflow: visible; }
        .stat-pill { display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.2rem 0.7rem; border-radius: 999px; font-size: 0.78rem; background: rgba(255,255,255,0.08); color: var(--muted-text); }
        body.light .stat-pill { background: rgba(31,36,48,0.08); }
        pre { background: var(--json-bg); padding: 1rem; border-radius: 10px; overflow-x: auto; margin: 0; font-size: 0.85rem; }
        .param-grid { display: flex; flex-direction: column; gap: 0.2rem; font-family: "JetBrains Mono", "SFMono-Regular", Consolas, monospace; font-size: 0.85rem; }
        .param-row { display: flex; justify-content: space-between; gap: 0.5rem; }
        .param-row span:last-child { color: var(--param-value); }
        .status-reason { display: block; font-size: 0.8rem; color: #ff9b9b; margin-top: 0.15rem; line-height: 1.3; }
        .page-header { display: flex; justify-content: space-between; align-items: center; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem; }
        .page-header h1 { font-size: 1.6rem; margin: 0 0 0.35rem 0; }
        .page-header p { margin: 0; opacity: 0.65; }
        .ghost-button { margin-right: 0; padding: 0.35rem 0.9rem; background: var(--ghost-button-bg); color: var(--text-color); border: 1px solid var(--ghost-button-border); border-radius: 999px; font-weight: 600; cursor: pointer; transition: background 0.15s ease, color 0.15s ease; }
        .ghost-button:hover { background: rgba(255, 255, 255, 0.18); }
        body.light .ghost-button:hover { background: rgba(38, 84, 164, 0.2); }
        .series-select { background: var(--panel-bg); color: var(--text-color); border: 1px solid var(--ghost-button-border); border-radius: 6px; padding: 0.35rem 0.6rem; font-size: 0.9rem; transition: background 0.2s ease, color 0.2s ease; }
        @media (max-width: 960px) {
          body { margin: 1rem; }
          .layout { flex-direction: column; }
          .panel { width: 100%; min-width: 0; }
        }
      </style>
      <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    </head>
    """
)

HOME_BODY = textwrap.dedent(
    """\
    <body>
      <header class="page-header">
        <div>
          <h1>Optimization Runs</h1>
          <p>Monitor optimisation progress, inspect iteration history, and review run artefacts.</p>
        </div>
        <button id="themeToggle" class="ghost-button">Light mode</button>
      </header>
      <div class="layout">
        <section class="panel" id="runsPanel">
          <h2>Active Runs</h2>
          <div class="runs-actions">
            <div class="action-buttons">
              <button id="refresh">Refresh</button>
              <button id="deleteAll">Delete All</button>
            </div>
            <button id="toggleSelection" class="ghost-button" type="button">Select runs</button>
          </div>
          <div class="selection-toolbar" id="selectionToolbar" hidden>
            <strong id="selectionCount">0 selected</strong>
            <div class="selection-toolbar-actions">
              <button id="bulkDeleteSelected" class="button-danger" type="button" disabled>Delete selected</button>
              <button id="cancelSelection" class="ghost-button" type="button">Cancel</button>
            </div>
          </div>
          <table>
            <thead>
              <tr>
                <th class="select-col" aria-label="Select"></th>
                <th>Run</th>
                <th>Status</th>
                <th>Iterations</th>
                <th>Last Cost</th>
                <th>Updated</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="runRows">
              <tr><td colspan="7">Loading...</td></tr>
            </tbody>
          </table>
        </section>
        <section class="panel" id="detail" hidden>
          <div style="display:flex; justify-content:space-between; align-items:center; gap:0.75rem;">
            <h2 id="detailTitle" style="margin-bottom:0.4rem;">Run detail</h2>
            <button id="closeDetail" style="margin-right:0; background:rgba(99,115,255,0.25); padding:0.35rem 0.8rem;">Close</button>
          </div>
          <div class="detail-grid" id="detailSummary"></div>
          <div class="chart-wrapper">
            <h3>Cost per iteration</h3>
            <div id="costChart" class="chart"></div>
          </div>
          <div class="chart-wrapper">
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <h3 style="margin:0; font-size:0.95rem; letter-spacing:0.03rem; text-transform:uppercase; opacity:0.75;">Experiment vs Simulation</h3>
              <select id="seriesSelect" class="series-select"></select>
            </div>
            <div id="seriesChart" class="chart"></div>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1.1rem;">
            <h3 style="font-size:0.95rem; letter-spacing:0.03rem; text-transform:uppercase; opacity:0.75; margin:0;">Raw JSON</h3>
            <button id="toggleJson" class="ghost-button" style="margin-right:0;">Expand</button>
          </div>
          <pre id="detailJson" style="max-height:0; overflow:hidden; transition:max-height 0.18s ease; visibility:hidden;">{}</pre>
        </section>
        <section class="panel full-width" id="systemPanel">
          <div class="panel-header">
            <h2>Server Health</h2>
            <button
              id="toggleSystemPanel"
              class="ghost-button collapse-toggle"
              type="button"
              aria-controls="systemContent"
              aria-expanded="true"
            >Collapse</button>
          </div>
          <div class="stats-grid collapsible-body" id="systemContent">
            <div class="chart-wrapper">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <h3 style="margin:0;">CPU &amp; Memory</h3>
                <span class="stat-pill" id="systemSummary">--</span>
              </div>
              <div id="systemChart" class="chart chart-sm"></div>
            </div>
            <div class="chart-wrapper">
              <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.4rem;">
                <h3 style="margin:0;">Disk Utilisation</h3>
                <span class="stat-pill" id="diskSummary">--</span>
              </div>
              <div class="disk-pies" id="diskPies">
                <p style="margin:0; color:var(--muted-text);">Collecting disk information...</p>
              </div>
            </div>
          </div>
        </section>
      </div>
      <script>
        const rows = document.getElementById('runRows');
        const detail = document.getElementById('detail');
        const detailJson = document.getElementById('detailJson');
        const detailTitle = document.getElementById('detailTitle');
        const detailSummary = document.getElementById('detailSummary');
        const costChartEl = document.getElementById('costChart');
        const closeDetailBtn = document.getElementById('closeDetail');
        const toggleJsonBtn = document.getElementById('toggleJson');
        const seriesSelect = document.getElementById('seriesSelect');
        const seriesChartEl = document.getElementById('seriesChart');
        const systemChartEl = document.getElementById('systemChart');
        const diskPiesEl = document.getElementById('diskPies');
        const systemSummaryEl = document.getElementById('systemSummary');
        const diskSummaryEl = document.getElementById('diskSummary');
        const runsPanel = document.getElementById('runsPanel');
        const toggleSelectionBtn = document.getElementById('toggleSelection');
        const selectionToolbar = document.getElementById('selectionToolbar');
        const selectionCountEl = document.getElementById('selectionCount');
        const bulkDeleteBtn = document.getElementById('bulkDeleteSelected');
        const cancelSelectionBtn = document.getElementById('cancelSelection');
        let systemPanel = document.getElementById('systemPanel');
        let systemContent = document.getElementById('systemContent');
        let toggleSystemPanelBtn = document.getElementById('toggleSystemPanel');
        let jsonExpanded = false;
        let selectedRunId = null;
        let lastTableHtml = '';
        let currentSeriesData = {};
        let currentDetailData = null;
        let latestSystemSnapshot = null;
        const selectedRuns = new Set();
        let selectionMode = false;
        let systemPanelCollapsed = false;
        const systemHistory = [];
        const maxSystemPoints = 90;
        const themeStorageKey = 'interfebio-monitor-theme';
        const plotlyConfig = { displayModeBar: false, responsive: true };
        const themeToggleBtn = document.getElementById('themeToggle');
        function escapeHtml(value) {
          return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
        }
        function formatParameterValue(value) {
          if (typeof value === 'number' && Number.isFinite(value)) {
            return value.toExponential(4);
          }
          const numeric = Number(value);
          if (Number.isFinite(numeric)) {
            return numeric.toExponential(4);
          }
          if (value === undefined || value === null) {
            return '-';
          }
          return escapeHtml(String(value));
        }
        function formatBytes(value) {
          if (typeof value !== 'number' || !Number.isFinite(value)) {
            return '-';
          }
          const units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB'];
          let idx = 0;
          let num = value;
          while (num >= 1024 && idx < units.length - 1) {
            num /= 1024;
            idx += 1;
          }
          const precision = num >= 100 || idx === 0 ? 0 : 1;
          return `${num.toFixed(precision)} ${units[idx]}`;
        }
        function latestThetaSnapshot(data) {
          const iterations = Array.isArray(data?.iterations) ? data.iterations : [];
          for (let i = iterations.length - 1; i >= 0; i -= 1) {
            const entry = iterations[i];
            if (entry && entry.theta && typeof entry.theta === 'object' && Object.keys(entry.theta).length) {
              return entry.theta;
            }
          }
          return null;
        }
        function extractLatestSeries(data) {
          const iterations = Array.isArray(data?.iterations) ? data.iterations : [];
          for (let idx = iterations.length - 1; idx >= 0; idx -= 1) {
            const entry = iterations[idx];
            if (entry && entry.series && typeof entry.series === 'object' && Object.keys(entry.series).length) {
              return entry.series;
            }
          }
          return {};
        }
        function getCssColor(name, fallback) {
          const value = getComputedStyle(document.body).getPropertyValue(name);
          return value && value.trim().length ? value.trim() : fallback;
        }
        function chartColors() {
          return {
            bg: getCssColor('--chart-bg', '#10172a'),
            card: getCssColor('--card-bg', '#232c42'),
            text: getCssColor('--chart-axis', '#f5f5f5'),
            grid: getCssColor('--chart-grid', 'rgba(255,255,255,0.08)'),
            zero: getCssColor('--chart-zero', 'rgba(255,255,255,0.18)'),
            muted: getCssColor('--muted-text', 'rgba(255,255,255,0.5)'),
            cpu: getCssColor('--cpu-line', '#8b5cf6'),
            memory: getCssColor('--mem-line', '#34d399'),
            diskUsed: getCssColor('--disk-used', '#fb7185'),
            diskFree: getCssColor('--disk-free', '#2dd4bf'),
          };
        }
        function updateThemeButton() {
          if (!themeToggleBtn) {
            return;
          }
          const isLight = document.body.classList.contains('light');
          themeToggleBtn.textContent = isLight ? 'Dark mode' : 'Light mode';
        }
        function refreshThemeDependentViews() {
          if (currentDetailData) {
            renderDetailPanels(currentDetailData, { refreshJson: false });
          } else {
            renderEmptyPlot(costChartEl, 'Select a run');
            renderEmptyPlot(seriesChartEl, 'Select a run');
          }
          renderSystemUsage();
          renderDiskPies();
        }
        function applyTheme(theme, { skipStorage = false } = {}) {
          const isLight = theme === 'light';
          document.body.classList.toggle('light', isLight);
          if (!skipStorage) {
            try {
              localStorage.setItem(themeStorageKey, isLight ? 'light' : 'dark');
            } catch (err) {
              // ignore storage errors
            }
          }
          updateThemeButton();
          refreshThemeDependentViews();
        }
        (function initTheme() {
          let saved = null;
          try {
            saved = localStorage.getItem(themeStorageKey);
          } catch (err) {
            saved = null;
          }
          applyTheme(saved === 'light' ? 'light' : 'dark', { skipStorage: true });
          if (themeToggleBtn) {
            themeToggleBtn.addEventListener('click', () => {
              const nextTheme = document.body.classList.contains('light') ? 'dark' : 'light';
              applyTheme(nextTheme);
            });
          }
        })();
        function hasPlotly() {
          return typeof Plotly !== 'undefined' && Plotly.react;
        }
        function renderEmptyPlot(container, message, height = 420) {
          if (!container) {
            return;
          }
          const colors = chartColors();
          if (!hasPlotly()) {
            container.innerHTML = `<div style="display:flex; align-items:center; justify-content:center; height:100%; color:${colors.muted}; font-size:16px;">${message}</div>`;
            return;
          }
          Plotly.react(
            container,
            [],
            {
              paper_bgcolor: colors.bg,
              plot_bgcolor: colors.bg,
              autosize: true,
              height,
              margin: { l: 0, r: 0, t: 0, b: 0 },
              xaxis: { visible: false },
              yaxis: { visible: false },
              annotations: [
                {
                  text: message,
                  x: 0.5,
                  y: 0.5,
                  xref: 'paper',
                  yref: 'paper',
                  showarrow: false,
                  font: { color: colors.muted, size: 16 },
                },
              ],
            },
            plotlyConfig,
          );
          if (typeof Plotly !== 'undefined' && Plotly.Plots && typeof Plotly.Plots.resize === 'function') {
            Plotly.Plots.resize(container);
          }
        }
        function refreshSelectionStyles() {
          if (!rows) {
            return;
          }
          const rowNodes = rows.querySelectorAll('tr[data-run]');
          rowNodes.forEach((row) => {
            const runId = row.dataset.run;
            const isSelected = runId && selectedRuns.has(runId);
            row.classList.toggle('selected', Boolean(isSelected && selectionMode));
            const toggleBtn = row.querySelector('.row-select');
            if (toggleBtn) {
              const pressed = Boolean(isSelected && selectionMode);
              toggleBtn.classList.toggle('selected', pressed);
              toggleBtn.setAttribute('aria-pressed', pressed ? 'true' : 'false');
            }
          });
        }
        function updateSelectionToolbar() {
          if (runsPanel) {
            runsPanel.classList.toggle('selecting', selectionMode);
          }
          if (selectionToolbar) {
            selectionToolbar.hidden = !selectionMode;
          }
          if (selectionCountEl) {
            selectionCountEl.textContent = `${selectedRuns.size} selected`;
          }
          if (bulkDeleteBtn) {
            bulkDeleteBtn.disabled = !selectionMode || !selectedRuns.size;
          }
          if (toggleSelectionBtn) {
            toggleSelectionBtn.textContent = selectionMode ? 'Done selecting' : 'Select runs';
          }
        }
        function enterSelectionMode() {
          if (selectionMode) {
            return;
          }
          selectionMode = true;
          selectedRuns.clear();
          updateSelectionToolbar();
          refreshSelectionStyles();
        }
        function exitSelectionMode() {
          if (!selectionMode) {
            return;
          }
          selectionMode = false;
          selectedRuns.clear();
          updateSelectionToolbar();
          refreshSelectionStyles();
        }
        function toggleRunSelection(runId) {
          if (!selectionMode || !runId) {
            return;
          }
          if (selectedRuns.has(runId)) {
            selectedRuns.delete(runId);
          } else {
            selectedRuns.add(runId);
          }
          refreshSelectionStyles();
          updateSelectionToolbar();
        }
        function ensureSystemPanelElements() {
          if (!systemPanel) {
            systemPanel = document.getElementById('systemPanel');
          }
          if (!systemContent) {
            systemContent = document.getElementById('systemContent');
          }
          if (!toggleSystemPanelBtn) {
            toggleSystemPanelBtn = document.getElementById('toggleSystemPanel');
          }
        }
        function setSystemPanelCollapsed(collapsed, { skipRender = false } = {}) {
          ensureSystemPanelElements();
          systemPanelCollapsed = Boolean(collapsed);
          if (systemContent) {
            systemContent.hidden = systemPanelCollapsed;
            systemContent.setAttribute('aria-hidden', systemPanelCollapsed ? 'true' : 'false');
          }
          if (systemPanel) {
            systemPanel.classList.toggle('collapsed', systemPanelCollapsed);
          }
          if (toggleSystemPanelBtn) {
            toggleSystemPanelBtn.textContent = systemPanelCollapsed ? 'Expand' : 'Collapse';
            toggleSystemPanelBtn.setAttribute('aria-expanded', systemPanelCollapsed ? 'false' : 'true');
          }
          if (!systemPanelCollapsed && !skipRender) {
            if (hasPlotly() && systemChartEl && Plotly.Plots && typeof Plotly.Plots.resize === 'function') {
              try {
                Plotly.Plots.resize(systemChartEl);
              } catch (err) {
                // ignore resize errors
              }
            }
            renderSystemUsage();
            renderDiskPies();
          }
        }
        function handleSystemPanelToggle(event) {
          if (event) {
            if (typeof event.preventDefault === 'function') {
              event.preventDefault();
            }
            if (typeof event.stopPropagation === 'function') {
              event.stopPropagation();
            }
          }
          setSystemPanelCollapsed(!systemPanelCollapsed);
        }
        function pushSystemHistory(snapshot) {
          if (!snapshot || typeof snapshot.timestamp !== 'number') {
            return;
          }
          systemHistory.push({
            time: new Date(snapshot.timestamp * 1000),
            cpu: typeof snapshot.cpu_percent === 'number' ? snapshot.cpu_percent : null,
            memory:
              snapshot && snapshot.memory && typeof snapshot.memory.percent === 'number'
                ? snapshot.memory.percent
                : null,
          });
          while (systemHistory.length > maxSystemPoints) {
            systemHistory.shift();
          }
        }
        function updateSystemSummary(snapshot) {
          if (!systemSummaryEl) {
            return;
          }
          const cpuText = snapshot && typeof snapshot.cpu_percent === 'number'
            ? `${snapshot.cpu_percent.toFixed(1)}% CPU`
            : 'CPU N/A';
          const memPercent = snapshot && snapshot.memory && typeof snapshot.memory.percent === 'number'
            ? `${snapshot.memory.percent.toFixed(1)}% RAM`
            : 'RAM N/A';
          systemSummaryEl.textContent = `${cpuText} • ${memPercent}`;
        }
        function renderSystemUsage() {
          if (!systemChartEl || systemPanelCollapsed) {
            return;
          }
          if (!systemHistory.length) {
            renderEmptyPlot(systemChartEl, 'Collecting metrics...', 300);
            return;
          }
          if (!hasPlotly()) {
            renderEmptyPlot(systemChartEl, 'Plotly.js not available', 300);
            return;
          }
          const scrollElement = document.scrollingElement || document.documentElement || document.body;
          let prevScrollTop = 0;
          if (scrollElement) {
            prevScrollTop = scrollElement.scrollTop;
          } else if (typeof window !== 'undefined') {
            prevScrollTop = window.scrollY || window.pageYOffset || 0;
          }
          const colors = chartColors();
          const times = systemHistory.map((entry) => entry.time);
          const cpuValues = systemHistory.map((entry) => (typeof entry.cpu === 'number' ? entry.cpu : null));
          const memValues = systemHistory.map((entry) =>
            typeof entry.memory === 'number' ? entry.memory : null,
          );
          const cpuTrace = {
            x: times,
            y: cpuValues,
            name: 'CPU %',
            mode: 'lines',
            line: { color: colors.cpu, width: 3 },
            hovertemplate: 'CPU %{y:.1f}%<extra></extra>',
          };
          const memTrace = {
            x: times,
            y: memValues,
            name: 'RAM %',
            mode: 'lines',
            line: { color: colors.memory, width: 3 },
            hovertemplate: 'RAM %{y:.1f}%<extra></extra>',
          };
          const layout = {
            paper_bgcolor: colors.bg,
            plot_bgcolor: colors.bg,
            autosize: true,
            height: 300,
            margin: { l: 40, r: 8, t: 10, b: 40 },
            xaxis: {
              title: 'Time',
              color: colors.text,
              gridcolor: colors.grid,
              showgrid: false,
            },
            yaxis: {
              title: 'Percent',
              color: colors.text,
              gridcolor: colors.grid,
              zerolinecolor: colors.zero,
              range: [0, 100],
            },
            legend: { orientation: 'h', yanchor: 'bottom', y: 1.02, x: 0, font: { color: colors.text } },
          };
          Plotly.react(systemChartEl, [cpuTrace, memTrace], layout, plotlyConfig);
          if (typeof Plotly !== 'undefined' && Plotly.Plots && typeof Plotly.Plots.resize === 'function') {
            Plotly.Plots.resize(systemChartEl);
          }
          if (scrollElement) {
            scrollElement.scrollTop = prevScrollTop;
          } else if (typeof window !== 'undefined' && typeof window.scrollTo === 'function') {
            window.scrollTo(0, prevScrollTop);
          }
        }
        function renderDiskPies() {
          if (!diskPiesEl) {
            return;
          }
          const disks = latestSystemSnapshot && Array.isArray(latestSystemSnapshot.disks)
            ? latestSystemSnapshot.disks
            : [];
          if (!disks.length) {
            diskPiesEl.innerHTML = '<p style="margin:0; color:var(--muted-text);">No disk data.</p>';
            if (diskSummaryEl) {
              diskSummaryEl.textContent = 'No disks';
            }
            return;
          }
          let totalBytes = 0;
          let freeBytes = 0;
          const diskData = disks.map((disk, idx) => {
            const used = typeof disk.used === 'number' ? disk.used : 0;
            const available = typeof disk.free === 'number' ? disk.free : 0;
            const diskTotal = typeof disk.total === 'number' ? disk.total : used + available;
            totalBytes += diskTotal;
            freeBytes += available;
            return { disk, idx, used, available, diskTotal };
          });
          if (diskSummaryEl) {
            diskSummaryEl.textContent = `${formatBytes(freeBytes)} free / ${formatBytes(totalBytes)}`;
          }
          if (systemPanelCollapsed) {
            return;
          }
          if (hasPlotly()) {
            const prevCharts = diskPiesEl.querySelectorAll('.disk-chart');
            prevCharts.forEach((node) => {
              try {
                Plotly.purge(node);
              } catch (err) {
                // ignore purge errors
              }
            });
          }
          diskPiesEl.innerHTML = '';
          const colors = chartColors();
          const cardBg = colors.card || colors.bg;
          diskData.forEach(({ disk, idx, used, available, diskTotal }) => {
            const card = document.createElement('div');
            card.className = 'disk-pie';
            const heading = document.createElement('h4');
            heading.textContent = disk.mount || disk.device || `Disk ${idx + 1}`;
            card.appendChild(heading);
            const chart = document.createElement('div');
            chart.className = 'disk-chart';
            card.appendChild(chart);
            const percentLabel = typeof disk.percent === 'number' ? disk.percent.toFixed(1) : '0.0';
            const caption = document.createElement('p');
            caption.textContent = `${formatBytes(available)} free of ${formatBytes(diskTotal)} (${percentLabel}% used)`;
            card.appendChild(caption);
            diskPiesEl.appendChild(card);
            if (!hasPlotly()) {
              chart.innerHTML = '<div style="display:flex; height:100%; align-items:center; justify-content:center; color:var(--muted-text); font-size:0.9rem;">Plotly.js not available</div>';
              return;
            }
            const usedValue = used < 0 ? 0 : used;
            const freeValue = available < 0 ? 0 : available;
            const values = usedValue + freeValue > 0 ? [usedValue, freeValue] : [1, 0];
            const annotationText = `${percentLabel}%<br>used`;
            Plotly.react(
              chart,
              [
                {
                  values,
                  labels: ['Used', 'Free'],
                  type: 'pie',
                  hole: 0.65,
                  marker: { colors: [colors.diskUsed, colors.diskFree] },
                  textinfo: 'none',
                  hovertemplate: '%{label}: %{value:.2f} B (%{percent})<extra></extra>',
                },
              ],
              {
                paper_bgcolor: cardBg,
                plot_bgcolor: cardBg,
                height: 160,
                width: 160,
                margin: { l: 6, r: 6, t: 6, b: 6 },
                showlegend: false,
                annotations: [
                  {
                    text: annotationText,
                    x: 0.5,
                    y: 0.5,
                    font: { color: colors.text, size: 13 },
                    showarrow: false,
                    xref: 'paper',
                    yref: 'paper',
                    align: 'center',
                  },
                ],
              },
              plotlyConfig,
            );
          });
        }
        async function loadSystemMetrics(initial = false) {
          if (!systemChartEl && !diskPiesEl) {
            return;
          }
          try {
            const res = await fetch('/api/system/metrics');
            if (!res.ok) {
              if (initial && systemChartEl) {
                renderEmptyPlot(systemChartEl, 'Unable to load metrics', 300);
              }
              return;
            }
            const data = await res.json();
            latestSystemSnapshot = data;
            pushSystemHistory(data);
            updateSystemSummary(data);
            renderSystemUsage();
            renderDiskPies();
          } catch (err) {
            if (initial && systemChartEl) {
              renderEmptyPlot(systemChartEl, 'Unable to load metrics', 300);
            }
            console.error(err);
          }
        }
        if (seriesSelect) {
          seriesSelect.disabled = true;
        }

        async function loadRuns(showLoading = true) {
          if (showLoading && !lastTableHtml) {
            rows.innerHTML = "<tr><td colspan='7'>Loading...</td></tr>";
          }
          try {
            const res = await fetch('/api/runs', { cache: 'no-store' });
            const data = await res.json();
            const activeIds = new Set(data.map(run => run.run_id));
            Array.from(selectedRuns).forEach((runId) => {
              if (!activeIds.has(runId)) {
                selectedRuns.delete(runId);
              }
            });
            let html;
            if (!data.length) {
              html = "<tr><td colspan='7'>No runs yet</td></tr>";
            } else {
              html = data
                .map(run => {
                  const status = run.status || 'unknown';
                  const label = run.label || run.run_id;
                  const last = typeof run.last_cost === 'number' ? run.last_cost.toExponential(3) : '-';
                  const updated = run.updated_at ? new Date(run.updated_at * 1000).toLocaleTimeString() : '-';
                  const safeLabel = escapeHtml(label);
                  const isSelected = selectedRuns.has(run.run_id);
                  const selectionClass = selectionMode && isSelected ? ' class="selected"' : '';
                  const pressed = selectionMode && isSelected ? 'true' : 'false';
                  return `<tr data-run="${run.run_id}"${selectionClass}>
                    <td class="select-cell">
                      <button type="button" class="row-select${isSelected ? ' selected' : ''}" data-run="${run.run_id}" aria-pressed="${pressed}" aria-label="${isSelected ? 'Deselect' : 'Select'} ${safeLabel}"></button>
                      <span class="sr-only">${isSelected ? 'Selected' : 'Not selected'}</span>
                    </td>
                    <td><a href="#" data-run="${run.run_id}">${safeLabel}</a></td>
                    <td class="status ${status}">${status}</td>
                    <td>${run.iteration_count}</td>
                    <td>${last}</td>
                    <td>${updated}</td>
                    <td><button class="deleteBtn" data-run="${run.run_id}">Delete</button></td>
                  </tr>`;
                })
                .join('');
            }
            if (html !== lastTableHtml) {
              rows.innerHTML = html;
              lastTableHtml = html;
              refreshSelectionStyles();
            }
            updateSelectionToolbar();
            if (selectedRunId) {
              const exists = data.some(run => run.run_id === selectedRunId);
              if (exists) {
                await refreshDetail(selectedRunId, { silent: true });
              } else {
                closeDetail();
              }
            }
          } catch (err) {
            console.error(err);
          }
        }

        async function refreshDetail(runId, { silent = false } = {}) {
          if (!silent) {
            detailSummary.innerHTML = "<p style='opacity:0.6;'>Loading...</p>";
            renderEmptyPlot(costChartEl, 'Loading...');
            renderEmptyPlot(seriesChartEl, 'Loading...');
          }
          try {
            const res = await fetch(`/api/runs/${runId}`, { cache: 'no-store' });
            if (!res.ok) {
              throw new Error('Failed to load run details');
            }
            const data = await res.json();
            currentDetailData = data;
            renderDetailPanels(data);
          } catch (err) {
            console.error(err);
            detailSummary.innerHTML = "<p style='color:#ff6b6b;'>Failed to load run details.</p>";
            renderEmptyPlot(costChartEl, 'Failed to load data');
            renderEmptyPlot(seriesChartEl, 'Failed to load data');
          }
        }

        function renderDetailPanels(data, { refreshJson = true } = {}) {
          if (!data) {
            return;
          }
          detail.hidden = false;
          detailTitle.textContent = data.label || data.run_id || 'Run detail';
          renderSummary(data);
          renderIterations(data);
          renderSeries(extractLatestSeries(data));
          if (detailJson) {
            if (refreshJson) {
              const resetJson = detailJson.textContent === '' || detailJson.textContent === '{}';
              const text = JSON.stringify(data, null, 2);
              if (detailJson.textContent !== text) {
                detailJson.textContent = text;
              }
              if (resetJson) {
                collapseJson();
              } else if (jsonExpanded) {
                detailJson.style.maxHeight = '320px';
                detailJson.style.overflowY = 'auto';
                detailJson.style.visibility = 'visible';
              }
            } else if (jsonExpanded) {
              detailJson.style.maxHeight = '320px';
              detailJson.style.overflowY = 'auto';
              detailJson.style.visibility = 'visible';
            } else {
              detailJson.style.maxHeight = '0';
              detailJson.style.overflowY = 'hidden';
              detailJson.style.visibility = 'hidden';
            }
          }
        }

        async function showDetail(runId) {
          selectedRunId = runId;
          await refreshDetail(runId);
        }

        function renderSummary(data) {
          const summary = [];
          const meta = data.meta || {};
          const statusRaw = typeof data.status === 'string' ? data.status : 'unknown';
          const statusLower = statusRaw.toLowerCase();
          let statusValue = escapeHtml(statusRaw);
          const failureReason = typeof meta.failure_reason === 'string' ? meta.failure_reason : null;
          if (statusLower === 'failed' && failureReason) {
            statusValue += `<span class="status-reason">${escapeHtml(failureReason)}</span>`;
          }
          summary.push({ label: 'Status', value: statusValue });
          if (typeof meta.last_cost === 'number') {
            summary.push({ label: 'Last cost', value: meta.last_cost.toExponential(4) });
          }
          if (meta.r_squared && typeof meta.r_squared === 'object') {
            const entries = Object.entries(meta.r_squared)
              .map(([key, value]) => {
                const safeKey = escapeHtml(key);
                if (value === null || value === undefined) {
                  return `${safeKey}: -`;
                }
                const num = Number(value);
                if (!Number.isFinite(num)) {
                  return `${safeKey}: -`;
                }
                return `${safeKey}: ${num.toFixed(4)}`;
              });
            if (entries.length) {
              summary.push({ label: 'R²', value: entries.join('<br>') });
            }
          }
          if (meta.optimizer) {
            let label = meta.optimizer;
            if (typeof meta.optimizer === 'object') {
              const values = Object.values(meta.optimizer);
              if (values.length === 1) {
                label = values[0];
              } else {
                label = JSON.stringify(meta.optimizer);
              }
            }
            const safeLabel = typeof label === 'string' ? escapeHtml(label) : escapeHtml(String(label));
            summary.push({ label: 'Optimizer', value: safeLabel });
          }
          const paramCard = buildParameterCard(data);
          if (paramCard) {
            summary.push(paramCard);
          }
          detailSummary.innerHTML = summary.map(item => `
            <dl class="summary-card">
              <dt>${item.label}</dt>
              <dd>${item.value}</dd>
            </dl>`).join('') || "<p style='opacity:0.6;'>No summary metadata</p>";
        }

        function buildParameterCard(data) {
          if (!data) {
            return null;
          }
          const parameterMeta = data.parameters || {};
          const names = Array.isArray(parameterMeta.names)
            ? parameterMeta.names.filter(name => typeof name === 'string')
            : [];
          const theta0 = Array.isArray(parameterMeta.theta0) ? parameterMeta.theta0 : null;
          const latestTheta = latestThetaSnapshot(data);
          const rows = [];
          const seen = new Set();
          names.forEach((name, idx) => {
            let value;
            if (latestTheta && Object.prototype.hasOwnProperty.call(latestTheta, name)) {
              value = latestTheta[name];
            } else if (theta0 && theta0[idx] !== undefined) {
              value = theta0[idx];
            }
            if (value !== undefined) {
              rows.push(`<div class="param-row"><span>${escapeHtml(name)}</span><span>${formatParameterValue(value)}</span></div>`);
              seen.add(name);
            }
          });
          if (latestTheta) {
            Object.entries(latestTheta).forEach(([name, value]) => {
              if (typeof name === 'string' && !seen.has(name)) {
                rows.push(`<div class="param-row"><span>${escapeHtml(name)}</span><span>${formatParameterValue(value)}</span></div>`);
              }
            });
          }
          if (rows.length) {
            return { label: 'Parameters', value: `<div class="param-grid">${rows.join('')}</div>` };
          }
          if (names.length) {
            return { label: 'Parameters', value: names.map(escapeHtml).join(', ') };
          }
          return null;
        }

        function renderIterations(data) {
          renderCostChart(Array.isArray(data.iterations) ? data.iterations : []);
        }

        function renderCostChart(iterations) {
          if (!costChartEl) {
            return;
          }
          if (!iterations.length) {
            renderEmptyPlot(costChartEl, 'No iteration data');
            return;
          }
          const points = iterations
            .map((it, idx) => {
              const cost = Number(it.cost);
              return Number.isFinite(cost) ? { idx, cost } : null;
            })
            .filter(Boolean);
          if (!points.length) {
            renderEmptyPlot(costChartEl, 'No valid cost values');
            return;
          }
          if (!hasPlotly()) {
            renderEmptyPlot(costChartEl, 'Plotly.js not available');
            return;
          }
          const colors = chartColors();
          const tickStep = points.length > 1 ? Math.max(1, Math.ceil(points.length / 10)) : 1;
          const trace = {
            x: points.map(item => item.idx),
            y: points.map(item => item.cost),
            mode: 'lines+markers',
            name: 'Cost',
            marker: { color: '#6ec1ff', size: 6 },
            line: { color: '#6ec1ff', width: 2.5 },
            hovertemplate: 'Iteration %{x}<br>Cost %{y:.4e}<extra></extra>',
          };
          const layout = {
            paper_bgcolor: colors.bg,
            plot_bgcolor: colors.bg,
            autosize: true,
            height: 420,
            margin: { l: 64, r: 28, t: 32, b: 64 },
            font: { color: colors.text },
            xaxis: {
              title: { text: 'Iteration' },
              dtick: tickStep,
              color: colors.text,
              gridcolor: colors.grid,
              zerolinecolor: colors.zero,
            },
            yaxis: {
              title: { text: 'Cost' },
              color: colors.text,
              gridcolor: colors.grid,
              zerolinecolor: colors.zero,
              tickformat: '.2e',
            },
            hovermode: 'closest',
            uirevision: 'cost-chart',
          };
          Plotly.react(costChartEl, [trace], layout, plotlyConfig);
          if (typeof Plotly !== 'undefined' && Plotly.Plots && typeof Plotly.Plots.resize === 'function') {
            Plotly.Plots.resize(costChartEl);
          }
        }

        function collapseJson() {
          jsonExpanded = false;
          toggleJsonBtn.textContent = 'Expand';
          detailJson.style.maxHeight = '0';
          detailJson.style.overflowY = 'hidden';
          detailJson.style.visibility = 'hidden';
        }

        function closeDetail() {
          detail.hidden = true;
          detailJson.textContent = '';
          detailSummary.innerHTML = '';
          jsonExpanded = false;
          toggleJsonBtn.textContent = 'Expand';
          detailJson.style.maxHeight = '0';
          detailJson.style.overflowY = 'hidden';
          detailJson.style.visibility = 'hidden';
          selectedRunId = null;
          currentSeriesData = {};
           currentDetailData = null;
          if (seriesSelect) {
            seriesSelect.innerHTML = '';
            seriesSelect.disabled = true;
          }
          if (hasPlotly()) {
            if (costChartEl) {
              Plotly.purge(costChartEl);
            }
            if (seriesChartEl) {
              Plotly.purge(seriesChartEl);
            }
          }
          if (seriesChartEl) {
            seriesChartEl.innerHTML = '';
          }
          if (costChartEl) {
            costChartEl.innerHTML = '';
          }
        }

        function renderSeries(series) {
          currentSeriesData = series || {};
          if (!seriesSelect || !seriesChartEl) {
            return;
          }
          const keys = Object.keys(currentSeriesData);
          if (!keys.length) {
            seriesSelect.innerHTML = '<option>No data</option>';
            seriesSelect.disabled = true;
            renderEmptyPlot(seriesChartEl, 'No series data');
            return;
          }
          seriesSelect.disabled = false;
          const desiredValue =
            !seriesSelect.value || !currentSeriesData[seriesSelect.value]
              ? keys[0]
              : seriesSelect.value;
          seriesSelect.innerHTML = keys
            .map(key => `<option value="${key}" ${key === desiredValue ? 'selected' : ''}>${key}</option>`)
            .join('');
          seriesSelect.value = desiredValue;
          renderSeriesChart(desiredValue);
        }

        function renderSeriesChart(key) {
          if (!seriesChartEl) {
            return;
          }
          const dataset = currentSeriesData && key ? currentSeriesData[key] : null;
          if (!dataset) {
            renderEmptyPlot(seriesChartEl, 'No series data');
            return;
          }
          const xRaw = Array.isArray(dataset.x) ? dataset.x : [];
          const expRaw = Array.isArray(dataset.y_exp) ? dataset.y_exp : [];
          const simRaw = Array.isArray(dataset.y_sim) ? dataset.y_sim : [];
          const length = Math.min(xRaw.length, expRaw.length, simRaw.length);
          const points = [];
          for (let i = 0; i < length; i++) {
            const xVal = Number(xRaw[i]);
            const expVal = Number(expRaw[i]);
            const simVal = Number(simRaw[i]);
            if (Number.isFinite(xVal) && Number.isFinite(expVal) && Number.isFinite(simVal)) {
              points.push({ x: xVal, exp: expVal, sim: simVal });
            }
          }
          if (!points.length) {
            renderEmptyPlot(seriesChartEl, 'No series data');
            return;
          }
          if (!hasPlotly()) {
            renderEmptyPlot(seriesChartEl, 'Plotly.js not available');
            return;
          }
          const colors = chartColors();
          const traces = [
            {
              x: points.map(p => p.x),
              y: points.map(p => p.exp),
              mode: 'markers',
              name: 'Experimental',
              marker: { color: '#ffb347', size: 7 },
              hovertemplate: 'x %{x}<br>Experimental %{y:.4e}<extra></extra>',
            },
            {
              x: points.map(p => p.x),
              y: points.map(p => p.sim),
              mode: 'lines',
              name: 'Simulation',
              line: { color: '#6ec1ff', width: 3 },
              hovertemplate: 'x %{x}<br>Simulation %{y:.4e}<extra></extra>',
            },
          ];
          const layout = {
            paper_bgcolor: colors.bg,
            plot_bgcolor: colors.bg,
            autosize: true,
            height: 420,
            margin: { l: 64, r: 28, t: 32, b: 64 },
            font: { color: colors.text },
            xaxis: {
              title: { text: 'x' },
              color: colors.text,
              gridcolor: colors.grid,
              zerolinecolor: colors.zero,
            },
            yaxis: {
              title: { text: 'y' },
              color: colors.text,
              gridcolor: colors.grid,
              zerolinecolor: colors.zero,
            },
            legend: {
              orientation: 'h',
              yanchor: 'bottom',
              y: 1.02,
              xanchor: 'left',
              x: 0,
            },
            hovermode: 'closest',
            uirevision: 'series-chart',
          };
          Plotly.react(seriesChartEl, traces, layout, plotlyConfig);
          if (typeof Plotly !== 'undefined' && Plotly.Plots && typeof Plotly.Plots.resize === 'function') {
            Plotly.Plots.resize(seriesChartEl);
          }
        }

        if (seriesSelect) {
          seriesSelect.addEventListener('change', (event) => {
            renderSeriesChart(event.target.value);
          });
        }
        document.getElementById('refresh').addEventListener('click', () => loadRuns());
        document.getElementById('deleteAll').addEventListener('click', () => deleteAllRuns());
        closeDetailBtn.addEventListener('click', () => {
          closeDetail();
        });
        toggleJsonBtn.addEventListener('click', () => {
          jsonExpanded = !jsonExpanded;
          if (jsonExpanded) {
            toggleJsonBtn.textContent = 'Collapse';
            detailJson.style.maxHeight = '320px';
            detailJson.style.overflowY = 'auto';
            detailJson.style.visibility = 'visible';
          } else {
            collapseJson();
          }
        });
        ensureSystemPanelElements();
        if (toggleSystemPanelBtn && systemPanel && systemContent) {
          toggleSystemPanelBtn.addEventListener('click', handleSystemPanelToggle);
        }
        document.addEventListener('click', (event) => {
          const target = event && event.target && typeof event.target.closest === 'function'
            ? event.target.closest('#toggleSystemPanel')
            : null;
          if (!target) {
            return;
          }
          handleSystemPanelToggle(event);
        });
        setSystemPanelCollapsed(false, { skipRender: true });
        if (toggleSelectionBtn) {
          toggleSelectionBtn.addEventListener('click', () => {
            if (selectionMode) {
              exitSelectionMode();
            } else {
              enterSelectionMode();
            }
          });
        }
        if (cancelSelectionBtn) {
          cancelSelectionBtn.addEventListener('click', () => {
            exitSelectionMode();
          });
        }
        if (bulkDeleteBtn) {
          bulkDeleteBtn.addEventListener('click', async () => {
            await deleteSelectedRuns();
          });
        }
        updateSelectionToolbar();

        async function deleteRun(runId, force = false, skipConfirm = false) {
          if (!skipConfirm) {
            const msg = force
              ? `Force delete run ${runId}? Only use this if the monitor cannot determine its state.`
              : `Delete run ${runId}?`;
            if (!confirm(msg)) {
              return;
            }
          }
          try {
            const endpoint = force ? `/api/runs/${runId}?force=1` : `/api/runs/${runId}`;
            const res = await fetch(endpoint, { method: 'DELETE' });
            if (res.status === 409 && !force) {
              const err = await res.json().catch(() => ({}));
              const confirmForce = confirm(
                `${err.detail || 'Run appears to be active.'}\nForce delete ${runId}?`,
              );
              if (confirmForce) {
                await deleteRun(runId, true, true);
              }
              return;
            }
            if (!res.ok) {
              const err = await res.json().catch(() => ({}));
              alert(err.detail || 'Failed to delete run');
              return;
            }
            if (selectedRunId === runId) {
              closeDetail();
            }
            await loadRuns(false);
          } catch (err) {
            alert('Failed to delete run');
            console.error(err);
          }
        }
        async function deleteSelectedRuns() {
          if (!selectionMode || !selectedRuns.size) {
            return;
          }
          const confirmBulk = confirm(`Delete ${selectedRuns.size} run(s)?`);
          if (!confirmBulk) {
            return;
          }
          let previousLabel = null;
          if (bulkDeleteBtn) {
            previousLabel = bulkDeleteBtn.textContent;
            bulkDeleteBtn.disabled = true;
            bulkDeleteBtn.textContent = 'Deleting...';
          }
          const targets = Array.from(selectedRuns);
          for (const runId of targets) {
            try {
              await deleteRun(runId, false, true);
            } catch (err) {
              console.error(`Failed to delete run ${runId}`, err);
            }
          }
          if (bulkDeleteBtn && previousLabel !== null) {
            bulkDeleteBtn.textContent = previousLabel;
            bulkDeleteBtn.disabled = false;
          }
          exitSelectionMode();
          loadRuns();
        }
        async function deleteAllRuns(force = false, skipConfirm = false) {
          if (!skipConfirm) {
            const msg = force
              ? 'Force delete all runs? This will also remove entries still marked as running.'
              : 'Delete all completed runs? This cannot be undone.';
            if (!confirm(msg)) {
              return;
            }
          }
          try {
            const endpoint = force ? '/api/runs?force=1' : '/api/runs';
            const res = await fetch(endpoint, { method: 'DELETE' });
            if (!res.ok) {
              alert('Failed to delete runs');
              return;
            }
            const data = await res.json().catch(() => ({}));
            if (!force && Array.isArray(data.protected) && data.protected.length) {
              const askForce = confirm(
                `Skipped ${data.protected.length} active run(s): ${data.protected.join(', ')}\nForce delete them?`,
              );
              if (askForce) {
                await deleteAllRuns(true, true);
              }
            }
            closeDetail();
            await loadRuns(false);
          } catch (err) {
            alert('Failed to delete runs');
            console.error(err);
          }
        }

        rows.addEventListener('click', (event) => {
          const target = event.target;
          if (!target) {
            return;
          }
          const selectBtn = target.closest('.row-select');
          const deleteBtn = target.closest('.deleteBtn');
          const linkTarget = target.closest('a[data-run]');
          const row = target.closest('tr[data-run]');
          const runId =
            (selectBtn && selectBtn.dataset.run) ||
            (deleteBtn && deleteBtn.dataset.run) ||
            (linkTarget && linkTarget.dataset.run) ||
            (row && row.dataset.run);
          if (!runId) {
            return;
          }
          if (selectionMode) {
            event.preventDefault();
            toggleRunSelection(runId);
            return;
          }
          if (deleteBtn) {
            event.preventDefault();
            deleteRun(runId);
            return;
          }
          if (linkTarget) {
            event.preventDefault();
            showDetail(runId);
          }
        });
        loadRuns();
        setInterval(() => loadRuns(false), 4000);
        loadSystemMetrics(true);
        setInterval(() => loadSystemMetrics(false), 5000);
      </script>
    </body>
    """
)


def _render_home_page() -> str:
    """Render the static HTML shell for the monitor dashboard."""
    return "\n".join(
        [
            "<!doctype html>",
            '<html lang="en">',
            HOME_HEAD,
            HOME_BODY,
            "</html>",
        ]
    )


def create_app(
    registry: RunRegistry,
    *,
    inventory: Optional[StorageInventory] = None,
    event_socket: Optional[Path] = None,
) -> FastAPI:
    """Build a FastAPI application exposing the monitor endpoints."""
    registry.refresh()
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
        return payload

    @app.get("/api/system/metrics")
    async def system_metrics():
        """Return current CPU, memory, and disk utilisation."""
        return _collect_system_metrics()

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
        return data

    @app.get("/api/runs/{run_id}/iterations")
    async def run_iterations(run_id: str):
        """Return the iteration history for the selected run."""
        registry.refresh()
        run = registry.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return [record.to_dict() for record in run.iterations]

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
        return HTMLResponse(_render_home_page())

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
