from __future__ import annotations

import argparse
import logging
import textwrap
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

from .events import EventSocketListener
from .paths import default_registry_path, default_socket_path
from .registry import ActiveRunDeletionError, RunRegistry
from .state import StorageInventory

logger = logging.getLogger(__name__)


HOME_HEAD = textwrap.dedent(
    """\
    <head>
      <meta charset="utf-8">
      <title>interFEBio Monitor</title>
      <style>
        :root { color-scheme: dark; }
        * { box-sizing: border-box; }
        body { font-family: "Inter", Arial, sans-serif; margin: 1.5rem; background: #0f1526; color: #f5f5f5; }
        a { color: #65b7ff; text-decoration: none; }
        a:hover { text-decoration: underline; }
        .layout { display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: flex-start; }
        .panel { background: #1d2538; border-radius: 12px; box-shadow: 0 10px 24px rgba(10, 18, 46, 0.4); padding: 1rem 1.25rem; flex: 1 1 360px; min-width: 320px; }
        .panel h2 { margin-top: 0; font-size: 1.1rem; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.8; }
        table { border-collapse: collapse; width: 100%; background: #232c42; border-radius: 8px; overflow: hidden; }
        th, td { padding: 0.6rem 0.8rem; text-align: left; font-size: 0.9rem; }
        th { background: #2f3954; text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05rem; }
        tr:nth-child(even) { background: #1c2438; }
        tr:hover { background: #314162; }
        .status { font-weight: 600; text-transform: uppercase; letter-spacing: 0.03rem; }
        .status.finished { color: #81d887; }
        .status.failed { color: #ff6b6b; }
        .status.running { color: #ffd166; }
        .status.created { color: #9bb4ff; }
        button { margin-right: 0.5rem; padding: 0.45rem 0.9rem; background: linear-gradient(135deg, #3a7bd5, #6f8fdb); color: #fff; border: none; border-radius: 999px; cursor: pointer; font-weight: 600; transition: transform 0.12s ease, box-shadow 0.12s ease; }
        button:hover { transform: translateY(-1px); box-shadow: 0 8px 16px rgba(58, 123, 213, 0.35); }
        .table-actions button { margin-right: 0; }
        .detail-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 1rem; margin-bottom: 1rem; }
        .summary-card { background: #232c42; border-radius: 10px; padding: 0.75rem 0.9rem; }
        .summary-card dt { font-size: 0.75rem; text-transform: uppercase; opacity: 0.6; margin-bottom: 0.25rem; letter-spacing: 0.04rem; }
        .summary-card dd { margin: 0; font-size: 1rem; font-weight: 600; }
        .chart-wrapper { background: #232c42; border-radius: 10px; padding: 0.75rem 1rem; }
        .chart-wrapper h3 { margin: 0 0 0.5rem 0; font-size: 0.95rem; letter-spacing: 0.03rem; text-transform: uppercase; opacity: 0.75; }
        .chart { width: 100%; height: 420px; display: block; border-radius: 6px; background: #10172a; }
        pre { background: #111727; padding: 1rem; border-radius: 10px; overflow-x: auto; margin: 0; font-size: 0.85rem; }
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
      <header style="margin-bottom: 1rem;">
        <h1 style="font-size: 1.6rem; margin: 0 0 0.6rem 0;">Optimization Runs</h1>
        <p style="margin: 0; opacity: 0.6;">Monitor optimisation progress, inspect iteration history, and review run artefacts.</p>
      </header>
      <div class="layout">
        <section class="panel" id="runsPanel">
          <h2>Active Runs</h2>
          <div style="display:flex; gap:0.5rem; margin-bottom:0.6rem;">
            <button id="refresh">Refresh</button>
            <button id="deleteAll">Delete All</button>
          </div>
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Iterations</th>
                <th>Best Cost</th>
                <th>Updated</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody id="runRows">
              <tr><td colspan="6">Loading...</td></tr>
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
              <select id="seriesSelect" style="background:#151f34; color:#f5f5f5; border:1px solid rgba(255,255,255,0.2); border-radius:6px; padding:0.35rem 0.6rem; font-size:0.9rem;"></select>
            </div>
            <div id="seriesChart" class="chart"></div>
          </div>
          <div style="display:flex; justify-content:space-between; align-items:center; margin-top:1.1rem;">
            <h3 style="font-size:0.95rem; letter-spacing:0.03rem; text-transform:uppercase; opacity:0.75; margin:0;">Raw JSON</h3>
            <button id="toggleJson" style="margin-right:0; background:rgba(255,255,255,0.08); padding:0.3rem 0.8rem;">Expand</button>
          </div>
          <pre id="detailJson" style="max-height:0; overflow:hidden; transition:max-height 0.18s ease; visibility:hidden;">{}</pre>
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
        let jsonExpanded = false;
        let selectedRunId = null;
        let lastTableHtml = '';
        let currentSeriesData = {};
        const plotlyConfig = { displayModeBar: false, responsive: true };
        function hasPlotly() {
          return typeof Plotly !== 'undefined' && Plotly.react;
        }
        function renderEmptyPlot(container, message) {
          if (!container) {
            return;
          }
          if (!hasPlotly()) {
            container.innerHTML = `<div style="display:flex; align-items:center; justify-content:center; height:100%; color:rgba(255,255,255,0.5); font-size:16px;">${message}</div>`;
            return;
          }
          Plotly.react(
            container,
            [],
            {
              paper_bgcolor: '#10172a',
              plot_bgcolor: '#10172a',
              autosize: true,
              height: 420,
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
                  font: { color: 'rgba(255,255,255,0.5)', size: 16 },
                },
              ],
            },
            plotlyConfig,
          );
          if (typeof Plotly !== 'undefined' && Plotly.Plots && typeof Plotly.Plots.resize === 'function') {
            Plotly.Plots.resize(container);
          }
        }
        if (seriesSelect) {
          seriesSelect.disabled = true;
        }

        async function loadRuns(showLoading = true) {
          if (showLoading && !lastTableHtml) {
            rows.innerHTML = "<tr><td colspan='6'>Loading...</td></tr>";
          }
          try {
            const res = await fetch('/api/runs', { cache: 'no-store' });
            const data = await res.json();
            let html;
            if (!data.length) {
              html = "<tr><td colspan='6'>No runs yet</td></tr>";
            } else {
              html = data
                .map(run => {
                  const status = run.status || 'unknown';
                  const label = run.label || run.run_id;
                  const best = typeof run.best_cost === 'number' ? run.best_cost.toExponential(3) : '-';
                  const updated = run.updated_at ? new Date(run.updated_at * 1000).toLocaleTimeString() : '-';
                  return `<tr data-run="${run.run_id}">
                    <td><a href="#" data-run="${run.run_id}">${label}</a></td>
                    <td class="status ${status}">${status}</td>
                    <td>${run.iteration_count}</td>
                    <td>${best}</td>
                    <td>${updated}</td>
                    <td><button class="deleteBtn" data-run="${run.run_id}">Delete</button></td>
                  </tr>`;
                })
                .join('');
            }
            if (html !== lastTableHtml) {
              rows.innerHTML = html;
              lastTableHtml = html;
            }
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
            detail.hidden = false;
            detailTitle.textContent = data.label || data.run_id || 'Run detail';
            renderSummary(data);
            renderIterations(data);
            const iterations = Array.isArray(data.iterations) ? data.iterations : [];
            let latestSeries = {};
            for (let idx = iterations.length - 1; idx >= 0; idx--) {
              const entry = iterations[idx];
              if (entry && entry.series && typeof entry.series === 'object' && Object.keys(entry.series).length) {
                latestSeries = entry.series;
                break;
              }
            }
            renderSeries(latestSeries);
            if (detailJson) {
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
            }
          } catch (err) {
            console.error(err);
            detailSummary.innerHTML = "<p style='color:#ff6b6b;'>Failed to load run details.</p>";
            renderEmptyPlot(costChartEl, 'Failed to load data');
            renderEmptyPlot(seriesChartEl, 'Failed to load data');
          }
        }

        async function showDetail(runId) {
          selectedRunId = runId;
          await refreshDetail(runId);
        }

        function renderSummary(data) {
          const summary = [];
          const meta = data.meta || {};
          summary.push({ label: 'Status', value: data.status || 'unknown' });
          if (typeof meta.best_cost === 'number') {
            summary.push({ label: 'Best cost', value: meta.best_cost.toExponential(4) });
          }
          if (typeof meta.last_cost === 'number') {
            summary.push({ label: 'Last cost', value: meta.last_cost.toExponential(4) });
          }
          if (meta.r_squared && typeof meta.r_squared === 'object') {
            const entries = Object.entries(meta.r_squared)
              .map(([key, value]) => {
                if (value === null || value === undefined) {
                  return `${key}: -`;
                }
                const num = Number(value);
                if (!Number.isFinite(num)) {
                  return `${key}: -`;
                }
                return `${key}: ${num.toFixed(4)}`;
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
            summary.push({ label: 'Optimizer', value: label });
          }
          if (data.parameters && Array.isArray(data.parameters.names)) {
            summary.push({ label: 'Parameters', value: data.parameters.names.join(', ') });
          }
          detailSummary.innerHTML = summary.map(item => `
            <dl class="summary-card">
              <dt>${item.label}</dt>
              <dd>${item.value}</dd>
            </dl>`).join('') || "<p style='opacity:0.6;'>No summary metadata</p>";
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
            paper_bgcolor: '#10172a',
            plot_bgcolor: '#10172a',
            autosize: true,
            height: 420,
            margin: { l: 64, r: 28, t: 32, b: 64 },
            font: { color: '#f5f5f5' },
            xaxis: {
              title: { text: 'Iteration' },
              dtick: tickStep,
              color: 'rgba(255,255,255,0.75)',
              gridcolor: 'rgba(255,255,255,0.08)',
              zerolinecolor: 'rgba(255,255,255,0.18)',
            },
            yaxis: {
              title: { text: 'Cost' },
              color: 'rgba(255,255,255,0.75)',
              gridcolor: 'rgba(255,255,255,0.08)',
              zerolinecolor: 'rgba(255,255,255,0.18)',
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
            paper_bgcolor: '#10172a',
            plot_bgcolor: '#10172a',
            autosize: true,
            height: 420,
            margin: { l: 64, r: 28, t: 32, b: 64 },
            font: { color: '#f5f5f5' },
            xaxis: {
              title: { text: 'x' },
              color: 'rgba(255,255,255,0.75)',
              gridcolor: 'rgba(255,255,255,0.08)',
              zerolinecolor: 'rgba(255,255,255,0.18)',
            },
            yaxis: {
              title: { text: 'y' },
              color: 'rgba(255,255,255,0.75)',
              gridcolor: 'rgba(255,255,255,0.08)',
              zerolinecolor: 'rgba(255,255,255,0.18)',
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
        document.getElementById('deleteAll').addEventListener('click', deleteAllRuns);
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

        async function deleteRun(runId) {
          if (!confirm(`Delete run ${runId}?`)) {
            return;
          }
          try {
            const res = await fetch(`/api/runs/${runId}`, { method: 'DELETE' });
            if (res.status === 409) {
              const err = await res.json().catch(() => ({}));
              alert(err.detail || 'Run is still active and cannot be deleted.');
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

        async function deleteAllRuns() {
          if (!confirm('Delete all runs? This cannot be undone.')) {
            return;
          }
          try {
            const res = await fetch('/api/runs', { method: 'DELETE' });
            if (!res.ok) {
              alert('Failed to delete runs');
              return;
            }
            const data = await res.json().catch(() => ({}));
            if (Array.isArray(data.protected) && data.protected.length) {
              alert(`Skipped ${data.protected.length} active run(s): ${data.protected.join(', ')}`);
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
          if (target && target.dataset && target.dataset.run) {
            if (target.classList.contains('deleteBtn')) {
              event.preventDefault();
              deleteRun(target.dataset.run);
            } else {
              event.preventDefault();
              showDetail(target.dataset.run);
            }
          }
        });
        loadRuns();
        setInterval(() => loadRuns(false), 4000);
      </script>
    </body>
    """
)


def _render_home_page() -> str:
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
    registry.refresh()
    app = FastAPI(title="interFEBio Monitor", version="2.0.0")
    listener: Optional[EventSocketListener] = None

    @app.on_event("startup")
    async def _startup() -> None:
        nonlocal listener
        logger.info("Monitor web app starting up")
        if event_socket:
            listener = EventSocketListener(Path(event_socket), registry)
            listener.start()
            logger.info("Event socket listener active at %s", event_socket)

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        if listener:
            listener.stop()

    @app.get("/api/runs")
    async def list_runs():
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
                    "best_cost": meta.get("best_cost"),
                    "last_cost": meta.get("last_cost"),
                    "iteration_count": len(run.iterations),
                    "optimizer": meta.get("optimizer"),
                }
            )
        return payload

    @app.get("/api/runs/{run_id}")
    async def run_detail(run_id: str):
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
        registry.refresh()
        run = registry.get_run(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return [record.to_dict() for record in run.iterations]

    @app.delete("/api/runs/{run_id}")
    async def delete_run(run_id: str):
        try:
            removed = registry.delete_run(run_id)
        except ActiveRunDeletionError:
            raise HTTPException(
                status_code=409,
                detail="Run is still active; stop the optimisation before deleting.",
            ) from None
        if not removed:
            raise HTTPException(status_code=404, detail="Run not found")
        return {"status": "deleted", "run_id": run_id}

    @app.delete("/api/runs")
    async def delete_all_runs():
        registry.refresh()
        total_before = len(registry.list_runs())
        protected = registry.clear()
        total_after = len(registry.list_runs())
        removed = max(total_before - total_after, 0)
        return {"status": "cleared", "removed": removed, "protected": protected}

    @app.get("/", response_class=HTMLResponse)
    async def home() -> HTMLResponse:
        return HTMLResponse(_render_home_page())


    return app


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
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
    import uvicorn

    args = parse_args(argv)
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
