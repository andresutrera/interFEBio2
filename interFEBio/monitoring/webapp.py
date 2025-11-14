"""FastAPI web UI for monitoring optimization runs."""

from __future__ import annotations

import argparse
import logging
import textwrap
from pathlib import Path
from typing import Iterable, Optional, Sequence

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
        }
        * { box-sizing: border-box; }
        body { font-family: "Inter", Arial, sans-serif; margin: 1.5rem; background: var(--bg-color); color: var(--text-color); transition: background 0.2s ease, color 0.2s ease; }
        a { color: var(--link-color); text-decoration: none; }
        a:hover { text-decoration: underline; }
        .layout { display: flex; flex-wrap: wrap; gap: 1.25rem; align-items: flex-start; }
        .panel { background: var(--panel-bg); border-radius: 12px; box-shadow: 0 10px 24px rgba(10, 18, 46, 0.18); padding: 1rem 1.25rem; flex: 1 1 360px; min-width: 320px; transition: background 0.2s ease, color 0.2s ease; }
        .panel h2 { margin-top: 0; font-size: 1.1rem; letter-spacing: 0.04em; text-transform: uppercase; opacity: 0.8; }
        table { border-collapse: collapse; width: 100%; background: var(--table-bg); border-radius: 8px; overflow: hidden; }
        th, td { padding: 0.6rem 0.8rem; text-align: left; font-size: 0.9rem; }
        th { background: var(--table-header-bg); text-transform: uppercase; font-size: 0.7rem; letter-spacing: 0.05rem; }
        tr:nth-child(even) { background: var(--table-row-alt); }
        tr:hover { background: var(--table-row-hover); }
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
                <th>Last Cost</th>
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
        let currentDetailData = null;
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
            text: getCssColor('--chart-axis', '#f5f5f5'),
            grid: getCssColor('--chart-grid', 'rgba(255,255,255,0.08)'),
            zero: getCssColor('--chart-zero', 'rgba(255,255,255,0.18)'),
            muted: getCssColor('--muted-text', 'rgba(255,255,255,0.5)'),
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
        function renderEmptyPlot(container, message) {
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
                  const last = typeof run.last_cost === 'number' ? run.last_cost.toExponential(3) : '-';
                  const updated = run.updated_at ? new Date(run.updated_at * 1000).toLocaleTimeString() : '-';
                  return `<tr data-run="${run.run_id}">
                    <td><a href="#" data-run="${run.run_id}">${label}</a></td>
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
    async def delete_run(run_id: str):
        """Delete a single run if it has finished."""
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
        """Delete every completed run and report protected ones."""
        registry.refresh()
        total_before = len(registry.list_runs())
        protected = registry.clear()
        total_after = len(registry.list_runs())
        removed = max(total_before - total_after, 0)
        return {"status": "cleared", "removed": removed, "protected": protected}

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
