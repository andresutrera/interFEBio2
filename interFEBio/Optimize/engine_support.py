"""Helper components that keep the FEBio optimisation engine lean."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, cast

import numpy as np
from numpy.typing import NDArray
from prettytable import PrettyTable

from ..monitoring.client import MonitorConfig, OptimizationMonitorClient
from .options import EngineOptions, MonitorOptions
from .Parameters import BoundsPayload, ParameterSpace
from .Storage import StorageWorkspace

Array = NDArray[np.float64]


class RunReporter:
    """Handle logging and monitoring events for optimisation runs."""

    def __init__(
        self,
        logger,
        parameter_space: ParameterSpace,
        case_descriptions: List[Mapping[str, Any]],
        monitor_opts: MonitorOptions,
        workspace: StorageWorkspace,
        *,
        reparam_enabled: bool,
    ) -> None:
        self.logger = logger
        self.parameter_space = parameter_space
        self.case_descriptions = case_descriptions
        self.workspace = workspace
        self.reparam_enabled = reparam_enabled
        self._monitor_enabled = bool(monitor_opts.enabled)
        self._monitor_socket = (
            Path(monitor_opts.socket).expanduser() if monitor_opts.socket else None
        )
        self._monitor_label = monitor_opts.label
        self._monitor_client: OptimizationMonitorClient | None = None

    def log_banner(self) -> None:
        banner_lines = [
            "",
            "================================================",
            " _       _            _____ _____ ____  _       ",
            "(_)_ __ | |_ ___ _ __|  ___| ____| __ )(_) ___  ",
            "| | '_ \\| __/ _ \\ '__| |_  |  _| |  _ \\| |/ _ \\ ",
            "| | | | | ||  __/ |  |  _| | |___| |_) | | (_) |",
            "|_|_| |_|\\__\\___|_|  |_|   |_____|____/|_|\\___/ ",
            "                                                ",
            "              Optimization Engine               ",
            "================================================",
            "",
        ]
        self.logger.info("\n" + "\n".join(banner_lines))

    def log_configuration(
        self,
        *,
        options: EngineOptions,
        runner_command: tuple[str, ...],
        runner_env: Mapping[str, str] | None,
        optimizer_adapter: str,
    ) -> None:
        optimizer_opts = options.optimizer
        jacobian_opts = options.jacobian
        monitor_opts = options.monitor
        runner_opts = options.runner

        params = self.parameter_space.parameters()
        param_lines: List[str] = []
        if params:
            for param in params:
                lo, hi = param.bounds if param.bounds is not None else (None, None)
                param_lines.append(
                    f"• {param.name}: θ₀={param.theta0:.6g}, vary={param.vary}, bounds=({lo},{hi})"
                )
        else:
            param_lines.append("• (no parameters)")

        def _section(title: str, rows: Sequence[str]) -> List[str]:
            lines = [f"│ {title}"]
            if rows:
                lines.extend(f"│   {row}" for row in rows)
            else:
                lines.append("│   • (none)")
            lines.append("│")
            return lines

        lines: List[str] = ["┌────────────── ENGINE CONFIGURATION ────────────────"]
        lines += _section(
            "Optimizer",
            [
                f"• name: {optimizer_opts.name}",
                f"• adapter: {optimizer_adapter}",
                f"• reparametrize: {optimizer_opts.reparametrize}",
                f"• settings: {optimizer_opts.settings or {}}",
            ],
        )
        lines += _section(
            "Jacobian",
            [
                f"• enabled: {jacobian_opts.enabled}",
                f"• perturbation: {float(jacobian_opts.perturbation)}",
                f"• parallel: {bool(jacobian_opts.parallel)}",
            ],
        )
        lines += _section("Storage", self.workspace.describe())
        lines += _section(
            "Runner",
            [
                f"• jobs: {runner_opts.jobs}",
                f"• command: {' '.join(runner_command)}",
                f"• env keys: {sorted((runner_env or {}).keys())}",
            ],
        )
        lines += _section(
            "Monitor",
            [
                f"• enabled: {monitor_opts.enabled}",
                f"• socket: {monitor_opts.socket}",
                f"• label: {monitor_opts.label}",
            ],
        )
        case_lines: List[str] = []
        for entry in self.case_descriptions:
            subfolder = entry.get("subfolder") or "."
            experiments = entry.get("experiments") or []
            case_lines.append(f"• subfolder='{subfolder}'")
            case_lines.append(f"    experiments: {experiments or '(none)'}")
        lines += _section("Cases", case_lines)
        lines += _section("Parameters", param_lines)
        lines[-1] = "└─────────────────────────────────────────────────────"
        self.logger.info("\n" + "\n".join(lines))

    def ensure_monitor(self) -> OptimizationMonitorClient | None:
        if not self._monitor_enabled:
            return None
        if self._monitor_client is not None:
            return self._monitor_client
        try:
            config = MonitorConfig(
                socket_path=self._monitor_socket,
                label=self._monitor_label or self._default_label(),
            )
            self._monitor_client = OptimizationMonitorClient(config)
        except Exception:
            self.logger.exception(
                "Monitor initialisation failed; disabling monitoring."
            )
            self._monitor_client = None
        return self._monitor_client

    def notify_run_started(
        self,
        phi0_vec: Array,
        theta0_vec: Array,
        bounds: BoundsPayload,
        optimizer_name: str,
        runner_jobs: int | None,
    ) -> None:
        if self.ensure_monitor() is None:
            return
        parameters = {
            "names": list(self.parameter_space.names),
            "phi0": [float(x) for x in np.asarray(phi0_vec, dtype=float)],
            "theta0": [float(x) for x in np.asarray(theta0_vec, dtype=float)],
            "bounds": self._serialise_bounds(bounds),
        }
        meta = {
            "storage_root": str(self.workspace.persist_root),
            "runner_jobs": runner_jobs,
        }
        assert self._monitor_client is not None
        self._monitor_client.run_started(
            parameters=parameters,
            cases=self.case_descriptions,
            optimizer={"adapter": optimizer_name},
            meta={k: v for k, v in meta.items() if v is not None},
        )

    def record_iteration(
        self,
        index: int,
        phi_vec: Array,
        theta_vec: Array,
        cost: float,
        metrics: Mapping[str, Any],
        series: Mapping[str, Dict[str, Any]],
        *,
        log_output: bool,
    ) -> None:
        r_squared = metrics.get("r_squared", {})
        nrmse_val = metrics.get("nrmse")
        if log_output:
            table = PrettyTable()
            if self.reparam_enabled:
                table.field_names = ["parameter", "phi", "theta"]
            else:
                table.field_names = ["parameter", "theta"]
            for name, phi_value, theta_value in zip(
                self.parameter_space.names, phi_vec, theta_vec, strict=True
            ):
                theta_float = float(theta_value)
                if self.reparam_enabled:
                    table.add_row(
                        [name, f"{float(phi_value):.6e}", f"{theta_float:.6e}"]
                    )
                else:
                    table.add_row([name, f"{theta_float:.6e}"])
            nrmse_display = (
                "nan"
                if nrmse_val is None or not np.isfinite(nrmse_val)
                else f"{float(nrmse_val):.6e}"
            )
            rsq_table = PrettyTable()
            rsq_table.field_names = ["case/experiment", "R^2"]
            if r_squared:
                for key in sorted(r_squared):
                    value = r_squared[key]
                    rsq_table.add_row(
                        [
                            key,
                            "nan"
                            if value is None or not np.isfinite(value)
                            else f"{float(value):.6f}",
                        ]
                    )
            else:
                rsq_table.add_row(["-", "-"])
            self.logger.info(
                "\n[iter {iter:03d}] cost={cost:.6e} nrmse={nrmse}\n{param_table}\n{rsq_table}",
                iter=index,
                cost=cost,
                nrmse=nrmse_display,
                param_table=table.get_string(),
                rsq_table=rsq_table.get_string(),
            )

        client = self.ensure_monitor()
        if client is not None:
            try:
                client.record_iteration(
                    index=index,
                    cost=float(cost),
                    theta={
                        name: float(val)
                        for name, val in zip(
                            self.parameter_space.names, theta_vec, strict=True
                        )
                    },
                    metrics=self._sanitize_metrics(metrics),
                    series=series,
                )
            except Exception:
                self.logger.exception("Failed to emit monitor iteration event.")

    def log_final_summary(
        self, phi_vec: Array, theta: Mapping[str, float], metrics: Mapping[str, Any]
    ) -> None:
        table = PrettyTable()
        table.field_names = ["parameter", "phi", "theta"]
        phi_values = np.asarray(phi_vec, dtype=float).reshape(-1)
        theta_values = {k: float(v) for k, v in theta.items()}
        for name, phi_value in zip(self.parameter_space.names, phi_values, strict=True):
            theta_value = theta_values.get(name, float("nan"))
            theta_display = (
                "nan" if not np.isfinite(theta_value) else f"{theta_value:+.6e}"
            )
            phi_display = f"{float(phi_value):+.6e}"
            table.add_row([name, phi_display, theta_display])
        nrmse_val = metrics.get("nrmse")
        nrmse_display = (
            "nan"
            if nrmse_val is None or not np.isfinite(nrmse_val)
            else f"{float(nrmse_val):.6e}"
        )
        self.logger.info(
            "\nOptimization complete nrmse={nrmse}\n{table}",
            nrmse=nrmse_display,
            table=table.get_string(),
        )

    def notify_completed(
        self,
        theta_opt: Mapping[str, float],
        optimizer_meta: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> None:
        client = self._monitor_client
        if client is None:
            return
        try:
            summary = {
                "theta_opt": {name: float(val) for name, val in theta_opt.items()},
                **self._sanitize_metrics(metrics),
                "optimizer": self._simplify_meta(dict(optimizer_meta or {})),
            }
            client.run_completed(summary=summary)
        except Exception:
            self.logger.exception("Failed to emit monitor completion event.")

    def notify_failed(self, reason: str) -> None:
        client = self._monitor_client
        if client is None:
            return
        try:
            client.run_failed(reason=reason)
        except Exception:
            self.logger.exception("Failed to emit monitor failure event.")

    def close(self) -> None:
        self._monitor_client = None

    def _default_label(self) -> str:
        for candidate in (
            self.workspace.persist_root.name,
            self.workspace.workdir.name,
        ):
            if candidate:
                return candidate
        return datetime.now().strftime("run-%Y%m%d%H%M%S")

    def _sanitize_metrics(self, metrics: Mapping[str, Any]) -> Dict[str, Any]:
        clean: Dict[str, Any] = {}
        nrmse = metrics.get("nrmse")
        if isinstance(nrmse, (int, float)) and np.isfinite(nrmse):
            clean["nrmse"] = float(nrmse)
        r_squared = metrics.get("r_squared")
        if isinstance(r_squared, Mapping):
            clean["r_squared"] = {
                key: (
                    float(value)
                    if isinstance(value, (int, float)) and np.isfinite(value)
                    else None
                )
                for key, value in r_squared.items()
            }
        return clean

    def _simplify_meta(self, data: Mapping[str, Any]) -> Dict[str, Any]:
        return {str(key): self._simplify_value(value) for key, value in data.items()}

    def _simplify_value(self, value: Any) -> Any:
        if isinstance(value, (int, float, str, bool)) or value is None:
            return value
        if isinstance(value, Mapping):
            limited_items = list(value.items())[:16]
            return {str(k): self._simplify_value(v) for k, v in limited_items}
        if isinstance(value, (list, tuple, set)):
            limited = list(value)[:16]
            return [self._simplify_value(item) for item in limited]
        if hasattr(value, "tolist"):
            arr = np.asarray(value)
            if arr.ndim == 0:
                scalar = arr.item()
                if isinstance(scalar, (int, float)):
                    return float(scalar)
                return self._simplify_value(scalar)
            flat = arr.reshape(-1)
            limited = flat[:16].tolist()
            return [float(x) for x in limited]
        return str(value)

    def _serialise_bounds(self, bounds: BoundsPayload) -> List[tuple[float, float]]:
        if isinstance(bounds, tuple) and len(bounds) == 2:
            lo_arr, hi_arr = bounds
            lo_vec = np.asarray(lo_arr, dtype=float).reshape(-1)
            hi_vec = np.asarray(hi_arr, dtype=float).reshape(-1)
            return list(zip(lo_vec.tolist(), hi_vec.tolist(), strict=True))
        serialised: List[tuple[float, float]] = []
        seq_bounds = cast(Sequence[tuple[float, float]], bounds)
        for pair in seq_bounds:
            try:
                lo, hi = pair
            except Exception:
                continue
            serialised.append((float(lo), float(hi)))
        return serialised


__all__ = ["RunReporter"]
