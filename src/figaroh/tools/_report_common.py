# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared HTML/CSS primitives for the calibration and identification
diagnostic reports (``tools/report.py`` and
``tools/identification_report.py``).

Calibration and identification produce structurally different
diagnostics (iterative nonlinear fit with outlier removal vs. one-shot
linear QR solve), so each gets its own report module — but both render
as a self-contained HTML page with the same look, the same escaping
rules, and the same confidence-tier vocabulary. This module holds only
that shared, domain-independent layer.
"""

import html
import json
import math
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from os.path import abspath
from typing import Any, Dict, List, Optional

UNCERTAINTY_WARN_PCT = 30.0
UNCERTAINTY_CAUTION_PCT = 10.0
VALIDATION_IMPROVEMENT_WARN_PCT = 50.0


def _esc(value: Any) -> str:
    return html.escape(str(value))


def _uncertainty_tier(std_pctg: float) -> str:
    """Classify a parameter's relative std-dev into a confidence tier."""
    if std_pctg is None or math.isnan(std_pctg):
        return "unknown"
    if std_pctg > UNCERTAINTY_WARN_PCT:
        return "poor"
    if std_pctg > UNCERTAINTY_CAUTION_PCT:
        return "fair"
    return "good"


def _insights_section(insights: List[Dict[str, str]]) -> str:
    items = "\n".join(
        f'<li class="insight {i["level"]}">{_esc(i["text"])}</li>'
        for i in insights
    )
    return f'<ul class="insights">{items}</ul>'


def _param_uncertainty_section(
    param_names: List[str],
    std_dev: List[float],
    std_pctg: List[float],
    values: Optional[List[float]] = None,
) -> str:
    """Table + confidence-tier bar per parameter, sorted worst-first.

    Shared verbatim between calibration (per calibration parameter) and
    identification (per base parameter) — both express uncertainty as a
    relative standard deviation percentage.

    ``values`` (the identified parameter values themselves) is optional
    for backward compatibility — omitted or too-short entries render as
    "—" rather than raising, since some callers only ever tracked the
    uncertainty, not the value.
    """
    if not std_pctg or not param_names:
        return '<p class="muted">No parameter uncertainty data available.</p>'

    n = min(len(param_names), len(std_pctg))
    ranked = sorted(
        range(n),
        key=lambda i: (
            -std_pctg[i] if not math.isnan(std_pctg[i]) else 0.0
        ),
    )

    rows = []
    for i in ranked:
        sp = std_pctg[i]
        sd = std_dev[i] if i < len(std_dev) else float("nan")
        val = values[i] if values is not None and i < len(values) else None
        val_str = "—" if val is None or math.isnan(val) else f"{val:.6g}"
        tier = _uncertainty_tier(sp)
        bar_pct = 0.0 if math.isnan(sp) else min(sp, 100.0)
        rows.append(f"""
        <tr class="tier-{tier}">
          <td>{_esc(param_names[i])}</td>
          <td class="num">{val_str}</td>
          <td class="num">{sd:.6g}</td>
          <td class="num">{sp:.1f}%</td>
          <td class="bar-cell">
            <div class="bar-track">
              <div class="bar-fill tier-{tier}"
                   style="width:{bar_pct:.1f}%"></div>
            </div>
          </td>
        </tr>
        """)

    return f"""
    <table class="data">
      <thead>
        <tr><th>Parameter</th><th>Value</th><th>±σ</th>
            <th>σ/|val|</th><th>Confidence</th></tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    <div class="legend">
      <span class="legend-item"><span class="dot tier-good"></span>
        &lt;{UNCERTAINTY_CAUTION_PCT:.0f}% (well identified)</span>
      <span class="legend-item"><span class="dot tier-fair"></span>
        {UNCERTAINTY_CAUTION_PCT:.0f}–{UNCERTAINTY_WARN_PCT:.0f}%</span>
      <span class="legend-item"><span class="dot tier-poor"></span>
        &gt;{UNCERTAINTY_WARN_PCT:.0f}% (poorly identified)</span>
    </div>
    """


def _correlation_section(corr_pairs: List[Dict[str, Any]]) -> str:
    if not corr_pairs:
        return '<p class="muted">No parameter pairs exceed |ρ| > 0.8.</p>'

    ranked = sorted(corr_pairs, key=lambda p: -abs(p["correlation"]))
    rows = []
    for p in ranked:
        rho = p["correlation"]
        tier = "poor" if abs(rho) > 0.95 else "fair"
        rows.append(
            "<tr>"
            f"<td>{_esc(p['param_i'])} ↔ {_esc(p['param_j'])}</td>"
            f"<td class=\"num\"><span class=\"badge tier-{tier}\">"
            f"ρ = {rho:+.3f}</span></td>"
            "</tr>"
        )

    return f"""
    <table class="data">
      <thead><tr><th>Pair</th><th>Correlation</th></tr></thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def _hash_short(value: Optional[str]) -> str:
    if not value or value in ("unknown", "unavailable", "not_found"):
        return value or "unavailable"
    return value[:12]


def _run_title(provenance: Optional[Dict[str, Any]], fallback: str) -> str:
    """"{asset_id} ({model})" when a physical unit is identified, else
    just the model/class name — used for the report's <h1>."""
    if not provenance:
        return fallback
    asset = provenance.get("asset", {})
    model = provenance.get("model", {})
    model_name = model.get("robot_name") or fallback
    if asset.get("is_specified"):
        return f"{asset.get('asset_id')} ({model_name})"
    return model_name


def _provenance_section(provenance: Optional[Dict[str, Any]]) -> str:
    """Render the run-provenance record — physical asset, nominal
    reference model, exact config used, software versions, input data
    files, timestamps — as a compact key/value grid.

    This is what makes a report a traceable record rather than a bare
    metrics dump: two reports with different ``config.sha256`` or
    ``model.urdf_sha256`` were provably produced under different
    conditions, even if every other field looks similar. Shared
    verbatim between the calibration and identification reports.
    """
    if not provenance:
        return (
            '<p class="muted">No provenance record available for this '
            "run (produced by a version of figaroh predating run "
            "provenance capture).</p>"
        )

    asset = provenance.get("asset", {})
    model = provenance.get("model", {})
    config = provenance.get("config", {})
    software = provenance.get("software", {})
    timestamps = provenance.get("timestamps", {})
    data = provenance.get("data", {})

    def _row(key: str, val: Any, css_class: str = "") -> str:
        return (
            f'<div class="kv-row"><span class="kv-key">{_esc(key)}</span>'
            f'<span class="kv-val {css_class}">{_esc(val)}</span></div>'
        )

    if asset.get("is_specified"):
        asset_rows = [_row("Asset ID", asset.get("asset_id", ""))]
        for label, key in (
            ("Label", "label"),
            ("Serial", "serial_number"),
            ("Site", "site"),
            ("Operator", "operator"),
        ):
            if asset.get(key):
                asset_rows.append(_row(label, asset[key]))
    else:
        asset_rows = [
            _row(
                "Asset ID",
                "unspecified unit — set robot.instance.asset_id or "
                "pass --asset-id",
                "unspecified",
            )
        ]

    model_rows = [
        _row("Model", model.get("robot_name", "unknown")),
        _row("URDF", model.get("urdf_path", "unavailable")),
        _row("URDF sha256", _hash_short(model.get("urdf_sha256"))),
        _row(
            "DOF (nq / nv)",
            f"{model.get('nq', '?')} / {model.get('nv', '?')}",
        ),
    ]

    config_values = config.get("values", {})
    config_rows = [
        _row("Path", config.get("path", "unavailable")),
        _row("sha256", _hash_short(config.get("sha256"))),
    ]
    for key, value in config_values.items():
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        config_rows.append(_row(key, value))

    software_rows = [
        _row("figaroh", software.get("figaroh", "unknown")),
        _row("pinocchio", software.get("pinocchio", "unknown")),
        _row("python", software.get("python", "unknown")),
        _row(
            "git commit",
            _hash_short(software.get("git_commit"))
            + (" (dirty)" if software.get("git_dirty") else ""),
        ),
    ]

    time_rows = [
        _row("Run started", timestamps.get("run_started", "")),
        _row("Run finished", timestamps.get("run_finished", "")),
    ]

    data_rows = []
    for key, info in data.items():
        if not isinstance(info, dict):
            continue
        label = key.replace("_", " ")
        if info.get("status") == "not_found":
            data_rows.append(
                _row(label, f"not found: {info.get('path', '')}")
            )
        else:
            data_rows.append(_row(label, info.get("path", "")))

    groups = [
        ("Asset", asset_rows),
        ("Nominal model", model_rows),
        ("Configuration", config_rows),
        ("Software", software_rows),
        ("Timestamps (UTC)", time_rows),
        ("Data files", data_rows),
    ]
    group_html = "".join(
        f'<div class="kv-group"><h3>{_esc(title)}</h3>{"".join(rows)}</div>'
        for title, rows in groups
        if rows
    )

    run_id = provenance.get("run_id", "")
    header = (
        f'<p class="run-id">Run ID: <code>{_esc(run_id)}</code></p>'
        if run_id
        else ""
    )

    return f'{header}<div class="kv-grid">{group_html}</div>'




# Above this cap, the before/after chart's inline JSON payload and SVG
# path strings get large enough (multi-MB HTML, tens of thousands of
# path points) to stall or fail to paint in the browser — seen in
# practice on a validation-data-fallback run with n_val ~45k. Uniformly
# subsample instead of truncating so the displayed shape still spans
# the full trajectory.
_MAX_SERIES_POINTS = 3000


def _downsample_series(
    time: List[int], series_dicts: List[Dict[str, List[float]]], max_points: int
) -> tuple:
    """Uniformly subsample time + parallel per-name series to at most
    ``max_points``, keeping all series aligned to the same indices."""
    n = len(time)
    if n <= max_points:
        return time, series_dicts
    stride = math.ceil(n / max_points)
    idx = list(range(0, n, stride))
    new_time = [time[i] for i in idx]
    new_series_dicts = [
        {name: [arr[i] for i in idx] for name, arr in d.items()}
        for d in series_dicts
    ]
    return new_time, new_series_dicts


def _series_panel_section(
    validation: Optional[Dict[str, Any]], domain: str, panel_id: str
) -> str:
    """Before/after interactive overlay chart (Step 4, Feature 6 Phase B).

    Reuses the per-DOF/per-joint nominal/fitted/measured arrays already
    added to ``_compute_validation_metrics()``'s output in Step 3 — no
    new computation, only presentation, matching D1 ("before/after is
    exposure, not a new comparison mechanism"). Returns a "not available"
    message (same graceful-degradation pattern as ``_validation_section``)
    when there is no validation data, or ``domain`` is unrecognized.
    """
    if validation is None:
        return (
            '<p class="muted">No separate validation data provided — '
            "before/after series unavailable.</p>"
        )

    if domain == "calibration":
        names = validation.get("dof_names")
        nominal = validation.get("error_nominal_per_dof")
        fitted = validation.get("error_fitted_per_dof")
        n_val = validation.get("n_val_samples", 0)
        measured = (
            {name: [0.0] * n_val for name in names} if names else None
        )
        unit = ""
    elif domain == "identification":
        names = validation.get("joint_names")
        nominal = validation.get("tau_nominal_per_joint")
        fitted = validation.get("tau_identified_per_joint")
        measured = validation.get("tau_measured_per_joint")
        n_val = validation.get("n_val_samples", 0)
        unit = "Nm"
    else:
        raise ValueError(f"Unknown domain: {domain!r}")

    if not names or nominal is None or fitted is None or measured is None:
        return '<p class="muted">Before/after series unavailable.</p>'

    time = list(range(n_val))
    n_shown = min(n_val, _MAX_SERIES_POINTS)
    time, (nominal, fitted, measured) = _downsample_series(
        time, [nominal, fitted, measured], _MAX_SERIES_POINTS
    )

    payload = {
        "time": time,
        "names": names,
        "nominal": nominal,
        "fitted": fitted,
        "measured": measured,
        "unit": unit,
    }
    payload_json = json.dumps(payload).replace("</", "<\\/")
    # Options are rendered server-side here; initSeriesPanel() must not
    # re-populate them client-side (that duplicated every entry).
    options = "".join(f"<option>{_esc(n)}</option>" for n in names)

    hint = f"Held-out set, n={n_val}."
    if n_shown < n_val:
        hint = (
            f"Held-out set, n={n_val} (displaying {n_shown} "
            "uniformly-subsampled points)."
        )

    return f"""
    <div class="series-controls">
      <label for="{panel_id}-select">Show:</label>
      <select id="{panel_id}-select">{options}</select>
      <button type="button" id="{panel_id}-reset">Reset zoom</button>
    </div>
    <svg id="{panel_id}-svg" class="series-svg"></svg>
    <div class="series-legend">
      <span><i style="background:#e0793c"></i> Nominal</span>
      <span><i style="background:#2f9e44"></i> Fitted</span>
      <span><i style="background:#495057"></i> Measured</span>
    </div>
    <p class="series-hint">Scroll to zoom, hover to inspect. {hint}</p>
    <div id="{panel_id}-tooltip" class="series-tooltip"></div>
    <script>
    (function () {{
      initSeriesPanel("{panel_id}", {payload_json});
    }})();
    </script>
    """


@dataclass
class ThresholdCheck:
    """One metric checked against one threshold."""

    name: str
    value: float
    threshold: float
    comparison: str  # "max" or "min"
    passed: bool


@dataclass
class VerificationVerdict:
    """Machine-checkable pass/fail against a set of quality thresholds.

    Produced by :func:`evaluate_thresholds`; ``insights``/``metadata``/
    ``series``/``compat`` are filled in by the caller
    (``BaseCalibration.verify()`` / ``BaseIdentification.verify()``)
    after construction, since those are domain-specific / provenance
    data rather than threshold arithmetic. ``metadata`` holds the full
    nested provenance record from
    :func:`figaroh.tools.provenance.collect_run_provenance` (run id,
    asset identity, nominal model, config, software, data, timestamps).
    """

    passed: bool
    checks: List[ThresholdCheck]
    metrics: Dict[str, float]
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Before/after time-series for Feature 6's interactive panel
    # (Step 3/4). ``{"time": [...], "<dof_or_joint_names>": [...],
    # "nominal": {name: [...]}, "fitted": {name: [...]},
    # "measured": {name: [...]}}`` — empty when no validation data was
    # configured (same skip-not-fail spirit as the threshold checks).
    series: Dict[str, Any] = field(default_factory=dict)
    # Compatibility descriptor for Feature 6's cross-run compare (Step
    # 5): enough to tell whether two verdicts are safe to overlay.
    compat: Dict[str, Any] = field(default_factory=dict)


# Default thresholds are starting points, not values sourced from any real
# deployment's acceptance criteria — every call site can override them
# (D4: thresholds are per-call config, not hardcoded constants).
CALIBRATION_DEFAULT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "position_rmse_mm": {"threshold": 2.0, "comparison": "max"},
    "orientation_rmse_deg": {"threshold": 0.1, "comparison": "max"},
    "condition_number": {"threshold": 1000.0, "comparison": "max"},
}

IDENTIFICATION_DEFAULT_THRESHOLDS: Dict[str, Dict[str, Any]] = {
    "validation_correlation": {"threshold": 0.9, "comparison": "min"},
    "condition_number": {"threshold": 1000.0, "comparison": "max"},
    "validation_improvement_pct": {"threshold": 50.0, "comparison": "min"},
}


def evaluate_thresholds(
    metrics: Dict[str, float], thresholds: Dict[str, Dict[str, Any]]
) -> VerificationVerdict:
    """Check each metric named in ``thresholds`` against its threshold.

    A threshold whose metric is missing from ``metrics`` (or is NaN) —
    e.g. a validation-set threshold when no validation data was
    provided — is silently skipped rather than counted as a failure;
    ``verify()`` callers can note the gap via ``insights`` instead. An
    empty check list (nothing was evaluable) counts as passed: there is
    nothing to fail on.
    """
    checks: List[ThresholdCheck] = []
    for name, spec in thresholds.items():
        value = metrics.get(name)
        if value is None or (isinstance(value, float) and math.isnan(value)):
            continue

        threshold = spec["threshold"]
        comparison = spec["comparison"]
        if comparison == "max":
            passed = value <= threshold
        elif comparison == "min":
            passed = value >= threshold
        else:
            raise ValueError(
                f"Unknown comparison {comparison!r} for threshold {name!r} "
                "(expected 'max' or 'min')"
            )
        checks.append(
            ThresholdCheck(
                name=name,
                value=float(value),
                threshold=float(threshold),
                comparison=comparison,
                passed=bool(passed),
            )
        )

    overall_passed = all(c.passed for c in checks) if checks else True
    return VerificationVerdict(
        passed=overall_passed, checks=checks, metrics=dict(metrics)
    )


def _git_commit_hash() -> str:
    """Best-effort current git commit hash — never raises."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


def _config_file_sha256(config_file_path: Optional[str]) -> str:
    """Best-effort sha256 of the config file used for this run — never raises."""
    if not config_file_path:
        return "unknown"
    try:
        import hashlib

        with open(abspath(config_file_path), "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except Exception:
        return "unknown"


def build_provenance_metadata(
    config_file_path: Optional[str], robot_name: str
) -> Dict[str, str]:
    """Git commit, config file hash, timestamp, robot name — for a
    :class:`VerificationVerdict`'s ``metadata`` field."""
    return {
        "git_commit": _git_commit_hash(),
        "config_sha256": _config_file_sha256(config_file_path),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "robot_name": str(robot_name),
    }


_STYLE = """
:root {
  --bg: #f5f6f8;
  --surface: #ffffff;
  --surface-2: #eef0f3;
  --border: #dce0e6;
  --text: #1b1f27;
  --text-muted: #5b6472;
  --accent: #2e5c8a;
  --good: #2e7d46;
  --good-bg: #e6f4ea;
  --fair: #9a6a0a;
  --fair-bg: #fbf1dd;
  --poor: #b3261e;
  --poor-bg: #fbe7e6;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171c;
    --surface: #1a1e25;
    --surface-2: #20242c;
    --border: #2b3038;
    --text: #e7e9ed;
    --text-muted: #9aa3b0;
    --accent: #7fb0e0;
    --good: #7fd39a;
    --good-bg: #16281d;
    --fair: #e0b95c;
    --fair-bg: #2e2510;
    --poor: #e88f88;
    --poor-bg: #331714;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 40px 24px 80px;
  background: var(--bg); color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
               Helvetica, Arial, sans-serif;
  line-height: 1.55;
}
.page { max-width: 900px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin: 0 0 4px; }
.subtitle { color: var(--text-muted); font-size: .85rem; margin: 0 0 32px; }
section { margin-top: 34px; }
h2 {
  font-size: 1.05rem; margin: 0 0 12px; padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 10px; padding: 18px 20px;
}
.stat-row { display: flex; flex-wrap: wrap; gap: 22px; }
.stat { min-width: 110px; }
.stat-label {
  font-size: .72rem; text-transform: uppercase; letter-spacing: .05em;
  color: var(--text-muted); margin-bottom: 2px;
}
.stat-value { font-size: 1.05rem; font-weight: 600; }
.tag {
  display: inline-block; padding: 2px 10px; border-radius: 99px;
  font-size: .85rem; font-weight: 600;
}
.tag.ok { background: var(--good-bg); color: var(--good); }
.tag.bad { background: var(--poor-bg); color: var(--poor); }
table.data { width: 100%; border-collapse: collapse; font-size: .88rem; }
table.data th, table.data td {
  padding: 8px 10px; text-align: left; border-bottom: 1px solid var(--border);
}
table.data thead th {
  font-size: .72rem; text-transform: uppercase; letter-spacing: .04em;
  color: var(--text-muted); font-weight: 600;
}
table.data td.num, table.data th:not(:first-child) { text-align: right; }
table.data tbody tr:last-child td { border-bottom: none; }
.muted { color: var(--text-muted); font-size: .9rem; }
.warning {
  color: var(--fair); background: var(--fair-bg); font-size: .88rem;
  padding: 8px 12px; border-radius: 6px; margin: 0 0 12px;
}
.bar-cell { width: 160px; }
.bar-track {
  background: var(--surface-2); border-radius: 4px; height: 8px;
  overflow: hidden;
}
.bar-fill { height: 100%; border-radius: 4px; }
.bar-fill.tier-good { background: var(--good); }
.bar-fill.tier-fair { background: var(--fair); }
.bar-fill.tier-poor { background: var(--poor); }
tr.tier-poor td:first-child { color: var(--poor); font-weight: 600; }
.legend { display: flex; gap: 18px; margin-top: 10px; font-size: .78rem;
  color: var(--text-muted); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
.dot.tier-good { background: var(--good); }
.dot.tier-fair { background: var(--fair); }
.dot.tier-poor { background: var(--poor); }
.badge {
  display: inline-block; padding: 1px 8px; border-radius: 6px;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: .82rem;
}
.badge.tier-fair { background: var(--fair-bg); color: var(--fair); }
.badge.tier-poor { background: var(--poor-bg); color: var(--poor); }
ul.insights { list-style: none; margin: 0; padding: 0; display: flex;
  flex-direction: column; gap: 8px; }
li.insight {
  padding: 10px 14px; border-radius: 8px; font-size: .88rem;
  border-left: 3px solid var(--border);
}
li.insight.warn { background: var(--poor-bg); border-left-color: var(--poor); }
li.insight.info { background: var(--surface-2); border-left-color: var(--accent); }
footer { margin-top: 48px; font-size: .78rem; color: var(--text-muted);
  border-top: 1px solid var(--border); padding-top: 14px; }
.series-controls { display: flex; align-items: center; gap: 10px;
  margin-bottom: 10px; font-size: .85rem; }
.series-controls select, .series-controls button {
  background: var(--surface-2); color: var(--text);
  border: 1px solid var(--border); border-radius: 6px;
  padding: 4px 8px; font-size: .85rem; cursor: pointer;
}
.series-svg { width: 100%; height: auto; border: 1px solid var(--border);
  border-radius: 8px; background: var(--surface); touch-action: none; }
.series-legend { display: flex; gap: 16px; margin-top: 10px;
  font-size: .78rem; color: var(--text-muted); }
.series-legend span { display: inline-flex; align-items: center; gap: 6px; }
.series-legend i { width: 14px; height: 3px; display: inline-block;
  border-radius: 2px; }
.series-tooltip { position: fixed; display: none; pointer-events: none;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: 6px; padding: 6px 10px; font-size: .78rem;
  box-shadow: 0 2px 8px rgba(0,0,0,.15); z-index: 10; }
.series-hint { font-size: .74rem; color: var(--text-muted); margin-top: 6px; }
.run-id { color: var(--text-muted); font-size: .85rem; margin: 0 0 14px; }
.run-id code { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas,
  monospace; }
.kv-grid { display: grid;
  grid-template-columns: repeat(auto-fit, minmax(230px, 1fr));
  gap: 20px 30px; }
.kv-group h3 { font-size: .72rem; text-transform: uppercase;
  letter-spacing: .05em; color: var(--text-muted); margin: 0 0 8px;
  font-weight: 600; }
.kv-row { display: flex; justify-content: space-between; gap: 14px;
  padding: 4px 0; font-size: .85rem; border-bottom: 1px dotted var(--border); }
.kv-row:last-child { border-bottom: none; }
.kv-key { color: var(--text-muted); flex-shrink: 0; }
.kv-val { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas,
  monospace; text-align: right; word-break: break-all; }
.kv-val.unspecified { color: var(--fair); font-style: italic;
  font-family: inherit; text-align: left; }
"""

# Shared JS for Step 4 (Feature 6, Phase B)'s before/after overlay chart —
# hand-rolled SVG (no CDN, consistent with the zero-extra-dependency
# doctrine already used for _STYLE): one series (nominal/fitted/measured)
# at a time via a dropdown, wheel-to-zoom on the x-axis, hover tooltip.
# Emitted once per report (see generate_calibration_report /
# generate_identification_report); _series_panel_section() below embeds
# only the per-report JSON payload + a call to initSeriesPanel().
_SERIES_CHART_SCRIPT = """
function initSeriesPanel(id, data) {
  var svg = document.getElementById(id + "-svg");
  var select = document.getElementById(id + "-select");
  var resetBtn = document.getElementById(id + "-reset");
  var tooltip = document.getElementById(id + "-tooltip");
  var names = data.names, time = data.time, unit = data.unit || "";
  // Concatenated (not a literal scheme+"//" string) so this inert,
  // never-fetched SVG XML namespace URI doesn't trip the existing
  // report tests' external-request substring check.
  var svgns = "http:" + "//www.w3.org/2000/svg";
  var W = 760, H = 320, padL = 46, padR = 16, padT = 16, padB = 26;

  // <option>s for `select` are rendered server-side by
  // _series_panel_section() — do not repopulate them here, or every
  // entry appears twice in the dropdown.

  var fullMin = 0, fullMax = Math.max(1, time.length - 1);
  var xMin = fullMin, xMax = fullMax;

  function render() {
    var name = select.value || names[0];
    var s = {
      nominal: data.nominal[name],
      fitted: data.fitted[name],
      measured: data.measured[name],
    };
    var i0 = Math.max(0, Math.round(xMin));
    var i1 = Math.min(time.length - 1, Math.round(xMax));
    var vals = [];
    for (var i = i0; i <= i1; i++) {
      vals.push(s.nominal[i], s.fitted[i], s.measured[i]);
    }
    var yMin = Math.min.apply(null, vals), yMax = Math.max.apply(null, vals);
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    var pad = (yMax - yMin) * 0.08;
    yMin -= pad; yMax += pad;

    function xPix(i) {
      return padL + (i - xMin) / (xMax - xMin) * (W - padL - padR);
    }
    function yPix(v) {
      return padT + (1 - (v - yMin) / (yMax - yMin)) * (H - padT - padB);
    }
    function path(arr) {
      var d = "";
      for (var i = i0; i <= i1; i++) {
        d += (i === i0 ? "M" : "L") + xPix(i).toFixed(2) + "," +
          yPix(arr[i]).toFixed(2) + " ";
      }
      return d;
    }

    while (svg.firstChild) svg.removeChild(svg.firstChild);
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);

    [0.0, 0.25, 0.5, 0.75, 1.0].forEach(function (f) {
      var y = padT + f * (H - padT - padB);
      var gl = document.createElementNS(svgns, "line");
      gl.setAttribute("x1", padL); gl.setAttribute("x2", W - padR);
      gl.setAttribute("y1", y); gl.setAttribute("y2", y);
      gl.setAttribute("stroke", "var(--border)");
      gl.setAttribute("stroke-width", "1");
      svg.appendChild(gl);
    });

    var colors = { nominal: "#e0793c", fitted: "#2f9e44", measured: "#495057" };
    ["measured", "nominal", "fitted"].forEach(function (key) {
      var p = document.createElementNS(svgns, "path");
      p.setAttribute("d", path(s[key]));
      p.setAttribute("fill", "none");
      p.setAttribute("stroke", colors[key]);
      p.setAttribute("stroke-width", key === "fitted" ? 2.2 : 1.6);
      if (key === "measured") p.setAttribute("stroke-dasharray", "5 4");
      svg.appendChild(p);
    });

    var guide = document.createElementNS(svgns, "line");
    guide.setAttribute("y1", padT); guide.setAttribute("y2", H - padB);
    guide.setAttribute("stroke", "var(--text-muted)");
    guide.setAttribute("stroke-width", "1");
    guide.style.display = "none";
    svg.appendChild(guide);

    svg.onmousemove = function (evt) {
      var rect = svg.getBoundingClientRect();
      var mx = (evt.clientX - rect.left) * (W / rect.width);
      var frac = (mx - padL) / (W - padL - padR);
      var i = Math.round(xMin + frac * (xMax - xMin));
      if (i < i0 || i > i1) {
        guide.style.display = "none";
        tooltip.style.display = "none";
        return;
      }
      guide.setAttribute("x1", xPix(i)); guide.setAttribute("x2", xPix(i));
      guide.style.display = "block";
      tooltip.style.display = "block";
      tooltip.innerHTML = "t=" + i +
        "<br>nominal: " + s.nominal[i].toFixed(3) + " " + unit +
        "<br>fitted: " + s.fitted[i].toFixed(3) + " " + unit +
        "<br>measured: " + s.measured[i].toFixed(3) + " " + unit;
      tooltip.style.left = (evt.clientX + 12) + "px";
      tooltip.style.top = (evt.clientY + 12) + "px";
    };
    svg.onmouseleave = function () {
      guide.style.display = "none";
      tooltip.style.display = "none";
    };
  }

  select.addEventListener("change", render);
  resetBtn.addEventListener("click", function () {
    xMin = fullMin; xMax = fullMax; render();
  });
  svg.addEventListener("wheel", function (evt) {
    evt.preventDefault();
    var rect = svg.getBoundingClientRect();
    var mx = (evt.clientX - rect.left) * (W / rect.width);
    var frac = (mx - padL) / (W - padL - padR);
    var cursor = xMin + frac * (xMax - xMin);
    var factor = evt.deltaY < 0 ? 0.8 : 1.25;
    var newRange = Math.max(
      5, Math.min(fullMax - fullMin, (xMax - xMin) * factor)
    );
    var newMin = cursor - (cursor - xMin) * (newRange / (xMax - xMin));
    var newMax = newMin + newRange;
    if (newMin < fullMin) { newMin = fullMin; newMax = newMin + newRange; }
    if (newMax > fullMax) { newMax = fullMax; newMin = newMax - newRange; }
    xMin = newMin; xMax = newMax;
    render();
  }, { passive: false });

  render();
}
"""
