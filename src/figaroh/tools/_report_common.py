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
    param_names: List[str], std_dev: List[float], std_pctg: List[float]
) -> str:
    """Table + confidence-tier bar per parameter, sorted worst-first.

    Shared verbatim between calibration (per calibration parameter) and
    identification (per base parameter) — both express uncertainty as a
    relative standard deviation percentage.
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
        tier = _uncertainty_tier(sp)
        bar_pct = 0.0 if math.isnan(sp) else min(sp, 100.0)
        rows.append(f"""
        <tr class="tier-{tier}">
          <td>{_esc(param_names[i])}</td>
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
        <tr><th>Parameter</th><th>±σ</th>
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

    Produced by :func:`evaluate_thresholds`; ``insights``/``metadata`` are
    filled in by the caller (``BaseCalibration.verify()`` /
    ``BaseIdentification.verify()``) after construction, since those are
    domain-specific / provenance data rather than threshold arithmetic.
    """

    passed: bool
    checks: List[ThresholdCheck]
    metrics: Dict[str, float]
    insights: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


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
"""
