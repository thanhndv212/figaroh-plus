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

"""HTML diagnostic report generation for dynamic identification results.

Renders the same statistics as
``BaseIdentification.print_quality_report()`` (condition number, torque
residuals, base-parameter uncertainty, held-out validation, optional
physical-consistency / reconstruction status) as a self-contained,
shareable HTML page — the identification analogue of
``tools/report.py``'s calibration report.

Identification and calibration produce structurally different
diagnostics (a one-shot linear QR solve vs. an iterative nonlinear fit
with outlier removal), so this is a separate adapter rather than a
forced re-use of ``generate_calibration_report`` — see
``docs/decisions/roadmap-mujoco-sysid-inspired-features.md`` (Feature
1b) for the comparison that motivated this split. Only the
domain-independent HTML/CSS layer is shared, via
``tools/_report_common.py``.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from figaroh.tools._report_common import (
    _STYLE,
    UNCERTAINTY_WARN_PCT,
    VALIDATION_IMPROVEMENT_WARN_PCT,
    _esc,
    _insights_section,
    _param_uncertainty_section,
)

CONDITION_ILL_THRESHOLD = 1000.0
CONDITION_MODERATE_THRESHOLD = 100.0
LOW_CORRELATION_WARN = 0.9


def _condition_label(cond_num: float) -> str:
    if math.isnan(cond_num):
        return "unavailable"
    if cond_num > CONDITION_ILL_THRESHOLD:
        return "ill-conditioned"
    if cond_num > CONDITION_MODERATE_THRESHOLD:
        return "moderately conditioned"
    return "well-conditioned"


def _build_insights(
    result: Dict[str, Any],
    std_relative: Optional[List[float]],
    base_names: List[str],
    validation: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Auto-flag things worth a second look. Returns list of
    {'level': 'warn'|'info', 'text': str}."""
    insights: List[Dict[str, str]] = []

    cond_num = result.get("condition number", float("nan"))
    cond_label = _condition_label(cond_num)
    if cond_label == "ill-conditioned":
        insights.append({
            "level": "warn",
            "text": f"Condition number {cond_num:.1f} is ill-conditioned "
                    "— consider a richer excitation trajectory or fewer "
                    "simultaneously identified base parameters.",
        })
    elif cond_label == "moderately conditioned":
        insights.append({
            "level": "info",
            "text": f"Condition number {cond_num:.1f} is moderately "
                    "conditioned — usable, but not optimal.",
        })

    if std_relative:
        poor = [
            base_names[i] if i < len(base_names) else f"param_{i}"
            for i, sp in enumerate(std_relative)
            if sp is not None and not math.isnan(sp)
            and abs(sp) > UNCERTAINTY_WARN_PCT
        ]
        if poor:
            names = ", ".join(poor[:6]) + (", ..." if len(poor) > 6 else "")
            insights.append({
                "level": "warn",
                "text": f"{len(poor)} base parameter(s) have >"
                        f"{UNCERTAINTY_WARN_PCT:.0f}% relative uncertainty "
                        f"and are poorly identified: {names}.",
            })

    if validation is None:
        insights.append({
            "level": "info",
            "text": "No held-out validation data provided — these "
                    "metrics reflect fit quality on the training set "
                    "only, not generalization. Set "
                    "validation_data_file in the identification config "
                    "to enable it.",
        })
    else:
        corr = validation.get("correlation", 1.0)
        if corr < LOW_CORRELATION_WARN:
            insights.append({
                "level": "warn",
                "text": f"Validation torque correlation is only "
                        f"{corr:.3f} — the identified model may not "
                        "generalize well beyond the excitation "
                        "trajectory used for fitting.",
            })
        improvement = validation.get("improvement_pct", 100.0)
        if improvement < VALIDATION_IMPROVEMENT_WARN_PCT:
            insights.append({
                "level": "warn",
                "text": "Validation torque RMSE improved by only "
                        f"{improvement:.1f}% over the nominal/CAD "
                        "parameters — check model assumptions or "
                        "excitation trajectory coverage.",
            })

    pc = result.get("physical consistency")
    if isinstance(pc, dict) and pc.get("status") not in (
        None, "already_feasible", "feasible",
    ):
        insights.append({
            "level": "warn" if pc.get("status") in ("error", "unavailable")
            else "info",
            "text": f"Physical-consistency projection status: "
                    f"{pc.get('status')}.",
        })

    recon = result.get("reconstruction")
    if isinstance(recon, dict) and recon.get("status") not in (
        None, "success",
    ):
        insights.append({
            "level": "warn",
            "text": f"Full-parameter reconstruction status: "
                    f"{recon.get('status')}.",
        })

    if not insights:
        insights.append({
            "level": "info",
            "text": "No issues detected — fit looks healthy.",
        })

    return insights


def _summary_section(result: Dict[str, Any], correlation: float) -> str:
    cond_num = result.get("condition number", float("nan"))
    cond_label = _condition_label(cond_num)
    rmse_norm = result.get("rmse norm (N/m)", float("nan"))
    cond_str = (
        f"{cond_num:.1f} ({_esc(cond_label)})"
        if not math.isnan(cond_num)
        else _esc(cond_label)
    )
    n_base = len(result.get("base parameters names", []))
    return f"""
    <div class="stat-row">
      <div class="stat">
        <div class="stat-label">Base parameters</div>
        <div class="stat-value">{n_base}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Samples</div>
        <div class="stat-value">{result.get("num samples", 0)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">RMSE</div>
        <div class="stat-value">{rmse_norm:.4f}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Correlation</div>
        <div class="stat-value">{correlation:.4f}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Condition number</div>
        <div class="stat-value">{cond_str}</div>
      </div>
    </div>
    """


def _per_joint_section(per_joint: Optional[Dict[str, Any]]) -> str:
    if not per_joint or not per_joint.get("joint_names"):
        return '<p class="muted">Per-joint residuals unavailable.</p>'

    names = per_joint["joint_names"]
    rows = []
    for i, name in enumerate(names):
        def _at(key):
            arr = per_joint.get(key, [])
            return arr[i] if i < len(arr) else float("nan")

        rows.append(
            "<tr>"
            f"<td>{_esc(name)}</td>"
            f"<td class=\"num\">{_at('mean'):.4f}</td>"
            f"<td class=\"num\">{_at('std'):.4f}</td>"
            f"<td class=\"num\">{_at('rmse'):.4f}</td>"
            f"<td class=\"num\">{_at('max_abs'):.4f}</td>"
            "</tr>"
        )

    return f"""
    <table class="data">
      <thead>
        <tr><th>Joint</th><th>Mean</th><th>Std</th><th>RMSE</th>
            <th>Max</th></tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def _validation_section(validation: Optional[Dict[str, Any]]) -> str:
    if validation is None:
        return (
            '<p class="muted">No separate validation data provided. '
            "Set validation_data_file in the identification config to "
            "a held-out trajectory (a genuinely different dataset, not "
            "a split of the training data) to test generalization.</p>"
        )

    improvement = validation.get("improvement_pct", 0.0)
    arrow = "↓" if improvement > 0 else "↑"
    rows = (
        "<tr>"
        "<td>Torque RMSE</td>"
        f"<td class=\"num\">{validation['rmse_nominal']:.4f}</td>"
        f"<td class=\"num\">{validation['rmse_identified']:.4f}</td>"
        f"<td class=\"num\">{improvement:.1f}% {arrow}</td>"
        "</tr>"
        "<tr>"
        "<td>Torque max |error|</td>"
        f"<td class=\"num\">{validation['max_nominal']:.4f}</td>"
        f"<td class=\"num\">{validation['max_identified']:.4f}</td>"
        "<td class=\"num\">—</td>"
        "</tr>"
    )

    n_val = validation.get("n_val_samples", 0)
    val_corr = validation.get("correlation", float("nan"))
    return f"""
    <p class="muted">Held-out set, n={n_val}
      &middot; correlation {val_corr:.4f}</p>
    <table class="data">
      <thead>
        <tr><th>Metric</th><th>Nominal</th><th>Identified</th>
            <th>Improvement</th></tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """


def _consistency_section(result: Dict[str, Any]) -> str:
    pc = result.get("physical consistency")
    recon = result.get("reconstruction")
    if pc is None and recon is None:
        return (
            '<p class="muted">Physical-consistency projection and '
            "full-parameter reconstruction were not enabled for this "
            "run.</p>"
        )

    parts = []
    if pc is not None:
        pc_status = _esc(pc.get("status", "unknown"))
        parts.append(
            "<div class=\"stat\">"
            "<div class=\"stat-label\">Physical consistency</div>"
            f"<div class=\"stat-value\">{pc_status}</div>"
            "</div>"
        )
    if recon is not None:
        recon_status = _esc(recon.get("status", "unknown"))
        parts.append(
            "<div class=\"stat\">"
            "<div class=\"stat-label\">Reconstruction</div>"
            f"<div class=\"stat-value\">{recon_status}</div>"
            "</div>"
        )
    return f'<div class="stat-row">{"".join(parts)}</div>'


def generate_identification_report(
    identifier, output_path: Optional[str] = None, title: Optional[str] = None
) -> str:
    """Render a self-contained HTML diagnostic report for an
    identification run.

    Reuses the metrics already computed by
    ``BaseIdentification.print_quality_report()`` — no new computation
    is performed here, only presentation.

    Args:
        identifier: A ``BaseIdentification`` instance after ``solve()``
            has been called (i.e. ``result`` is populated).
        output_path: If given, the HTML is also written to this path.
        title: Optional report title. Defaults to the robot's class name.

    Returns:
        The rendered HTML document as a string.

    Raises:
        AttributeError: If ``identifier`` has not been solved yet.
    """
    result = getattr(identifier, "result", None)
    if result is None:
        raise AttributeError(
            "No identification results available. Run solve() first."
        )

    base_names = result.get("base parameters names", [])
    std_relative_raw = getattr(identifier, "std_relative", None)
    std_relative: List[float] = (
        list(std_relative_raw) if std_relative_raw is not None else []
    )
    phi_base = getattr(identifier, "phi_base", None)
    correlation = getattr(identifier, "correlation", float("nan"))
    validation = result.get("validation_metrics")

    std_dev_abs: List[float] = []
    if std_relative is not None and phi_base is not None:
        for i in range(len(std_relative)):
            sp = std_relative[i]
            val = phi_base[i] if i < len(phi_base) else float("nan")
            std_dev_abs.append(
                abs(sp / 100.0 * val) if not math.isnan(sp) else float("nan")
            )

    per_joint = None
    if hasattr(identifier, "_compute_per_joint_stats"):
        per_joint = identifier._compute_per_joint_stats()

    insights = _build_insights(result, std_relative, base_names, validation)

    report_title = title or f"{type(identifier).__name__} Quality Report"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(report_title)}</title>
<style>{_STYLE}</style>
</head>
<body>
<div class="page">
  <h1>{_esc(report_title)}</h1>
  <p class="subtitle">Generated {_esc(timestamp)}</p>

  <section>
    <h2>Summary</h2>
    <div class="card">{_summary_section(result, correlation)}</div>
  </section>

  <section>
    <h2>Insights</h2>
    {_insights_section(insights)}
  </section>

  <section>
    <h2>Per-joint torque residuals</h2>
    <div class="card">{_per_joint_section(per_joint)}</div>
  </section>

  <section>
    <h2>Validation</h2>
    <div class="card">{_validation_section(validation)}</div>
  </section>

  <section>
    <h2>Base-parameter uncertainty</h2>
    <div class="card">{_param_uncertainty_section(
        base_names,
        std_dev_abs,
        std_relative or [],
    )}</div>
  </section>

  <section>
    <h2>Physical consistency / reconstruction</h2>
    <div class="card">{_consistency_section(result)}</div>
  </section>

  <footer>Generated by figaroh.tools.identification_report</footer>
</div>
</body>
</html>
"""

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(doc)

    return doc
