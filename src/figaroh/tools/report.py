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

"""HTML diagnostic report generation for calibration results.

Renders the same statistics as
``BaseCalibration.print_quality_report()`` (convergence, per-DOF
residuals, parameter uncertainty, correlation, validation) as a
self-contained, shareable HTML page instead of terminal text — with a
visual encoding of which parameters are well identified and which are
not, and an auto-generated "insights" list flagging things worth a
second look.
"""

import math
from datetime import datetime
from typing import Any, Dict, List, Optional

from figaroh.tools._report_common import (
    _SERIES_CHART_SCRIPT,
    _STYLE,
    UNCERTAINTY_CAUTION_PCT,
    UNCERTAINTY_WARN_PCT,
    VALIDATION_IMPROVEMENT_WARN_PCT,
    _correlation_section,
    _esc,
    _insights_section,
    _param_uncertainty_section,
    _provenance_section,
    _run_title,
    _series_panel_section,
    _uncertainty_tier,
)

OUTLIER_WARN_PCT = 10.0


def _build_insights(
    eval_: Dict[str, Any],
    n_samples: int,
    param_names: List[str],
    validation: Optional[Dict[str, Any]],
) -> List[Dict[str, str]]:
    """Auto-flag things worth a second look. Returns list of
    {'level': 'warn'|'info', 'text': str}."""
    insights: List[Dict[str, str]] = []

    if not eval_.get("optimization_success", True):
        insights.append({
            "level": "warn",
            "text": "Optimization did not report success — treat "
                    "results with caution.",
        })

    cond_label = eval_.get("condition_label", "unavailable")
    cond_num = eval_.get("condition_number", float("nan"))
    if cond_label == "ill-conditioned":
        insights.append({
            "level": "warn",
            "text": f"Condition number {cond_num:.1f} is ill-conditioned "
                    "— consider a richer excitation trajectory or fewer "
                    "simultaneously identified parameters.",
        })
    elif cond_label == "moderately conditioned":
        insights.append({
            "level": "info",
            "text": f"Condition number {cond_num:.1f} is moderately "
                    "conditioned — usable, but not optimal.",
        })

    outlier_pct = eval_.get("outlier_percentage", 0.0)
    n_outliers = eval_.get("n_outliers", 0)
    if outlier_pct > OUTLIER_WARN_PCT:
        insights.append({
            "level": "warn",
            "text": f"{n_outliers} outliers removed "
                    f"({outlier_pct:.1f}% of {n_samples} samples) — "
                    "check data quality if this seems high.",
        })

    std_pctg = eval_.get("param_stddev_percentage", [])
    poor = [
        param_names[i] if i < len(param_names) else f"param_{i}"
        for i, sp in enumerate(std_pctg)
        if sp is not None and not math.isnan(sp) and sp > UNCERTAINTY_WARN_PCT
    ]
    if poor:
        names = ", ".join(poor[:6]) + (", ..." if len(poor) > 6 else "")
        insights.append({
            "level": "warn",
            "text": f"{len(poor)} parameter(s) have >"
                    f"{UNCERTAINTY_WARN_PCT:.0f}% relative uncertainty "
                    f"and are poorly identified: {names}.",
        })

    corr_pairs = eval_.get("correlated_pairs", [])
    if corr_pairs:
        insights.append({
            "level": "warn",
            "text": f"{len(corr_pairs)} parameter pair(s) are strongly "
                    "correlated (|ρ| > 0.8) — the excitation "
                    "trajectory may not separate them; consider fixing "
                    "one or redesigning the trajectory.",
        })

    if validation is None:
        insights.append({
            "level": "info",
            "text": "No held-out validation data provided — these "
                    "metrics reflect fit quality on the training set "
                    "only, not generalization.",
        })
    else:
        if validation.get("validation_source") == "calibration_data_fallback":
            insights.append({
                "level": "warn",
                "text": "No separate validation data provided — "
                        "validation metrics fall back to the "
                        "calibration data itself and do NOT test "
                        "generalization to new configurations.",
            })
        if validation.get("pos_improvement_pct", 100.0) < (
            VALIDATION_IMPROVEMENT_WARN_PCT
        ):
            insights.append({
                "level": "warn",
                "text": "Validation position RMSE improved by only "
                        f"{validation['pos_improvement_pct']:.1f}% over "
                        "nominal — check model assumptions or "
                        "configuration.",
            })

    if not insights:
        insights.append({
            "level": "info",
            "text": "No issues detected — fit looks healthy.",
        })

    return insights


def _summary_section(eval_: Dict[str, Any], n_samples: int) -> str:
    status_ok = eval_.get("optimization_success", False)
    status = "converged" if status_ok else "failed"
    status_class = "ok" if status_ok else "bad"
    cond_num = eval_.get("condition_number", float("nan"))
    cond_label = eval_.get("condition_label", "unavailable")
    cond_str = (
        f"{cond_num:.1f} ({_esc(cond_label)})"
        if not math.isnan(cond_num)
        else _esc(cond_label)
    )
    return f"""
    <div class="stat-row">
      <div class="stat">
        <div class="stat-label">Convergence</div>
        <div class="stat-value tag {status_class}">{_esc(status)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Iterations</div>
        <div class="stat-value">{eval_.get("n_iterations", 0)}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Cost</div>
        <div class="stat-value">{eval_.get("cost", float("nan")):.6f}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Samples</div>
        <div class="stat-value">{n_samples}</div>
      </div>
      <div class="stat">
        <div class="stat-label">Outliers</div>
        <div class="stat-value">{eval_.get("n_outliers", 0)}
          ({eval_.get("outlier_percentage", 0.0):.1f}%)</div>
      </div>
      <div class="stat">
        <div class="stat-label">Condition number</div>
        <div class="stat-value">{cond_str}</div>
      </div>
    </div>
    """


def _per_dof_section(per_dof: Dict[str, Any]) -> str:
    names = per_dof.get("dof_names", [])
    if not names:
        return "<p class=\"muted\">No per-DOF residual data available.</p>"

    rows = []
    for i, name in enumerate(names):
        def _at(key):
            arr = per_dof.get(key, [])
            return arr[i] if i < len(arr) else float("nan")

        rows.append(
            "<tr>"
            f"<td>{_esc(name)}</td>"
            f"<td class=\"num\">{_at('mean'):.4f}</td>"
            f"<td class=\"num\">{_at('std'):.4f}</td>"
            f"<td class=\"num\">{_at('rmse'):.4f}</td>"
            f"<td class=\"num\">{_at('max_abs'):.4f}</td>"
            f"<td class=\"num\">{_at('r_squared'):.4f}</td>"
            "</tr>"
        )

    overall = per_dof.get("overall", {})
    overall_html = ""
    if overall:
        mae_html = ""
        if "pos_mae_mm" in overall:
            mae_html = f"""
          <div class="stat">
            <div class="stat-label">Position MAE</div>
            <div class="stat-value">{overall["pos_mae_mm"]:.2f} mm</div>
          </div>
          <div class="stat">
            <div class="stat-label">Orientation MAE</div>
            <div class="stat-value">{overall["orient_mae_deg"]:.4f} deg</div>
          </div>
            """
        overall_html = f"""
        <div class="stat-row" style="margin-top:14px;">
          <div class="stat">
            <div class="stat-label">Position RMSE</div>
            <div class="stat-value">{overall["pos_rmse_mm"]:.2f} mm</div>
          </div>
          <div class="stat">
            <div class="stat-label">Orientation RMSE</div>
            <div class="stat-value">{overall["orient_rmse_deg"]:.4f} deg</div>
          </div>
          {mae_html}
          <div class="stat">
            <div class="stat-label">Position max</div>
            <div class="stat-value">{overall["pos_max_mm"]:.2f} mm</div>
          </div>
          <div class="stat">
            <div class="stat-label">Orientation max</div>
            <div class="stat-value">{overall["orient_max_deg"]:.4f} deg</div>
          </div>
        </div>
        """

    return f"""
    <table class="data">
      <thead>
        <tr><th>DOF</th><th>Mean</th><th>Std</th><th>RMSE</th>
            <th>Max</th><th>R²</th></tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    {overall_html}
    """


def _validation_section(validation: Optional[Dict[str, Any]]) -> str:
    if validation is None:
        return (
            '<p class="muted">No separate validation data provided. '
            "Collect measurements at random configurations to test "
            "generalization beyond the excitation trajectory.</p>"
        )

    def _row(label, nominal, calibrated, improvement, unit):
        arrow = "↓" if improvement > 0 else "↑"
        return (
            "<tr>"
            f"<td>{_esc(label)}</td>"
            f"<td class=\"num\">{nominal:.2f} {unit}</td>"
            f"<td class=\"num\">{calibrated:.2f} {unit}</td>"
            f"<td class=\"num\">{improvement:.1f}% {arrow}</td>"
            "</tr>"
        )

    warning_html = ""
    if validation.get("validation_source") == "calibration_data_fallback":
        warning_html = (
            '<p class="warning">⚠ No separate validation data was '
            "provided — falling back to calibration data. These "
            "metrics are <strong>not</strong> an independent "
            "generalization test.</p>"
        )

    rows = [
        _row(
            "Position RMSE",
            validation["pos_rmse_nominal_mm"],
            validation["pos_rmse_calibrated_mm"],
            validation["pos_improvement_pct"],
            "mm",
        ),
        _row(
            "Orientation RMSE",
            validation["orient_rmse_nominal_deg"],
            validation["orient_rmse_calibrated_deg"],
            validation["orient_improvement_pct"],
            "deg",
        ),
        _row(
            "Position max",
            validation["pos_max_nominal_mm"],
            validation["pos_max_calibrated_mm"],
            validation["pos_improvement_pct"],
            "mm",
        ),
        _row(
            "Orientation max",
            validation["orient_max_nominal_deg"],
            validation["orient_max_calibrated_deg"],
            validation["orient_improvement_pct"],
            "deg",
        ),
    ]

    set_label = (
        "calibration set (fallback)"
        if validation.get("validation_source") == "calibration_data_fallback"
        else "held-out set"
    )
    return f"""
    {warning_html}
    <p class="muted">{set_label}, n={validation.get("n_val_samples", 0)}</p>
    <table class="data">
      <thead>
        <tr><th>Metric</th><th>Nominal</th><th>Calibrated</th>
            <th>Improvement</th></tr>
      </thead>
      <tbody>{"".join(rows)}</tbody>
    </table>
    """


def generate_calibration_report(
    calibrator, output_path: Optional[str] = None, title: Optional[str] = None
) -> str:
    """Render a self-contained HTML diagnostic report for a calibration run.

    Reuses the metrics already computed by
    ``BaseCalibration.print_quality_report()`` — no new computation is
    performed here, only presentation.

    Args:
        calibrator: A ``BaseCalibration`` instance after ``solve()`` has
            been called (i.e. ``evaluation_metrics``/``results_data`` are
            populated).
        output_path: If given, the HTML is also written to this path.
        title: Optional report title. Defaults to the robot's class name.

    Returns:
        The rendered HTML document as a string.

    Raises:
        AttributeError: If ``calibrator`` has not been solved yet.
    """
    eval_ = calibrator.evaluation_metrics
    calib_config = calibrator.calib_config
    n_samples = calib_config.get("NbSample", 0)
    param_names = calib_config.get("param_name", [])

    validation = None
    results_data = getattr(calibrator, "results_data", None) or {}
    if "validation_metrics" in results_data:
        validation = results_data["validation_metrics"]

    insights = _build_insights(eval_, n_samples, param_names, validation)

    provenance = getattr(calibrator, "_run_provenance", None)
    report_title = title or (
        f"{_run_title(provenance, type(calibrator).__name__)} "
        "— Calibration Quality Report"
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{_esc(report_title)}</title>
<style>{_STYLE}</style>
<script>{_SERIES_CHART_SCRIPT}</script>
</head>
<body>
<div class="page">
  <h1>{_esc(report_title)}</h1>
  <p class="subtitle">Generated {_esc(timestamp)}</p>

  <section>
    <h2>Provenance</h2>
    <div class="card">{_provenance_section(provenance)}</div>
  </section>

  <section>
    <h2>Summary</h2>
    <div class="card">{_summary_section(eval_, n_samples)}</div>
  </section>

  <section>
    <h2>Insights</h2>
    {_insights_section(insights)}
  </section>

  <section>
    <h2>Per-DOF residuals</h2>
    <div class="card">{_per_dof_section(eval_.get("per_dof_stats", {}))}</div>
  </section>

  <section>
    <h2>Validation</h2>
    <div class="card">{_validation_section(validation)}</div>
  </section>

  <section>
    <h2>Before / after</h2>
    <div class="card">{_series_panel_section(
        validation, "calibration", "series-panel"
    )}</div>
  </section>

  <section>
    <h2>Parameter uncertainty</h2>
    <div class="card">{_param_uncertainty_section(
        param_names,
        eval_.get("param_stdev", []),
        eval_.get("param_stddev_percentage", []),
    )}</div>
  </section>

  <section>
    <h2>Parameter correlation</h2>
    <div class="card">{_correlation_section(
        eval_.get("correlated_pairs", [])
    )}</div>
  </section>

  <footer>Generated by figaroh.tools.report</footer>
</div>
</body>
</html>
"""

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(doc)

    return doc
