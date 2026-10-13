# Tools

## Reporting & Verification

The [reporting & verification suite](https://thanhndv212.github.io/figaroh-plus/reporting_and_verification/index.md)'s HTML report generators and the static two-run compare page.

### report

HTML diagnostic report generation for calibration results.

Renders the same statistics as `BaseCalibration.print_quality_report()` (convergence, per-DOF residuals, parameter uncertainty, correlation, validation) as a self-contained, shareable HTML page instead of terminal text — with a visual encoding of which parameters are well identified and which are not, and an auto-generated "insights" list flagging things worth a second look.

## `generate_calibration_report(calibrator, output_path=None, title=None)`

Render a self-contained HTML diagnostic report for a calibration run.

Reuses the metrics already computed by `BaseCalibration.print_quality_report()` — no new computation is performed here, only presentation.

Parameters:

| Name          | Type            | Description                                                                                                    | Default    |
| ------------- | --------------- | -------------------------------------------------------------------------------------------------------------- | ---------- |
| `calibrator`  |                 | A BaseCalibration instance after solve() has been called (i.e. evaluation_metrics/results_data are populated). | *required* |
| `output_path` | `Optional[str]` | If given, the HTML is also written to this path.                                                               | `None`     |
| `title`       | `Optional[str]` | Optional report title. Defaults to the robot's class name.                                                     | `None`     |

Returns:

| Type  | Description                             |
| ----- | --------------------------------------- |
| `str` | The rendered HTML document as a string. |

Raises:

| Type             | Description                            |
| ---------------- | -------------------------------------- |
| `AttributeError` | If calibrator has not been solved yet. |

Source code in `src/figaroh/tools/report.py`

```
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

    redistributed = None
    try:
        redistributed = calibrator.redistribute_parameters()
    except Exception as e:
        logger.debug("redistribute_parameters unavailable for report: %s", e)

    provenance = getattr(calibrator, "_run_provenance", None)
    report_title = title or (
        f"{_run_title(provenance, type(calibrator).__name__)} "
        "— Calibration Quality Report"
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    from figaroh.tools.provenance import collect_splits
    from figaroh.tools.stages import stages_as_dicts

    stage_records = stages_as_dicts(calibrator)
    splits = collect_splits(calibrator)

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
    <h2>Stages and data</h2>
    <div class="card">{_stages_section(stage_records, splits)}</div>
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
    <div class="card">{_calibration_uncertainty_section(eval_, param_names)}</div>
  </section>

  <section>
    <h2>Redistributed standard parameters</h2>
    <div class="card">{_redistributed_section(redistributed)}</div>
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
```

### identification_report

HTML diagnostic report generation for dynamic identification results.

Renders the same statistics as `BaseIdentification.print_quality_report()` (condition number, torque residuals, base-parameter uncertainty, held-out validation, optional physical-consistency / reconstruction status) as a self-contained, shareable HTML page — the identification analogue of `tools/report.py`'s calibration report.

Identification and calibration produce structurally different diagnostics (a one-shot linear QR solve vs. an iterative nonlinear fit with outlier removal), so this is a separate adapter rather than a forced re-use of `generate_calibration_report` — see `docs/decisions/roadmap-mujoco-sysid-inspired-features.md` (Feature 1b) for the comparison that motivated this split. Only the domain-independent HTML/CSS layer is shared, via `tools/_report_common.py`.

## `generate_identification_report(identifier, output_path=None, title=None)`

Render a self-contained HTML diagnostic report for an identification run.

Reuses the metrics already computed by `BaseIdentification.print_quality_report()` — no new computation is performed here, only presentation.

Parameters:

| Name          | Type            | Description                                                                             | Default    |
| ------------- | --------------- | --------------------------------------------------------------------------------------- | ---------- |
| `identifier`  |                 | A BaseIdentification instance after solve() has been called (i.e. result is populated). | *required* |
| `output_path` | `Optional[str]` | If given, the HTML is also written to this path.                                        | `None`     |
| `title`       | `Optional[str]` | Optional report title. Defaults to the robot's class name.                              | `None`     |

Returns:

| Type  | Description                             |
| ----- | --------------------------------------- |
| `str` | The rendered HTML document as a string. |

Raises:

| Type             | Description                            |
| ---------------- | -------------------------------------- |
| `AttributeError` | If identifier has not been solved yet. |

Source code in `src/figaroh/tools/identification_report.py`

```
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
        raise AttributeError("No identification results available. Run solve() first.")

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

    provenance = getattr(identifier, "_run_provenance", None)
    report_title = title or (
        f"{_run_title(provenance, type(identifier).__name__)} "
        "— Identification Quality Report"
    )
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    from figaroh.tools.provenance import collect_splits
    from figaroh.tools.stages import stages_as_dicts

    stage_records = stages_as_dicts(identifier)
    splits = collect_splits(identifier)

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
    <div class="card">{_summary_section(result, correlation)}</div>
  </section>

  <section>
    <h2>Stages and data</h2>
    <div class="card">{_stages_section(stage_records, splits)}</div>
  </section>

  <section>
    <h2>Insights</h2>
    {_insights_section(insights)}
  </section>

  <section>
    <h2>Per-joint torque residuals</h2>
    <div class="card">{_per_joint_section(per_joint, result.get("selected"))}</div>
  </section>

  <section>
    <h2>Validation</h2>
    <div class="card">{_validation_section(validation)}</div>
  </section>

  <section>
    <h2>Before / after</h2>
    <div class="card">{_series_panel_section(
        validation, "identification", "series-panel"
    )}</div>
  </section>

  <section>
    <h2>Base-parameter uncertainty</h2>
    <div class="card">{_param_uncertainty_section(
        base_names,
        std_dev_abs,
        std_relative or [],
        list(phi_base) if phi_base is not None else None,
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
```

### compare_report

Static, offline, two-run comparison page (Feature 6, Phase C).

Unlike `tools/report.py` / `tools/identification_report.py`, this module does not render a specific run's data at generation time. It emits a self-contained HTML *shell* that a user opens directly in a browser and then loads two `*_verification.json` files into (drag-and- drop or a file picker) — the two exports produced by `BaseCalibration.export_verification_report()` / `BaseIdentification.export_verification_report()` (Step 2, extended in Step 3 with the `series`/`compat` fields this page reads). Everything happens client-side in JavaScript: no backend, no network request, no change to either base class (see the roadmap's Step 5 "Files (modified): none required").

Per D3 (roadmap), a compatibility check (same domain, same DOF/joint names, same `decimate` setting for identification, comparable sample counts) runs before anything is rendered; on mismatch the page blocks the comparison with a clear message and offers an explicit "compare anyway" override rather than silently overlaying incompatible runs.

## `generate_compare_page(output_path=None, title=None)`

Render the static, self-contained two-run comparison page.

Unlike the calibration/identification report generators, this takes no run object — it emits a template that loads two `export_verification_report()` JSON files client-side (drag-and- drop or a file picker) and, after a mandatory compatibility check (D3), renders a per-metric diff table and an overlaid before/after series chart. No network request, no backend: the returned document is fully self-contained and can be opened directly from disk.

Parameters:

| Name          | Type            | Description                                                  | Default |
| ------------- | --------------- | ------------------------------------------------------------ | ------- |
| `output_path` | `Optional[str]` | If given, the HTML is also written to this path.             | `None`  |
| `title`       | `Optional[str]` | Optional page title. Defaults to a generic comparison title. | `None`  |

Returns:

| Type  | Description                             |
| ----- | --------------------------------------- |
| `str` | The rendered HTML document as a string. |

Source code in `src/figaroh/tools/compare_report.py`

```
def generate_compare_page(
    output_path: Optional[str] = None, title: Optional[str] = None
) -> str:
    """Render the static, self-contained two-run comparison page.

    Unlike the calibration/identification report generators, this takes
    no run object — it emits a template that loads two
    ``export_verification_report()`` JSON files client-side (drag-and-
    drop or a file picker) and, after a mandatory compatibility check
    (D3), renders a per-metric diff table and an overlaid before/after
    series chart. No network request, no backend: the returned document
    is fully self-contained and can be opened directly from disk.

    Args:
        output_path: If given, the HTML is also written to this path.
        title: Optional page title. Defaults to a generic comparison title.

    Returns:
        The rendered HTML document as a string.
    """
    report_title = title or "FIGAROH — Two-Run Comparison"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{report_title}</title>
<style>{_STYLE}</style>
<style>{_COMPARE_STYLE}</style>
<script>{_COMPARE_CHART_SCRIPT}</script>
</head>
<body>
<div class="page">
  <h1>{report_title}</h1>
  <p class="subtitle">Generated {timestamp} &middot; static, offline —
    load two verification exports below. Nothing leaves this page.</p>

  <section>
    <h2>Load runs</h2>
    <div class="card compare-loader">
      <div class="drop-zone" id="zone-a" data-slot="a">
        <p class="drop-label">Run A</p>
        <input type="file" id="file-a" accept="application/json,.json">
        <p class="drop-filename" id="filename-a">No file loaded</p>
      </div>
      <div class="drop-zone" id="zone-b" data-slot="b">
        <p class="drop-label">Run B</p>
        <input type="file" id="file-b" accept="application/json,.json">
        <p class="drop-filename" id="filename-b">No file loaded</p>
      </div>
    </div>
  </section>

  <section id="compat-section" style="display:none;">
    <h2>Compatibility</h2>
    <div class="card">
      <div id="compat-status"></div>
      <div id="force-compare-row">
        <label>
          <input type="checkbox" id="force-compare">
          Compare anyway (results may be misleading)
        </label>
      </div>
    </div>
  </section>

  <section id="results-section" style="display:none;">
    <div class="forced-banner" id="forced-banner">
      These runs failed the compatibility check above — the comparison
      below was forced and may be misleading.
    </div>

    <h2>Metric comparison</h2>
    <div class="card">
      <table class="data">
        <thead>
          <tr>
            <th>Metric</th><th>Run A</th><th>Run B</th>
            <th>&Delta;</th><th>% change</th>
          </tr>
        </thead>
        <tbody id="diff-table-body"></tbody>
      </table>
    </div>

    <h2>Series overlay</h2>
    <div class="card" id="compare-series-card">
      <div class="series-controls">
        <label><input type="checkbox" id="compare-toggle-a" checked>
          Run A</label>
        <label><input type="checkbox" id="compare-toggle-b" checked>
          Run B</label>
        <label for="compare-select">Show:</label>
        <select id="compare-select"></select>
        <button type="button" id="compare-reset">Reset zoom</button>
      </div>
      <svg id="compare-svg" class="series-svg"></svg>
      <div class="series-legend" id="compare-legend"></div>
      <p class="series-hint">Scroll to zoom, hover to inspect.</p>
      <div id="compare-tooltip" class="series-tooltip"></div>
    </div>
  </section>

  <footer>Generated by figaroh.tools.compare_report — static, offline,
    no backend.</footer>
</div>
<script>{_COMPARE_DRIVER_SCRIPT}</script>
</body>
</html>
"""

    if output_path is not None:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(doc)

    return doc
```

### stages

`StageResult` records: which step (data, fit, validation, physical, export) ran, failed, fell back or did not run (#55).

Which step produced a result, and whether it ran, failed or fell back.

See docs/decisions/data-result-contract.md (#55). Identification and calibration record one :class:`StageResult` per step they run on `obj.stages`; results dictionaries carry them under `"stages"` and the run archive writes them to `stages.json`. A stage that did not run has no record, or an explicit `not_run` when that is worth stating.

## `record_stage(obj, stage, status, reason='', metrics=None, artifacts=None)`

Record `stage` on `obj.stages`, replacing an earlier record of it.

`metrics` maps a name to `(value, unit)` or to a bare value (unit `""`).

Source code in `src/figaroh/tools/stages.py`

```
def record_stage(
    obj,
    stage: str,
    status: str,
    reason: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[str]] = None,
) -> StageResult:
    """Record ``stage`` on ``obj.stages``, replacing an earlier record of it.

    ``metrics`` maps a name to ``(value, unit)`` or to a bare value (unit
    ``""``).
    """
    clean = {}
    for name, value in (metrics or {}).items():
        if isinstance(value, tuple):
            value, unit = value
        else:
            unit = ""
        clean[name] = {"value": value, "unit": unit}
    result = StageResult(stage, status, reason, clean, list(artifacts or []))
    stages = [s for s in getattr(obj, "stages", None) or [] if s.stage != stage]
    stages.append(result)
    stages.sort(key=lambda s: STAGES.index(s.stage))
    obj.stages = stages
    return result
```

## `drop_stage(obj, stage)`

Remove the record of `stage` from `obj.stages` (a no-op if none).

Source code in `src/figaroh/tools/stages.py`

```
def drop_stage(obj, stage: str) -> None:
    """Remove the record of ``stage`` from ``obj.stages`` (a no-op if none)."""
    obj.stages = [s for s in getattr(obj, "stages", None) or [] if s.stage != stage]
```

## `stages_as_dicts(obj)`

`obj.stages` as plain dicts (for results dictionaries and JSON).

Source code in `src/figaroh/tools/stages.py`

```
def stages_as_dicts(obj) -> List[Dict[str, Any]]:
    """``obj.stages`` as plain dicts (for results dictionaries and JSON)."""
    return [asdict(s) for s in getattr(obj, "stages", None) or []]
```

## `with_schema(obj, verdict)`

A verdict dict with `schema_version` and the run's stage records.

The records go under `stage_records`: the verdict's own `stages` is the per-stage status map and must not be replaced (#63).

Source code in `src/figaroh/tools/stages.py`

```
def with_schema(obj, verdict: Dict[str, Any]) -> Dict[str, Any]:
    """A verdict dict with ``schema_version`` and the run's stage records.

    The records go under ``stage_records``: the verdict's own ``stages`` is
    the per-stage status map and must not be replaced (#63).
    """
    return {
        "schema_version": SCHEMA_VERSION,
        **verdict,
        "stage_records": stages_as_dicts(obj),
    }
```

## `apply_to_verdict(verdict, obj, selected_stage='fit')`

Separate per-stage verdicts from the run's records (#63).

`verdict.stages` keeps its scoped entries (`numerical_execution`, `prediction`, `solver`) and gains `data`, `fit`, `validation`, `physical` and `export` from the records: `pass`, `fail`, `fallback` (validation on training data: not held-out evidence) or `not_run`; a stage with no record keeps `not_evaluated`. `data_provenance` mirrors `data`.

Source code in `src/figaroh/tools/stages.py`

```
def apply_to_verdict(verdict, obj, selected_stage: str = "fit") -> None:
    """Separate per-stage verdicts from the run's records (#63).

    ``verdict.stages`` keeps its scoped entries (``numerical_execution``,
    ``prediction``, ``solver``) and gains ``data``, ``fit``,
    ``validation``, ``physical`` and ``export`` from the records: ``pass``,
    ``fail``, ``fallback`` (validation on training data: not held-out
    evidence) or ``not_run``; a stage with no record keeps
    ``not_evaluated``. ``data_provenance`` mirrors ``data``.
    """
    from figaroh.tools.provenance import collect_splits

    records = {s.stage: s for s in getattr(obj, "stages", None) or []}
    for stage in STAGES:
        if stage in records:
            verdict.stages[stage] = _VERDICT_STATUS[records[stage].status]
        else:
            verdict.stages.setdefault(stage, "not_evaluated")
    verdict.stages["data_provenance"] = verdict.stages["data"]
    verdict.stage_records = stages_as_dicts(obj)
    fit = records.get("fit")
    verdict.selected_stage = (
        selected_stage if fit is not None and fit.status == "ok" else "none"
    )
    verdict.splits = collect_splits(obj)
```

## `stages_line(obj)`

One-line terminal summary, e.g. `data ok · fit ok · validation fallback (training data, not held-out)`.

Source code in `src/figaroh/tools/stages.py`

```
def stages_line(obj) -> str:
    """One-line terminal summary, e.g. ``data ok · fit ok · validation
    fallback (training data, not held-out)``."""
    parts = []
    for s in getattr(obj, "stages", None) or []:
        text = f"{s.stage} {s.status}"
        if s.stage == "validation" and s.status == "fallback":
            text += " (training data, not held-out)"
        elif s.status == "failed" and s.reason:
            text += f" ({s.reason})"
        parts.append(text)
    return " · ".join(parts) if parts else "no stage records"
```

### \_report_common

Shared HTML/CSS primitives and the `VerificationVerdict`/`ThresholdCheck` dataclasses used by both report generators.

Shared HTML/CSS primitives for the calibration and identification diagnostic reports (`tools/report.py` and `tools/identification_report.py`).

Calibration and identification produce structurally different diagnostics (iterative nonlinear fit with outlier removal vs. one-shot linear QR solve), so each gets its own report module — but both render as a self-contained HTML page with the same look, the same escaping rules, and the same confidence-tier vocabulary. This module holds only that shared, domain-independent layer.

## `ThresholdCheck(name, value, threshold, comparison, passed, status='pass', required=True, reason='')`

One metric checked against one threshold.

## `VerificationVerdict(passed, checks, metrics, insights=list(), metadata=dict(), series=dict(), compat=dict(), status='not_evaluated', scope='thresholds', stages=dict(), policy=dict(), stage_records=list(), selected_stage='fit', splits=dict())`

Machine-checkable pass/fail against a set of quality thresholds.

Produced by :func:`evaluate_thresholds`; `insights`/`metadata`/ `series`/`compat` are filled in by the caller (`BaseCalibration.verify()` / `BaseIdentification.verify()`) after construction, since those are domain-specific / provenance data rather than threshold arithmetic. `metadata` holds the full nested provenance record from :func:`figaroh.tools.provenance.collect_run_provenance` (run id, asset identity, nominal model, config, software, data, timestamps).

## `evaluate_thresholds(metrics, thresholds)`

Evaluate explicit limits; missing required evidence cannot pass.

Specs accept `required=False` for advisory checks. Missing metrics are recorded as not evaluated; computed nonfinite values fail. Empty policies have no acceptance evidence and return not evaluated, with passed=False.

Source code in `src/figaroh/tools/_report_common.py`

```
def evaluate_thresholds(metrics, thresholds):
    """Evaluate explicit limits; missing required evidence cannot pass.

    Specs accept ``required=False`` for advisory checks. Missing metrics are
    recorded as not evaluated; computed nonfinite values fail. Empty policies
    have no acceptance evidence and return not evaluated, with passed=False.
    """
    checks = []
    for name, spec in thresholds.items():
        threshold = float(spec["threshold"])
        comparison = spec["comparison"]
        if comparison not in ("min", "max"):
            raise ValueError(f"Unknown comparison {comparison!r} for {name!r}")
        if not math.isfinite(threshold):
            raise ValueError(f"Threshold for {name!r} must be finite")
        value = metrics.get(name)
        status, reason = "not_evaluated", "Required metric is unavailable"
        if value is not None:
            value = float(value)
            if not math.isfinite(value):
                status, reason = "fail", "Computed metric is nonfinite"
            else:
                passed = (
                    value <= threshold if comparison == "max" else value >= threshold
                )
                status, reason = ("pass" if passed else "fail"), ""
        checks.append(
            ThresholdCheck(
                name,
                value,
                threshold,
                comparison,
                status == "pass",
                status,
                bool(spec.get("required", True)),
                reason,
            )
        )
    status = _check_status(checks)
    return VerificationVerdict(
        status == "pass", checks, dict(metrics), status=status, policy=dict(thresholds)
    )
```

## `scoped_verification(metrics, thresholds, scope, arrays, independent, prediction_keys, solver_success=None, facts=None)`

Build a scoped verdict without claiming unrequested model stages.

Execution verifies finite, nonempty numerical outputs. Prediction also requires independent measurements and explicit error limits for every requested output. Neither scope certifies physical feasibility or export.

`facts` maps a name to `None` (not evaluated), a bool, or a `(bool, reason)` tuple whose reason is the message of a failing fact.

Source code in `src/figaroh/tools/_report_common.py`

```
def scoped_verification(
    metrics,
    thresholds,
    scope,
    arrays,
    independent,
    prediction_keys,
    solver_success=None,
    facts=None,
):
    """Build a scoped verdict without claiming unrequested model stages.

    Execution verifies finite, nonempty numerical outputs. Prediction also
    requires independent measurements and explicit error limits for every
    requested output. Neither scope certifies physical feasibility or export.

    ``facts`` maps a name to ``None`` (not evaluated), a bool, or a
    ``(bool, reason)`` tuple whose reason is the message of a failing fact.
    """
    import numpy as np

    if scope not in ("execution", "prediction"):
        raise ValueError("Verification scope must be 'execution' or 'prediction'")
    verdict = evaluate_thresholds(metrics, thresholds)
    guard = []
    for name, array in arrays.items():
        status, reason = "not_evaluated", "Numerical evidence is unavailable"
        if array is not None:
            values = np.asarray(array, dtype=float)
            valid = values.size > 0 and np.all(np.isfinite(values))
            status, reason = (
                ("pass", "")
                if valid
                else ("fail", "Empty or nonfinite numerical output")
            )
        guard.append(
            ThresholdCheck(
                name,
                None if status == "not_evaluated" else float(status == "pass"),
                1.0,
                "min",
                status == "pass",
                status,
                True,
                reason,
            )
        )
    for name, value in (facts or {}).items():
        # a fact is None (unknown), a bool, or ``(bool, reason)`` whose
        # reason is the message when it fails
        reason = "Numerical dimensions are missing or inconsistent"
        if isinstance(value, tuple):
            value, reason = value
        status = "not_evaluated" if value is None else ("pass" if value else "fail")
        guard.append(
            ThresholdCheck(
                name,
                None if status == "not_evaluated" else float(status == "pass"),
                1.0,
                "min",
                status == "pass",
                status,
                True,
                reason if status != "pass" else "",
            )
        )
    if solver_success is not None:
        status = "pass" if solver_success else "fail"
        guard.append(
            ThresholdCheck(
                "solver_success",
                float(solver_success),
                1.0,
                "min",
                bool(solver_success),
                status,
            )
        )
    numerical_status = _check_status(guard)
    verdict.checks = guard + verdict.checks
    prediction_status = "not_evaluated"
    if scope == "prediction":
        for name, satisfied, reason in [
            (
                "independent_validation",
                independent,
                "No independently loaded validation data",
            ),
            (
                "prediction_error_policy",
                bool(prediction_keys)
                and all(
                    k in thresholds
                    and thresholds[k].get("required", True)
                    and thresholds[k]["comparison"] == "max"
                    and float(thresholds[k]["threshold"]) >= 0
                    for k in prediction_keys
                ),
                "Specify nonnegative per-output maximum error limits before acceptance",
            ),
        ]:
            status = "pass" if satisfied else "not_evaluated"
            verdict.checks.append(
                ThresholdCheck(
                    name,
                    1.0 if satisfied else None,
                    1.0,
                    "min",
                    satisfied,
                    status,
                    True,
                    "" if satisfied else reason,
                )
            )
        prediction_status = _check_status(verdict.checks)
    verdict.status = _check_status(verdict.checks)
    verdict.passed = verdict.status == "pass"
    verdict.scope = scope
    verdict.stages = {
        "numerical_execution": numerical_status,
        "prediction": prediction_status,
        "data_provenance": "not_evaluated",
        "solver": (
            ("pass" if solver_success else "fail")
            if solver_success is not None
            else "not_evaluated"
        ),
        "physical": "not_evaluated",
        "export": "not_evaluated",
    }
    return verdict
```

## `verification_json_data(data)`

Keep invalid evidence readable in strict JSON; status retains failure.

Source code in `src/figaroh/tools/_report_common.py`

```
def verification_json_data(data):
    """Keep invalid evidence readable in strict JSON; status retains failure."""
    import numpy as np

    if isinstance(data, dict):
        return {k: verification_json_data(v) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return [verification_json_data(v) for v in data]
    if isinstance(data, np.ndarray):
        return verification_json_data(data.tolist())
    if isinstance(data, np.generic):
        return verification_json_data(data.item())
    if isinstance(data, float) and not math.isfinite(data):
        return None
    return data
```

## `build_provenance_metadata(config_file_path, robot_name)`

Git commit, config file hash, timestamp, robot name — for a :class:`VerificationVerdict`'s `metadata` field.

Source code in `src/figaroh/tools/_report_common.py`

```
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
```

### provenance

Run provenance metadata — git commit, config hash, timestamps, robot/asset identity — attached to every calibration/identification run.

Run provenance capture — the single source of truth for "what produced this fit": the nominal reference model, the exact config used, software versions, input data files, and (if configured) the physical asset identity. Consumed identically by the terminal report, the HTML report, the JSON verification verdict, and the run archive, so the four can never disagree about what produced a given result.

Reuses :func:`figaroh.tools._report_common._git_commit_hash` and :func:`figaroh.tools._report_common._config_file_sha256` rather than duplicating hashing logic.

## `collect_splits(obj)`

Which data trained and which validated this run (#63).

From the data-contract objects when the run used them (`trajectory` / `observations` and their validation counterparts: files with sha256, session, adapter); `"legacy input"` otherwise, whose files are listed under provenance `data`. `validation_source` comes from the run's `validation` stage: `held_out`, or `training_fallback` when no held-out data was available.

Source code in `src/figaroh/tools/provenance.py`

```
def collect_splits(obj: Any) -> Dict[str, Any]:
    """Which data trained and which validated this run (#63).

    From the data-contract objects when the run used them
    (``trajectory`` / ``observations`` and their validation counterparts:
    files with sha256, session, adapter); ``"legacy input"`` otherwise,
    whose files are listed under provenance ``data``. ``validation_source``
    comes from the run's ``validation`` stage: ``held_out``, or
    ``training_fallback`` when no held-out data was available.
    """

    def describe(data):
        if data is None:
            return None
        source = data.source
        return {
            "files": dict(source.files),
            "session": source.session.id if source.session else None,
            "adapter": source.adapter,
            "samples": int(data.n_samples),
            "masked_samples": int((~data.mask).sum()),
        }

    training = getattr(obj, "trajectory", None) or getattr(obj, "observations", None)
    validation = getattr(obj, "_val_trajectory", None) or getattr(
        obj, "_val_observations", None
    )
    status = {s.stage: s.status for s in getattr(obj, "stages", None) or []}
    source = {"ok": "held_out", "fallback": "training_fallback"}.get(
        status.get("validation"), "none"
    )
    return {
        "training": describe(training) or "legacy input (see provenance data)",
        "validation": describe(validation),
        "validation_source": source,
    }
```

## `collect_run_provenance(obj, task)`

Build the full provenance record for one V&V run.

Parameters:

| Name   | Type  | Description                                                                                                                                                                                                                                                                                                             | Default    |
| ------ | ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `obj`  | `Any` | A BaseIdentification or BaseCalibration instance (or duck-typed equivalent) after its config has been loaded — i.e. safe to call any time after load_param(), though it is normally captured once in \_store_results/ \_store_optimization_results right after solve() finishes computing, so run_finished is accurate. | *required* |
| `task` | `str` | "identification" or "calibration" — selects which curated config-key allowlist to surface under config.values.                                                                                                                                                                                                          | *required* |

Returns:

| Type             | Description                                                     |
| ---------------- | --------------------------------------------------------------- |
| `Dict[str, Any]` | A JSON-serializable nested dict (see module docstring for the   |
| `Dict[str, Any]` | four consumers). Never raises: every sub-lookup is best-effort. |

Source code in `src/figaroh/tools/provenance.py`

```
def collect_run_provenance(obj: Any, task: str) -> Dict[str, Any]:
    """Build the full provenance record for one V&V run.

    Args:
        obj: A ``BaseIdentification`` or ``BaseCalibration`` instance
            (or duck-typed equivalent) after its config has been loaded
            — i.e. safe to call any time after ``load_param()``, though
            it is normally captured once in ``_store_results``/
            ``_store_optimization_results`` right after ``solve()``
            finishes computing, so ``run_finished`` is accurate.
        task: ``"identification"`` or ``"calibration"`` — selects which
            curated config-key allowlist to surface under
            ``config.values``.

    Returns:
        A JSON-serializable nested dict (see module docstring for the
        four consumers). Never raises: every sub-lookup is best-effort.
    """
    config = (
        getattr(obj, "identif_config", None) or getattr(obj, "calib_config", None) or {}
    )
    config_keys = (
        _IDENTIFICATION_CONFIG_KEYS
        if task == "identification"
        else _CALIBRATION_CONFIG_KEYS
    )

    asset = _asset_identity(config)
    now = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit_hash()
    run_started = getattr(obj, "_run_started_at", None) or now
    run_finished = getattr(obj, "_run_finished_at", None) or now

    run_id = "_".join(
        [
            asset["asset_id"].replace(" ", "-"),
            task,
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            git_commit[:8] if git_commit != "unknown" else "nogit",
        ]
    )

    config_path = getattr(obj, "_config_file_path", None)

    return {
        "run_id": run_id,
        "task": task,
        "asset": asset,
        "model": _model_identity(obj),
        "config": {
            "path": config_path or "unavailable",
            "sha256": _config_file_sha256(config_path),
            "values": _config_values(config, config_keys),
        },
        "software": {
            **_software_versions(),
            # working directory's repository (an example run: examples)
            "git_commit": git_commit,
            "git_dirty": _git_dirty(),
            # the figaroh package's own checkout, the other half of the pair
            "figaroh_revision": _package_revision(),
            "hostname": socket.gethostname(),
        },
        "data": _data_files_provenance(config),
        "timestamps": {
            "run_started": run_started,
            "run_finished": run_finished,
            "report_generated": now,
        },
    }
```

### run_archive

Writes each run to a timestamped `results/runs/<robot>/<task>/<timestamp>/` directory (config snapshot, provenance JSON, report) instead of overwriting a single `results/` path.

Longitudinal run archive — a fleet's V&V history.

`export_html_report()`/`export_verification_report()` write to fixed filenames that get overwritten by the next run, so a robot calibrated monthly only ever has its most recent report on disk. :func:`archive_run` instead writes each run to its own immutable, timestamped directory under `results/runs/{asset_id}/{task}/{timestamp}_{git_short}/` and appends a one-line summary to `results/runs/index.jsonl` — a zero-dependency, git-diffable, `jq`-able index of every run ever performed, per asset.

Call after :meth:`~figaroh.identification.base_identification.BaseIdentification.solve` (and, typically, after `export_html_report()`/`export_verification_report()` so their output gets copied in alongside the raw provenance/parameters).

## `compute_run_dir(obj, root='results/runs')`

Compute this run's dedicated directory path from provenance.

Deterministic for a given `obj._run_provenance` — safe to call more than once (e.g. once for the HTML report's output_path, again for the verdict's) and get back the identical path, since the timestamp is read from the already-fixed provenance record, never re-derived from wall clock.

Parameters:

| Name   | Type  | Description                                                                                                           | Default          |
| ------ | ----- | --------------------------------------------------------------------------------------------------------------------- | ---------------- |
| `obj`  | `Any` | A BaseIdentification/BaseCalibration instance after :meth:solve has been called (i.e. \_run_provenance is populated). | *required*       |
| `root` | `str` | Archive root directory (created if missing).                                                                          | `'results/runs'` |

Returns:

| Name   | Type   | Description                                     |
| ------ | ------ | ----------------------------------------------- |
| `Path` | `Path` | The directory for this run, created if missing. |

Raises:

| Type             | Description                   |
| ---------------- | ----------------------------- |
| `AttributeError` | If called before :meth:solve. |

Source code in `src/figaroh/tools/run_archive.py`

```
def compute_run_dir(obj: Any, root: str = "results/runs") -> Path:
    """Compute this run's dedicated directory path from provenance.

    Deterministic for a given ``obj._run_provenance`` — safe to call more
    than once (e.g. once for the HTML report's output_path, again for the
    verdict's) and get back the identical path, since the timestamp is
    read from the already-fixed provenance record, never re-derived from
    wall clock.

    Args:
        obj: A ``BaseIdentification``/``BaseCalibration`` instance after
            :meth:`solve` has been called (i.e. ``_run_provenance`` is
            populated).
        root: Archive root directory (created if missing).

    Returns:
        Path: The directory for this run, created if missing.

    Raises:
        AttributeError: If called before :meth:`solve`.
    """
    provenance = getattr(obj, "_run_provenance", None)
    if provenance is None:
        raise AttributeError("No run provenance available. Run solve() first.")

    task = provenance.get("task", "unknown")
    asset_id = provenance.get("asset", {}).get("asset_id", "unknown")

    run_finished = provenance.get("timestamps", {}).get("run_finished")
    try:
        dt = datetime.fromisoformat(run_finished)
    except (TypeError, ValueError):
        dt = datetime.now(timezone.utc)
    ts_compact = dt.strftime("%Y%m%dT%H%M%SZ")
    git_short = provenance.get("software", {}).get("git_commit", "nogit")[:8]

    root_path = Path(root)
    run_dir = (
        root_path
        / _safe_path_component(asset_id)
        / _safe_path_component(task)
        / f"{ts_compact}_{_safe_path_component(git_short)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
```

## `archive_run(obj, run_dir)`

Finalize archiving into an already-computed run_dir.

Writes provenance.json / config.snapshot.yaml / parameters.csv, and appends the index.jsonl entry (reading run_dir/verdict.json for pass/metrics if the caller already wrote one there).

Parameters:

| Name      | Type   | Description                                                                                                                                                                                              | Default    |
| --------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `obj`     | `Any`  | A BaseIdentification/BaseCalibration instance after :meth:solve has been called (i.e. \_run_provenance is populated).                                                                                    | *required* |
| `run_dir` | `Path` | The directory for this run (typically obtained via :func:compute_run_dir). Expected to already exist; the caller should have written report.html and/or verdict.json there before calling this function. | *required* |

Returns:

| Name  | Type  | Description                                                             |
| ----- | ----- | ----------------------------------------------------------------------- |
| `str` | `str` | The path to this run's archive directory (echoed back for convenience). |

Raises:

| Type             | Description                   |
| ---------------- | ----------------------------- |
| `AttributeError` | If called before :meth:solve. |

Source code in `src/figaroh/tools/run_archive.py`

```
def archive_run(obj: Any, run_dir: Path) -> str:
    """Finalize archiving into an already-computed run_dir.

    Writes provenance.json / config.snapshot.yaml / parameters.csv, and
    appends the index.jsonl entry (reading run_dir/verdict.json for
    pass/metrics if the caller already wrote one there).

    Args:
        obj: A ``BaseIdentification``/``BaseCalibration`` instance after
            :meth:`solve` has been called (i.e. ``_run_provenance`` is
            populated).
        run_dir: The directory for this run (typically obtained via
            :func:`compute_run_dir`). Expected to already exist; the
            caller should have written ``report.html`` and/or
            ``verdict.json`` there before calling this function.

    Returns:
        str: The path to this run's archive directory (echoed back for
            convenience).

    Raises:
        AttributeError: If called before :meth:`solve`.
    """
    provenance = getattr(obj, "_run_provenance", None)
    if provenance is None:
        raise AttributeError("No run provenance available. Run solve() first.")

    with open(run_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    config = getattr(obj, "identif_config", None) or getattr(obj, "calib_config", None)
    if config:
        with open(run_dir / "config.snapshot.yaml", "w") as f:
            yaml.dump(_yaml_safe(config), f, default_flow_style=False, sort_keys=True)

    stages = getattr(obj, "stages", None)
    if stages:
        from figaroh.tools.stages import SCHEMA_VERSION, stages_as_dicts

        with open(run_dir / "stages.json", "w") as f:
            json.dump(
                {"schema_version": SCHEMA_VERSION, "stages": stages_as_dicts(obj)},
                f,
                indent=2,
                default=str,
            )

    param_names, param_values = _extract_parameters(obj)
    if param_names:
        with open(run_dir / "parameters.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerows(zip(param_names, param_values))

    result = getattr(obj, "result", None)
    if isinstance(result, dict) and isinstance(result.get("selected"), dict):
        # the base-parameter fit stays available beside the selected estimate
        with open(run_dir / "fit_parameters.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerows(
                zip(
                    result.get("base parameters names", []),
                    result.get("base parameters values", []),
                )
            )

    # Determine the root for index (run_dir is .../root/asset/task/timestamp/)
    root_path = run_dir.parent.parent.parent
    _append_index(root_path, provenance, run_dir)
    logger.info(f"Run archived to {run_dir}")
    return str(run_dir)
```

## Linear Solver

Multivariate linear solver for robot parameter identification.

This module provides advanced linear regression solvers optimized for overdetermined systems (thin, tall matrices) with support for:

- Multiple regularization methods (Ridge, Lasso, Elastic Net, Tikhonov)
- Linear equality and inequality constraints
- Robust regression methods
- Physical parameter bounds
- Iterative refinement for improved accuracy

## `LinearSolver(method='lstsq', regularization=None, alpha=0.0, l1_ratio=0.5, weights=None, constraints=None, bounds=None, max_iter=1000, tol=1e-06, verbose=False)`

Advanced linear solver for overdetermined systems: Ax = b

Optimized for robot identification where A is typically:

- Dense (not sparse)
- Large (many samples)
- Thin (more rows than columns, overdetermined)

Attributes:

| Name             | Type    | Description                    |
| ---------------- | ------- | ------------------------------ |
| `method`         | `str`   | Solving method to use          |
| `regularization` | `str`   | Type of regularization         |
| `alpha`          | `float` | Regularization strength        |
| `constraints`    | `dict`  | Linear constraints on solution |
| `bounds`         | `tuple` | Box constraints on variables   |
| `solver_info`    | `dict`  | Information about the solution |

Initialize linear solver.

Parameters:

| Name             | Type      | Description                                                                                                                                                                                                               | Default   |
| ---------------- | --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| `method`         | `str`     | Solving method (see METHODS)                                                                                                                                                                                              | `'lstsq'` |
| `regularization` | `str`     | 'l1', 'l2', 'elastic_net', or None                                                                                                                                                                                        | `None`    |
| `alpha`          | `float`   | Regularization strength (>=0)                                                                                                                                                                                             | `0.0`     |
| `l1_ratio`       | `float`   | Elastic net mixing (0=Ridge, 1=Lasso)                                                                                                                                                                                     | `0.5`     |
| `weights`        | `ndarray` | Sample weights for weighted least squares                                                                                                                                                                                 | `None`    |
| `constraints`    | `dict`    | Linear constraints: - 'A_eq': Equality constraint matrix (A_eq @ x = b_eq) - 'b_eq': Equality constraint vector - 'A_ineq': Inequality constraint matrix (A_ineq @ x \<= b_ineq) - 'b_ineq': Inequality constraint vector | `None`    |
| `bounds`         | `tuple`   | Box constraints (lower, upper) for each variable                                                                                                                                                                          | `None`    |
| `max_iter`       | `int`     | Maximum iterations for iterative methods                                                                                                                                                                                  | `1000`    |
| `tol`            | `float`   | Convergence tolerance                                                                                                                                                                                                     | `1e-06`   |
| `verbose`        | `bool`    | Print solver information                                                                                                                                                                                                  | `False`   |

Source code in `src/figaroh/tools/solver.py`

```
def __init__(
    self,
    method="lstsq",
    regularization=None,
    alpha=0.0,
    l1_ratio=0.5,
    weights=None,
    constraints=None,
    bounds=None,
    max_iter=1000,
    tol=1e-6,
    verbose=False,
):
    """
    Initialize linear solver.

    Args:
        method (str): Solving method (see METHODS)
        regularization (str): 'l1', 'l2', 'elastic_net', or None
        alpha (float): Regularization strength (>=0)
        l1_ratio (float): Elastic net mixing (0=Ridge, 1=Lasso)
        weights (ndarray): Sample weights for weighted least squares
        constraints (dict): Linear constraints:
            - 'A_eq': Equality constraint matrix (A_eq @ x = b_eq)
            - 'b_eq': Equality constraint vector
            - 'A_ineq': Inequality constraint matrix (A_ineq @ x <= b_ineq)
            - 'b_ineq': Inequality constraint vector
        bounds (tuple): Box constraints (lower, upper) for each variable
        max_iter (int): Maximum iterations for iterative methods
        tol (float): Convergence tolerance
        verbose (bool): Print solver information
    """
    if method not in self.METHODS:
        raise ValueError(f"Method must be one of {self.METHODS}")

    self.method = method
    self.regularization = regularization
    self.alpha = alpha
    self.l1_ratio = l1_ratio
    self.weights = weights
    self.constraints = constraints or {}
    self.bounds = bounds
    self.max_iter = max_iter
    self.tol = tol
    self.verbose = verbose
    self.solver_info = {}
```

### `solve(A, b)`

Solve the linear system Ax = b.

Parameters:

| Name | Type      | Description                           | Default    |
| ---- | --------- | ------------------------------------- | ---------- |
| `A`  | `ndarray` | Design matrix (n_samples, n_features) | *required* |
| `b`  | `ndarray` | Target vector (n_samples,)            | *required* |

Returns:

| Name      | Type | Description                     |
| --------- | ---- | ------------------------------- |
| `ndarray` |      | Solution vector x (n_features,) |

Raises:

| Type          | Description                |
| ------------- | -------------------------- |
| `ValueError`  | If inputs are incompatible |
| `LinAlgError` | If solution fails          |

Source code in `src/figaroh/tools/solver.py`

```
def solve(self, A, b):
    """
    Solve the linear system Ax = b.

    Args:
        A (ndarray): Design matrix (n_samples, n_features)
        b (ndarray): Target vector (n_samples,)

    Returns:
        ndarray: Solution vector x (n_features,)

    Raises:
        ValueError: If inputs are incompatible
        np.linalg.LinAlgError: If solution fails
    """
    # Validate inputs
    A, b = self._validate_inputs(A, b)

    # Select and execute solving method
    if self.method == "lstsq":
        x = self._solve_lstsq(A, b)
    elif self.method == "qr":
        x = self._solve_qr(A, b)
    elif self.method == "svd":
        x = self._solve_svd(A, b)
    elif self.method == "ridge":
        x = self._solve_ridge(A, b)
    elif self.method == "lasso":
        x = self._solve_lasso(A, b)
    elif self.method == "elastic_net":
        x = self._solve_elastic_net(A, b)
    elif self.method == "tikhonov":
        x = self._solve_tikhonov(A, b)
    elif self.method == "constrained":
        x = self._solve_constrained(A, b)
    elif self.method == "robust":
        x = self._solve_robust(A, b)
    elif self.method == "weighted":
        x = self._solve_weighted(A, b)
    else:
        raise ValueError(f"Method {self.method} not implemented")

    # Compute solution quality metrics
    self._compute_solution_quality(A, b, x)

    if self.verbose:
        self._print_solution_info()

    return x
```

## `solve_linear_system(A, b, method='lstsq', **kwargs)`

Convenience function to solve linear system Ax = b.

Parameters:

| Name       | Type      | Description                           | Default    |
| ---------- | --------- | ------------------------------------- | ---------- |
| `A`        | `ndarray` | Design matrix (n_samples, n_features) | *required* |
| `b`        | `ndarray` | Target vector (n_samples,)            | *required* |
| `method`   | `str`     | Solving method                        | `'lstsq'`  |
| `**kwargs` |           | Additional arguments for LinearSolver | `{}`       |

Returns:

| Name    | Type | Description                    |
| ------- | ---- | ------------------------------ |
| `tuple` |      | (solution x, solver_info dict) |

Example

> > > A = np.random.randn(1000, 50) # 1000 samples, 50 features b = np.random.randn(1000) x, info = solve_linear_system(A, b, method='ridge', alpha=0.1)

Source code in `src/figaroh/tools/solver.py`

```
def solve_linear_system(A, b, method="lstsq", **kwargs):
    """
    Convenience function to solve linear system Ax = b.

    Args:
        A (ndarray): Design matrix (n_samples, n_features)
        b (ndarray): Target vector (n_samples,)
        method (str): Solving method
        **kwargs: Additional arguments for LinearSolver

    Returns:
        tuple: (solution x, solver_info dict)

    Example:
        >>> A = np.random.randn(1000, 50)  # 1000 samples, 50 features
        >>> b = np.random.randn(1000)
        >>> x, info = solve_linear_system(A, b, method='ridge', alpha=0.1)
    """
    solver = LinearSolver(method=method, **kwargs)
    x = solver.solve(A, b)
    return x, solver.solver_info
```

The linear solver provides comprehensive solving methods for robot parameter identification:

- **Basic methods**: lstsq, QR decomposition, SVD
- **Regularized methods**: Ridge (L2), Lasso (L1), Elastic Net, Tikhonov
- **Advanced methods**: Constrained optimization, robust regression, weighted least squares
- **Constraint support**: Box constraints, linear equality/inequality constraints
- **Quality metrics**: RMSE, R², condition number, residual analysis

## Robot Management

## `Robot(robot_urdf, package_dirs, isFext=False, freeflyer_ori=None, freeflyer_limits=None)`

Bases: `RobotWrapper`

Enhanced Robot class with advanced free-flyer support and visualization utilities.

This class extends RobotWrapper with:

- Sophisticated free-flyer configuration and limits
- Integrated visualization support
- Convenient API for common robot operations
- Enhanced error handling and validation

Initialize enhanced robot model from URDF.

Parameters:

| Name               | Type                            | Description                                                | Default    |
| ------------------ | ------------------------------- | ---------------------------------------------------------- | ---------- |
| `robot_urdf`       | `str`                           | Path to URDF file                                          | *required* |
| `package_dirs`     | `str`                           | Package directories for mesh files                         | *required* |
| `isFext`           | `bool`                          | Whether to add floating base joint                         | `False`    |
| `freeflyer_ori`    | `Optional[ndarray]`             | Optional 3x3 rotation matrix for floating base orientation | `None`     |
| `freeflyer_limits` | `Optional[Tuple[float, float]]` | Optional custom limits for free-flyer (default: (-1, 1))   | `None`     |

Source code in `src/figaroh/tools/robot.py`

```
def __init__(
    self,
    robot_urdf: str,
    package_dirs: str,
    isFext: bool = False,
    freeflyer_ori: Optional[np.ndarray] = None,
    freeflyer_limits: Optional[Tuple[float, float]] = None,
):
    """Initialize enhanced robot model from URDF.

    Args:
        robot_urdf: Path to URDF file
        package_dirs: Package directories for mesh files
        isFext: Whether to add floating base joint
        freeflyer_ori: Optional 3x3 rotation matrix for floating base orientation
        freeflyer_limits: Optional custom limits for free-flyer (default: (-1, 1))
    """
    self.isFext = isFext
    self.robot_urdf = robot_urdf
    self._freeflyer_limits = freeflyer_limits or self.DEFAULT_FREEFLYER_LIMITS

    # Initialize robot model
    root_joint = pin.JointModelFreeFlyer() if isFext else None
    self.initFromURDF(robot_urdf, package_dirs=package_dirs, root_joint=root_joint)

    # Configure free-flyer with advanced options
    if isFext:
        self._configure_freeflyer(freeflyer_ori)

    # Set aliases for backward compatibility
    self.geom_model = self.collision_model

    # Lazy backend for backend-aware computations
    self._backend = None
```

### `num_joints`

Number of actuated joints (excluding universe and free-flyer).

### `actuated_joint_names`

Names of actuated joints only.

### `freeflyer_limits`

Current free-flyer position limits.

### `backend`

Lazily-created DynamicsBackend wrapping this robot's model.

Returns a PinocchioBackend that shares the same pin.Model and pin.Data as this Robot instance — no URDF re-loading, no model duplication. Used by RegressorBuilder and other tools to route dynamics computation through the backend abstraction.

Returns:

| Type | Description               |
| ---- | ------------------------- |
|      | PinocchioBackend instance |

### `update_freeflyer_limits(limits)`

Update free-flyer limits dynamically.

Source code in `src/figaroh/tools/robot.py`

```
def update_freeflyer_limits(self, limits: Tuple[float, float]) -> None:
    """Update free-flyer limits dynamically."""
    if not self.isFext:
        raise ValueError("Robot does not have a free-flyer joint")

    self._freeflyer_limits = limits
    self._set_freeflyer_limits()
```

### `display_q0(visualizer_type=None, q=None)`

Enhanced visualization with configuration options.

Source code in `src/figaroh/tools/robot.py`

```
def display_q0(
    self, visualizer_type: Optional[str] = None, q: Optional[np.ndarray] = None
) -> None:
    """Enhanced visualization with configuration options."""
    visualizer_class = self._get_visualizer_class(visualizer_type)

    if visualizer_class:
        self.setVisualizer(visualizer_class())
        self.initViewer()
        self.loadViewerModel(self.robot_urdf)

        # Display specified configuration or initial configuration
        config = q if q is not None else self.q0
        self.display(config)
```

### `get_configuration_bounds()`

Get configuration space bounds with proper handling of quaternions.

Source code in `src/figaroh/tools/robot.py`

```
def get_configuration_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
    """Get configuration space bounds with proper handling of quaternions."""
    lb, ub = (
        self.model.lowerPositionLimit.copy(),
        self.model.upperPositionLimit.copy(),
    )

    if self.isFext:
        # Ensure quaternion bounds are properly set
        lb[3:7] = -1.0
        ub[3:7] = 1.0

    return lb, ub
```

### `validate_configuration(q)`

Validate if configuration is within bounds and constraints.

Source code in `src/figaroh/tools/robot.py`

```
def validate_configuration(self, q: np.ndarray) -> bool:
    """Validate if configuration is within bounds and constraints."""
    if len(q) != self.model.nq:
        return False

    lb, ub = self.get_configuration_bounds()

    # Check bounds
    if not np.all((q >= lb) & (q <= ub)):
        return False

    # Check quaternion normalization for free-flyer
    if self.isFext:
        quat = q[3:7]
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-6):
            return False

    return True
```

### `__repr__()`

Enhanced string representation.

Source code in `src/figaroh/tools/robot.py`

```
def __repr__(self) -> str:
    """Enhanced string representation."""
    return (
        f"Robot(urdf='{self.robot_urdf}', "
        f"actuated_joints={self.num_joints}, "
        f"total_dofs={self.model.nv}, "
        f"free_flyer={self.isFext})"
    )
```

## `create_robot(robot_urdf, package_dirs=None, loader='figaroh', enhanced_features=True, **kwargs)`

Factory function to create robots with optional enhanced features.

Parameters:

| Name                | Type            | Description                                         | Default     |
| ------------------- | --------------- | --------------------------------------------------- | ----------- |
| `robot_urdf`        | `str`           | Robot identifier or URDF path                       | *required*  |
| `package_dirs`      | `Optional[str]` | Package directories                                 | `None`      |
| `loader`            | `str`           | Loader type from load_robot.py                      | `'figaroh'` |
| `enhanced_features` | `bool`          | If True, returns Robot class with enhanced features | `True`      |
| `**kwargs`          |                 | Additional arguments for loaders                    | `{}`        |

Returns:

| Type                         | Description                                                |
| ---------------------------- | ---------------------------------------------------------- |
| `Union[Robot, RobotWrapper]` | Robot instance (if enhanced_features=True) or RobotWrapper |

Source code in `src/figaroh/tools/robot.py`

```
def create_robot(
    robot_urdf: str,
    package_dirs: Optional[str] = None,
    loader: str = "figaroh",
    enhanced_features: bool = True,
    **kwargs,
) -> Union[Robot, RobotWrapper]:
    """Factory function to create robots with optional enhanced features.

    Args:
        robot_urdf: Robot identifier or URDF path
        package_dirs: Package directories
        loader: Loader type from load_robot.py
        enhanced_features: If True, returns Robot class with enhanced features
        **kwargs: Additional arguments for loaders

    Returns:
        Robot instance (if enhanced_features=True) or RobotWrapper
    """
    if enhanced_features and loader == "figaroh":
        # Use enhanced Robot class
        from .load_robot import _prepare_package_dirs, _validate_urdf_exists

        isFext = kwargs.get("isFext", False)
        robot_pkg = kwargs.get("robot_pkg")

        package_dirs = _prepare_package_dirs(robot_urdf, package_dirs, robot_pkg)
        _validate_urdf_exists(robot_urdf)

        return Robot(
            robot_urdf,
            package_dirs=package_dirs,
            isFext=isFext,
            freeflyer_ori=kwargs.get("freeflyer_ori"),
            freeflyer_limits=kwargs.get("freeflyer_limits"),
        )
    else:
        # Use load_robot.py for other loaders
        from .load_robot import load_robot

        return load_robot(robot_urdf, package_dirs, loader=loader, **kwargs)
```

## `load_robot(robot_urdf, package_dirs=None, isFext=False, load_by_urdf=True, robot_pkg=None, loader='figaroh', **kwargs)`

Load robot model from various sources with multiple loader options.

Parameters:

| Name           | Type            | Description                                              | Default     |
| -------------- | --------------- | -------------------------------------------------------- | ----------- |
| `robot_urdf`   | `str`           | Path to URDF file or robot name for robot_description    | *required*  |
| `package_dirs` | `Optional[str]` | Package directories for mesh files                       | `None`      |
| `isFext`       | `bool`          | Whether to add floating base joint                       | `False`     |
| `load_by_urdf` | `bool`          | Whether to load from URDF file (vs ROS param server)     | `True`      |
| `robot_pkg`    | `Optional[str]` | Name of robot package for path resolution                | `None`      |
| `loader`       | `str`           | Loader type - "figaroh", "robot_description", "yourdfpy" | `'figaroh'` |
| `**kwargs`     |                 | Additional arguments passed to the specific loader       | `{}`        |

Returns:

| Type                              | Description                                                    |
| --------------------------------- | -------------------------------------------------------------- |
| `Union[Robot, RobotWrapper, Any]` | Robot object based on loader type:                             |
| `Union[Robot, RobotWrapper, Any]` | "figaroh": Robot class instance (default, backward compatible) |
| `Union[Robot, RobotWrapper, Any]` | "robot_description": RobotWrapper from pinocchio               |
| `Union[Robot, RobotWrapper, Any]` | "yourdfpy": URDF object from yourdfpy (suitable for viser)     |

Raises:

| Type                | Description                              |
| ------------------- | ---------------------------------------- |
| `FileNotFoundError` | If URDF file not found                   |
| `ImportError`       | If required packages not available       |
| `ValueError`        | If the specified loader is not supported |

For loading by URDF, robot_urdf and package_dirs can be different.

1/ If package_dirs is not provided directly, robot_pkg is used to look up the package directory. 2/ If no mesh files, package_dirs and robot_pkg are not used. 3/ If load_by_urdf is False, the robot is loaded from the ROS parameter server. 4/ For robot_description loader, robot_urdf should be the robot name. 5/ For yourdfpy loader, returns URDF object suitable for viser visualization.

Source code in `src/figaroh/tools/load_robot.py`

```
def load_robot(
    robot_urdf: str,
    package_dirs: Optional[str] = None,
    isFext: bool = False,
    load_by_urdf: bool = True,
    robot_pkg: Optional[str] = None,
    loader: str = "figaroh",
    **kwargs,
) -> Union[Robot, RobotWrapper, Any]:
    """Load robot model from various sources with multiple loader options.

    Args:
        robot_urdf: Path to URDF file or robot name for robot_description
        package_dirs: Package directories for mesh files
        isFext: Whether to add floating base joint
        load_by_urdf: Whether to load from URDF file (vs ROS param server)
        robot_pkg: Name of robot package for path resolution
        loader: Loader type - "figaroh", "robot_description", "yourdfpy"
        **kwargs: Additional arguments passed to the specific loader

    Returns:
        Robot object based on loader type:
        - "figaroh": Robot class instance (default, backward compatible)
        - "robot_description": RobotWrapper from pinocchio
        - "yourdfpy": URDF object from yourdfpy (suitable for viser)

    Raises:
        FileNotFoundError: If URDF file not found
        ImportError: If required packages not available
        ValueError: If the specified loader is not supported

    Note: For loading by URDF, robot_urdf and package_dirs can be different.
          1/ If package_dirs is not provided directly, robot_pkg is used to
          look up the package directory.
          2/ If no mesh files, package_dirs and robot_pkg are not used.
          3/ If load_by_urdf is False, the robot is loaded from the ROS
          parameter server.
          4/ For robot_description loader, robot_urdf should be the robot name.
          5/ For yourdfpy loader, returns URDF object suitable for viser visualization.
    """
    # Handle different loaders
    if loader == "robot_description":
        return _load_robot_description(robot_urdf, isFext, **kwargs)
    elif loader == "yourdfpy":
        return _load_yourdfpy(robot_urdf, package_dirs, robot_pkg, **kwargs)
    elif loader == "figaroh":
        # Original figaroh implementation (backward compatible)
        return _load_figaroh_original(
            robot_urdf, package_dirs, isFext, load_by_urdf, robot_pkg
        )
    else:
        raise ValueError(
            f"Unsupported loader: {loader}. Supported loaders: figaroh, robot_description, yourdfpy"
        )
```

## `get_available_loaders()`

Get information about available robot loaders.

Returns:

| Name   | Type   | Description                                        |
| ------ | ------ | -------------------------------------------------- |
| `dict` | `dict` | Information about each loader and its availability |

Source code in `src/figaroh/tools/load_robot.py`

```
def get_available_loaders() -> dict:
    """Get information about available robot loaders.

    Returns:
        dict: Information about each loader and its availability
    """
    loaders = {
        "figaroh": {
            "description": "Original figaroh Robot class",
            "available": True,
            "returns": "Robot instance",
            "features": ["URDF loading", "ROS param server", "Free-flyer support"],
        },
        "robot_description": {
            "description": "Load from robot_descriptions package",
            "available": _check_package_available("robot_descriptions"),
            "returns": "RobotWrapper instance",
            "features": ["Pre-defined robot models", "Easy robot switching"],
        },
        "yourdfpy": {
            "description": "Load with yourdfpy (for visualization)",
            "available": _check_package_available("yourdfpy"),
            "returns": "URDF object",
            "features": ["Visualization support", "Mesh loading", "Scene graphs"],
        },
    }

    return loaders
```

## `list_available_robots(loader='robot_description')`

List available robot descriptions for a given loader.

Parameters:

| Name     | Type  | Description                               | Default               |
| -------- | ----- | ----------------------------------------- | --------------------- |
| `loader` | `str` | Loader type to check for available robots | `'robot_description'` |

Returns:

| Name   | Type   | Description           |
| ------ | ------ | --------------------- |
| `list` | `list` | Available robot names |

Source code in `src/figaroh/tools/load_robot.py`

```
def list_available_robots(loader: str = "robot_description") -> list:
    """List available robot descriptions for a given loader.

    Args:
        loader: Loader type to check for available robots

    Returns:
        list: Available robot names
    """
    if loader == "robot_description":
        if not _check_package_available("robot_descriptions"):
            return []

        try:
            import robot_descriptions

            # Get all available robot descriptions with URDF format
            available_robots = []

            # Check if DESCRIPTIONS attribute exists
            if hasattr(robot_descriptions, "DESCRIPTIONS"):
                descriptions = robot_descriptions.DESCRIPTIONS
                for robot_name, robot_info in descriptions.items():
                    # Check if the robot has URDF format available
                    if hasattr(robot_info, "has_urdf") and robot_info.has_urdf:
                        available_robots.append(robot_name)
                    # Fallback: check if robot_info has URDF_PATH attribute
                    elif hasattr(robot_info, "URDF_PATH"):
                        available_robots.append(robot_name)
            else:
                # Fallback to original method if DESCRIPTIONS not available
                for attr_name in dir(robot_descriptions):
                    if attr_name.startswith("_"):
                        continue
                    attr_obj = getattr(robot_descriptions, attr_name)
                    if hasattr(attr_obj, "URDF_PATH"):
                        available_robots.append(attr_name)

            return sorted(available_robots)
        except Exception:
            return []

    return []
```

## Regressor Builder

Regressor matrix computation utilities for robot dynamic identification.

## `RegressorConfig(has_friction=False, has_actuator_inertia=False, has_joint_offset=False, is_joint_torques=True, is_external_wrench=False, force_torque=None, additional_columns=0)`

Configuration for regressor computation.

### `extra_blocks()`

Enabled per-joint extra parameter blocks, in column order.

The order matches `add_standard_additional_parameters`: viscous friction, static friction, actuator inertia, joint offset.

Source code in `src/figaroh/tools/regressor.py`

```
def extra_blocks(self) -> List[str]:
    """Enabled per-joint extra parameter blocks, in column order.

    The order matches ``add_standard_additional_parameters``:
    viscous friction, static friction, actuator inertia, joint offset.
    """
    blocks = []
    if self.has_friction:
        blocks += ["fv", "fs"]
    if self.has_actuator_inertia:
        blocks.append("ia")
    if self.has_joint_offset:
        blocks.append("off")
    return blocks
```

## `RegressorBuilder(robot, config=None)`

Enhanced regressor builder with better organization.

Source code in `src/figaroh/tools/regressor.py`

```
def __init__(self, robot, config: Optional[RegressorConfig] = None):
    self.robot = robot
    self.config = config or RegressorConfig()
    # Use backend if available, fall back to direct model access
    self._has_backend = hasattr(robot, "backend")
    self.nv = robot.backend.nv if self._has_backend else robot.model.nv
    self.nonzero_inertias = self._get_nonzero_inertias()
    # Start of each enabled extra block, in units of blocks after the
    # 10 inertial columns per body; disabled blocks take no columns.
    self._extra_block_index = {
        name: k for k, name in enumerate(self.config.extra_blocks())
    }
    self._n_extra_columns = len(self._extra_block_index)
```

### `build_basic_regressor(q, v, a, identif_config=None)`

Build basic regressor matrix.

Source code in `src/figaroh/tools/regressor.py`

```
def build_basic_regressor(
    self, q: np.ndarray, v: np.ndarray, a: np.ndarray, identif_config=None
) -> np.ndarray:
    """Build basic regressor matrix."""
    # Normalize inputs
    Q, V, A, N = self._normalize_inputs(q, v, a)

    if self.config.is_joint_torques:
        return self._build_joint_torque_regressor(Q, V, A, N, identif_config)
    elif self.config.is_external_wrench:
        return self._build_external_wrench_regressor(Q, V, A, N, identif_config)
    else:
        raise ValueError(
            "Must specify either joint_torques or external_wrench mode"
        )
```

## `build_regressor_basic(robot, q, v, a, identif_config, tau=None)`

Legacy function for backward compatibility.

Source code in `src/figaroh/tools/regressor.py`

```
def build_regressor_basic(robot, q, v, a, identif_config, tau=None):
    """Legacy function for backward compatibility."""
    # Calculate additional columns based on enabled options
    additional_columns = sum(
        [
            2 if identif_config.get("has_friction", False) else 0,  # fv and fs
            1 if identif_config.get("has_actuator_inertia", False) else 0,  # ia
            1 if identif_config.get("has_joint_offset", False) else 0,  # offset
        ]
    )

    config = RegressorConfig(
        has_friction=identif_config.get("has_friction", False),
        has_actuator_inertia=identif_config.get("has_actuator_inertia", False),
        has_joint_offset=identif_config.get("has_joint_offset", False),
        is_joint_torques=identif_config.get("is_joint_torques", True),
        is_external_wrench=identif_config.get("is_external_wrench", False),
        force_torque=identif_config.get("force_torque", None),
        additional_columns=additional_columns,
    )

    builder = RegressorBuilder(robot, config)
    return builder.build_basic_regressor(q, v, a, identif_config)
```

## `eliminate_non_dynaffect(W, params_std, tol_e=1e-06)`

Eliminate columns with small L2 norm.

Source code in `src/figaroh/tools/regressor.py`

```
def eliminate_non_dynaffect(W, params_std, tol_e=1e-6):
    """Eliminate columns with small L2 norm."""
    col_norms = np.diag(W.T @ W)
    param_keys = list(params_std.keys())

    keep_indices = []
    keep_params = []

    for i, norm in enumerate(col_norms):
        if norm >= tol_e and i < len(param_keys):
            keep_indices.append(i)
            keep_params.append(param_keys[i])

    W_reduced = W[:, keep_indices]
    return W_reduced, keep_params
```

## `get_index_eliminate(W, params_std, tol_e=1e-06)`

Get indices of columns to eliminate based on tolerance.

Parameters:

| Name         | Type | Description                    | Default    |
| ------------ | ---- | ------------------------------ | ---------- |
| `W`          |      | Joint torque regressor matrix  | *required* |
| `params_std` |      | Standard parameters dictionary | *required* |
| `tol_e`      |      | Tolerance value                | `1e-06`    |

Returns:

| Name    | Type | Description                                               |
| ------- | ---- | --------------------------------------------------------- |
| `tuple` |      | List of indices to eliminate List of remaining parameters |

Source code in `src/figaroh/tools/regressor.py`

```
def get_index_eliminate(W, params_std, tol_e=1e-6):
    """Get indices of columns to eliminate based on tolerance.

    Args:
        W: Joint torque regressor matrix
        params_std: Standard parameters dictionary
        tol_e: Tolerance value

    Returns:
        tuple:
            - List of indices to eliminate
            - List of remaining parameters
    """
    col_norm = np.diag(np.dot(W.T, W))
    idx_e = []
    params_r = []
    for i in range(col_norm.shape[0]):
        if col_norm[i] < tol_e:
            idx_e.append(i)
        else:
            params_r.append(list(params_std.keys())[i])
    return idx_e, params_r
```

## `build_regressor_reduced(W, idx_e)`

Build reduced regressor matrix.

Parameters:

| Name    | Type | Description                     | Default    |
| ------- | ---- | ------------------------------- | ---------- |
| `W`     |      | Input regressor matrix          | *required* |
| `idx_e` |      | Indices of columns to eliminate | *required* |

Returns:

| Name      | Type | Description              |
| --------- | ---- | ------------------------ |
| `ndarray` |      | Reduced regressor matrix |

Source code in `src/figaroh/tools/regressor.py`

```
def build_regressor_reduced(W, idx_e):
    """Build reduced regressor matrix.

    Args:
        W: Input regressor matrix
        idx_e: Indices of columns to eliminate

    Returns:
        ndarray: Reduced regressor matrix
    """
    W_e = np.delete(W, idx_e, 1)
    return W_e
```

## `build_total_regressor_current(W_b_u, W_b_l, W_l, I_u, I_l, param_standard_l, identif_config)`

Build regressor for total least squares with current measurements.

Parameters:

| Name               | Type | Description                        | Default    |
| ------------------ | ---- | ---------------------------------- | ---------- |
| `W_b_u`            |      | Base regressor for unloaded case   | *required* |
| `W_b_l`            |      | Base regressor for loaded case     | *required* |
| `W_l`              |      | Full regressor for loaded case     | *required* |
| `I_u`              |      | Joint currents in unloaded case    | *required* |
| `I_l`              |      | Joint currents in loaded case      | *required* |
| `param_standard_l` |      | Standard parameters in loaded case | *required* |
| `identif_config`   |      | Dictionary of settings             | *required* |

Returns:

| Name    | Type | Description                                                                 |
| ------- | ---- | --------------------------------------------------------------------------- |
| `tuple` |      | Total regressor matrix Normalized identif_configeter vector Residual vector |

Source code in `src/figaroh/tools/regressor.py`

```
def build_total_regressor_current(
    W_b_u, W_b_l, W_l, I_u, I_l, param_standard_l, identif_config
):
    """Build regressor for total least squares with current measurements.

    Args:
        W_b_u: Base regressor for unloaded case
        W_b_l: Base regressor for loaded case
        W_l: Full regressor for loaded case
        I_u: Joint currents in unloaded case
        I_l: Joint currents in loaded case
        param_standard_l: Standard parameters in loaded case
        identif_config: Dictionary of settings

    Returns:
        tuple:
            - Total regressor matrix
            - Normalized identif_configeter vector
            - Residual vector
    """
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    nb_joints = int(len(I_u) / identif_config["nb_samples"])
    n_samples = identif_config["nb_samples"]

    V_a = np.concatenate(
        [
            I_u[:n_samples].reshape(n_samples, 1),
            np.zeros(((nb_joints - 1) * n_samples, 1)),
        ],
        axis=0,
    )

    V_b = np.concatenate(
        [
            I_l[:n_samples].reshape(n_samples, 1),
            np.zeros(((nb_joints - 1) * n_samples, 1)),
        ],
        axis=0,
    )

    for ii in range(1, nb_joints):
        V_a_ii = np.concatenate(
            [
                np.zeros((n_samples * ii, 1)),
                I_u[n_samples * ii : (ii + 1) * n_samples].reshape(n_samples, 1),
                np.zeros((n_samples * (nb_joints - ii - 1), 1)),
            ],
            axis=0,
        )
        V_b_ii = np.concatenate(
            [
                np.zeros((n_samples * ii, 1)),
                I_l[n_samples * ii : (ii + 1) * n_samples].reshape(n_samples, 1),
                np.zeros((n_samples * (nb_joints - ii - 1), 1)),
            ],
            axis=0,
        )
        V_a = np.concatenate((V_a, V_a_ii), axis=1)
        V_b = np.concatenate((V_b, V_b_ii), axis=1)

    W_current = np.concatenate((V_a, V_b), axis=0)
    W_tot = np.concatenate((W_tot, W_current), axis=1)

    if identif_config["has_friction"]:
        W_l_temp = np.zeros((len(W_l), 12))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11]:
            W_l_temp[:, k] = W_l[:, (identif_config["which_body_loaded"]) * 12 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (identif_config["which_body_loaded"]) * 12 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    elif identif_config["has_actuator_inertia"]:
        W_l_temp = np.zeros((len(W_l), 14))
        for k in [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13]:
            W_l_temp[:, k] = W_l[:, (identif_config["which_body_loaded"]) * 14 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (identif_config["which_body_loaded"]) * 14 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    else:
        W_l_temp = np.zeros((len(W_l), 9))
        for k in range(9):
            W_l_temp[:, k] = W_l[:, (identif_config["which_body_loaded"]) * 10 + k]
        idx_e_temp, params_r_temp = get_index_eliminate(
            W_l_temp, param_standard_l, 1e-6
        )
        W_e_l = build_regressor_reduced(W_l_temp, idx_e_temp)
        W_upayload = np.concatenate(
            (np.zeros((len(W_l), W_e_l.shape[1])), -W_e_l), axis=0
        )
        W_tot = np.concatenate((W_tot, W_upayload), axis=1)
        W_kpayload = np.concatenate(
            (
                np.zeros((len(W_l), 1)),
                -W_l[:, (identif_config["which_body_loaded"]) * 10 + 9].reshape(
                    len(W_l), 1
                ),
            ),
            axis=0,
        )
        W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    V = np.transpose(Vh).conj()
    V_norm = identif_config["mass_load"] * np.divide(V[:, -1], V[-1, -1])
    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue
```

## `build_total_regressor_wrench(W_b_u, W_b_l, W_l, tau_u, tau_l, param_standard_l, identif_config)`

Build regressor for total least squares with external wrench measurements.

Parameters:

| Name               | Type | Description                                                           | Default    |
| ------------------ | ---- | --------------------------------------------------------------------- | ---------- |
| `W_b_u`            |      | Base regressor for unloaded case                                      | *required* |
| `W_b_l`            |      | Base regressor for loaded case                                        | *required* |
| `W_l`              |      | Full regressor for loaded case                                        | *required* |
| `tau_u`            |      | External wrench in unloaded case                                      | *required* |
| `tau_l`            |      | External wrench in loaded case                                        | *required* |
| `param_standard_l` |      | Standard parameters in loaded case                                    | *required* |
| `identif_config`   |      | Dictionary of settings. Must provide which_body_loaded and mass_load. | *required* |

Returns:

| Name    | Type | Description                                                        |
| ------- | ---- | ------------------------------------------------------------------ |
| `tuple` |      | Total regressor matrix Normalized parameter vector Residual vector |

Source code in `src/figaroh/tools/regressor.py`

```
def build_total_regressor_wrench(
    W_b_u, W_b_l, W_l, tau_u, tau_l, param_standard_l, identif_config
):
    """Build regressor for total least squares with external wrench measurements.

    Args:
        W_b_u: Base regressor for unloaded case
        W_b_l: Base regressor for loaded case
        W_l: Full regressor for loaded case
        tau_u: External wrench in unloaded case
        tau_l: External wrench in loaded case
        param_standard_l: Standard parameters in loaded case
        identif_config: Dictionary of settings. Must provide
            ``which_body_loaded`` and ``mass_load``.

    Returns:
        tuple:
            - Total regressor matrix
            - Normalized parameter vector
            - Residual vector
    """
    W_tot = np.concatenate((-W_b_u, -W_b_l), axis=0)

    tau_meast_ul = np.reshape(tau_u, (len(tau_u), 1))
    tau_meast_l = np.reshape(tau_l, (len(tau_l), 1))

    nb_samples_ul = int(len(tau_meast_ul) / 6)
    nb_samples_l = int(len(tau_meast_l) / 6)

    tau_ul = np.concatenate(
        [
            tau_meast_ul[:nb_samples_ul],
            np.zeros((len(tau_meast_ul) - nb_samples_ul, 1)),
        ],
        axis=0,
    )

    tau_l = np.concatenate(
        [tau_meast_l[:nb_samples_l], np.zeros((len(tau_meast_l) - nb_samples_l, 1))],
        axis=0,
    )

    for ii in range(1, 6):
        tau_ul_ii = np.concatenate(
            [
                np.concatenate(
                    [
                        np.zeros((nb_samples_ul * ii, 1)),
                        tau_meast_ul[nb_samples_ul * ii : (ii + 1) * nb_samples_ul],
                    ],
                    axis=0,
                ),
                np.zeros((nb_samples_ul * (5 - ii), 1)),
            ],
            axis=0,
        )

        tau_l_ii = np.concatenate(
            [
                np.concatenate(
                    [
                        np.zeros((nb_samples_l * ii, 1)),
                        tau_meast_l[nb_samples_l * ii : (ii + 1) * nb_samples_l],
                    ],
                    axis=0,
                ),
                np.zeros((nb_samples_l * (5 - ii), 1)),
            ],
            axis=0,
        )

        tau_ul = np.concatenate((tau_ul, tau_ul_ii), axis=1)
        tau_l = np.concatenate((tau_l, tau_l_ii), axis=1)

    W_tau = np.concatenate((tau_ul, tau_l), axis=0)
    W_tot = np.concatenate((W_tot, W_tau), axis=1)

    W_l_temp = np.zeros((len(W_l), 9))
    for k in range(9):
        W_l_temp[:, k] = W_l[:, (identif_config["which_body_loaded"]) * 10 + k]
    W_upayload = np.concatenate(
        (np.zeros((len(W_l), W_l_temp.shape[1])), -W_l_temp), axis=0
    )
    W_tot = np.concatenate((W_tot, W_upayload), axis=1)

    W_kpayload = np.concatenate(
        [
            np.zeros((len(W_l), 1)),
            -W_l[:, identif_config["which_body_loaded"] * 10 + 9].reshape(len(W_l), 1),
        ],
        axis=0,
    )
    W_tot = np.concatenate((W_tot, W_kpayload), axis=1)

    U, S, Vh = np.linalg.svd(W_tot, full_matrices=False)
    V = np.transpose(Vh).conj()
    V_norm = identif_config["mass_load"] * np.divide(V[:, -1], V[-1, -1])
    residue = np.matmul(W_tot, V_norm)

    return W_tot, V_norm, residue
```

## Robot Visualization

Enhanced robot visualization utilities with better performance.

## `VisualizationConfig(com_color=None, axes_color=None, force_color=None, bbox_color=None, scale_factor=1.0)`

Configuration for robot visualization.

## `RobotVisualizer(model, data, viz, config=None)`

Enhanced robot visualizer with better organization.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def __init__(
    self,
    model: pin.Model,
    data: pin.Data,
    viz,
    config: Optional[VisualizationConfig] = None,
):
    self.model = model
    self.data = data
    self.viz = viz
    self.config = config or VisualizationConfig()

    # Cache frequently used values
    self.joint_names = model.names
    self.nv = model.nv
```

### `update_kinematics(q)`

Update robot kinematics efficiently.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def update_kinematics(self, q: np.ndarray) -> None:
    """Update robot kinematics efficiently."""
    pin.forwardKinematics(self.model, self.data, q)
    pin.updateFramePlacements(self.model, self.data)
    pin.centerOfMass(self.model, self.data, q, True)
    pin.computeSubtreeMasses(self.model, self.data)
```

### `display_com(q, frame_indices)`

Display center of mass positions with better naming.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_com(self, q: np.ndarray, frame_indices: List[int]) -> None:
    """Display center of mass positions with better naming."""
    self.update_kinematics(q)

    for i, frame_idx in enumerate(frame_indices[:-1]):  # Exclude last frame
        next_frame_idx = frame_indices[i + 1]

        # Calculate link properties
        link_length = np.linalg.norm(
            self.data.oMf[next_frame_idx].translation
            - self.data.oMf[frame_idx].translation
        )

        # Mass-based radius scaling
        mass_ratio = (
            self.data.mass[i] / self.data.mass[0] if self.data.mass[0] > 0 else 0.1
        )
        radius = link_length * mass_ratio * self.config.scale_factor

        # COM placement
        placement = self.data.oMf[frame_idx].copy()
        parent_id = self.model.frames[frame_idx].parentJoint - 1
        if parent_id >= 0:
            placement.translation = self.data.com[parent_id]

        # Create and place sphere
        sphere_name = f"world/com_sphere_{i}"
        self.viz.viewer.gui.addSphere(sphere_name, radius, self.config.com_color)
        self._apply_placement(sphere_name, placement)
```

### `display_axes(q, axis_length=0.15, axis_radius=0.01)`

Display coordinate axes for joints.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_axes(
    self, q: np.ndarray, axis_length: float = 0.15, axis_radius: float = 0.01
) -> None:
    """Display coordinate axes for joints."""
    self.update_kinematics(q)

    for i, joint_name in enumerate(self.joint_names):
        joint_id = self.model.getJointId(joint_name)
        axis_name = f"world/axis_{joint_name}_{i}"

        # Create axis visualization
        self.viz.viewer.gui.addXYZaxis(
            axis_name,
            self.config.axes_color,
            axis_radius * self.config.scale_factor,
            axis_length * self.config.scale_factor,
        )

        # Apply joint placement
        self._apply_placement(axis_name, self.data.oMi[joint_id])
```

### `display_force(force, placement, scale=0.001, name='force_vector')`

Display force vector with better scaling.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_force(
    self,
    force: pin.Force,
    placement: pin.SE3,
    scale: float = 1e-3,
    name: str = "force_vector",
) -> None:
    """Display force vector with better scaling."""
    # Transform force to world frame
    world_force = force.se3Action(placement)
    force_magnitude = np.linalg.norm(world_force.linear)

    if force_magnitude < 1e-10:  # Skip negligible forces
        return

    # Calculate arrow properties
    arrow_length = force_magnitude * scale * self.config.scale_factor
    arrow_radius = arrow_length * 0.05

    # Align arrow with force direction
    force_direction = world_force.linear / force_magnitude
    rotation = self._rotation_from_vectors([1, 0, 0], force_direction)

    arrow_placement = placement.copy()
    arrow_placement.rotation = rotation

    # Create and place arrow
    arrow_name = f"world/{name}_arrow"
    self.viz.viewer.gui.addArrow(
        arrow_name, arrow_radius, arrow_length, self.config.force_color
    )
    self._apply_placement(arrow_name, arrow_placement)
```

### `display_bounding_boxes(q, com_min, com_max, frame_indices)`

Display COM bounding boxes for optimization visualization.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_bounding_boxes(
    self,
    q: np.ndarray,
    com_min: np.ndarray,
    com_max: np.ndarray,
    frame_indices: List[int],
) -> None:
    """Display COM bounding boxes for optimization visualization."""
    self.update_kinematics(q)

    for i, frame_idx in enumerate(frame_indices):
        # Get COM bounds for this segment
        bounds_start = 3 * i
        min_bounds = com_min[bounds_start : bounds_start + 3]
        max_bounds = com_max[bounds_start : bounds_start + 3]

        # Calculate box dimensions
        box_size = (max_bounds - min_bounds) * self.config.scale_factor

        # Box placement at COM location
        placement = self.data.oMf[frame_idx].copy()
        parent_id = self.model.frames[frame_idx].parentJoint - 1
        if parent_id >= 0:
            placement.translation = self.data.com[parent_id]

        # Create and place box
        box_name = f"world/com_bbox_{i}"
        self.viz.viewer.gui.addBox(
            box_name, box_size[0], box_size[1], box_size[2], self.config.bbox_color
        )
        self._apply_placement(box_name, placement)
```

### `display_joints(q)`

Display joint frames efficiently.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_joints(self, q: np.ndarray) -> None:
    """Display joint frames efficiently."""
    self.update_kinematics(q)

    for i in range(self.nv):
        joint_placement = self.data.oMi[i]
        joint_name = f"world/joint_frame_{i}"

        # Create joint frame visualization
        self.viz.viewer.gui.addXYZaxis(
            joint_name,
            self.config.axes_color,
            0.01 * self.config.scale_factor,
            0.15 * self.config.scale_factor,
        )

        # Apply configuration
        self._apply_placement(joint_name, joint_placement)
```

## `place(viz, name, M)`

Legacy placement function.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def place(viz, name: str, M: pin.SE3) -> None:
    """Legacy placement function."""
    _warn_deprecated("place", "viz.viewer.gui.applyConfiguration directly")
    viz.viewer.gui.applyConfiguration(name, pin.SE3ToXYZQUATtuple(M))
```

## `display_COM(model, data, viz, q, IDX)`

Legacy COM display function.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_COM(
    model: pin.Model, data: pin.Data, viz, q: np.ndarray, IDX: list
) -> None:
    """Legacy COM display function."""
    _warn_deprecated("display_COM", "RobotVisualizer.display_com")
    visualizer = RobotVisualizer(model, data, viz)
    visualizer.display_com(q, IDX)
```

## `display_axes(model, data, viz, q)`

Legacy axes display function.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_axes(model: pin.Model, data: pin.Data, viz, q: np.ndarray) -> None:
    """Legacy axes display function."""
    _warn_deprecated("display_axes", "RobotVisualizer.display_axes")
    visualizer = RobotVisualizer(model, data, viz)
    visualizer.display_axes(q)
```

## `rotation_matrix_from_vectors(vec1, vec2)`

Legacy rotation function.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def rotation_matrix_from_vectors(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """Legacy rotation function."""
    _warn_deprecated("rotation_matrix_from_vectors", "RobotVisualizer")
    visualizer = RobotVisualizer(None, None, None)
    return visualizer._rotation_from_vectors(vec1, vec2)
```

## `display_force(viz, phi, M_se3)`

Legacy force display function.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_force(viz, phi: pin.Force, M_se3: pin.SE3) -> None:
    """Legacy force display function."""
    _warn_deprecated("display_force", "RobotVisualizer.display_force")
    visualizer = RobotVisualizer(None, None, viz)
    visualizer.display_force(phi, M_se3)
```

## `display_bounding_boxes(viz, model, data, q, COM_min, COM_max, IDX)`

Legacy bounding box display function.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_bounding_boxes(
    viz,
    model: pin.Model,
    data: pin.Data,
    q: np.ndarray,
    COM_min: np.ndarray,
    COM_max: np.ndarray,
    IDX: list,
) -> None:
    """Legacy bounding box display function."""
    _warn_deprecated("display_bounding_boxes", "RobotVisualizer.display_bounding_boxes")
    visualizer = RobotVisualizer(model, data, viz)
    visualizer.display_bounding_boxes(q, COM_min, COM_max, IDX)
```

## `display_joints(viz, model, data, q)`

Legacy joint display function.

Source code in `src/figaroh/tools/robotvisualization.py`

```
def display_joints(viz, model: pin.Model, data: pin.Data, q: np.ndarray) -> None:
    """Legacy joint display function."""
    _warn_deprecated("display_joints", "RobotVisualizer.display_joints")
    visualizer = RobotVisualizer(model, data, viz)
    visualizer.display_joints(q)
```

## Robot Collisions

Enhanced collision detection and visualization utilities.

## `CollisionManager(robot, geom_model=None, geom_data=None, viz=None)`

Enhanced collision detection with better performance and safety.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def __init__(self, robot, geom_model=None, geom_data=None, viz=None):
    self.robot = robot
    self.viz = viz
    self.model = robot.model
    self.data = robot.model.createData()

    # Initialize geometry models
    self.geom_model = geom_model or getattr(robot, "geom_model", None)
    self.geom_data = geom_data or (
        self.geom_model.createData() if self.geom_model else None
    )

    if self.geom_data:
        self.geom_data.collisionRequests.enable_contact = True

    # Visualization cache
    self._vis_cache = {}
    self._max_patches = 10
```

### `setup_collision_pairs(srdf_model_path=None)`

Setup collision pairs with optional SRDF filtering.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def setup_collision_pairs(self, srdf_model_path: Optional[str] = None) -> None:
    """Setup collision pairs with optional SRDF filtering."""
    if not self.geom_model:
        raise ValueError("Geometry model not available for collision setup")

    # Add all collision pairs
    self.geom_model.addAllCollisionPairs()

    # Remove pairs specified in SRDF
    if srdf_model_path and self._file_exists(srdf_model_path):
        pin.removeCollisionPairs(self.model, self.geom_model, srdf_model_path)
```

### `check_collisions(q, update_geometry=True)`

Check for collisions at given configuration.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def check_collisions(self, q: np.ndarray, update_geometry: bool = True) -> bool:
    """Check for collisions at given configuration."""
    if not self.geom_model or not self.geom_data:
        return False

    if update_geometry:
        pin.updateGeometryPlacements(
            self.model, self.data, self.geom_model, self.geom_data, q
        )

    return pin.computeCollisions(
        self.model, self.data, self.geom_model, self.geom_data, q, False
    )
```

### `get_collision_details()`

Get detailed collision information.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def get_collision_details(self) -> List[Tuple[int, any, any]]:
    """Get detailed collision information."""
    if not self.geom_data:
        return []

    return [
        (idx, self.geom_model.collisionPairs[idx], result)
        for idx, result in enumerate(self.geom_data.collisionResults)
        if result.isCollision()
    ]
```

### `get_collision_distances(collision_details=None)`

Get minimum distances for collision pairs.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def get_collision_distances(
    self, collision_details: Optional[List] = None
) -> np.ndarray:
    """Get minimum distances for collision pairs."""
    if not self.geom_data:
        return np.array([])

    if collision_details is None:
        collision_details = self.get_collision_details()

    if not collision_details:
        return np.array([])

    return np.array(
        [
            self.geom_data.distanceResults[idx].min_distance
            for idx, _, _ in collision_details
        ]
    )
```

### `get_all_distances()`

Get distances for all collision pairs.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def get_all_distances(self) -> np.ndarray:
    """Get distances for all collision pairs."""
    if not self.geom_model or not self.geom_data:
        return np.array([])

    return np.array(
        [
            pin.computeDistance(self.geom_model, self.geom_data, k).min_distance
            for k in range(len(self.geom_model.collisionPairs))
        ]
    )
```

### `print_collision_pairs()`

Print all collision pair information.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def print_collision_pairs(self) -> None:
    """Print all collision pair information."""
    if not self.geom_model or not self.geom_data:
        print("No geometry model available")
        return

    print(f"Total collision pairs: {len(self.geom_model.collisionPairs)}")
    print("-" * 60)

    for k in range(len(self.geom_model.collisionPairs)):
        result = self.geom_data.collisionResults[k]
        pair = self.geom_model.collisionPairs[k]

        name1 = self.geom_model.geometryObjects[pair.first].name
        name2 = self.geom_model.geometryObjects[pair.second].name
        status = "COLLISION" if result.isCollision() else "FREE"

        print(f"Pair {k:3d}: {name1:20s} <-> {name2:20s} [{status}]")
```

### `visualize_collisions(collision_details=None)`

Visualize collision contacts with enhanced display.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def visualize_collisions(self, collision_details: Optional[List] = None) -> None:
    """Visualize collision contacts with enhanced display."""
    if not self.viz:
        return

    if collision_details is None:
        collision_details = self.get_collision_details()

    # Clean up old visualizations
    self._cleanup_old_contacts(len(collision_details))

    # Display new contacts
    for i, (idx, pair, result) in enumerate(collision_details[: self._max_patches]):
        if result.getNbContacts() > 0:
            contact = result.getContact(0)  # First contact point
            self._display_contact(i, contact, pair)
```

## `CollisionWrapper(robot, geom_model=None, geom_data=None, viz=None)`

Bases: `CollisionManager`

Legacy wrapper.

.. deprecated:: 1.0 Use :class:`CollisionManager`. Removal in 2.0.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def __init__(self, robot, geom_model=None, geom_data=None, viz=None):
    warnings.warn(
        "CollisionWrapper is deprecated in 1.0, will be removed in 2.0; "
        "use CollisionManager",
        DeprecationWarning,
        stacklevel=2,
    )
    super().__init__(robot, geom_model, geom_data, viz)

    # Legacy aliases
    self.rmodel = self.model
    self.rdata = self.data
    self.gmodel = self.geom_model
    self.gdata = self.geom_data
```

### `add_collisions()`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def add_collisions(self):
    """Legacy method."""
    if self.geom_model:
        self.geom_model.addAllCollisionPairs()
```

### `remove_collisions(srdf_model_path)`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def remove_collisions(self, srdf_model_path):
    """Legacy method."""
    if srdf_model_path and self.geom_model:
        pin.removeCollisionPairs(self.model, self.geom_model, srdf_model_path)
```

### `computeCollisions(q, geom_data=None)`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def computeCollisions(self, q, geom_data=None):
    """Legacy method."""
    if geom_data:
        self.geom_data = geom_data
    return self.check_collisions(q)
```

### `getCollisionList()`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def getCollisionList(self):
    """Legacy method."""
    return self.get_collision_details()
```

### `getCollisionDistances(collisions=None)`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def getCollisionDistances(self, collisions=None):
    """Legacy method."""
    return self.get_collision_distances(collisions)
```

### `getDistances()`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def getDistances(self):
    """Legacy method."""
    return self.get_all_distances()
```

### `getAllpairs()`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def getAllpairs(self):
    """Legacy method."""
    self.print_collision_pairs()
```

### `check_collision(q)`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def check_collision(self, q):
    """Legacy method."""
    return self.check_collisions(q)
```

### `displayCollisions(collisions=None)`

Legacy method.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def displayCollisions(self, collisions=None):
    """Legacy method."""
    self.visualize_collisions(collisions)
```

## `add_non_adjacent_pairs(model, geom_model)`

Add a collision pair for every two geometries on non-adjacent bodies.

Geometries on the same joint, or on a joint and its parent (adjacent bodies, usually in contact by design), are not paired. Fixed bodies belong to the joint they are merged into (`universe` for the world). Returns the number of pairs added.

Source code in `src/figaroh/tools/robotcollisions.py`

```
def add_non_adjacent_pairs(model, geom_model) -> int:
    """Add a collision pair for every two geometries on non-adjacent bodies.

    Geometries on the same joint, or on a joint and its parent (adjacent
    bodies, usually in contact by design), are not paired. Fixed bodies
    belong to the joint they are merged into (``universe`` for the world).
    Returns the number of pairs added.
    """
    objects = geom_model.geometryObjects
    added = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            a, b = objects[i].parentJoint, objects[j].parentJoint
            if a == b or model.parents[a] == b or model.parents[b] == a:
                continue
            geom_model.addCollisionPair(pin.CollisionPair(i, j))
            added += 1
    return added
```

## QR Decomposition

QR decomposition utilities for robot parameter identification.

This module provides QR-based helpers to:

- identify a set of base parameters (a full-rank subset) from a rank-deficient regressor
- express those base parameters as linear combinations of the original/full parameters

In particular, both QR workflows compute a matrix $M$ such that:

$$ \\phi\_{base} = M,\\theta\_{full} $$

where columns of $M$ correspond to the *remaining* parameter vector ordered as `params_r`.

Note

In FIGAROH, `params_r` typically refers to the list of *standard parameter names remaining after column elimination* (e.g., after removing all-zero regressor columns). It is not necessarily the complete set of standard parameters (which is often available as a dictionary like `params_std`).

Use :meth:`QRDecomposer.expand_mapping_matrix_to_full` if you need an "M" defined over a full, canonical parameter ordering.

The resulting row ordering matches the returned base-parameter expressions.

## `QRResult(rank, base_indices, pivot_order, W_b, beta, M, base_param_expressions, phi_b, phi_b_nom, method, diag_R, cond_R1)`

Structured output of a QR base-parameter decomposition.

All numerical fields are stored at full float64 precision. Rounding for display is done only at the expression-building stage.

Attributes:

| Name                     | Type                  | Description                                                                                                          |
| ------------------------ | --------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `rank`                   | `int`                 | Identified numerical rank of the regressor.                                                                          |
| `base_indices`           | `List[int]`           | Column indices into params_r that form the independent (base) set.                                                   |
| `pivot_order`            | `Optional[List[int]]` | Full pivot permutation used by the pivoting path (from :func:deterministic_column_pivots); None for the double path. |
| `W_b`                    | `ndarray`             | Base regressor matrix, shape (m, rank).                                                                              |
| `beta`                   | `ndarray`             | Dependency coefficient matrix, shape (rank, n - rank). Full precision — not rounded.                                 |
| `M`                      | `ndarray`             | Base mapping matrix satisfying phi_base = M @ theta_r, shape (rank, n).                                              |
| `base_param_expressions` | `List[str]`           | Human-readable expression strings, length rank.                                                                      |
| `phi_b`                  | `Optional[ndarray]`   | Identified base-parameter values, shape (rank,); None when computed without a tau vector.                            |
| `phi_b_nom`              | `Optional[ndarray]`   | Nominal base-parameter values from params_std prior, shape (rank,); None when params_std was not provided.           |
| `method`                 | `str`                 | "pivoting" or "double".                                                                                              |
| `diag_R`                 | `ndarray`             | Absolute diagonal of R at the factorisation point, shape (min(m, n),). Useful for stability diagnostics.             |
| `cond_R1`                | `float`               | Condition number of the R[:rank, :rank] block. Measures how well-conditioned the base part of the factorisation is.  |

## `QRDecomposer(tolerance=TOL_QR, beta_tolerance=TOL_BETA, relative_tolerance=None, deterministic=True)`

Enhanced QR decomposition handler for robot parameter identification.

Parameters:

| Name            | Type   | Description                                                                                                                                                                                                                                             | Default |
| --------------- | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `deterministic` | `bool` | Choose base columns with :func:deterministic_column_pivots (default), so ties between equally large columns are broken by parameter name instead of BLAS rounding noise (#116). False keeps LAPACK's pivoted QR, whose tie order is platform dependent. | `True`  |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def __init__(
    self,
    tolerance: float = TOL_QR,
    beta_tolerance: float = TOL_BETA,
    relative_tolerance: Optional[float] = None,
    deterministic: bool = True,
):
    """
    Args:
        deterministic: Choose base columns with
            :func:`deterministic_column_pivots` (default), so ties
            between equally large columns are broken by parameter name
            instead of BLAS rounding noise (#116). ``False`` keeps
            LAPACK's pivoted QR, whose tie order is platform dependent.
    """
    self.deterministic = deterministic
    self.tolerance = tolerance
    self.beta_tolerance = beta_tolerance
    self.relative_tolerance = relative_tolerance

    # Last computed base-mapping matrix and its labels.
    # M maps the remaining parameter vector (ordered as params_r) to base.
    self.M: Optional[np.ndarray] = None
    self.M_params_r: Optional[List[str]] = None
    self.M_base_params_expr: Optional[List[str]] = None
    self.M_method: Optional[str] = None
    # Column indices into params_r selected as the base (independent) set
    # by the last decomposition. Only populated by double_decomposition().
    self.base_indices: Optional[List[int]] = None
    self.regroup_indices: Optional[List[int]] = None
    # Diagnostics from the last decomposition (available via get_diagnostics)
    self._last_diag_R: Optional[np.ndarray] = None
    self._last_cond_R1: Optional[float] = None
```

### `get_M()`

Return the last computed base mapping matrix `M` (or None).

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_M(self) -> Optional[np.ndarray]:
    """Return the last computed base mapping matrix ``M`` (or None)."""
    return self.M
```

### `get_M_labels()`

Return (base_param_expr, params_r) labels for the last stored M.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_M_labels(self) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    """Return (base_param_expr, params_r) labels for the last stored M."""
    return self.M_base_params_expr, self.M_params_r
```

### `get_base_indices()`

Return the base-column indices (into `params_r`) from the last `double_decomposition()` call, or `None` if not yet run.

These indices select the columns of a *reduced* regressor (built with the same `params_r` ordering) that reproduce base-parameter predictions: `tau ≈ W_reduced[:, base_indices] @ phi_base`. Since the base/dependent column split reflects the robot's kinematic structure rather than a specific trajectory, this selection is valid for any regressor built with the same `params_r` ordering — e.g. to evaluate an identified model on held-out validation data.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_base_indices(self) -> Optional[List[int]]:
    """Return the base-column indices (into ``params_r``) from the last
    ``double_decomposition()`` call, or ``None`` if not yet run.

    These indices select the columns of a *reduced* regressor (built
    with the same ``params_r`` ordering) that reproduce base-parameter
    predictions: ``tau ≈ W_reduced[:, base_indices] @ phi_base``. Since
    the base/dependent column split reflects the robot's kinematic
    structure rather than a specific trajectory, this selection is
    valid for any regressor built with the same ``params_r`` ordering
    — e.g. to evaluate an identified model on held-out validation data.
    """
    return self.base_indices
```

### `get_diagnostics()`

Return stability diagnostics from the last decomposition.

Keys

`rank`: identified rank. `diag_R`: absolute diagonal of R at the factorisation point. `cond_R1`: condition number of the R[:rank, :rank] block. `method`: `"pivoting"` or `"double"`.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_diagnostics(self) -> Dict[str, Any]:
    """Return stability diagnostics from the last decomposition.

    Keys:
        ``rank``: identified rank.
        ``diag_R``: absolute diagonal of R at the factorisation point.
        ``cond_R1``: condition number of the R[:rank, :rank] block.
        ``method``: ``"pivoting"`` or ``"double"``.
    """
    return {
        "rank": self.M.shape[0] if self.M is not None else None,
        "diag_R": self._last_diag_R,
        "cond_R1": self._last_cond_R1,
        "method": self.M_method,
    }
```

### `decompose(W_e, params_r, tau=None, params_std=None, method='double')`

Unified entry point that returns a structured :class:`QRResult`.

This is the preferred API for new code. It delegates to either :meth:`decompose_with_pivoting` (`method="pivoting"`) or :meth:`double_decomposition` (`method="double"`), but always returns a :class:`QRResult` regardless of path.

Parameters:

| Name         | Type                         | Description                                                                                                                                | Default    |
| ------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ---------- |
| `W_e`        | `ndarray`                    | Full regressor matrix, shape (m, n).                                                                                                       | *required* |
| `params_r`   | `List[str]`                  | Remaining parameter names (columns of W_e), length n.                                                                                      | *required* |
| `tau`        | `Optional[ndarray]`          | Torque/effort vector, shape (m,). Defaults to zeros, which is sufficient for rank/basis computation; required for meaningful phi_b values. | `None`     |
| `params_std` | `Optional[Dict[str, float]]` | Optional prior parameter dict used to compute phi_b_nom (available in the double path only).                                               | `None`     |
| `method`     | `str`                        | "double" (default) or "pivoting".                                                                                                          | `'double'` |

Returns:

| Type       | Description                        |
| ---------- | ---------------------------------- |
| `QRResult` | A fully-populated :class:QRResult. |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def decompose(
    self,
    W_e: np.ndarray,
    params_r: List[str],
    tau: Optional[np.ndarray] = None,
    params_std: Optional[Dict[str, float]] = None,
    method: str = "double",
) -> "QRResult":
    """Unified entry point that returns a structured :class:`QRResult`.

    This is the preferred API for new code.  It delegates to either
    :meth:`decompose_with_pivoting` (``method="pivoting"``) or
    :meth:`double_decomposition` (``method="double"``), but always
    returns a :class:`QRResult` regardless of path.

    Args:
        W_e: Full regressor matrix, shape ``(m, n)``.
        params_r: Remaining parameter names (columns of ``W_e``), length n.
        tau: Torque/effort vector, shape ``(m,)``.  Defaults to zeros,
            which is sufficient for rank/basis computation; required for
            meaningful ``phi_b`` values.
        params_std: Optional prior parameter dict used to compute
            ``phi_b_nom`` (available in the double path only).
        method: ``"double"`` (default) or ``"pivoting"``.

    Returns:
        A fully-populated :class:`QRResult`.
    """
    if tau is None:
        tau = np.zeros(W_e.shape[0])

    method_norm = method.strip().lower()
    if method_norm in {"pivoting", "pivot", "qr_pivoting", "qr-pivoting"}:
        Q, R, P, rank = self._pivoted_qr(W_e, params_r)
        params_sorted = [params_r[P[i]] for i in range(P.shape[0])]
        R1, Q1, R2 = self._extract_base_components(R, Q, rank)
        beta = np.linalg.solve(R1, R2) if R2.size else np.empty((rank, 0))
        phi_b = np.linalg.solve(R1, Q1.T @ tau)
        W_b = Q1 @ R1
        base_param_expressions = self._build_parameter_expressions(
            params_sorted[:rank], params_sorted[rank:], beta
        )
        n_params = len(params_r)
        M = self._build_M_from_pivoting(
            beta=beta, P=P, rank=rank, n_params=n_params
        )
        diag_R = np.abs(np.diag(R))
        cond_R1 = float(np.linalg.cond(R1)) if rank > 0 else 0.0
        base_indices = sorted(P[:rank].tolist())
        return QRResult(
            rank=rank,
            base_indices=base_indices,
            pivot_order=P.tolist(),
            W_b=W_b,
            beta=beta,
            M=M,
            base_param_expressions=base_param_expressions,
            phi_b=phi_b,
            phi_b_nom=None,
            method="pivoting",
            diag_R=diag_R,
            cond_R1=cond_R1,
        )

    if method_norm in {"double", "double_qr", "double-qr"}:
        base_indices, regroup_indices = self._identify_base_parameters(
            W_e, params_r
        )
        W_base, W_regroup, params_base, params_regroup = self._regroup_parameters(
            W_e, params_r, base_indices, regroup_indices
        )
        W_regrouped = np.c_[W_base, W_regroup]
        Q_r, R_r = np.linalg.qr(W_regrouped)
        rank = len(base_indices)
        R1, Q1, R2 = self._extract_base_components(R_r, Q_r, rank)
        beta = np.linalg.solve(R1, R2) if R2.size else np.empty((rank, 0))
        phi_b = np.linalg.solve(R1, Q1.T @ tau)
        W_b = Q1 @ R1
        base_param_expressions = self._build_parameter_expressions(
            params_base, params_regroup, beta
        )
        n_params = len(params_r)
        M = self._build_M_from_partition(
            beta=beta,
            base_indices=base_indices,
            regroup_indices=regroup_indices,
            n_params=n_params,
        )
        diag_R = np.abs(np.diag(R_r))
        cond_R1 = float(np.linalg.cond(R1)) if rank > 0 else 0.0
        phi_b_nom = None
        if params_std is not None:
            phi_b_nom = self.compute_nominal_base_parameters(
                params_base, params_regroup, beta, params_std
            )
        return QRResult(
            rank=rank,
            base_indices=base_indices,
            pivot_order=None,
            W_b=W_b,
            beta=beta,
            M=M,
            base_param_expressions=base_param_expressions,
            phi_b=phi_b,
            phi_b_nom=phi_b_nom,
            method="double",
            diag_R=diag_R,
            cond_R1=cond_R1,
        )

    raise ValueError(f"Unknown method={method!r}. Expected 'double' or 'pivoting'.")
```

### `decompose_with_pivoting(tau, W_e, params_r)`

Perform QR decomposition with column pivoting.

This identifies an effective rank, computes base parameters, and returns a base regressor `W_b` along with a dict mapping base-parameter expressions to values.

Parameters:

| Name       | Type        | Description                                           | Default    |
| ---------- | ----------- | ----------------------------------------------------- | ---------- |
| `tau`      | `ndarray`   | Torque/effort vector, shape (m,).                     | *required* |
| `W_e`      | `ndarray`   | Full regressor matrix, shape (m, n).                  | *required* |
| `params_r` | `List[str]` | Remaining parameter names (columns of W_e), length n. | *required* |

Returns:

| Type                               | Description                                                                                                                             |
| ---------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| `Tuple[ndarray, Dict[str, float]]` | (W_b, base_parameters) where: - W_b has shape (m, r) with r = rank(W_e) - base_parameters maps expression strings (length r) to values. |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def decompose_with_pivoting(
    self, tau: np.ndarray, W_e: np.ndarray, params_r: List[str]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Perform QR decomposition with column pivoting.

    This identifies an effective rank, computes base parameters, and
    returns a base regressor ``W_b`` along with a dict mapping
    base-parameter expressions to values.

    Args:
        tau: Torque/effort vector, shape (m,).
        W_e: Full regressor matrix, shape (m, n).
        params_r: Remaining parameter names (columns of W_e), length n.

    Returns:
        (W_b, base_parameters) where:
          - W_b has shape (m, r) with r = rank(W_e)
          - base_parameters maps expression strings (length r) to values.
    """
    Q, R, P, rank = self._pivoted_qr(W_e, params_r)

    # Reorder parameters according to pivoting
    params_sorted = [params_r[P[i]] for i in range(P.shape[0])]

    # Extract base components
    R1, Q1, R2 = self._extract_base_components(R, Q, rank)

    # Compute base parameters — full precision; round only for display (#142)
    beta = np.linalg.solve(R1, R2) if R2.size else np.empty((rank, 0))
    phi_b = np.linalg.solve(R1, Q1.T @ tau)
    W_b = Q1 @ R1

    # Build parameter expressions (display rounding happens inside)
    base_params = self._build_parameter_expressions(
        params_sorted[:rank], params_sorted[rank:], beta
    )

    # Store mapping matrix M such that: phi_base = M @ theta_r
    n_params = len(params_r)
    M = self._build_M_from_pivoting(
        beta=beta,
        P=P,
        rank=rank,
        n_params=n_params,
    )

    self.M = M
    self.M_params_r = list(params_r)
    self.M_base_params_expr = list(base_params)
    self.M_method = "pivoting"
    self._last_diag_R = np.abs(np.diag(R))
    self._last_cond_R1 = float(np.linalg.cond(R1)) if rank > 0 else 0.0

    return W_b, dict(zip(base_params, phi_b))
```

### `double_decomposition(tau, W_e, params_r, params_std=None)`

Perform the "double QR" workflow for base-parameter identification.

This follows a two-stage procedure

1. Identify an independent/base subset via QR on `W_e`.
1. Regroup columns and run a second QR to compute linear dependencies.

The output base parameters are returned as expression strings built from the original parameter names.

Parameters:

| Name         | Type                         | Description                                                                                                                                                                                                                                                                           | Default    |
| ------------ | ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `tau`        | `ndarray`                    | Torque/effort vector, shape (m,).                                                                                                                                                                                                                                                     | *required* |
| `W_e`        | `ndarray`                    | Full regressor matrix, shape (m, n).                                                                                                                                                                                                                                                  | *required* |
| `params_r`   | `List[str]`                  | Remaining parameter names (columns of W_e), length n.                                                                                                                                                                                                                                 | *required* |
| `params_std` | `Optional[Dict[str, float]]` | Optional mapping from full parameter name -> value. If provided, also returns the standard-parameter values of the base expressions. Returns: If params_std is None: (W_b, base_parameters, params_base_expr, phi_b) else: (W_b, base_parameters, params_base_expr, phi_b, phi_b_nom) | `None`     |
| `Where`      |                              | W_b has shape (m, r) params_base_expr is a list[str] of length r phi_b is a vector of length r - phi_b_nom is length r if params_std is provided                                                                                                                                      | *required* |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def double_decomposition(
    self,
    tau: np.ndarray,
    W_e: np.ndarray,
    params_r: List[str],
    params_std: Optional[Dict[str, float]] = None,
) -> Union[
    Tuple[np.ndarray, Dict, List, np.ndarray],
    Tuple[np.ndarray, Dict, List, np.ndarray, np.ndarray],
]:
    """Perform the "double QR" workflow for base-parameter identification.

    This follows a two-stage procedure:
        1) Identify an independent/base subset via QR on ``W_e``.
        2) Regroup columns and run a second QR to compute linear
           dependencies.

    The output base parameters are returned as expression strings built
    from the original parameter names.

    Args:
        tau: Torque/effort vector, shape (m,).
        W_e: Full regressor matrix, shape (m, n).
        params_r: Remaining parameter names (columns of W_e), length n.
        params_std: Optional mapping from full parameter name -> value. If
            provided, also returns the standard-parameter values of
            the base expressions.

            Returns:
                    If params_std is None:
                        (W_b, base_parameters, params_base_expr, phi_b)
                    else:
                        (W_b, base_parameters, params_base_expr, phi_b,
                         phi_b_nom)

        Where:
          - W_b has shape (m, r)
          - params_base_expr is a list[str] of length r
          - phi_b is a vector of length r
                        - phi_b_nom is length r if params_std is provided
    """

    # First QR to identify base parameters
    base_indices, regroup_indices = self._identify_base_parameters(W_e, params_r)

    # Regroup and second QR
    (
        W_base,
        W_regroup,
        params_base,
        params_regroup,
    ) = self._regroup_parameters(
        W_e,
        params_r,
        base_indices,
        regroup_indices,
    )

    # Second QR decomposition
    W_regrouped = np.c_[W_base, W_regroup]
    Q_r, R_r = np.linalg.qr(W_regrouped)

    rank = len(base_indices)
    R1, Q1, R2 = self._extract_base_components(R_r, Q_r, rank)

    # Compute parameters — full precision; round only for display (#142)
    beta = np.linalg.solve(R1, R2) if R2.size else np.empty((rank, 0))
    phi_b = np.linalg.solve(R1, Q1.T @ tau)
    W_b = Q1 @ R1

    # Build expressions and compute standard parameters if provided
    params_base_expr = self._build_parameter_expressions(
        params_base, params_regroup, beta
    )
    base_parameters = dict(zip(params_base_expr, phi_b))

    # Store mapping matrix M such that: phi_base = M @ theta_r
    n_params = len(params_r)
    M = self._build_M_from_partition(
        beta=beta,
        base_indices=base_indices,
        regroup_indices=regroup_indices,
        n_params=n_params,
    )

    self.M = M
    self.M_params_r = list(params_r)
    self.M_base_params_expr = list(params_base_expr)
    self.M_method = "double"
    self.base_indices = list(base_indices)
    self.regroup_indices = list(regroup_indices)
    self._last_diag_R = np.abs(np.diag(R_r))
    self._last_cond_R1 = float(np.linalg.cond(R1)) if rank > 0 else 0.0

    if params_std is not None:
        phi_b_nom = self.compute_nominal_base_parameters(
            params_base, params_regroup, beta, params_std
        )

        return W_b, base_parameters, params_base_expr, phi_b, phi_b_nom

    return W_b, base_parameters, params_base_expr, phi_b
```

### `compute_nominal_base_parameters(base_params, regroup_params, beta, params_std)`

Compute numerical values of base parameters from priors.

This computes the numerical value of each base expression using:

```
phi_std[i] = params_std[base_param]
             + sum_j beta[i, j] * params_std[dep_param]
```

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def compute_nominal_base_parameters(
    self,
    base_params: List[str],
    regroup_params: List[str],
    beta: np.ndarray,
    params_std: Dict[str, float],
) -> np.ndarray:
    """Compute numerical values of base parameters from priors.

    This computes the numerical value of each base expression using:

        phi_std[i] = params_std[base_param]
                     + sum_j beta[i, j] * params_std[dep_param]
    """
    phi_std = [params_std[param] for param in base_params]

    for i in range(len(phi_std)):
        for j, regroup_param in enumerate(regroup_params):
            if j < beta.shape[1]:
                phi_std[i] += beta[i, j] * params_std[regroup_param]

    return np.array(phi_std, dtype=float)
```

### `get_base_mapping_matrix_pivoting(W_e, params_r)`

Return the base-parameter mapping matrix M for pivoting-QR.

The matrix `M` satisfies `phi_base = M @ theta_r` where `theta_r` is ordered as `params_r` (the remaining parameters).

Row order matches the returned base-parameter expression strings.

Parameters:

| Name       | Type        | Description                                           | Default    |
| ---------- | ----------- | ----------------------------------------------------- | ---------- |
| `W_e`      | `ndarray`   | Full regressor matrix, shape (m, n).                  | *required* |
| `params_r` | `List[str]` | Remaining parameter names (columns of W_e), length n. | *required* |

Returns:

| Type                        | Description                                                                                                             |
| --------------------------- | ----------------------------------------------------------------------------------------------------------------------- |
| `Tuple[ndarray, List[str]]` | (M, base_params_expr) - M has shape (r, n) where r is the identified rank - base_params_expr is a list[str] of length r |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_base_mapping_matrix_pivoting(
    self, W_e: np.ndarray, params_r: List[str]
) -> Tuple[np.ndarray, List[str]]:
    r"""Return the base-parameter mapping matrix M for pivoting-QR.

    The matrix ``M`` satisfies ``phi_base = M @ theta_r`` where
    ``theta_r`` is ordered as ``params_r`` (the remaining parameters).

    Row order matches the returned base-parameter expression strings.

    Args:
        W_e: Full regressor matrix, shape (m, n).
        params_r: Remaining parameter names (columns of W_e), length n.

    Returns:
        (M, base_params_expr)
          - M has shape (r, n) where r is the identified rank
          - base_params_expr is a list[str] of length r
    """
    _, R, P, rank = self._pivoted_qr(W_e, params_r)

    # Dependency coefficients in the pivoted ordering — full precision
    R1 = R[:rank, :rank]
    R2 = R[:rank, rank:] if rank < R.shape[1] else np.array([]).reshape(rank, 0)
    beta = np.linalg.solve(R1, R2) if R2.size else np.empty((rank, 0))
    n_params = len(params_r)
    M = self._build_M_from_pivoting(
        beta=beta,
        P=P,
        rank=rank,
        n_params=n_params,
    )

    params_sorted = [params_r[P[i]] for i in range(P.shape[0])]
    base_params_expr = self._build_parameter_expressions(
        params_sorted[:rank], params_sorted[rank:], beta
    )
    return M, base_params_expr
```

### `get_base_mapping_matrix_double(W_e, params_r)`

Return the base-parameter mapping matrix M for the double-QR workflow.

The returned `M` satisfies `phi_base = M @ theta_r` with columns ordered as `params_r` (the remaining parameters).

Parameters:

| Name       | Type        | Description                                           | Default    |
| ---------- | ----------- | ----------------------------------------------------- | ---------- |
| `W_e`      | `ndarray`   | Full regressor matrix, shape (m, n).                  | *required* |
| `params_r` | `List[str]` | Remaining parameter names (columns of W_e), length n. | *required* |

Returns:

| Type                                              | Description                                                                                                                                                                    |
| ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `Tuple[ndarray, List[str], List[int], List[int]]` | (M, base_params_expr, base_indices, regroup_indices) - M has shape (r, n) - base_params_expr is a list[str] of length r - base_indices and regroup_indices index into params_r |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_base_mapping_matrix_double(
    self, W_e: np.ndarray, params_r: List[str]
) -> Tuple[np.ndarray, List[str], List[int], List[int]]:
    """Return the base-parameter mapping matrix M for the double-QR
    workflow.

    The returned ``M`` satisfies ``phi_base = M @ theta_r`` with columns
    ordered as ``params_r`` (the remaining parameters).

    Args:
        W_e: Full regressor matrix, shape (m, n).
        params_r: Remaining parameter names (columns of W_e), length n.

    Returns:
        (M, base_params_expr, base_indices, regroup_indices)
          - M has shape (r, n)
          - base_params_expr is a list[str] of length r
          - base_indices and regroup_indices index into params_r
    """
    base_indices, regroup_indices = self._identify_base_parameters(W_e, params_r)
    (
        W_base,
        W_regroup,
        params_base,
        params_regroup,
    ) = self._regroup_parameters(
        W_e,
        params_r,
        base_indices,
        regroup_indices,
    )

    W_regrouped = np.c_[W_base, W_regroup]
    _, R_r = np.linalg.qr(W_regrouped)
    rank = len(base_indices)

    R1 = R_r[:rank, :rank]
    R2 = R_r[:rank, rank:] if rank < R_r.shape[1] else np.array([]).reshape(rank, 0)
    beta = np.linalg.solve(R1, R2) if R2.size else np.empty((rank, 0))

    n_params = len(params_r)
    M = self._build_M_from_partition(
        beta=beta,
        base_indices=base_indices,
        regroup_indices=regroup_indices,
        n_params=n_params,
    )

    base_params_expr = self._build_parameter_expressions(
        params_base, params_regroup, beta
    )
    return M, base_params_expr, base_indices, regroup_indices
```

### `get_base_mapping_matrix(W_e, params_r, method='double')`

Convenience wrapper to retrieve the base mapping matrix.

Parameters:

| Name       | Type        | Description                                           | Default    |
| ---------- | ----------- | ----------------------------------------------------- | ---------- |
| `W_e`      | `ndarray`   | Full regressor matrix, shape (m, n).                  | *required* |
| `params_r` | `List[str]` | Remaining parameter names (columns of W_e), length n. | *required* |
| `method`   | `str`       | "double" (default) or "pivoting".                     | `'double'` |

Returns:

| Type                        | Description           |
| --------------------------- | --------------------- |
| `Tuple[ndarray, List[str]]` | (M, base_params_expr) |

Raises:

| Type         | Description                  |
| ------------ | ---------------------------- |
| `ValueError` | if method is not recognized. |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_base_mapping_matrix(
    self, W_e: np.ndarray, params_r: List[str], method: str = "double"
) -> Tuple[np.ndarray, List[str]]:
    """Convenience wrapper to retrieve the base mapping matrix.

    Args:
        W_e: Full regressor matrix, shape (m, n).
        params_r: Remaining parameter names (columns of W_e), length n.
        method: "double" (default) or "pivoting".

    Returns:
        (M, base_params_expr)

    Raises:
        ValueError: if method is not recognized.
    """
    method_norm = method.strip().lower()
    if method_norm in {"double", "double_qr", "double-qr"}:
        M, base_params_expr, _, _ = self.get_base_mapping_matrix_double(
            W_e, params_r
        )
        return M, base_params_expr
    if method_norm in {"pivoting", "pivot", "qr_pivoting", "qr-pivoting"}:
        return self.get_base_mapping_matrix_pivoting(W_e, params_r)
    raise ValueError(f"Unknown method={method!r}. Expected 'double' or 'pivoting'.")
```

### `expand_mapping_matrix_to_full(M, params_r, full_param_names)`

Expand an M defined on `params_r` into a full parameter ordering.

This is useful when the QR routines operate on a reduced set of parameters (after removing all-zero columns), but you want a mapping matrix defined over the complete standard-parameter list.

Parameters:

| Name               | Type        | Description                                                   | Default    |
| ------------------ | ----------- | ------------------------------------------------------------- | ---------- |
| `M`                | `ndarray`   | Base mapping matrix for remaining parameters, shape (r, n_r). | *required* |
| `params_r`         | `List[str]` | Remaining parameter names (columns of M), length n_r.         | *required* |
| `full_param_names` | `List[str]` | Full/canonical parameter ordering to expand into.             | *required* |

Returns:

| Type      | Description                                              |
| --------- | -------------------------------------------------------- |
| `ndarray` | M_full of shape (r, n_full) where columns not present in |
| `ndarray` | params_r are set to 0.                                   |

Raises:

| Type         | Description                                                                            |
| ------------ | -------------------------------------------------------------------------------------- |
| `ValueError` | if M has incompatible shape or if a name in params_r is missing from full_param_names. |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
@staticmethod
def expand_mapping_matrix_to_full(
    M: np.ndarray,
    params_r: List[str],
    full_param_names: List[str],
) -> np.ndarray:
    """Expand an M defined on ``params_r`` into a full parameter ordering.

    This is useful when the QR routines operate on a reduced set of
    parameters (after removing all-zero columns), but you want a mapping
    matrix defined over the complete standard-parameter list.

    Args:
        M: Base mapping matrix for remaining parameters, shape (r, n_r).
        params_r: Remaining parameter names (columns of ``M``), length n_r.
        full_param_names: Full/canonical parameter ordering to expand into.

    Returns:
        M_full of shape (r, n_full) where columns not present in
        ``params_r`` are set to 0.

    Raises:
        ValueError: if M has incompatible shape or if a name in params_r is
            missing from full_param_names.
    """
    if M.ndim != 2:
        raise ValueError("M must be a 2D array")
    if M.shape[1] != len(params_r):
        raise ValueError(
            "M has incompatible column count: "
            f"M.shape[1]={M.shape[1]} but len(params_r)={len(params_r)}"
        )

    full_index = {name: idx for idx, name in enumerate(full_param_names)}
    missing = [name for name in params_r if name not in full_index]
    if missing:
        raise ValueError(
            "params_r contains names not present in full_param_names: "
            f"{', '.join(missing)}"
        )

    M_full = np.zeros((M.shape[0], len(full_param_names)), dtype=M.dtype)
    for j_r, name in enumerate(params_r):
        M_full[:, full_index[name]] = M[:, j_r]
    return M_full
```

## `deterministic_column_pivots(W, names=None, *, tol=TOL_QR, relative_tol=None, tie_rtol=TIE_RTOL)`

Pick independent regressor columns the same way on every machine.

LAPACK's pivoted QR (`scipy.linalg.qr(pivoting=True)`) takes the column of largest residual norm at each step. Columns whose norms tie to rounding error (common in rigid-body regressors) are then ordered by BLAS noise, so the selected set, and with it `phi_base`, depends on the platform.

This greedy selection (Gram-Schmidt on the `R` factor, with re-orthogonalisation) takes the largest-residual column too, but treats all columns within `tie_rtol` of the largest as tied and picks the one with the smallest name (falling back to the smallest index when `names` is None). The outcome then depends on the column names, not on their order in `W` or on rounding noise below `tie_rtol`.

Parameters:

| Name           | Type                  | Description                                                                                      | Default    |
| -------------- | --------------------- | ------------------------------------------------------------------------------------------------ | ---------- |
| `W`            | `ndarray`             | Regressor, shape (m, n).                                                                         | *required* |
| `names`        | `Optional[List[str]]` | Column names, length n. Ties are broken by name.                                                 | `None`     |
| `tol`          | `float`               | Absolute rank threshold on the residual norm.                                                    | `TOL_QR`   |
| `relative_tol` | `Optional[float]`     | If set, threshold is max(relative_tol * first_pivot, tol), as in :meth:QRDecomposer.\_find_rank. | `None`     |
| `tie_rtol`     | `float`               | Relative width of the tie band.                                                                  | `TIE_RTOL` |

Returns:

| Type                  | Description                                                          |
| --------------------- | -------------------------------------------------------------------- |
| `ndarray`             | (P, rank): a permutation of range(n) whose first rank                |
| `int`                 | entries are the selected columns in selection order, followed by the |
| `Tuple[ndarray, int]` | remaining columns in ascending name (index) order.                   |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def deterministic_column_pivots(
    W: np.ndarray,
    names: Optional[List[str]] = None,
    *,
    tol: float = TOL_QR,
    relative_tol: Optional[float] = None,
    tie_rtol: float = TIE_RTOL,
) -> Tuple[np.ndarray, int]:
    """Pick independent regressor columns the same way on every machine.

    LAPACK's pivoted QR (``scipy.linalg.qr(pivoting=True)``) takes the column
    of largest residual norm at each step. Columns whose norms tie to rounding
    error (common in rigid-body regressors) are then ordered by BLAS noise,
    so the selected set, and with it ``phi_base``, depends on the platform.

    This greedy selection (Gram-Schmidt on the ``R`` factor, with
    re-orthogonalisation) takes the largest-residual column too, but treats
    all columns within ``tie_rtol`` of the largest as tied and picks the one
    with the smallest name (falling back to the smallest index when ``names``
    is None). The outcome then depends on the column names, not on their
    order in ``W`` or on rounding noise below ``tie_rtol``.

    Args:
        W: Regressor, shape ``(m, n)``.
        names: Column names, length n. Ties are broken by name.
        tol: Absolute rank threshold on the residual norm.
        relative_tol: If set, threshold is ``max(relative_tol * first_pivot,
            tol)``, as in :meth:`QRDecomposer._find_rank`.
        tie_rtol: Relative width of the tie band.

    Returns:
        ``(P, rank)``: a permutation of ``range(n)`` whose first ``rank``
        entries are the selected columns in selection order, followed by the
        remaining columns in ascending name (index) order.
    """
    n = W.shape[1]
    if n == 0:
        return np.empty(0, dtype=int), 0
    keys = list(names) if names is not None else list(range(n))
    rank_order = sorted(range(n), key=lambda j: (keys[j], j))
    prio = np.empty(n, dtype=int)
    prio[rank_order] = np.arange(n)

    A = np.linalg.qr(np.asarray(W, dtype=float), mode="r")
    remaining = np.ones(n, dtype=bool)
    chosen: List[int] = []
    thr = None
    for _ in range(min(A.shape[0], n)):
        norms = np.linalg.norm(A, axis=0)
        norms[~remaining] = -1.0
        top = norms.max()
        if thr is None:
            thr = tol
            if relative_tol is not None:
                thr = max(relative_tol * top, tol)
        if top <= thr:
            break
        tied = np.flatnonzero(norms >= top * (1.0 - tie_rtol))
        p = int(tied[np.argmin(prio[tied])])
        chosen.append(p)
        remaining[p] = False
        q = A[:, p] / norms[p]
        for _ in range(2):
            A -= np.outer(q, q @ A)
    rest = [j for j in rank_order if remaining[j]]
    return np.array(chosen + rest, dtype=int), len(chosen)
```

## `QR_pivoting(tau, W_e, params_r, tol_qr=TOL_QR)`

Legacy QR pivoting function.

.. deprecated:: 1.0 Use :meth:`QRDecomposer.decompose_with_pivoting`. Removal in 2.0.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def QR_pivoting(
    tau: np.ndarray,
    W_e: np.ndarray,
    params_r: list,
    tol_qr: float = TOL_QR,
) -> tuple:
    """Legacy QR pivoting function.

    .. deprecated:: 1.0
        Use :meth:`QRDecomposer.decompose_with_pivoting`. Removal in 2.0.
    """
    warnings.warn(
        "QR_pivoting is deprecated in 1.0, will be removed in 2.0; "
        "use QRDecomposer.decompose_with_pivoting",
        DeprecationWarning,
        stacklevel=2,
    )
    decomposer = QRDecomposer(tolerance=tol_qr)
    return decomposer.decompose_with_pivoting(tau, W_e, params_r)
```

## `double_QR(tau, W_e, params_r, params_std=None, tol_qr=TOL_QR)`

Legacy double QR function.

.. deprecated:: 1.0 Use :meth:`QRDecomposer.double_decomposition`. Removal in 2.0.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def double_QR(
    tau: np.ndarray,
    W_e: np.ndarray,
    params_r: list,
    params_std: dict = None,
    tol_qr: float = TOL_QR,
) -> tuple:
    """Legacy double QR function.

    .. deprecated:: 1.0
        Use :meth:`QRDecomposer.double_decomposition`. Removal in 2.0.
    """
    warnings.warn(
        "double_QR is deprecated in 1.0, will be removed in 2.0; "
        "use QRDecomposer.double_decomposition",
        DeprecationWarning,
        stacklevel=2,
    )
    decomposer = QRDecomposer(tolerance=tol_qr)
    return decomposer.double_decomposition(tau, W_e, params_r, params_std)
```

## `get_baseParams(W_e, params_r, params_std=None, tol_qr=TOL_QR)`

Legacy function for getting base parameters.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_baseParams(
    W_e: np.ndarray,
    params_r: list,
    params_std: dict = None,
    tol_qr: float = TOL_QR,
) -> tuple:
    """Legacy function for getting base parameters."""
    decomposer = QRDecomposer(tolerance=tol_qr)
    dummy_tau = np.zeros(W_e.shape[0])  # Not used in this function
    result = decomposer.double_decomposition(dummy_tau, W_e, params_r, params_std)
    base_indices, _ = decomposer._identify_base_parameters(W_e, params_r)
    return result[0], result[2], base_indices
```

## `get_baseIndex(W_e, params_r, tol_qr=TOL_QR)`

Legacy function for getting base indices.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def get_baseIndex(
    W_e: np.ndarray,
    params_r: list,
    tol_qr: float = TOL_QR,
) -> tuple:
    """Legacy function for getting base indices."""
    decomposer = QRDecomposer(tolerance=tol_qr)
    base_indices, _ = decomposer._identify_base_parameters(W_e, params_r)
    return tuple(base_indices)
```

## `build_baseRegressor(W_e, idx_base)`

Legacy function for building base regressor.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def build_baseRegressor(W_e: np.ndarray, idx_base: tuple) -> np.ndarray:
    """Legacy function for building base regressor."""
    return W_e[:, list(idx_base)]
```

## `cond_num(W_b, norm_type=None)`

Calculate condition number with various norms.

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def cond_num(W_b: np.ndarray, norm_type: str = None) -> float:
    """Calculate condition number with various norms."""
    if norm_type == "fro":
        return np.linalg.cond(W_b, "fro")
    elif norm_type == "max_over_min_sigma":
        return np.linalg.cond(W_b, 2) / np.linalg.cond(W_b, -2)
    else:
        return np.linalg.cond(W_b)
```

## `redistribute_min_norm(M, phi_base)`

Minimum-norm redistribution of base-parameter values onto the full standard-parameter set, via the Moore-Penrose pseudoinverse of `M`.

Given $\\phi\_{base} = M,\\theta_r$, returns $\\hat\\theta_r = M^+,\\phi\_{base}$: the minimum-L2-norm $\\theta_r$ consistent with the identified $\\phi\_{base}$, i.e. each redundant group's fitted value is spread across its members instead of being concentrated on one arbitrary representative (the effective default when only the base subset is reported and everything else is left at its nominal value). Round-trips exactly: `M @ redistribute_min_norm(M, phi_base) == phi_base` (up to floating point), so predictions built from the redistributed vector are unchanged from those built from the base-only fit.

This does not add information: non-identifiable directions stay non-identifiable. It only picks a less arbitrary point among the (infinite) equally-valid ways to split an identified combination back onto its individually non-identifiable members.

Parameters:

| Name       | Type      | Description                                                                    | Default    |
| ---------- | --------- | ------------------------------------------------------------------------------ | ---------- |
| `M`        | `ndarray` | Base-mapping matrix, shape (rank, n_params), such that phi_base = M @ theta_r. | *required* |
| `phi_base` | `ndarray` | Fitted base-parameter values, shape (rank,).                                   | *required* |

Returns:

| Type      | Description                     |
| --------- | ------------------------------- |
| `ndarray` | theta_r_hat, shape (n_params,). |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def redistribute_min_norm(M: np.ndarray, phi_base: np.ndarray) -> np.ndarray:
    r"""Minimum-norm redistribution of base-parameter values onto the full
    standard-parameter set, via the Moore-Penrose pseudoinverse of ``M``.

    Given $\phi_{base} = M\,\theta_r$, returns
    $\hat\theta_r = M^+\,\phi_{base}$: the minimum-L2-norm $\theta_r$
    consistent with the identified $\phi_{base}$, i.e. each redundant
    group's fitted value is spread across its members instead of being
    concentrated on one arbitrary representative (the effective default
    when only the base subset is reported and everything else is left at
    its nominal value). Round-trips exactly:
    ``M @ redistribute_min_norm(M, phi_base) == phi_base`` (up to floating
    point), so predictions built from the redistributed vector are
    unchanged from those built from the base-only fit.

    This does not add information: non-identifiable directions stay
    non-identifiable. It only picks a less arbitrary point among the
    (infinite) equally-valid ways to split an identified combination back
    onto its individually non-identifiable members.

    Args:
        M: Base-mapping matrix, shape ``(rank, n_params)``, such that
            ``phi_base = M @ theta_r``.
        phi_base: Fitted base-parameter values, shape ``(rank,)``.

    Returns:
        ``theta_r_hat``, shape ``(n_params,)``.
    """
    return np.linalg.pinv(M) @ phi_base
```

## `propagate_covariance_min_norm(M, C_base)`

Propagate base-parameter covariance through the minimum-norm map.

Returns $C\_{full} = M^+,C\_{base},(M^+)^T$ — the covariance of the minimum-norm *estimator* (:func:`redistribute_min_norm`), conditional on having committed to that redistribution choice. It is **not** the true uncertainty of each individual standard parameter: along directions `M` cannot see, the data provides literally zero information, and this propagation reports a finite number reflecting the estimator's own sensitivity to noise in `phi_base`, not an unconditional physical uncertainty. Callers surfacing this to users must label it as such.

Parameters:

| Name     | Type      | Description                                                                    | Default    |
| -------- | --------- | ------------------------------------------------------------------------------ | ---------- |
| `M`      | `ndarray` | Base-mapping matrix, shape (rank, n_params), such that phi_base = M @ theta_r. | *required* |
| `C_base` | `ndarray` | Covariance of the fitted base parameters, shape (rank, rank).                  | *required* |

Returns:

| Type      | Description                         |
| --------- | ----------------------------------- |
| `ndarray` | C_full, shape (n_params, n_params). |

Source code in `src/figaroh/tools/qrdecomposition.py`

```
def propagate_covariance_min_norm(M: np.ndarray, C_base: np.ndarray) -> np.ndarray:
    r"""Propagate base-parameter covariance through the minimum-norm map.

    Returns $C_{full} = M^+\,C_{base}\,(M^+)^T$ — the covariance of the
    minimum-norm *estimator* (:func:`redistribute_min_norm`), conditional
    on having committed to that redistribution choice. It is **not** the
    true uncertainty of each individual standard parameter: along
    directions ``M`` cannot see, the data provides literally zero
    information, and this propagation reports a finite number reflecting
    the estimator's own sensitivity to noise in ``phi_base``, not an
    unconditional physical uncertainty. Callers surfacing this to users
    must label it as such.

    Args:
        M: Base-mapping matrix, shape ``(rank, n_params)``, such that
            ``phi_base = M @ theta_r``.
        C_base: Covariance of the fitted base parameters, shape
            ``(rank, rank)``.

    Returns:
        ``C_full``, shape ``(n_params, n_params)``.
    """
    M_pinv = np.linalg.pinv(M)
    return M_pinv @ C_base @ M_pinv.T
```

## Robot IPOPT

General IPOPT-based optimization framework for robotics applications.

This module provides a comprehensive, flexible framework for setting up and solving nonlinear optimization problems using IPOPT (Interior Point OPTimizer), with specialized support for robotics applications including trajectory optimization, parameter identification, and general robot optimization problems.

Key Features

- Unified IPOPT interface with robotics-specific configurations
- Automatic differentiation support for gradients and Jacobians
- Built-in problem validation and result analysis
- Specialized classes for trajectory optimization problems
- Factory functions for rapid problem setup
- Comprehensive logging and iteration tracking

Main Classes

- IPOPTConfig: Configuration management for IPOPT solver settings
- BaseOptimizationProblem: Abstract base class for optimization problems
- RobotIPOPTSolver: High-level solver interface with result analysis
- TrajectoryOptimizationProblem: Specialized class for trajectory problems

Usage Examples

Basic trajectory optimization:

```
from figaroh.tools.robotipopt import IPOPTConfig, RobotIPOPTSolver

# Create custom problem class
class MyProblem(BaseOptimizationProblem):
    def objective(self, x):
        return np.sum(x**2)  # Minimize sum of squares

    def constraints(self, x):
        return np.array([x[0] + x[1] - 1.0])  # x[0] + x[1] = 1

    # ... implement other required methods

# Solve the problem
problem = MyProblem()
config = IPOPTConfig.for_trajectory_optimization()
solver = RobotIPOPTSolver(problem, config)
success, results = solver.solve()
```

Using the factory function:

```
from figaroh.tools.robotipopt import create_trajectory_solver

def my_objective(x):
    return np.sum(x**2)

def my_constraints(x):
    return np.array([x[0] + x[1] - 1.0])

def get_bounds():
    var_bounds = ([-10, -10], [10, 10])
    cons_bounds = ([0], [0])
    return var_bounds, cons_bounds

def initial_guess():
    return [0.5, 0.5]

solver = create_trajectory_solver(
    my_objective, my_constraints, get_bounds, initial_guess
)
success, results = solver.solve()
```

Complete robotics example:

```
import numpy as np
from figaroh.tools.robotipopt import (
    BaseOptimizationProblem, RobotIPOPTSolver, IPOPTConfig
)

class RobotTrajectoryProblem(BaseOptimizationProblem):
    def __init__(self, robot, waypoints, active_joints):
        super().__init__("RobotTrajectory")
        self.robot = robot
        self.waypoints = waypoints
        self.active_joints = active_joints
        self.n_joints = len(active_joints)
        self.n_waypoints = len(waypoints)
        self.n_vars = self.n_joints * self.n_waypoints

    def get_variable_bounds(self):
        # Joint limits for all waypoints
        lb = [-np.pi] * self.n_vars
        ub = [np.pi] * self.n_vars
        return lb, ub

    def get_constraint_bounds(self):
        # Boundary conditions (start/end positions)
        n_constraints = 2 * self.n_joints
        return [0] * n_constraints, [0] * n_constraints

    def get_initial_guess(self):
        # Linear interpolation between start and end
        return np.linspace(
            self.waypoints[0], self.waypoints[-1],
            self.n_waypoints * self.n_joints
        )

    def objective(self, x):
        # Minimize trajectory smoothness (squared accelerations)
        q = x.reshape(self.n_waypoints, self.n_joints)
        acc = np.diff(q, n=2, axis=0)  # Second differences
        return np.sum(acc**2)

    def constraints(self, x):
        # Enforce boundary conditions
        q = x.reshape(self.n_waypoints, self.n_joints)
        constraints = np.concatenate([
            q[0] - self.waypoints[0],   # Start position
            q[-1] - self.waypoints[-1]  # End position
        ])
        return constraints

# Usage
problem = RobotTrajectoryProblem(robot, waypoints, active_joints)
config = IPOPTConfig.for_trajectory_optimization()
solver = RobotIPOPTSolver(problem, config)

if solver.validate_problem_setup():
    success, results = solver.solve()
    if success:
        optimal_trajectory = results['x_opt'].reshape(
            problem.n_waypoints, problem.n_joints
        )
        print("Optimization successful!")
        print(f"Final objective: {results['obj_val']:.6f}")
        print(solver.get_solution_summary())
```

Dependencies

- cyipopt: Python interface to IPOPT solver
- numpy: Numerical computations
- numdifftools: Automatic differentiation (fallback)

References

- IPOPT documentation: https://coin-or.github.io/Ipopt/
- cyipopt: https://github.com/mechmotum/cyipopt

## `IPOPTConfig(tolerance=1e-06, acceptable_tolerance=0.0001, max_iterations=3000, max_cpu_time=1000000.0, print_level=5, output_file=None, hessian_approximation='limited-memory', warm_start=True, check_derivatives=True, linear_solver='mumps', custom_options=dict())`

Configuration parameters for IPOPT solver.

This class provides a convenient way to set IPOPT solver options with sensible defaults for robotics applications. It includes predefined configurations for common robotics optimization scenarios.

The configuration is designed to work well with:

- Trajectory optimization problems (smooth, continuous trajectories)
- Parameter identification (high-precision parameter estimation)
- General robotics optimization (balanced performance/accuracy)

Attributes:

| Name                    | Type             | Description                                                                                                   |
| ----------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------- |
| `tolerance`             | `float`          | Primary convergence tolerance for optimization. Default: 1e-6. Smaller values increase precision but runtime. |
| `acceptable_tolerance`  | `float`          | Fallback tolerance if primary fails. Default: 1e-4. Used when primary tolerance cannot be achieved.           |
| `max_iterations`        | `int`            | Maximum number of optimization iterations. Default: 3000. Increase for complex problems.                      |
| `max_cpu_time`          | `float`          | Maximum CPU time in seconds. Default: 1e6. Prevents infinite loops in difficult problems.                     |
| `print_level`           | `int`            | IPOPT output verbosity (0-12). Default: 5. Higher values provide more detailed output.                        |
| `output_file`           | `Optional[str]`  | File to save IPOPT output. Default: None. Useful for debugging and analysis.                                  |
| `hessian_approximation` | `str`            | Hessian computation method. Default: "limited-memory". Options: "exact", "limited-memory".                    |
| `warm_start`            | `bool`           | Use previous solution as starting point. Default: True. Speeds up subsequent optimizations.                   |
| `check_derivatives`     | `bool`           | Enable derivative checking. Default: True. Helps detect implementation errors.                                |
| `linear_solver`         | `str`            | Linear algebra solver backend. Default: "mumps". Options: "mumps", "ma27", "ma57", "ma77".                    |
| `custom_options`        | `Dict[str, Any]` | Additional IPOPT options. Default: {}. For advanced users to set specialized options.                         |

Examples:

Basic usage:

```
# Use default configuration
config = IPOPTConfig()

# Customize specific parameters
config = IPOPTConfig(
    tolerance=1e-8,
    max_iterations=5000,
    print_level=3
)
```

Predefined configurations:

```
# For trajectory optimization (speed-focused)
config = IPOPTConfig.for_trajectory_optimization()

# For parameter identification (precision-focused)
config = IPOPTConfig.for_parameter_identification()
```

Custom IPOPT options:

```
config = IPOPTConfig(
    custom_options={
        "mu_strategy": "adaptive",
        "nlp_scaling_method": "gradient-based"
    }
)
```

Note

All string options in custom_options should be provided as strings. They will be automatically encoded to bytes for IPOPT compatibility.

### `to_ipopt_options()`

Convert configuration to IPOPT option dictionary.

Transforms the configuration parameters into the format expected by IPOPT, including proper encoding of string options to bytes.

Returns:

| Type             | Description                                                                                   |
| ---------------- | --------------------------------------------------------------------------------------------- |
| `Dict[str, Any]` | Dict\[str, Any\]: Dictionary of IPOPT options ready for use with cyipopt.Problem.add_option() |

Note

All string values are automatically encoded to bytes as required by the IPOPT C interface via cyipopt.

Source code in `src/figaroh/tools/robotipopt.py`

```
def to_ipopt_options(self) -> Dict[str, Any]:
    """
    Convert configuration to IPOPT option dictionary.

    Transforms the configuration parameters into the format expected
    by IPOPT, including proper encoding of string options to bytes.

    Returns:
        Dict[str, Any]: Dictionary of IPOPT options ready for use
                       with cyipopt.Problem.add_option()

    Note:
        All string values are automatically encoded to bytes as required
        by the IPOPT C interface via cyipopt.
    """
    options = {
        # Basic convergence settings
        b"tol": self.tolerance,
        b"acceptable_tol": self.acceptable_tolerance,
        b"max_iter": self.max_iterations,
        b"max_cpu_time": self.max_cpu_time,
        # Output settings
        b"print_level": self.print_level,
        # Algorithm settings
        b"hessian_approximation": self.hessian_approximation.encode(),
        b"linear_solver": self.linear_solver.encode(),
        # Derivative checking
        b"check_derivatives_for_naninf": (
            b"yes" if self.check_derivatives else b"no"
        ),
        # Warm start
        b"warm_start_init_point": b"yes" if self.warm_start else b"no",
        b"acceptable_obj_change_tol": self.acceptable_tolerance,
    }

    # Add output file if specified
    if self.output_file:
        options[b"output_file"] = self.output_file.encode()

    # Add custom options
    for key, value in self.custom_options.items():
        if isinstance(key, str):
            key = key.encode()
        if isinstance(value, str):
            value = value.encode()
        options[key] = value

    return options
```

### `for_trajectory_optimization()`

Create configuration optimized for trajectory optimization problems.

This configuration prioritizes convergence speed over extreme precision, making it suitable for real-time or interactive trajectory planning. Uses adaptive barrier parameter strategy for better convergence.

Returns:

| Name          | Type          | Description                                      |
| ------------- | ------------- | ------------------------------------------------ |
| `IPOPTConfig` | `IPOPTConfig` | Optimized configuration for trajectory problems. |

Source code in `src/figaroh/tools/robotipopt.py`

```
@classmethod
def for_trajectory_optimization(cls) -> "IPOPTConfig":
    """
    Create configuration optimized for trajectory optimization problems.

    This configuration prioritizes convergence speed over extreme
    precision, making it suitable for real-time or interactive
    trajectory planning. Uses adaptive barrier parameter strategy
    for better convergence.

    Returns:
        IPOPTConfig: Optimized configuration for trajectory problems.
    """
    return cls(
        tolerance=1e-4,
        acceptable_tolerance=1e-3,
        max_iterations=1000,
        print_level=4,
        hessian_approximation="limited-memory",
        custom_options={
            b"mu_strategy": b"adaptive",
            b"adaptive_mu_globalization": b"obj-constr-filter",
        },
    )
```

### `for_parameter_identification()`

Create configuration optimized for parameter identification problems.

This configuration prioritizes high precision over speed, making it suitable for accurate parameter estimation in robotics applications. Uses exact Hessian computation and monotone barrier strategy for numerical stability.

Returns:

| Name          | Type          | Description                                           |
| ------------- | ------------- | ----------------------------------------------------- |
| `IPOPTConfig` | `IPOPTConfig` | Optimized configuration for parameter identification. |

Source code in `src/figaroh/tools/robotipopt.py`

```
@classmethod
def for_parameter_identification(cls) -> "IPOPTConfig":
    """
    Create configuration optimized for parameter identification problems.

    This configuration prioritizes high precision over speed, making it
    suitable for accurate parameter estimation in robotics applications.
    Uses exact Hessian computation and monotone barrier strategy for
    numerical stability.

    Returns:
        IPOPTConfig: Optimized configuration for parameter identification.
    """
    return cls(
        tolerance=1e-8,
        acceptable_tolerance=1e-6,
        max_iterations=5000,
        print_level=3,
        hessian_approximation="exact",
        custom_options={
            b"mu_strategy": b"monotone",
            b"fixed_variable_treatment": b"make_parameter",
        },
    )
```

## `BaseOptimizationProblem(name='OptimizationProblem')`

Bases: `ABC`

Abstract base class for IPOPT optimization problems.

This class defines the interface that all IPOPT problems must implement. Subclasses should implement the objective function, constraints, and their derivatives (or use automatic differentiation).

The class provides default implementations for gradient and Jacobian computation using automatic differentiation via numdifftools. For better performance, subclasses can override these methods with analytical derivatives.

Attributes:

| Name             | Type     | Description                                       |
| ---------------- | -------- | ------------------------------------------------- |
| `name`           | `str`    | Human-readable name for the optimization problem. |
| `logger`         | `Logger` | Logger instance for this problem.                 |
| `iteration_data` | `Dict`   | Storage for tracking optimization iterations.     |
| `callback_data`  | `Dict`   | Storage for passing data between methods.         |

Required Methods (must be implemented by subclasses):

- get_variable_bounds(): Return variable bounds as (lower, upper)
- get_constraint_bounds(): Return constraint bounds as (lower, upper)
- get_initial_guess(): Return initial guess for optimization variables
- objective(x): Evaluate objective function at point x
- constraints(x): Evaluate constraint functions at point x

Optional Methods (have default implementations):

- gradient(x): Compute objective gradient (uses auto-diff by default)
- jacobian(x): Compute constraint Jacobian (uses auto-diff by default)
- hessian(x, lagrange, obj_factor): Compute Hessian (IPOPT approx by default)
- intermediate(...): Callback for iteration tracking

Examples:

Basic implementation:

```
class QuadraticProblem(BaseOptimizationProblem):
    def get_variable_bounds(self):
        return ([-10, -10], [10, 10])

    def get_constraint_bounds(self):
        return ([0], [0])  # Equality constraint

    def get_initial_guess(self):
        return [1.0, 1.0]

    def objective(self, x):
        return x[0]**2 + x[1]**2  # Minimize sum of squares

    def constraints(self, x):
        return np.array([x[0] + x[1] - 1.0])  # x[0] + x[1] = 1
```

With custom derivatives:

```
class CustomProblem(BaseOptimizationProblem):
    # ... implement required methods ...

    def gradient(self, x):
        # Custom analytical gradient
        return np.array([2*x[0], 2*x[1]])

    def jacobian(self, x):
        # Custom analytical Jacobian
        return np.array([[1.0, 1.0]])
```

Initialize the optimization problem.

Source code in `src/figaroh/tools/robotipopt.py`

```
def __init__(self, name: str = "OptimizationProblem"):
    """Initialize the optimization problem."""
    self.name = name
    self.logger = logging.getLogger(f"{__name__}.{name}")

    # Storage for iteration tracking
    self.iteration_data = {
        "iterations": [],
        "obj_values": [],
        "constraint_violations": [],
        "solve_times": [],
    }

    # Callback storage for passing data between methods
    self.callback_data: Dict[str, Any] = {}
```

### `get_variable_bounds()`

Get bounds for optimization variables.

Returns:

| Type                              | Description                           |
| --------------------------------- | ------------------------------------- |
| `Tuple[List[float], List[float]]` | Tuple of (lower_bounds, upper_bounds) |

Source code in `src/figaroh/tools/robotipopt.py`

```
@abstractmethod
def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
    """
    Get bounds for optimization variables.

    Returns:
        Tuple of (lower_bounds, upper_bounds)
    """
    pass
```

### `get_constraint_bounds()`

Get bounds for constraints.

Returns:

| Type                              | Description                                                 |
| --------------------------------- | ----------------------------------------------------------- |
| `Tuple[List[float], List[float]]` | Tuple of (constraint_lower_bounds, constraint_upper_bounds) |

Source code in `src/figaroh/tools/robotipopt.py`

```
@abstractmethod
def get_constraint_bounds(self) -> Tuple[List[float], List[float]]:
    """
    Get bounds for constraints.

    Returns:
        Tuple of (constraint_lower_bounds, constraint_upper_bounds)
    """
    pass
```

### `get_initial_guess()`

Get initial guess for optimization variables.

Returns:

| Type          | Description                     |
| ------------- | ------------------------------- |
| `List[float]` | Initial guess as list of floats |

Source code in `src/figaroh/tools/robotipopt.py`

```
@abstractmethod
def get_initial_guess(self) -> List[float]:
    """
    Get initial guess for optimization variables.

    Returns:
        Initial guess as list of floats
    """
    pass
```

### `objective(x)`

Evaluate objective function.

Parameters:

| Name | Type      | Description            | Default    |
| ---- | --------- | ---------------------- | ---------- |
| `x`  | `ndarray` | Optimization variables | *required* |

Returns:

| Type    | Description              |
| ------- | ------------------------ |
| `float` | Objective function value |

Source code in `src/figaroh/tools/robotipopt.py`

```
@abstractmethod
def objective(self, x: np.ndarray) -> float:
    """
    Evaluate objective function.

    Args:
        x: Optimization variables

    Returns:
        Objective function value
    """
    pass
```

### `constraints(x)`

Evaluate constraint functions.

Parameters:

| Name | Type      | Description            | Default    |
| ---- | --------- | ---------------------- | ---------- |
| `x`  | `ndarray` | Optimization variables | *required* |

Returns:

| Type      | Description                |
| --------- | -------------------------- |
| `ndarray` | Constraint function values |

Source code in `src/figaroh/tools/robotipopt.py`

```
@abstractmethod
def constraints(self, x: np.ndarray) -> np.ndarray:
    """
    Evaluate constraint functions.

    Args:
        x: Optimization variables

    Returns:
        Constraint function values
    """
    pass
```

### `gradient(x)`

Compute gradient of objective function.

Default implementation uses automatic differentiation. Override for custom implementations.

Parameters:

| Name | Type      | Description            | Default    |
| ---- | --------- | ---------------------- | ---------- |
| `x`  | `ndarray` | Optimization variables | *required* |

Returns:

| Type      | Description     |
| --------- | --------------- |
| `ndarray` | Gradient vector |

Source code in `src/figaroh/tools/robotipopt.py`

```
def gradient(self, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient of objective function.

    Default implementation uses automatic differentiation.
    Override for custom implementations.

    Args:
        x: Optimization variables

    Returns:
        Gradient vector
    """
    try:
        return nd.Gradient(self.objective)(x)
    except Exception as e:
        self.logger.warning(f"Error computing gradient: {e}")
        return np.zeros_like(x)
```

### `jacobian(x)`

Compute Jacobian of constraint functions.

Default implementation uses automatic differentiation. Override for custom implementations.

Parameters:

| Name | Type      | Description            | Default    |
| ---- | --------- | ---------------------- | ---------- |
| `x`  | `ndarray` | Optimization variables | *required* |

Returns:

| Type      | Description     |
| --------- | --------------- |
| `ndarray` | Jacobian matrix |

Source code in `src/figaroh/tools/robotipopt.py`

```
def jacobian(self, x: np.ndarray) -> np.ndarray:
    """
    Compute Jacobian of constraint functions.

    Default implementation uses automatic differentiation.
    Override for custom implementations.

    Args:
        x: Optimization variables

    Returns:
        Jacobian matrix
    """
    try:
        constraints = self.constraints(x)
        if len(constraints) == 0:
            # No constraints
            return np.zeros((0, len(x)))

        jac = nd.Jacobian(self.constraints)(x)

        # Handle case where constraint returns scalar
        if constraints.ndim == 0 or len(constraints) == 1:
            jac = jac.reshape(1, -1)

        self.logger.debug(f"Constraint jacobian shape: {jac.shape}")
        return jac
    except Exception as e:
        self.logger.warning(f"Error computing jacobian: {e}")
        # Return identity matrix as fallback
        constraints = self.constraints(x)
        n_constraints = len(constraints) if hasattr(constraints, "__len__") else 1
        return np.eye(n_constraints, len(x))
```

### `hessian(x, lagrange, obj_factor)`

Compute Hessian of Lagrangian.

Default implementation returns False to use IPOPT's approximation. Override for custom implementations.

Parameters:

| Name         | Type      | Description              | Default    |
| ------------ | --------- | ------------------------ | ---------- |
| `x`          | `ndarray` | Optimization variables   | *required* |
| `lagrange`   | `ndarray` | Lagrange multipliers     | *required* |
| `obj_factor` | `float`   | Objective scaling factor | *required* |

Returns:

| Type                   | Description                                  |
| ---------------------- | -------------------------------------------- |
| `Union[bool, ndarray]` | Hessian matrix or False to use approximation |

Source code in `src/figaroh/tools/robotipopt.py`

```
def hessian(
    self, x: np.ndarray, lagrange: np.ndarray, obj_factor: float
) -> Union[bool, np.ndarray]:
    """
    Compute Hessian of Lagrangian.

    Default implementation returns False to use IPOPT's approximation.
    Override for custom implementations.

    Args:
        x: Optimization variables
        lagrange: Lagrange multipliers
        obj_factor: Objective scaling factor

    Returns:
        Hessian matrix or False to use approximation
    """
    return False
```

### `intermediate(alg_mod, iter_count, obj_value, inf_pr, inf_du, mu, d_norm, regularization_size, alpha_du, alpha_pr, ls_trials)`

Intermediate callback for iteration tracking.

Parameters:

| Name                  | Type    | Description              | Default    |
| --------------------- | ------- | ------------------------ | ---------- |
| `alg_mod`             | `int`   | Algorithm mode           | *required* |
| `iter_count`          | `int`   | Iteration count          | *required* |
| `obj_value`           | `float` | Current objective value  | *required* |
| `inf_pr`              | `float` | Primal infeasibility     | *required* |
| `inf_du`              | `float` | Dual infeasibility       | *required* |
| `mu`                  | `float` | Barrier parameter        | *required* |
| `d_norm`              | `float` | Step size                | *required* |
| `regularization_size` | `float` | Regularization parameter | *required* |
| `alpha_du`            | `float` | Dual step size           | *required* |
| `alpha_pr`            | `float` | Primal step size         | *required* |
| `ls_trials`           | `int`   | Line search trials       | *required* |

Returns:

| Type   | Description                                  |
| ------ | -------------------------------------------- |
| `bool` | True to continue optimization, False to stop |

Source code in `src/figaroh/tools/robotipopt.py`

```
def intermediate(
    self,
    alg_mod: int,
    iter_count: int,
    obj_value: float,
    inf_pr: float,
    inf_du: float,
    mu: float,
    d_norm: float,
    regularization_size: float,
    alpha_du: float,
    alpha_pr: float,
    ls_trials: int,
) -> bool:
    """
    Intermediate callback for iteration tracking.

    Args:
        alg_mod: Algorithm mode
        iter_count: Iteration count
        obj_value: Current objective value
        inf_pr: Primal infeasibility
        inf_du: Dual infeasibility
        mu: Barrier parameter
        d_norm: Step size
        regularization_size: Regularization parameter
        alpha_du: Dual step size
        alpha_pr: Primal step size
        ls_trials: Line search trials

    Returns:
        True to continue optimization, False to stop
    """
    self.iteration_data["iterations"].append(iter_count)
    self.iteration_data["obj_values"].append(obj_value)
    self.iteration_data["constraint_violations"].append(max(inf_pr, inf_du))

    return True
```

## `RobotIPOPTSolver(problem, config=None)`

General IPOPT solver for robotics optimization problems.

This class provides a high-level interface for solving optimization problems using IPOPT, with built-in support for result analysis, error handling, and problem validation specifically designed for robotics applications.

Features

- Automatic problem setup and configuration
- Built-in problem validation before solving
- Comprehensive result analysis and logging
- Solution history tracking for multiple solves
- Robotics-specific error handling and diagnostics
- Support for iterative solving and warm starts

Attributes:

| Name            | Type                      | Description                           |
| --------------- | ------------------------- | ------------------------------------- |
| `problem`       | `BaseOptimizationProblem` | The optimization problem to solve.    |
| `config`        | `IPOPTConfig`             | IPOPT solver configuration.           |
| `logger`        | `Logger`                  | Logger for solver operations.         |
| `last_solution` | `ndarray`                 | Most recent optimization solution.    |
| `last_info`     | `Dict`                    | Most recent IPOPT solver information. |
| `solve_history` | `List[Dict]`              | History of all solve attempts.        |

Examples:

Basic usage:

```
# Create problem and solver
problem = MyOptimizationProblem()
solver = RobotIPOPTSolver(problem)

# Solve the problem
success, results = solver.solve()

if success:
    print(f"Optimal solution: {results['x_opt']}")
    print(f"Objective value: {results['obj_val']}")
```

With custom configuration:

```
# Create custom configuration
config = IPOPTConfig(
    tolerance=1e-8,
    max_iterations=5000,
    print_level=3
)

# Create solver with custom config
solver = RobotIPOPTSolver(problem, config)

# Validate problem before solving
if solver.validate_problem_setup():
    success, results = solver.solve()
```

Multiple solves with warm start:

```
solver = RobotIPOPTSolver(problem)

# First solve
success1, results1 = solver.solve()

# Modify problem parameters...
problem.update_parameters(new_params)

# Second solve (uses warm start automatically)
success2, results2 = solver.solve()

# Access solve history
print(f"Solved {len(solver.solve_history)} problems")
```

Initialize the IPOPT solver.

Parameters:

| Name      | Type                      | Description                                                   | Default    |
| --------- | ------------------------- | ------------------------------------------------------------- | ---------- |
| `problem` | `BaseOptimizationProblem` | Optimization problem to solve                                 | *required* |
| `config`  | `Optional[IPOPTConfig]`   | IPOPT configuration (default: trajectory optimization config) | `None`     |

Source code in `src/figaroh/tools/robotipopt.py`

```
def __init__(
    self, problem: BaseOptimizationProblem, config: Optional[IPOPTConfig] = None
):
    """
    Initialize the IPOPT solver.

    Args:
        problem: Optimization problem to solve
        config: IPOPT configuration (default: trajectory optimization
               config)
    """
    self.problem = problem
    self.config = config or IPOPTConfig.for_trajectory_optimization()
    self.logger = logging.getLogger(f"{__name__}.RobotIPOPTSolver")

    # Results storage
    self.last_solution = None
    self.last_info = None
    self.solve_history = []
```

### `solve()`

Solve the optimization problem.

Performs the complete optimization process including problem setup, IPOPT configuration, solving, and result analysis. The method handles all IPOPT interactions and provides comprehensive error handling and logging.

Returns:

| Type                          | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `Tuple[bool, Dict[str, Any]]` | Tuple\[bool, Dict[str, Any]\]: A tuple containing: - success (bool): True if optimization succeeded, False otherwise - results (Dict[str, Any]): Dictionary containing: * 'x_opt': Optimal solution vector * 'obj_val': Final objective function value * 'status': IPOPT exit status code * 'status_msg': Human-readable status message * 'solve_time': Total optimization time in seconds * 'iterations': Number of iterations performed * 'iteration_data': Detailed iteration history * 'callback_data': Custom data from problem callbacks * 'ipopt_info': Complete IPOPT solver information * 'success': Copy of success flag for convenience |

Raises:

| Type        | Description                                                                                                       |
| ----------- | ----------------------------------------------------------------------------------------------------------------- |
| `Exception` | If critical errors occur during problem setup or solving. Non-critical errors are caught and returned in results. |

Note

IPOPT status codes for success: 0 (solved) and 1 (solved to acceptable level). All other codes, including -1 (maximum number of iterations exceeded), indicate failure or early termination; the last iterate is still returned in `x_opt`.

Source code in `src/figaroh/tools/robotipopt.py`

```
def solve(self) -> Tuple[bool, Dict[str, Any]]:
    """
    Solve the optimization problem.

    Performs the complete optimization process including problem setup,
    IPOPT configuration, solving, and result analysis. The method
    handles all IPOPT interactions and provides comprehensive error
    handling and logging.

    Returns:
        Tuple[bool, Dict[str, Any]]: A tuple containing:
            - success (bool): True if optimization succeeded,
              False otherwise
            - results (Dict[str, Any]): Dictionary containing:
                * 'x_opt': Optimal solution vector
                * 'obj_val': Final objective function value
                * 'status': IPOPT exit status code
                * 'status_msg': Human-readable status message
                * 'solve_time': Total optimization time in seconds
                * 'iterations': Number of iterations performed
                * 'iteration_data': Detailed iteration history
                * 'callback_data': Custom data from problem callbacks
                * 'ipopt_info': Complete IPOPT solver information
                * 'success': Copy of success flag for convenience

    Raises:
        Exception: If critical errors occur during problem setup
                  or solving. Non-critical errors are caught and
                  returned in results.

    Note:
        IPOPT status codes for success: 0 (solved) and 1 (solved to
        acceptable level). All other codes, including -1 (maximum number
        of iterations exceeded), indicate failure or early termination;
        the last iterate is still returned in ``x_opt``.
    """
    try:
        # Import cyipopt only when needed
        try:
            import cyipopt
        except ImportError:
            raise ImportError(
                "cyipopt is required for IPOPT optimization. "
                "Install with: pip install cyipopt"
            )

        self.logger.info(f"Setting up IPOPT problem: {self.problem.name}")

        # Get problem dimensions and bounds
        x0 = self.problem.get_initial_guess()
        lb, ub = self.problem.get_variable_bounds()
        cl, cu = self.problem.get_constraint_bounds()

        self.logger.info(
            f"Problem dimensions: {len(x0)} variables, " f"{len(cl)} constraints"
        )

        # Create IPOPT problem
        nlp = cyipopt.Problem(
            n=len(x0),
            m=len(cl),
            problem_obj=self.problem,
            lb=lb,
            ub=ub,
            cl=cl,
            cu=cu,
        )

        # Apply configuration
        for key, value in self.config.to_ipopt_options().items():
            nlp.add_option(key, value)

        # Solve optimization
        self.logger.info("Starting IPOPT optimization...")
        start_time = time.time()
        x_opt, info = nlp.solve(x0)
        solve_time = time.time() - start_time

        # Store results
        self.last_solution = x_opt
        self.last_info = info

        # Analyze results
        # Converged IPOPT exit codes; -1 is Maximum_Iterations_Exceeded
        success = info["status"] in [0, 1]
        if info["status"] == -1:
            self.logger.warning(
                "IPOPT stopped at the iteration limit without converging"
            )

        self.logger.info(f"Optimization completed in {solve_time:.2f} seconds")
        self.logger.info(f"Status: {info['status']} - {info['status_msg']}")
        self.logger.info(f"Final objective: {info['obj_val']:.6e}")

        # Prepare results dictionary
        results = {
            "success": success,
            "x_opt": x_opt,
            "obj_val": info["obj_val"],
            "status": info["status"],
            "status_msg": info["status_msg"],
            "solve_time": solve_time,
            "iterations": len(self.problem.iteration_data["iterations"]),
            "iteration_data": self.problem.iteration_data.copy(),
            "callback_data": self.problem.callback_data.copy(),
            "ipopt_info": info,
        }

        # Store in history
        self.solve_history.append(results)

        return success, results

    except Exception as e:
        self.logger.error(f"Error during optimization: {e}")
        return False, {"error": str(e)}
```

### `get_solution_summary()`

Get a summary of the last solution.

Source code in `src/figaroh/tools/robotipopt.py`

```
def get_solution_summary(self) -> str:
    """Get a summary of the last solution."""
    if self.last_solution is None:
        return "No solution available"

    info = self.last_info
    return (
        f"IPOPT Solution Summary:\n"
        f"  Status: {info['status']} - {info['status_msg']}\n"
        f"  Objective: {info['obj_val']:.6e}\n"
        f"  Iterations: {len(self.problem.iteration_data['iterations'])}\n"
        f"  Variables: {len(self.last_solution)}\n"
        f"  Constraints: {len(self.problem.get_constraint_bounds()[0])}\n"
    )
```

### `validate_problem_setup()`

Validate that the optimization problem is properly set up.

Returns:

| Type   | Description                                    |
| ------ | ---------------------------------------------- |
| `bool` | True if problem appears valid, False otherwise |

Source code in `src/figaroh/tools/robotipopt.py`

```
def validate_problem_setup(self) -> bool:
    """
    Validate that the optimization problem is properly set up.

    Returns:
        True if problem appears valid, False otherwise
    """
    try:
        self.logger.info("Validating problem setup...")

        # Check dimensions
        x0 = self.problem.get_initial_guess()
        lb, ub = self.problem.get_variable_bounds()
        cl, cu = self.problem.get_constraint_bounds()

        if len(lb) != len(ub) or len(lb) != len(x0):
            self.logger.error("Variable bounds dimension mismatch")
            return False

        if len(cl) != len(cu):
            self.logger.error("Constraint bounds dimension mismatch")
            return False

        # Test function evaluations
        obj_val = self.problem.objective(np.array(x0))
        if not np.isfinite(obj_val):
            self.logger.error(
                f"Objective function returns non-finite value: {obj_val}"
            )
            return False

        constraints = self.problem.constraints(np.array(x0))
        if len(constraints) != len(cl):
            self.logger.error("Constraint function dimension mismatch")
            return False

        if not np.all(np.isfinite(constraints)):
            self.logger.error("Constraint function returns non-finite values")
            return False

        self.logger.info("Problem validation passed")
        return True

    except Exception as e:
        self.logger.error(f"Problem validation failed: {e}")
        return False
```

## `TrajectoryOptimizationProblem(name='TrajectoryOptimization')`

Bases: `BaseOptimizationProblem`

Specialized optimization problem for trajectory optimization.

This class provides a framework for trajectory optimization problems with common patterns like waypoint parameterization and constraint handling.

Initialize trajectory optimization problem.

Source code in `src/figaroh/tools/robotipopt.py`

```
def __init__(self, name: str = "TrajectoryOptimization"):
    """Initialize trajectory optimization problem."""
    super().__init__(name)

    # Trajectory-specific data
    self.trajectory_data = {
        "times": None,
        "positions": None,
        "velocities": None,
        "accelerations": None,
    }
```

### `set_objective_function(obj_func)`

Set custom objective function.

Source code in `src/figaroh/tools/robotipopt.py`

```
def set_objective_function(self, obj_func: Callable[[np.ndarray], float]):
    """Set custom objective function."""
    self._custom_objective = obj_func
```

### `set_constraint_function(cons_func)`

Set custom constraint function.

Source code in `src/figaroh/tools/robotipopt.py`

```
def set_constraint_function(self, cons_func: Callable[[np.ndarray], np.ndarray]):
    """Set custom constraint function."""
    self._custom_constraints = cons_func
```

### `objective(x)`

Evaluate objective function.

Source code in `src/figaroh/tools/robotipopt.py`

```
def objective(self, x: np.ndarray) -> float:
    """Evaluate objective function."""
    if hasattr(self, "_custom_objective"):
        return self._custom_objective(x)
    else:
        raise NotImplementedError(
            "Must implement objective function or set custom function"
        )
```

### `constraints(x)`

Evaluate constraint functions.

Source code in `src/figaroh/tools/robotipopt.py`

```
def constraints(self, x: np.ndarray) -> np.ndarray:
    """Evaluate constraint functions."""
    if hasattr(self, "_custom_constraints"):
        return self._custom_constraints(x)
    else:
        raise NotImplementedError(
            "Must implement constraints function or set custom function"
        )
```

## `create_trajectory_solver(objective_func, constraint_func, get_bounds_func, initial_guess_func, name='CustomTrajectory')`

Factory function to create a trajectory optimization solver.

Parameters:

| Name                 | Type                                                        | Description                                                  | Default              |
| -------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ | -------------------- |
| `objective_func`     | `Callable[[ndarray], float]`                                | Function to evaluate objective                               | *required*           |
| `constraint_func`    | `Callable[[ndarray], ndarray]`                              | Function to evaluate constraints                             | *required*           |
| `get_bounds_func`    | `Callable[[], Tuple[Tuple[List, List], Tuple[List, List]]]` | Function that returns ((var_lb, var_ub), (cons_lb, cons_ub)) | *required*           |
| `initial_guess_func` | `Callable[[], List[float]]`                                 | Function that returns initial guess                          | *required*           |
| `name`               | `str`                                                       | Problem name                                                 | `'CustomTrajectory'` |

Returns:

| Type               | Description             |
| ------------------ | ----------------------- |
| `RobotIPOPTSolver` | Configured IPOPT solver |

Source code in `src/figaroh/tools/robotipopt.py`

```
def create_trajectory_solver(
    objective_func: Callable[[np.ndarray], float],
    constraint_func: Callable[[np.ndarray], np.ndarray],
    get_bounds_func: Callable[[], Tuple[Tuple[List, List], Tuple[List, List]]],
    initial_guess_func: Callable[[], List[float]],
    name: str = "CustomTrajectory",
) -> RobotIPOPTSolver:
    """
    Factory function to create a trajectory optimization solver.

    Args:
        objective_func: Function to evaluate objective
        constraint_func: Function to evaluate constraints
        get_bounds_func: Function that returns ((var_lb, var_ub),
                        (cons_lb, cons_ub))
        initial_guess_func: Function that returns initial guess
        name: Problem name

    Returns:
        Configured IPOPT solver
    """

    class CustomProblem(TrajectoryOptimizationProblem):
        def __init__(self):
            super().__init__(name)
            self.set_objective_function(objective_func)
            self.set_constraint_function(constraint_func)
            self._get_bounds = get_bounds_func
            self._get_initial = initial_guess_func

        def get_variable_bounds(self):
            return self._get_bounds()[0]

        def get_constraint_bounds(self):
            return self._get_bounds()[1]

        def get_initial_guess(self):
            return self._get_initial()

    problem = CustomProblem()
    config = IPOPTConfig.for_trajectory_optimization()
    return RobotIPOPTSolver(problem, config)
```

## Random Data Generation

## `generate_waypoints(N, robot, mlow, mhigh)`

Generate random values for joint positions, velocities, accelerations.

Parameters:

| Name    | Type | Description                   | Default    |
| ------- | ---- | ----------------------------- | ---------- |
| `N`     |      | Number of samples             | *required* |
| `robot` |      | Robot model object            | *required* |
| `mlow`  |      | Lower bound for random values | *required* |
| `mhigh` |      | Upper bound for random values | *required* |

Returns:

| Name    | Type | Description                                             |
| ------- | ---- | ------------------------------------------------------- |
| `tuple` |      | (q, v, a) joint position, velocity, acceleration arrays |

Source code in `src/figaroh/tools/randomdata.py`

```
def generate_waypoints(N, robot, mlow, mhigh):
    """Generate random values for joint positions, velocities, accelerations.

    Args:
        N: Number of samples
        robot: Robot model object
        mlow: Lower bound for random values
        mhigh: Upper bound for random values

    Returns:
        tuple: (q, v, a) joint position, velocity, acceleration arrays
    """
    has_backend = hasattr(robot, "backend")
    nq = robot.backend.nq if has_backend else robot.model.nq
    nv = robot.backend.nv if has_backend else robot.model.nv
    q = np.empty((1, nq))
    v = np.empty((1, nv))
    a = np.empty((1, nv))
    for i in range(N):
        q = np.vstack((q, np.random.uniform(low=mlow, high=mhigh, size=(nq,))))
        v = np.vstack((v, np.random.uniform(low=mlow, high=mhigh, size=(nv,))))
        a = np.vstack((a, np.random.uniform(low=mlow, high=mhigh, size=(nv,))))
    return q, v, a
```

## `generate_waypoints_fext(N, robot, mlow, mhigh)`

Generate random values for joints with external forces.

Parameters:

| Name    | Type | Description                   | Default    |
| ------- | ---- | ----------------------------- | ---------- |
| `N`     |      | Number of samples             | *required* |
| `robot` |      | Robot model object            | *required* |
| `mlow`  |      | Lower bound for random values | *required* |
| `mhigh` |      | Upper bound for random values | *required* |

Returns:

| Name    | Type | Description                                             |
| ------- | ---- | ------------------------------------------------------- |
| `tuple` |      | (q, v, a) joint position, velocity, acceleration arrays |

Source code in `src/figaroh/tools/randomdata.py`

```
def generate_waypoints_fext(N, robot, mlow, mhigh):
    """Generate random values for joints with external forces.

    Args:
        N: Number of samples
        robot: Robot model object
        mlow: Lower bound for random values
        mhigh: Upper bound for random values

    Returns:
        tuple: (q, v, a) joint position, velocity, acceleration arrays
    """
    has_backend = hasattr(robot, "backend")
    nq = robot.backend.nq if has_backend else robot.model.nq
    nv = robot.backend.nv if has_backend else robot.model.nv
    q0 = robot.q0[:7]
    v0 = np.zeros(6)
    a0 = np.zeros(6)
    q = np.empty((1, nq))
    v = np.empty((1, nv))
    a = np.empty((1, nv))
    for i in range(N):
        q_ = np.append(q0, np.random.uniform(low=mlow, high=mhigh, size=(nq - 7,)))
        q = np.vstack((q, q_))

        v_ = np.append(v0, np.random.uniform(low=mlow, high=mhigh, size=(nv - 6,)))
        v = np.vstack((v, v_))

        a_ = np.append(a0, np.random.uniform(low=mlow, high=mhigh, size=(nv - 6,)))
        a = np.vstack((a, a_))
    return q, v, a
```

## `get_torque_rand(N, robot, q, v, a, identif_config)`

Calculate random torque values including various dynamic effects.

Parameters:

| Name             | Type | Description                      | Default    |
| ---------------- | ---- | -------------------------------- | ---------- |
| `N`              |      | Number of samples                | *required* |
| `robot`          |      | Robot model object               | *required* |
| `q`              |      | Joint positions array            | *required* |
| `v`              |      | Joint velocities array           | *required* |
| `a`              |      | Joint accelerations array        | *required* |
| `identif_config` |      | Dictionary of dynamic parameters | *required* |

Returns:

| Name    | Type | Description         |
| ------- | ---- | ------------------- |
| `array` |      | Joint torques array |

Source code in `src/figaroh/tools/randomdata.py`

```
def get_torque_rand(N, robot, q, v, a, identif_config):
    """Calculate random torque values including various dynamic effects.

    Args:
        N: Number of samples
        robot: Robot model object
        q: Joint positions array
        v: Joint velocities array
        a: Joint accelerations array
        identif_config: Dictionary of dynamic parameters

    Returns:
        array: Joint torques array
    """
    has_backend = hasattr(robot, "backend")
    nv = robot.backend.nv if has_backend else robot.model.nv
    tau = np.zeros(nv * N)
    for i in range(N):
        if has_backend:
            tau_vec = robot.backend.compute_inverse_dynamics(q[i, :], v[i, :], a[i, :])
        else:
            tau_vec = pin.rnea(robot.model, robot.data, q[i, :], v[i, :], a[i, :])
        for j in range(nv):
            tau[j * N + i] = tau_vec[j]
    if identif_config["has_friction"]:
        for i in range(N):
            for j in range(nv):
                fv_term = v[i, j] * identif_config["fv"][j]
                fs_term = np.sign(v[i, j]) * identif_config["fs"][j]
                tau[j * N + i] += fv_term + fs_term
    if identif_config["has_actuator_inertia"]:
        for i in range(N):
            for j in range(nv):
                tau[j * N + i] += identif_config["Ia"][j] * a[i, j]
    if identif_config["has_joint_offset"]:
        for i in range(N):
            for j in range(nv):
                tau[j * N + i] += identif_config["off"][j]
    if identif_config["has_coupled_wrist"]:
        for i in range(N):
            for j in range(nv):
                if j == nv - 2:
                    tau[j * N + i] += (
                        identif_config["Iam6"] * v[i, nv - 1]
                        + identif_config["fvm6"] * v[i, nv - 1]
                        + identif_config["fsm6"] * np.sign(v[i, nv - 2] + v[i, nv - 1])
                    )
                if j == nv - 1:
                    tau[j * N + i] += (
                        identif_config["Iam6"] * v[i, nv - 2]
                        + identif_config["fvm6"] * v[i, nv - 2]
                        + identif_config["fsm6"] * np.sign(v[i, nv - 2] + v[i, nv - 1])
                    )
    return tau
```

## URDF Export

Applies calibrated joint parameters back onto a URDF file.

URDF exporter for identified/calibrated robot parameters.

Reads a nominal URDF and produces a modified URDF by applying parameter overlays. Parameter names encode the semantics: `d_px_joint2` is additive (geometric offset), `m_link1` is absolute (mass override).

This is used after identification or calibration to materialize results into a tangible URDF file for downstream use (simulation, visualization, model-based control).

**Two parameter categories**

Joint-level parameters (auto-applied to the URDF): These directly modify the URDF's joint origins, link inertias, and dynamics attributes. They come from figaroh's identification or calibration solvers and can be applied automatically:

```
- Joint placement (in the joint frame): ``d_px_{joint}``,
  ``d_py_{joint}``, ``d_pz_{joint}``, ``d_phix_{joint}``,
  ``d_phiy_{joint}``, ``d_phiz_{joint}``; a joint's six values are
  composed as ``origin * SE3(exp3(d_phi), d_p)``
- Joint offset / calibration (additive): ``offsetPX_{joint}``,
  ``offsetPY_{joint}``, ``offsetPZ_{joint}``, ``offsetRX_{joint}``,
  ``offsetRY_{joint}``, ``offsetRZ_{joint}``
- Mass (absolute): ``m_{link}``
- Standard inertial parameters (absolute, all ten per link):
  ``m_``, ``mx_``, ``my_``, ``mz_``, ``Ixx_``, ``Ixy_``, ``Iyy_``,
  ``Ixz_``, ``Iyz_``, ``Izz_`` + ``{link}``; see
  :func:`_apply_standard_inertial` for the conventions
- Viscous/static friction (absolute): ``fv_{joint}``, ``fs_{joint}``
- Armature (absolute): ``Ia_{joint}``
- Joint elasticity (additive): ``k_PX_{joint}``, ..., ``k_RZ_{joint}``,
  stored as a custom ``<dynamics elasticity>`` attribute that URDF
  parsers ignore: it does not change the reloaded kinematics

Inertial targets (``m_`` ... ``Izz_``) name a link, or a moving joint
as identification does (``get_standard_parameters`` keys by Pinocchio
joint); a joint resolves to its child link.

Refused (``ValueError``, nothing written): legacy ``off_{joint}``; names
targeting a joint or link the URDF does not have; a joint carrying both
``offset*`` and ``d_*``; a partial inertial set (first moments or tensor
entries without all ten parameters); a joint target whose child link has
massive links attached by fixed joints (Pinocchio merges them into the
joint's body, so the estimate does not belong to one URDF link); a set
with ``m <= 0``; and, unless ``allow_infeasible=True``, a set that is not
physically consistent.
```

Metrology frame parameters (user-defined, not auto-applied): These define the transformation between the robot (URDF) and the external measurement system (mocap, camera, chessboard, etc.). They depend on the **calibration setup**, not on the URDF itself. `export_urdf()` will **not** auto-apply them; instead it logs a reminder and returns them as metadata for the user to configure::

```
    base_px, base_py, base_pz, base_phix, base_phiy, base_phiz
        Transform from the **metrology frame** (e.g. mocap world,
        Vicon origin) to the robot's ``base_link``.
        Default: identity (no offset from origin).

    pEEx_{frame}, pEEy_{frame}, pEEz_{frame}
    phiEEx_{frame}, phiEEy_{frame}, phiEEz_{frame}
        Transform from the last robot joint (e.g. ``arm_7_joint``,
        ``head_2_link``) to the **measurement frame** mounted on the
        end-effector. What this frame is depends on calibration type:

        - **Mocap calibration**: optical marker cluster frame
          (markers attached to the end-effector).
        - **Eye-hand calibration**: camera optical frame
          (e.g. ``xtion_rgb_optical_frame``) or chessboard frame
          (pattern on the gripper).
        Default: identity (measurement frame coincides with the joint).
```

Typical usage::

```
from figaroh.tools.urdf_exporter import export_urdf, frame_settings_doc

# Joint-level params (auto-applied to URDF)
params = {
    "d_px_joint2": 0.05,
    "m_link1": 2.5,
    "fv_joint1": 0.2,
}
modified_path = export_urdf("robot.urdf", params, verbose=True)

# Metrology frames (user-defined — see frame_settings_doc())
defaults = frame_settings_doc()
# → prints descriptions + default values for base and EE frame params
```

## `is_kinematic_correction(name)`

Whether `name` is a joint correction that changes the reloaded kinematics: a `full_params` placement (`d_*`) or a joint offset (`offset*`). Frame, elastic and dynamic parameters are not.

Source code in `src/figaroh/tools/urdf_exporter.py`

```
def is_kinematic_correction(name: str) -> bool:
    """Whether ``name`` is a joint correction that changes the reloaded
    kinematics: a ``full_params`` placement (``d_*``) or a joint offset
    (``offset*``). Frame, elastic and dynamic parameters are not."""
    parsed = _parse_param_name(name)
    return parsed is not None and parsed[0] in ("joint_placement", "joint_offset")
```

## `frame_settings_doc(*, calibration_type=None, verbose=True)`

Return default metrology-frame parameter values with explanations.

These parameters define the transformation between the robot and the external measurement system. They are **not** intrinsic to the URDF and must be configured by the user for each calibration setup.

Parameters:

| Name               | Type            | Description                                                                          | Default |
| ------------------ | --------------- | ------------------------------------------------------------------------------------ | ------- |
| `calibration_type` | `Optional[str]` | Optional hint for context-specific defaults. "mocap", "eye_hand", or None (generic). | `None`  |
| `verbose`          | `bool`          | If True (default), prints descriptions to stderr.                                    | `True`  |

Returns:

| Type   | Description                                                                                                           |
| ------ | --------------------------------------------------------------------------------------------------------------------- |
| `dict` | dict with default values for all base-frame and EE-frame params:: { "base_px": 0.0, ... "pEEx_arm_7_link": 0.0, ... } |

Source code in `src/figaroh/tools/urdf_exporter.py`

```
def frame_settings_doc(
    *, calibration_type: Optional[str] = None, verbose: bool = True
) -> dict:
    """Return default metrology-frame parameter values with explanations.

    These parameters define the transformation between the robot and the
    external measurement system.  They are **not** intrinsic to the URDF
    and must be configured by the user for each calibration setup.

    Args:
        calibration_type: Optional hint for context-specific defaults.
            ``"mocap"``, ``"eye_hand"``, or ``None`` (generic).
        verbose: If True (default), prints descriptions to stderr.

    Returns:
        dict with default values for all base-frame and EE-frame params::

            {
                "base_px": 0.0, ...
                "pEEx_arm_7_link": 0.0, ...
            }
    """
    defaults: dict = {}

    # Base frame defaults — identity (metrology origin = robot base)
    for name, _, _, _ in _METROLOGY_REGISTRY:
        if name.startswith("base_"):
            defaults[name] = 0.0

    # EE frame defaults — identity (sensor/marker frame = last joint)
    if verbose:
        if calibration_type == "mocap":
            target = "arm_7_link"
            note = (
                "Mocap calibration: EE measurement frame is the optical marker "
                "cluster attached to the end-effector (e.g. arm_7_link)."
            )
        elif calibration_type == "eye_hand":
            target = "head_2_link"
            note = (
                "Eye-hand calibration: EE measurement frame is the camera "
                "optical frame (relative to head_2_link) or the chessboard "
                "attached to the gripper."
            )
        else:
            target = "<frame_name>"
            note = (
                "EE measurement frame: typically a marker cluster, camera "
                "optical frame, or chessboard attached to the end-effector.  "
                "Replace ``<frame_name>`` with the actual robot joint/link name."
            )
        logger.info(
            "Metrology frame defaults (see frame_settings_doc()):\n"
            "  Base frame : metrology origin → robot base_link\n"
            "              default = identity (no offset)\n"
            "  EE frame   : last joint → measurement frame\n"
            "              default = identity\n"
            "  %s\n"
            "  To customize, pass e.g. pEEx_%s = <value> to export_urdf() "
            "and configure your\n"
            "  controller or calibration pipeline accordingly.",
            note,
            target,
        )

    return defaults
```

## `export_urdf(nominal_urdf_path, params, *, output_path=None, verbose=False, allow_infeasible=False, merged_bodies='refuse', psd_eig_tol=-1e-10)`

Apply identified/calibrated **joint-level** parameters to a nominal URDF.

This function **auto-applies** parameters that modify the URDF directly (joint placements, mass, inertias, friction, etc. — see module docstring). It does **not** apply metrology frame parameters (`base_*`, `pEE*`, `phiEE*`); those depend on the calibration setup and must be configured by the user — see :func:`frame_settings_doc`.

Parameters:

| Name                | Type                         | Description                                                                                                                                                                                                                                                                                                                                                     | Default    |
| ------------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `nominal_urdf_path` | `Union[str, Path]`           | Path to the nominal (reference) URDF file.                                                                                                                                                                                                                                                                                                                      | *required* |
| `params`            | `dict`                       | Dictionary of {parameter_name: value} pairs. Joint-level params are applied automatically. Metrology frame params (base\_\*, pEE\*, phiEE\*) are logged and collected for the caller but not auto-applied to the URDF — see :func:frame_settings_doc.                                                                                                           | *required* |
| `output_path`       | `Optional[Union[str, Path]]` | Path for the modified URDF. If None (default), writes to <stem>\_modified.urdf beside the nominal URDF.                                                                                                                                                                                                                                                         | `None`     |
| `verbose`           | `bool`                       | If True, log which params were applied.                                                                                                                                                                                                                                                                                                                         | `False`    |
| `allow_infeasible`  | `bool`                       | Write a complete inertial set whose pseudo-inertia is not positive semidefinite, with a warning, instead of raising.                                                                                                                                                                                                                                            | `False`    |
| `merged_bodies`     | `str`                        | What to do when an inertial set names a joint whose Pinocchio body also holds massive links attached by fixed joints. "refuse" (default) raises. "subtract_fixed" writes the child link p10_est - p10_fixed_nominal (the fixed links' nominal share, from the nominal model, removed; they keep their CAD values) and refuses a remainder that is not physical. | `'refuse'` |
| `psd_eig_tol`       | `float`                      | Smallest pseudo-inertia eigenvalue still accepted as feasible. Pass the tolerance the estimate was accepted with, so an estimate accepted upstream is not refused here.                                                                                                                                                                                         | `-1e-10`   |

Returns:

| Type  | Description                                                         |
| ----- | ------------------------------------------------------------------- |
| `str` | Absolute path to the modified URDF file (joint params applied). Use |
| `str` | func:frame_settings_doc to configure metrology frame params         |
| `str` | separately.                                                         |

Raises:

| Type                | Description                                                                                                                                                                                                                                                                                                                                                          |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `FileNotFoundError` | If nominal_urdf_path does not exist.                                                                                                                                                                                                                                                                                                                                 |
| `ValueError`        | If a parameter name is unknown, names a joint or link the URDF does not have, belongs to a category the exporter does not write (legacy off\_\*), a joint carries both offset\* and d\_\* corrections, or an inertial set is partial, targets a merged Pinocchio body, has m \<= 0 or (unless allow_infeasible) is physically inconsistent. Nothing is written then. |

Source code in `src/figaroh/tools/urdf_exporter.py`

```
def export_urdf(
    nominal_urdf_path: Union[str, Path],
    params: dict,
    *,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    allow_infeasible: bool = False,
    merged_bodies: str = "refuse",
    psd_eig_tol: float = -1e-10,
) -> str:
    """Apply identified/calibrated **joint-level** parameters to a nominal URDF.

    This function **auto-applies** parameters that modify the URDF directly
    (joint placements, mass, inertias, friction, etc. — see module docstring).
    It does **not** apply metrology frame parameters (``base_*``, ``pEE*``,
    ``phiEE*``); those depend on the calibration setup and must be
    configured by the user — see :func:`frame_settings_doc`.

    Args:
        nominal_urdf_path: Path to the nominal (reference) URDF file.
        params: Dictionary of ``{parameter_name: value}`` pairs.
            **Joint-level params** are applied automatically.
            **Metrology frame params** (``base_*``, ``pEE*``, ``phiEE*``)
            are logged and collected for the caller but **not** auto-applied
            to the URDF — see :func:`frame_settings_doc`.
        output_path: Path for the modified URDF. If ``None`` (default),
            writes to ``<stem>_modified.urdf`` beside the nominal URDF.
        verbose: If True, log which params were applied.
        allow_infeasible: Write a complete inertial set whose pseudo-inertia
            is not positive semidefinite, with a warning, instead of raising.
        merged_bodies: What to do when an inertial set names a joint whose
            Pinocchio body also holds massive links attached by fixed joints.
            ``"refuse"`` (default) raises. ``"subtract_fixed"`` writes the
            child link ``p10_est - p10_fixed_nominal`` (the fixed links'
            nominal share, from the nominal model, removed; they keep their
            CAD values) and refuses a remainder that is not physical.
        psd_eig_tol: Smallest pseudo-inertia eigenvalue still accepted as
            feasible. Pass the tolerance the estimate was accepted with, so
            an estimate accepted upstream is not refused here.

    Returns:
        Absolute path to the modified URDF file (joint params applied).  Use
        :func:`frame_settings_doc` to configure metrology frame params
        separately.

    Raises:
        FileNotFoundError: If *nominal_urdf_path* does not exist.
        ValueError: If a parameter name is unknown, names a joint or link
            the URDF does not have, belongs to a category the exporter does
            not write (legacy ``off_*``), a joint carries both ``offset*``
            and ``d_*`` corrections, or an inertial set is partial, targets
            a merged Pinocchio body, has ``m <= 0`` or (unless
            *allow_infeasible*) is physically inconsistent. Nothing is
            written then.
    """
    if merged_bodies not in ("refuse", "subtract_fixed"):
        raise ValueError(
            f"merged_bodies must be 'refuse' or 'subtract_fixed', "
            f"not {merged_bodies!r}"
        )
    nominal_path = Path(nominal_urdf_path)
    if not nominal_path.exists():
        raise FileNotFoundError(f"URDF not found: {nominal_path}")

    # Determine output path
    if output_path is None:
        output_path = nominal_path.with_stem(nominal_path.stem + "_modified")
    output_path = Path(output_path)

    # Parse URDF
    tree = ET.parse(str(nominal_path))
    doc = tree.getroot()

    # Separate joint params (auto-apply) from frame params (user-defined)
    frame_params: dict = {}
    placements: dict = {}
    inertials: dict = {}
    offsets: set = set()

    for name, value in params.items():
        parsed = _parse_param_name(name)
        if parsed is not None:
            category, target, idx, is_additive = parsed
            if category == "joint_placement":
                placements.setdefault(target, np.zeros(6))[idx] += value
                if verbose:
                    logger.info("%s → joint_placement.%s (%.4f)", name, target, value)
                continue
            if category in _INERTIAL_CATEGORIES:
                inertials.setdefault(target, {})[_inertial_key(name, target)] = value
                if verbose:
                    logger.info("%s → inertial.%s (%.4f)", name, target, value)
                continue
            handler = _HANDLERS.get(category)
            if handler is None:
                raise ValueError(
                    f"Parameter '{name}': the exporter does not write "
                    f"{_UNSUPPORTED.get(category, category)}"
                )
            if category == "joint_offset":
                offsets.add(target)
            if verbose:
                action = "additive" if is_additive else "absolute"
                logger.info(
                    "%s → %s.%s %s (%.4f)", name, category, target, action, value
                )
            handler(doc, target, idx, value, is_additive)
            continue

        # Check if it's a metrology frame param (base_*, pEE*, phiEE*)
        frame_parsed = _parse_frame_param_name(name)
        if frame_parsed is not None:
            cat, f_target, idx = frame_parsed
            frame_params[name] = value
            desc = _FRAME_PARAM_DESCRIPTIONS.get(cat, cat)
            logger.info(
                "Metrology frame param '%s' = %.4f — not auto-applied to URDF.  "
                "This defines: %s  See frame_settings_doc() for defaults and "
                "explanations.",
                name,
                value,
                desc,
            )
            continue

        raise ValueError(
            f"Unknown parameter '{name}'. "
            f"Recognized joint-level categories: joint placement (d_px_*), "
            f"joint offset (offsetRX_*), mass (m_*), friction (fv_*, fs_*), "
            f"armature (Ia_*), elasticity (k_*), inertial (mx_*, Ixx_*).  "
            f"Metrology frame params: base_*, pEE*, phiEE*."
        )

    both = sorted(offsets & set(placements))
    if both:
        # the fit never produces both; their order would be a guess
        raise ValueError(
            f"Joint(s) {both} carry both offset* and d_* corrections; "
            f"use one calibration level per joint"
        )
    for target, xyz_rpy in placements.items():
        _apply_joint_placement(doc, target, xyz_rpy)
    _apply_inertials(
        doc, inertials, allow_infeasible, merged_bodies, nominal_path, psd_eig_tol
    )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), xml_declaration=True, encoding="utf-8")

    if frame_params and verbose:
        logger.info(
            "The following metrology frame parameters were **not** applied "
            "to the URDF:\n  %s\nUse frame_settings_doc() to review defaults.",
            "  ".join(f"{k}={v}" for k, v in frame_params.items()),
        )

    return str(output_path.resolve())
```

## Geometric Calibration Export

Produces the PAL Robotics `robot_state_publisher.geometric_calibration` runtime-correction YAML (e.g. `/etc/calibration/master_calibration.yaml`) from a solved calibrator's redistributed parameters — a different deploy target from URDF Export above (a small runtime overlay applied on top of the original URDF, not a modified URDF file).

PAL Robotics `robot_state_publisher` geometric-calibration deploy file.

Produces the runtime joint-correction YAML PAL robots (TIAGo, TIAGo Pro, TALOS, ...) read at `/etc/calibration/master_calibration.yaml` (`robot_state_publisher: geometric_calibration: {<joint>_<axis>: value}`), directly from a solved `BaseCalibration`.

This is a different deploy target from :mod:`figaroh.tools.urdf_exporter`: that module bakes corrections into a *modified URDF file*; this one produces a small *runtime correction overlay* PAL's `robot_state_publisher` applies on top of the original, unmodified URDF at startup. Both read the same `d_px_{joint}`-style (`full_params`) and `offsetRZ_{joint}`-style (`joint_offset`) parameter names — this module reuses :func:`figaroh.tools.urdf_exporter._parse_param_name` rather than re-deriving that parsing. The keys are additive deltas on the URDF `<origin>` xyz and rpy as written; pass `nominal_urdf` so they are exact (figaroh-plus#123).

The values come from :meth:`~figaroh.calibration.base_calibration.BaseCalibration.redistribute_parameters`: for the `structural` method, the weighted minimum-norm lift of the fitted base parameters onto every joint (figaroh-plus#111), so a joint whose correction was represented by another parameter in the fit still gets its share; for the other estimation methods, the fitted joint parameters. Use :meth:`~figaroh.calibration.base_calibration.BaseCalibration.joint_corrections` to write the same values into a URDF.

## `build_geometric_calibration(calibrator, *, min_sigma=None, nominal_urdf=None, allow_failed_fit=False)`

Build a PAL `robot_state_publisher.geometric_calibration` dict from a solved `BaseCalibration` instance's redistributed parameters.

Per-joint kinematic corrections are included: `full_params` placements (`d_px_{joint}` etc.) and `joint_offset` offsets (`offsetRZ_{joint}` etc., figaroh-plus#123), both converted to additive deltas on the joint's URDF origin. Marker/tip, base-frame and elasticity parameters are not kinematic joint corrections in :func:`~figaroh.tools.urdf_exporter._parse_param_name`'s registry and are dropped by the category check. The base frame needs no exclusion: the lift holds the base-frame rows at 0, so every lifted value is a joint correction.

Parameters:

| Name               | Type              | Description                                                                                                                                                                                                                                                   | Default    |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `calibrator`       |                   | A solved BaseCalibration instance (solve() already called).                                                                                                                                                                                                   | *required* |
| `min_sigma`        | `Optional[float]` | If given, only include parameters with                                                                                                                                                                                                                        | value      |
| `nominal_urdf`     | `Optional[str]`   | The URDF the robot runs, to which the deltas are added. Its <origin> values, as written, are the reference (figaroh-plus#123). Without it they are recovered from calibrator.model, which is exact unless an origin's rpy is at or near pitch = +-pi/2 or has | pitch      |
| `allow_failed_fit` | `bool`            | export although the fit stage failed, for diagnosis only (warned and recorded; see BaseCalibration.require_exportable).                                                                                                                                       | `False`    |

Returns:

| Type                                     | Description                                                        |
| ---------------------------------------- | ------------------------------------------------------------------ |
| `Dict[str, Dict[str, Dict[str, float]]]` | {"robot_state_publisher": {"geometric_calibration": {key: value}}} |

Raises:

| Type               | Description                                                                                                                                    |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `CalibrationError` | The fit stage failed (export is recorded as failed), or propagated from redistribute_parameters() if solve()/ create_param_list() haven't run. |

Source code in `src/figaroh/tools/geometric_calibration_export.py`

```
def build_geometric_calibration(
    calibrator,
    *,
    min_sigma: Optional[float] = None,
    nominal_urdf: Optional[str] = None,
    allow_failed_fit: bool = False,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build a PAL ``robot_state_publisher.geometric_calibration`` dict
    from a solved ``BaseCalibration`` instance's redistributed parameters.

    Per-joint kinematic corrections are included: ``full_params``
    placements (``d_px_{joint}`` etc.) and ``joint_offset`` offsets
    (``offsetRZ_{joint}`` etc., figaroh-plus#123), both converted to
    additive deltas on the joint's URDF origin. Marker/tip, base-frame and
    elasticity parameters are not kinematic joint corrections in
    :func:`~figaroh.tools.urdf_exporter._parse_param_name`'s registry and
    are dropped by the category check. The base frame needs no exclusion:
    the lift holds the base-frame rows at 0, so every lifted value is a joint
    correction.

    Args:
        calibrator: A solved ``BaseCalibration`` instance (``solve()``
            already called).
        min_sigma: If given, only include parameters with
            ``|value| / std_dev >= min_sigma``: a conservative deploy
            that leaves statistically insignificant corrections at
            nominal. ``None`` (default) includes every joint-placement
            parameter. Whether it generalizes better is robot- and
            data-specific; check on held-out postures.
        nominal_urdf: The URDF the robot runs, to which the deltas are
            added. Its ``<origin>`` values, as written, are the reference
            (figaroh-plus#123). Without it they are recovered from
            ``calibrator.model``, which is exact unless an origin's rpy is
            at or near pitch = +-pi/2 or has |pitch| > pi/2.
        allow_failed_fit: export although the ``fit`` stage ``failed``,
            for diagnosis only (warned and recorded; see
            ``BaseCalibration.require_exportable``).

    Returns:
        ``{"robot_state_publisher": {"geometric_calibration": {key: value}}}``

    Raises:
        CalibrationError: The ``fit`` stage ``failed`` (``export`` is
            recorded as failed), or propagated from
            ``redistribute_parameters()`` if ``solve()``/
            ``create_param_list()`` haven't run.
    """
    require = getattr(calibrator, "require_exportable", None)
    if require is not None:
        require(allow_failed_fit)
    redistributed = calibrator.redistribute_parameters()

    corrections: Dict[str, np.ndarray] = {}
    offsets = set()
    for name, info in redistributed.items():
        parsed = _parse_param_name(name)
        # joint_offset: same joint-frame transform as the matching d_* (#123)
        if parsed is None or parsed[0] not in ("joint_placement", "joint_offset"):
            continue
        category, target, sub_idx, _ = parsed

        value, std_dev = info["value"], info["std_dev"]
        if min_sigma is not None:
            sigma = abs(value) / std_dev if std_dev > 0 else float("inf")
            if sigma < min_sigma:
                continue

        corrections.setdefault(target, np.zeros(6))[sub_idx] = value
        if category == "joint_offset":
            offsets.add(target)

    model = getattr(calibrator, "model", None)
    origins = _urdf_origins(nominal_urdf) if nominal_urdf is not None else {}
    geometric_calibration: Dict[str, float] = {}
    for target, xyz_rpy in corrections.items():
        placement, rpy = pin.SE3.Identity(), None
        if target in origins:
            xyz, rpy = origins[target]
            placement = pin.SE3(pin.rpy.rpyToMatrix(rpy), xyz)
        elif nominal_urdf is not None:
            raise ValueError(f"Joint '{target}' not found in {nominal_urdf}")
        elif model is not None and model.existJointName(target):
            placement = _urdf_origin(model, target)
        delta = _origin_delta(placement, xyz_rpy, offset=target in offsets, rpy=rpy)
        for sub_idx, value in enumerate(delta):
            if abs(value) > 1e-12:
                key = f"{_pal_joint_name(target)}_{_AXIS_SUFFIX[sub_idx]}"
                geometric_calibration[key] = float(value)

    return {"robot_state_publisher": {"geometric_calibration": geometric_calibration}}
```

## `export_geometric_calibration_yaml(calibrator, output_path, *, min_sigma=None, header_comment=None, nominal_urdf=None, allow_failed_fit=False)`

:func:`build_geometric_calibration` + write as YAML, PAL deploy-ready.

PAL's `master_calibration.yaml` layout (`robot_state_publisher: geometric_calibration:` with `<joint>_<axis>` keys), so the output drops in at `/etc/calibration/master_calibration.yaml` on a PAL robot.

Parameters:

| Name               | Type              | Description                                                                                                               | Default    |
| ------------------ | ----------------- | ------------------------------------------------------------------------------------------------------------------------- | ---------- |
| `calibrator`       |                   | A solved BaseCalibration instance.                                                                                        | *required* |
| `output_path`      | `str`             | Destination YAML file path.                                                                                               | *required* |
| `min_sigma`        | `Optional[float]` | See :func:build_geometric_calibration.                                                                                    | `None`     |
| `nominal_urdf`     | `Optional[str]`   | See :func:build_geometric_calibration; pass it.                                                                           | `None`     |
| `header_comment`   | `Optional[str]`   | Optional single-line comment written above the YAML document (e.g. source data file, sample count, RMSE), for provenance. | `None`     |
| `allow_failed_fit` | `bool`            | See :func:build_geometric_calibration.                                                                                    | `False`    |

Returns:

| Type  | Description                           |
| ----- | ------------------------------------- |
| `str` | output_path, unchanged, for chaining. |

Source code in `src/figaroh/tools/geometric_calibration_export.py`

```
def export_geometric_calibration_yaml(
    calibrator,
    output_path: str,
    *,
    min_sigma: Optional[float] = None,
    header_comment: Optional[str] = None,
    nominal_urdf: Optional[str] = None,
    allow_failed_fit: bool = False,
) -> str:
    """:func:`build_geometric_calibration` + write as YAML, PAL deploy-ready.

    PAL's ``master_calibration.yaml`` layout (``robot_state_publisher:
    geometric_calibration:`` with ``<joint>_<axis>`` keys), so the output
    drops in at ``/etc/calibration/master_calibration.yaml`` on a PAL robot.

    Args:
        calibrator: A solved ``BaseCalibration`` instance.
        output_path: Destination YAML file path.
        min_sigma: See :func:`build_geometric_calibration`.
        nominal_urdf: See :func:`build_geometric_calibration`; pass it.
        header_comment: Optional single-line comment written above the
            YAML document (e.g. source data file, sample count, RMSE), for
            provenance.
        allow_failed_fit: See :func:`build_geometric_calibration`.

    Returns:
        ``output_path``, unchanged, for chaining.
    """
    data = build_geometric_calibration(
        calibrator,
        min_sigma=min_sigma,
        nominal_urdf=nominal_urdf,
        allow_failed_fit=allow_failed_fit,
    )
    with open(output_path, "w") as f:
        if header_comment:
            f.write(f"# {header_comment}\n")
        yaml.dump(data, f, sort_keys=True, default_flow_style=False)
    logger.info("Geometric calibration written to %s", output_path)
    return output_path
```

## Export Validation

FK consistency checks and interactive viser visualization for a calibration-exported URDF.

URDF model comparison and validation utilities.

Provides tools to compare two URDF models (nominal vs. modified after parameter application) through forward-kinematics analysis and viser-based visualization.

Typical usage::

```
from figaroh.tools.export_validation import URDFComparison

comp = URDFComparison("robot.urdf", "robot_modified.urdf")
print(comp.fk_consistency_check())
# → FkConsistencyResult(rmse_position=0.051, rmse_orientation=0.098, ...)

# Interactive viser visualization:
comp.show_trajectory_animation()
comp.show_static_grid()
```

## `PoseDelta(translation, rotation, twist, q)`

FK pose difference between two models at a single configuration.

The delta is expressed as `M_nominal⁻¹ · M_modified`, i.e. the SE3 transform from the nominal end-effector frame to the modified one.

### `position_error()`

Euclidean distance in meters.

Source code in `src/figaroh/tools/export_validation.py`

```
def position_error(self) -> float:
    """Euclidean distance in meters."""
    return float(np.linalg.norm(self.translation))
```

### `orientation_error()`

Angle-axis magnitude in radians.

Source code in `src/figaroh/tools/export_validation.py`

```
def orientation_error(self) -> float:
    """Angle-axis magnitude in radians."""
    return float(np.linalg.norm(self.twist[3:]))
```

## `FkConsistencyResult(rmse_position, rmse_orientation, max_position, max_orientation, per_sample)`

Aggregated FK consistency check — compares nominal vs. exported URDF FK.

## `PoseError(q, position_error_mm, orientation_error_deg, pose_delta, label='')`

FK error at a single static configuration with user label.

## `URDFComparison(nominal_urdf, modified_urdf)`

Compare two URDF models via FK analysis and viser visualization.

Loads both URDFs into Pinocchio for FK computation and into yourdfpy (when needed) for viser rendering.

Parameters:

| Name            | Type               | Description                                   | Default    |
| --------------- | ------------------ | --------------------------------------------- | ---------- |
| `nominal_urdf`  | `Union[str, Path]` | Path to the original/reference URDF.          | *required* |
| `modified_urdf` | `Union[str, Path]` | Path to the URDF after parameter application. | *required* |

Source code in `src/figaroh/tools/export_validation.py`

```
def __init__(
    self,
    nominal_urdf: Union[str, Path],
    modified_urdf: Union[str, Path],
):
    if pin is None:
        raise ImportError("pinocchio is required for URDFComparison")

    self.nominal_path = Path(nominal_urdf)
    self.modified_path = Path(modified_urdf)

    for p in (self.nominal_path, self.modified_path):
        if not p.exists():
            raise FileNotFoundError(f"URDF not found: {p}")

    # Pinocchio models for FK computation
    self.model_a = pin.buildModelFromUrdf(str(self.nominal_path))
    self.model_b = pin.buildModelFromUrdf(str(self.modified_path))
    self.data_a = self.model_a.createData()
    self.data_b = self.model_b.createData()

    # Identify end-effector: deepest frame in the model tree.
    # We skip frames attached directly to the universe (joint parent = 0)
    # to avoid picking fixed-base sensor/camera links.
    self.ee_frame_a, self.ee_id_a = self._detect_ee(self.model_a)
    self.ee_frame_b, self.ee_id_b = self._detect_ee(self.model_b)
```

### `fk_consistency_check(n_samples=100, seed=42)`

Check the exported URDF reproduces the calibrated FK numerically.

This is a file-integrity check — it verifies the exported URDF file is self-consistent with the calibration results. It does NOT test against ground-truth measurements (use calibration validation for that).

Parameters:

| Name        | Type  | Description                                | Default |
| ----------- | ----- | ------------------------------------------ | ------- |
| `n_samples` | `int` | Number of random configurations to sample. | `100`   |
| `seed`      | `int` | Random seed for reproducibility.           | `42`    |

Returns:

| Type                  | Description                                                 |
| --------------------- | ----------------------------------------------------------- |
| `FkConsistencyResult` | FkConsistencyResult with aggregated FK consistency metrics. |

Source code in `src/figaroh/tools/export_validation.py`

```
def fk_consistency_check(
    self,
    n_samples: int = 100,
    seed: int = 42,
) -> FkConsistencyResult:
    """Check the exported URDF reproduces the calibrated FK numerically.

    This is a file-integrity check — it verifies the exported URDF file
    is self-consistent with the calibration results. It does NOT test
    against ground-truth measurements (use calibration validation for
    that).

    Args:
        n_samples: Number of random configurations to sample.
        seed: Random seed for reproducibility.

    Returns:
        FkConsistencyResult with aggregated FK consistency metrics.
    """
    configs = self._sample_configs(n_samples, seed)
    deltas = [self._compute_delta(q) for q in configs]
    return self._aggregate(deltas)
```

### `static_poses(poses=None)`

Compute FK errors at specific joint configurations.

Parameters:

| Name    | Type                                          | Description                                                                                                                                   | Default |
| ------- | --------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| `poses` | `Optional[List[Union[List[float], ndarray]]]` | List of joint configuration vectors. If None, a default set of 20 meaningful poses is used (covering home, limits, mixed-angle combinations). | `None`  |

Returns:

| Type              | Description                               |
| ----------------- | ----------------------------------------- |
| `List[PoseError]` | List of PoseError, one per configuration. |

Source code in `src/figaroh/tools/export_validation.py`

```
def static_poses(
    self,
    poses: Optional[List[Union[List[float], np.ndarray]]] = None,
) -> List[PoseError]:
    """Compute FK errors at specific joint configurations.

    Args:
        poses: List of joint configuration vectors. If None, a default
            set of 20 meaningful poses is used (covering home, limits,
            mixed-angle combinations).

    Returns:
        List of PoseError, one per configuration.
    """
    if poses is None:
        poses = self._default_poses()
    results = []
    for i, q in enumerate(poses):
        q_arr = np.asarray(q, dtype=float)
        delta = self._compute_delta(q_arr)
        results.append(
            PoseError(
                q=q_arr,
                position_error_mm=delta.position_error() * 1000,
                orientation_error_deg=delta.orientation_error() * 180 / np.pi,
                pose_delta=delta,
                label=f"pose_{i}",
            )
        )
    return results
```

### `show_overlay(server=None, port=8080, orig_color=(51, 102, 229), mod_color=(229, 51, 51), duration=5.0)`

Display both models overlaid in viser.

Parameters:

| Name         | Type                   | Description                                       | Default          |
| ------------ | ---------------------- | ------------------------------------------------- | ---------------- |
| `server`     |                        | Existing viser server, or None to create one.     | `None`           |
| `port`       | `int`                  | Port for new server (ignored if server is given). | `8080`           |
| `orig_color` | `Tuple[int, int, int]` | RGB color for the original model (0-255).         | `(51, 102, 229)` |
| `mod_color`  | `Tuple[int, int, int]` | RGB color for the modified model (0-255).         | `(229, 51, 51)`  |
| `duration`   | `float`                | Seconds to keep the display open.                 | `5.0`            |

Source code in `src/figaroh/tools/export_validation.py`

```
def show_overlay(
    self,
    server=None,
    port: int = 8080,
    orig_color: Tuple[int, int, int] = (51, 102, 229),
    mod_color: Tuple[int, int, int] = (229, 51, 51),
    duration: float = 5.0,
):
    """Display both models overlaid in viser.

    Args:
        server: Existing viser server, or None to create one.
        port: Port for new server (ignored if *server* is given).
        orig_color: RGB color for the original model (0-255).
        mod_color: RGB color for the modified model (0-255).
        duration: Seconds to keep the display open.
    """
    viz = self._get_viser_server(server, port)
    urdf_orig, urdf_mod = self._load_yourdfpy()

    from viser.extras import ViserUrdf

    try:
        viz.scene.remove("/robot")
    except Exception:
        pass

    ViserUrdf(
        viz,
        urdf_or_path=urdf_orig,
        root_node_name="/robot/original",
        load_meshes=True,
    )
    ViserUrdf(
        viz,
        urdf_or_path=urdf_mod,
        root_node_name="/robot/modified",
        load_meshes=True,
    )
    print(f"[vis] Both models displayed at http://localhost:{viz.get_port()}")
    print(f"[vis] Original (RGB={orig_color}) / Modified (RGB={mod_color})")
    time.sleep(duration)
```

### `show_trajectory_animation(n_configs=50, seed=42, server=None, port=8080, duration=10.0)`

Animate through random joint configs, tracing end-effector paths.

Parameters:

| Name        | Type    | Description                                       | Default |
| ----------- | ------- | ------------------------------------------------- | ------- |
| `n_configs` | `int`   | Number of random configurations to animate.       | `50`    |
| `seed`      | `int`   | Random seed.                                      | `42`    |
| `server`    |         | Existing viser server, or None to create one.     | `None`  |
| `port`      | `int`   | Port for new server.                              | `8080`  |
| `duration`  | `float` | Seconds to keep the display open after animation. | `10.0`  |

Source code in `src/figaroh/tools/export_validation.py`

```
def show_trajectory_animation(
    self,
    n_configs: int = 50,
    seed: int = 42,
    server=None,
    port: int = 8080,
    duration: float = 10.0,
):
    """Animate through random joint configs, tracing end-effector paths.

    Args:
        n_configs: Number of random configurations to animate.
        seed: Random seed.
        server: Existing viser server, or None to create one.
        port: Port for new server.
        duration: Seconds to keep the display open after animation.
    """
    import trimesh

    viz = self._get_viser_server(server, port)
    urdf_orig, urdf_mod = self._load_yourdfpy()
    from viser.extras import ViserUrdf

    vis_orig = ViserUrdf(
        viz,
        urdf_or_path=urdf_orig,
        root_node_name="/robot/orig_anim",
        load_meshes=True,
    )
    vis_mod = ViserUrdf(
        viz,
        urdf_or_path=urdf_mod,
        root_node_name="/robot/mod_anim",
        load_meshes=True,
    )

    configs = self._sample_configs(n_configs, seed)
    orig_trace: List[np.ndarray] = []
    mod_trace: List[np.ndarray] = []

    for q in configs:
        cfg_dict = self._pinocchio_config_to_dict(q)
        vis_orig.update_cfg(cfg_dict)
        vis_mod.update_cfg(cfg_dict)
        pin.forwardKinematics(self.model_a, self.data_a, q)
        pin.updateFramePlacements(self.model_a, self.data_a)
        pin.forwardKinematics(self.model_b, self.data_b, q)
        pin.updateFramePlacements(self.model_b, self.data_b)
        orig_trace.append(self.data_a.oMf[self.ee_id_a].translation.copy())
        mod_trace.append(self.data_b.oMf[self.ee_id_b].translation.copy())
        time.sleep(0.05)

    # Draw path traces
    r = 0.01
    sphere_mesh = trimesh.primitives.Sphere(radius=r)
    for pts, color, label in [
        (orig_trace, (51, 102, 229), "orig"),
        (mod_trace, (229, 51, 51), "mod"),
    ]:
        for i, pt in enumerate(pts):
            viz.scene.add_mesh_simple(
                f"/traces/{label}_{i}",
                sphere_mesh.vertices,
                sphere_mesh.faces,
                color=color,
                position=tuple(pt),
            )
    print(f"[vis] Traced {n_configs} configs at http://localhost:{viz.get_port()}")
    time.sleep(duration)
```

### `show_static_grid(poses=None, server=None, port=8080, duration=15.0, spacing=2.0)`

Display a grid of static configurations with error labels.

Each cell shows both models at the same configuration, with the position (mm) and orientation (deg) error overlaid.

Parameters:

| Name       | Type                                          | Description                                           | Default |
| ---------- | --------------------------------------------- | ----------------------------------------------------- | ------- |
| `poses`    | `Optional[List[Union[List[float], ndarray]]]` | Joint configurations. If None, uses 24 default poses. | `None`  |
| `server`   |                                               | Existing viser server, or None to create one.         | `None`  |
| `port`     | `int`                                         | Port for new server.                                  | `8080`  |
| `duration` | `float`                                       | Seconds to keep the display open.                     | `15.0`  |
| `spacing`  | `float`                                       | Distance between grid cells in meters.                | `2.0`   |

Source code in `src/figaroh/tools/export_validation.py`

```
def show_static_grid(
    self,
    poses: Optional[List[Union[List[float], np.ndarray]]] = None,
    server=None,
    port: int = 8080,
    duration: float = 15.0,
    spacing: float = 2.0,
):
    """Display a grid of static configurations with error labels.

    Each cell shows both models at the same configuration, with the
    position (mm) and orientation (deg) error overlaid.

    Args:
        poses: Joint configurations. If None, uses 24 default poses.
        server: Existing viser server, or None to create one.
        port: Port for new server.
        duration: Seconds to keep the display open.
        spacing: Distance between grid cells in meters.
    """
    from viser.extras import ViserUrdf

    viz = self._get_viser_server(server, port)
    urdf_orig, urdf_mod = self._load_yourdfpy()

    if poses is None:
        poses = self._default_poses()

    for i, q in enumerate(poses):
        row, col = divmod(i, 4)
        x_off = col * spacing
        z_off = row * spacing
        q_arr = np.asarray(q, dtype=float)

        vis_orig = ViserUrdf(
            viz,
            urdf_or_path=urdf_orig,
            root_node_name=f"/grid/{i}/orig",
            load_meshes=True,
        )
        vis_mod = ViserUrdf(
            viz,
            urdf_or_path=urdf_mod,
            root_node_name=f"/grid/{i}/mod",
            load_meshes=True,
        )
        cfg_dict = self._pinocchio_config_to_dict(q_arr)
        vis_orig.update_cfg(cfg_dict)
        vis_mod.update_cfg(cfg_dict)

        delta = self._compute_delta(q_arr)
        pos_err = delta.position_error() * 1000  # mm
        orient_err = delta.orientation_error() * 180 / np.pi  # deg

        viz.scene.add_label(
            f"/grid/{i}/label",
            f"{pos_err:.1f}mm / {orient_err:.1f}°",
            position=(x_off, z_off + 1.2, 0),
        )
    print(
        f"[vis] {len(poses)} configs in grid at http://localhost:{viz.get_port()}"
    )
    time.sleep(duration)
```

### `show_interactive_validation(n_trajectory=50, seed=42, port=8080)`

Interactive combined visualization with trajectory, static comparison, error plots, replay, and opacity controls.

Opens a single viser server containing:

- **Trajectory animation** — 50 random configurations with path traces for both nominal and modified models. Replay button allows re-running.
- **Static pose comparison** — overlays both models at the same configuration with opacity slider for the modified model.
- **Error plots** — uplot charts for trajectory and static pose errors (position in mm, orientation in degrees).
- **Pose selector** — slider to cycle through static configurations.

Parameters:

| Name           | Type  | Description                                 | Default |
| -------------- | ----- | ------------------------------------------- | ------- |
| `n_trajectory` | `int` | Number of random configurations to animate. | `50`    |
| `seed`         | `int` | Random seed for reproducibility.            | `42`    |
| `port`         | `int` | Port for the viser server.                  | `8080`  |

Source code in `src/figaroh/tools/export_validation.py`

```
def show_interactive_validation(
    self,
    n_trajectory: int = 50,
    seed: int = 42,
    port: int = 8080,
):
    """Interactive combined visualization with trajectory, static comparison,
    error plots, replay, and opacity controls.

    Opens a single viser server containing:

    * **Trajectory animation** — 50 random configurations with path traces
      for both nominal and modified models. Replay button allows re-running.
    * **Static pose comparison** — overlays both models at the same
      configuration with opacity slider for the modified model.
    * **Error plots** — uplot charts for trajectory and static pose errors
      (position in mm, orientation in degrees).
    * **Pose selector** — slider to cycle through static configurations.

    Args:
        n_trajectory: Number of random configurations to animate.
        seed: Random seed for reproducibility.
        port: Port for the viser server.
    """
    import threading

    from viser import uplot
    from viser.extras import ViserUrdf

    server = self._get_viser_server(None, port)
    urdf_orig, urdf_mod = self._load_yourdfpy()
    traj_configs = self._sample_configs(n_trajectory, seed)
    static_configs = self._default_poses()

    # --- 3D Scene: trajectory robots ---
    # Parent frames to offset trajectory models from static comparison
    server.scene.add_frame("/trajectory", show_axes=False)
    traj_vis_orig = ViserUrdf(
        server,
        urdf_or_path=urdf_orig,
        root_node_name="/trajectory/orig",
        load_meshes=True,
        mesh_color_override=(51, 102, 229),
    )
    traj_vis_mod = ViserUrdf(
        server,
        urdf_or_path=urdf_mod,
        root_node_name="/trajectory/mod",
        load_meshes=True,
        mesh_color_override=(229, 51, 51),
    )

    # --- 3D Scene: static comparison robots (offset to the right) ---
    server.scene.add_frame(
        "/static",
        show_axes=False,
        position=(2.5, 0, 0),
    )
    static_vis_nom = ViserUrdf(
        server,
        urdf_or_path=urdf_orig,
        root_node_name="/static/nominal",
        load_meshes=True,
        mesh_color_override=(51, 102, 229),
    )
    static_vis_mod = ViserUrdf(
        server,
        urdf_or_path=urdf_mod,
        root_node_name="/static/modified",
        load_meshes=True,
        mesh_color_override=(229, 51, 51, 0.5),
    )
    # Labels for static section
    server.scene.add_label(
        "/static/label_nom",
        "Nominal",
        position=(0, 0, 1.8),
    )
    server.scene.add_label(
        "/static/label_mod",
        "Modified",
        position=(0, 0, 2.0),
    )

    # --- Track static error state ---
    static_state = {
        "idx": 0,
        "opacity": 0.5,
        "configs": static_configs,
        "errors": [self._compute_delta(q) for q in static_configs],
    }

    # --- Compute trajectory errors upfront ---
    traj_errors = [self._compute_delta(q) for q in traj_configs]
    traj_pos_mm = np.array([e.position_error() * 1000 for e in traj_errors])
    traj_orient_deg = np.array(
        [e.orientation_error() * 180 / np.pi for e in traj_errors]
    )
    static_pos_mm = np.array(
        [e.position_error() * 1000 for e in static_state["errors"]]
    )
    static_orient_deg = np.array(
        [e.orientation_error() * 180 / np.pi for e in static_state["errors"]]
    )

    # --- Animation state ---
    anim_state: dict = {
        "traj_configs": traj_configs,
        "running": False,
        "speed": 0.05,
        "orig_trace": [],
        "mod_trace": [],
    }

    def _clear_traces():
        try:
            server.scene.remove_by_name("/traces/orig_line")
        except Exception:
            pass
        try:
            server.scene.remove_by_name("/traces/mod_line")
        except Exception:
            pass

    def _draw_traces():
        """Draw EE path traces as line segments."""
        orig_pts = np.array(anim_state["orig_trace"])
        mod_pts = np.array(anim_state["mod_trace"])
        if len(orig_pts) < 2:
            return

        # Reshape to (N-1, 2, 3) for line segments
        def _to_segments(pts: np.ndarray) -> np.ndarray:
            return np.stack([pts[:-1], pts[1:]], axis=1)

        # Nominal trace (blue)
        server.scene.add_line_segments(
            "/traces/orig_line",
            _to_segments(orig_pts),
            colors=(51, 102, 229),
            line_width=2,
        )
        # Modified trace (red)
        server.scene.add_line_segments(
            "/traces/mod_line",
            _to_segments(mod_pts),
            colors=(229, 51, 51),
            line_width=2,
        )

    def _run_animation():
        if anim_state["running"]:
            return
        anim_state["running"] = True
        _clear_traces()
        anim_state["orig_trace"].clear()
        anim_state["mod_trace"].clear()

        for i, q in enumerate(anim_state["traj_configs"]):
            cfg_dict = self._pinocchio_config_to_dict(q)
            traj_vis_orig.update_cfg(cfg_dict)
            traj_vis_mod.update_cfg(cfg_dict)

            pin.forwardKinematics(self.model_a, self.data_a, q)
            pin.updateFramePlacements(self.model_a, self.data_a)
            pin.forwardKinematics(self.model_b, self.data_b, q)
            pin.updateFramePlacements(self.model_b, self.data_b)

            anim_state["orig_trace"].append(
                self.data_a.oMf[self.ee_id_a].translation.copy()
            )
            anim_state["mod_trace"].append(
                self.data_b.oMf[self.ee_id_b].translation.copy()
            )
            time.sleep(anim_state["speed"])

        _draw_traces()
        anim_state["running"] = False
        replay_btn.disabled = False

    def _update_static_pose():
        """Apply the current static pose to both overlaid models."""
        idx = static_state["idx"]
        q = static_state["configs"][idx]
        cfg_dict = self._pinocchio_config_to_dict(q)
        static_vis_nom.update_cfg(cfg_dict)
        static_vis_mod.update_cfg(cfg_dict)
        error = static_state["errors"][idx]
        pos_mm = error.position_error() * 1000
        orient_deg = error.orientation_error() * 180 / np.pi
        pose_label.content = f"Pose {idx}: {pos_mm:.1f}mm / {orient_deg:.1f}°"

    # --- GUI Controls ---
    with server.gui.add_folder("Controls"):
        # Trajectory section
        server.gui.add_markdown("**Trajectory**")
        replay_btn = server.gui.add_button(
            "Replay Trajectory", icon="player-play-filled"
        )
        speed_slider = server.gui.add_slider(
            "Animation Speed",
            0.01,
            0.2,
            0.01,
            0.05,
            hint="Delay between configurations (seconds)",
        )

        # Static comparison section
        server.gui.add_markdown("**Static Comparison**")
        pose_slider = server.gui.add_slider(
            "Pose Index",
            0,
            len(static_configs) - 1,
            1,
            0,
            hint="Select which static configuration to display",
        )
        opacity_slider = server.gui.add_slider(
            "Modified Model Opacity",
            0.0,
            1.0,
            0.01,
            0.5,
            hint="Transparency of the modified model overlay",
        )
        pose_label = server.gui.add_markdown(
            "Pose 0: ...",
        )

    # --- Error Plots ---
    traj_x = np.arange(len(traj_pos_mm))
    static_x = np.arange(len(static_pos_mm))

    with server.gui.add_folder("Error Plots"):
        server.gui.add_markdown("**Trajectory FK Errors**")
        server.gui.add_uplot(
            data=(traj_x, traj_pos_mm, traj_orient_deg),
            series=(
                uplot.Series(),
                uplot.Series(
                    label="Position Error (mm)",
                    color=(51, 102, 229),
                ),
                uplot.Series(
                    label="Orientation Error (deg)",
                    color=(229, 51, 51),
                ),
            ),
            title="Trajectory",
            aspect=2.0,
        )

        server.gui.add_markdown("**Static Pose FK Errors**")
        server.gui.add_uplot(
            data=(static_x, static_pos_mm, static_orient_deg),
            series=(
                uplot.Series(),
                uplot.Series(
                    label="Position Error (mm)",
                    color=(51, 102, 229),
                ),
                uplot.Series(
                    label="Orientation Error (deg)",
                    color=(229, 51, 51),
                ),
            ),
            title="Static Poses",
            aspect=2.0,
        )

    # --- Callbacks ---
    @replay_btn.on_click
    def _on_replay(_event):
        replay_btn.disabled = True
        thread = threading.Thread(target=_run_animation, daemon=True)
        thread.start()

    @speed_slider.on_update
    def _on_speed_change(_event):
        anim_state["speed"] = _event.target.value

    @pose_slider.on_update
    def _on_pose_change(_event):
        static_state["idx"] = int(_event.target.value)
        _update_static_pose()

    @opacity_slider.on_update
    def _on_opacity_change(_event):
        static_state["opacity"] = _event.target.value
        for mesh in static_vis_mod._meshes:
            mesh.opacity = _event.target.value

    # --- Initial state ---
    _update_static_pose()

    print("\n" + "=" * 60)
    print(f"Interactive validation at http://localhost:{server.get_port()}")
    print("  Trajectory:  blue (nominal), red (modified)")
    print("  Static:      overlaid models (offset right)")
    print("  Controls:    Replay | Speed | Pose | Opacity")
    print("  Error plots: scroll GUI panel")
    print("=" * 60)

    # --- Run initial animation ---
    replay_btn.disabled = True
    _run_animation()

    # Block until user stops the server
    try:
        server.sleep_forever()
    except KeyboardInterrupt:
        pass
```
