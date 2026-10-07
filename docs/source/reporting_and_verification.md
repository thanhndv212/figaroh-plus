# Reporting & Verification (V&V)

Every `BaseCalibration`/`BaseIdentification` run produces the same
underlying metrics — condition number, per-DOF/per-joint residuals,
parameter uncertainty, held-out validation. This guide covers the four
ways those metrics get surfaced, and when to reach for each one.

| Artifact | Call | Audience |
|---|---|---|
| Terminal quality report | automatic, after `solve()` | you, right now, in the terminal |
| Self-contained HTML report | `solve(html_report=True)` / `export_html_report()` | you, or anyone you send the file to |
| Machine-readable verdict (JSON) | `verify()` / `export_verification_report()` | CI, scripts, anything that needs a pass/fail |
| Static two-run compare page | `figaroh.tools.compare_report.generate_compare_page()` | comparing two exported verdicts, offline |

All four read from data `solve()` already computed — none of them re-run
the calibration/identification, and none of them require network access or
a backend.

## Interpreting a verdict

Use [Plan, Fit and Validate](example_workflow.md) to establish input correctness,
parameter scope and independent validation before interpreting these reports.
Inspect computed and skipped checks: a passing verdict with unavailable
validation metrics does not establish held-out accuracy. Keep solver termination,
physical feasibility, prediction quality and export/reload parity as separate
conclusions. The compare page's compatibility checks do not establish matching
raw inputs, processing, objectives or absence of leakage; retain that provenance.

### Stages, splits and revisions

Each run records what each step did (`figaroh.tools.stages`, #55/#63). The
terminal report prints them on a `Stages:` line, and the HTML report shows a
**Stages and data** section.

Each step gets its own verdict in `verdict.stages`:
- the steps are `data`, `fit`, `validation`, `physical` and `export`;
- each is `pass`, `fail`, `fallback` or `not_run`, or `not_evaluated` when
  the run recorded nothing for it;
- the existing entries (`numerical_execution`, `prediction`, `solver`) are
  kept beside them.

A `validation` **fallback** means the metrics were computed on the
training data. It is never held-out evidence, and prediction acceptance
cannot pass on it.

| Field | Meaning |
|---|---|
| `verdict.stage_records` | every step's status, reason and metrics with units |
| `verdict.selected_stage` | the step whose parameters the result reports (`fit`; `none` if the fit failed) |
| `verdict.splits` | which files, sessions and sample counts trained and validated the run, and whether validation was held out |
| provenance `software.git_commit` | the working directory's commit (an example run: the examples repository) |
| provenance `software.figaroh_revision` | the figaroh checkout's own commit and dirty state |

Together, the two commits name the paired core/examples revisions. The
exported JSON carries the records under `stage_records` and a
`schema_version`. The run archive writes `stages.json` and adds each run's
stage verdicts to `index.jsonl`.

## Terminal quality reports

`print_quality_report()` runs automatically at the end of `solve()`; call
it again any time after a successful solve to reprint it:

```python
calibrator.solve()
calibrator.print_quality_report()   # same output, printed again
```

For calibration this prints convergence status, outlier count, condition
number, per-DOF residual statistics (mean/std/RMSE/max/R²), held-out
validation (if `validation_data_file` is configured), and any parameter
pairs with `|ρ| > 0.8`. For identification it prints base-parameter count,
condition number, RMSE, correlation, per-joint torque residuals, the top-5
worst-identified base parameters, and (if enabled) physical-consistency /
reconstruction status.

## Self-contained HTML reports

```python
calibrator.solve(html_report=True)
# or, after the fact:
calibrator.export_html_report(output_path="results/calibration_report.html")
```

The identification side is the same shape:

```python
identifier.solve(html_report=True)
identifier.export_html_report(output_path="results/identification_report.html")
```

Each report is a single self-contained HTML file — no external requests,
no CDN dependency, inline CSS/JS only, light/dark theme aware — so it opens
correctly straight from disk and is safe to attach to an email or a ticket.
It contains:

- **Summary** — the same headline numbers as the terminal report.
- **Insights** — auto-flagged issues (ill-conditioning, poorly-identified
  parameters, weak validation improvement, no validation data configured).
- **Per-DOF / per-joint residuals** and **parameter uncertainty** tables,
  each parameter's relative uncertainty rendered as a confidence-tier bar.
- **Validation** — nominal vs. fitted vs. measured, when a genuinely
  separate held-out dataset was configured.
- **Before / after** — an interactive overlay chart of the same
  nominal/fitted/measured series, with wheel-to-zoom and hover-to-inspect,
  so you don't have to read the numbers out of a table to see the fit
  improve. This needs no extra call — it's populated automatically whenever
  validation data is available.

## Machine-readable verification

`verify()` checks the run's metrics against a set of pass/fail thresholds
and returns a `VerificationVerdict` — this is the piece that turns a report
(for a human to read) into something a script can branch on:

```python
verdict = identifier.verify(scope="execution")
print(verdict.passed)          # bool
for check in verdict.checks:
    print(check.name, check.value, check.comparison, check.threshold, check.passed)
```

A bare library `verify()` defaults to **prediction acceptance**, which stays
incomplete without a validation set and explicit limits. Routine example CLIs
explicitly request **numerical execution**:

```python
verdict = identifier.verify(scope="execution")
print(verdict.scope, verdict.status, verdict.stages)
```

This checks finite, nonempty fitted parameters, predictions, measurements and
RMSE, matching effort dimensions, and whether explicitly requested validation
loaded. Calibration additionally checks its reported optimization success and
finite parameter/residual outputs. It does not certify data provenance, a full
physical model or export/reload parity. Those stages remain `not_evaluated`.

There are **no universal improvement, correlation, raw condition-number or
calibration-error gates**. These remain report diagnostics. A good nominal model
may have little improvement; high correlation can coexist with biased torque;
raw conditioning changes with parameter units and basis scaling.

For prediction acceptance, provide independent validation data and explicit
application limits. For identification, supply an absolute maximum RMSE in Nm
for **every active validation joint**:

```python
profile = {
    f"validation_rmse:{joint}": {
        "threshold": allowed_error_nm[joint],
        "comparison": "max",
        "rationale": "measurement uncertainty and the application requirement",
    }
    for joint in active_joints
}
verdict = identifier.verify(scope="prediction", thresholds=profile)
```

Identification also exposes `validation_abs_bias:<joint>` and
`validation_peak_error:<joint>` in Nm (or `training_…` for fallback). Add explicit
limits where the application needs them; the per-joint RMSE limits remain required.

For calibration, use `position_rmse_mm` and, when rotational DOFs are measured,
`orientation_rmse_deg`, with application-specific maximum limits. No replacement
universal error value is supplied. Record units, the requirement behind each
limit and the validation split before evaluating acceptance. Loading a separate
file is necessary but does not prove experimental independence: split integrity,
coverage and acquisition provenance remain the experiment owner's responsibility.

`passed` remains available for existing consumers and is true only if all
required checks **in the named scope** pass. Additive `status`, `scope`, `stages`
and applied `policy` fields make the meaning explicit:

- `pass`: all required checks in scope succeeded.
- `fail`: a required evaluated check failed, including computed NaN/Inf evidence.
- `not_evaluated`: required evidence or a prediction profile is unavailable.

Missing checks are retained with their reason, rather than silently skipped.
An empty threshold-only policy is `not_evaluated`, never PASS. A spec may set
`required=False` for an advisory check; it is still reported. A failed explicitly
requested validation load prevents execution acceptance. Training fallback can
support diagnostics but cannot pass prediction acceptance. Physical and export
acceptance require their own evidence; a scoped prediction PASS is not a full
physical-model approval. Compare historical verdicts with care: their unscoped
boolean used the earlier universal gates, not this policy.

Write the verdict to disk as JSON (numpy-safe, includes git commit / config
hash / timestamp provenance):

```python
path = identifier.export_verification_report(
    output_path=None,       # defaults to {output_dir}/identification_verification.json
    output_dir="results",
    thresholds=None,        # explicit limits, when evaluating prediction
    scope="execution",
)
```

The same call exists on `BaseCalibration` (writes
`{output_dir}/calibration_verification.json`).

### Wiring `--verify` into a CLI script

This is the pattern already used by the `identification.py` entry points in
[figaroh-examples](https://github.com/thanhndv212/figaroh-examples):

```python
verdict = identifier.verify(scope="execution")
identifier.export_verification_report(output_path=str(run_dir / "verdict.json"), scope="execution")

for check in verdict.checks:
    status = check.status.upper()
    print(f"  [{status}] {check.name}: {check.value} ({check.comparison} {check.threshold:.4g})")

if not verdict.passed:
    sys.exit(1)
```

Run it as `python identification.py --verify`. The scope must be printed with
the verdict: zero means numerical execution passed by default, not that an
independent prediction or physical model was accepted. Prediction-scoped
incomplete evidence produces a nonzero exit.

## Comparing two runs

`figaroh.tools.compare_report.generate_compare_page()` renders a static,
self-contained HTML page that loads **two** `export_verification_report()`
JSON files client-side (drag-and-drop or a file picker) and, once they
pass a compatibility check, shows a per-metric diff table and an overlaid
before/after series chart with a per-run visibility toggle:

```python
from figaroh.tools.compare_report import generate_compare_page

generate_compare_page(output_path="results/compare.html")
```

Open `results/compare.html` in a browser and load two verdict JSON files —
everything after that runs entirely client-side; there's no server, no
network request, and nothing is uploaded anywhere.

**The compatibility check runs before anything renders.** Two verdicts are
compared only if they agree on:

- domain (calibration vs. identification — inferred from `compat.dof_names`
  vs. `compat.active_joints`)
- the same DOF/joint names
- the same `decimate` setting (identification only)
- comparable sample counts (within ~20%)

On a mismatch the page blocks the comparison and explains why (e.g.
"Different decimate setting: false vs. true") rather than silently
overlaying two runs that aren't actually comparable — you can still force
the comparison via an explicit "compare anyway" checkbox, with a persistent
warning banner while forced.

This is deliberately a static, no-backend artifact, not a run library or
dashboard — see the design rationale in
[`docs/decisions/external-tool-comparisons.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/external-tool-comparisons.md)
Part C if you're curious why (short version: no validated need yet for a
run history/backend, and comparing incompatible runs silently was judged
actively dangerous).

## CI integration example

A minimal build-gate recipe combining the pieces above:

```python
identifier.solve(html_report=True)
verdict = identifier.verify()
identifier.export_verification_report(output_dir="results")

if not verdict.passed:
    failed = [c.name for c in verdict.checks if not c.passed]
    print(f"Verification failed: {', '.join(failed)}")
    sys.exit(1)
```

Point your CI at `results/*_verification.json` as a build artifact, and at
`results/*_report.html` if you want a human-readable report attached to
the same run.
