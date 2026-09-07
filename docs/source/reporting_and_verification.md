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
verdict = identifier.verify()
print(verdict.passed)          # bool
for check in verdict.checks:
    print(check.name, check.value, check.comparison, check.threshold, check.passed)
```

Default thresholds (every one is overridable per call — see below):

**Calibration:**

| Metric | Default | Comparison |
|---|---|---|
| `position_rmse_mm` (validation set) | 2.0 | max |
| `orientation_rmse_deg` (validation set) | 0.1 | max |
| `condition_number` | 1000.0 | max |

**Identification:**

| Metric | Default | Comparison |
|---|---|---|
| `validation_correlation` | 0.9 | min |
| `condition_number` | 1000.0 | max |
| `validation_improvement_pct` | 50.0 | min |

A threshold whose metric can't be computed (typically because no
`validation_data_file` was configured) is **skipped, not failed** — an
empty check list counts as passed, since there's nothing to fail on.

Override thresholds per call:

```python
verdict = calibrator.verify(thresholds={
    "condition_number": {"threshold": 500.0, "comparison": "max"},
})
```

Write the verdict to disk as JSON (numpy-safe, includes git commit / config
hash / timestamp provenance):

```python
path = identifier.export_verification_report(
    output_path=None,       # defaults to {output_dir}/identification_verification.json
    output_dir="results",
    thresholds=None,        # defaults, same as verify()
)
```

The same call exists on `BaseCalibration` (writes
`{output_dir}/calibration_verification.json`).

### Wiring `--verify` into a CLI script

This is the pattern already used by the `identification.py` entry points in
[figaroh-examples](https://github.com/thanhndv212/figaroh-examples):

```python
verdict = identifier.verify()
identifier.export_verification_report(output_path=str(run_dir / "verdict.json"))

for check in verdict.checks:
    status = "PASS" if check.passed else "FAIL"
    print(f"  [{status}] {check.name}: {check.value:.4g} ({check.comparison} {check.threshold:.4g})")

if not verdict.passed:
    sys.exit(1)
```

Run it as `python identification.py --verify`, and the script's exit code
is a real CI gate — non-zero on a failing run, zero on a passing one.

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
