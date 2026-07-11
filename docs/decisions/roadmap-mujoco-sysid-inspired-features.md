# Roadmap: FIGAROH Quality & Reporting Infrastructure

## Date
2026-07-10 (originated) / 2026-07-11 (consolidated into one document, reordered to match
the decided build sequence, Step 1 implemented, Step 2 implemented)

## Status
In Progress — Feature 1, Feature 1b, Step 1 (Feature 5 Phases 1–2), and Step 2 (Feature 5
Phase 4) implemented and tested. Everything else (Steps 3–7, Features 2–4) is unimplemented,
and this document's section order now **is** the build order: read top to bottom from "Step
3" onward to get the plan in the sequence it should actually be executed. "Later" (Features
2, 3, 4) comes last, not because those matter less, but because Steps 1–7 build the
verification/reporting tooling that makes correctness-sensitive modeling work (2–4)
checkable as it's built.

## Context

[`sysid-comparison-mujoco-vs-figaroh.md`](sysid-comparison-mujoco-vs-figaroh.md)
compared FIGAROH's regressor-based identification/calibration against MuJoCo's
black-box rollout `sysid` module. Four ideas from that comparison (Features 1–4) were
worth adopting, independently of each other. Verifying Feature 1b against a real UR10
example then surfaced a real bug and several gaps in FIGAROH's reporting/plotting stack
that were bigger than Feature 1b's own scope — those became Feature 5 (reporting
infrastructure consolidation) and, after a design discussion and self-critique, Feature 6
(a deliberately cut-down interactive V&V MVP).

This single document tracks all six. It went through two prior structural changes, both on
request, both preserved in spirit here:

1. Originally split across four files (this roadmap plus two standalone implementation
   plans plus a doc tying them together) — merged into one file, since four cross-linked
   documents were harder to track than one longer one.
2. Originally organized by Feature number (1, 1b, 2, 3, 4, 5, 6) regardless of build order
   — **reordered** (this revision) so the document's physical section order matches the
   decided implementation sequence instead of requiring a separate lookup table. Feature
   5's and Feature 6's phases are interleaved in the real build order (Feature 5 Phase 4
   before Feature 6, but Feature 5 Phases 3 and 5 after Feature 6), so their content is
   split across Steps below rather than kept as two monolithic Feature sections — each
   Step says which Feature/Phase it corresponds to, so `F5.x`/`P#.x`/`A.x`-style task IDs
   from earlier revisions still resolve to the same content, just relocated.

No task content, decision, risk note, or open question was dropped in either restructuring
— only section order and connective tissue changed.

| Build order | Feature / Phase | Effort | Status |
|---|---|---|---|
| done | Feature 1 — HTML diagnostic report (calibration) | Small | ✅ Implemented |
| done | Feature 1b — Terminal + HTML quality report for `BaseIdentification` | Small–Medium | ✅ Implemented |
| done | Step 1 — Feature 5, Phases 1+2 — fix reshape bug, dedup fallback plotting | Small | ✅ Implemented |
| done | Step 2 — Feature 5, Phase 4 — machine-readable verification verdict | Medium | ✅ Implemented |
| Step 3 | Feature 6, Phase A (redefined) — extend Step 2's export with `series`/`compat` | Small | Proposed |
| Step 4 | Feature 6, Phase B — before/after interactive panel | Small | Proposed |
| Step 5 | Feature 6, Phase C — static two-run compare page | Small | Proposed |
| Step 6 | Feature 5, Phase 3 — optimal-\* reports | Medium | Proposed |
| Step 7 | Feature 5, Phase 5 — unified report schema (spike only) | Small (spike) | Proposed |
| Later | Feature 2 — generic `Parameter` + modifier abstraction | Medium | Not started |
| Later | Feature 3 — log-Cholesky pseudo-inertia parameterization | Medium | Not started |
| Later | Feature 4 — black-box rollout refinement stage | Large | Not started |

Why this order (condensed — each Step below repeats the reasoning local to it): Steps 1–2
are small, zero-dependency bug/dedup fixes plus the one piece (a machine-checkable verdict)
that changes what the reporting stack fundamentally *is*. Steps 3–5 build directly on
Step 2's export rather than inventing a second one. Steps 6–7 are real but lower-urgency
coverage/consistency work. Features 2–4 are new, correctness-sensitive modeling work that
benefits from Steps 1–7's verification tooling existing first.

---

## Done

### Feature 1: HTML diagnostic report

**Inspired by:** MuJoCo's `report/builder.py` + `report/sections/*` — an
interactive HTML report with a parameter-distribution plot (value vs.
nominal vs. bounds, colored by identifiability), a correlation heatmap, and
an auto-generated "insights" section flagging parameters stuck at bounds or
poorly identified.

**Why FIGAROH needs it:** `print_quality_report()` already computes
condition number, per-DOF residuals, parameter uncertainty, and correlated
pairs — but only prints them as terminal text. That's hard to skim, hard to
share, and has no visual encoding of "which parameters should I distrust."

**Scope:** Calibration only, reusing `evaluation_metrics` /
`results_data` already computed by `BaseCalibration`. No new dependency
(no jinja2/plotly) — self-contained HTML/CSS string template, matching the
zero-extra-dependency footprint of the existing terminal report. Light/dark
theme aware. Identification support is a possible follow-up, not in scope
here (its diagnostics live in a different class shape:
`base_identification.py`'s `self.result`/`std_relative`/`correlation`).

#### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| F1.1 | `tools/report.py` module | `generate_calibration_report(calibrator, output_path=None, title=None) -> str` builds a self-contained HTML string from `calibrator.evaluation_metrics`, `calibrator.calib_config`, `calibrator.results_data` | ✅ |
| F1.2 | Summary section | Convergence, iterations, cost, outliers, condition number — same fields as the terminal report's header | ✅ |
| F1.3 | Insights section | Auto-flag: ill-conditioned fit, outlier rate, parameters with relative uncertainty above a threshold, no validation data provided | ✅ |
| F1.4 | Per-DOF residual table | Mirrors `_compute_per_dof_stats` output (mean/std/RMSE/max/R² per DOF) | ✅ |
| F1.5 | Parameter uncertainty section | Table + inline CSS bar per parameter, sorted by relative std-dev (highest first), analogous to MuJoCo's confidence-interval plot | ✅ |
| F1.6 | Correlation section | List of `|ρ| > 0.8` pairs with a magnitude-colored badge; "none" message otherwise | ✅ |
| F1.7 | Validation section | Nominal vs. calibrated vs. improvement %, when validation data is available; graceful "not provided" message otherwise | ✅ |
| F1.8 | `BaseCalibration.export_html_report()` | Opt-in method, plus an `html_report=False` flag on `solve()` mirroring the existing `plotting`/`save_results` flags | ✅ |
| F1.9 | Unit tests | `tests/unit/test_report.py` — synthetic `evaluation_metrics` dict, no full calibration fixture required | ✅ |

#### Verification

- [x] `generate_calibration_report()` produces valid, self-contained HTML (no external requests, inline CSS only)
- [x] Renders correctly with and without validation data
- [x] Renders correctly with zero correlated pairs
- [x] `calibrator.solve(html_report=True)` writes a file without raising
- [x] Unit tests pass (19/19, `tests/unit/test_report.py`)

**Files touched:**
`src/figaroh/tools/report.py` (new),
`src/figaroh/tools/__init__.py`,
`src/figaroh/calibration/base_calibration.py`
(`solve(html_report=...)`, `export_html_report()`),
`tests/unit/test_report.py` (new).

---

### Feature 1b: Terminal + HTML quality report for `BaseIdentification`

**Status:** ✅ Implemented (2026-07-10), including held-out validation
support. Planned earlier the same day after confirming (by grep + reading
both base classes) that `print_quality_report()` and `export_html_report()`
existed **only** on `BaseCalibration`. `BaseIdentification` had no report
method at all: its `solve()` only called `plot_results()` /
`save_results()`, and diagnostics lived in a flat `self.result` dict plus
`self.std_relative` / `self.rms_error` / `self.correlation`.

**Validation design decision (resolved, not omitted):** the user directed
that identification validation be treated *exactly like calibration* — a
genuinely separate held-out dataset (`identif_config["validation_data_file"]`),
never a runtime split of the training data. This closes the "open question"
below as: **implement it**, not omit it.

**Why not a drop-in reuse of Feature 1:** calibration and identification
produce structurally different diagnostics, so `generate_calibration_report`
cannot be pointed at an identification instance as-is:

| | Calibration (`evaluation_metrics`) | Identification (`self.result` / attrs) |
|---|---|---|
| Solve type | Iterative nonlinear LM with outlier removal | One-shot linear QR / least-squares |
| Convergence | `optimization_success`, `n_iterations`, `n_outliers` | None — no iteration, no outlier step |
| Residual axis | Per-DOF Cartesian pose (mm / deg) | Per-joint torque (Nm), currently only pooled, not sliced per joint |
| Parameter uncertainty | `param_stdev` / `param_stddev_percentage` per calibration param | `std_relative` per **base** parameter (already relative %, different scale) |
| Held-out validation | Separate CSV → `validation_metrics` | Not implemented for identification at all |
| Extras | Parameter correlation pairs (`correlated_pairs`) | `physical consistency` / `reconstruction` sub-dicts (opt-in, SDP-based) |

Given that, the right shape is a **parallel adapter + parallel report
module that shares only the low-level HTML/CSS primitives** with Feature 1
— not a forced common `evaluation_metrics` schema. Trying to unify the
schemas now would mean bending identification's linear-solve diagnostics
into calibration's iterative-solve vocabulary (e.g. inventing a fake
"iterations" or "outliers" field that doesn't mean anything for a QR
solve), which is worse than two adapters sharing a style sheet.

#### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| F1b.1 | Extract shared primitives | `tools/_report_common.py`: `_esc`, `_uncertainty_tier`, `UNCERTAINTY_*`/`VALIDATION_IMPROVEMENT_WARN_PCT`, `_insights_section`, `_param_uncertainty_section`, `_correlation_section`, `_STYLE`. `tools/report.py` re-exports them (calibration's own `_summary_section`/`_per_dof_section`/`_validation_section`/`_build_insights` stay put — they're calibration-shaped). All 19 pre-existing calibration report tests still pass unchanged | ✅ |
| F1b.2 | Validation support in `BaseIdentification` | `validation_data_file` config key (unified `tasks.identification.data.validation_data_file` + legacy top-level key, both threaded through `identification/config.py`). `_load_validation_data()` reuses the exact per-robot `load_trajectory_data`/`process_kinematics_data`/`process_torque_data` pipeline (save/restore training state) so robot-specific overrides apply identically. `_compute_validation_metrics()` predicts nominal (standard/CAD params) vs. identified (`phi_base`) torque on the held-out set and compares against measured — direct analogue of calibration's nominal-vs-calibrated FK check | ✅ |
| F1b.2a | Base-column exposure in `QRDecomposer` | Evaluating `phi_base` against a *new* regressor requires knowing which columns of a reduced regressor are the base set. `double_decomposition()` now stores `self.base_indices` (+ `get_base_indices()`); `base_indices`/`regroup_indices` are a structural property of the robot's kinematic chain, not the training trajectory, so the same selection is valid for held-out data of the same robot | ✅ |
| F1b.3 | Per-joint torque residual breakdown | `_compute_per_joint_stats()`. Originally gated on `solve(decimate=True)` only; the gate was removed once F1b.10 fixed `_prepare_undecimated_data()` to flatten joint-major, so both paths now produce rows in the same known order. See "Findings" below | ✅ |
| F1b.4 | `BaseIdentification.print_quality_report()` | Terminal report: base-parameter count/samples, condition number + label, RMSE, correlation, per-joint residual table (or "unavailable" note), top-5 worst base-parameter uncertainties, validation section, physical-consistency/reconstruction status lines when those optional steps ran | ✅ |
| F1b.5 | `tools/identification_report.py` — `generate_identification_report()` | HTML counterpart on the F1b.1 shared primitives: Summary, Insights (ill-conditioned fit, poorly-identified base parameters, low validation correlation, weak validation improvement, physical-consistency/reconstruction problems), per-joint residual table, validation table, base-parameter uncertainty table + bars, consistency/reconstruction status card | ✅ |
| F1b.6 | `BaseIdentification.export_html_report()` + `solve(html_report=False)` | Same opt-in pattern as F1.8 | ✅ |
| F1b.7 | Unit tests | `tests/unit/test_identification_report.py` — 25 tests, `FakeIdentifier` fixture; explicitly covers numpy-array `std_relative`/`phi_base` (a real `truth value of an array is ambiguous` trap), missing validation/per-joint/consistency data, and NaN condition number | ✅ |
| F1b.8 | Verify against a real example | Ran `figaroh-examples/examples/ur10/identification.py`'s `UR10Identification` directly (both `decimate=True` and `decimate=False`) with a genuinely separate validation dataset and `html_report=True`. Report renders correctly in both cases; per-joint section now renders identically under both paths (see F1b.10) | ✅ |
| F1b.9 | Example subclasses: `data_source` support | `load_trajectory_data(self, data_source=None)` signature added to the abstract method and all three example subclasses (UR10, TIAGo, Staubli TX40) — `data_source` is a directory override; each subclass reads its usual filenames from that directory instead of its default when given | ✅ |
| F1b.10 | Fix `decimate=False` row-order bug | `_prepare_undecimated_data()` did `self.processed_data["torques"].flatten()` on a `(N, n_active)` array — sample-major — while regressor rows (and `_apply_decimation()`'s output) are joint-major. Fixed by transposing before flattening (`tau_data.T.flatten()`). This was a real bug, not a hypothetical one — see "Findings" | ✅ |
| F1b.11 | Regenerate UR10 example identification data | Bundled `examples/ur10/data/identification_{q,tau}_simulation.csv` had torque values ~2 orders of magnitude above physically-realistic RNEA output. Regenerated both training (`data/`) and validation (`data/validation/`) sets from a persistent-excitation (multi-harmonic Fourier + dither) trajectory, computing torque via `pin.rnea()` on **exactly** the same finite-difference `q→dq→ddq` pipeline (`calculate_first_second_order_differentiation`, called the same way `load_trajectory_data` calls it) that the identification pipeline itself uses — guaranteeing self-consistency, not just physical plausibility in isolation. Different random seed/harmonics for validation vs. training (genuinely separate dataset). See "Findings" | ✅ |

**Findings from F1b.8–F1b.11 (worth keeping — informed decisions above, not
just implementation notes):**

- **`decimate=False` row-order bug was real, and is now fixed (F1b.10).**
  Before the fix: running the UR10 example with `decimate=False` (the
  example script's own default call pattern) gave correlation 0.44 and RMSE
  ~4036 on training data, vs. correlation 0.966 and RMSE ~374 for the exact
  same data with `decimate=True`. After transposing before flattening in
  `_prepare_undecimated_data()`, both paths agree (correlation ≈1.0000,
  matching RMSE/condition number on the regenerated data below). The
  `_compute_per_joint_stats()` gate on `decimate=True` was removed
  accordingly — both paths now produce joint-major rows safely.
- **UR10's bundled example data is now physically realistic (F1b.11).**
  The original `data/identification_*_simulation.csv` used torque values
  unrelated to real Nm (thousands of Nm against a robot whose configured
  torque limits are 330/330/150/54/54/54 Nm). Regenerated via `pin.rnea()`
  on a Fourier-series excitation trajectory (500 training samples @ 500 Hz,
  400 validation samples, different seed/harmonics), with torques computed
  through the identical finite-difference `dq`/`ddq` pipeline the
  identification code uses internally — not independently-computed
  "textbook-correct" derivatives that the framework would then silently
  contradict. Result: training correlation 1.0000, validation correlation
  0.99999, 99.8% RMSE improvement over nominal parameters on the held-out
  set — the validation mechanism now demonstrates what it's meant to
  demonstrate, instead of masking the row-order/data-unit issues with a
  "warn" insight.
- **A third, separate bug was found but deliberately left unfixed:**
  `calculate_first_second_order_differentiation()` in
  `identification/identification_tools.py` computes `ddq` via
  `for jj in range(nq - 1): ddq[:, jj] = np.gradient(...)` — for any
  robot without continuous/spherical joints (`nq == nv`, e.g. UR10's 6
  revolute joints), this silently skips the **last** joint, whose
  acceleration column stays zero regardless of true motion. Because both
  the regressor and any hand-computed validation torque go through this
  same function, the UR10 data regeneration above is self-consistent with
  it (not affected in practice), but the bug still means the last active
  joint's inertial/acceleration-dependent terms are always under-modeled
  for *every* robot using this helper with the default `dt=None` path. Not
  fixed here: broader blast radius (every identification example, not just
  UR10) and outside the scope of what was asked; flagged for a future fix.

**Files touched:** `src/figaroh/tools/_report_common.py` (new),
`src/figaroh/tools/report.py` (refactored to use it),
`src/figaroh/tools/identification_report.py` (new),
`src/figaroh/tools/__init__.py`,
`src/figaroh/tools/qrdecomposition.py` (`base_indices`/`get_base_indices()`),
`src/figaroh/identification/config.py` (`validation_data_file`),
`src/figaroh/identification/base_identification.py`
(`solve(html_report=...)`, `_load_validation_data()`,
`_compute_validation_metrics()`, `_compute_per_joint_stats()`,
`print_quality_report()`, `export_html_report()`,
`_prepare_undecimated_data()` row-order fix),
`tests/unit/test_identification_report.py` (new).
In `figaroh-examples`: `examples/{ur10,tiago,staubli_tx40}/utils/*_tools.py`
(`load_trajectory_data(data_source=...)`),
`examples/ur10/config/ur10_unified_config.yaml`
(`tasks.identification.data.validation_data_file`),
`examples/ur10/data/identification_{q,tau}_simulation.csv` (regenerated),
`examples/ur10/data/validation/identification_{q,tau}_simulation.csv`
(new, regenerated).

---

## Step 1: Fix the reshape bug and dedup fallback plotting (Feature 5, Phases 1–2)

**Status:** ✅ Implemented (2026-07-11). Verified with
`pytest tests/unit/test_results_manager.py` (6 new tests) and
`pytest tests/unit` (333 passed, 5 skipped — the 4 pre-existing `test_robotipopt.py`
failures are an unrelated missing `cyipopt` dependency, confirmed present before this
work started too). Also verified against real runs: UR10 identification
(`decimate=False`) via `UR10Identification.plot_results()` directly (no manual
per-joint workaround) now produces 12 correctly-labeled lines (6 joints ×
measured/identified) instead of one mislabeled "Joint 1" trace; TIAGo calibration
via `TiagoCalibration.plot_results()` still produces its plot unchanged.

Implementation notes (see also "Findings" below for one correction to this section's
original plan):

**Why first:** small, low-risk, fixes things already confirmed broken/duplicated, zero
dependencies. Also unblocks Step 3 (Feature 6 Phase A), which reuses the same joint-major
convention this step fixes.

**Context — the survey that produced Feature 5.** While verifying Feature 1b against the
UR10 identification example, a full survey of the reporting stack turned up gaps and a
real bug:

| Layer | Calibration | Identification | Optimal calibration | Optimal trajectory |
|---|---|---|---|---|
| Terminal report | `print_quality_report()` | `print_quality_report()` | **none** | **none** |
| HTML report | `generate_calibration_report()` | `generate_identification_report()` | **none** | **none** |
| Plotting | `plot_results()` → `ResultsManager` → fallback | same pattern | same pattern | same pattern |
| Save (yaml/csv/npz) | `save_results()` | `save_results()` | `save_results()` | `save_results()` |
| Held-out validation | `validation_data_file` | `validation_data_file` | n/a | n/a |
| Machine-readable pass/fail | **none** | **none** | **none** | **none** |

Findings from the survey:

1. **A real plotting bug.** `ResultsManager.plot_identification_results()`
   (`figaroh/src/figaroh/utils/results_manager.py:233`) reshapes any 1D torque array via
   `tau.reshape(-1, 1)` before plotting. `BaseIdentification`'s torque arrays
   (`self.result["torque processed"/"torque estimated"]`) are flattened **joint-major**
   (all samples of joint 0, then joint 1, ...), so this reshape produces a single column
   labeled "Joint 1" spanning all joints concatenated — confirmed by hand when the UR10
   identification plot mislabeled 6 joints as one. Worked around manually in that session;
   not fixed in the library. **Fixed in this step.**
2. **Four near-identical fallback blocks.** Every one of `BaseCalibration`,
   `BaseIdentification`, `BaseOptimalCalibration`, `BaseOptimalTrajectory` implements
   `plot_results()` as `try: self.results_manager.plot_X() except: <raw matplotlib
   fallback>` — copy-pasted, not shared. **Deduplicated in this step.**
3. **Two domains have no report at all.** `BaseOptimalCalibration` and
   `BaseOptimalTrajectory` only have `plot_results()`/`save_results()` — no
   `print_quality_report()`, no HTML report, no insights. Addressed in Step 6.
4. **No machine-checkable verdict anywhere.** Every report — terminal, HTML, or plot — is
   for a human to read. Nothing emits a structured pass/fail against a threshold. Addressed
   in Step 2.
5. **Two independently-evolved HTML report generators already share a kernel.**
   `tools/_report_common.py` (`_STYLE`, `_esc`, `_uncertainty_tier`, `_insights_section`,
   `_param_uncertainty_section`, `_correlation_section`) is reused by both
   `tools/report.py` (calibration) and `tools/identification_report.py` (identification).
   This is the one place consolidation already happened, and it's the foundation Steps
   1–7 build on rather than replace.

**Feature 5 decisions** (apply across Steps 1, 2, 6, and 7 — recorded once, here, at first
appearance):

**D1: Fix the reshape bug by making the caller state joint count explicitly.**
`ResultsManager` has no way to *infer* joint-major vs. sample-major layout from a bare 1D
array — it's ambiguous by construction. The caller (`BaseIdentification`) already knows
`n_active` and the joint-major convention (the same one `_compute_per_joint_stats()`
already relies on). Fix at the boundary: require the caller to pass `n_joints`/
`joint_names` explicitly rather than have `ResultsManager` guess.

**D2: Fallback-plotting dedup as a plain helper function, not a shared base class.**
`BaseCalibration`, `BaseIdentification`, `BaseOptimalCalibration`, `BaseOptimalTrajectory`
have no common ancestor today and shouldn't be forced into one just for this. A
module-level helper `plot_with_fallback(primary, fallback, logger)` in
`utils/results_manager.py` is a lower-risk refactor than introducing a mixin into four
class hierarchies.

**D3: Optimal-\* reports (Step 6) reuse `_report_common.py`, get their own adapter files.**
Matches the existing calibration/identification split — shared CSS/insight/uncertainty
primitives, domain-specific section builders. Consistent with the same call already made
once for Feature 1b.

**D4: Verification thresholds (Step 2) are per-call config, not hardcoded constants.**
"Ill-conditioned" for a 6-DOF arm and for a 30-DOF humanoid are different numbers. Default
thresholds ship as sensible starting points (reusing the same cutoffs already used for
insight-flagging, e.g. condition number 100/1000), but every threshold is overridable per
call so a robot's own config can tighten or loosen the bar.

**D5: Verification (Step 2) is opt-in output, not a new gate inside `solve()`.** `solve()`
already does a lot (compute → store → print → optionally plot/save/HTML). Adding a
threshold check that could raise felt like scope creep into a method that currently never
fails after a successful numerical solve. Instead: a new `verify()` method computed from
already-stored `self.result`, callable whenever, returning a verdict object;
`export_verification_report()` writes it to JSON. CI scripts call these explicitly.

**D6 (resolved):** Step 6 (optimal-\* reports) builds bespoke `_summary_section`/
`_build_insights` per domain now, the same way Features 1/1b were, rather than waiting on
Step 7's schema spike. See Step 6 for why.

### Phase 1: Fix the `ResultsManager` joint-major reshape bug

**Files:** `figaroh/src/figaroh/utils/results_manager.py`,
`figaroh/src/figaroh/identification/base_identification.py`

| # | Task | Details | Status |
|---|------|---------|--------|
| P1.1 | Add explicit shape info to the plot call | `plot_identification_results()` gains optional `n_joints: int` and `joint_names` params, both defaulting to `None`/looked up from `self.result` | ✅ |
| P1.2 | Fix the reshape | Replace `tau.reshape(-1, 1)` with `tau.reshape(n_joints, -1).T` when the input is 1D **and** `n_joints` is given; falls back to the old `reshape(-1, 1)` only when `n_joints` is not provided (rather than guessing) — matches the joint-major convention already used in `_compute_per_joint_stats()` (`base_identification.py`) | ✅ |
| P1.3 | Thread `joint_names` through | `_plot_torque_comparison`/`_plot_torque_residuals` already accept `joint_names` — confirmed they render one legend entry per joint once P1.2 lands | ✅ |
| P1.4 | Update call site | `BaseIdentification.plot_results()` passes `n_joints=len(self.identif_config["act_idxv"])`, `joint_names=self.identif_config.get("active_joints")` | ✅ |
| P1.5 | Regression test | `tests/unit/test_results_manager.py` (new file): synthetic joint-major flattened array with known per-joint values; asserts each line's y-data equals the correct joint's slice, plus a no-`n_joints` fallback case and a 2D-input-unaffected case | ✅ |
| P1.6 | Manual verification | Re-ran the UR10 identification example (`decimate=False`); `ur10_identif.plot_results()` (no manual workaround) now produces 12 lines (6 joints × measured/identified), each correctly labeled by joint name | ✅ |

**Acceptance criteria:** `plot_results()` on a multi-joint identification produces one
correctly-labeled subplot/trace per joint; existing calibration plotting (already 2D,
unaffected by this bug) has zero behavior change; `tests/unit` still green.

**Risk:** low — the function's current behavior is already wrong, so any caller relying on
the old output was relying on a bug. Confirmed the only caller of
`plot_identification_results` outside its own module is `BaseIdentification.plot_results()`.

### Phase 2: Deduplicate the fallback-plotting block

**Files:** `figaroh/src/figaroh/utils/results_manager.py`,
`figaroh/src/figaroh/calibration/base_calibration.py`,
`figaroh/src/figaroh/identification/base_identification.py`,
`figaroh/src/figaroh/optimal/base_optimal_calibration.py`,
`figaroh/src/figaroh/optimal/base_optimal_trajectory.py`

| # | Task | Details | Status |
|---|------|---------|--------|
| P2.1 | Add `plot_with_fallback()` helper | In `utils/results_manager.py`: `def plot_with_fallback(primary: Callable[[], None], fallback: Callable[[], None], logger, context: str) -> None` — calls `primary()`, on exception logs and calls `fallback()` | ✅ |
| P2.2 | Refactor `BaseCalibration.plot_results()` | Replaced inline try/except with `plot_with_fallback(lambda: self.results_manager.plot_calibration_results(), _basic_plots, logger, "calibration")` | ✅ |
| P2.3 | Refactor `BaseIdentification.plot_results()` | Same pattern; the managed-plot lambda now also passes `n_joints`/`joint_names` per Phase 1 | ✅ |
| P2.4 | Refactor `BaseOptimalCalibration.plot_results()`/`.plot()` | **Confirmed:** two genuinely different methods, not dead code — `.plot()` (D-optimality-ratio/weight plots, no `ResultsManager`) is what `solve()` actually calls; `plot_results()` (the `ResultsManager` + fallback pattern) has no callers anywhere in `figaroh`/`figaroh-examples` today but is public API. Only `plot_results()` matches the pattern being deduped, so only it was refactored; `.plot()` was left untouched (out of scope — it doesn't use `ResultsManager` at all) | ✅ |
| P2.5 | Refactor `BaseOptimalTrajectory.plot_results()` | Same pattern | ✅ |
| P2.6 | Unit test | `tests/unit/test_results_manager.py`: fallback triggers when primary raises; primary's return value used when it succeeds (fallback not called); no double-plotting | ✅ |

**Acceptance criteria:** all four `plot_results()` methods behaviorally unchanged (same
plots produced, same fallback-on-error behavior), each now ~5 lines instead of ~15;
duplicated try/except block count goes from 4 to 0.

**Finding (worth keeping):** `base_optimal_calibration.py` and `base_optimal_trajectory.py`
originally caught only `ImportError` around their `ResultsManager` call (so an internal
exception thrown *inside* `plot_optimal_calibration_results()`/`plot_optimal_trajectory_results()`
— e.g. from a bad argument — would propagate uncaught), while `base_calibration.py` and
`base_identification.py` caught broad `Exception` (so an internal error there safely falls
back). Routing all four through the same `plot_with_fallback()` helper (which always
catches `Exception`) fixes this inconsistency as a side effect of the dedup — the two
optimal-\* classes are now as robust to an internal plotting error as the other two,
matching the spirit of "four near-identical blocks" this step set out to unify. Also
removed two now-redundant inline `from .results_manager import ResultsManager` imports in
`BaseOptimalCalibration.save_results()`/`BaseOptimalTrajectory.save_results()` that would
otherwise shadow (and lint-flag, `F811`) the new module-level import.

**Risk:** low — pure refactor, no behavior change intended. Verify via before/after manual
run on one calibration and one identification example.

---

## Step 2: Machine-readable verification verdict (Feature 5, Phase 4)

**Status:** ✅ Implemented (2026-07-11). Verified with `pytest tests/unit/test_verification.py`
(29 new tests, all passing) and `pytest tests/unit` (362 passed, 5 skipped, the same 4
pre-existing unrelated `cyipopt` failures as Step 1). Also verified against real runs:
`BaseIdentification.verify()` on the UR10 identification example (both without and with the
genuinely separate validation dataset from Feature 1b — condition-number check correctly
fails against the default 1000 threshold given the example's known ill-conditioning,
`validation_correlation`/`validation_improvement_pct` checks correctly pass at 1.0000/99.8%
when validation data is present and are correctly *skipped* — not failed — when it isn't);
`BaseCalibration.verify()` on the TIAGo calibration example (condition number 282, passes).
The UR10 example's new `--verify` CLI flag was run end-to-end and confirmed to exit 1 on
the current (ill-conditioned) run and write a valid, numpy-safe
`results/identification_verification.json`.

Implementation notes (deviations from the original design sketch, and why):

- **Metric names differ from the design sketch's threshold table** to match what
  `evaluation_metrics`/`self.result` actually expose: calibration validation RMSE fields are
  `position_rmse_mm`/`orientation_rmse_deg` (sourced from `results_data["validation_metrics"]`
  populated by `_compute_validation_metrics()`), identification uses
  `validation_correlation`/`validation_improvement_pct` (sourced from the identification
  analogue). Both are populated only when `results_data`/`result` actually contains
  `validation_metrics` — i.e., only when the caller configured `validation_data_file`.
- **A threshold whose metric is absent or NaN is skipped, not failed** (`evaluate_thresholds`
  in `_report_common.py`). This was necessary, not just convenient: with no validation data
  configured, `verify()` should say "these checks don't apply" rather than report a false
  failure on a validation-set threshold that was never measurable. An empty check list counts
  as passed — there's nothing to fail on.
- **`self._config_file_path` is new state**, set at the top of `load_param()` in both
  `BaseCalibration` and `BaseIdentification` (neither class previously stored the config path
  after construction). Needed for the config-hash provenance field (P4.5); a
  one-line, behavior-preserving addition to an existing method.
- **P4.9 (stretch) — wiring a `figaroh-examples` smoke test to assert on the verdict JSON —
  was not done.** It was explicitly marked stretch in the original plan; the manual real-run
  verification above (including the exit-code check) covers the same acceptance criteria for
  now.

**Why second, ahead of Step 6:** this is the phase that changes what the reporting stack
*is* — today nothing produces a pass/fail a CI job can branch on; every report is for a
human to read. It's also the foundation Step 3 (Feature 6) builds on, so it needs to exist
before Feature 6 starts, not after.

**Files (new):** additions to `figaroh/src/figaroh/tools/_report_common.py`,
`tests/unit/test_verification.py`
**Files (modified):** `figaroh/src/figaroh/calibration/base_calibration.py`,
`figaroh/src/figaroh/identification/base_identification.py`

Design sketch:

```python
# tools/_report_common.py
@dataclass
class ThresholdCheck:
    name: str
    value: float
    threshold: float
    comparison: str  # "max" or "min"
    passed: bool

@dataclass
class VerificationVerdict:
    passed: bool
    checks: list[ThresholdCheck]
    metrics: dict[str, float]      # everything checked, pass or fail
    insights: list[str]            # reuse existing insight text
    metadata: dict[str, str]       # git commit, config hash, timestamp, robot name

def evaluate_thresholds(metrics: dict, thresholds: dict) -> VerificationVerdict: ...
```

Default thresholds (overridable, per D4):

| Domain | Metric | Default | Comparison |
|---|---|---|---|
| Calibration | position RMSE (validation set) | 2 mm | max |
| Calibration | orientation RMSE (validation set) | 0.1 deg | max |
| Calibration | condition number | 1000 | max |
| Identification | correlation (validation set) | 0.9 | min |
| Identification | condition number | 1000 | max |
| Identification | validation RMSE improvement | 50% | min |

| # | Task | Details | Status |
|---|------|---------|--------|
| P4.1 | `ThresholdCheck`/`VerificationVerdict` dataclasses + `evaluate_thresholds()` | In `_report_common.py`, shared by both domains | ✅ |
| P4.2 | Default threshold tables | One dict per domain, module-level constants near the existing `*_WARN_PCT` constants | ✅ |
| P4.3 | `BaseCalibration.verify(thresholds: dict = None) -> VerificationVerdict` | Pulls from `self.evaluation_metrics`/validation metrics already computed in `solve()` | ✅ |
| P4.4 | `BaseIdentification.verify(thresholds: dict = None) -> VerificationVerdict` | Pulls from `self.result`/`self.correlation` | ✅ |
| P4.5 | Provenance metadata | Git commit (`git rev-parse HEAD`, best-effort — don't fail if not in a git repo), config file sha256, ISO-8601 timestamp, robot name | ✅ |
| P4.6 | `export_verification_report(output_path=None) -> str` | Writes the verdict as JSON (reuse `ResultsManager._convert_for_serialization` for numpy-safety); same opt-in pattern as `export_html_report()`. **Note (from Step 3's resolution):** this method's output is extended in Step 3 with `series`/`compat` fields for Feature 6 — design it with that extension in mind, not as a closed schema | ✅ |
| P4.7 | Unit tests | Threshold pass/fail in both directions, missing-metric handling, JSON round-trip, metadata present | ✅ (`tests/unit/test_verification.py`, 29 tests) |
| P4.8 | Example CLI wiring | Add a `--verify` flag to **one** example script (`figaroh-examples/examples/ur10/identification.py`) that calls `verify()` + `export_verification_report()` and `sys.exit(1)` on failure — a concrete demonstration of CI-gateable usage, not yet applied everywhere | ✅ |
| P4.9 (stretch) | Wire into `figaroh-examples` smoke tests | At least one smoke test asserts on the verdict JSON's `passed` field instead of only "script didn't crash" | Not done (stretch — see implementation notes above) |

**Acceptance criteria:** `identifier.verify()` / `calibrator.verify()` return a
`VerificationVerdict` with correct pass/fail against both default and overridden
thresholds ✅; `export_verification_report()` produces valid JSON with provenance metadata
✅; the UR10 example's `--verify` flag exits nonzero on the current run (ill-conditioned by
the default 1000 threshold — a real, not manufactured, bad case) ✅. Not separately verified:
a config change that flips the same run from failing to passing (not needed to demonstrate
the exit-code mechanism works).

**Risk:** medium. The default thresholds in the table above are proposed starting points,
not values validated against real acceptance criteria from any actual deployment — confirmed
during verification: the UR10 example's condition number (~20713) fails the default 1000
threshold in every run so far, training or validation-augmented, which is a property of the
example's excitation trajectory (already flagged as "ill-conditioned" by the pre-existing
insights code), not a defect in `verify()` itself.

**Open questions:**

1. The default thresholds above (2mm position RMSE, 0.9 min correlation, etc.) are
   proposed starting points, not sourced from any actual deployment acceptance criteria.
   Should these ship as-is, or replace them with real numbers from prior calibration/
   identification work?
2. Should `verify()`/`export_verification_report()` be added to `BaseCalibration`/
   `BaseIdentification` only (as scoped above), or also to the optimal-\* classes once
   Step 6 exists (e.g., "did the optimal trajectory search actually reduce condition
   number by more than X%")?

---

## Step 3: Bundle export, redefined (Feature 6, Phase A)

**Status:** Proposed — awaiting review, not yet implemented.

**Context — why Feature 6 exists.** A follow-on discussion to Feature 5 explored turning
the reporting stack into a "product-level" interactive V&V suite: a run library, a
backend, trend dashboards, before/after and cross-run comparison, annotations, live
threshold editing. Self-critique of that idea before committing to it surfaced four
problems worth acting on rather than building past:

1. **No validated need for that scope yet.** Current usage is: run an example script, read
   one HTML report. A backend, run history, and a trend dashboard solve a problem (many
   runs, multiple people comparing them) that hasn't been observed, only anticipated.
2. **Purpose-built tools already do the hard part.** Tools like rerun.io are specifically
   built for loading dynamic, multi-series time-series data and visualizing it
   interactively — better than a hand-rolled Plotly page would be, especially as the
   number of joints/DOFs and sample counts grow. Building that ourselves is the wrong
   place to spend effort right now.
3. **"Before/after" and "compare two runs" are different features, not one.** Before/after
   (nominal vs. calibrated, or nominal vs. identified) is data that already exists inside a
   single `solve()` call's `result` — no second file, no comparison *mechanism* needed,
   just better visibility into data already computed. Comparing two *separate* runs is a
   genuinely different problem: it needs two files and a check that they're comparable at
   all.
4. **Comparing incompatible runs silently is actively dangerous here.** Feature 1b found
   and fixed a bug where `decimate=True` vs. `decimate=False` produced different row
   semantics for the same identification code path. A compare feature that overlays two
   runs' curves without checking they used the same settings could produce exactly that
   kind of misleading result, dressed up as a definitive-looking chart.

**Decision:** cut scope to an MVP that only does what's already justified — expose
already-computed before/after data interactively, and support comparing two runs safely as
a static, no-backend artifact. Everything speculative (backend, run library, trend
dashboard, annotations, live threshold editing, rich dynamic multi-series visualization) is
explicitly deferred, not designed away — revisit only if this MVP gets reached for
repeatedly.

**Feature 6 decisions** (apply across Steps 3, 4, and 5 — recorded once, here, at first
appearance):

**D1: Before/after is exposure, not a new comparison mechanism.** Both `BaseCalibration`
and `BaseIdentification` already compute nominal-vs-fitted comparisons (calibration's FK
validation section, identification's `_compute_validation_metrics()`). The MVP work here
is making that data interactively visible (zoom/hover on an overlay plot) instead of only
a static image/table — not building a new before/after concept.

**D2: Cross-run compare (Step 5) ships as a static, self-contained HTML page.** No
backend. One HTML file (following the same self-contained/theme-aware doctrine already
used for `report.html`/`identification_report.html`) that loads two JSON bundle files via
drag-and-drop/`<input type=file>`, entirely client-side. Shareable as a single file, same
as today's reports.

**D3: A compatibility check is mandatory, not optional, before rendering a comparison.**
Before rendering anything, check: same domain (calibration vs. identification), same
active joints/DOF names, same `decimate` setting (identification), similar sample counts.
On mismatch: block the comparison and say why, don't render a misleading overlay with a
footnote. This directly follows from Context point 4.

**D4 (resolved — see below):** originally scoped as a bundle schema separate from Step 2's
`VerificationVerdict`. Resolved: Step 2's `export_verification_report()` is extended with
the two fields below (`series`, `compat`) instead of a second, separate
`export_json_bundle()` method. One export method per domain, not two.

**D5: Rich/dynamic multi-series visualization is explicitly not built here.** If a future
need emerges for genuinely rich exploration (many joints, long trajectories, live/
streaming data, deep drill-down), adopt an existing tool (e.g. rerun.io) rather than
extend this MVP's simple overlay plots into that territory. This boundary is intentional,
not a placeholder for "later we'll build our own."

**D6: Explicitly deferred, not designed for.** No backend, no run library/index, no
history/trend dashboard, no annotations, no live threshold editor, no provenance/config
diffing beyond the D3 compatibility check. All of these were discussed in the earlier
brainstorm; none are scoped here.

### Phase A: Extend Step 2's export with `series` and `compat`

**Files:** `figaroh/src/figaroh/calibration/base_calibration.py`,
`figaroh/src/figaroh/identification/base_identification.py`,
`figaroh/src/figaroh/tools/_report_common.py` (same files as Step 2 — no new export
method, no new file)

| # | Task | Details |
|---|------|---------|
| A.1 | Add `series` and `compat` to `VerificationVerdict` | `series: {time, nominal, fitted, measured}`, `compat: {active_joints/dof_names, decimate, sample_count, config_hash}` — extends the dataclass from Step 2, doesn't replace it |
| A.2 | `export_verification_report()` already writes these | No second export method: Step 2's `export_verification_report(output_path=None)` now includes `series`/`compat` in its JSON output |
| A.3 | Populate `series` | Calibration: per-DOF nominal/calibrated/measured pose errors. Identification: per-joint nominal/identified/measured torque (reuse the joint-major slicing convention from `_compute_per_joint_stats()`, fixed in Step 1) |
| A.4 | Populate `compat` | `active_joints` (or DOF names), `decimate` (identification only), `num samples`, config file sha256 |
| A.5 | Numpy-safe JSON serialization | Reuse `ResultsManager._convert_for_serialization` rather than reimplementing (same as Step 2's P4.6) |
| A.6 | Unit tests | Valid JSON, round-trips, numpy arrays/NaN handled, `series`/`compat` fields present for both domains — extend Step 2's `test_verification.py` rather than writing a parallel test file |

**Acceptance criteria:** `export_verification_report()` (from Step 2, now extended) produces
a valid, numpy-safe JSON file for both domains, containing everything Steps 4 and 5 need
and nothing more.

**Open question:** bundle/verdict output location — alongside the existing
`results/{report}.html`/`.npz` output (e.g. `results/identification_verification.json`),
or somewhere else?

---

## Step 4: Before/after interactive panel (Feature 6, Phase B)

**Status:** Proposed — awaiting review, not yet implemented.

**Files:** `figaroh/src/figaroh/tools/report.py`,
`figaroh/src/figaroh/tools/identification_report.py`,
`figaroh/src/figaroh/tools/_report_common.py`

| # | Task | Details |
|---|------|---------|
| B.1 | Decide embedding approach | Embed inline in the existing `generate_calibration_report()`/`generate_identification_report()` output (inline a small charting snippet with the series data as an embedded `<script>` payload) rather than a separate viewer page — keeps the "open one file, see everything" property the current reports already have |
| B.2 | Interactive overlay chart | Nominal vs. fitted vs. measured as a zoomable/hoverable line/scatter overlay, replacing the current static plot embedded via matplotlib PNG |
| B.3 | Keep it simple per D5 | One chart type (line/scatter with zoom+hover), no multi-panel dashboards, no attempt to generalize beyond what's already in `series` |
| B.4 | Verify against real examples | UR10 identification and one calibration example (e.g. TIAGo) — confirm the panel renders correctly and matches the existing static numbers |

**Acceptance criteria:** existing reports gain an interactive before/after chart with no
loss of the "single self-contained file" property; existing report tests
(`test_report.py`, `test_identification_report.py`) still pass.

**Risk:** low-medium — depends on choosing a charting approach that can be inlined without
an external CDN dependency (consistent with the existing no-CDN rule). A minimal
hand-rolled canvas/SVG chart or a small vendored charting library are both viable; pick
during implementation based on file-size/complexity trade-off.

**Open question (shared with Step 5):** charting approach — minimal hand-rolled SVG/canvas
vs. a small vendored (inlined, no-CDN) charting library. Worth a quick spike before
committing, given it's shared by this step and Step 5.

---

## Step 5: Static two-run compare page (Feature 6, Phase C)

**Status:** Proposed — awaiting review, not yet implemented.

**Files (new):** `figaroh/src/figaroh/tools/compare_report.py` (generates the static HTML
shell), a template following the existing `_STYLE` doctrine
**Files (modified):** none required — this consumes the JSON from Step 3, doesn't change
either Base class further

| # | Task | Details |
|---|------|---------|
| C.1 | Compatibility check (D3) | Compare `compat` blocks of the two loaded JSON files: same domain, same active joints/DOF names, same decimate flag, comparable sample counts. Block rendering with a clear message on mismatch; allow an explicit "compare anyway" override for the user, but never silently proceed |
| C.2 | Metric diff table | Per summary stat: run A value, run B value, Δ, % change, colored by improve/regress |
| C.3 | Overlaid series plot | Both runs' nominal/fitted/measured curves on shared axes, per-run visibility toggle, reusing Step 4's chart approach |
| C.4 | Client-side file loading | Drag-and-drop or `<input type=file>` for both JSON files; no server, no network request |
| C.5 | Manual verification | Compare two real UR10 identification runs — e.g. a run from before the `decimate=False` row-order fix (Step 1) vs. after, as a real-world test of both the diff table and the compatibility check (these two runs differ only in the fix, not in `compat` fields, so this specific pair should compare cleanly and show the RMSE/correlation improvement) |

**Acceptance criteria:** compare page renders correctly on two compatible exports; refuses
or clearly warns on incompatible ones (different `decimate`, different active joints,
different domain); works fully offline as a single opened HTML file plus two dropped JSON
files.

**Risk:** low — no backend, no new state to manage, smallest-blast-radius way to deliver
the comparison feature. Main risk is scope creep back toward the deferred list below —
resist adding "just one more" feature (history, annotations) into this page.

### Feature 6 verification checklist (Steps 3–5 together)

- [ ] `export_verification_report()` (Step 2, extended in Step 3) produces valid,
      numpy-safe JSON with `series`/`compat` for both domains
- [ ] Before/after panel renders inline in existing HTML reports, interactive (zoom/hover)
- [ ] Existing report unit tests unaffected
- [ ] Compare page blocks/warns on incompatible exports (verified with a deliberately
      mismatched pair, e.g. `decimate=True` vs `decimate=False`)
- [ ] Compare page renders correctly on a real compatible pair (before/after the Step 1
      row-order fix), showing the expected RMSE/correlation improvement
- [ ] Zero new running processes required for any of Steps 3–5 — everything is either an
      existing `solve()`-time export or a static file opened in a browser

### Feature 6 explicitly deferred (do not build now — revisit only if this MVP gets used repeatedly)

- Run library / index / backend of any kind
- History / trend dashboard across many runs
- Annotations or notes on a run
- Live threshold editing in a UI
- Provenance/config diffing beyond the pass/fail compatibility check in C.1
- Rich dynamic/multi-series visualization, live/streaming data — adopt an existing tool
  (e.g. rerun.io) instead of extending this MVP's simple charts into that territory

---

## Step 6: Terminal + HTML reports for optimal-\* tasks (Feature 5, Phase 3)

**Status:** Proposed — awaiting review, not yet implemented.

**Why here, after Feature 6:** independent of Steps 2–5 (doesn't touch calibration or
identification), but D6 (Step 1) resolved that this should use bespoke sections rather
than wait on Step 7's schema spike — so there's no reason to block it on anything except
being the largest remaining well-scoped chunk of net-new coverage.

**Files (new):** `figaroh/src/figaroh/tools/optimal_calibration_report.py`,
`figaroh/src/figaroh/tools/optimal_trajectory_report.py`,
`tests/unit/test_optimal_calibration_report.py`,
`tests/unit/test_optimal_trajectory_report.py`
**Files (modified):** `figaroh/src/figaroh/optimal/base_optimal_calibration.py`,
`figaroh/src/figaroh/optimal/base_optimal_trajectory.py`

Content design (what "quality" means for a *design* step, not a *fit* step):

- **Optimal calibration:** number of candidate configurations considered vs. selected,
  D-optimality criterion value (final vs. theoretical max from using all candidates),
  final information-matrix condition number, top-weighted configurations. Insight
  candidates: "fewer than N configurations selected — verify sufficient excitation",
  "D-optimality within X% of the all-candidates ceiling — selection is near-optimal" /
  "far from ceiling — consider more candidates or iterations."
- **Optimal trajectory:** best condition number found, number of iterations/attempts,
  fraction of attempts that satisfied constraints, waypoint count, trajectory duration.
  Insight candidates: "condition number did not improve over the last N iterations —
  consider more waypoints or iterations", "constraint-satisfying attempts < X% — soft
  limits or velocity/acceleration bounds may be too tight."

| # | Task | Details |
|---|------|---------|
| P3.1 | `BaseOptimalCalibration.print_quality_report()` | Terminal report: candidates considered/selected, D-optimality value + ceiling comparison, condition number, top-5 configurations by weight |
| P3.2 | `tools/optimal_calibration_report.py` | `generate_optimal_calibration_report()` on the `_report_common.py` kernel; own `_build_insights`/`_summary_section` (bespoke, per D6) |
| P3.3 | `BaseOptimalCalibration.export_html_report()` | Same opt-in pattern as calibration/identification (`solve(..., html_report=False)` or a `generate(html_report=...)` param — confirm current `solve()`-equivalent method name during implementation) |
| P3.4 | `BaseOptimalTrajectory.print_quality_report()` | Terminal report: best condition number, iterations/attempts, constraint-satisfaction rate, trajectory parameters |
| P3.5 | `tools/optimal_trajectory_report.py` | `generate_optimal_trajectory_report()`, same kernel |
| P3.6 | `BaseOptimalTrajectory.export_html_report()` | Same opt-in pattern |
| P3.7 | Unit tests | Mirror `tests/unit/test_identification_report.py`'s structure: a `Fake*` fixture per domain, insight-triggering cases, HTML-escaping, missing-data cases — ~15-20 tests per new report module |
| P3.8 | Verify against a real example | `figaroh-examples`' UR10 `UR10OptimalCalibration`/`OptimalTrajectoryIPOPT` (`examples/ur10/utils/ur10_tools.py`) — run and inspect both reports by hand |

**Acceptance criteria:** both optimal-\* classes have terminal + HTML reports at parity
(one report type each, not necessarily identical sections) with calibration/identification;
new unit tests pass; verified against a real UR10 run.

**Risk:** medium. Unlike calibration/identification (which had an obvious "nominal vs.
fitted" quality axis to report on), "quality" for a *design* step is less standardized —
the insight thresholds proposed above (D-optimality-vs-ceiling %, constraint-satisfaction
rate) are a best first draft, not validated against domain literature the way condition-
number thresholds were reused from the existing calibration/identification code. Expect to
revise these after the first real run.

---

## Step 7: Unified report schema — spike only (Feature 5, Phase 5)

**Status:** exploratory — recommend a feasibility spike before committing, same pattern as
Feature 3's spike below.

**Why last:** lowest urgency of everything in this document. Do the spike (`P5.1`)
opportunistically whenever; the full migration (`P5.3`–`P5.5`) stays explicitly deferred
until a concrete need exists.

Rationale for even considering it: `report.py` (`_summary_section`, `_per_dof_section`,
`_validation_section`, `_build_insights`) and `identification_report.py`
(`_summary_section`, `_per_joint_section`, `_validation_section`, `_build_insights`) are
structurally parallel but independently written — same shape, different field names,
hand-duplicated. Step 6 adds two more structurally-parallel-but-distinct pairs. A shared
schema would mean writing the renderer once.

Rationale against doing it now: two duplicated pairs (current) is annoying; four (after
Step 6, unchanged) is more annoying but not unmanageable; a fifth domain (Feature 4's
rollout-refinement stage, in "Later" below — not started) is the point where "write the
renderer once" clearly pays for itself. Until then this is a refactor of already-shipped,
already-tested code for stylistic consistency, not new capability.

| # | Task | Details |
|---|------|---------|
| P5.1 | Feasibility spike | Draft `ReportSchema` dataclass (`summary_stats`, `insights`, `tables: list[Table]`, `validation: Optional[ValidationBlock]`) against the *actual* current field sets of all 4 existing report generators (2 from Features 1/1b, 2 from Step 6) — confirm it doesn't force an awkward fit for any of them |
| P5.2 | Decision | Go/no-go, appended to this section, based on the spike |
| P5.3 (if go) | Adapters | `calibration_to_schema()`, `identification_to_schema()`, `optimal_calibration_to_schema()`, `optimal_trajectory_to_schema()` |
| P5.4 (if go) | Shared renderers | One `render_html(schema)`, one `render_terminal(schema)` in `_report_common.py` |
| P5.5 (if go) | Migrate with a safety net | New adapters/renderers land alongside the old functions first; existing test suites (`test_report.py`, `test_identification_report.py`, Step 6's new tests, ...) re-run against output from the new path before the old bespoke functions are deleted |

**Acceptance criteria (if pursued):** all existing report tests pass against the new
renderer's output; no bespoke `_summary_section`/`_per_*_section` functions remain across
any domain; adding a 5th domain requires only writing an adapter, not a new renderer.

**Risk:** high relative to payoff *right now* — real regression risk against four
already-shipped, user-facing report generators, for a consistency benefit that only fully
pays off once a 5th domain exists. Recommend deferring until Feature 4 (rollout refinement,
in "Later" below) is actually scheduled, and doing only P5.1 (the spike) as a cheap way to
keep the option open.

---

## Later — not yet started

Features 2, 3, and 4 are placed last in this document not because they matter less in the
abstract, but because they're new, correctness-sensitive modeling work (new
parameterizations, a nonlinear reparameterization, a second dynamics engine), and Steps
1–7 above are what make that kind of change verifiable as it's built rather than caught by
accident the way Feature 1b's three real bugs were caught this session. Within this group,
keep the existing order: Feature 4 already depends on Feature 2's generic parameter type,
and Feature 3 is an independent spike that can slot in front of or behind Feature 2 without
affecting anything downstream.

### Feature 2: Generic `Parameter` + modifier abstraction

**Inspired by:** MuJoCo's `Parameter` dataclass, which pairs a value/bounds
with a `modifier(MjSpec, Parameter)` callback — decoupling "what's being
identified" from "how it's applied to the model." Any spec-settable
quantity becomes identifiable without touching the core solver.

**Why FIGAROH could use it:** `identification/parameter.py` and
`calibration/parameter.py` currently model fixed categories (inertial,
friction, actuator inertia, geometric offsets). Adding a new identifiable
quantity (e.g. F/T sensor bias, a transmission ratio) means extending
`RegressorBuilder` itself. A generic `Parameter` + `modifier` type would let
new quantities plug in without changing the regressor-building code —
though anything that isn't linear-in-τ can't live in the linear regressor
and would need to be solved outside it (see Feature 4).

**Scope (draft, not started):** Define a `Parameter` protocol/dataclass in
`figaroh/utils/` with `name`, `value`, `bounds`, and an `apply(model)`
callback; make it opt-in alongside the existing hardcoded parameter
columns rather than replacing them.

#### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| F2.1 | Design `Parameter` dataclass + `apply(model)` contract | Decide where it lives (`figaroh/utils/parameter.py`?) and how it composes with existing `identification/parameter.py` / `calibration/parameter.py` | Not started |
| F2.2 | Prototype one non-regressor parameter | E.g. sensor bias, to validate the abstraction outside the linear regressor path | Not started |
| F2.3 | Document the extension pattern | Short doc/example showing how to add a custom parameter | Not started |

### Feature 3: Log-Cholesky pseudo-inertia parameterization

**Inspired by:** MuJoCo's `InertiaType.Pseudo`, which parameterizes the
10-D pseudo-inertia through a Rucker & Wensing log-Cholesky encoding so
`J = UUᵀ ⪰ 0` holds by construction — physical consistency guaranteed
during the solve, not projected afterward.

**Why FIGAROH could use it:** `identification/physical_consistency.py`
currently enforces consistency via a convex SDP/LMI projection
(`picos` + cvxopt/mosek) as a *post-processing* step after the linear
least-squares solve. A log-Cholesky reparameterization would guarantee
consistency inline and drop the external SDP solver dependency — but it
also breaks linearity: the inertial fit would become a nonlinear
least-squares problem instead of closed-form OLS, so it only pays off where
FIGAROH is already doing (or willing to do) a nonlinear solve.

**Scope (draft, not started):** Exploratory — evaluate whether replacing
the SDP-projection step with an in-loop reparameterization is worth the
loss of closed-form linearity, before committing to an implementation.

#### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| F3.1 | Feasibility spike | Compare SDP-projection vs. log-Cholesky reparameterization on one fixture robot; measure accuracy and runtime cost of going nonlinear | Not started |
| F3.2 | Decision | Go/no-go write-up based on the spike, appended to this section | Not started |
| F3.3 | Implementation (if go) | `theta_from_pseudoinertia`/`pi_from_theta`-equivalent in `identification/`, wired as an alternative to `physical_consistency.py` | Not started |

### Feature 4: Black-box rollout refinement stage

**Inspired by:** MuJoCo `sysid`'s simulate-and-compare nonlinear least
squares over full trajectory rollouts — capable of identifying anything
expressible in the model (contact friction, actuator gains) with no
analytic regressor required.

**Why FIGAROH could use it:** FIGAROH already has a `backends/mujoco.py`
regressor backend. A natural pipeline: solve the linear RNEA regressor for
a fast, cheap first estimate (current FIGAROH pipeline), then optionally
refine with a rollout-based residual to absorb contact/friction
nonlinearities the linear regressor can't represent. This is the largest
and riskiest feature — it pulls in a second dynamics engine and a nonlinear
solver loop, and needs its own design pass.

**Scope:** Not designed yet. Largest of the group; should not be started
before Features 2 and 3 land, since it depends on having generic parameters
(Feature 2) to describe the rollout's decision variables.

#### Tasks

| # | Task | Details | Status |
|---|------|---------|--------|
| F4.1 | Design doc | Separate `docs/decisions/` entry once scoped — how the linear estimate seeds the rollout refinement, which solver, which backend | Not started |

---

## Notes

- Only Feature 1 has been implemented as of this writing (2026-07-10),
  verified with `pytest tests/unit/test_report.py` (19 passed) and a
  manually rendered sample report inspected for correct section content,
  HTML escaping, and insight flagging. It was additionally verified against
  a real run of `figaroh-examples/examples/tiago/calibration.py`.
- Feature 1b (identification reports + held-out validation) is implemented
  as of this writing (2026-07-10), verified with
  `pytest tests/unit/test_identification_report.py` (25 passed),
  `pytest tests/unit` (318 passed, 18 skipped, 0 failed), and a real run of
  `figaroh-examples/examples/ur10/identification.py`'s `UR10Identification`
  with a genuinely separate validation dataset. Verifying it surfaced two
  real pre-existing bugs, both since fixed: the `decimate=False` row-order
  mismatch in `_prepare_undecimated_data()` (F1b.10), and the bundled UR10
  example data using non-physical torque magnitudes, now regenerated via
  `pin.rnea()` on a self-consistent excitation trajectory (F1b.11). Training
  correlation is now 1.0000 and validation correlation 0.99999 (99.8% RMSE
  improvement over nominal), on both `decimate=True` and `decimate=False`.
  A third bug (`calculate_first_second_order_differentiation()`'s
  `range(nq - 1)` loop silently zeroing the last joint's acceleration) was
  found but deliberately left unfixed — see "Findings" under Feature 1b.
- Step 1 (Feature 5, Phases 1–2: reshape-bug fix + fallback-plotting dedup) is
  implemented as of this writing (2026-07-11), verified with
  `pytest tests/unit/test_results_manager.py` (6 new tests, all passing) and
  `pytest tests/unit` (333 passed, 5 skipped, 4 pre-existing unrelated
  `cyipopt`-dependency failures in `test_robotipopt.py`). Verified against real
  runs of both `figaroh-examples/examples/ur10/identification.py`'s
  `UR10Identification.plot_results()` (now produces 12 correctly-labeled lines
  instead of one mislabeled "Joint 1" trace) and
  `figaroh-examples/examples/tiago/calibration.py`'s
  `TiagoCalibration.plot_results()` (unchanged output, confirming the dedup
  refactor is behavior-preserving there). One correction to the original plan
  surfaced during implementation: `BaseOptimalCalibration` really does have two
  distinct methods (`.plot()`, called by `solve()`, not using `ResultsManager`
  at all; `plot_results()`, unused by any caller today but public API, using
  the `ResultsManager`+fallback pattern) — only `plot_results()` was in scope
  for the dedup. See Step 1's Phase 2 "Finding" for a second fix that fell out
  of the dedup: two of the four classes previously caught only `ImportError`
  around their `ResultsManager` call (not `Exception`), so an internal
  plotting error there would have propagated instead of falling back; routing
  all four through the shared `plot_with_fallback()` helper fixed this
  inconsistency too.
- Step 2 (Feature 5, Phase 4: machine-readable `VerificationVerdict`) is
  implemented as of this writing (2026-07-11), verified with
  `pytest tests/unit/test_verification.py` (29 new tests, all passing) and
  `pytest tests/unit` (362 passed, 5 skipped, the same 4 pre-existing
  `cyipopt` failures). Verified against real runs: `BaseIdentification.verify()`
  on the UR10 identification example correctly fails its `condition_number`
  check (~20713 vs. the default 1000 threshold — a real ill-conditioning, not
  a manufactured failure) while correctly passing `validation_correlation`
  (1.0000) and `validation_improvement_pct` (99.8%) once the genuinely
  separate validation dataset from Feature 1b is wired in, and correctly
  *skips* (not fails) those two checks when no validation data is configured;
  `BaseCalibration.verify()` on the TIAGo calibration example passes
  (condition number 282 vs. 1000). The UR10 example's new `--verify` CLI flag
  was run end-to-end and confirmed to exit 1 on the current run and write a
  valid, numpy-safe `results/identification_verification.json`. P4.9 (a
  stretch task to wire the verdict into a `figaroh-examples` smoke test) was
  not done — see Step 2's "Implementation notes" for why that's an acceptable
  gap for now.
- Steps 3–7 (formerly part of "Feature 5" and "Feature 6") are proposed, not
  implemented, as of this writing (2026-07-11). This document has gone
  through two structural revisions since they were written: merged from
  four files into one, then reordered from Feature-number order into build
  order. No task content, decision, risk note, or open question was dropped
  in either revision — only section order and connective tissue changed.
- Features 2–4 ("Later") are intentionally left at "draft scope" — detailed
  task breakdowns will be filled in immediately before each is started,
  once the design questions above are resolved, and only after Steps 1–7
  land per the recommended order.
