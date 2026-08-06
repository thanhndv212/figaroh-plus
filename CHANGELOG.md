# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.6] - 2026-08-06

### Fixed

- `fix(calibration)`: `calc_updated_fkm` now supports joint elasticity
  (`non_geom`) and camera/ref-frame base composition — capabilities
  previously implemented only in a second, dead-code function,
  `update_forward_kinematics` (zero callers in `figaroh`/`figaroh-examples`),
  which had regressed relative to its own predecessor and was deleted as
  part of this fix:
  - The base/camera transform (`bMo`/`wMo`) was computed but never composed
    into the returned pose.
  - Elasticity indexing (`xyz_rpy[elas_id + 3]`) went out of bounds for any
    rotary joint — i.e. every revolute joint.
  - The elasticity parameter-match loop reused a stale `key` left over from
    an unrelated earlier loop instead of iterating its own.
  - The per-sample pose write was gated on a parameter-count bookkeeping
    variable that accumulated across the whole sample loop instead of
    resetting per sample, silently zeroing out later samples.
  - (Found while implementing the merge) the elasticity branch also read
    `data.oMf`/`get_rel_transform` without re-running forward kinematics
    after perturbing `model.jointPlacements` for the deflection, so the
    elastic offset never actually reached the output pose — fixed by
    re-running FK before computing `oMee`.
  - `calc_updated_fkm` previously silently ignored `calib_config["non_geom"]`
    entirely (every real caller uses this function, not the dead one) — it
    now actually applies elasticity when the flag is set.
  - `NbMarkers > 1` now raises `NotImplementedError` instead of silently
    logging a warning and falling back to an identity marker transform.
  - Verified against the TIAGo Pro reference calibration dataset
    (`figaroh-examples/examples/tiago_pro`, 94 samples): the existing
    `full_params`/`known_baseframe=False` geometric path is numerically
    unchanged (bit-identical output for a fixed parameter set and data).
- `fix(config)`: `unified_to_legacy_config` read `parameters.free_flyer`, a
  key the unified schema never defines (the template puts it at
  `kinematics.free_flying_base`) — a unified config setting
  `free_flying_base: true` was silently ignored, always resolving to
  `free_flyer=False`. Extraction moved to `_extract_frame_info`, reading
  the correct key.
- `fix(config)`: eye-hand calibration (`base_to_ref_frame`/`ref_frame`) had
  no unified-format equivalent, even though the schema template already
  reserved `tasks.calibration.eye_hand.{camera_frame,reference_frame}` for
  it — `unified_to_legacy_config` just never read that section. Added
  `_extract_eye_hand_params`, wired in.
- `fix(calibration)`: `BaseCalibration` reported RMSE/MAE two different ways
  that could disagree by exactly `sqrt(n_dofs)` depending on which number
  you looked at — `_evaluate_solution`'s `evaluation_metrics["rmse"]`/`["mae"]`
  flattened every x/y/z/... residual component into one list (a "mean of
  squared components" convention), while `_compute_per_dof_stats`'s
  `pos_rmse_mm`/`orient_rmse_deg` and the validation table reduced each
  sample's residual vector to one Euclidean-norm distance first, then
  aggregated. For position-only calibration these differed by exactly
  `sqrt(3)`, always. Converged `_evaluate_solution`, `_detect_outliers`, and
  `_store_optimization_results`'s `_PEE_dist` onto the per-sample-distance
  convention — the physically meaningful one (an actual 3D positioning
  error in mm) and the one most of the report already used. Outlier
  detection is unaffected (scale-invariant comparison). Also added
  `pos_mae_mm`/`orient_mae_deg` to `_compute_per_dof_stats`'s "overall"
  block (RMSE already had a position/orientation split; MAE didn't),
  surfaced as "Position MAE"/"Orientation MAE" in both the terminal
  quality report and the HTML report (`report.py`). Re-verified against
  tiago, tiago_pro, ur10, and talos calibration runs.
- `fix(calibration)`: `calc_stddev()` ran *after*
  `_compute_parameter_correlation()` in `BaseCalibration`, but correlation
  reads `self._C_param`, which `calc_stddev()` is what sets — on a fresh
  instance's first `solve()`, correlation silently returned `[]` instead
  of the actual pairs. Reordered so `calc_stddev()` runs first.

### Added

- `feat(config)`: Legacy flat config format (`calibration:`/`identification:`
  top-level sections) is now deprecated in favor of the unified format —
  loading one emits a `DeprecationWarning` (+ `logger.warning`) from
  `figaroh.calibration.config.get_param_from_yaml` /
  `figaroh.identification.config.get_param_from_yaml`, the single parser
  every legacy-format caller funnels through regardless of entry point.
- `feat(utils)`: `figaroh.utils.config_migration` — converts a legacy config
  to unified format automatically (`python -m figaroh.utils.config_migration
  --input legacy.yaml --output unified.yaml`), including the eye-hand and
  free-flyer fields fixed above. An optional `--urdf` flag runs a
  round-trip self-check (converts, then re-parses through the real
  `unified_to_legacy_*config` and diffs against the original) rather than
  trusting the conversion blindly. Fields with no unified-format consumer
  today are preserved under `custom:` instead of silently dropped. See
  "Migrating from legacy to unified format" in
  `docs/source/concepts/config_parameters.md`.

## [0.4.5] - 2026-07-13

### Added

- `feat(identification)`: Weighted least squares (WLS) support —
  `BaseIdentification.solve(wls=True)` refines the OLS base-parameter
  estimate with iteratively-weighted least squares (Gautier, 1997) before
  computing quality metrics, so the terminal/HTML report and `verify()`
  verdict reflect the WLS-refined parameters rather than a stale OLS
  estimate. Config-driven via `identification.problem.wls` in the unified
  YAML config, following the existing `has_friction`/`has_joint_offset`
  pattern.
- `feat(tooling)`: `figaroh.tools.provenance` and `figaroh.tools.run_archive`
  — every calibration/identification run now carries a provenance record
  (git commit, config hash, timestamps, robot/asset identity), and
  `archive_run()` writes each run to a timestamped
  `results/runs/<robot>/<task>/<timestamp>/` directory (config snapshot,
  provenance JSON, report) instead of overwriting a single `results/` path.
- `feat(validation)`: When no separate validation dataset is configured,
  calibration and identification now fall back to evaluating validation
  metrics against the training data itself — rather than silently omitting
  the validation section — with an explicit warning surfaced in the
  terminal report, HTML report, and machine-readable insights.
- `feat(reporting)`: Self-contained HTML diagnostic reports for both
  calibration and identification.
  - `figaroh.tools.report.generate_calibration_report()` /
    `BaseCalibration.export_html_report()` — summary, auto-generated
    insights, per-DOF residual table, parameter uncertainty bars,
    correlation section, validation section.
  - `figaroh.tools.identification_report.generate_identification_report()` /
    `BaseIdentification.export_html_report()` — same shape, adapted to
    identification's per-joint torque residuals and base-parameter
    uncertainty.
  - `figaroh.tools._report_common` — shared HTML/CSS kernel (styling,
    escaping, insight/uncertainty/correlation section builders) reused by
    both generators.
  - Opt-in via `solve(html_report=True)` on both base classes, matching the
    existing `plotting`/`save_results` flag pattern.
- `feat(identification)`: `BaseIdentification.print_quality_report()` —
  terminal quality report (condition number, RMSE, per-joint residuals,
  base-parameter uncertainty, validation, physical-consistency/
  reconstruction status), the identification analogue of calibration's
  existing terminal report.
- `feat(identification)`: Held-out validation support for identification,
  mirroring calibration's — a genuinely separate dataset via
  `identification.data.validation_data_file`, never a runtime split of
  training data.
- `feat(reporting)`: Machine-readable verification verdicts.
  - `VerificationVerdict` / `ThresholdCheck` (`figaroh.tools._report_common`)
    — a structured pass/fail against per-metric thresholds, with
    provenance (git commit, config hash, timestamp).
  - `verify(thresholds=None)` and `export_verification_report(output_path=None,
    output_dir="results", thresholds=None)` added to both `BaseCalibration`
    and `BaseIdentification`. Default thresholds cover condition number and,
    when validation data is configured, validation-set RMSE/correlation/
    improvement — every threshold is overridable per call. A threshold whose
    metric isn't computable is skipped, not failed.
  - Called explicitly (not from inside `solve()`) — verification is opt-in
    output computed from already-stored results.
- `feat(reporting)`: Interactive before/after panel embedded in both HTML
  reports — a zoomable, hoverable overlay of nominal vs. fitted vs. measured
  values, rendered as hand-rolled inline SVG (no new dependency).
- `feat(reporting)`: `figaroh.tools.compare_report.generate_compare_page()`
  — a static, self-contained, offline HTML page that loads two exported
  verification JSON files client-side and renders a per-metric diff table
  plus an overlaid before/after chart, gated by a mandatory compatibility
  check (domain, joint/DOF names, `decimate`, sample count) with an
  explicit "compare anyway" override on mismatch. No backend, no run
  history.
- `fix(results_manager)`: `plot_identification_results()` reshaped 1D
  torque arrays as `reshape(-1, 1)`, silently collapsing a multi-joint,
  joint-major-flattened array into a single mislabeled trace. Now accepts
  explicit `n_joints`/`joint_names` and reshapes joint-major
  (`reshape(n_joints, -1).T`) when provided.
- `refactor(results_manager)`: `plot_with_fallback()` helper deduplicates
  the four near-identical try/except plotting blocks across
  `BaseCalibration`, `BaseIdentification`, `BaseOptimalCalibration`, and
  `BaseOptimalTrajectory`.

### Fixed

- `fix(calibration)`: `BaseCalibration.calc_stddev()` computed the
  calibrated-parameter count (`self.nvars`) once in `__init__`, before
  `initialize()` populated the actual parameter list — leaving it stuck at
  0 and silently producing empty `std_dev`/`std_pctg` lists. The "Parameter
  uncertainty" section was therefore always empty in both the terminal and
  HTML calibration reports. Now derives the count from the solved parameter
  vector (`len(result.x)`) at computation time.
- `fix(identification)`: `_compute_validation_metrics()`'s before/after
  torque chart sliced the first `n_active` joint-blocks off the full-model
  regressor, which is only correct when the active/identified joints are
  exactly DOFs `0..n_active-1`. For robots identifying a strict subset of
  joints (e.g. TIAGo: torso+arm out of many more DOFs), this silently
  returned the wrong joints' nominal/identified torque predictions —
  visible as identical nominal/fitted curves and wildly wrong magnitudes in
  the before/after chart. Now indexes by the actual `act_idxv` DOF indices,
  mirroring the row-selection already used correctly in
  `_decimate_regressor_matrix`.
- `fix(identification)`: `_apply_weighted_least_squares()` sliced the
  stacked regressor into `model.nv` equal blocks to estimate per-joint
  residual variance, which is only correct when every model joint is
  identified. For subset-identified robots (TIAGo: `model.nv=24`, 8 joints
  identified), this misaligned the slicing entirely, producing an RMSE
  ~2000x worse than plain OLS. Now uses `len(identif_config["act_idxv"])`,
  the same quantity `_decimate_regressor_matrix` already uses.
- `fix(identification)`: `_prepare_undecimated_data()` flattened torque
  data sample-major while regressor rows are joint-major, corrupting
  `solve(decimate=False)`'s correlation/RMSE (0.44 vs 0.966 on identical
  data pre-fix).
- `fix(reporting)`: the before/after HTML panel's joint/DOF dropdown
  rendered its options twice — once server-side in the generated HTML,
  once again client-side via JS — so every entry appeared duplicated in
  the selector.
- `fix(reporting)`: the before/after HTML panel embedded the full
  validation-set series inline with no cap; for large validation sets
  (e.g. a fallback-to-full-training-data run with ~45k samples) this
  produced multi-MB HTML pages whose SVG path strings were too large to
  render reliably in the browser. Now uniformly subsampled to at most 3000
  points, with the displayed sample count disclosed in the chart caption.
- `fix(export_validation)`: the interactive viser GUI's speed/pose/opacity
  sliders crashed with `AttributeError: 'GuiEvent' object has no attribute
  'value'` on every drag — `GuiEvent` exposes the changed control via
  `.target`, not `.value` directly.

Full design history and rationale:
[`docs/decisions/roadmap-mujoco-sysid-inspired-features.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/roadmap-mujoco-sysid-inspired-features.md).

## [0.4.4] - 2026-06-27

### Added

- `feat(calibration)`: Add validation data support for ground-truth FK testing.
  - `_load_validation_data(path)` — load a separate validation measurement CSV.
  - `_compute_validation_metrics()` — compare nominal vs calibrated FK against
    held-out measurements; reports position/orientation RMSE, max error, and
    improvement percentage.
  - Validation data path configurable via YAML (`validation_data_file`) or CLI
    (`--validation-data`) in example scripts.
- `feat(calibration)`: Add comprehensive statistical quality report.
  - `print_quality_report()` — formatted terminal table with convergence status,
    per-DOF residuals (X/Y/Z/rx/ry/rz with R²), overall RMSE, validation metrics,
    parameter uncertainty (top 5), and strongly correlated pairs (|ρ| > 0.8).
  - `_compute_per_dof_stats()` — mean, std, RMSE, max, R² per kinematic DOF.
  - `_compute_condition_number()` — condition number of Jacobian with
    well/moderate/ill-conditioned label.
  - `_compute_parameter_correlation()` — flags parameter pairs with |ρ| > 0.8
    from the covariance matrix.
  - Report printed automatically after `solve()` completes.
- `feat(calibration)`: Add `position_frame='body'|'world'` option to
  `_compute_logmap_residuals()` — choose body-frame or world-frame position error.
- `feat(urdf_exporter)`: New module `figaroh.tools.urdf_exporter` for applying
  calibrated joint parameters to URDF files with 12 joint-level categories.
- `feat(export_validation)`: New module `figaroh.tools.export_validation` with
  `URDFComparison` class — FK consistency checks, interactive viser visualization
  (trajectory animation, static overlay, error plots, replay controls).
- `feat(backends)`: Multi-simulator backend architecture.
  - `PinocchioBackend` — 371 lines wrapping all Pinocchio dynamics/kinematics calls.
  - `MuJoCoBackend` — 441 lines with URDF→MJCF auto-conversion.
  - `DynamicsBackend` abstract interface with 9 abstract + 6 optional methods.
  - `RobotIdentificationSystem` high-level integration API.
- `feat(tests)`: Cross-backend validation suite, backend unit tests (32 Pinocchio
  + 13 MuJoCo), TIAGo integration test fixture, URDF exporter tests.
- `docs`: Combined ROADMAP.md (Track A + Track B), comprehensive architecture
  docs, ADR for calibration validation plan.

### Fixed

- `fix(calibration)`: Replace element-wise RPY subtraction with proper SE3
  log-map pose error in `_compute_logmap_residuals()` — geometrically correct
  orientation residuals without singularity issues.
- `fix(backends)`: MuJoCoBackend regressor (delegates to Pinocchio analytical),
  Coriolis matrix (finite-difference Jacobian/2), mass matrix (adds missing
  `mj_forward` call).

### Changed

- `breaking(export_validation)`: Rename `trajectory_errors()` → `fk_consistency_check()`
  and `TrajectoryErrors` → `FkConsistencyResult` — clarifies this is a file-integrity
  check, NOT ground-truth FK validation.
- `chore`: Add black formatter to pre-commit hooks.
- `chore`: Modernize pyproject.toml with dev dependencies and tool configs.

### Tests

- 293 passed, 20 skipped, 0 failed (from 208 passed with 4 pre-existing failures).
- New: `test_backends.py` (45 tests), `test_cross_backend.py` (integration),
  `test_tiago_fixture.py` (287 lines), `test_urdf_exporter.py` (296 lines),
  `test_integration.py` (250 lines).

## [0.4.3] - 2026-06-02

### Added

- `feat(cad_constraints)`: New module `figaroh.identification.cad_constraints`
  with convex CAD-informed constraints for inertial parameter identification.
- `CADConstraints` dataclass — container for per-link mass bounds, first-moment
  (CoM) bounds, and symmetry equality constraints.
- `add_mass_bounds(cad, joint, *, m_min, m_max)` — add box constraint
  `m_min ≤ m_j ≤ m_max` for a link.
- `add_com_bounds(cad, joint, *, axis, h_min, h_max)` — add first-moment
  (linear) box constraint `h_min ≤ h_k ≤ h_max` for axis `k ∈ {x,y,z}`.
- `add_symmetry_constraints(cad, joint_a, joint_b, *, keys)` — add equality
  constraints between inertial parameters of two symmetric links.
- `bounds_from_urdf(model, ...)` — derive mass and CoM bounds automatically
  from Pinocchio model URDF inertials (CAD data source strategy).
- `build_cad_constraints_from_config(cfg, *, model)` — build `CADConstraints`
  from a YAML config sub-dict; returns `None` when empty (safe default-off).
- `apply_cad_constraints_to_problem(problem, theta, params_r, cad)` — inject
  CAD constraints into a picos `Problem` for the full-robot SDP.
- `feat(physical_consistency)`: `project_p10_lmi` gains optional `mass_bounds`
  and `com_bounds` kwargs — per-link box constraints on the SDP.
- `feat(physical_consistency)`: `project_robot_p10_lmi` gains optional
  `cad_constraints` kwarg — passes per-link bounds to `project_p10_lmi`.
- `feat(reconstruction)`: `_reconstruct_sdp` and `reconstruct_full_parameters`
  gain optional `cad_constraints` kwarg — injects CAD constraints after the
  per-joint LMI loop in the full-robot SDP.
- `feat(base_identification)`: Both `_apply_physical_consistency_if_enabled`
  and `_apply_reconstruction_if_enabled` hooks parse `cad_constraints` from
  config and pass the result to the underlying solvers.
- Example YAML comment block for `cad_constraints` in
  `figaroh-examples/examples/templates/manipulator_robot.yaml`.

### Changed

- All new parameters are optional with safe defaults — fully backward-compatible.

## [0.4.2] - Unreleased

### Added

- `feat(reconstruction)`: `ReconstructionResult.status` — outcome code (`"ok"`,
  `"solver_missing"`, `"error"`) on every result
- `feat(reconstruction)`: `ReconstructionResult.base_residual_norm` — L2 norm
  of the base constraint residual `M θ − φ_base` (scalar, always set)
- `feat(reconstruction)`: `ReconstructionResult.objective` — SDP objective value
  for Option B (`None` for nullspace)
- `feat(reconstruction)`: `BaseResult` frozen dataclass — structured container
  for `(M, phi_base, params_r)` as input to `reconstruct_full_parameters()`
- `feat(reconstruction)`: `reconstruct_full_parameters()` — unified entry point
  with `method="nullspace" | "sdp" | "auto"` dispatch
- `feat(reconstruction)`: Option B SDP (`method="sdp"`) — minimises
  `‖diag(w)(θ−θ₀)‖²` subject to `M θ = φ_base` and per-joint pseudo-inertia
  LMI `P_j ≽ 0` via Schur-complement epigraph (requires picos + cvxopt/mosek)
- `feat(reconstruction)`: `method="auto"` — silently falls back to nullspace
  when picos is not installed; `status="solver_missing"` when picos unavailable
  and `method="sdp"` was requested explicitly
- `feat(reconstruction)`: `_load_prior_from_urdf()` — builds flat prior dict
  from Pinocchio model inertias (`prior_source="urdf"`)
- `feat(reconstruction)`: `_load_prior_from_yaml()` — loads prior from a flat
  YAML file (`prior_source="yaml"`)
- `feat(identification)`: `_apply_reconstruction_if_enabled()` hook in
  `BaseIdentification._store_results()` — called after physical consistency;
  stores result under `self.result["reconstruction"]`
- `feat(identification)`: `_calculate_base_parameters()` now uses
  `QRDecomposer` directly to expose `self._M_matrix` / `self._params_r_for_recon`
  for downstream reconstruction; result dict includes `"M"` and `"params_r"` keys
- `feat(config)`: `reconstruction` block parsed from both legacy YAML
  (`get_param_from_yaml`) and unified config (`unified_to_legacy_identif_config`)
- `feat(identification/__init__)`: `BaseResult` and `reconstruct_full_parameters`
  added to public exports and `__all__`
- `tests`: 13 new unit tests in `tests/unit/test_reconstruction.py` covering
  new fields, BaseResult, prior helpers, `_p10_indices_for_joints`, nullspace
  end-to-end, auto fallback, YAML prior, and unsupported method error

### Fixed

- `fix(reconstruction)`: alternation loop in `run_reconstruction` now correctly
  unpacks `project_robot_p10_lmi()` as `(projected_p10_dict, robot_report)`;
  removes `AttributeError: tuple has no attribute 'p10_by_joint'`
- `fix(identification)`: `"projected parameters"` key (space) renamed to
  `"projected_parameters"` (underscore) in `_apply_physical_consistency_if_enabled`
  result dict for consistency with `"raw_parameters"` and Python conventions
- `fix(tools/robotcollisions)`: `print_collision_pairs()` uses `print()` instead
  of `logger.info()` so output is visible in interactive use and captured by tests
- `fix(tools/solver)`: `LinearSolver._print_solution_info()` uses `print()`
  instead of `logger.info()` so verbose output is visible in interactive use
  and captured by tests

## [0.4.1] - Unreleased

### Added

- `feat(physical-consistency)`: `is_feasible_link()` — public alias for
  `check_p10_feasibility()` matching roadmap spec naming
- `feat(physical-consistency)`: `project_link()` — public alias for
  `project_p10_lmi()` matching roadmap spec naming
- `feat(physical-consistency)`: `ProjectionReport.runtime` — per-link solve
  time (seconds) recorded via `time.perf_counter()` around `problem.solve()`
- `feat(physical-consistency)`: `weights` keyword argument to
  `project_robot_p10_lmi()` for passing a 10-element weight vector to all
  per-link projection calls
- `feat(config)`: `weights.mode: "auto" | "manual"` parsed from the
  `physical_consistency` YAML block in `_apply_physical_consistency_if_enabled`
- `feat(config)`: `weights.manual.{m, h, Sigma}` per-group manual weights built
  into a 10-element array and forwarded to `project_robot_p10_lmi`
- `feat(identification)`: `physical consistency` result dict now stores both
  `raw_parameters` (pre-projection) and `projected_parameters` (post-projection)
  as separate keys, preserving the original identified values for comparison
- `tests`: 28 unit tests in `tests/unit/test_physical_consistency.py` covering
  TC-1 through TC-12 (projection correctness, aliases, runtime, robot aggregation)
  plus 3 config-wiring tests for weights and raw/projected separation

### Fixed

- `fix(physical-consistency)`: SDP formulation in `project_p10_lmi` no longer
  uses the picos 2.x-incompatible `pc.vstack`, `pc.multiply`, or `pc.sum_squares`
  functions; replaced with element-wise objective using `pc.SquaredNorm` and
  explicit sigma-entry loop
- `fix(physical-consistency)`: solver keyword `verbose=` replaced with
  `verbosity=` (picos 2.x API); eliminates `DeprecationWarning` on every call
- `fix(physical-consistency)`: feasibility check after projection uses a
  relaxed `psd_eig_tol=-1e-8` tolerance to absorb inevitable SDP solver
  numerical noise, preventing valid projections from being reported as
  `"infeasible"` due to tiny eigenvalue violations (~1e-9)

## [0.3.1] - 2026-04-01

### Added
- `feat(physical-consistency)`: optional projection of inertial parameters onto
  a physically consistent set (`identification/physical_consistency.py`)
- `feat(reconstruction)`: reconstruction utilities — `run_reconstruction` and
  `run_option_a_reconstruction` entry points (`identification/reconstruction.py`)
- `feat(qrdecomposition)`: `get_M` / `get_M_labels` methods for retrieving the
  stored base mapping matrix after decomposition
- `feat(qrdecomposition)`: `QRResult` dataclass for structured, full-precision
  decomposition output (rank, base_indices, W_b, beta, M, phi_b, diag_R,
  cond_R1, method, …)
- `feat(qrdecomposition)`: `relative_tolerance` constructor parameter for
  scale-invariant rank detection (threshold relative to largest pivot)
- `feat(qrdecomposition)`: `get_diagnostics()` method returning rank, diag_R,
  cond_R1, and method after every decomposition
- `feat(qrdecomposition)`: `decompose()` unified entry point returning a
  `QRResult`; delegates to pivoting or double path based on `method` argument
- `feat(identification)`: enhanced parameter management with standard and
  custom parameter support in `BaseIdentification`

### Fixed
- `fix(qrdecomposition)`: `_find_rank` now correctly returns 0 for zero/empty
  matrices (previously returned row count, causing phantom base parameters)
- `fix(reconstruction)`: add missing `run_option_a_reconstruction` alias that
  blocked package-level import

### Changed
- `refactor(qrdecomposition)`: replace non-pivoted `numpy.linalg.qr` with
  `scipy.linalg.qr(..., pivoting=True)` in `_identify_base_parameters` —
  permutation-stable, deterministic basis selection
- `refactor(qrdecomposition)`: remove premature `np.around` from `beta` in all
  code paths; rounding deferred to display-only `_build_parameter_expressions`
- `refactor(qrdecomposition)`: remove fragile `assert np.allclose(W_base, W_b)`
- `refactor(qrdecomposition)`: enhanced docstrings and nominal-parameter handling
- `refactor(identification)`: patch filter configuration management in
  `BaseIdentification`
- `refactor(calibration)`: update deprecated `Frame.parent` →
  `Frame.parentJoint` for Pinocchio 3.x compatibility
- `refactor(logging)`: replace print statements with `logging` calls across
  calibration and identification modules
- `refactor(config)`: rename `load_from_yaml` → `load_param` for clarity

### Tests
- `test(qrdecomposition)`: `TestNumericalImprovements` suite — 14 new tests,
  23/23 total pass (rank-zero, relative tolerance, permutation stability ×20,
  mapping matrix property, column space, diagnostics, QRResult structure,
  full-precision beta)
- `test(reconstruction)`: unit tests for reconstruction utilities

---

## [0.3.0] - 2025-12-09

### Added
- **Advanced Linear Solver (`figaroh.tools.solver`)**: Comprehensive multivariate linear solver for robot parameter identification
  - Multiple solving methods: lstsq, QR, SVD, Ridge, Lasso, Elastic Net, Tikhonov, constrained, robust, weighted
  - Regularization support: L1 (Lasso), L2 (Ridge), Elastic Net, custom Tikhonov
  - Constraint handling: Box constraints (bounds), linear equality/inequality constraints
  - Robust regression with iterative reweighting for outlier resistance
  - Comprehensive solution quality metrics (RMSE, R², condition number, residuals)
  - Optimized for dense, large, thin matrices typical in robot dynamics
  - Full unit test coverage (18 tests) with robot identification scenarios

- **Module Reorganization**: Better code organization and separation of concerns
  - **Calibration module restructuring**:
    - `calibration/config.py`: Configuration parsing and YAML handling (624 lines)
    - `calibration/parameter.py`: Parameter management utilities (240 lines)
    - `calibration/data_loader.py`: Data loading and I/O operations (160 lines)
  - **Identification module restructuring**:
    - `identification/config.py`: Configuration parsing for identification (334 lines)
    - `identification/parameter.py`: Parameter management for identification (388 lines)
  - Maintains 100% backward compatibility through re-exports

- **BaseIdentification Enhancement**:
  - `solve_with_custom_solver()`: New method using advanced linear solver with regularization and constraints
  - Flexible solving with multiple methods and custom constraints
  - Support for physical parameter bounds (e.g., positive masses/inertias)

### Improved
- **Parameter Ordering**: Changed to Pinocchio dynamic parameter ordering for consistency
  - New order: [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, m]
  - Previous order: [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]

- **Regressor Module**: Cleaned up build_basic_regressor methods
  - Removed unused `tau` parameter for better API clarity
  - Improved method signatures and documentation

- **Code Quality**: Significant reduction in code duplication
  - `calibration_tools.py`: Reduced from ~1500 to ~630 lines (-58%)
  - `identification_tools.py`: Reduced from ~900 to ~295 lines (-67%)
  - Modular design with clear single responsibilities

### Fixed
- Parameter naming: Changed from numbered indices to parent joint names for clarity
- Regressor handling: Better support for additional columns in regressor matrices
- Results manager imports and formatting issues

### Technical Details
- **Files Changed**: 21 files
- **Lines Added**: +3,372
- **Lines Removed**: -1,604
- **Net Change**: +1,768 lines
- **Test Coverage**: All 18 new solver tests passing

## [0.2.4] - 2025-09-08

### Changed
- **Optional Dependencies**: Removed `cyipopt` from required dependencies
  - cyipopt is now truly optional and loaded only when IPOPT optimization is used
  - Users can install without cyipopt and still use all other features
  - Install cyipopt separately when needed: `pip install cyipopt` or via conda environment

### Improved
- **Installation Flexibility**: Package now installs without requiring heavy optimization dependencies
- **Error Handling**: Better error messages when optional dependencies are missing

## [0.2.3] - 2025-09-08

### Added
- **Streamlined Dependencies**: All core dependencies now available via PyPI with automatic installation
- **Lazy Loading**: Optional dependencies (cyipopt) now loaded only when needed to improve startup time
- **Enhanced Installation Notes**: Clear documentation of simplified dependency management

### Improved
- **Dependency Management**: Complete cleanup and optimization of package dependencies
  - Removed redundant `requirements.txt` and `setup.py` files
  - Consolidated all dependencies in `pyproject.toml`
  - Updated to use PyPI versions of robotics libraries (`pin` for Pinocchio)
- **Installation Process**: Significantly simplified installation with better cross-platform compatibility
- **Documentation**: Comprehensive README updates reflecting modern packaging standards
  - Combined development installation methods for clarity
  - Added official Pinocchio repository reference
  - Updated dependency documentation with descriptions
- **Performance**: Faster module loading through localized imports
- **Environment Setup**: Streamlined conda environment with minimal dependencies

### Enhanced
- **Import Strategy**: Localized cyipopt import to specific functions for better error handling
- **Error Messages**: More informative import error messages with installation instructions
- **Package Structure**: Modern Python packaging standards with pyproject.toml-only approach

### Removed
- **Redundant Files**: Eliminated `requirements.txt` and `setup.py` in favor of modern `pyproject.toml`
- **Unnecessary Dependencies**: Cleaned up unused dependencies for leaner installation

### Fixed
- **Package Name**: Corrected dependency references (e.g., proper use of `pin` for Pinocchio PyPI version)
- **Installation Conflicts**: Resolved potential conflicts between conda and pip installations

## [0.2.0] - 2025-09-05

### Added
- **Unified Configuration System**: Complete overhaul of configuration management
  - New `UnifiedConfigParser` with YAML template inheritance
  - Automatic format detection for seamless legacy compatibility
  - Advanced parameter mapping between configuration formats
  - Comprehensive configuration validation with helpful error messages

- **Enhanced Base Classes**: Modern object-oriented workflow management
  - `BaseCalibration`: Standardized calibration workflow with unified config support
  - `BaseIdentification`: Standardized identification workflow with unified config support
  - Automatic configuration format detection and conversion

- **Advanced Regressor Builder**: Complete redesign of regressor computation
  - `RegressorBuilder`: Object-oriented, extensible regressor construction
  - `RegressorConfig`: Configuration dataclass for regressor parameters
  - Enhanced input validation and error handling
  - Support for joint torque and external wrench modes

- **Configuration Format Mapping**: Seamless format conversion utilities
  - `unified_to_legacy_config`: Calibration parameter mapping function
  - `unified_to_legacy_identif_config`: Identification parameter mapping function
  - Perfect compatibility with existing legacy configurations

### Improved
- **Parameter Processing**: Enhanced parameter handling with better defaults
- **Error Messages**: More informative validation and error reporting
- **Documentation**: Comprehensive updates to README and module documentation
- **Code Organization**: Better structured modules with clear responsibilities
- **Type Safety**: Added type hints throughout the codebase

### Enhanced
- **Cross-Platform Support**: Improved compatibility across operating systems
- **Input Validation**: Robust parameter validation and type checking
- **Template System**: Flexible configuration template inheritance
- **Backward Compatibility**: Full support for existing legacy configurations

### Removed
- **quadprog dependency**: Removed unused quadprog dependency to reduce package size

### Fixed
- **Configuration Parsing**: Resolved edge cases in YAML parsing
- **Parameter Mapping**: Accurate conversion between configuration formats
- **Validation Logic**: Improved configuration validation accuracy
- **Error Handling**: Better error recovery and user feedback

### Technical Improvements
- Modern Python practices with dataclasses and type hints
- Enhanced error handling with custom exception classes
- Improved testing framework with comprehensive validation
- Better code documentation and examples

### Documentation
- Updated README with modern API examples
- Enhanced configuration system documentation
- New API usage patterns and best practices
- Comprehensive module documentation updates

## [0.1.0] - 2025-01-25

### Added
- Initial release of FIGAROH package
- Dynamic identification algorithms for rigid multi-body systems
- Geometric calibration algorithms for serial and tree-structure robots
- Support for URDF modeling convention
- Optimal trajectory generation for dynamic identification
- Optimal posture generation for geometric calibration
- Integration with Pinocchio for efficient computations
- Support for various optimization algorithms
- Data filtering and pre-processing utilities
- Model parameter update utilities

### Features
- **Dynamic Identification**:
  - Dynamic model including friction, actuator inertia, and joint torque offset
  - Continuous optimal exciting trajectory generation
  - Multiple parameter estimation algorithms
  - Physically consistent standard inertial parameters calculation

- **Geometric Calibration**:
  - Full kinematic parameter calibration
  - Optimal calibration posture generation via combinatorial optimization
  - Support for external sensors (cameras, motion capture)
  - Non-external methods (planar constraints)

### Dependencies
- Core scientific computing: numpy, scipy, matplotlib, pandas
- Robotics: pinocchio (via conda)
- Optimization: cyipopt (via conda)
- Visualization: meshcat
- Additional: numdifftools, ndcurves, rospkg

### Documentation
- Comprehensive README with installation and usage instructions
- Examples moved to separate repository (figaroh-examples)
- API documentation structure prepared

### Notes
- Examples and URDF models moved to separate repository for clean package distribution
- Package optimized for PyPI distribution
- Supports Python 3.8+
