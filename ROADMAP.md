# FIGAROH Combined Roadmap

**Document Version:** 1.4 (Track B Step 5 complete — core algorithm path migrated)
**Date:** June 19, 2026
**Current Release:** `figaroh` 0.4.3 (PyPI)
**Status:** v0.4.x algorithmic features largely landed; Track B Phase 1 backend abstraction functional and core algorithm path migrated (identification + calibration route through PinocchioBackend)

> **Single source of truth.** This document merges the private algorithmic roadmap
> (version-tracked features: physical consistency, reconstruction, online ID, etc.)
> with the public multi-simulator integration roadmap (backend abstraction,
> Pinocchio/MuJoCo/Genesis/IsaacSim). The two tracks are orthogonal but intersect
> at v1.0 (API stabilization requires the backend abstraction to be real) and at
> `figaroh-examples` (both tracks need examples).
>
> **Accuracy note:** Status markers below are reconciled against the actual codebase
> as of June 2026, not the aspirational claims in the former `ARCHITECTURE.md` /
> `IMPLEMENTATION_ROADMAP.md`. Where a prior doc claimed "✅ completed" but the
> code does not exist, this document marks it ❌ or 🔄.

---

## 1. Vision

**Transform FIGAROH into a universal, simulator-agnostic calibration and
identification toolbox** that enables researchers and engineers to:

- Identify **physically feasible** full per-link inertial parameters — not just
  reduced base-parameter sets — suitable for URDF export, simulation, and control.
- Switch between dynamics backends (Pinocchio, MuJoCo, Genesis, IsaacSim) with
  minimal code changes, leveraging each simulator's strengths while maintaining
  consistent algorithms.
- Integrate seamlessly into existing workflows with straightforward, step-by-step
  APIs and comprehensive documentation.
- Achieve production-grade reliability with comprehensive tests, CI, and examples.

---

## 2. Strategic Principles

1. **Leverage, Don't Reinvent**
   - Use existing simulator converters instead of building custom ones.
   - Build on FIGAROH's mature algorithms; extend, don't fork.
   - Integrate shared infrastructure from `figaroh-examples` where it belongs.

2. **Abstraction Over Implementation**
   - Define clean interfaces for dynamics backends.
   - Enable pluggable simulator implementations.
   - Maintain algorithm consistency across backends.

3. **Do Not Break Existing Workflows**
   - New functionality is **opt-in** through config flags and/or new classes until v1.0.
   - Separate concerns: (A) identify parameters, (B) enforce physical feasibility,
     (C) export back to URDF/YAML + visualize/validate.
   - Prefer small, testable building blocks over monolithic drops.

4. **User Experience First**
   - Prioritize ease of use and clear documentation.
   - Provide step-by-step workflows; getting started in <5 minutes.
   - Minimize learning curve for new users and new simulators.

5. **Incremental, Validated Progress**
   - Deliver working features incrementally.
   - Validate each backend and feature thoroughly.
   - Maintain backward compatibility.

---

## 3. Current State (Verified Snapshot, June 19, 2026)

> Status below is verified against the actual codebase (file existence, test runs,
> grep checks), not prior doc claims. Test run: `pytest --co` = 214 collected;
> `pytest` = 208 passed, 4 failed, 2 skipped.

### 3.1 FIGAROH Core (`figaroh/`, v0.4.3)

**Algorithmic features — solid and actively developed:**
- Mature identification (regressor-based, QR base parameters, 10+ linear solver methods)
- Geometric calibration (multi-sensor, optimal postures)
- Optimal trajectory generation (IPOPT)
- **v0.4.1** LMI physical-consistency projection ✅ (`identification/physical_consistency.py`, tested)
- **v0.4.2** Base → full parameter reconstruction ✅ (`identification/reconstruction.py`, nullspace + SDP)
- **v0.4.3** CAD-informed constraints ✅ (`identification/cad_constraints.py`, released 2026-06-02)
- **Reporting & verification (V&V) suite** ✅ — HTML diagnostic reports, machine-readable
  pass/fail verdicts (`verify()`/`export_verification_report()`), an interactive
  before/after panel, and a static two-run compare page
  (`tools/compare_report.py`). Tracked separately from this roadmap's Track A/B —
  full design history and status in
  [`docs/decisions/roadmap-mujoco-sysid-inspired-features.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/roadmap-mujoco-sysid-inspired-features.md).

**Test suite (verified by running `pytest`):**
- 259 tests collected across 11 files in `tests/unit/`:
  `test_solver.py`, `test_qr_decomposition.py`, `test_reconstruction.py`,
  `test_physical_consistency.py`, `test_cad_constraints.py`, `test_regressor.py`,
  `test_robotvisualization.py`, `test_robotcollisions.py`, `test_robotipopt.py`,
  `test_backends.py` (45 tests: 32 Pinocchio + 13 MuJoCo)
- **253 passed, 4 failed, 2 skipped.**
- The 4 failures are all in `test_robotipopt.py` (IPOPT solver tests) — pre-existing.
- ⚠️ Prior docs claimed "198/198 pass" — inaccurate on both counts (count and pass rate).
- `test_cad_constraints.py` exists but was not mentioned in the old roadmap checklist.

**Backend abstraction — functional (PinocchioBackend + MuJoCoBackend implemented):**
- `DynamicsBackend` abstract interface (`backends/base.py`): ✅ exists, extended with 6 optional model-introspection methods (`get_inertias`, `get_frame_id`, `compute_difference`, `compute_integrate`, `random_configuration`, `get_model_object`)
- `PinocchioBackend` (`backends/pinocchio.py`): ✅ **implemented** (371 lines) — wraps `pin.crba`, `pin.computeCoriolisMatrix`, `pin.computeGeneralizedGravity`, `pin.rnea`, `pin.aba`, `pin.computeJointTorqueRegressor`, `pin.computeFrameJacobian`, FK, plus all optional Lie group methods. Includes `from_model()` classmethod for wrapping existing `pin.Model`/`pin.Data` without URDF re-loading. Factory `get_backend("pinocchio")` now works.
- `MuJoCoBackend` (`backends/mujoco.py`): ✅ implemented and functional — dynamics (mass matrix, Coriolis, gravity, inverse/forward dynamics), kinematics (FK, Jacobian), regressor. ⚠️ Regressor delegates to Pinocchio's analytical `computeJointTorqueRegressor` (MuJoCo doesn't support runtime inertial parameter perturbation for finite differences). Coriolis matrix uses finite-difference Jacobian/2 approach (Euler's theorem for quadratic homogeneous functions). Mass matrix fixed to call `mj_forward` before `mj_crb`.
- `GenesisBackend`: ❌ not started
- `IsaacSimBackend`: ❌ not started
- `integration/` directory: ❌ does not exist (high-level API not started)
- `cli/` directory: ❌ does not exist (CLI tool not started)
- Backend test suite (`tests/unit/test_backends.py`): ✅ 45 tests (32 Pinocchio + 13 MuJoCo) covering factory, initialization, numerical correctness, kinematics, regressor self-consistency, Coriolis consistency, optional methods, interface conformance.
- **Core module migration (Step 5):** `Robot` class has lazy `backend` property (wraps existing model via `PinocchioBackend.from_model()`). Full algorithm path migrated:
  - **Identification:** `RegressorBuilder` routes through `robot.backend.compute_regressor()`; `identification_tools.py` accepts optional `backend` for `compute_difference()`; `randomdata.py` routes through `robot.backend.compute_inverse_dynamics()`.
  - **Calibration:** `calibration_tools.py` — 7 functions migrated (`get_rel_kinreg`, `get_rel_jac`, `update_forward_kinematics`, `calc_updated_fkm`, `calculate_kinematics_model`, `calculate_identifiable_kinematics_model`, `calculate_base_kinematics_regressor`) with optional `backend=None` parameter routing FK, Jacobian, gravity, and random-configuration through backend. `computeFrameKinematicRegressor` uses `get_model_object()` escape hatch (no backend equivalent yet).
  - All changes are "strangler fig" pattern — new path when backend available, old `pin.*` path as fallback. 6 files modified, +268/-79 lines.

**Tooling/infra:**
- `robot_format_converter`: deprecated (notice + migration guide done; GitHub
  archiving pending July 1, 2026)
- `figaroh-mujoco`: deprecated & merged into core as `backends/mujoco.py`
  (archiving pending July 1, 2026)
- CI builds docs only (`.github/workflows/docs.yml`); **no test/lint CI workflow**.
  Quality gate = `pre-commit run --all-files`.
- Dev branch: `devel`; release branch: `main`.

### 3.2 FIGAROH-Examples (`figaroh-examples/`)

**Robot examples (verified real-vs-partial):**

| Example | Status | Scripts present | Notes |
|---|---|---|---|
| `ur10/` | ✅ Full | calibration, identification, optimal_config, optimal_trajectory, update_model | All real (52–122 lines each); utils ~1656 lines; 6 config YAMLs; 4 data CSVs |
| `tiago/` | ✅ Full | calibration, identification, optimal_config, optimal_trajectory | No `update_model.py`; utils ~7482 lines (incl. 7242-line collision model); 8 config YAMLs; 7 URDF variants; rich data |
| `talos/` | ⚠️ Partial | calibration_upperbody, update_model | Missing identification, optimal_config, optimal_trajectory; utils ~3876 lines |
| `staubli_tx40/` | ⚠️ Partial | identification | Missing calibration, optimal_config, optimal_trajectory, update_model; utils ~20306 lines; 2 large data CSVs (2.1–2.3MB) |
| `templates/` | ⚠️ Scaffold | — | `base_robot_config.yaml` (221L), `manipulator_robot.yaml` (43L), `humanoid_robot.yaml` (71L). Starter templates only, not exercised examples. |

**`examples/shared/` — ❌ DOES NOT EXIST.**
The former `FIGAROH_ECOSYSTEM_ANALYSIS.md` described 9 shared files in detail
(base_calibration.py, base_identification.py, config_manager.py, etc.) but this
directory is absent. Each robot example has its own `utils/` folder instead.
All references to `shared/` infrastructure in prior docs are stale.

**`web-interface/` — ✅ REAL WORKING APP (not a scaffold).**
- Built on **Viser** (3D web framework), not the React/Three.js stack the old
  Phase 4 plan described.
- 21 Python files, ~20K+ lines total:
  - `app.py` (206L), `main.py` (58L, CLI entry with argparse)
  - `components/` (8 files, ~4209L): setup_panel, robot_panel, task_panel, data_panel,
    config_panel, results_panel, visualization_panel, streamlined_task_panel
    (results_panel and visualization_panel are thin stubs; rest substantial)
  - `core/` (6 files): interface.py (744L), streamlined_interface.py (11448L),
    robot_manager.py (378L), task_manager.py, example_loader.py (95L), session_manager.py
  - `utils/file_utils.py` (3919L)
- Features: interactive 3D viz, multiple robot loaders (figaroh/robot_description/yourdfpy),
  task management with threading, progress tracking, session management.
- Marked "⚠️ Under development" in its README.

**Simulator integration examples — ❌ ALL MISSING.**
`examples/mujoco_integration/`, `genesis_integration/`, `isaacsim_integration/`
do not exist.

**End-to-end example (ID → projection → reconstruction → URDF export) — ❌ MISSING.**
No single script chains these steps. Individual scripts exist per robot, and
`update_model.py` (ur10, talos) handles URDF/xacro updating, but no script
combines identification output → physical projection → reconstruction → URDF export.

**CAD-constraints example — ❌ MISSING.**
Grep across all of `figaroh-examples` for `cad_constraint`, `physical_consistency`,
`reconstruction` returns **zero matches**. The v0.4.3 CHANGELOG claim "Example YAML
comment block for cad_constraints in figaroh-examples/examples/templates/manipulator_robot.yaml"
is **false** — that file is 43 lines with no CAD block. No config anywhere exercises
CAD constraints, physical consistency, or reconstruction.

**`tests/` — ✅ REAL.**
9 files, ~3684 lines: `run_tests.py`, `test_backward_compatibility.py` (634L),
`test_robot_configs.py` (659L), `test_robot_integration.py` (536L),
`test_config_integration.py` (391L), `test_unified_config_parser.py` (470L),
`test_error_handling.py` (209L), `test_ur10_backward_compatibility.py` (259L),
`test_tiago_backward_compatibility.py` (302L), `conftest.py`.

**`create_example.sh` — ✅ REAL (934 lines).**
Scaffolds a new robot example folder from the TIAGO template.

---

## 4. Track A — Algorithmic Features

> Source: former `ROADMAP_PRIVATE.md`. Version-tracked, opt-in, non-breaking until v1.0.
> Problem context: classic inverse-dynamics identification yields a reduced/base
> parameter set because the regressor is rank-deficient; users want full per-link
> physical parameters (mass, CoM, inertia) that are **physically feasible**
> ($m>0$, valid/PSD inertia). Promising direction: Lee et al. (2020)-style convex
> identification using spatial inertia + convex feasibility (SDP/LMI).

### v0.4 (Current Minor) — Physical Consistency & Reconstruction

#### v0.4.1 — LMI Physical-Consistency Projection ✅ DONE

Opt-in post-processing stage: given a full per-link inertial set (10D `p10` per
link), return the closest physically feasible set.

- **Module:** `figaroh.identification.physical_consistency`
- **API:**
  - `is_feasible_link(p10, ...) -> FeasibilityReport` (alias of `check_p10_feasibility`)
  - `project_link(p10, weights, mass_min, solver, tol) -> (p10_proj, ProjectionReport)` (alias of `project_p10_lmi`)
  - `project_robot_p10_lmi(params_by_link, ...) -> (projected_by_link, RobotProjectionReport)`
- **Constraints (per-link convex SDP):**
  - C1: $m \ge m_{\min}$ (linear)
  - C2: $P(p_i) \succeq 0$ (pseudo-inertia 4×4 PSD, via `pinocchio.PseudoInertia.FromDynamicParameters`)
- **Objective:** $\min_{p_i} \| W(p_i - \hat{p}_i) \|_2^2$ s.t. C1, C2
- **Config:** `identification.physical_consistency.{enabled, method, per_link, mass_min, weights.{mode,manual}, solver.{backend,verbose,max_seconds}, tolerances.psd_eig}`
- **Solver:** `picos` + CVXOPT (open-source) or MOSEK (preferred). Clear `ImportError` when backend missing.
- **Tests:** TC-1 through TC-6 pass in `test_physical_consistency.py` (feasible-unchanged, Pinocchio round-trip, negative-mass corrected, indefinite-inertia corrected, weighting sanity, backend-missing error).
- **Minor gaps:** `solver.max_seconds` not yet mapped from YAML; export-to-URDF path not yet using projected inertials.

#### v0.4.2 — Base → Full Parameter Reconstruction ✅ DONE

Given identified base parameters, return a specific full-parameter set by selecting
a point in the equivalence class.

- **Module:** `figaroh.identification.reconstruction`
- **Entry point:** `reconstruct_full_parameters(base_result, *, method="nullspace"|"sdp"|"auto", prior="urdf"|"yaml"|dict, config) -> (parameter_dict_full, report)`
- **Base mapping:** $\phi_{\text{base}} \approx M\,\theta_{\text{full}}$; $M$ retrieved via `QRDecomposer.get_M()` / `get_M_labels()`. `BaseResult` frozen dataclass carries $(M, \phi_{\text{base}}, \text{params}_r)$.
- **Option A (nullspace, fast):** $\hat\theta_0 = \arg\min \|M\theta - \phi_{\text{base}}\|^2$; parameterize via nullspace $N$; regularize toward prior $\theta_{\text{prior}}$. Alternating loop with v0.4.1 projection for feasibility.
- **Option B (SDP, convex):** $\min \|W(\theta - \theta_{\text{prior}})\|^2$ s.t. $M\theta = \phi_{\text{base}}$ and per-joint $P(p_{10,j}) \succeq 0$ via Schur-complement epigraph. Feasibility by construction.
- **`method: "auto"`:** tries SDP when picos available, falls back silently to nullspace.
- **Report:** `ReconstructionResult` with `status` (`ok|solver_missing|error`), `base_residual_norm`, `objective`, `theta_r_dict`, `params_r`.
- **Config:** `identification.reconstruction.{enabled, method, prior.source, tolerances.{base_residual,psd_eig}, nullspace.{lambda,max_iters}, sdp.{solver,max_seconds,verbose}}`
- **Pipeline:** `_apply_reconstruction_if_enabled()` hook in `BaseIdentification._store_results()` after physical-consistency step.
- **Tests:** 17 tests in `test_reconstruction.py` (prior labels, dict output, prior propagation, BaseResult, residual norm, YAML prior, p10 indices, nullspace end-to-end, auto fallback, unsupported method error). Total suite: 214 collected, 208 passed, 4 failed (IPOPT), 2 skipped.

#### v0.4.3 — CAD-Informed Constraints ✅ DONE (released 2026-06-02)

Convex CAD-informed constraints for inertial parameter identification.

- **Module:** `figaroh.identification.cad_constraints`
- **API:**
  - `CADConstraints` dataclass — per-link mass bounds, first-moment (CoM) bounds, symmetry equalities
  - `add_mass_bounds(cad, joint, *, m_min, m_max)`
  - `add_com_bounds(cad, joint, *, axis, h_min, h_max)`
  - `add_symmetry_constraints(cad, joint_a, joint_b, *, keys)`
  - `bounds_from_urdf(model, ...)` — derive bounds from Pinocchio model URDF inertials
  - `build_cad_constraints_from_config(cfg, *, model)` — from YAML; `None` when empty (safe default-off)
  - `apply_cad_constraints_to_problem(problem, theta, params_r, cad)` — inject into picos `Problem`
- **Integration:** `project_p10_lmi` / `project_robot_p10_lmi` / `_reconstruct_sdp` / `reconstruct_full_parameters` all accept optional `cad_constraints` kwarg. Both `_apply_physical_consistency_if_enabled` and `_apply_reconstruction_if_enabled` parse CAD config.
- **Tests:** `tests/unit/test_cad_constraints.py` exists and passes.
- **All new params optional with safe defaults — fully backward-compatible.**
- ⚠️ **Example gap:** The v0.4.3 CHANGELOG claims "Example YAML comment block for
  cad_constraints in figaroh-examples/examples/templates/manipulator_robot.yaml" —
  this is **false** (file is 43 lines, no CAD block; grep confirms zero matches
  across all of figaroh-examples). No example config exercises CAD constraints.

#### v0.4.x — Cross-Cutting Gates (REMAINING)

- [ ] **One end-to-end example** in `figaroh-examples`: classic ID → physical projection → (optional) reconstruction → URDF export
- [ ] **Docs**: Sphinx page for `physical_consistency` and `reconstruction` APIs with config reference
- [ ] **CHANGELOG entry** for 0.4.x summarizing all new public APIs
- [ ] `picos` backend availability documented in `README.md` and install guide (solver requirements + graceful error when missing)
- [ ] **CI**: test matrix runs with and without optional `picos`/`cvxopt` to verify lazy-import errors surface cleanly
- [ ] `solver.max_seconds` forwarded from config (field exists, not mapped from YAML)
- [ ] Export-to-URDF/YAML path uses projected inertials when enabled (currently overwrites in-place; raw should be preserved alongside projected)

---

### v0.5 — Modular Refactor & Visualization

#### 0.5.1 Modular refactor (internal, no public API break)
Goal: make later convex ID integration clean.
- Extract "identification pipeline stages" into internal helpers:
  - regressor prep / decimation / column elimination
  - "base parameter computation"
  - "parameter packaging (per-link mapping)"
- Keep `BaseIdentification.solve()` behavior unchanged.

#### 0.5.2 Docs & examples
- One focused example: "classic ID → physical projection → URDF export".
- Document solver setup for `picos` backends and graceful fallback when not installed.

#### 0.5.3 Major visualization update (inspection of physicality)
Deliverables:
- Inertia ellipsoid / principal axes visualization
- CoM overlay vs link geometry
- Before/after comparison (raw ID vs projected/reconstructed)

Essential because convex feasibility can still produce surprising inertias without good priors — visual QA is critical.

---

### v0.6 — Online Identification, Friction, FIM-OED

#### 0.6.1 Online identification
- `figaroh.identification.online` with at least one robust baseline:
  - RLS (with forgetting factor)
  - optional sliding-window LS
  - (later) EKF/UKF for combined state+parameter estimation
- Must reuse the same regressor interface already used by offline ID.

#### 0.6.2 New friction models
Behind a clean interface:
- Stribeck (nonlinear)
- dead-zone/backlash as experimental
- keep existing viscous/Coulomb intact

#### 0.6.3 OED refactor: Fisher Information Metrics
- FIM computation utilities and metrics: $\log\det(\text{FIM})$, A-/E-optimality
- Keep existing condition-number objective available for backward compatibility.

---

### v0.7 — Multi-Sensor Calibration & Round-Tripping

#### 0.7.1 Multi-sensor calibration
- Sensor-agnostic residual + Jacobian interface
- At least one new sensor type end-to-end (IMU or AprilTag-based camera) as reference implementation

#### 0.7.2 URDF↔YAML round-tripping
- Export identified parameters to a diff-friendly YAML "model delta"
- Regenerate URDF inertials from YAML with version tagging + provenance metadata

#### 0.7.3 Test suite (real gates, not just examples)
- Unit tests: physical feasibility checks, projection/reconstruction invariants, regressor assembly sanity
- Integration tests: at least one small robot model fixture
- CI matrix (pragmatic): Python 3.10–3.12, macOS+Linux first

---

### v1.0 — API Stabilization & Convex ID

#### 1.0.1 Architecture consolidation (breaking changes allowed)
- Stabilize public APIs: `BaseCalibration`, `BaseIdentification`, OED class(es)
- Move "experimental" features behind clear namespaces
- Migration guide v0.7 → v1.0
- **Backend abstraction must be real by this point** (see Track B) — PinocchioBackend
  wrapping existing direct Pinocchio calls is the minimum gate.

#### 1.0.2 Physically consistent pipeline as first-class
- Users choose: "classic ID (fast)" vs "physically consistent ID (recommended for export)"
- Consistent outputs for URDF export
- Full Lee et al.-style convex identification backend becomes "official" — but only
  after v0.4–v0.7 deliver the necessary primitives (feasibility/projection/reconstruction,
  QA, tests, solver story).

---

## 5. Track B — Multi-Simulator Backend Integration

> Source: former `IMPLEMENTATION_ROADMAP.md`. Phase-based, quarterly timeline.
> **Reality check:** Phase 1 was claimed "✅ COMPLETED" in the prior doc but is
> only partially done — the PinocchioBackend file does not exist and Pinocchio is
> used directly throughout core. Status below is reconciled to actual code.

### Phase 1: Foundation (Q2 2026) — 🔄 MOSTLY COMPLETE

**Objective:** Establish core architecture for multi-simulator support.

| Deliverable | Prior claim | Actual status |
|---|---|---|
| `DynamicsBackend` abstract interface (`backends/base.py`) | ✅ | ✅ exists + 6 optional methods added |
| `PinocchioBackend` (`backends/pinocchio.py`) | ✅ mostly | ✅ **implemented** (371 lines, 32 tests pass) |
| `MuJoCoBackend` (`backends/mujoco.py`) | ✅ mostly | ✅ functional (regressor delegates to Pinocchio analytical; Coriolis via FD Jacobian/2; mass matrix fixed) |
| Backend factory `get_backend()` | ✅ | ✅ works (`pinocchio` + `mujoco` available) |
| Backend test suite (`tests/unit/test_backends.py`) | 🔄 | ✅ 32 tests, all pass |
| Deprecate `robot_format_converter` | ✅ | ✅ notice+guide done; ⏳ GitHub archive pending Jul 1, 2026 |
| Merge `figaroh-mujoco` into core | ✅ | ✅ merged; ⏳ GitHub archive pending Jul 1, 2026 |
| Architecture docs (`ARCHITECTURE.md`) | ✅ | ⚠️ partly aspirational/out of date (per `AGENTS.md`) |
| High-level integration API (`integration/api.py`) | — | ❌ not started |
| Performance benchmarks (Pinocchio vs MuJoCo) | 🔄 | ❌ not started |
| Migrate existing examples to backend abstraction | 🔄 | ❌ not started |
| Migrate core modules to use backend abstraction | — | ✅ done (identification + calibration algorithm path migrated; visualization/collisions/data-types remain Pinocchio-specific by design) |

**Highest-leverage next step:** Implement `PinocchioBackend` by wrapping the
Pinocchio calls already scattered through `tools/`, `calibration/`,
`identification/`. This is the default backend the roadmap claims is done but
isn't, and it's the prerequisite for making the abstraction meaningful.

#### Task 1.2.1: DynamicsBackend Interface — ✅ DONE
- `backends/base.py`: abstract class, 9 abstract methods (mass matrix, Coriolis,
  gravity, FK, Jacobian, regressor, inverse dynamics, forward dynamics),
  3 properties (`nq`, `nv`, `model_format`), context manager support.
- 9 optional methods: `compute_dynamics_derivatives`, `get_joint_names`,
  `get_frame_names`, `get_inertias`, `get_frame_id`, `compute_difference`,
  `compute_integrate`, `random_configuration`, `get_model_object`.

#### Task 1.2.2: PinocchioBackend — ✅ DONE
- **Location:** `figaroh/backends/pinocchio.py` (371 lines)
- Wraps `pin.crba`, `pin.computeCoriolisMatrix`, `pin.computeGeneralizedGravity`,
  `pin.rnea`, `pin.aba`, `pin.computeJointTorqueRegressor`, `pin.computeFrameJacobian`,
  `pin.forwardKinematics`/`pin.updateFramePlacements`, `pin.difference`, `pin.integrate`,
  `pin.randomConfiguration`.
- Supports free-flyer root joint via `free_flyer`/`isFext` kwargs.
- All return values `.copy()`'d to prevent shared-memory mutations.
- 32 tests in `test_backends.py` — numerical correctness verified against direct
  `pin.*` calls at atol=1e-10.
- **Remaining:** Core modules still use direct `pin.*` calls in calibration, visualization, collisions, and model loading (Step 5 — partial migration done, calibration pending).

#### Task 1.2.3: Core Module Migration — ✅ DONE (Algorithm Path)

**Strategy:** "Strangler fig" pattern — `Robot` class gains lazy `backend` property (wraps existing `pin.Model`/`pin.Data` via `PinocchioBackend.from_model()`). Core modules route through `robot.backend` when available, fall back to direct `pin.*` calls otherwise. Fully backward-compatible.

**Migrated (✅ — full algorithm path):**
- `tools/regressor.py` — `RegressorBuilder` uses `robot.backend.compute_regressor()`, `robot.backend.nv`, `robot.backend.get_inertias()` when backend available
- `identification/identification_tools.py` — `calculate_first_second_order_differentiation` accepts optional `backend` parameter for `compute_difference()`
- `tools/randomdata.py` — `get_torque_rand` uses `robot.backend.compute_inverse_dynamics()`, `generate_waypoints`/`generate_waypoints_fext` use backend-aware `nq`/`nv`
- `calibration/calibration_tools.py` — 7 functions migrated with optional `backend=None`: `get_rel_kinreg`, `get_rel_jac`, `update_forward_kinematics`, `calc_updated_fkm`, `calculate_kinematics_model`, `calculate_identifiable_kinematics_model`, `calculate_base_kinematics_regressor`. Routes FK, Jacobian, gravity, random-configuration through backend. `computeFrameKinematicRegressor` uses `get_model_object()` escape hatch.

**Intentionally not migrated (Pinocchio-specific by design):**
- `calibration/parameter.py` — `pin.SE3`, `pin.Frame`, `pin.Inertia.Zero()`, `pin.rpy.rpyToMatrix` data type construction + model introspection (`joint.shortname()`). Would require parallel type system.
- `measurements/measurement.py` — `pin.SE3`, `pin.Force`, `pin.Frame` data type construction. No callers in active codebase.
- `tools/robotvisualization.py` — Meshcat/Gepetto visualization is inherently Pinocchio-coupled. Out of scope for backend track.
- `tools/robotcollisions.py` — hppfcl/coal collision detection is Pinocchio-specific. No other simulator has equivalent collision interface.
- `utils/cubic_spline.py` — `pin.rnea` + FK for visualization. Deferred to v0.5.
- Data type construction (`pin.SE3`, `pin.Quaternion`, `pin.Force`, `pin.rpy.rpyToMatrix`, `pin.rpy.matrixToRpy`) — used throughout calibration. Would require a common math library layer (e.g., `scipy.spatial.transform.Rotation`) — under investigation for future backend track expansion.

#### Task 1.3.1: High-Level Integration API — ❌ NOT STARTED
- **Location:** `figaroh/integration/api.py`
- `RobotIdentificationSystem(backend="pinocchio"|"mujoco"|"genesis"|"isaacsim")`
- `from_urdf()` / `from_mjcf()` classmethods
- `identify_parameters(data_path, **config) -> Results` one-line API
- Tutorial notebook + API docs

---

### Phase 2: MuJoCo & Genesis Backends (Q3 2026)

**Objective:** Implement and validate MuJoCo and Genesis backends.

#### 2.1 MuJoCo Backend — 🔄 IMPLEMENTED, VALIDATION PENDING
- `backends/mujoco.py`: ✅ core methods (mass matrix, inverse dynamics, FK, Jacobian, regressor, URDF→MJCF auto-conversion)
- **Remaining:**
  - [ ] All interface tests passing
  - [ ] Performance benchmarks (target: 2-3x faster than Pinocchio)
  - [ ] Format conversion guide (URDF → MJCF)
- **Examples** (`figaroh-examples/examples/mujoco_integration/`):
  1. UR10 identification with MuJoCo (load URDF → MJCF → identify → compare with Pinocchio)
  2. Humanoid calibration with MuJoCo (TALOS, geometric calibration, optimal trajectories)
  3. Contact identification (unique to MuJoCo's contact solver)

#### 2.2 Genesis Backend — ❌ NOT STARTED
- **Location:** `figaroh/backends/genesis.py` (does not exist)
- **Dependencies:** `genesis-world>=0.1.0`
- GPU acceleration, native Python API, USD/MJCF/URDF support
- **Examples** (`figaroh-examples/examples/genesis_integration/`):
  1. Humanoid identification with Genesis (TALOS USD, GPU-accelerated, batch trajectories)
  2. Mobile manipulator calibration (TIAGo, mobile base + arm, Genesis rendering)
  3. Large-scale parallel identification (multiple robots, GPU batch, benchmarks)

#### 2.3 Cross-Backend Validation (Week 21-24)
- **Dynamics consistency:** same robot/config, compare M, C, g across backends; tolerance <0.1%
- **Identification consistency:** same trajectory data/algorithm, compare parameters; tolerance <1%
- **Performance benchmarks:** mass matrix (1000 samples), regressor (1000 samples), full ID (10000 samples)
  - Target: MuJoCo 2-3x, Genesis 5-10x (GPU), IsaacSim 3-5x (GPU) vs Pinocchio
- Automated CI/CD integration, benchmark dashboard, validation report

---

### Phase 3: IsaacSim Backend & Ecosystem (Q4 2026)

**Objective:** Complete simulator coverage and build ecosystem tools.

#### 3.1 IsaacSim Backend — ❌ NOT STARTED
- **Location:** `figaroh/backends/isaacsim.py` (does not exist)
- **Dependencies:** `isaacsim>=2023.1.0`
- USD-based workflow, NVIDIA GPU acceleration, Isaac Lab integration, photorealistic rendering
- **Challenges:** USD format requirement (conversion needed), licensing/installation, Omniverse API paradigm
- **Examples** (`figaroh-examples/examples/isaacsim_integration/`):
  1. Industrial manipulator identification (UR10 USD, URDF→USD converter, photorealistic)
  2. Mobile manipulator in Isaac Lab (TIAGo, RL integration, sim-to-real)
  3. Multi-robot scenario (shared environment, parallel identification, GPU dynamics)

#### 3.2 Unified Documentation (Week 33-36)
- **Location:** `figaroh/docs/`
- Structure: `getting-started/`, `simulators/{pinocchio,mujoco,genesis,isaacsim}.md`, `tutorials/{industrial,humanoid,mobile,custom}.md`, `api-reference/{backends,identification,calibration,optimal}.md`, `migration/{from-pinocchio-only,choosing-simulator,performance-guide}.md`
- Sphinx or MkDocs, auto-generated API reference, tutorials with code, migration guides

#### 3.3 Tooling & Developer Experience (Week 37-40)
- **CLI** (`figaroh/cli/`, Click): `figaroh identify`, `optimize-trajectory`, `calibrate`, `backends list/info`, `convert`
- **VS Code extension** (optional, P3): syntax highlighting, IntelliSense, integrated terminal, result visualization

---

### Phase 4: Ecosystem & Advanced Features (2027+)

#### 4.1 Web Interface (Q1 2027)
- Drag-and-drop robot models, interactive trajectory design, real-time identification progress, result dashboard, report export
- **Note:** `figaroh-examples/web-interface/` is already a **real working Viser-based app**
  (21 files, ~20K lines, 3D visualization, task management, multiple robot loaders) —
  see §3.2. Marked "under development" but substantially built. Phase 4 work is
  hardening, completing stub panels (results/visualization), and productionizing
  rather than greenfield.
- Original plan called for FastAPI + React/Three.js; actual implementation uses
  **Viser**. Reconcile the tech-stack decision.

#### 4.2 ROS 2 Integration (Q1-Q2 2027)
- Real-time identification nodes, service interfaces, topic-based data streaming, launch files, MoveIt 2 integration

#### 4.3 ML-Enhanced Identification (Q3-Q4 2027)
- Neural network regressors, physics-informed neural networks, hybrid models (physics + ML), uncertainty quantification, transfer learning across robots

#### 4.4 Cloud Deployment (Q4 2027)
- AWS/GCP/Azure, Kubernetes orchestration, distributed identification, multi-tenant, API management

---

## 6. Cross-Cutting Work

### Documentation
- [ ] Sphinx docs site (Phase 3, but `physical_consistency`/`reconstruction` pages needed sooner — v0.4.x gate)
- [ ] API reference auto-generation
- [ ] 10+ comprehensive tutorials
- [ ] Migration guides (from Pinocchio-only, choosing simulator, performance)
- [ ] `picos` solver-availability docs in README + install guide (v0.4.x gate)

### Examples (`figaroh-examples`)
- [ ] v0.4.x end-to-end example: classic ID → projection → reconstruction → URDF export (❌ missing)
- [ ] CAD-constraints example config (❌ missing — v0.4.3 CHANGELOG claim of a template comment block is false)
- [ ] Physical-consistency / reconstruction example configs (❌ zero matches in figaroh-examples)
- [ ] Complete `talos/` example (⚠️ partial — missing identification, optimal_config, optimal_trajectory)
- [ ] Complete `staubli_tx40/` example (⚠️ partial — missing calibration, optimal_config, optimal_trajectory, update_model)
- [ ] MuJoCo integration examples (3, Phase 2)
- [ ] Genesis integration examples (3, Phase 2)
- [ ] IsaacSim integration examples (3, Phase 3)
- [ ] Example gallery & showcase (Phase 3)
- ✅ `ur10/` and `tiago/` are full real examples
- ✅ `tests/` is real (9 files, ~3684 lines)
- ✅ `create_example.sh` is real (934 lines)
- ❌ `examples/shared/` does not exist (prior docs described it in detail — stale)

### CI / Testing
- [ ] **Test/lint CI workflow** (currently only docs CI exists — this is a gap for both tracks)
- [ ] CI matrix: Python 3.10–3.12, macOS+Linux (v0.7.3, but pragmatic to start sooner)
- [ ] CI matrix with/without optional `picos`/`cvxopt` (v0.4.x gate)
- [ ] Cross-backend validation suite (Phase 2.3)
- [ ] Performance benchmark dashboard (Phase 2.3)

### Deprecation / Archiving
- [ ] Archive `robot_format_converter` on GitHub (July 1, 2026)
- [ ] Archive `figaroh-mujoco` on GitHub (July 1, 2026)
- [ ] Mark PyPI packages deprecated (December 31, 2026)

---

## 7. Resource Planning

### Team Structure
**Core Team (4):** Project Lead (1), Core Developers (2), QA Engineer (1)
**Specialist Teams (2-3 each):** MuJoCo (2), Genesis (2), IsaacSim (2), Documentation (2), Examples (2)
**Total:** 14-16 people

### Timeline Summary

| Track | Milestone | Target | Key Deliverables |
|---|---|---|---|
| A | v0.4.x gates | imminent | End-to-end example, docs, CI matrix, CHANGELOG |
| A | v0.5 | next | Modular refactor, viz update (inertia ellipsoids, CoM overlay) |
| A | v0.6 | — | Online ID, friction models, FIM-OED |
| A | v0.7 | — | Multi-sensor calibration, URDF↔YAML round-trip, test suite |
| A | v1.0 | — | API stabilization, convex ID first-class, migration guide |
| B | Phase 1 | Q2 2026 | PinocchioBackend (❌ critical gap), integration API, tests, benchmarks |
| B | Phase 2 | Q3 2026 | MuJoCo validation, Genesis backend, cross-backend suite |
| B | Phase 3 | Q4 2026 | IsaacSim backend, unified docs, CLI tool |
| B | Phase 4 | 2027+ | Web interface, ROS 2, ML-enhanced, cloud |

### Budget Estimate (Rough, Phase 1-3)
- Personnel: 14-16 people × 9 months × $10K/month = $1.26M - $1.44M
- Infrastructure: GPU compute $45K + CI/CD $18K + docs hosting $9K = $72K
- **Total Phase 1-3:** ~$1.35M - $1.55M

---

## 8. Success Metrics

### Technical
- [ ] All backends pass 100% of interface tests
- [ ] <1% difference in identified parameters across backends
- [ ] <0.1% difference in dynamics computations across backends
- [ ] MuJoCo 2-3x faster than Pinocchio; Genesis (GPU) 5-10x; IsaacSim (GPU) 3-5x
- [ ] 95%+ test coverage; zero critical bugs; CI/CD passing on all PRs
- [ ] All projected link inertias satisfy $m \ge m_{\min}$ and `min_eig(P) >= -psd_eig_tol`

### User Experience
- [ ] Getting started in <5 minutes for any simulator
- [ ] Single-line integration for common workflows
- [ ] <10 lines of code for complete identification
- [ ] 100% API coverage in docs; 10+ tutorials; video demos per simulator

### Community
- [ ] 1000+ GitHub stars (combined repos); 100+ active users; 10+ external contributors

---

## 9. Risk Management

### High-Risk
1. **Simulator API changes** — version pinning, automated testing, regular updates
2. **Performance regressions from abstraction layer** — continuous benchmarking, profiling, optimization
3. **Format conversion inconsistencies** — validation tests, user guides, fallback options
4. **PinocchioBackend debt** — the abstraction is claimed done but isn't; every day core uses direct `pin.*` calls makes the eventual refactor harder. Prioritize wrapping.

### Medium-Risk
1. **Team availability** — cross-training, documentation, external contractors
2. **Dependency conflicts** (simulators require conflicting deps) — optional dependencies, separate environments, Docker
3. **SDP solver availability** (`picos`/`cvxopt`/`mosek`) — lazy imports, graceful errors, documented requirements

### Low-Risk
1. **Licensing issues** — clear documentation, open-source alternatives

---

## 10. Revision History

### Version 1.5 (June 19, 2026) — Track B Step 4 Implemented (MuJoCo Backend Fixed)
- **`MuJoCoBackend.compute_regressor()`** fixed — replaced identity placeholder with
  Pinocchio's analytical `computeJointTorqueRegressor` (delegated via lazy-loaded
  `pin.Model`). MuJoCo 3.9.0 doesn't support runtime inertial parameter perturbation,
  making finite-difference approach infeasible. Pinocchio is always available as a
  FIGAROH dependency, so this is a pragmatic solution.
- **`MuJoCoBackend.compute_coriolis_matrix()`** fixed — replaced incorrect outer-product
  approximation with proper finite-difference Jacobian/2 approach. Uses Euler's theorem
  for quadratic homogeneous functions: since Coriolis forces f(v) = C(q,v)*v are
  quadratic in v, the Jacobian df/dv = 2*C (Christoffel symbols are symmetric).
- **`MuJoCoBackend.compute_mass_matrix()`** fixed — added `mj_forward` call before
  `mj_crb` (required in MuJoCo 3.x; without it, mass matrix returns zeros).
- **`__init__.py` availability check fixed** — `list_backends()` now checks
  `MUJOCO_AVAILABLE`/`PINOCCHIO_AVAILABLE` flags, not just whether the class imports.
  Previously reported `mujoco: True` even when mujoco wasn't installed.
- **13 new MuJoCo tests** added to `test_backends.py`: initialization, dynamics,
  regressor (self-consistency + cross-backend match with Pinocchio), Coriolis
  (shape + zero-velocity + consistency), kinematics.
- **Test suite:** 253 passed, 4 failed (pre-existing IPOPT), 2 skipped. Zero regressions.
- **Key finding:** MuJoCo's model parameters (body_mass, body_inertia, body_ipos) are
  NOT mutable at runtime in 3.9.0 — changes don't affect `mj_inverse` output. This is
  the same class of limitation as the calibration model manipulation blocker.

### Version 1.4 (June 19, 2026) — Track B Step 5 Complete (Core Algorithm Path Migrated)
- **`calibration/calibration_tools.py`** migrated — 7 functions with dynamics computation now accept
  optional `backend=None` parameter: `get_rel_kinreg`, `get_rel_jac`, `update_forward_kinematics`,
  `calc_updated_fkm`, `calculate_kinematics_model`, `calculate_identifiable_kinematics_model`,
  `calculate_base_kinematics_regressor`. Routes FK, Jacobian, gravity, random-configuration through
  backend. `computeFrameKinematicRegressor` uses `get_model_object()` escape hatch (no backend equiv).
- **Step 5 marked complete** — the full algorithm path (identification + calibration) now routes
  through `PinocchioBackend` when a backend is available. 6 files modified, +268/-79 lines total.
- **Remaining Pinocchio coupling** explicitly categorized as "by design": data type construction
  (`SE3`, `Quaternion`, `Force`, `Frame`), model manipulation (`jointPlacements`), visualization
  (Meshcat/Gepetto), collision detection (hppfcl/coal). These require parallel type systems or
  simulator-specific APIs — out of scope for current backend track, under investigation.
- **Test suite:** 240 passed, 4 failed (pre-existing IPOPT), 2 skipped. Zero regressions.
- **MuJoCo equivalents research** launched (librarian, background) — investigating whether
  SE3/transforms, frame management, kinematic regressor, model manipulation, collision detection,
  and visualization can be abstracted across simulators for future full migration.

### Version 1.3 (June 19, 2026) — Track B Step 5 Implemented (Partial Core Migration)
- **`PinocchioBackend.from_model()`** classmethod added — wraps existing `pin.Model`/`pin.Data`
  without URDF re-loading, enabling model sharing with `Robot`/`RobotWrapper` instances.
- **`Robot.backend`** lazy property added — creates `PinocchioBackend.from_model(self.model, self.data)`
  on first access. This is the integration seam for the "strangler fig" migration.
- **`RegressorBuilder`** migrated — routes through `robot.backend.compute_regressor()`,
  `robot.backend.nv`, `robot.backend.get_inertias()` when backend available. Backward-compatible
  `hasattr(robot, 'backend')` fallback to direct `pin.*` calls.
- **`identification_tools.py`** migrated — `calculate_first_second_order_differentiation` accepts
  optional `backend` parameter for `compute_difference()`.
- **`randomdata.py`** migrated — `get_torque_rand` uses `robot.backend.compute_inverse_dynamics()`,
  `generate_waypoints`/`generate_waypoints_fext` use backend-aware `nq`/`nv`.
- **Smoke test confirmed:** `Robot` → `backend` → `PinocchioBackend` → `RegressorBuilder` exercises
  the backend path (not the fallback). Regressor results identical to direct `pin.*` calls.
- **Test suite:** 240 passed, 4 failed (pre-existing IPOPT), 2 skipped. Zero regressions.
- **5 files modified:** pinocchio.py (+from_model), robot.py (+backend property), regressor.py
  (backend-aware), identification_tools.py (+backend param), randomdata.py (backend-aware).
- **Remaining migration:** calibration, visualization, collisions, model loading, data types.

### Version 1.2 (June 19, 2026) — Track B Steps 1-3 Implemented
- **PinocchioBackend** implemented (`backends/pinocchio.py`, 371 lines) — wraps all
  Pinocchio dynamics/kinematics/regressor calls + Lie group ops. `get_backend("pinocchio")`
  now works (was raising ValueError).
- **DynamicsBackend interface extended** (`base.py`) — 6 new optional methods:
  `get_inertias`, `get_frame_id`, `compute_difference`, `compute_integrate`,
  `random_configuration`, `get_model_object`.
- **Backend test suite** created (`tests/unit/test_backends.py`, 32 tests) — factory,
  initialization, numerical correctness vs direct `pin.*` (atol=1e-10), kinematics,
  optional methods, interface conformance. All pass.
- **Test suite updated:** 246 collected, 240 passed, 4 failed (pre-existing IPOPT),
  2 skipped. Zero regressions.
- Phase 1 upgraded from "🔄 PARTIALLY COMPLETE" to "🔄 MOSTLY COMPLETE".
- Remaining Phase 1 gaps: core module migration (Step 5), integration API (Step 6),
  MuJoCo regressor fix (Step 4), performance benchmarks.

### Version 1.1 (June 19, 2026) — Implementation State Verification
- Ran `pytest` and verified file/directory existence against actual codebase.
- **Test suite corrected:** 214 collected, 208 passed, 4 failed (test_robotipopt.py),
  2 skipped — not "198/198 pass" as prior docs claimed.
- **`test_cad_constraints.py`** added to v0.4.3 inventory (was missing from checklist).
- **`examples/shared/`** marked ❌ — does not exist; prior ecosystem analysis described
  9 files in detail but the directory is absent. All references removed/corrected.
- **`web-interface/`** upgraded from "scaffolded early" to "real working Viser app"
  (21 files, ~20K lines). Tech stack corrected from React/Three.js to Viser.
- **Robot examples verified real-vs-partial:** ur10 ✅ full, tiago ✅ full,
  talos ⚠️ partial, staubli_tx40 ⚠️ partial, templates ⚠️ scaffold only.
- **CAD-constraints example claim retracted:** v0.4.3 CHANGELOG claim of a YAML
  comment block in `manipulator_robot.yaml` is false (grep confirms zero matches
  across all of figaroh-examples).
- **`integration/` and `cli/` directories** confirmed absent (not just "not started").
- Phase 4 web interface section updated to reflect existing Viser implementation.

### Version 1.0 (June 19, 2026) — Combined Roadmap
- Merged `ROADMAP_PRIVATE.md` (algorithmic features, v0.4–v1.0) and `IMPLEMENTATION_ROADMAP.md` (multi-simulator backends, Phase 1–4) into a single document.
- Reconciled status markers against actual codebase: PinocchioBackend marked ❌ (file does not exist); Phase 1 downgraded from "✅ COMPLETED" to "🔄 PARTIALLY COMPLETE".
- Identified PinocchioBackend implementation as the highest-leverage next step for Track B.
- Preserved detailed design specs (v0.4.1–v0.4.3) and phase plans as engineering reference.
- Added cross-cutting work section consolidating docs/examples/CI/deprecation items from both tracks.

### Prior documents (superseded, moved to `deprecated/`)
- `deprecated/ROADMAP_PRIVATE.md` (v0.4 design specs, was gitignored in `figaroh/`) — content folded into Track A
- `deprecated/IMPLEMENTATION_ROADMAP.md` v2.0 (June 3, 2026, was workspace root) — content folded into Track B
- `deprecated/FIGAROH_ECOSYSTEM_ANALYSIS.md` v1.0 (June 3, 2026) — analysis input, not a roadmap; findings incorporated
- `deprecated/TASK_1.1_IMPLEMENTATION_SUMMARY.md` (June 3, 2026) — deprecation task summary; status folded into §3 and §6

---

**End of Combined Roadmap**
