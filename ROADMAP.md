# FIGAROH Roadmap

**Roadmap version:** 2.1 — restructured (adds Tracks C/D/E, deduplicates
change-log detail out to `CHANGELOG.md`, adds a consolidated timeline and a
references section) + Track F added as ongoing research (deployment &
sim-to-real positioning, not yet committed)
**Date:** August 16, 2026
**Current release:** `figaroh` 0.4.7 (PyPI, 2026-08-07)
**Test health (verified 2026-08-16):** 451 collected, 442 passed, 4 failed
(pre-existing, unrelated `cyipopt`/IPOPT solver tests), 5 skipped

> **Single source of truth.** This document is the strategic view — vision,
> tracks, status, timeline. It intentionally does **not** duplicate two
> other documents that already do their jobs well:
> - **`CHANGELOG.md`** — the authoritative, dated record of what shipped in
>   each release. If you want to know exactly what changed in 0.4.6, look
>   there, not here.
> - **`docs/decisions/`** — the design-rationale layer: why a feature looks
>   the way it does, what alternatives were considered, what's proposed but
>   not yet built. Each track below links to its source document(s).
>
> **Accuracy note:** every status marker below was checked against the
> actual codebase and test suite on 2026-08-16 (file existence, `grep`,
> `pytest`), not carried forward from a prior draft. Where a track wasn't
> independently re-verified this pass (noted inline), that's flagged rather
> than silently assumed current.

---

## Table of Contents

1. [Vision](#1-vision)
2. [Strategic Principles](#2-strategic-principles)
3. [Current State Snapshot](#3-current-state-snapshot-2026-08-16)
4. [Track A — Algorithmic Core](#4-track-a--algorithmic-core-identification--calibration)
5. [Track B — Multi-Simulator Backend Integration](#5-track-b--multi-simulator-backend-integration)
6. [Track C — Reporting, Verification & Quality Infrastructure](#6-track-c--reporting-verification--quality-infrastructure)
7. [Track D — Calibration Layer Composability](#7-track-d--calibration-layer-composability)
8. [Track E — Example Ecosystem Parity & Robot Ports](#8-track-e--example-ecosystem-parity--robot-ports)
9. [Track F — Deployment & Sim-to-Real Integration (ongoing research)](#9-track-f--deployment--sim-to-real-integration-ongoing-research)
10. [Cross-Cutting Work](#10-cross-cutting-work)
11. [Estimated Timeline](#11-estimated-timeline)
12. [Resource Planning](#12-resource-planning)
13. [Success Metrics](#13-success-metrics)
14. [Risk Management](#14-risk-management)
15. [References](#15-references)
16. [Roadmap Document History](#16-roadmap-document-history)

---

## 1. Vision

**Transform FIGAROH into a universal, simulator-agnostic calibration and
identification toolbox** that enables researchers and engineers to:

- Identify **physically feasible** full per-link inertial parameters — not
  just reduced base-parameter sets — suitable for URDF export, simulation,
  and control.
- Switch between dynamics backends (Pinocchio, MuJoCo, Genesis, IsaacSim)
  with minimal code changes, leveraging each simulator's strengths while
  maintaining consistent algorithms.
- Trust every run's output without reading source code: a machine-readable
  verification verdict, an HTML report, and a redistribution of fitted
  parameters onto the full standard set (not an arbitrary one-hot pick)
  accompany every calibration/identification.
- Integrate seamlessly into existing workflows with straightforward,
  step-by-step APIs and comprehensive documentation.
- Achieve production-grade reliability with comprehensive tests, CI, and
  examples across a growing family of real robots (UR10, TIAGo, TIAGo Pro,
  TALOS, Staubli TX40).

---

## 2. Strategic Principles

1. **Leverage, Don't Reinvent**
   Use existing simulator converters instead of building custom ones.
   Build on FIGAROH's mature algorithms; extend, don't fork. Adopt proven
   ideas from adjacent tools (`robot_calibration`, MuJoCo `sysid`) rather
   than reinventing them from scratch — Tracks C and D exist because of
   this principle.

2. **Abstraction Over Implementation**
   Define clean interfaces for dynamics backends. Enable pluggable
   simulator implementations. Maintain algorithm consistency across
   backends.

3. **Do Not Break Existing Workflows**
   New functionality is **opt-in** through config flags and/or new classes
   until v1.0. Separate concerns: (A) identify parameters, (B) enforce
   physical feasibility, (C) export back to URDF/YAML + visualize/validate.
   Prefer small, testable building blocks over monolithic drops.

4. **User Experience First**
   Prioritize ease of use and clear documentation. Provide step-by-step
   workflows; getting started in <5 minutes. Minimize learning curve for
   new users and new simulators.

5. **Incremental, Validated Progress**
   Deliver working features incrementally. Validate each backend and
   feature thoroughly — against real robot data, not just unit tests where
   possible (Track C's V&V suite exists specifically to make this cheap).
   Maintain backward compatibility.

---

## 3. Current State Snapshot (2026-08-16)

### 3.1 Release & test health

- **PyPI release:** 0.4.7 (2026-08-07). Five releases have shipped since
  the last roadmap pass (0.4.3 → 0.4.7, 2026-06-02 → 2026-08-07) — see
  `CHANGELOG.md` for the full per-release detail.
- **Test suite:** 451 tests collected (up from 259 in June), 442 passed, 4
  failed (all in `test_robotipopt.py`, a pre-existing `cyipopt`-dependency
  gap, not a regression), 5 skipped.
- **New test modules since June:** `test_backends.py`,
  `test_base_calibration_redistribution.py`, `test_compare_report.py`,
  `test_config_deprecation.py`, `test_config_migration.py`,
  `test_geometric_calibration_export.py`, `test_identification_report.py`,
  `test_integration.py`, `test_report.py`, `test_results_manager.py`,
  `test_urdf_exporter.py`, `test_verification.py`.
- **CI:** `figaroh` core still runs docs-only CI (`.github/workflows/docs.yml`)
  — no test/lint workflow. `figaroh-examples` (the sibling repo) does have
  one (`.github/workflows/ci.yml`). This asymmetry is a real, unresolved
  gap — see §10 and §14.

### 3.2 Track summary

| Track | Covers | Status | Detail |
|---|---|---|---|
| A — Algorithmic Core | Inertial ID, physical consistency, reconstruction, CAD constraints, redistribution | v0.4.1–v0.4.7 shipped; v0.5–v1.0 planned | [§4](#4-track-a--algorithmic-core-identification--calibration) |
| B — Backend Integration | Pinocchio/MuJoCo/Genesis/IsaacSim, high-level API, CLI | Phase 1 substantially complete; Phases 2–4 not started | [§5](#5-track-b--multi-simulator-backend-integration) |
| C — Reporting & Verification | HTML reports, machine-readable verdicts, before/after panel, compare page | 7 of 12 items shipped; optimal-* reports and 3 research features remain | [§6](#6-track-c--reporting-verification--quality-infrastructure) |
| D — Calibration Composability | Residual abstraction, multi-step calibration, camera intrinsics (`robot_calibration`-inspired) | Proposed roadmap only; 0 of 7 steps started (1 partial) | [§7](#7-track-d--calibration-layer-composability) |
| E — Example Ecosystem & Ports | TALOS/Staubli script parity, TIAGo eye-hand + suspension ports, URDF exporter gaps | Mixed — some done, most open | [§8](#8-track-e--example-ecosystem-parity--robot-ports) |
| F — Deployment & Sim-to-Real | Model-based control layer for RL/IL deployment, sim-grounding, retargeting | 🔬 **Ongoing research — plan iteration, nothing built** | [§9](#9-track-f--deployment--sim-to-real-integration-ongoing-research) |

Tracks A and B are the original roadmap tracks (algorithmic features and
backend abstraction). **Tracks C, D, and E are new in this revision** —
they formalize work that already had detailed design docs in
`docs/decisions/` but had never been rolled up into the top-level roadmap.
**Track F is a proposed strategic direction, not yet committed** — see §9.

---

## 4. Track A — Algorithmic Core (Identification & Calibration)

> Problem context: classic inverse-dynamics identification yields a
> reduced/base parameter set because the regressor is rank-deficient; users
> want full per-link physical parameters (mass, CoM, inertia) that are
> **physically feasible** ($m>0$, valid/PSD inertia), and a defensible way
> to deploy them.

### Shipped (v0.4.1 – v0.4.7)

| Version | Date | Headline | Module |
|---|---|---|---|
| 0.4.1 | — | LMI physical-consistency projection | `identification/physical_consistency.py` |
| 0.4.2 | — | Base → full parameter reconstruction (nullspace / SDP) | `identification/reconstruction.py` |
| 0.4.3 | 2026-06-02 | CAD-informed constraints (mass/CoM bounds, symmetry) | `identification/cad_constraints.py` |
| 0.4.4 | 2026-06-27 | `urdf_exporter`, `export_validation`, backend architecture (Pinocchio+MuJoCo), SE3 log-map orientation residual fix | `tools/urdf_exporter.py`, `tools/export_validation.py`, `backends/` |
| 0.4.5 | 2026-07-13 | Weighted least squares, provenance/run-archive tooling, held-out validation fallback — **plus the bulk of Track C** (see §6) | `tools/provenance.py`, `tools/run_archive.py` |
| 0.4.6 | 2026-08-06 | `calc_updated_fkm` correctness fixes (elasticity, camera-frame composition, multi-marker now raises instead of silently degrading), legacy-config deprecation + migration tool, RMSE/MAE convention unification | `calibration/calibration_tools.py`, `utils/config_migration.py` |
| 0.4.7 | 2026-08-07 | Base-parameter **redistribution** (`redistribute_parameters()`), `geometric_calibration_export` (PAL runtime-correction YAML) | `tools/qrdecomposition.py`, `tools/geometric_calibration_export.py` |

Full per-item detail for every release above lives in `CHANGELOG.md` — not
repeated here. Design rationale for the redistribution work specifically
is in
[`docs/decisions/tiago-calibration-and-port-review.md`](docs/decisions/tiago-calibration-and-port-review.md)
Part A §A.8.

### v0.4.x — Remaining gates before calling this minor "closed"

Re-verified 2026-08-16 (not just carried forward):

- [ ] **One end-to-end example** in `figaroh-examples`: classic ID →
  physical projection → (optional) reconstruction → URDF export. Still
  **missing** — a fresh grep across `figaroh-examples/examples/**/*.py`
  for `cad_constraint`/`physical_consistency`/`reconstruct_full_parameters`
  returns zero matches in source (only in generated report HTML, which
  just means the report *template* has a section for it, not that any
  example exercises it).
- [x] `picos` backend availability documented — `README.md` lines 131 and
  244 both cover it now.
- [ ] CI test matrix with/without optional `picos`/`cvxopt` — no test/lint
  CI exists yet at all (see §10), so this is blocked on that gap first.
- [ ] `solver.max_seconds` forwarded from config (field exists, still not
  mapped from YAML).
- [ ] Export-to-URDF/YAML path uses *projected* inertials when physical
  consistency is enabled (currently overwrites in place; raw should be
  preserved alongside projected).

### v0.5 — Modular Refactor & Visualization

- **0.5.1** Extract identification pipeline stages (regressor prep,
  decimation/elimination, base-parameter computation, per-link packaging)
  into internal helpers, behavior-preserving, to make future convex-ID
  integration clean.
- **0.5.2** Docs + one focused "classic ID → physical projection → URDF
  export" example (closes the v0.4.x gate above, formally, if not already
  closed sooner).
- **0.5.3** Visualization: inertia ellipsoid/principal-axes overlay, CoM
  vs. link geometry, before/after (raw vs. projected/reconstructed)
  comparison. Convex feasibility can still produce surprising inertias
  without good priors — visual QA matters here.

### v0.6 — Online Identification, Friction, FIM-OED

- **0.6.1** `identification.online` — RLS with forgetting factor at
  minimum, optional sliding-window LS, EKF/UKF later. Reuses the existing
  regressor interface. **Track F's Phase 5 (§9, ongoing research) has a
  more detailed spec for this item** (RLS + EKF + LuGre actuator friction)
  — if that track's plan firms up, its spec should supersede this one
  rather than the two diverging.
- **0.6.2** New friction models (Stribeck, dead-zone/backlash as
  experimental) behind a clean interface; existing viscous/Coulomb stays.
- **0.6.3** OED refactor: Fisher Information Metrics ($\log\det(\text{FIM})$,
  A-/E-optimality) alongside the existing condition-number objective.

### v0.7 — Multi-Sensor Calibration & Round-Tripping

- **0.7.1** Sensor-agnostic residual + Jacobian interface, one new sensor
  type (IMU or AprilTag camera) as reference implementation. This is
  conceptually the same direction as Track D's residual-abstraction work
  (§7) — the two should converge on one design, not diverge.
- **0.7.2** URDF↔YAML round-tripping: diff-friendly "model delta" export,
  regenerate URDF inertials from YAML with version/provenance metadata.
- **0.7.3** Real test gates: physical-feasibility unit tests, projection/
  reconstruction invariants, at least one integration-test robot fixture,
  a pragmatic CI matrix (Python 3.10–3.12, macOS+Linux).

### v1.0 — API Stabilization & Convex ID

- **1.0.1** Stabilize public APIs (`BaseCalibration`, `BaseIdentification`,
  OED classes), move experimental features behind clear namespaces,
  publish a v0.7→v1.0 migration guide. **Gate: backend abstraction must be
  real by this point** (Track B) — PinocchioBackend wrapping direct
  Pinocchio calls is the minimum bar, and that part is already done.
- **1.0.2** Physically-consistent pipeline as first-class: users choose
  "classic ID (fast)" vs. "physically consistent ID (recommended for
  export)"; full Lee et al.-style convex identification becomes official
  once v0.4–v0.7's primitives (feasibility, projection, reconstruction, QA,
  tests, solver story) are all in place.

---

## 5. Track B — Multi-Simulator Backend Integration

> **Re-verification note:** unlike Tracks C/D/E, this track wasn't the
> subject of the recent `docs/decisions` work — but every claim below was
> spot-checked against the filesystem on 2026-08-16 (not blindly carried
> forward), and one real change surfaced: the integration API, previously
> marked "not started," now exists.

### Phase 1: Foundation — 🔄 substantially complete

| Deliverable | Status |
|---|---|
| `DynamicsBackend` abstract interface (`backends/base.py`) | ✅ Done — 9 abstract + 9 optional methods |
| `PinocchioBackend` (`backends/pinocchio.py`) | ✅ Done — 371 lines, 32 tests, numerically verified vs. direct `pin.*` calls at atol=1e-10 |
| `MuJoCoBackend` (`backends/mujoco.py`) | ✅ Done, with known limitations (regressor delegates to Pinocchio's analytical regressor since MuJoCo 3.9 doesn't support runtime inertial-parameter perturbation; Coriolis via finite-difference Jacobian/2) |
| Backend factory `get_backend()` | ✅ Done |
| Backend test suite (`test_backends.py`) | ✅ Done — 45 tests (32 Pinocchio + 13 MuJoCo) |
| Core module migration (identification + calibration algorithm path) | ✅ Done — "strangler fig" pattern, `Robot.backend` lazy property, both `RegressorBuilder` and 7 `calibration_tools.py` functions route through it when available |
| High-level integration API (`integration/api.py`) | ✅ **Newly confirmed done** — `RobotIdentificationSystem` with `from_urdf()`/`from_mjcf()` classmethods and `identify_parameters()`, 413 lines, covered by `test_integration.py`. Previous roadmap draft marked this "❌ not started" — that was stale. |
| Deprecate `robot_format_converter` / merge `figaroh-mujoco` | ✅ Notice + migration guide done; GitHub archiving still pending |
| Performance benchmarks (Pinocchio vs. MuJoCo) | ❌ Not started |
| Migrate existing `figaroh-examples` scripts to backend abstraction | ❌ Not started — examples still call Pinocchio directly |
| CLI tool (`figaroh/cli/`) | ❌ Not started — directory doesn't exist |
| `GenesisBackend` / `IsaacSimBackend` | ❌ Not started — no `backends/genesis.py` or `backends/isaacsim.py` |

**Intentionally not migrated to the backend abstraction** (Pinocchio-specific
by design, not a gap): `calibration/parameter.py`'s SE3/Frame/Inertia data
construction, `measurements/measurement.py`, Meshcat/Gepetto visualization,
hppfcl/coal collision detection. These would need a parallel math/type
layer to abstract — deferred, not forgotten.

### Phase 2: MuJoCo & Genesis Backends — not started

- MuJoCo: interface tests, performance benchmarks (target 2–3× vs.
  Pinocchio), a format-conversion guide, and three worked examples
  (identification, humanoid calibration, contact identification).
- Genesis: `backends/genesis.py` does not exist yet. GPU acceleration,
  native Python API, USD/MJCF/URDF support; three worked examples.
- Cross-backend validation: dynamics consistency (<0.1% tolerance),
  identification consistency (<1% tolerance), performance benchmarks,
  automated CI dashboard. **Track F's Phase 3 (§9) has a more detailed spec
  for this exact deliverable** (`backends/validation.py`,
  `BackendConsistencyReport`) — use that spec if/when this item is picked
  up, rather than re-deriving one.

### Phase 3: IsaacSim Backend & Ecosystem — not started

- `backends/isaacsim.py`, USD-based workflow, Isaac Lab integration.
- Unified documentation restructure (`getting-started/`, per-simulator
  guides, migration guides).
- CLI tool (`figaroh identify`, `optimize-trajectory`, `calibrate`,
  `backends list/info`, `convert`).

### Phase 4: Ecosystem & Advanced Features (2027+) — not started

- Web interface hardening — note: `figaroh-examples/web-interface/` is
  **already a real, working Viser-based app** (21 files, ~20K lines,
  interactive 3D viz, task management), just marked "under development."
  Phase 4 here means hardening and completing stub panels, not greenfield
  work — and it settled on Viser, not the originally-planned
  FastAPI+React/Three.js stack.
- ROS 2 integration, ML-enhanced identification, cloud deployment — all
  still at the "not started, 2027+" horizon.

---

## 6. Track C — Reporting, Verification & Quality Infrastructure

**Origin:** verifying an early reporting feature against a real UR10
example surfaced bugs bigger than that feature's own scope, which became a
seven-step build-out. Full design rationale, every implementation
deviation, and the bugs found along the way are documented in
[`docs/decisions/external-tool-comparisons.md`](docs/decisions/external-tool-comparisons.md)
Part C.

| Item | Status |
|---|---|
| Feature 1 — HTML diagnostic report (calibration) | ✅ Shipped (0.4.5) |
| Feature 1b — Terminal + HTML quality report (identification) + held-out validation | ✅ Shipped (0.4.5) |
| Step 1 — Reshape-bug fix + fallback-plotting dedup | ✅ Shipped (0.4.5) |
| Step 2 — Machine-readable `VerificationVerdict` (`verify()`) | ✅ Shipped (0.4.5) |
| Step 3 — `series`/`compat` export extension | ✅ Shipped (0.4.5) |
| Step 4 — Before/after interactive panel (embedded SVG chart) | ✅ Shipped (0.4.5) |
| Step 5 — Static two-run compare page (`compare_report.py`) | ✅ Shipped (0.4.5) |
| Step 6 — Terminal + HTML reports for optimal-* tasks | ⬜ Proposed, not started |
| Step 7 — Unified report schema (feasibility spike only) | ⬜ Not started |
| Feature 2 — Generic `Parameter` + modifier abstraction | ⬜ Not started |
| Feature 3 — Log-Cholesky pseudo-inertia parameterization | ⬜ Not started — overlaps Track A's physical-consistency work; if pursued, coordinate rather than duplicate |
| Feature 4 — Black-box rollout-refinement stage (MuJoCo backend) | ⬜ Not started — depends on Feature 2, and on Track B's MuJoCo backend being validated (Phase 2) |

**Known, deliberately-unfixed bug:** `calculate_first_second_order_differentiation()`
in `identification/identification_tools.py` (lines 245, 254) loops
`range(nq - 1)` when computing `ddq`, silently zeroing the last active
joint's acceleration for any robot without continuous/spherical joints.
Confirmed still present 2026-08-16. Flagged, not yet scheduled — broad
blast radius (every identification example), tracked here so it isn't
lost.

---

## 7. Track D — Calibration Layer Composability

**Origin:** a comparison against `robot_calibration` (ROS 2, Ceres-based
kinematic/sensor calibration) found FIGAROH's calibration layer is
comparatively monolithic — one hard-coded SE3-pose residual, one solve —
against `robot_calibration`'s composable "models × error blocks ×
multi-step × regularizer" architecture. FIGAROH wins decisively on
dynamics identification, physical consistency, OED, and reporting (Tracks
A and C); this track is about closing the calibration-layer gap in the
other direction. Full comparison and gap analysis:
[`docs/decisions/external-tool-comparisons.md`](docs/decisions/external-tool-comparisons.md)
Part A.

| Step | Description | Status |
|---|---|---|
| 1 | `Residual`/`ErrorBlock` abstraction — decompose `BaseCalibration.cost_function` into stackable residual types | ⬜ Not started |
| 2 | Multi-step calibration (`calibration_steps` config) — solve in stages, freezing earlier results | ⬜ Not started |
| 3 | Prior/regularization residual (`PriorResidual`, "stay near nominal") | ⬜ Not started |
| 4 | Camera intrinsic model (pinhole `fx,fy,cx,cy` + optional distortion) | ⬜ Not started |
| 5 | Feature-finder / measurement ingestion layer (raw sensor data → `Measurement`, not just pre-extracted poses) | ⬜ Not started — biggest practical gap for real-robot use, per the source doc |
| 6 | Export convention parity — `<calibration rising>` tag export + camera-intrinsics YAML | 🟡 **Half done** — `rising` tag export already shipped in `tools/urdf_exporter.py`; camera-YAML export is the only remaining piece (bundle with Track E's URDF-exporter work, §8) |
| 7 | Mobile-base + magnetometer calibration (example-level, niche) | ⬜ Not started, low priority — candidate for `figaroh-examples`, not core |

**Recommended build order** (per the source doc, unchanged): residual
abstraction → multi-step → prior → camera intrinsics → data ingestion.
Steps 4–5 also overlap Track A's v0.7.1 (multi-sensor calibration) — these
should converge on one residual/sensor-abstraction design, not ship two
competing ones.

---

## 8. Track E — Example Ecosystem Parity & Robot Ports

Three previously-separate audits, none fully closed, rolled up here
because they're all "make the example/tooling surface match what core
already supports" work rather than new algorithmic capability.

### 8.1 `figaroh-examples` script parity (Phase 7 of the examples audit)

Source: [`docs/decisions/figaroh-examples-improvement_plan.md`](docs/decisions/figaroh-examples-improvement_plan.md).
Phases 1–6 of that audit (35 items — argparse/CLI, logging, error handling,
config cleanup, dead-code removal, CI, smoke tests) are fully done. Phase 7
is not:

| Item | Status |
|---|---|
| 7.1 — Add `identification.py`, `optimal_config.py`, `optimal_trajectory.py` to TALOS | ⬜ Not started |
| 7.2 — Add `calibration.py`, `optimal_config.py`, `optimal_trajectory.py`, `update_model.py` to Staubli TX40 | ⬜ Not started |
| 7.3 — Add `update_model.py` to TIAGo | ✅ Done |
| 7.4 — Rename TALOS `calibration_upperbody.py` → `calibration.py` | ⬜ Not started |

### 8.2 TIAGo feature ports (eye-hand calibration + suspension identification)

Source: [`docs/decisions/tiago-calibration-and-port-review.md`](docs/decisions/tiago-calibration-and-port-review.md)
Part B. Two features sitting on old, pre-architecture-split branches,
never ported forward. Neither has landed — no `eye_hand_calibration.py`,
`suspension_identification.py`, `vicon_utils.py`, or `suspension_utils.py`
exist in `figaroh-examples/examples/tiago/` as of 2026-08-16.

| Feature | Risk | Status |
|---|---|---|
| Eye-hand (camera) calibration | Low — no core changes needed, style-only port | ⬜ Not started; 8 open decisions (which gripper variants, XACRO vs. `update_model.py` output, standalone vs. `BaseCalibration` subclass, …) |
| Mobile-base / suspension identification | Medium–High — needs an architecture decision (standalone script vs. `BaseIdentification` subclass vs. documented research example) and a 1068-line monolith refactored into focused modules | ⬜ Not started |

When eye-hand calibration is ported, its deploy step should reuse Track A's
`redistribute_parameters()` + `geometric_calibration_export()` (already
shipped, §4) rather than reintroducing the old branch's one-hot
`write_to_xacro` output — it hits the exact same base-parameter
redistribution ambiguity Track A's work already solved.

### 8.3 URDF exporter gaps

Source: [`docs/decisions/urdf_exporter.md`](docs/decisions/urdf_exporter.md).
`export_urdf()` shipped in 0.4.4 and is tested (18 unit tests + pendulum
fixture + viser visual tests), but with real deviations from its original
spec:

| Gap | Status |
|---|---|
| Inertia-tensor handler (`Ixx_*` etc.) | 🟡 Registered but a no-op stub — silently accepts and discards these params |
| Camera-YAML export | ⬜ Not started — same deliverable as Track D Step 6, bundle the two |
| Multi-format export (MJCF/SDF/USD) | ⬜ Deferred as planned, demand-driven, no fixed date |
| `base_*`/`pEE*`/`phiEE*` auto-apply | Deliberate design deviation, not a gap — surfaced via `frame_settings_doc()` for the caller instead of auto-applied; documented behavior, not a bug |

---

## 9. Track F — Deployment & Sim-to-Real Integration (ongoing research)

**Status: 🔬 Ongoing research — plan iteration, nothing built.** Unlike
Tracks A–E, this track is not yet a committed roadmap item. It's tracked
here so it's visible and cross-referenced against the rest of the roadmap
while the plan is still being iterated — not as a promise of what ships
next.

**Full plan, research citations, and open decisions:**
[`docs/decisions/sim2real-modelbased-deployment.md`](docs/decisions/sim2real-modelbased-deployment.md).

### What it proposes

A reframing of FIGAROH's headline value proposition: not just "a toolbox
that *produces* an identified model," but a toolbox that also **runs that
model under a deployed RL/IL policy** — gravity compensation, computed
torque, and a residual-policy interface so a learned policy only has to
learn what the identified model doesn't already predict. Secondary goals:
grounding training simulators with identified parameters, and whole-body
QP-based motion retargeting.

This is a materially larger scope commitment than Tracks A–E — it adds a
new `application/` namespace and a new optional dependency (Crocoddyl, for
MPC/whole-body control) that don't exist in FIGAROH today.

### Where it overlaps existing tracks (by design — not duplication)

| This track's phase | Overlaps | Resolution |
|---|---|---|
| Phase 3 — cross-backend dynamics validation | Track B Phase 2.3 (already scoped: M/C/g <0.1%, regressor <1%) | This phase's spec **supersedes** Track B 2.3's description — one deliverable, not two |
| Phase 5 — online ID + LuGre friction | Track A v0.6.1 ("online identification") | This phase's spec (RLS + EKF + LuGre) **supersedes** v0.6.1's original scope |
| Phase 2 — friction/armature URDF export | Track D Step 6 + Track E §8.3 (camera-YAML/URDF exporter gaps) | Same underlying deliverable — implement once, reuse across all three |
| Phase 2 — covariance export | Track A v0.4.7 (`redistribute_parameters()`/`propagate_covariance_min_norm`, already shipped) | **Corrected in the source doc** — covariance export already exists; this phase should *consume* it, not rebuild it |

### What's genuinely new (no existing home)

- **Phase 1** — the deployment control layer itself (`application/control.py`,
  `application/estimator.py`, `RobotDeploymentSystem`). The core of the pivot.
- **Phase 4** — real-robot data-plane adapters (`integration/adapters/`,
  canonical `TrajectoryData` contract). Anticipated conceptually by the
  existing `figaroh-sim2real` skill, but never built.
- **Phase 6** — end-to-end sim2real loop + whole-body QP retargeting
  interface.

### Open research flagged, not yet scoped into any phase

How FIGAROH actually integrates with the tools practitioners already use
for the real use case — e.g. **mjlab** and existing **motion-retargeting
libraries** — rather than only exposing its own API in isolation. Is
FIGAROH an adapter *into* those tools, or a library they import *from*?
This shapes Phase 1's API surface and should be answered before that
surface is treated as final. See the source doc's "Open decisions to
iterate," item 8.

### Naming collision to resolve before any of this ships

"Residual" currently means three different things across the roadmap:
Track D's calibration measurement residual, Track A v0.7.1's sensor
residual, and this track's control-torque/policy residual. They'll need
distinct names in code.

---

## 10. Cross-Cutting Work

### Documentation
- [ ] Sphinx docs pages for `physical_consistency`/`reconstruction` APIs
  (v0.4.x gate, §4)
- [ ] API reference auto-generation, 10+ tutorials, migration guides
  (Track B Phase 3)
- [x] `picos` solver-availability docs in `README.md` (done, verified)

### Examples (`figaroh-examples`)
- [ ] v0.4.x end-to-end example: ID → projection → reconstruction → URDF
  export (still missing, §4)
- [ ] Complete TALOS / Staubli TX40 script parity (§8.1)
- [ ] TIAGo eye-hand + suspension ports (§8.2)
- [ ] MuJoCo / Genesis / IsaacSim integration examples (Track B Phase 2–3)
- [x] `ur10/` and `tiago/` are full, real examples
- [x] `redistribute_parameters()`/`geometric_calibration_export` already
  wired into `ur10`, `tiago`, and `talos` calibration scripts

### CI / Testing
- [ ] **Test/lint CI workflow for `figaroh` core** — still the single
  biggest cross-cutting gap. `figaroh-examples` has one; `figaroh` core
  doesn't. Every gate above that says "CI matrix" is blocked on this
  existing first.
- [ ] CI matrix: Python 3.10–3.12, macOS+Linux
- [ ] CI matrix with/without optional `picos`/`cvxopt` (v0.4.x gate)
- [ ] Cross-backend validation suite + performance benchmark dashboard
  (Track B Phase 2)

### Deprecation / Archiving
- [ ] Archive `robot_format_converter` and `figaroh-mujoco` on GitHub
  (notice already shipped; archiving itself still pending)
- [ ] Mark PyPI packages deprecated on schedule

---

## 11. Estimated Timeline

These are engineering estimates based on effort/priority ordering already
established in each track's source document — not committed dates. Tracks
run in parallel; a row's quarter is when that item is *targeted to start or
land*, not a hard deadline.

| When | Track | Item |
|---|---|---|
| **Q3 2026 (now)** | A | Close remaining v0.4.x gates: end-to-end CAD/physical-consistency example, `solver.max_seconds` config wiring |
| Q3 2026 | E | TALOS/Staubli script parity (7.1, 7.2, 7.4) — mechanical, low-effort, matches existing patterns |
| Q3 2026 | E / D | Camera-YAML URDF export (closes Track D Step 6 + Track E §8.3 together — same deliverable) |
| Q3–Q4 2026 | B | Finish Phase 1 spillover: performance benchmarks, migrate `figaroh-examples` scripts to backend abstraction, CLI scaffold |
| **Q4 2026** | A | v0.5 — modular refactor + visualization (inertia ellipsoids, CoM overlay, before/after) |
| Q4 2026 | C | Steps 6–7 — optimal-* task reports, unified-schema feasibility spike |
| Q4 2026 | E | TIAGo eye-hand calibration port (§8.2, low risk) |
| Q4 2026 | Cross-cutting | Stand up test/lint CI for `figaroh` core (§10) — unblocks several other gated items |
| Q3–Q4 2026 | F | Ecosystem-integration research pass (mjlab, retargeting libraries — §9's flagged open decision) before any Track F API surface is treated as final |
| **Q1 2027** | A | v0.6 — online identification, new friction models, FIM-OED |
| Q1 2027 | D | Steps 1–3 — residual/error-block abstraction, multi-step calibration, prior regularization |
| Q1–Q2 2027 | B | Phase 2 — MuJoCo validation + benchmarks, Genesis backend, cross-backend suite |
| Q1–Q2 2027 | E | TIAGo suspension/mobile-base identification port (§8.2, needs an architecture decision first) |
| **Q2 2027** | D | Steps 4–5 — camera intrinsics, feature-finder/measurement ingestion (coordinate with Track A's v0.7.1, don't duplicate) |
| **H2 2027** | A | v0.7 — multi-sensor calibration, URDF↔YAML round-trip, real test gates |
| H2 2027 | B | Phase 3 — IsaacSim backend, unified docs, CLI tool |
| **2027+** | A | v1.0 — API stabilization, convex ID first-class (gated on Track B's backend abstraction being real, already true today for the minimum bar) |
| 2027+ | B | Phase 4 — web interface hardening, ROS 2, ML-enhanced ID, cloud |
| 2027+ | C | Features 2–4 — generic parameter abstraction, log-Cholesky reparameterization (coordinate with Track A), black-box rollout refinement (needs Track B's MuJoCo backend validated) |
| Opportunistic | D | Step 7 — mobile-base/magnetometer (example-level, no fixed date) |
| Opportunistic | E | URDF multi-format export (MJCF/SDF/USD) — demand-driven |
| **No committed date** | F | Phases 1–6 (§9) — deliberately excluded from the schedule above. This track is still plan iteration, not a committed body of work; it gets a date once the plan is finalized and the ecosystem-integration research above has answered what Phase 1's API actually needs to look like. |

---

## 12. Resource Planning

### Team structure (unchanged from prior planning)
**Core Team (4):** Project Lead (1), Core Developers (2), QA Engineer (1)
**Specialist Teams (2–3 each):** MuJoCo (2), Genesis (2), IsaacSim (2),
Documentation (2), Examples (2)
**Total:** 14–16 people

### Budget estimate (rough, Track B Phases 1–3)
- Personnel: 14–16 people × 9 months × $10K/month ≈ $1.26M–$1.44M
- Infrastructure: GPU compute $45K + CI/CD $18K + docs hosting $9K = $72K
- **Total:** ~$1.35M–$1.55M

This estimate covers Track B only (the resource-heavy multi-simulator
work). Tracks A, C, D, and E are comparatively small, incremental efforts
that have historically shipped with much smaller footprints — e.g. Track
C's five shipped steps landed across roughly one month of focused work.

---

## 13. Success Metrics

### Technical
- [ ] All backends pass 100% of interface tests
- [ ] <1% difference in identified parameters across backends
- [ ] <0.1% difference in dynamics computations across backends
- [ ] MuJoCo 2–3× faster than Pinocchio; Genesis (GPU) 5–10×; IsaacSim
  (GPU) 3–5×
- [ ] Test/lint CI passing on every PR (currently: no such CI exists)
- [ ] All projected link inertias satisfy $m \ge m_{\min}$ and
  `min_eig(P) >= -psd_eig_tol`
- [x] Every calibration/identification run produces a machine-readable
  pass/fail verdict (Track C, shipped)

### User Experience
- [ ] Getting started in <5 minutes for any simulator
- [ ] Single-line integration for common workflows (Track B's
  `RobotIdentificationSystem.identify_parameters()` is a first step, done)
- [ ] <10 lines of code for complete identification
- [ ] 100% API coverage in docs; 10+ tutorials; video demos per simulator

### Community
- [ ] 1000+ GitHub stars (combined repos); 100+ active users; 10+ external
  contributors

---

## 14. Risk Management

### High-risk
1. **Simulator API changes** — version pinning, automated testing, regular
   updates.
2. **Performance regressions from the abstraction layer** — continuous
   benchmarking, profiling, optimization.
3. **No test/lint CI on `figaroh` core** — every commit to core algorithms
   (Tracks A, C, D) currently lands without an automated correctness gate
   beyond `pre-commit run --all-files`. This is the highest-leverage
   process risk in the whole roadmap; closing it (§10) unblocks several
   other gated items too.
4. **Format conversion inconsistencies** — validation tests, user guides,
   fallback options.

### Medium-risk
1. **Known-unfixed `ddq` indexing bug** (Track C) — silently under-models
   the last active joint's acceleration-dependent terms for any robot
   without continuous/spherical joints, on the default `dt=None` path.
   Broad blast radius (every identification example); not yet scheduled.
2. **Track A/C/D/F convergent-but-separate residual-abstraction work** — v0.7.1
   (multi-sensor calibration), Track D Steps 1 & 4–5 (residual abstraction,
   camera intrinsics), Track C Feature 3 (log-Cholesky), and Track F's
   control-torque/policy residual (§9) are all variations on "generalize
   how a residual/parameter is represented" — four candidates, three
   different domains (measurement, physical consistency, control). Left
   unmanaged, these could ship as incompatible abstractions with clashing
   names instead of one coherent design. Flagged in §4, §7, and §9; needs
   an explicit design decision before more than one of them starts.
3. **Team availability** — cross-training, documentation, external
   contractors.
4. **Dependency conflicts** (simulators require conflicting deps) —
   optional dependencies, separate environments, Docker.
5. **SDP solver availability** (`picos`/`cvxopt`/`mosek`) — lazy imports,
   graceful errors, documented requirements (mostly done — see §4).

### Low-risk
1. **Licensing issues** — clear documentation, open-source alternatives.
2. **Doc/reality drift** — mitigated going forward by this roadmap's
   practice of verifying every status marker against the codebase rather
   than trusting a prior draft; the same practice should apply to future
   revisions.

---

## 15. References

**Root-level docs:**
- [`CHANGELOG.md`](CHANGELOG.md) — authoritative per-release change record
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system architecture (per `AGENTS.md`,
  partly aspirational/out of date in places — cross-check against code)
- [`README.md`](README.md) — install/quickstart, optional-dependency matrix

**`docs/decisions/` (design rationale, working documents for contributors):**

| Document | Covers | Referenced from |
|---|---|---|
| [`external-tool-comparisons.md`](docs/decisions/external-tool-comparisons.md) | Part A: `robot_calibration` comparison (Track D's source). Part B: MuJoCo `sysid` comparison. Part C: the full reporting/verification build-out (Track C's source) | §6, §7 |
| [`tiago-calibration-and-port-review.md`](docs/decisions/tiago-calibration-and-port-review.md) | Part A: TIAGo/TIAGo Pro structural & statistical calibration analysis, redistribution rationale (Track A). Part B: eye-hand + suspension port review (Track E's source) | §4, §8.2 |
| [`figaroh-examples-improvement_plan.md`](docs/decisions/figaroh-examples-improvement_plan.md) | Cross-example audit of UR10/TIAGo/TALOS/Staubli (Track E's source) | §8.1 |
| [`urdf_exporter.md`](docs/decisions/urdf_exporter.md) | URDF exporter plan & spec, implementation deviations (Track E's source) | §8.3 |
| [`validation-quality-report.md`](docs/decisions/validation-quality-report.md) | FK validation + statistical quality matrix — fully implemented, superseded in practice by Track C's reporting suite | §6 |
| [`sim2real-modelbased-deployment.md`](docs/decisions/sim2real-modelbased-deployment.md) | Positioning research: FIGAROH as a sim-to-real deployment/model-based-control toolbox for RL/IL policies and motion retargeting (Track F's source — ongoing research, not yet committed) | §9 |

See also `docs/source/further_reading/decisions.md` for the same index,
rendered into the built docs site.

---

## 16. Roadmap Document History

This section tracks changes to the **roadmap document itself**. For
software changes, see `CHANGELOG.md`.

- **v2.1 (2026-08-16)** — added Track F (Deployment & Sim-to-Real
  Integration), sourced from
  `docs/decisions/sim2real-modelbased-deployment.md` (moved into
  `docs/decisions/` from a working `.slim/deepwork/` draft in this same
  pass). Marked explicitly as ongoing research, not a committed track —
  distinct from Tracks A–E. Cross-referenced its phase-level overlaps with
  existing tracks rather than letting them exist as separate,
  potentially-diverging items: Phase 3 supersedes Track B's cross-backend
  validation item, Phase 5 supersedes Track A's v0.6.1 online-ID item, and
  Phase 2's URDF/friction export shares a deliverable with Track D Step 6
  and Track E §8.3. Corrected a stale claim in the source doc itself
  (covariance export was described as not-yet-built; it shipped in 0.4.7).
  Flagged an added open question — how FIGAROH integrates with ecosystem
  tools practitioners actually use (mjlab, motion-retargeting libraries) —
  as a prerequisite research pass before Track F's API surface is treated
  as final. Track F is deliberately excluded from the committed timeline
  (§11) pending that research and plan finalization.
- **v2.0 (2026-08-16)** — this revision. Added Tracks C (Reporting &
  Verification), D (Calibration Composability), and E (Example Ecosystem &
  Ports), formalizing work that previously only existed as detailed
  `docs/decisions/` documents with no roadmap-level rollup. Removed the
  old per-version "Revision History" section (it had drifted into
  duplicating `CHANGELOG.md`'s job) in favor of this shorter document-only
  history plus a References section. Added a Table of Contents and a
  single consolidated Estimated Timeline spanning all five tracks.
  Re-verified every Track A/B status marker against the codebase rather
  than carrying the June draft forward; one real correction surfaced
  (`integration/api.py` is implemented, previously marked "not started").
  Updated current release to 0.4.7 and test counts to 451/442/4/5.
- **v1.0–v1.5 (June 2026)** — original merge of `ROADMAP_PRIVATE.md`
  (algorithmic features) and `IMPLEMENTATION_ROADMAP.md` (multi-simulator
  backends) into one document, followed by five incremental verification
  passes reconciling status markers against the actual codebase. Superseded
  prior documents: `deprecated/ROADMAP_PRIVATE.md`,
  `deprecated/IMPLEMENTATION_ROADMAP.md`,
  `deprecated/FIGAROH_ECOSYSTEM_ANALYSIS.md`,
  `deprecated/TASK_1.1_IMPLEMENTATION_SUMMARY.md`.

---

**End of Roadmap**
