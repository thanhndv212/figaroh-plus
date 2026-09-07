# Positioning: FIGAROH as Sim-to-Real (robot-learning) + Model-Based Deployment

**Started:** 2026-08-16
**Status:** 🔬 **Ongoing research — plan iteration, not yet building.** Tracked
as `ROADMAP.md` Track F (see there for how this aligns with the rest of the
roadmap). Nothing in this document is committed; phases, module names, and
dependency choices (Crocoddyl, `application/` namespace, etc.) are all
still open.
**Repo:** `figaroh/` (git root: `/Users/thanhndv212/Develop/figaroh-ws/figaroh`)

---

## Goal & understanding

**User's clarified thesis (2026-08-16):** FIGAROH closes the **sim-to-real gap in
deployment of robot learning** — RL/imitation-learning policies and motion
retargeting. The gap is dynamics mismatch: a policy trained in sim fails on the
real robot. FIGAROH's sysid (inertial + friction + actuator identification) closes
that gap by producing a **faithful identified model of the real robot** that:

1. **Underpins deployment (PRIMARY)** — a model-based control layer (gravity comp /
   computed torque / inverse dynamics) under the learned policy, so the policy only
   learns residuals (residual policy learning / torque adaptation).
2. **Grounds the simulator for training (SECONDARY)** — inject identified params into
   the sim (match-sim-to-real); domain randomization around identified params is a
   differentiator, not the lead.
3. **Enables retargeting** — accurate kinematics + dynamics for human→robot motion
   retargeting, via a whole-body QP controller interface.

**Core thesis (refined after research):** the identified, *physically-consistent*,
*backend-agnostic* **model** is the shared currency, but the **deployment control
interface is the headline value prop** — research confirms deployment correction is
more impactful than sim-grounding. SysID point estimates beat wide DR after minimal
measurement (3–5 rollouts), so accurate estimation precedes (not replaces) DR.

## Ground-truth context (verified Aug 2026)

- `integration/api.py` **exists** (413 L) — `RobotIdentificationSystem`
  (`from_urdf()`, `identify_parameters()`, `IdentificationResult`). Identification-only;
  `from_mjcf()` raises `NotImplementedError`.
- `backends/` = `base.py` (abstract `DynamicsBackend`), `pinocchio.py` (397 L),
  `mujoco.py`. Genesis + IsaacSim **not started** (RL training targets are Isaac
  Lab/Genesis — note as dependency, not pulled into critical path).
- `tools/urdf_exporter.py` exports 12 joint-offset categories + inertials (to URDF).
  No YAML "model delta", no provenance, no covariance export, no friction/armature
  field wiring, no round-trip.
- Cross-backend validation (Track B 2.3), online ID (v0.6.1) — **not started**.
- `figaroh-sim2real` skill defines "offline log compatibility first" + data contract.
  No adapters/examples built.
- No deployment/control layer exists. `figaroh.optimal` does OED (IPOPT), not MPC.
- `DynamicsBackend` already has optional `compute_dynamics_derivatives` (analytical
  RNEA/ABA derivatives) — the hook MPC/WBC needs.

> **Correction (2026-08-16, post-alignment-review):** the covariance-export claim
> above is stale. `figaroh.tools.qrdecomposition.propagate_covariance_min_norm` +
> `BaseCalibration.redistribute_parameters()` shipped in **0.4.7** (2026-08-07) and
> are already surfaced in the HTML calibration report's "Redistributed standard
> parameters" section. Phase 2 below should read as "wire the *existing* covariance
> export into `model_delta.py`'s provenance schema," not "build covariance export
> as a new differentiator." See `ROADMAP.md` Track A and
> `docs/decisions/tiago-calibration-and-port-review.md` Part A §A.8.4 for what
> already exists.

## Research findings (reconciled)

### lib-1 — sim2real sysid vs DR; control/estimation libs; digital-twin provenance ✅
- **F1 (MPC derivatives):** MPC needs *analytical* derivatives → Pinocchio
  `computeRNEADerivatives`/`computeABADerivatives` (already optional on
  `DynamicsBackend`); numeric diff is wrong path. → **Crocoddyl-first** (not acados).
- **F2 (IDF-MPC):** Inverse-dynamics MPC (Mastalli 2023, ~47% compute reduction) uses
  **τ as control variable**; FIGAROH's identified M, C, g map *directly*.
- **F3 (inertial manifold):** Martinez 2024/2025 multi-contact inertial estimation uses
  an **inertial manifold** + Riccati recursion for joint state+param estimation.
  FIGAROH's `physical_consistency` (LMI/SDP) **is** the inertial manifold — align/extend.
- **F4 (Crocoddyl > acados):** Crocoddyl natively consumes Pinocchio models
  (`DifferentialActionModelContactInvDynamics`); acados needs CasADi glue.
- **F5 (friction/armature = dominant gap):** Pinocchio `armature`, `damping`,
  `lowerDryFrictionLimit`/`upperDryFrictionLimit` are the right knobs. → P2 export
  must write identified fv/fs/ia to these fields.
- **F6 (h1v2 assets ready):** `figaroh-examples` h1v2 `urdf_exporter.py` +
  `shared/exporters.py` (neutral SysID schema → URDF/SDF/MJCF/USD) are portable assets.
- **F7 (gap metrics):** torque-prediction RMSE + tracking divergence; **<5% torque
  error** is state-of-art after full inertial+friction ID.
- **F8 (provenance):** `confidence_intervals` (covariance) is standard model-artifact
  provenance metadata.

### lib-2 — RL/IL deployment correction + retargeting + what FIGAROH must expose ✅
- **G1 (deployment > sim-grounding):** residual RL / TAM (Torque Adaptation Module) /
  SPARR / SEEC is the **dominant 2026 deployment strategy**. FIGAROH should **lead with
  the deployment control layer** (M, C, g, J, inverse dynamics, gravity comp, residual
  output) as its **primary interface**. Sim-grounding is necessary but not sufficient.
  → **RE-PRIORITIZATION: deployment layer becomes Phase 1 (lead).**
- **G2 (sysid beats DR after minimal measurement):** 3–5 identification rollouts with
  accurate estimates **outperform** wide DR that includes the true value (2026 arXiv
  "How Should a Sim-to-Reality Transfer Budget Be Spent?"). SysID should **precede**
  DR, not replace it. Validates FIGAROH's regressor-based estimation investment.
- **G3 (covariance-aware DR NOT adopted):** the field uses **2–5x heuristic inflation**,
  never the OLS covariance `σ²(ΦᵀΦ)⁻¹` directly. → covariance export is an **untapped
  differentiator** but **low priority** (nobody consumes it yet). Demote from P1 lead.
  *(See the ground-truth correction above — the export itself already exists; what's
  actually low-priority/undone is a DR sampler that consumes it.)*
- **G4 (retargeting needs whole-body QP):** humanoid pipelines need **QP-based
  whole-body controllers** (tsid/crocoddyl) with contact constraints + balance criteria,
  not just inverse dynamics. FIGAROH must **expose an interface to WBC frameworks**
  (crocoddyl `ContactModelMultiple` + QP) or provide a minimal QP wrapper. → sharpens
  retargeting scope.
- **G5 (LuGre friction = frontier differentiator):** MuJoCo's LuGre model
  (stiffness/damping/Coulomb/static/Stribeck) is state-of-art for actuator dynamics;
  most sysid tools stop at inertia + viscous friction. Supporting LuGre from torque data
  is a **FIGAROH differentiator**.
- **G6 (residual interface must be expressive):** additive residuals **fail** when base
  distribution is geometrically mismatched (Warp RL 2026 uses invertible spline flows);
  TAM uses history-encoder + torque-adaptor reusable across action spaces. → residual
  interface should support **additive (first) + expressive correction (second mode)**.
- **G7 (required API surface):** `M(q)`, `C(q,q̇)`, `g(q)`, `J(q)`,
  `τ_id = Mq̈+Cq̇+g`, `τ_gc = g(q)`, `τ_residual = τ_desired − τ_id`, + actuator-level
  models (armature/damping/LuGre).
- **G8 (competitor baseline):** MuJoCo's native `mujoco.sysid` (differentiable sysid,
  per-sensor weighting, Huber/Cauchy loss, multiple optimizers) sets the accessibility
  bar. FIGAROH differentiators: physical-consistency (inertial manifold), covariance,
  multi-backend, regressor-based (analytical), LuGre. HALO/SPI-Active/Vid2Sid are 2026
  real2sim references.
- **G9 (retargeting→controller interface is the limiting factor):** kinematically
  feasible but dynamically infeasible references are the key failure mode; a robust
  gravity-compensated torque interface + residual learning is the solution.

## Plan (6 phases, RE-PRIORITIZED — deployment layer leads)

> Tracks: **D**eployment (P1, P5), **F**oundation (P2, P3), **S**im2real (P4, P6).
> Oracle gate after each phase — **6 total**.
> **Re-prioritization rationale (G1):** deployment correction is more impactful than
> sim-grounding; the deployment control interface is FIGAROH's headline value prop.

### Phase 1 — Deployment control layer (D) **[LEAD / CRITICAL]**
- **Goal:** run the identified model UNDER a learned policy — FIGAROH's primary interface.
- **Deliver:** `application/control.py` — `GravityCompensationController`, `ComputedTorqueController` (ID + PD),
  `ResidualPolicyInterface` (additive residual torque **+** expressive correction mode per G6);
  `application/estimator.py` — stateless `DynamicsEKF`; expose `M/C/g/J/τ_id/τ_gc/τ_residual` (G7);
  extend `integration/api.py` → `RobotDeploymentSystem`.
- **Serves:** residual RL / torque adaptation at deployment + retargeting tracking controller.
- **Depends:** backend abstraction (exists) + identified params (exist from `BaseIdentification`).
  **No hard dep on export → can lead.** ∥ P2/P3/P4.
- **Oracle gate:** control-law + estimator correctness + residual-interface API design.

### Phase 2 — Sim grounding & model export (F)
- **Goal:** inject identified model into the training sim; (low-pri) principled DR.
- **Deliver:** `tools/model_delta.py` — `ModelDelta` (inertials + offsets + **friction/armature** →
  Pinocchio `armature`/`damping`/`lowerDryFrictionLimit` fields per F5) + provenance + **covariance**
  (already exported elsewhere per the ground-truth correction above — wire it in, don't rebuild it);
  export to URDF/MJCF/USD (reuse h1v2 `urdf_exporter.py` +
  `shared/exporters.py` per F6); `domain_randomization()` sampler (point-estimate-centered; covariance
  optional); positioning ADR. **Shares its URDF/friction-export deliverable with
  `ROADMAP.md` Track D Step 6 and Track E §8.3 — one implementation, not three.**
- **Serves:** match-sim-to-real + (optional) DR around identified params.
- **Depends:** existing exporter + identification output. **∥ P1/P3/P4.**
- **Oracle gate:** public data-format + friction-field semantics.

### Phase 3 — Cross-backend dynamics validation (F)
- **Goal:** prove the identified model is faithfully reproduced in the target training sim.
- **Deliver:** `backends/validation.py` — `compare_{mass_matrix,coriolis,gravity,inverse_dynamics,
  regressor,forward_dynamics}` + `BackendConsistencyReport` (M/C/g <0.1%, regressor <1%).
  **This is `ROADMAP.md` Track B's existing Phase 2.3 backlog item — this spec supersedes
  that item's description rather than existing alongside it as a separate deliverable.**
- **Serves:** trust that a model identified for the real robot behaves identically in MuJoCo/Genesis/Isaac.
- **Depends:** backends exist (Pinocchio+MuJoCo). **∥ P1/P2/P4.**
- **Oracle gate:** numerical correctness + tolerances.

### Phase 4 — Real-robot data plane (S)
- **Goal:** ingest real sysid data (torque/encoder/IMU/F-T logs) into a canonical contract.
- **Deliver:** `integration/adapters/` — `LogAdapter` ABC + `rosbag2.py`, `mujoco.py`, `csv_npz.py`;
  canonical `TrajectoryData` (t,q,dq,ddq,tau,wrench,marker_poses,sampling_frequency,joint_order,units);
  joint/order/unit/frame mapping to URDF; fixture + round-trip test.
- **Serves:** the input to sysid (grounds everything downstream).
- **Depends:** none hard. **∥ P1/P2/P3.**
- **Oracle gate:** data-contract + adapter robustness.

### Phase 5 — Online identification + LuGre friction (D)
- **Goal:** keep the deployed model faithful; frontier actuator-friction differentiator.
- **Deliver:** `identification/online.py` — RLS (forgetting factor) + sliding-window LS reusing
  regressor interface; joint state+param EKF (Riccati-form, align with F3 inertial manifold);
  **LuGre actuator friction identification** from torque data (G5 differentiator); consumes P4 contract.
  **This is `ROADMAP.md` Track A's existing v0.6.1 ("online identification") item — this spec
  (RLS + EKF + LuGre) supersedes that item's original scope rather than sitting beside it.**
- **Serves:** live gap-closing under deployment + high-fidelity actuator sysid.
- **Depends:** P4 (streaming) + existing regressor interface.
- **Oracle gate:** estimation/numerical correctness + LuGre identifiability.

### Phase 6 — End-to-end sim2real loop + whole-body QP retargeting (S)
- **Goal:** prove the full loop; retargeting via WBC (not just ID tracking).
- **Deliver:** `tools/sim2real_gap.py` (torque-prediction RMSE, tracking divergence; <5% target per F7);
  **whole-body QP interface** via crocoddyl `ContactModelMultiple` (G4) or minimal QP wrapper;
  retargeting reference→controller handoff helper; `figaroh-examples` e2e (real log → sysid → export
  delta → sim-ground → policy deploys with computed-torque base + residual → gap report; + retargeting demo).
- **Serves:** the positioning, proven end-to-end.
- **Depends:** P1 + P2 + P3 (+P4/P5 for the full loop).
- **Oracle gate:** e2e correctness + metric validity + WBC interface.

### Dependency / critical path
- Independent: P1 ∥ P2 ∥ P3 ∥ P4.
- P5 → P4. P6 → P1 + P2 + P3 (+P4/P5 optional).
- Critical path: **P1 → P6** and **P4 → P5 → P6**.

## Oracle review budget — **6 total** (one gate after each phase; reasons above)

## Open decisions to iterate (with user)

1. **DR depth:** point-estimate-centered Gaussian (lean, per G2) + covariance export as
   **low-pri differentiator** (per G3). Leaning lean Gaussian first; covariance optional.
2. **Residual interface:** additive residual torque (first) **+ expressive correction mode**
   (history-encoded adaptor / spline-flow per G6/TAM/Warp RL) as second. Leaning additive first.
3. **Retargeting scope:** **whole-body QP via crocoddyl** (per G4) + ID tracking; not ID-only.
   Leaning crocoddyl WBC interface (reuses MPC choice) + minimal QP fallback.
4. **Genesis/IsaacSim backends:** stay in existing Track B backlog (RL training targets are
   Isaac Lab/Genesis — note as dependency, not pulled into this critical path).
5. **Estimation placement:** stateless `DynamicsEKF` in P1; online state+param EKF (Riccati,
   inertial-manifold-aligned per F3) in P5.
6. **MPC:** **Crocoddyl-first** (optional dep, native Pinocchio, IDF-MPC per F1/F4); IPOPT stays
   for OED; acados dropped.
7. **LuGre friction:** in-scope as P5 differentiator (G5), or defer? Leaning in-scope (differentiator).
8. **Ecosystem integration surface (added 2026-08-16, needs deeper research):** how does
   FIGAROH actually plug into the tools practitioners already use for the *actual use case*
   — e.g. **mjlab** (MuJoCo-based RL training/environment tooling) and existing
   **motion-retargeting libraries** — rather than only exposing its own
   `RobotDeploymentSystem`/WBC interface in isolation? Open questions this raises that
   Phases 1–6 above don't yet answer: Does FIGAROH ship an adapter/plugin *into* those
   tools, or does it stay a standalone library those tools import *from*? What's the
   actual integration surface (a Python API, a config/asset format, a sim wrapper)? Who
   is the consumer of `ModelDelta`/`RobotDeploymentSystem` in a real mjlab or retargeting
   pipeline, and does Phase 1's API design (G7) need to be shaped around that consumer
   from the start rather than retrofitted later? Not scoped into any phase above yet —
   flagged as a prerequisite research pass before Phase 1/6 API surfaces are treated as
   final, since G9 already identifies the retargeting→controller handoff as the limiting
   factor and this is the concrete version of that question.

## Phase status

- [ ] Phase 1 — deployment control layer (LEAD)
- [ ] Phase 2 — sim grounding & model export
- [ ] Phase 3 — cross-backend dynamics validation
- [ ] Phase 4 — real-robot data plane
- [ ] Phase 5 — online identification + LuGre friction
- [ ] Phase 6 — e2e sim2real loop + whole-body QP retargeting
- [ ] **Prerequisite research (open decision 8):** ecosystem integration surface
  (mjlab, retargeting libraries) — not started, no phase owns it yet
