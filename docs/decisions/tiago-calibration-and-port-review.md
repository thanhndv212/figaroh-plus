# TIAGo Calibration Analysis & Feature Port Review

**Status:** Part A implemented and closed out (2026-08-07); Part B still a
pending review — no implementation yet.
**Scope:** Two independent TIAGo-focused documents combined here because
both concern the same robot family's calibration pipelines and inform each
other — Part A's base-parameter/redistribution findings are directly
relevant to how Part B's eye-hand calibration port should deploy its
identified corrections.

- Part A source: `TIAGO_CALIBRATION_ANALYSIS.md` (2026-08-05, updated
  2026-08-06/07)
- Part B source: `TIAGO_PORT_REVIEW.md` (2026-06-19)

---

## 0. Cross-cutting takeaway

- **Part A** is a deep structural/statistical dive into the *existing*,
  already-merged TIAGo and TIAGo Pro mocap calibration examples — kinematic
  structure, base-parameter derivation, a real bug fix
  (`calc_stddev`/`_compute_parameter_correlation` ordering), a
  statistical-reliability + held-out-validation study, and, in §A.8, a
  general fix for the "base parameters are combinations, not standalone
  values" problem. §A.8's redistribution work **shipped in core `figaroh`**
  (`redistribute_min_norm`, `propagate_covariance_min_norm`,
  `BaseCalibration.redistribute_parameters()`, HTML report integration,
  `geometric_calibration_export`) — this is done, not proposed.
- **Part B** is an *earlier* (2026-06-19), still-open review of porting two
  unrelated TIAGo features — eye-hand (camera) calibration and mobile-base/
  suspension identification — from old pre-architecture-split branches into
  the current `figaroh` + `figaroh-examples` structure. It ends in open
  questions and a recommended strategy, not code. As of this writing, none
  of Part B's proposed files (`eye_hand_calibration.py`,
  `suspension_identification.py`, `vicon_utils.py`, `suspension_utils.py`,
  the new data directories/configs) exist on `main`.
- **Where they connect:** Part B's eye-hand calibration (§B.1 there) would be
  built on `TiagoCalibration`/`BaseCalibration` — the exact class Part A's
  §A.8 redistribution and `geometric_calibration_export` work already
  targets. When eye-hand calibration is eventually ported, its deploy step
  should reuse `redistribute_parameters()` +
  `export_geometric_calibration_yaml()` rather than reintroducing the
  branch's old one-hot `write_to_xacro` output — the exact "arbitrary
  member of a redundant group absorbs the whole correction" problem Part A
  §A.8.1 diagnoses would otherwise resurface in the ported eye-hand pipeline
  too, since it uses the same `calculate_base_kinematics_regressor`
  machinery.

---

## Implementation status (verified 2026-08-16, against current `main`)

### Part A — implemented and closed out

| Item | Status | Where |
|---|---|---|
| `calc_stddev`/`_compute_parameter_correlation` ordering bug fix (§A.3.1) | ✅ Done | `calibration/base_calibration.py` |
| `TiagoProCalibration` migrated onto `BaseCalibration` (§A.6 item 7) | ✅ Done | `figaroh_tiagoPro` (external repo) |
| `redistribute_min_norm()` / `propagate_covariance_min_norm()` (§A.8.4) | ✅ Done | `tools/qrdecomposition.py:836,867` |
| `BaseCalibration.redistribute_parameters()` (§A.8.4) | ✅ Done | `calibration/base_calibration.py:1547` |
| `base_mapping_matrix` stashed instead of discarded (§A.8.4) | ✅ Done | `calculate_base_kinematics_regressor` |
| HTML report "Redistributed standard parameters" section (§A.8.4) | ✅ Done | `tools/report.py` |
| `geometric_calibration_export` module (§A.8.4) | ✅ Done | `tools/geometric_calibration_export.py` |
| Unit tests (37 new: 12 redistribution + 25 export) | ✅ Present | `test_base_calibration_redistribution.py`, `test_geometric_calibration_export.py` |
| Wired into `figaroh_tiagoPro`'s production deploy script | ⬜ Deferred (§A.8.4) | not started |
| LMI/manifold-constrained recovery for inertial params (§A.8.2 option 2) | ⬜ Deferred (§A.8.4) | pending identification→MPC/RL path |

### Part B — still not implemented

Checked against `figaroh-examples/examples/tiago/` — none of the proposed
files exist:

| Proposed item (§B.4) | Status |
|---|---|
| `eye_hand_calibration.py` | ⬜ Does not exist |
| `suspension_identification.py` | ⬜ Does not exist |
| `utils/vicon_utils.py`, `utils/suspension_utils.py` | ⬜ Do not exist — `utils/` only has `tiago_tools.py`, `simplified_collision_model.py` |
| `data/eye_hand/`, `data/suspension/` | ⬜ Do not exist — `data/` only has `calibration/`, `identification/` |
| `tiago_eye_hand_config.yaml`, `tiago_suspension_config.yaml` | ⬜ Do not exist |
| All 8 decisions in §B.5 | ⬜ Still open — no architecture choice made |
| Phase 1/2/3 port strategy (§B.3) | ⬜ Not started |

---

# Part A — TIAGo vs. TIAGo Pro Mocap Calibration: Structural & Statistical Analysis

Deep-dive comparison of the two `figaroh-examples` calibration pipelines
(`examples/tiago`, `examples/tiago_pro`), covering kinematic structure, base
(identifiable) parameters, calibration results, parameter correlation, and
data-collection design. All numbers below are from real runs of the checked-in
code and data (not simulated), captured 2026-08-05.

> **Updated 2026-08-06** — §A.1.1's TIAGo Pro `pEEx_1`/`pEEy_1` structural
> singularity has since been fixed (config change, not a data-collection
> workaround) and a new, smaller calibration run with full 6-DOF pose
> measurement replaces the position-only one used through §A.1-§A.6. See new
> **§A.7** for the fix, the new run's results, a statistical-reliability study,
> and a held-out validation experiment. §A.1-§A.6 are left as originally written
> (they document the July run and the singularity as it stood then); read
> them together with §A.7's corrections, not as fully superseded. **§A.8** (added
> same day) generalizes a problem implicit throughout §A.2-§A.7 — deployed
> "base parameters" are really linear combinations of standard parameters,
> not standalone values — traces it in FIGAROH's own code, surveys how the
> calibration/identification literature handles it, and recommends a
> redistribution approach given this model's intended consumers (MPC, RL,
> planning).

## A.0 Executive summary

- **Both setups do use a base marker set, contrary to what the CSV column
  count alone suggests.** Per confirmation from the person running the rig:
  the `x1,y1,z1` column figaroh actually reads is not a raw mocap-frame
  reading — it's the end-effector marker's position *already projected into
  the base marker's own rigid-body frame*, computed upstream (e.g. in
  Qualisys Track Manager) before the CSV is written. figaroh never sees the
  raw global mocap coordinates or the base marker's own motion in the room.
  This reframes what `known_baseframe: False` actually estimates: **not**
  "mocap-room-origin → robot base" (an arbitrary, large, session-dependent
  pose), but "base-marker-rigid-body frame → robot's own `base_frame`" — a
  fixed *mounting/alignment offset* between where the base marker plate is
  physically bolted on and where the URDF defines the robot's kinematic
  origin. That's a much better-constrained quantity, and it's consistent
  with TIAGo's small, specific `base_pose` initial guess
  (`[0.09, 0.08, 0.0, -1.57, 0.0, 0.0]` — a plausible plate-mounting offset,
  not "where the robot happened to stand"). It also explains the
  `qualysis_base_hand_calibration.csv` filename and its 4 marker columns
  from earlier (§ not repeated here): most likely raw base-rigid-body +
  hand markers were logged together, with only the already-projected hand
  position (column 1) actually consumed by figaroh. The *math* in the rest
  of this report is unaffected by this (co-estimating an unknown 6-DOF root
  transform works identically regardless of what it physically represents),
  but every "mocap→base" phrase below should be read as
  "base-marker→robot-base."
- **TIAGo Pro has one extra active joint TIAGo doesn't**:
  `gripper_right_tool_mount_joint`, the PAL "ATC" quick-connect tool coupler.
  It's a real, mechanically-rotating joint — just never exercised during data
  collection — which made two marker-offset parameters (`pEEx_1`, `pEEy_1`)
  *exactly* structurally unobservable (SVD singular value ~1e-17) **in the
  July run analyzed below**. TIAGo's tool frame sits on a purely-fixed chain
  past its last actuated joint, so it has no equivalent hard singularity —
  though it does have a strong (0.988), non-singular correlation in the same
  neighborhood (§A.5). **Update 2026-08-05: this singularity has since been
  fixed** by moving `tool_frame` to stop *before* the ATC joint
  (`gripper_right_pal_atc_base_link`) and switching to full 6-DOF pose
  measurement — `pEEx_1`/`pEEy_1` are now free and well-conditioned (SVD
  confirms). See §A.7.
- **A real bug was found and fixed** in `figaroh`'s `BaseCalibration`
  (`base_calibration.py`): the printed "correlated pairs" report always read
  an empty/stale covariance matrix due to a call-order bug, so TIAGo's
  console report claimed **zero** correlated parameters when the true number,
  recomputed directly, is **10** (up to ρ=0.988). This has been fixed in this
  repo checkout (uncommitted) — see §A.4.2.
- With the bug fixed, **both robots show the same qualitative correlation
  signature**: strong pairwise correlation dominated by *adjacent-joint,
  short-link* redundancy, a smaller contribution from *axis-family
  collinearity*, and a base-frame/first-joint coupling cluster common to both.
- **`TiagoProCalibration` has been migrated onto `BaseCalibration`** (same
  pattern as `TiagoCalibration`) — see §A.3. This surfaced a second finding:
  the two robots' RMSE figures as first reported here mixed two different
  RMSE conventions `BaseCalibration` reports (they differ by exactly √3) —
  corrected in §A.3. Same-convention, TIAGo fits ~4× tighter than TIAGo Pro
  (not ~2.3× as first stated) while still being far worse conditioned
  (15,685 vs 264.5) — consistent with TIAGo's checked-in example CSV having
  only 34 of the ~500 samples the config/README describe, the likely
  dominant driver of its conditioning, independent of any structural
  kinematic issue.

---

## A.1 Kinematic structure

### A.1.1 Active joint chains

Both configs use `base_frame: universe` with `known_baseframe: False`
(co-estimated), and a single position-only marker (`measure: [T,T,T,F,F,F]`)
near the wrist/gripper.

| | TIAGo Pro (`universe → gripper_right_tool_holder`) | TIAGo (`universe → wrist_ft_tool_link`) |
|---|---|---|
| Active joints | 9 | 8 |
| Chain | `torso_lift_joint` (P), `arm_right_1..7_joint` (R, `arm_right_6` is `RevoluteUnaligned`), `gripper_right_tool_mount_joint` (**continuous**, RUBZ) | `torso_lift_joint` (P), `arm_1..7_joint` (R, `arm_6` is `RevoluteUnaligned`) |
| Path from last arm joint to tool frame | `arm_right_7_joint` → **continuous joint** → fixed → tool frame | `arm_7_joint` → fixed → fixed → fixed → tool frame |

TIAGo Pro's extra joint is the **PAL ATC (Automatic Tool Changer)** coupler —
confirmed from URDF link/joint naming (`gripper_right_pal_atc_base_link` →
`gripper_right_tool_mount_joint` [continuous] → `gripper_right_tool_mount` →
fixed → `gripper_right_tool_holder`). It's a genuine twist-lock mechanical
interface (hence URDF type `continuous`, no `<limit>`), just not
servo-driven — nothing commands it during normal operation or during this
calibration capture, so every sample records it at q=0.

> **Update 2026-08-05**: fixed, by config rather than by re-locking the
> mechanism. `tiago_pro_calibration_config.yaml`'s `tool_frame` now stops the
> FK chain at `gripper_right_pal_atc_base_link` — *before*
> `gripper_right_tool_mount_joint` — so that joint is excluded from
> `actJoint_idx` entirely and never gets candidate DH parameters of its own;
> `pEEx_1`/`pEEy_1`/`pEEz_1` (now joined by `phiEEx_1`/`phiEEy_1`/`phiEEz_1`,
> since measurement is now full 6-DOF) become the single owner of that whole
> fixed downstream offset (mount + tool_holder + true marker mounting), with
> no other parameter left to be degenerate against. Re-verified by SVD of the
> residual Jacobian at the new solution: `pEEx_1`/`pEEy_1`'s own dominant
> singular values are 8.6/6.1 (comparable to the largest sv, 33.2 — nothing
> close to singular), and their weight in the smallest singular direction is
> ~1e-21, down from ~1.0 in the old setup. See §A.7.

### A.1.2 Axis geometry — parallel/collinear joint families

Computed at neutral pose (world-frame axis directions + perpendicular
distance between axis lines):

**TIAGo Pro:**

| Pair | Angle | Axis-line distance | Relationship |
|---|---|---|---|
| joint 1 ↔ 3 | 0.00° | **0.0 mm** | exactly collinear |
| joint 3 ↔ 5 | 0.00° | **0.0 mm** | exactly collinear |
| joint 1 ↔ 5 | 0.00° | **0.0 mm** | exactly collinear |
| joints {1,3,5} ↔ 7 | 0.00° | 69.9 mm | parallel, not collinear |
| joint 2 ↔ 4 | 180.00° | 332.4 mm | antiparallel, not collinear |
| joint 4 ↔ 6 | 0.18° | 336.1 mm | parallel, not collinear |
| joint 2 ↔ 6 | 179.82° | 665.8 mm | antiparallel, not collinear |

**TIAGo:**

| Pair | Angle | Axis-line distance | Relationship |
|---|---|---|---|
| joint 3 ↔ 5 | 180.00° | **0.0 mm** | exactly collinear |
| joint 3 ↔ 7 | 180.00° | **0.0 mm** | exactly collinear |
| joint 5 ↔ 7 | 0.00° | **0.0 mm** | exactly collinear |
| joint 2 ↔ 4 | 180.00° | 312.1 mm | antiparallel, not collinear |
| joint 1 ↔ 6 | 0.00° | 748.8 mm | parallel (incidental, not a family — far apart in the chain) |
| joint 6 ↔ 7 | — | **0.0 mm origin offset** | share the same pivot point (`arm_7_joint` has zero link length from `arm_6_joint`) |

Both robots have an odd-numbered collinear family (TIAGo Pro: 1-3-5, TIAGo:
3-5-7) and an even-numbered parallel-but-offset family (2-4(-6)) — consistent
with the standard alternating roll/pitch design of 7-DOF anthropomorphic
arms. Link lengths are comparable in scale between the two robots (TIAGo Pro
9.8–20.5cm, TIAGo 9.0–22.5cm, TIAGo's `arm_6→arm_7` is 0cm — a true
wrist-point intersection).

**Effect on calibration (verified against the real fit, §A.5.3): collinearity
is a real but secondary contributor.** The closest collinear pair (TIAGo
Pro's 1↔3) does show elevated correlation (0.81), but it's weaker than most
of the *adjacent*-joint pairs (0.85–0.9998), and pairs further apart in the
same family (3↔5, 1↔5) decorrelate almost entirely (0.44, 0.05) once you look
at the real, workspace-spanning sample set rather than the static neutral
pose — the intervening joint's real motion across the dataset breaks the
zero-pose coincidence.

---

## A.2 Base parameters — linear combinations of independent corrections

### A.2.1 How they're derived

Both pipelines use the same underlying machinery
(`figaroh.calibration.calibration_tools.calculate_base_kinematics_regressor`):
for every joint in the active chain, 6 candidate DH-style correction
parameters are proposed (`d_px, d_py, d_pz, d_phix, d_phiy, d_phiz`  —
translation/rotation offset of that joint's own frame vs. the URDF nominal).
These are reduced in two stages, using **randomized configurations spanning
the whole joint space** (a structural, dataset-independent computation):

1. **`eliminate_non_dynaffect`** — drop any parameter whose effect on the
   6-DOF pose regressor is exactly zero across every random configuration
   (e.g. rotating a joint about its own axis when everything downstream sits
   exactly on that axis).
2. **QR pivoting (`get_baseIndex`)** — among survivors, keep only a maximal
   linearly-independent subset ("base parameters"); this is the same
   technique used for minimal inertial-parameter sets in dynamics
   identification, applied here to kinematics. Anything left is a genuine
   linear combination of the kept parameters — not independently
   identifiable no matter how much data you collect.

### A.2.2 TIAGo Pro — per-joint survivors (traced directly)

> **Update 2026-08-05**: this table reflects the July config
> (`tool_frame: gripper_right_tool_holder`, position-only). Since the
> `tool_frame` change (§A.1.1 update, §A.7), `gripper_right_tool_mount_joint` no
> longer appears in the active chain at all — its row below (2 kept
> parameters) no longer exists — and `pEE*`/`phiEE*` grew from 1 free
> parameter (`pEEz_1` alone) to 6 (`pEEx_1..phiEEz_1`, all free). Total free
> parameters for the new run: 38, not 39. Table left as-is below since it's
> still the correct trace for the July dataset referenced in §A.2-§A.6.

| Joint | Kept (of 6) | Dropped as zero-effect | Dropped as redundant |
|---|---|---|---|
| `torso_lift_joint` | 6/6 (renamed `base_*_torso`, merged with the co-estimated base transform) | — | — |
| `arm_right_1_joint` | 3 — `phix, phiy, phiz` | — | `px, py, pz` |
| `arm_right_2_joint` | 4 — `px, phix, phiy, phiz` | — | `py, pz` |
| `arm_right_3_joint` | 6/6 | — | — |
| `arm_right_4_joint` | 4 — `px, phix, phiy, phiz` | — | `py, pz` |
| `arm_right_5_joint` | 5 — `px, py, phix, phiy, phiz` | — | `pz` |
| `arm_right_6_joint` | 5 — `px, py, phix, phiy, phiz` | — | `pz` |
| `arm_right_7_joint` | 3 — `px, py, pz` | `phiz` | `phix, phiy` |
| `gripper_right_tool_mount_joint` | 2 — `px, py` | `phiz` | `pz, phix, phiy` |

Total: 38 DH parameters + `pEEz_1` (`pEEx_1`/`pEEy_1` are **manually** fixed
at 0 — see §A.1.1/§A.2.4) = **39 free parameters**.

### A.2.3 TIAGo — per-joint survivors (reconstructed from the fitted parameter list)

| Joint | Kept (of 6) |
|---|---|
| `torso_lift_joint` | 6/6 (→ `base_*`, same merge as TIAGo Pro) |
| `arm_1_joint` | 2 — `phix, phiy` |
| `arm_2_joint` | 3 — `px, phix, phiz` |
| `arm_3_joint` | 4 — `px, py, pz, phix` |
| `arm_4_joint` | 3 — `py, phix, phiy` |
| `arm_5_joint` | 6/6 |
| `arm_6_joint` | 2 — `py, phiy` |
| `arm_7_joint` | 3 — `px, py, pz` |

Total: 23 DH parameters + `base_*` (6) + `pEEx_1, pEEy_1, pEEz_1` (all 3 free,
no manual fixing) = **32 free parameters**.

### A.2.4 Comparison

- Both robots merge their first active joint (`torso_lift_joint`) into the
  co-estimated base-frame block — mathematically necessary (it's the first
  joint after the unknown base-marker→robot-base transform, so its own
  placement error is indistinguishable from that transform) — but **only TIAGo Pro's code
  documents this** (`base_*_torso` naming + an explanatory comment). TIAGo's
  generic `base_px..phiz` naming doesn't flag that it also encodes
  `torso_lift_joint`'s DH offset. Small, low-cost documentation gap worth
  porting back.
- The specific per-joint kept/dropped pattern differs between the two robots
  (TIAGo Pro: 3,4,6,4,5,5,3; TIAGo: 2,3,4,3,6,2,3) — this reflects each
  robot's specific joint-offset geometry, not a universal rule; no
  arm-design-independent formula should be assumed from one robot's pattern.
- TIAGo Pro's extra joint contributes 2 identifiable parameters
  (`d_px`/`d_py_gripper_right_tool_mount_joint`) that have **no structural
  analogue in TIAGo** — TIAGo's fixed tool-adapter joints (`arm_tool_joint`,
  `wrist_ft_joint`, `wrist_tool_joint`) get **zero** candidate parameters at
  all (Pinocchio doesn't instantiate joint objects for URDF `fixed` joints).
  Any real assembly/manufacturing offset in TIAGo's tool-adapter chain is
  therefore permanently unmodeled — TIAGo Pro's design can at least partially
  correct for the analogous error in its ATC mount (verified: those 2
  parameters are *not* degenerate with `pEEx_1`/`pEEy_1`, correlation
  0.000000 — the pEE singularity is isolated to those two parameters
  jointly, not shared with the mount's translation). **Update 2026-08-05:
  moot as of the `tool_frame` fix (§A.1.1, §A.7)** — the ATC joint no longer has
  candidate parameters of its own at all, so there's nothing left for
  `pEEx_1`/`pEEy_1` to be degenerate *with* or distinct *from*.

---

## A.3 Calibration results

**Update: `TiagoProCalibration` has since been migrated onto `BaseCalibration`**
(same pattern as `TiagoCalibration` — see §A.3.2), which is what surfaced the
correction below. Numbers in this table are post-migration.

**Correction — the RMSE figures below were originally not apples-to-apples.**
`BaseCalibration`'s quality report actually carries *two* RMSE conventions
that differ by exactly √3, and the two robots' numbers as first reported here
mixed them without realizing it:

- **"flat" convention** — `sqrt(mean(all x/y/z residual components²))`,
  treating every x, y, z value as an independent scalar sample. This is what
  TIAGo Pro's pre-migration bespoke code computed, and what
  `data/calibration_results_20260702_0756.yaml`'s "RMSE 6.46mm" reference
  figure (quoted in class docstrings/README elsewhere in this repo) means.
- **"Position RMSE" (vector-norm) convention** — `sqrt(mean(per-sample ‖x,y,z‖²))`,
  the RMS of each sample's *3D Euclidean distance* error. This is what
  `BaseCalibration`'s standard quality-report "Position RMSE" line and
  TIAGo's driver script both actually print — √3 times larger than the flat
  figure for the same underlying residuals.

TIAGo's originally-reported "2.75mm" is the **vector-norm** figure; TIAGo
Pro's originally-reported "6.46mm" was the **flat** figure. Same-convention
comparison:

| | TIAGo Pro | TIAGo |
|---|---|---|
| Data file | `calibration_samples_20260702_0756.csv` | `qualysis_base_hand_calibration.csv` |
| Samples used | 94 | **34** (config/README describe 500 — this checked-in CSV is a small stand-in, not the production dataset) |
| Free parameters | 39 | 32 |
| Residuals : params ratio | ~7.2 | ~3.2 |
| Position RMSE (vector-norm) | **11.20 mm** | 2.75 mm |
| Position RMSE (flat) | 6.46 mm | 1.59 mm |
| Position MAE (flat) | 4.95 mm | — (not separately reported) |
| Condition number | 264.5 — *moderately conditioned* | 15,685 — *ill-conditioned* (~59× worse) |
| Outliers removed | 0 (of 94) | 1 (of 34, 2.9%) |
| Architecture | `TiagoProCalibration(BaseCalibration)` | `TiagoCalibration(BaseCalibration)` |

Same-convention, the gap is larger than first reported (TIAGo fits ~4× tighter
than TIAGo Pro either way, not ~2.3×) — which *strengthens* §A.5's point rather
than undermining it: TIAGo's fit is tighter **and** its conditioning is worse,
consistent with a small/under-sampled dataset (§A.5) rather than a structural
kinematic problem, since a real structural issue would show up as a *worse*
fit too, not just worse conditioning.

> **Update 2026-08-05**: TIAGo Pro has a newer run (48 samples,
> `calibration_results_20260805_1246.yaml`) using the fixed `tool_frame` and
> full 6-DOF pose measurement (§A.1.1, §A.7) — deliberately **not** folded into
> the table above, since it changes both the active chain (no
> `gripper_right_tool_mount_joint` params) and what's being measured
> (position **and** orientation residuals, vs. TIAGo's still position-only
> here), so a direct RMSE comparison would repeat the exact
> apples-to-oranges mistake this section was written to correct. See §A.7 for
> its own results and reliability analysis.

### A.3.1 A bug found and fixed: correlation reporting was silently broken

While cross-checking the "0 correlated pairs" TIAGo console report against a
from-scratch recomputation of the same covariance matrix, the numbers
disagreed sharply (0 vs. 10 pairs, up to ρ=0.988). Root cause, in
`figaroh/src/figaroh/calibration/base_calibration.py`,
`BaseCalibration._evaluate_solution()`:

```python
# before (buggy order):
correlated_pairs = self._compute_parameter_correlation()  # reads self._C_param
self.calc_stddev(result)                                  # sets self._C_param
```

`_compute_parameter_correlation()` reads `self._C_param`, but that attribute
is only *set* by `calc_stddev()` — which ran one line later. On a freshly
constructed calibration object (the normal case — one `solve()` call per
script invocation), `self._C_param` doesn't exist yet at that point, so
`_compute_parameter_correlation()` hit its `if C_param is None: return []`
guard and **always** reported zero correlated pairs, regardless of the true
correlation structure. This affects every example built on `BaseCalibration`
(TIAGo confirmed; `staubli_TX40`, `talos`, `ur10` share the same `solve()`
path and are likely affected too, though not independently verified here).

**Fix applied** (this repo checkout, uncommitted): swapped the two calls so
`calc_stddev()` runs first. Verified: TIAGo's console report now correctly
prints all 10 correlated pairs (matches the from-scratch computation exactly).
The condition-number figure (254.2 / 15,685) was **not** affected — it's
computed independently from the raw Jacobian, before this bug's code path.

---

## A.4 Parameter correlation analysis

### A.4.1 TIAGo Pro — 25 pairs with |ρ| > 0.8

Dominated by **adjacent-joint, short-link** couplings (`d_py_6↔d_pz_7`:
0.9998, `d_px_2↔d_pz_3`: 0.9996, `d_phiy_3↔d_phiz_4`: 0.9976, …), plus a
base-frame/first-joint cluster (`base_phiz_torso↔d_phiz_arm_right_1`: 0.97),
plus exactly **one** axis-family (collinear) pair, joints 1↔3 at 0.81 — the
weakest entry on the list.

### A.4.2 TIAGo — 10 pairs with |ρ| > 0.8 (post-fix)

```
-0.988   d_px_arm_7_joint   ↔ pEEx_1
-0.983   d_py_arm_6_joint   ↔ d_pz_arm_7_joint
-0.961   d_py_arm_4_joint   ↔ d_pz_arm_5_joint
-0.958   base_phiz          ↔ d_phiz_arm_2_joint
-0.951   base_px            ↔ base_phiy
-0.917   base_phix          ↔ d_phiy_arm_1_joint
+0.906   base_py            ↔ base_phix
-0.905   base_px            ↔ d_phix_arm_1_joint
-0.901   base_py            ↔ d_phiy_arm_1_joint
+0.841   base_phiy          ↔ d_phix_arm_1_joint
```

Same qualitative signature as TIAGo Pro: adjacent-joint short-link pairs
(`d_py_6↔d_pz_7`, `d_py_4↔d_pz_5`), a base-frame/first-joint cluster (6 of
the 10 pairs), and — notably — **`d_px_arm_7_joint ↔ pEEx_1` at −0.988**,
the strongest pair in TIAGo's list. This refines the §A.1 finding: TIAGo's
marker offset isn't subject to TIAGo Pro's *exact* structural singularity
(no unexercised joint sits between the last arm joint and the marker), but
it *is* strongly — just not perfectly — correlated with the last joint's own
translation, because the marker sits relatively close to `arm_7_joint`'s
own frame (which has zero offset from `arm_6_joint`). "No hard joint stacked
in between" bought TIAGo a finite, data-improvable correlation instead of
TIAGo Pro's fixed, unfixable-by-data singular value ~1e-17 — a difference of
degree, not of kind, once you look past the aggregate condition number.

### A.4.3 Ranking the root causes (both robots)

1. **Adjacent-joint short-link redundancy** (dominant, both robots) — a
   correction at joint *i* and a different correction at joint *i+1*
   propagate almost identically to a marker that's far away relative to the
   link length between them. Position-only external metrology can't resolve
   *which of two nearby points* an error originated at.
2. **Base-frame / first-active-joint coupling** (both robots, ~6-8 pairs) —
   structural: the unknown base-marker→robot-base transform and the first joint's own
   offset are adjacent unknowns at the root of the chain with nothing
   upstream to disambiguate them locally.
3. **Axis-family collinearity** (present in both, but the smallest
   contributor, and it decays fast with chain distance — see §A.1.2).
4. **Tip/marker aliasing** — TIAGo Pro: exact singularity (unexercised joint).
   TIAGo: strong but finite correlation (no unexercised joint, but a
   short remaining lever arm).

---

## A.5 Generalizing to optimal static configuration design

Everything above converges on a small number of design principles for
*which* configurations to collect data at (relevant to both robots'
`generate_optimal_configs.py` / `optimal_config.py` D-optimal pipelines):

1. **Sample count relative to parameter count matters more than either
   number alone.** TIAGo Pro's 94 samples / 39 params (~7.2 residuals/param)
   vs. TIAGo's 34/32 (~3.2) tracks its much worse aggregate condition number
   reasonably well on its own, before invoking any structural explanation.
   Rule of thumb from this data: aim for at least ~7 position-residuals per
   free parameter, not the bare minimum needed for a determined system.
2. **Explicitly vary the joint sitting *between* a collinear-axis pair.**
   Collinearity is a zero-pose artifact — the D-optimal search should reward
   configurations where the intervening joint is *far* from the value that
   recreates the collinear alignment, not just reward generic "workspace
   coverage."
3. **Adjacent short-link pairs are a harder, more permanent target.** Unlike
   collinearity, this doesn't fully wash out with configuration diversity —
   it's bounded by the ratio of link length to lever arm to the marker. D-optimal
   design can still help at the margin (favor configurations that maximize
   the *difference* in how joint *i* and *i+1*'s corrections move the marker),
   but expect a floor on how much can be resolved by configuration choice alone.
4. **Exercise every joint that's mechanically capable of moving, even if it's
   not part of the normal task-space trajectory.** TIAGo Pro's ATC coupler
   is the clearest case: it's a real joint, it's just never commanded — a
   protocol that captures a few samples at a different (deliberately
   re-locked) ATC orientation would convert `pEEx_1`/`pEEy_1` from a
   permanent structural assumption into a genuinely estimated quantity.
5. **The base-frame/first-joint cluster is not fixable by configuration
   design alone** — it's a fundamental consequence of co-estimating an
   unknown root transform with nothing further upstream to reference against.
   Since that transform is a *fixed mechanical mounting offset* (base marker
   plate → `base_frame`, not a session-dependent room placement — see §A.0),
   it's plausibly a one-time metrology problem rather than a per-session one:
   a careful CAD/survey measurement of the plate's mounting position could
   supply a genuinely *known* base frame, removing 6 unknowns (and the
   correlated pairs riding on them) from every future calibration run rather
   than re-estimating it from mocap data each time.

---

## A.6 Recommended improvements, prioritized

1. **(Done, this session)** Fix the `calc_stddev`/`_compute_parameter_correlation`
   ordering bug in `figaroh/src/figaroh/calibration/base_calibration.py` —
   currently uncommitted in the `figaroh` checkout. Recommend: commit it,
   and spot-check `staubli_TX40`/`talos`/`ur10` (same `BaseCalibration.solve()`
   path) for the same silently-empty correlation report.
2. **Replace TIAGo's 34-sample demo CSV with the intended ~500-sample
   dataset** (or at minimum a larger, genuinely D-optimal set) — the
   single highest-leverage fix for its condition number, and it's
   orthogonal to any structural kinematic issue.
3. ~~Investigate whether a few TIAGo Pro samples can be collected with the
   ATC mount at a different lock orientation.~~ **Superseded 2026-08-05**:
   resolved a different way — a `tool_frame` config change (stop the chain
   before the ATC joint) plus switching to full 6-DOF pose measurement,
   no mount re-locking needed. See §A.1.1 update, §A.7.
4. **Weight D-optimal config generation (both robots) toward decoupling the
   specific worst-offending adjacent pairs** identified here — TIAGo Pro:
   joints 6-7, 2-3; TIAGo: `arm_6`-`arm_7`, `arm_4`-`arm_5`, and the
   base-frame/`arm_1` cluster — rather than only optimizing a generic
   aggregate condition number, which can hide badly-correlated subsets (as
   seen with TIAGo Pro's "moderately conditioned" 254.2 alongside 25
   individually near-degenerate pairs).
5. **Consider `calib_level: joint_offset` instead of `full_params`** for
   either robot if the resolved correlation structure means the extra DH
   parameters aren't buying real, separable information — a coarser model
   with well-separated parameters can be more trustworthy than a finer one
   riding on regularization to paper over near-collinearity.
6. **Port TIAGo Pro's two genuine documentation/robustness wins back
   upstream**: the explicit `base_*_torso` naming (makes the torso/base
   merge discoverable) and the reasoning behind `_fixed_tip_xy` (a documented
   pattern other `BaseCalibration`-based robots with a similar
   never-exercised terminal joint could reuse, rather than rediscovering the
   SVD/singular-value diagnosis from scratch each time).
7. **(Done, this session)** Migrated `TiagoProCalibration` onto
   `BaseCalibration`, contributing its two genuine customizations as
   documented overrides (`cost_function`/`get_pose_from_measure` for the
   fixed-tip-xy handling, `_pad_csv_missing_joints` for the CSV gap,
   `_detect_outliers` for the absolute-distance threshold,
   `_optimize_with_outlier_removal`/`_compute_condition_number` to keep the
   fixed parameters out of the LM search space and conditioning assessment
   rather than leaving noisy near-zero Jacobian columns in — an early
   attempt that zeroed them inside `cost_function` alone, without reducing
   the actual search dimensionality, produced a nonsensical
   4.8×10^17 condition number from exactly this). Re-verified: RMSE 6.46mm
   (flat) / 11.20mm (vector-norm), condition number 264.5 — closely matches
   pre-migration (6.46mm flat, 254.2) but not bit-identical, since
   `BaseCalibration`'s inherited outlier-removal loop calls the LM solver
   with different options than the original bespoke loop did.

---

## A.7 Update (2026-08-05/06): TIAGo Pro re-calibrated — mount-boundary fix + full-pose measurement

New commits in `figaroh_tiagoPro` (`5b794fe`, `5295b28`, `ea0f7e2`, all
2026-08-05) replace the July TIAGo Pro run analyzed in §A.1-§A.6 with a new one
that fixes the §A.1.1 structural singularity at the source and adds a
statistical-reliability + held-out-validation study on top. Source data:
`calibration_samples_20260805_1246.csv` (48 samples),
`calibration_results_20260805_1246.yaml`,
`calibration_reliability_20260805.md`, `mocap_frame_correction_20260805.md`,
`master_calibration_20260805{,_conservative}.yaml` — all in
`figaroh_tiagoPro/data/`.

### A.7.1 What changed

`tiago_pro_calibration_config.yaml`:

| | July run (§A.1-§A.6) | New run (2026-08-05) |
|---|---|---|
| `tool_frame` | `gripper_right_tool_holder` | `gripper_right_pal_atc_base_link` |
| `measure` | `[T,T,T,F,F,F]` (position only) | `[T,T,T,T,T,T]` (full pose) |
| `tip_pose` seed | `[0,0,0,0,0,0]` | `[0,0,0.141928,0,0,0]` (known mount+holder offset) |
| `_fixed_tip_xy` | `True` (`pEEx_1`/`pEEy_1` fixed at 0) | `False` (all 6 `pEE*`/`phiEE*` free) |
| Samples | 94 | 48 |
| Free parameters | 39 | 38 |

Moving `tool_frame` to stop *before* `gripper_right_tool_mount_joint`
excludes that joint from the active chain entirely — it no longer gets DH
correction candidates of its own, so there's nothing left for `pEE*` to be
degenerate with (§A.1.1, §A.2.4 updates). Switching to full-pose measurement
(the `tiago_endEffector` Qualisys rigid body already reports orientation,
per production code `mocap_ee_publisher.py`/`mocap_mpc_corrector.py` — this
run is the first calibration use of that data) is what makes `phiEE*`
identifiable at all, and gives 6 residual dimensions/sample instead of 3.

**Verified structurally sound, not just assumed**: SVD of the residual
Jacobian at the new solution puts `pEEx_1`/`pEEy_1`'s own dominant singular
values at 8.6/6.1 (vs. the largest sv, 33.2 — not close to degenerate), and
their weight in the smallest singular direction at ~1e-21, down from ~1.0
under the old config.

**Side finding, fixed in the same pass**: naive RPY-difference residuals
have no branch-cut handling — the same physical orientation on either side
of ±180° produces a spurious ~360° residual that dwarfs every other term.
Caught on real data: sample #17 of the new dataset showed a naive residual
norm of 356° against a true geodesic orientation error of 4.1°. Fixed by
wrapping rotation-block residuals into `[-π, π]` before computing distances
(`_wrap_orientation_residual` in `run_calibration.py`); outlier detection
was also split into separate position/orientation distance checks
(`_position_orientation_dist`), since the old single position-only check
would have let orientation outliers like sample #17 through unnoticed.

### A.7.2 Results

`calibration_results_20260805_1246.yaml`: **Position RMSE 7.93mm / MAE
7.28mm, orientation RMSE 2.18° / MAE 1.89°**, 48 samples, 38 free
parameters (288 residuals, 250 dof).

### A.7.3 Statistical reliability — most translations are noise at this sample count

`calibration_reliability_20260805.md` computes standard error per parameter
from `cov = σ²·(JᵀJ)⁻¹` at the solution (`σ² = SSR/dof`) and buckets each
into `solid` (≥3σ), `acceptable` (2-3σ), `marginal` (1-2σ), or `noise` (<1σ,
not distinguishable from 0 with this data):

**8 solid, 4 acceptable, 4 marginal, 22 noise — out of 38.**

- **Almost all per-joint translations (`d_px`/`d_py`/`d_pz`) are noise** —
  the only exception is `base_pz_torso` (2.52σ, acceptable).
- **Yaw rotations (`d_phiz`) are almost systematically solid** (5.7-11.6σ
  wherever identified): `d_phiz_arm_right_3` 9.70σ, `d_phiz_arm_right_4`
  11.57σ, `d_phiz_arm_right_6` 5.68σ, `d_phiz_arm_right_7` 11.31σ — joined
  by a few isolated roll/pitch entries in the same joints:
  `d_phiy_arm_right_6` 11.59σ (solid), `d_phix_arm_right_6` 2.74σ
  (acceptable), `d_phix_arm_right_3` 2.08σ (acceptable).
- Marker/tip params: `pEEz_1` 49.27σ, `phiEEz_1` 23.16σ, `phiEEx_1` 5.58σ —
  all solid; `phiEEy_1` 2.67σ acceptable; `pEEx_1`/`pEEy_1` 1.67σ/0.46σ —
  not reliable yet (§A.7.4).
- Reads the same way §A.5 predicted: translations are statistically harder
  to separate from mocap noise than rotations, and 48 samples for 38
  parameters (~7.6 residual-dims/param, but only 1.26 samples/param) is
  thin — more pose diversity is needed to pin down the noise/marginal set.

### A.7.4 Mocap marker-frame correction (`mocap_frame_correction_20260805.md`)

Goal: redefine the QTM `tiago_endEffector` rigid body's local origin to
directly report `gripper_right_tool_holder`'s pose. Per-axis reliability
mirrors §A.7.3's pEE/phiEE rows: `z` (49.3σ), `roll`/`yaw` (5.6σ/23.2σ) are
solid; `pitch` is marginal-to-acceptable (2.7σ); **`x`/`y` are not
reliable** (1.67σ/0.46σ — uncertainty on the same order as the value
itself, or larger). Physically sensible: a lateral (X/Y) marker offset has
a subtler effect on measured position than an axial (Z) offset or a yaw
rotation.

**Recommendation: apply Z + the 3 rotations, leave X/Y at 0** — z ≈
+4.77mm, RPY ≈ (+0.95°, +0.33°, -2.47°), quaternion
`[0.008314, 0.002692, -0.021566, 0.999729]`. The full identified value
(X/Y included) composes back to identity when checked against its own
inverse (not a math error), but per §A.7.5's validation, deploying the
noise-level X/Y measurably hurts generalization rather than just adding
theoretical risk.

### A.7.5 Held-out validation: conservative subset beats the full fit

Rather than trust the σ-thresholds alone, they were tested directly: 48
samples split 36 train / 12 held-out test (random, no overlap), fitting
both the full 38-param model and a conservative model (only the 12
parameters at ≥2σ across *all* categories — base frame, joints, and
marker/tip — left free, everything else pinned at 0) on train only, then
scoring FK error on the never-seen test set. Repeated over 4 independent
splits (seeds 0-3):

| | Position — conservative | Position — full | Orientation — conservative | Orientation — full |
|---|---|---|---|---|
| Mean (4 splits) | **21.2 mm** | 34.5 mm | **5.09°** | 6.27° |
| Range | 19.7-22.7 mm | 24.9-49.9 mm | 3.88-7.23° | 4.76-9.66° |

(Uncalibrated nominal model, same test splits, for reference: ~139mm /
~5.6°.)

**Consistent across all 4 splits, no exception**: the conservative model
wins on held-out data (~1.6× better position error on average) *and* is far
more stable split-to-split (a ~2× range for the full model vs. a tight
band for the conservative one) — the standard signature of the full model
overfitting the noise-level parameters rather than learning anything that
generalizes. This is a real behavioral finding, not just a statistical-
purity preference.

### A.7.6 What to deploy

- **Joint corrections**: `master_calibration_20260805_conservative.yaml` —
  the 7 of the 12 ≥2σ parameters that are genuine per-joint URDF DH
  corrections (`d_phix_arm_right_3`, `d_phiz_arm_right_3`,
  `d_phiz_arm_right_4`, `d_phix_arm_right_6`, `d_phiy_arm_right_6`,
  `d_phiz_arm_right_6`, `d_phiz_arm_right_7`). The other 5 of the 12
  (`base_pz_torso`, `pEEz_1`, `phiEEx_1`, `phiEEy_1`, `phiEEz_1`) aren't
  pure per-joint corrections — `base_*_torso` is mathematically merged with
  the co-estimated mocap→base transform (§A.2.4), and the marker/tip terms
  belong in the mocap rigid-body definition, not the URDF — so they're
  applied separately, per below. `master_calibration_20260805.yaml` (all 38,
  including noise-level ones) is kept for reference but **not
  recommended** — see §A.7.5.
- **Mocap rigid-body redefinition**: apply §A.7.4's Z + 3-rotation correction
  to `tiago_endEffector` in QTM; leave X/Y at 0 until more data arrives.

### A.7.7 Next step

Both §A.7.3 and §A.7.4 point at the same fix: recollect data toward ~96 samples
(close to the July run's 94, using the now-fixed config) — pipeline already
in place: `generate_optimal_configs.py` → `collect_calibration_data.py` →
`run_calibration.py`. Since `σ_error ∝ 1/√N`, doubling 48→96 would cut
standard errors by ~30%, plausibly pushing several `marginal`/`acceptable`
parameters to `solid` and some `noise` parameters to `marginal` — most
importantly, enough to finally get a reliable read on `pEEx_1`/`pEEy_1`
(§A.7.4), the one pair still unresolved now that the *structural* blocker
(§A.1.1) is gone and only a *statistical-power* one remains.

---

## A.8 Base parameters vs. standard parameters: the redistribution problem, and what the literature does about it

A general problem underlying §A.2-§A.7, made explicit here: everywhere this
document says a joint "keeps" N of its 6 candidate parameters, what's
actually identifiable is a **linear combination** of the original 6
(standard) parameters, not necessarily the N individual ones under their
own names. Deploying the fitted value under one representative name, with
the rest of its group implicitly at 0, is one choice among infinitely many
that reproduce the same fit — worth understanding precisely, since every
`master_calibration_*.yaml` in `figaroh_tiagoPro/data/` is built this way.

### A.8.1 The problem, traced in FIGAROH's own code

`calculate_base_kinematics_regressor` (`figaroh/src/figaroh/calibration/calibration_tools.py:738`)
runs two structural (data-value-independent) reduction steps — §A.2.1 already
describes them at a high level; here's the precise mechanism:

1. `eliminate_non_dynaffect` (`regressor.py:251`) drops candidate columns
   with ~zero effect across random configurations.
2. `get_baseIndex`/`get_baseParams` (`qrdecomposition.py:810`/`796`, via
   `QRDecomposer._identify_base_parameters`) run column-pivoted QR on the
   survivors and keep a maximal linearly-independent subset (`base_indices`).
   Everything else (`regroup_indices`) is an **exact** linear combination of
   the kept columns — not merely small, genuinely redundant.

`QRDecomposer` computes the actual combination — a second QR on
`[W_base, W_regroup]` gives `β = R1⁻¹R2` (`qrdecomposition.py:438`), and
`_build_parameter_expressions` (line 551) builds the textbook formula
(Gautier & Khalil's base-parameter framework):

$$\phi_{base,i} = \theta_{std,i} + \sum_j \beta_{ij}\,\theta_{std,\,dep_j}$$

**But this expression is computed, then discarded.** `calculate_base_kinematics_regressor`
returns it as `paramsrand_base` (line 834), and `base_calibration.py:441`
unpacks it — and never references it again. What actually gets stored as
`calib_config["param_name"]` is just the bare representative name of each
kept column (line 830-832: `paramsrand_e[j] for j in idx_base`).

This isn't merely a labeling gap: it's mathematically forced that the
*fitted value* is already the full combination, whether or not it's
labeled as such. Since every dropped column satisfies
`R_e[:,dep_j] = R_e[:,base] @ β[:,j]` exactly, regressing residuals against
`R_b` (the base columns alone — dependent columns are omitted from the
regressor entirely, not blended in) necessarily converges to
`θ_base + β·θ_dep`, not bare `θ_base`. The one case this document makes
*visible* is `torso_lift_joint`'s merge into `base_*_torso` (§A.2.4) — same
mechanism, just renamed to flag it; every other redundant group is folded
in silently under an unrelated joint's own name.

### A.8.2 How the literature handles the same problem (dynamics side, more mature)

Robot dynamics identification faces an identical structural issue —
recovering individual link mass/COM/inertia ("standard parameters") from
identified base parameters — and has three escalating answers:

1. **Minimum-norm pseudoinverse recovery** (classical, decades-old,
   e.g. [*Identification of consistent standard dynamic parameters of
   industrial robots*](https://ieeexplore.ieee.org/document/6584295/),
   [*Reduction of robot base parameters*](https://www.osti.gov/etdeweb/servlets/purl/455089)) —
   spread the fitted combination across its group via `M⁺` (the
   Moore-Penrose pseudoinverse of the base-mapping matrix) instead of a
   one-hot pick. Free in the sense that `M @ (M⁺·φ_base) = φ_base` exactly —
   predictions are unchanged, only the individual numbers change. Known
   failure mode, stated explicitly in the literature: no guarantee of
   physical plausibility (can return e.g. a negative rotor inertia).
2. **Physical-consistency-constrained recovery** — current state of the
   art for dynamics specifically. Reformulate the recovery as a constrained
   optimization over the manifold of physically-realizable rigid bodies
   (positive mass, positive-definite pseudo-inertia) instead of fitting
   unconstrained then hoping:
   - Wensing, Kim & Slotine, [*Linear Matrix Inequalities for
     Physically-Consistent Inertial Parameter Identification*](https://arxiv.org/abs/1701.04395) (2017) —
     physical consistency as an LMI, identified via SDP.
   - Traversaro et al., [*Identification of Fully Physical Consistent
     Inertial Parameters using Optimization on Manifolds*](https://arxiv.org/pdf/1610.08703) (2016) —
     same goal via Riemannian manifold optimization.
   - [*Sequential semidefinite optimization for physically and
     statistically consistent robot identification*](https://www.sciencedirect.com/science/article/abs/pii/S0967066120302690) —
     combines both consistency criteria in one SDP pipeline.
3. **Bayesian / prior-regularized identification** — treat CAD-derived
   nominal values as a real prior (mean + covariance), do MAP estimation
   instead of plain least squares, so non-identifiable directions default
   *softly* toward the prior with a defensible uncertainty rather than a
   hard 0 or an arbitrary split. Classical line: Ting, Mistry, Peters &
   Schaal. Active: [*A two-stage Bayesian framework for rapid dynamics
   identification*](https://onlinelibrary.wiley.com/doi/10.1002/rnc.7547) (2024),
   [*A Bayesian learning approach for dynamic parameter
   identification*](https://www.sciencedirect.com/science/article/abs/pii/S0045782525002233) (2025),
   [*Zero-Shot Parameter Learning of Robot Dynamics Using Bayesian
   Statistics and Prior Knowledge*](https://arxiv.org/html/2506.19350) (2025).
   Very recent alternative formalism, not yet load-bearing:
   [*A Geometric Method for Base Parameter Analysis via Projective
   Geometric Algebra*](https://arxiv.org/pdf/2509.02071) (Sept 2025).

**Notably, §A.7.5/§A.7.6's conservative-subset approach is already an ad-hoc,
hard-threshold version of option 3** — it shrinks statistically-insignificant
parameters to their prior (0) instead of trusting a noisy point estimate,
just via a binary ≥2σ cutoff rather than continuous MAP shrinkage. (This is
a related but *distinct* issue from §A.8.1's structural redundancy: §A.7.3's
noise/marginal parameters mostly did survive QR as independent base
parameters — they're just weakly constrained by only 48 samples, not
degenerate with anything.)

### A.8.3 How the literature handles it (kinematic/geometric side — mostly avoided, not solved)

Classical kinematic calibration theory sidesteps this problem rather than
solving it: it imposes a "minimality principle" (completeness + continuity
+ **minimality**) on the model up front, i.e. use standard DH (4
parameters/joint, by construction non-redundant) specifically so there's
never a combination to split later. FIGAROH's choice — 6 candidates/joint,
reduced after the fact (§A.2.1) — is the opposite move, trading DH's
near-parallel-axis pathologies (the motivation for Hayati/CPC-style fixes)
for exactly the redistribution ambiguity discussed here.

For deployment, the field draws an explicit **direct vs. indirect
compensation** distinction ([Robot calibration — overview](https://en.wikipedia.org/wiki/Robot_calibration);
industrial-calibration survey literature):
- **Direct compensation** — write identified deltas into the
  controller/URDF's individual kinematic parameters. What
  `master_calibration_*.yaml` does today; exactly where §A.8.1's ambiguity lives.
- **Indirect compensation** — keep the correction as a standalone error
  model, applied to the *commanded target pose* at runtime; the underlying
  kinematic parameters are never touched, so the redistribution question
  never has to be answered.

### A.8.4 What this means for FIGAROH, given the intended consumers (MPC, RL, planning)

Indirect compensation is not viable here: MPC's internal dynamics
linearization, an RL sim's physics step, and a planner's IK/collision
check all need a complete, internally-consistent model object (Pinocchio
model / MuJoCo XML / URDF) for their *own* rollouts — there's no hook to
apply a "correct the output pose afterward" function inside those solvers.
Direct compensation is effectively required; the redistribution question
has to be answered, not dodged.

- **Geometric parameters (this document's scope)**: no physical-validity
  manifold to violate — any real-valued joint offset keeps kinematics
  well-posed. **Minimum-norm (`M⁺`) redistribution is sufficient** — free
  relative to today's one-hot deploy (identical FK/Jacobian predictions,
  per §A.8.2 option 1), and removes the arbitrary "one joint absorbs the
  whole group" artifact that could otherwise bias a planner's per-joint
  offset assumptions or a linearized Jacobian at a specific joint.
- **Dynamics/inertial parameters (not yet in scope here, but on the same
  path)**: physical consistency is a hard requirement, not a nicety —
  Pinocchio's ABA/CRBA and MuJoCo/PyBullet's physics steps assume a
  positive-definite mass matrix; a plain-pseudoinverse recovery that hands
  back a non-PD inertia can crash or destabilize exactly the MPC/RL
  consumers this is being built for. Use §A.8.2 option 2 (LMI/manifold-
  constrained recovery) once FIGAROH's identification output feeds MPC/RL,
  not unconstrained minimum-norm.
- **Higher-leverage than either redistribution scheme**: export the
  covariance FIGAROH already computes (`BaseCalibration.calc_stddev`'s
  `_C_param` — the same matrix behind §A.7.3's σ-verdicts) alongside the
  redistributed point estimate, not just the point estimate alone.
  Robust/tube-MPC formulations and RL domain-randomization pipelines both
  consume parameter *uncertainty* directly — a real identified covariance
  is strictly more useful to them than a single best guess, and turns
  §A.8.1's ambiguity from a nuisance into the exact signal those consumers
  want (which directions are underdetermined, and by how much).

**Implemented (2026-08-07), core `figaroh` only** (scope decision: leave
`figaroh_tiagoPro`'s separate production deploy script untouched for now —
a natural follow-up once this proves useful):

- `figaroh.tools.qrdecomposition.redistribute_min_norm(M, phi_base)` /
  `propagate_covariance_min_norm(M, C_base)` — the `M⁺`-based
  redistribution and covariance-propagation primitives, as pure functions.
- `calculate_base_kinematics_regressor` now stashes the structural
  base-mapping matrix into `calib_config["base_mapping_matrix"]`
  (+ `base_mapping_param_names`/`base_mapping_row_names`/`base_mapping_slice`)
  instead of computing then discarding it — additive-only, no return-signature
  change, so no existing caller (including `figaroh_tiagoPro`) is affected.
- `BaseCalibration.redistribute_parameters()` — opt-in method, callable
  after `solve()`, returning `{name: {"value", "std_dev"}}` for every
  standard parameter in the reduced set, not just the base subset.

Verified on a small 2-joint synthetic case with a genuine, deterministic
redundancy (`d_px_joint2`/`d_phix_joint2` reduce to exact duplicates of
`d_px_joint1`/`d_phix_joint1`, coefficient 1.0): redistribution now assigns
both members of the pair the same value (previously: one at the fitted
value, the other implicitly 0), and round-trips exactly to the original
fitted base-parameter combination. 12 new unit tests across
`test_qr_decomposition.py`, `test_calibration_tools.py`, and the new
`test_base_calibration_redistribution.py`; full suite unaffected (457
passed, same 4 pre-existing `cyipopt`-unrelated failures).

**Follow-up, implemented same day**: the two natural next steps —
surfacing redistribution where it's actually visible, and turning it into
a real deploy artifact:

- The HTML calibration report (`figaroh.tools.report.generate_calibration_report`)
  now has a "Redistributed standard parameters" section, calling
  `redistribute_parameters()` on the live calibrator and rendering it
  through the same table used for the base-only fit — degrades to a muted
  message rather than failing when unavailable for a given run.
- `figaroh.tools.geometric_calibration_export` — generalizes the PAL
  Robotics `robot_state_publisher.geometric_calibration` runtime-correction
  YAML (the format every `figaroh_tiagoPro/data/master_calibration_*.yaml`
  to date was hand-curated into) into a reusable function,
  `build_geometric_calibration()`/`export_geometric_calibration_yaml()`,
  built on the *redistributed* parameter set rather than the base-only fit
  — so a joint silently left at nominal in today's manual process now gets
  its share of the identified correction. Reuses
  `urdf_exporter._parse_param_name` for joint/axis parsing (no re-derived
  parsing logic), and takes an optional `min_sigma` threshold reproducing
  the "conservative" (≥2σ) variant §A.7.5/§A.7.6 validated as generalizing
  better than deploying every identified value regardless of significance.
  The co-estimated-base-merged block is excluded structurally (via
  `base_mapping_row_names`, which is never renamed) rather than by name
  pattern, so it doesn't matter whether a subclass renames those slots to
  `base_px` or `base_px_torso` or leaves them alone.

25 new unit tests (`test_report.py`, new `test_geometric_calibration_export.py`).

**Still deferred**: wiring this into `figaroh_tiagoPro`'s actual production
deploy script (the module above is usable from there today, just not yet
called), and the LMI/manifold-constrained dynamics recovery (§A.8.2 option 2)
— pending the identification→MPC/RL path actually being built.

---

## A.9 Appendix — reproduction

All figures in §A.1-§A.6 were computed directly against the checked-in code/data
in this session (2026-08-05), using the `figaroh-dev` conda environment. §A.7's
figures instead come straight from the checked-in outputs of commits
`5b794fe`/`5295b28`/`ea0f7e2` in `figaroh_tiagoPro` (2026-08-05) — no
independent recomputation, since those already ship the fitted results,
per-parameter standard errors, and validation-split table as committed
markdown/YAML. Key scratch scripts for §A.1-§A.6 (not part of the repo, for
reference only):

- Base-parameter elimination trace: replicated
  `calculate_base_kinematics_regressor`'s internal steps for both robots,
  capturing `eliminate_non_dynaffect`/`get_baseIndex` output per candidate
  parameter.
- Axis geometry: Pinocchio `forwardKinematics` at `q = pin.neutral(model)`,
  world-frame joint axis directions + perpendicular distance between axis
  lines for every joint pair.
- Correlation matrices: `C = (cost²/dof) · pinv(JᵀJ)` from each robot's final
  LM `result.jac`, `corr = C / outer(sqrt(diag(C)), sqrt(diag(C)))` — same
  formula `BaseCalibration.calc_stddev`/`TiagoProCalibration._compute_uncertainty`
  already use internally; recomputed standalone for TIAGo to catch the §A.3.1 bug.

---

# Part B — TIAGo Feature Port Review: Eye-Hand Calibration & Mobile-Base Identification

> **Status:** Review document — NO implementation yet. Iterate before proceeding.
> **Date:** 2026-06-19
> **Source branches:** `pal-tiago-calib` (eye-hand), `tiago-suspension-calib` (suspension/mobile-base)
> **Target:** `figaroh/` (core) + `figaroh-examples/` (examples)

## B.0 Executive Summary

Two features are being ported from ancient branches in the `figaroh` repo. Both predate the architecture split (examples were inside `figaroh/`; now they live in `figaroh-examples/`) and the base-class refactoring (`BaseCalibration`, `BaseIdentification` template methods, unified config parser, backend abstraction).

| Feature | Branch | Core changes? | Risk | Effort |
|---------|--------|---------------|------|--------|
| **E: Eye-hand calibration** | `pal-tiago-calib` | None (style-only diff) | Low | Medium |
| **D: Mobile-base / suspension ID** | `tiago-suspension-calib` | Yes — but mostly already merged | Medium-High | High |

**Key finding:** Much of the suspension branch's core work (`xyzquat_to_SE3`, `non_geom` config, multi-marker in `update_forward_kinematics`) is **already in the current codebase**. The remaining unmerged piece is `update_forward_kinematics_2()` (multi-marker + base_placement). The `robot.py` try-except guards are obsolete — that logic was refactored into `regressor.py` with a proper dataclass.

---

## Part B.1: Eye-Hand Calibration (Feature E)

### B.1.1 What the branch contains

**Source:** `pal-tiago-calib` branch, `examples/tiago/` directory (old layout)

| Item | Path in branch | Lines | Notes |
|------|---------------|-------|-------|
| Calibration script | `examples/tiago/calibration.py` | ~120 | Custom cost function with regularization |
| Config (hey5) | `examples/tiago/config/tiago_config_hey5.yaml` | 63 | Eye-hand with hey5 gripper |
| Config (hey5_center) | `examples/tiago/config/tiago_config_hey5_center.yaml` | 63 | Chessboard center variant |
| Config (hey5_topleft) | `examples/tiago/config/tiago_config_hey5_topleft.yaml` | 63 | Chessboard topleft variant |
| Config (mocap) | `examples/tiago/config/tiago_config_mocap.yaml` | 62 | Mocap variant (no camera) |
| Config (palgripper) | `examples/tiago/config/tiago_config_palgripper.yaml` | 62 | PAL gripper variant |
| Config (schunk) | `examples/tiago/config/tiago_config_schunk.yaml` | 62 | Schunk gripper variant |
| Eye-hand data | `examples/tiago/data/eye_hand_calibration/` | ~18 rows each | 10 CSV files |
| Offset output | `data/calibration_parameters/offset.xacro` | 16 | XACRO properties |
| Master calib YAML | `data/calibration_parameters/tiago_master_calibration_*.yaml` | ~20 each | Identified params |
| Zero model | `data/calibration_parameters/zero_model_tiago.yaml` | 70 | Nominal joint placements |
| Docs | `doc/pal_calib_adaptation.md` | 46 | PAL pipeline adaptation notes |
| Docs | `doc/figaroh_tiago_description.md` | 347 | Full TIAGo workflow guide |

### B.1.2 How it works

**Kinematic chain:** `xtion_rgb_optical_frame` (camera) → `head_2_joint` → `head_1_joint` → `torso_lift_joint` → `arm_1..7_joint` → `hand_tool_link` (end-effector)

**Calibration parameters:**
- Camera pose (6 DOF: xyz + rpy) relative to `head_2_link`
- Joint angle offsets for `head_1`, `head_2`, `torso_lift`, `arm_1..7` (9 offsets)
- Tool tip pose (6 DOF) — optional, per gripper variant

**Data format (CSV):**
```
x1,y1,z1,phix1,phiy1,phiz1,arm_1_joint,...,arm_7_joint,head_1_joint,head_2_joint
```
- 6 DOF marker pose (position + Euler angles) from ARUCO/chessboard detection
- 9 joint angles from encoders
- ~18 samples per recording

**Cost function** (custom, not using `BaseCalibration.cost_function`):
- Residual: measured marker pose − FK-predicted pose
- Regularization terms:
  - Chessboard orientation toward known nominal (`[0, -π/2, -π/2]`)
  - Head joints toward zero (small weight)
  - Camera pose toward URDF nominal (weight 1.0)
  - arm_1/arm_2 toward small offsets, arm_3 toward zero (weight 0.5)
- Solver: `scipy.optimize.least_squares` with `method="lm"`

**Output:**
- `tiago_master_calibration.yaml` — camera pose + joint offsets + tip pose
- `offset.xacro` — XACRO properties for URDF update

### B.1.3 Reconciliation with current architecture

| Aspect | Branch (old) | Current | Action |
|--------|-------------|---------|--------|
| Config format | Legacy flat YAML (`calibration:` / `identification:` sections) | Unified config with `extends:` inheritance | **Convert to unified format** — `tiago_unified_config.yaml` already has a partial `eye_hand_calibration` variant (lines 177–192) but references non-existent `gripper_link` |
| Import paths | `from tiago_tools import TiagoCalibration, load_robot, write_to_xacro` (relative) | `from utils.tiago_tools import ...` (package-relative) | **Update imports** |
| Base class | Old `TiagoCalibration` with `self.param` dict | Refactored `BaseCalibration` with `self.calib_config` dict, template methods, `ResultsManager` | **Adapt subclass** — current `TiagoCalibration` in `tiago_tools.py` already exists and works; eye-hand just needs a different config + cost function variant |
| FK function | `update_forward_kinematics()` | `calc_updated_fkm()` (renamed, backend-aware) | **Update call** — current `TiagoCalibration.cost_function` already uses `calc_updated_fkm` |
| Robot loading | `load_robot("data/urdf/tiago_48_hey5.urdf", load_by_urdf=False)` | `load_robot(urdf, load_by_urdf=True, robot_pkg="tiago_description")` | **Update to use symlinked URDF** from `models/` |
| `write_to_xacro` | Custom function in `tiago_tools` | Not in current `tiago_tools.py` | **Port or replace** — current examples use `update_model.py` pattern (UR10 has one) |
| Core library | No functional changes | — | **Nothing to port** |

### B.1.4 Config schema — eye-hand specific fields

From the branch configs, the eye-hand calibration needs these fields in the unified config:

```yaml
tasks:
  calibration:
    calib_level: joint_offset          # or full_params
    kinematics:
      base_frame: xtion_rgb_optical_frame   # camera optical frame (chain start)
      tool_frame: hand_tool_link            # end-effector (chain end)
      free_flying_base: false
    eye_hand:
      enabled: true
      camera_frame: xtion_link              # camera mount frame
      reference_frame: head_2_link          # parent joint of camera
      camera_pose: [0.0908, 0.08, 0.0, -1.57, 0.0, 0.0]   # nominal [xyz, rpy]
      tip_pose: [0.2163, 0.03484, 0.004, 0.0, -1.57, -1.57]  # nominal tool tip
    measurements:
      markers:
        - ref_joint: arm_7_joint
          measure: [true, true, true, true, true, true]  # 6 DOF
    data:
      file: data/eye_hand/hey5_cb_center.csv
    optimization:
      coeff_regularize: 0.0
      outlier_eps: 0.05                    # meters
```

The current `base_robot_config.yaml` template already defines `tasks.calibration.eye_hand` with `enabled`, `camera_frame`, `reference_frame` — but is missing `camera_pose`, `tip_pose`, and the marker `measure` field for 6-DOF.

### B.1.5 Risks (eye-hand)

| Risk | Severity | Mitigation |
|------|----------|------------|
| Config conversion errors (legacy → unified) | Medium | Test with one variant first (hey5_center) |
| `gripper_link` referenced in current unified config but doesn't exist in URDF | Low | Fix to `hand_tool_link` or `wrist_ft_tool_link` |
| `write_to_xacro` not in current codebase | Low | Port from branch or use `update_model.py` pattern |
| Custom cost function bypasses `BaseCalibration.solve()` | Medium | Either: (a) keep as standalone script, or (b) integrate as `cost_function` override — **decision needed** |
| Data files are small (~18 samples) | Low | Acceptable for calibration; document as example data |
| 6-DOF marker measurement (position + orientation) — current `load_data` may only handle position | **Medium** | Need to verify `data_loader.py` handles 6-DOF markers; the branch config uses `measure: [True, True, True, True, True, True]` |

### B.1.6 Open questions (eye-hand)

1. **Standalone script or BaseCalibration subclass?** The branch uses a custom `least_squares` call with a complex multi-term cost function that doesn't fit cleanly into `BaseCalibration.cost_function`. Should we:
   - (a) Keep it as a standalone script that uses `TiagoCalibration` for data loading + FK but does its own optimization?
   - (b) Refactor the cost function into `TiagoCalibration.cost_function` and use `BaseCalibration.solve()`?
2. **Which gripper variant(s) to port?** The branch has 6 config variants (hey5, hey5_center, hey5_topleft, mocap, palgripper, schunk). Port all, or just hey5_center as the reference?
3. **XACRO output or update_model.py?** The branch writes `offset.xacro`. Current examples (UR10) use `update_model.py` to write updated URDF directly. Which pattern for TIAGo?
4. **Does `data_loader.py` support 6-DOF markers?** The eye-hand data has `phix, phiy, phiz` columns. Need to verify the loader handles orientation measurements.
5. **URDF variant:** Branch loads `tiago_48_hey5.urdf` (a specific variant). Current setup uses symlinked `tiago.urdf` from `models/tiago_description/`. Does the standard URDF have the xtion camera frame? (Yes — confirmed: `xtion_rgb_optical_frame` exists in `tiago.urdf`)

**Cross-reference:** whichever deploy path is chosen (§B.1.6 Q1/Q3), the
identified corrections should flow through the redistribution machinery Part
A §A.8.4 already shipped (`redistribute_parameters()` +
`geometric_calibration_export`) rather than a fresh one-hot `write_to_xacro`
— eye-hand calibration builds on the same `calculate_base_kinematics_regressor`
base-parameter reduction, so it inherits the identical "combination, not
standalone value" ambiguity Part A §A.8.1 diagnoses.

---

## Part B.2: Mobile-Base / Suspension Identification (Feature D)

### B.2.1 What the branch contains

**Source:** `tiago-suspension-calib` branch, `examples/tiago/` directory (old layout)

| Item | Path in branch | Lines | Notes |
|------|---------------|-------|-------|
| Suspension ID script | `examples/tiago/suspension_identification.py` | 370 | Main script — free-flyer + spring-damper |
| Processing utils | `examples/tiago/processing_utils.py` | 1068 | Monolithic: data I/O, filtering, frame math, cost function |
| Self notes | `examples/tiago/self_notes.md` | 170 | Methodology + `update_forward_kinematics_2` docs |
| Config | `examples/tiago/config/tiago_config.yaml` | ~50 | Added `non_geom`, `base_frame` fields |
| Extract scripts | `examples/tiago/extract_adream.py` | 178 | ADREAM rosbag extraction |
| Extract scripts | `examples/tiago/extract_creps.py` | 235 | CREPS + Vicon extraction |
| Extract scripts | `examples/tiago/extract_rosbag.py` | 262 | Rosbag → suspension ID |
| Vicon data | `examples/tiago/data/vicon_calibration_*.csv` | ~22 rows each | 24 CSV files |
| Optitrack data | `examples/tiago/data/optitrack_calibration_*.csv` | ~21-23 rows | 4 CSV files |
| URDF variants | `examples/tiago/data/tiago*.urdf` | 3201 + 1999 + 1427 lines | 3 URDF copies (pre-dedup) |
| **Core: calibration_tools.py** | `src/figaroh/calibration/calibration_tools.py` | +269 lines | `xyzquat_to_SE3`, `update_forward_kinematics_2` |
| **Core: robot.py** | `src/figaroh/tools/robot.py` | +18 lines | Try-except guards for param arrays |

### B.2.2 How it works

**Two distinct problems are solved:**

#### Problem A: Free-flyer base pose estimation from markers
- Robot loaded with `isFext=True` → adds free-flyer joint at base
- Vicon/optitrack markers on base → compute base pose in world frame
- Known offset `Mmarker_base` transforms marker cluster → base link frame
- Derivatives (velocity, acceleration) computed from filtered position data
- Result: `q_base, dq_base, ddq_base` (6 DOF free-flyer + derivatives)

#### Problem B: Suspension spring-damper parameter identification
- 12 parameters: 3 translation stiffness + 3 translation damping + 3 rotation stiffness + 3 rotation damping
- Regressor matrix `R` (6×NbSample × 12): `force = k*displacement + c*velocity`
- Uses force plate data (`F_x/y/z`, `M_x/y/z`, `COP_x/y/z`)
- Cost function: `cost_function_fb()` — estimates base pose, concatenates with arm data, computes centroidal momentum via Pinocchio, builds R matrix, returns residual
- Solver: `scipy.optimize.least_squares` (Levenberg-Marquardt)

**Data format (Vicon CSV):**
```
time, base1_x/y/z, base2_x/y/z, base3_x/y/z, shoulder1-4_x/y/z, gripper1-3_x/y/z, F_x/y/z, M_x/y/z, COP_x/y/z
```
- 3 base markers + 4 shoulder markers + 3 gripper markers (30 position columns)
- Force plate: 3 force + 3 moment + 3 center-of-pressure (9 columns)

### B.2.3 Reconciliation with current architecture

#### Core library changes — detailed comparison

| Change in branch | Current state | Action |
|-----------------|---------------|--------|
| `xyzquat_to_SE3()` added to `calibration_tools.py` | **Already exists** (line 121) | ✅ No action needed |
| `update_forward_kinematics()` — added `verbose`, multi-marker loop, `non_geom` | **Already exists** (line 291) with these features | ✅ No action needed |
| `update_forward_kinematics_2()` — new function with `base_placement` + multi-marker | **NOT in current code**. Current `calc_updated_fkm()` (line 502) says "NbMarkers=1 (only supports single marker)" and warns for >1 | ⚠️ **Gap — decision needed** (see below) |
| `non_geom` config field | **Already in config.py** (lines 227, 255, 476, 489) | ✅ No action needed |
| `robot.py` try-except guards for `has_friction`/`has_actuator_inertia`/`has_joint_offset` | **Obsolete** — logic refactored into `regressor.py` with `RegressorConfig` dataclass (lines 13-15) | ❌ **Do not port** — would be a regression |
| `cartesian_to_SE3()` | **Already exists** (line 102) | ✅ No action needed |

#### The `update_forward_kinematics_2` gap

The current `calc_updated_fkm()` has a **known limitation**: it only supports `NbMarkers=1`. For `NbMarkers > 1`, it logs a warning and falls back to identity. The branch's `update_forward_kinematics_2()` handles multiple markers properly with:
- `base_placement` as first 6 parameters (inverse applied)
- Per-marker `eeMf` computation
- Output shape: `PEE = np.zeros((NbMarkers * calibration_index, NbSample))`

**However:** The suspension work doesn't actually use `update_forward_kinematics_2` for the suspension ID itself — it uses `processing_utils.py`'s own frame math (`create_rigidbody_frame`, `project_frame`, `estimate_base_pose_from_marker`). The `update_forward_kinematics_2` was for **multi-marker calibration**, which is a separate concern.

#### Example-level changes

| Aspect | Branch (old) | Current | Action |
|--------|-------------|---------|--------|
| Config format | Legacy flat YAML | Unified config | **Convert** — add suspension-specific fields |
| Import paths | Relative (`from processing_utils import ...`) | Package-relative | **Update** |
| `processing_utils.py` | 1068-line monolith | Not in current examples | **Port + refactor** — split into focused modules |
| Hardcoded paths | `/home/thanhndv212/Downloads/experiment_data/...` | — | **Make relative/configurable** |
| URDF copies | 3 URDF files in `data/` | Symlinked from `models/` | **Use symlinks** — don't port URDF copies |
| `isFext=True` robot loading | Custom `Robot()` constructor arg | Current `Robot` class — need to verify `isFext` support | ⚠️ **Verify** |
| Extract scripts | 3 scripts for rosbag → CSV | Not needed for example | **Optional** — port as utilities, not core example |
| `BaseIdentification` integration | None — standalone script | Could subclass `BaseIdentification` | **Decision needed** |

### B.2.4 Risks (suspension)

| Risk | Severity | Mitigation |
|------|----------|------------|
| `processing_utils.py` is a 1068-line monolith with hardcoded paths, mixing I/O, math, and cost function | **High** | Refactor into: (a) `vicon_utils.py` (data I/O + filtering), (b) `frame_utils.py` (SE3 math), (c) suspension cost function in the subclass |
| Free-flyer robot loading (`isFext=True`) may not work with current `Robot` class | **Medium** | Verify `tools/robot.py` supports free-flyer injection; may need `pin.Model` manipulation |
| Suspension ID doesn't use `BaseIdentification` pipeline at all — it's a completely custom script | **High** | Either: (a) port as standalone script (honest but doesn't showcase the framework), or (b) refactor to use `BaseIdentification` with custom `load_trajectory_data` + custom solve (showcases framework but significant rework) |
| Data files reference external paths (`/home/thanhndv212/Downloads/...`) | **Medium** | Port the CSV data into `examples/tiago/data/suspension/` and update paths |
| Force plate data (`F_x/y/z`, `M_x/y/z`) is not standard figaroh input | **Medium** | Document as TIAGo-specific; the suspension model is a research feature, not a standard pipeline |
| `update_forward_kinematics_2` is not in current code — multi-marker calibration is broken | **Low** (for suspension) | Not needed for suspension ID; relevant only if multi-marker calibration is desired |
| 3 URDF copies in branch data (pre-dedup) | **Low** | Use current symlinked URDF; don't port copies |
| Extract scripts depend on rosbag-specific formats | **Low** | Port as optional utilities, not as part of the example pipeline |

### B.2.5 Open questions (suspension)

1. **Standalone script or BaseIdentification subclass?** The suspension ID is fundamentally different from standard dynamic ID:
   - It identifies suspension parameters (spring/damper), not inertial parameters
   - It uses force plate data, not joint torques
   - It requires free-flyer base pose estimation from markers
   - The regressor is custom (`create_R_matrix`), not the standard `build_regressor_basic`

   Should we:
   - (a) Port as a standalone script that uses figaroh tools (Robot, SE3 math) but not the BaseIdentification pipeline?
   - (b) Create a `BaseSuspensionIdentification` or extend `BaseIdentification` with custom regressor/solve?
   - (c) Port as a research example with clear documentation that it's not a standard pipeline?

2. **How much of `processing_utils.py` to port?** The 1068-line file contains:
   - Vicon/optitrack CSV reading → needed
   - Filtering (median + Butterworth) → figaroh already has `_apply_filters` in `BaseIdentification`
   - SE3 frame math → some exists in figaroh, some is custom
   - `estimate_base_pose_from_marker` → custom, needed
   - `cost_function_fb` → custom, needed
   - `create_R_matrix` → custom, needed
   - Plotting functions → optional

   Should we port the whole file, or extract only the essential functions?

3. **Is `isFext` supported in the current `Robot` class?** Need to verify `tools/robot.py` can add a free-flyer joint. The branch uses `Robot(..., isFext=True)`.

4. **Should the extract scripts be ported?** They convert rosbag data to CSV. Useful for real-robot workflows but not needed to run the example with included data.

5. **Multi-marker calibration (`update_forward_kinematics_2`):** Should we port this to core? It's not needed for suspension ID but would fix the `NbMarkers > 1` gap in `calc_updated_fkm`. This is a separate feature.

6. **Data provenance:** The Vicon/optitrack data files — are these from real experiments? Should they be included in the examples repo, or are they too large / proprietary?

---

## Part B.3: Recommended Port Strategy

### Phase 1: Eye-hand calibration (low risk, no core changes)

1. Convert one config variant (hey5_center) to unified config format
2. Verify `data_loader.py` handles 6-DOF markers (position + orientation)
3. Port `write_to_xacro` or adapt to `update_model.py` pattern
4. Create `eye_hand_calibration.py` script — either standalone or as `TiagoCalibration` variant
5. Port eye-hand CSV data to `examples/tiago/data/eye_hand/`
6. Test end-to-end
7. If successful, port remaining config variants

### Phase 2: Mobile-base / suspension ID (higher risk, needs decisions)

1. **Decide architecture** (standalone vs BaseIdentification subclass — see question 1 above)
2. Verify `isFext` free-flyer support in current `Robot` class
3. Refactor `processing_utils.py` into focused modules
4. Port suspension data to `examples/tiago/data/suspension/`
5. Convert config to unified format
6. Create `suspension_identification.py` script
7. Test end-to-end
8. Optionally port extract scripts as utilities

### Phase 3 (optional): Multi-marker calibration core fix

1. Port `update_forward_kinematics_2` logic into `calc_updated_fkm` to support `NbMarkers > 1`
2. Add tests for multi-marker FK
3. This is independent of both features but fixes a known gap

---

## Part B.4: Architecture Mapping

### Current figaroh-examples TIAGo structure

```
figaroh-examples/examples/tiago/
├── calibration.py              # Geometric calibration (mocap, position-only)
├── identification.py           # Dynamic ID (8 joints, no base)
├── optimal_config.py           # D-optimal posture selection
├── optimal_trajectory.py       # IPOPT exciting trajectory
├── config/
│   ├── tiago_config.yaml       # Legacy flat config
│   ├── tiago_config_hey5.yaml  # Legacy, eye-hand fields present
│   └── tiago_unified_config.yaml  # Unified, has partial eye_hand variant
├── utils/
│   └── tiago_tools.py          # TiagoCalibration, TiagoIdentification, etc.
├── data/
│   ├── calibration/mocap/      # Mocap CSV data
│   └── identification/dynamic/ # Joint position/velocity/effort CSVs
└── urdf/                       # Symlinks to models/
```

### Proposed additions

```
figaroh-examples/examples/tiago/
├── eye_hand_calibration.py     # NEW — Feature E
├── suspension_identification.py # NEW — Feature D
├── config/
│   ├── tiago_eye_hand_config.yaml      # NEW — unified config for eye-hand
│   └── tiago_suspension_config.yaml    # NEW — unified config for suspension
├── utils/
│   ├── tiago_tools.py          # EXISTING — add eye-hand cost function variant
│   ├── vicon_utils.py          # NEW — Vicon/optitrack data I/O + filtering
│   └── suspension_utils.py     # NEW — frame math, cost_function_fb, create_R_matrix
├── data/
│   ├── eye_hand/               # NEW — eye-hand CSV data
│   └── suspension/             # NEW — Vicon/optitrack CSV data
└── urdf/                       # EXISTING symlinks
```

### Core library changes needed

| File | Change | Risk |
|------|--------|------|
| `figaroh/src/figaroh/calibration/calibration_tools.py` | None needed for eye-hand. **Optional**: port `update_forward_kinematics_2` for multi-marker support | Low (optional) |
| `figaroh/src/figaroh/tools/robot.py` | None — `isFext` support needs verification, not changes | Unknown |
| `figaroh/src/figaroh/calibration/data_loader.py` | Verify 6-DOF marker support for eye-hand | Low |
| `figaroh/src/figaroh/calibration/config.py` | May need eye-hand specific config fields (`camera_pose`, `tip_pose`) | Low |

---

## Part B.5: Decision Summary

| # | Decision | Options | Recommendation |
|---|----------|---------|----------------|
| 1 | Eye-hand: standalone or BaseCalibration subclass? | (a) Standalone, (b) Subclass | (b) Subclass — use `TiagoCalibration` with eye-hand config + custom `cost_function` override |
| 2 | Eye-hand: which gripper variants? | All 6, or just hey5_center | Start with hey5_center, add others if needed |
| 3 | Eye-hand: XACRO or update_model.py output? | XACRO, or URDF update | XACRO (matches PAL workflow) + optional update_model.py |
| 4 | Suspension: standalone or BaseIdentification subclass? | (a) Standalone, (b) Subclass, (c) Research example | (c) Research example with standalone script using figaroh tools |
| 5 | Suspension: port all of processing_utils.py? | All 1068 lines, or extract essentials | Extract essentials (~300 lines) into `vicon_utils.py` + `suspension_utils.py` |
| 6 | Port extract scripts? | Yes/no | No — not needed for example; document data format instead |
| 7 | Port `update_forward_kinematics_2` to core? | Yes/no/defer | Defer — separate feature, not needed for either D or E |
| 8 | Include Vicon/optitrack data in examples repo? | Yes/no | Yes — small files (~22 rows each), needed to run example |

