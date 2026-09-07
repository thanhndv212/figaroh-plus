# TIAGo Suspension and Empirical Backlash Examples

**Status:** Done

**Date:** 2026-08-31 (split from the combined
`tiago-suspension-backlash-and-modular-terms-plan.md`, proposed 2026-08-28)

**Companion document:** [`modular-linear-residual-terms-plan.md`](modular-linear-residual-terms-plan.md)
covers the still-proposed, not-yet-started generic linear-regressor-term /
residual-term architecture this work deliberately did **not** require —
including the still-unbuilt physical/stateful backlash model. That
document's "Conditional Core Promotion" gates are why nothing described
here has been (or should yet be) promoted into core FIGAROH.

## Implementation Summary

Both examples described below are implemented, tested, and shipped as
explicitly experimental research examples in `figaroh-examples`:

- `examples/tiago/suspension_identification.py` — fixed-transform
  generalized-base suspension identification.
- `examples/tiago/backlash_empirical_surface.py` — empirical
  backlash-compensation surface, with a `--joint all` sweep across every
  arm joint, automatic per-joint polynomial-degree backoff, structurally-
  zero design-matrix column removal, and a self-contained HTML report
  (statistics table, fit-quality chart, per-joint interactive 3D
  regression plots) alongside the JSON report.
- Supporting utilities: `utils/suspension_data.py`,
  `utils/suspension_model.py`, `utils/backlash_surface.py`,
  `utils/reporting.py`.
- Tests: `tests/test_tiago_suspension_backlash.py` (9 unit tests —
  synthetic coefficient recovery, rank-deficiency rejection, the
  trajectory contract, real-data loaders, the experimental-config opt-in
  contract, extrapolation rejection).
- Shipped via `figaroh-examples` PR
  [#9](https://github.com/thanhndv212/figaroh-examples/pull/9)
  (branch `feature/tiago-suspension-backlash-examples`).

**Deviations from the original plan, discovered during implementation:**

- The real-data backlash load feature must be Pinocchio's computed
  generalized gravity torque (reconstructed per sample from every logged
  arm/torso/head joint position), not a raw logged-effort proxy — effort
  is heavily quantized on the wrist joints (arm_5–arm_7) and produces
  rank-deficient fits there. This matches the historical prototype's own
  `tau_g` feature more precisely than the plan anticipated.
- Polynomial degree 5 (not 1) is the correct default for the empirical
  backlash surface, matching the historical prototype's own
  order-selection study (`surface_fitting_rho100_0202.csv`: R²=0.72 at
  degree 1 vs. R²=0.90 at degree 5 on `arm_6_joint`).
- Not every joint supports the same degree. A joint whose rotation axis
  doesn't couple to gravity (`arm_1_joint`'s near-vertical shoulder-yaw
  axis) has an identically-zero gravity-torque feature — every design-
  matrix term involving a positive power of it is an exact zero column,
  which is rank deficiency by construction, not a "hard to fit" case.
  Dropping those structurally-zero columns before solving (rather than
  reducing the whole polynomial degree) lets the genuinely well-posed
  position-only terms fit at the full requested degree. A second,
  complementary automatic-degree-backoff mechanism handles the different
  case of a joint whose load feature is merely near-collinear with
  position at high degree (not exactly zero) — `arm_2_joint`.

**Representative real-data results** (`--joint all`, degree ≤5,
`sinus_amp3_period10_2023-07-24-13-28-42` fixture):

| joint | degree used | R² |
|---|---|---|
| arm_1 | 5 (30 structurally-zero terms dropped) | 0.952 |
| arm_2 | 4 (backed off from 5) | 0.972 |
| arm_3 | 5 | 0.945 |
| arm_4 | 5 | 0.996 |
| arm_5 | 5 | 0.973 |
| arm_6 | 5 | 0.990 |
| arm_7 | 5 | 0.928 |

## What the Historical Branch Actually Contains

The local `tiago-suspension-calib` branch diverged from current `main` in 2023
and its relevant prototype is in:

- `examples/tiago/suspension_identification.py`
- `examples/tiago/processing_utils.py`
- `examples/tiago/extract_rosbag.py`
- `examples/tiago/extract_adream.py`
- `examples/tiago/extract_creps.py`

The branch includes a TIAGo free-flyer setup (`Robot(..., isFext=True)`), Vicon
marker data, force-plate wrench data, filtering/synchronization utilities, and a
custom suspension solve. It also includes unrelated historical examples and
large model/data additions, so it was ported selectively rather than merged.

The `tiago-suspension-calib` branch itself contains no backlash implementation.
However, a separate older reachable lineage, referenced locally as
`figaroh-plus`, contains a TIAGo backlash inspection and compensation
prototype. The recovered prototype is described below and was the
empirical baseline for the port.

## Recovered Backlash History

The all-ref history contains the following relevant work:

| Commit | Date | Evidence |
| --- | --- | --- |
| `9f17127` | 2024-02-19 | Adds `examples/tiago/backlash_inspection.py`, `tiago_utils/backlash/calib_inspect.py`, and `polynomial_fitting.py` |
| `9251556` | 2024-02-22 | Compares pose estimation with relative versus absolute encoders on Vicon data |
| `9159c5c` | 2024-04-11 | Updates backlash inspection with Vicon data |
| `911fcac` | 2024-04-11 | Adds per-joint inspection visualizations and surface-fitting results |
| `a6dfd81` | 2026-06-19 | Roadmap reserves dead-zone/backlash as an experimental friction-model direction |

The prototype persists in the reachable historical `figaroh-plus` ref, but is
not part of current `main`, release branches, or the current `figaroh-examples`
layout. It was source material, not a supported API.

### Historical Empirical Compensation Model

The historical workflow compares relative and absolute encoder measurements on
Vicon calibration trajectories. For an inspected joint $j$, it constructs the
target encoder difference:

$$
e_j=q_{\mathrm{absolute},j}-q_{\mathrm{relative},j}.
$$

It fits two polynomial surfaces in relative joint position and Pinocchio gravity
torque, then blends them using a fixed steep logistic of joint velocity:

$$
\hat e_j=P_j^+(q_{\mathrm{relative},j},\tau_{g,j})\sigma(\rho\dot q_j)
+P_j^-(q_{\mathrm{relative},j},\tau_{g,j})[1-\sigma(\rho\dot q_j)].
$$

Here $\rho=100$ in the prototype. `SurfaceFitting` obtains the polynomial design
matrix from `create_poly`, uses `encoder_diff` as the target, and evaluates both
directional surfaces. The branch solves the coefficients with
`scipy.optimize.least_squares`, although the problem is linear in the polynomial
coefficients when $\rho$ and the features are fixed — the port uses an
equivalent weighted linear least-squares solve instead.

This is a direction- and load-conditioned **kinematic correction surface**. It
is not a physical backlash state model: it has no persistent contact state, no
explicit gap width, and no prediction of dynamic transmission torque. Its direct
evidence is calibration-pose improvement against Vicon/absolute-encoder data.

Ported as `EmpiricalBacklashSurface` in the TIAGo example namespace,
fit with weighted linear least squares using a combined design matrix
$[\sigma A^+, (1-\sigma)A^-]$, with polynomial degree and
velocity-transition steepness as explicit configuration, rejecting
evaluation outside the fitted position/torque domain by default.

A second, dynamic/stateful backlash model (predicting dynamic transmission
torque from a hysteretic deadband state, for simulation/torque
prediction rather than kinematic correction) remains future work — see
[`modular-linear-residual-terms-plan.md`](modular-linear-residual-terms-plan.md).

## Recovered Suspension Approach

### Inputs

The prototype combines:

- rigidly mounted base markers from optical motion capture;
- arm encoder trajectories;
- force-plate force, moment, and center-of-pressure signals;
- a TIAGo URDF with a Pinocchio free-flyer joint; and
- a calibrated static transform from marker frame to base frame.

It filters/resamples signals, estimates the base pose from marker motion, derives
base velocity and acceleration, concatenates base and arm state, and evaluates
floating-base dynamics with Pinocchio.

### Model

The model treats the mobile base as one generalized six-degree-of-freedom
spring-damper. Its parameter vector is:

$$
\theta_s = [k_x,c_x,k_y,c_y,k_z,c_z,\kappa_x,\gamma_x,\kappa_y,\gamma_y,\kappa_z,\gamma_z]^T.
$$

It uses translational displacement and velocity for force rows, and RPY angle
and angular-velocity signals for moment rows. In its intended linear form:

$$
w_s = R_s(x_b,\dot{x}_b,r_b,\omega_b)\theta_s.
$$

The historical branch built the $6N \times 12$ matrix $R_s$, obtained a
floating-base wrench from Pinocchio centroidal momentum variation (with an
inverse-dynamics alternative present but inactive), and called
`scipy.optimize.least_squares(..., method="lm")`.

### Corrections Made When Porting

The prototype's suspension coefficients are linear once base motion and the
marker-to-base transform are fixed. Levenberg-Marquardt is appropriate only for
the joint nonlinear problem that also estimates the marker transform. The port
made the two cases explicit:

| Problem | Unknowns | Solver used |
| --- | --- | --- |
| Fixed marker-to-base transform | $\theta_s$ | Weighted linear least squares with rank diagnostics (implemented) |
| Estimated marker-to-base transform | transform plus $\theta_s$ | Bounded nonlinear least squares, variable projection where possible (not implemented) |
| Per-wheel suspension model | wheel parameters plus geometry | Nonlinear least squares after single-base model validation (not implemented) |

The shipped example is the fixed-transform generalized-base model. It has
the smallest observability burden and offers a discriminating validation
target. It has not been promoted to a core FIGAROH API — see the
companion document's "Conditional Core Promotion" gates.

## Immediate Example Layout

The example layout as shipped:

```text
figaroh-examples/examples/tiago/
   suspension_identification.py       # fixed-transform generalized-base model
   backlash_empirical_surface.py      # relative-to-absolute encoder correction,
                                       # --joint all sweep + HTML report
   utils/
      suspension_data.py               # Vicon/force-plate parsing and filtering
      suspension_model.py              # pose estimation and R_s construction
      backlash_surface.py              # feature matrix, fitting, range checks
      reporting.py                     # shared plotting + JSON/HTML report generation
   config/
      tiago_suspension_config.yaml
      tiago_backlash_surface_config.yaml
   data/
      suspension/                      # small provenance-cleared fixture
      backlash/                        # small provenance-cleared fixture
```

The example owns all TIAGo-specific marker names, transform conventions,
force-plate frames, encoder naming, datasets, and plotting. ROS bag extraction
remains outside the runnable example — its input contract and optional
dependencies were never stabilized.

Neither script subclasses or modifies `BaseIdentification`; both stay
standalone. This intentionally does not require the `LinearRegressorTerm`/
`ResidualTerm`/`WeightPolicy` architecture in
[`modular-linear-residual-terms-plan.md`](modular-linear-residual-terms-plan.md) —
that architecture only becomes worthwhile once a primitive here has a
second concrete consumer (see that document's "Conditional Core
Promotion" gates, none of which are yet met).

## Migration Plan (Completed)

### Phase 1: Freeze the Research Contract — Done

Defined TIAGo-specific suspension and empirical-backlash contracts: SI units,
timestamps, marker names and frame definitions, base-marker transform
convention, force-plate wrench frame, center-of-pressure convention, relative
and absolute encoder naming, arm joint order, sampling/resampling policy, and
train/validation trajectory split.

Extracted a minimal branch fixture. Did not port absolute paths, notebooks,
duplicate URDFs, or the 1068-line utility module.

**Gate (met):** example parsers load fixtures and validate shapes, units,
timestamps, frame metadata, joint ordering, and fitted-domain metadata
without invoking an optimizer.

### Phase 2: Pure Suspension Mathematics — Done

Implemented and tested TIAGo example utilities:

- rigid-body pose estimation from a marker cluster;
- base-to-marker transform application;
- derivative/filter pipeline with explicit edge trimming;
- generalized-base spring-damper regressor $R_s$;
- wrench flattening/layout; and
- weighted linear solve with rank/condition diagnostics.

**Gate (met):** synthetic trajectories recover known suspension coefficients
under noise and reject rank-deficient excitation
(`test_suspension_regressor_recovers_known_coefficients`,
`test_suspension_fit_rejects_rank_deficient_motion`).

### Phase 3: TIAGo Suspension Example — Done

Created a standalone `suspension_identification.py` example using the fixed
marker-to-base transform and weighted linear solve. Reuses public
FIGAROH robot/dynamics utilities; does not subclass or modify
`BaseIdentification`. Produces a result artifact (JSON + PNG) with
rank/condition-number diagnostics.

**Gate (met):** fixed-transform TIAGo fixture has a reproducible report with
parameter units, condition number, and residuals by wrench component.
(Held-out train/validation split is not yet implemented — see "Known Gaps"
below.)

### Phase 4: TIAGo Empirical Backlash Example — Done

Created `backlash_empirical_surface.py` as a calibration-only example.
Reproduces the historical relative-versus-absolute encoder target,
directional polynomial blending, and pose comparison with a clean typed
artifact. Replaced the historical nonlinear coefficient fit with its
equivalent weighted linear solve, using historical results as the target
model form and polynomial basis.

**Gate (met):** real encoder data fits well across 6 of 7 arm joints
(R² 0.93–0.99 at degree ≤5, with automatic degree backoff/column-drop
handling the two axis/collinearity edge cases), reports fitted
position/torque ranges, and rejects extrapolation by default. (Held-out
train/validation split and an explicit no-compensation-baseline
comparison are not yet implemented — see "Known Gaps" below.)

## Known Gaps

Two items from the original validation criteria are not yet implemented,
despite the phases above being otherwise complete:

- **No held-out train/validation split.** Both examples currently report
  RMSE/R² on the same data they fit — not a genuine generalization test.
  The original plan's validation criteria explicitly call for separate
  train/validation RMSE and rejecting a model solely because in-sample
  residual decreases.
- **No explicit no-compensation baseline comparison** for the backlash
  example (Phase 4's gate asks for held-out improvement against the
  uncorrected baseline, not just a good in-sample fit).

## Validation and Acceptance Criteria

| Area | Minimum evidence | Status |
| --- | --- | --- |
| Frame correctness | Static and moving marker-transform unit tests | Covered by `SuspensionTrajectory` contract tests |
| Signal processing | Known sine/noise fixture, documented trim and phase behavior | Not separately unit-tested |
| Suspension | Synthetic recovery and withheld force/moment prediction | Synthetic recovery done; held-out prediction is a known gap |
| Empirical backlash | Historical-surface parity, held-out encoder/Vicon improvement, and extrapolation protection | Parity and extrapolation protection done; held-out improvement is a known gap |
| Integration | TIAGo fixture run from `figaroh-examples` with no absolute paths | Done |
| Reporting | Parameter units, bounds, terms, weights, solver, rank, and held-out metrics | Done except held-out metrics (rank, condition number, R², adjusted R², residual stats all reported) |

## Risks and Mitigations

| Risk | Mitigation | Status |
| --- | --- | --- |
| Confounding among offsets, friction, backlash, and compliance | Stage estimation and use targeted excitation | Not yet relevant — no joint identification combining these exists yet |
| RPY singularity and derivative noise | Store poses as SE(3)/quaternions; use tangent-space velocity and filtered derivatives | As implemented in `suspension_data.py` |
| Unknown marker-to-base transform | Calibrate separately first; use variable projection only after fixed-transform validation | Fixed-transform case shipped; variable-transform case not implemented |
| Force-plate frame/sign mismatch | Require explicit frame metadata and static-load sanity tests | Frame metadata required by the trajectory contract |
| Rank-deficient suspension experiment | Compute rank/condition before solve; reject insufficient excitation | Implemented (`fit_generalized_base_suspension` raises on rank deficiency) |

## Recommendation (Realized)

Started with the fixed-marker-transform generalized-base suspension model as a
TIAGo research example, followed by the recovered empirical backlash surface as
a separate TIAGo calibration example — as recommended. Core primitives were not
extracted, since no second consumer has appeared. Backlash has not advanced
from the empirical surface to a separately validated stateful model; that
remains proposed future work in
[`modular-linear-residual-terms-plan.md`](modular-linear-residual-terms-plan.md).
