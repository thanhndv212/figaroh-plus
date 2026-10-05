# Decision: Selectable calibration estimation methods

- Status: Accepted
- Date: 2026-10-05
- Issue: [#113](https://github.com/thanhndv212/figaroh-plus/issues/113)
- Supersedes / superseded by: None

## Context

Geometric calibration chose its parameters one way: structural base
parameters from a pivoted QR on random configurations, then (since #102) the
data-level removal of parameters the estimated frames absorb. Selected
parameters were estimated freely. Investigating TIAGo results that differed
between macOS and Linux (figaroh-examples#74) showed:

- the structural step breaks ties between equal columns by floating-point
  details, so the selected set depends on the platform;
- it evaluates identifiability at the tool frame origin, not at the measured
  marker point, and drops directions the data identifies (rank 31/30 against
  33 on the TIAGo reference);
- keeping every identifiable direction is not better: with 37 postures the
  extra directions are barely excited and fitting them freely degrades
  held-out prediction (validation 3.07 → 4.17 mm).

So the question is not only identifiability but excitation, and the right
answer depends on the robot, the posture plan, the noise and what the user
knows about expected error sizes. Prior-based estimation is best when the
priors are right and poor when they are wrong by 10×; users often do not know
their robot's tolerances.

## Decision

FIGAROH provides several methods and lets the user choose per robot and
dataset; it does not pick one. `parameters.estimation.method` in the unified
config (`calib_config["estimation"]`) selects:

- `structural` — default, unchanged behaviour;
- `excitation` — data-level selection over every joint parameter with an
  explicit excitation threshold relative to expected sizes;
- `map` — Gaussian priors of user-given expected sizes on every joint
  parameter;
- `map_cv` — `map` with the prior scale chosen by k-fold cross-validation on
  the training postures;
- `cv_subset` — nested excitation-ordered sets with the size chosen by k-fold
  cross-validation.

Contract:

- Frames are always estimated freely.
- The parameter layout (base frame, joint parameters, tool point) is kept for
  robot cost functions.
- Priors enter as residual rows appended by `BaseCalibration._objective`.
- Selection runs once in `create_param_list`, and its record is
  `calib_config["estimation_report"]`.

Unsupported:

- `include_non_geometric`;
- cost functions that add their own parameters (e.g. table contact).

The default stays `structural` so existing results and the figaroh-examples
golden outputs do not change. Changing the default is a separate decision.

## Alternatives and consequences

- **One fixed method (replace `structural`).** Simpler, but every option has
  a regime where it is wrong (wrong priors, too few postures for CV, a simple
  truth), and it would change every existing result.
- **Only fix the platform tie-break in `structural`.** Smallest change; keeps
  the overfitting and the wrong measurement model of the structural step.
- **Empirical-Bayes prior scales (marginal likelihood).** Possible future
  addition to `map_cv`; cross-validation was chosen first because it is more
  robust to unmodelled effects (backlash, deflection) that violate the noise
  model.

Costs:

- Five methods to document and test.
- `map_cv` and `cv_subset` take one fit per fold and candidate (25–45 s on
  TIAGo, against 1–3 s).
- Users must understand the choice, which the guide and diagnostics address.

## Validation and implementation

- **Implementation:** figaroh-plus #117: `src/figaroh/calibration/estimation.py`
  and `BaseCalibration` integration.
- **Unit tests:** `tests/unit/test_calibration_estimation.py`.
- **Evidence:** figaroh-examples TIAGo truth fixture, 5 seeds, held-out error
  against truth (reference page, "What a known truth shows").
  - Every non-structural method improves on `structural`.
  - `map` with correct priors is best; with priors off by 10× in either
    direction it loses most of that gain.
  - `map_cv` and `cv_subset` stay close to correct-prior `map` without robot
    sizes.
- **Docs:** reference `docs/source/concepts/calibration_estimation.md`; guide
  `docs/source/tutorials/calibration_estimation_guide.md`.
- **Unverified:**
  - robots other than TIAGo;
  - designed posture plans;
  - model mismatch (backlash, deflection);
  - whether a non-structural default would serve most users.
