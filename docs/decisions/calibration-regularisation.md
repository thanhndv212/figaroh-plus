# Decision: Deprecate the calibration regularisation coefficient in favour of `map`

- Status: Accepted
- Date: 2026-10-06
- Issue: [#120](https://github.com/thanhndv212/figaroh-plus/issues/120) (C4)
- Supersedes / superseded by: None. Builds on
  [calibration estimation methods](calibration-estimation-methods.md) (#113).

## Context

`regularization_coefficient` (`calib_config["coeff_regularize"]`, default
0.01) is a single weight `c`. Robot cost functions in figaroh-examples
appended `sqrt(c) * theta` for every joint parameter, cut from the parameter
vector by position (UR10, TIAGo, TIAGo Pro; TALOS hard-coded `1e-3` and
ignored the config). Core itself never applied it.

One weight for every parameter is a Gaussian prior of the same standard
deviation `s = sigma / sqrt(c)` for all of them, where `sigma` is the
residual noise in the cost function's units. So:

- a metre and a radian get the same prior size (with `sigma` = 0.5 mm and
  `c` = 0.01, 5 mm for a translation *and* 5 mrad for an encoder offset that
  is typically 20-50 mrad);
- the prior changes with the residual noise and with any measurement
  weighting the robot applies, so the same `c` means different things on
  different robots.

`estimation.method: map` (#113) already applies priors of physical size per
parameter group (`translation`, `rotation`, `joint_offset`,
`prismatic_offset`), scaled by the residual noise, as rows appended by
`BaseCalibration._objective`.

## Decision

Deprecate the coefficient; do not add a second, unit-aware regulariser.

- Default 0 (was 0.01) in the unified parser and in config migration.
- A non-zero value is still passed through in `calib_config["coeff_regularize"]`
  for robot classes that read it, and raises a `DeprecationWarning` naming
  `estimation.method: map`.
- Priors on joint parameters are configured only with
  `estimation.method: map` / `map_cv` (or parameters are selected with
  `excitation` / `cv_subset`).
- figaroh-examples removes the regularisation rows from its cost functions
  and sets each robot's method deliberately (paired PR).

Unsupported, unchanged: cost functions with their own extra parameters
(TALOS table contact: plane and contact-frame corrections) still carry their
own regularisation; `map` cannot place priors on those parameters yet
(follow-up issue).

## Alternatives and consequences

- **Per-group scales on the coefficient.** Rows `sqrt(c) * theta / s_group`
  are unit-consistent only after multiplying by the residual noise, which
  gives exactly `map`'s rows `sigma / s_group * theta`. A second copy of the
  same estimator with a different knob.
- **Keep the coefficient, change only the default to 0.** Leaves a setting
  that cannot be chosen consistently across units.

Consequences:

- Robots that relied on the rows must choose a method. On the examples:

| Robot | Old setting | Without rows | Chosen |
|---|---|---|---|
| TIAGo mocap (`joint_offset`) | 0 since examples#73 | unchanged | `structural` (unchanged) |
| TIAGo Pro (`full_params`) | 0.01 | RMSE unchanged, max error 14.40 → 14.41 mm | `structural` |
| UR10 (`full_params`) | 0.001 | fit RMS unchanged; export deviation 4.42 → 4.66 mm | `structural` |
| TALOS upper body (`full_params`) | 1e-3 hard-coded | fit 0.326 → 0.294 mm, but corrections up to 1.3 rad / 258 mm | `map` |

TALOS upper body, 61 postures, 5-fold CV over the training postures (no
held-out session exists):

| Estimation | Parameters | Fit | CV | Largest correction (offset / rotation / translation) |
|---|---|---|---|---|
| `structural` + old 1e-3 rows | 38 | 0.326 mm | 0.473 mm | 27.9 mrad / 17.3 mrad / 14.5 mm |
| `structural`, no rows | 38 | 0.294 mm | 0.478 mm | 1312 mrad / 645 mrad / 258 mm |
| `map`, default priors | 63 | 0.388 mm | **0.468 mm** | 27.4 mrad / 6.2 mrad / 3.7 mm |
| `map_cv` | 63 | 0.336 mm | 0.491 mm | 30.0 mrad / 15.0 mrad / 6.4 mm |

`cv_subset` was not completed: one fit per candidate set size inside each
fold took over an hour on these 61 postures.

## Validation and implementation

Acceptance criterion of #120: "Changing the coefficient by 100× moves no
retained parameter by more than its standard error on the TIAGo truth
fixture." Measured on figaroh-examples' TIAGo truth fixture (examples#78):
5 seeds, σ = 0.5 and 2.0 mm, `joint_offset` and `full_params` truths.
Shift = largest |Δθ| / SE over the joint parameters the unregularised
`structural` fit keeps, SE from that fit.

| Change in strength (100× in `c`, 10× in prior size) | `joint_offset` fit | `full_params` fit |
|---|---|---|
| old coefficient, 1e-4 → 1e-2 | 2.84 SE | 4.66 SE |
| `map`, priors ×10 → ×1 (default) | **0.94 SE** | 2.27 SE |
| `map`, priors ×1 → ×0.1 | 10.1 SE | 2.74 SE |

Held-out error against the noise-free truth, mean of 5 seeds
(`full_params` fit, `joint_offset` truth, σ = 0.5 / 2.0 mm): no
regularisation 1.17 / 3.45 mm, old 1e-2 0.80 / 2.39 mm, `map` default
0.47 / 1.12 mm.

What this supports:

- The old coefficient fails the criterion on both levels.
- `map` meets it at the `joint_offset` level for priors at or above the
  default size: weakening the prior 10× moves no retained parameter by more
  than 0.94 SE.
- At `full_params` with 37 postures, no prior strength meets it: several
  retained directions are decided mostly by the prior, and moving it moves
  them. This is the intended behaviour of a prior on weakly excited
  directions, and `map` at default priors gives the best held-out error of
  every setting. The criterion is therefore accepted in this scope: on
  well-determined parameters, not on prior-dominated ones.
- Priors 10× too small bias well-determined offsets (up to 10 SE): the
  default sizes should not be shrunk without evidence.

Implementation: core `src/figaroh/calibration/config.py`
(`regularization_coefficient`), `src/figaroh/utils/config_migration.py`;
tests `tests/unit/test_calibration_regularization.py`. Examples: cost
functions of UR10, TIAGo, TIAGo Pro and TALOS; TALOS config
`estimation.method: map`; golden reference updated.

Unverified: robots other than TIAGo for the strength criterion; designed
posture plans; table contact (out of scope above).
