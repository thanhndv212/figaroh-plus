# Calibration estimation methods

A geometric calibration estimates joint errors together with the unknown
base frame (where the measurement system sees the robot) and tool point
(where the marker sits). Two questions decide the result as much as the data
does:

1. **Which parameters to estimate.** Some are not identifiable at all from
   the measurements (they move nothing, or exactly what another parameter or
   a frame moves). Others are identifiable in principle but barely excited
   by the recorded postures: estimating them freely fits noise and
   extrapolates badly.
2. **How to estimate them.** Freely, or with prior knowledge of how large
   they are expected to be.

FIGAROH offers several methods. None is right for every robot and dataset;
this page lists what each one does, what it needs from you, and when it
tends to work. The choice is yours, in the configuration.

```yaml
tasks:
  calibration:
    parameters:
      calibration_level: full_params      # or joint_offset
      regularization_coefficient: 0.0     # keep 0 with map / map_cv
      estimation:
        method: map_cv                    # structural | excitation | map | map_cv | cv_subset
        priors:                           # expected error sizes (1 sigma)
          translation: 1.0e-3             # m
          rotation: 2.0e-3                # rad
          joint_offset: 2.0e-2            # rad, revolute joint-angle offset
          prismatic_offset: 2.0e-3        # m
        noise_std: null                   # residual units; null = estimated
        excitation_k: 1.0                 # excitation only
        cv_folds: 5                       # map_cv, cv_subset
        cv_seed: 0
        cv_multipliers: [0.1, 0.3, 1.0, 3.0, 10.0]   # map_cv
        cv_per_group: false               # map_cv: one factor per group
        cv_sizes: null                    # cv_subset: sizes to try (null = all)
        cv_rule: min                      # cv_subset: min | one_se
```

The base frame and tool point are always estimated freely, with no prior.
Parameter groups: `joint_offset` is the rotation about a revolute joint's
own axis (`d_phiz` for a z-axis joint at `full_params`, `offsetRZ` at
`joint_offset`); `prismatic_offset` the translation along a prismatic
joint's axis; `rotation` and `translation` every other placement error.

## The methods

### `structural` (default)

Selects base parameters by a QR decomposition on random configurations, then
drops those the estimated frames absorb on the measured postures
(figaroh-plus#102). Selected parameters are estimated freely.

- **Needs:** nothing.
- **Pros:** unchanged behaviour; results comparable with earlier versions.
- **Cons:** the structural step does not see the measured marker point, so
  it can drop parameters the data identifies; ties between equal columns are
  broken by floating-point details, so the selected set can differ between
  platforms (figaroh-plus#113); weakly excited parameters are estimated
  freely.
- **Use when:** you need results comparable with an earlier run.

### `excitation`

Starts from every joint parameter of the calibration level and drops exact
dependencies on the measured postures. Then, one at a time, removes the
parameter whose predicted standard error is largest relative to its
expected size (`priors`), while that ratio exceeds `excitation_k`. Kept
parameters are estimated freely.

- **Needs:** expected sizes (defaults above), `excitation_k`, noise (given
  or estimated).
- **Pros:** a named, reduced parameter set that is easy to read and export;
  kept parameters are not biased toward nominal; deterministic.
- **Cons:** still a scale assumption (the expected sizes and `k`); a hard
  cut, so a parameter close to the limit can flip with small data changes;
  dropped parameters are set exactly to nominal even if the data says
  something about them.
- **Use when:** you know roughly how large the errors are and want a short,
  interpretable list of corrections.

### `map`

Estimates every joint parameter with a zero-mean Gaussian prior of the
expected sizes. Directions the data excites well are set by the data;
weakly excited ones stay near nominal. Nothing is dropped.

- **Needs:** prior sizes per group, noise (given or estimated).
- **Pros:** no hard cut, so results change smoothly with the data and do not
  depend on the platform; uses all the information; every parameter gets a
  posterior standard deviation, and comparing it with the prior shows
  whether the data or the prior set it.
- **Cons:** results depend on the prior sizes. Too small: real errors are
  shrunk toward nominal (bias). Too large: behaves like a free fit and
  overfits. The priors are assumptions you must state.
- **Use when:** you know the robot's tolerances (vendor data, previous
  calibrations of the same model).

### `map_cv`

`map` with the prior sizes multiplied by a factor chosen by k-fold
cross-validation over the training postures: each fold is fitted on the
others and judged on the postures it left out. With `cv_per_group`, one
factor per group is chosen by coordinate search.

- **Needs:** nothing robot-specific; the default sizes only set the shape.
- **Pros:** adapts to the robot and the data; cross-validation also pushes
  the priors down when a parameter group only fits noise or unmodelled
  effects; uses the training data only.
- **Cons:** slower (one fit per fold and candidate factor); with few
  postures the folds are small and the choice is noisy; one more layer to
  explain.
- **Use when:** you have no reliable information on the robot's tolerances.

### `cv_subset`

Orders the joint parameters by the `excitation` removal order (least
excited first) and chooses how many to keep by k-fold cross-validation. With
`cv_rule: one_se`, it takes the smallest set within one standard error of
the best.

- **Needs:** nothing robot-specific.
- **Pros:** a named, reduced set like `excitation`, sized by prediction on
  left-out postures rather than by a threshold.
- **Cons:** slowest (one fit per fold and size); the chosen size can change
  with the fold split when there are few postures; a hard cut.
- **Use when:** you want a named parameter set and have no basis for a
  threshold.

## Choosing

| Situation | Methods to consider |
|---|---|
| Reproduce or compare with earlier FIGAROH results | `structural` |
| Tolerances known, want a short list of corrections | `excitation` |
| Tolerances known, want the best prediction | `map` |
| No information on the robot | `map_cv`, `cv_subset` |
| Few postures for many parameters (TIAGo mocap: 111 measurements for 31 free parameters already overfits a known truth) | `map`, `map_cv`, `cv_subset`; avoid free `full_params` |
| Postures designed for this model (`figaroh.optimal`) | any; differences shrink |

Whatever you choose, judge the result on postures that were not used for
fitting, ideally from another session. Cross-validation inside
`map_cv`/`cv_subset` uses training postures only and is not a substitute.
figaroh-examples has a fixture with a known truth
(`docs/development/tiago-calibration-synthetic-truth.md`) and a held-out
protocol (`docs/development/tiago-mocap-heldout-protocol.md`) showing how.

### What a known truth shows

figaroh-examples' TIAGo fixture with a known truth (real 37 training
postures, 184 held-out postures, 5 seeds; `calibration_truth.py`) gives the
held-out prediction error against the truth, in mm, at `full_params` unless
noted:

| Truth class / noise | `joint_offset` level | `structural` | `excitation` | `map`, correct priors | `map`, priors ×0.1 | `map`, priors ×10 | `map_cv` | `cv_subset` |
|---|---|---|---|---|---|---|---|---|
| `full_params` / 0.5 mm | 1.16 | 1.09 | 0.65 | **0.49** | 1.21 | 0.78 | 0.56 | 0.69 |
| `full_params` / 2.0 mm | 1.50 | 3.23 | 1.50 | **1.21** | 3.05 | 2.54 | 1.54 | 1.59 |
| `joint_offset` / 0.5 mm | **0.25** | 1.17 | 0.64 | 0.47 | 0.71 | 0.79 | 0.42 | 0.33 |
| `joint_offset` / 2.0 mm | **0.98** | 3.45 | **0.98** | 1.12 | 3.44 | 2.55 | 1.32 | 1.27 |

- Every method other than `structural` improves on it, by up to 3×.
- `map` is best when its priors are right; priors wrong by 10× in either
  direction lose most of that.
- `map_cv` and `cv_subset` need no robot sizes and stay close to `map` with
  correct priors (25–45 s instead of 1–3 s on this problem).
- A simpler model wins when the truth is simple; the data-driven methods
  come close to it without being told.

This is one robot, one posture plan and no model mismatch (no backlash or
deflection); repeat the comparison on your own setup before relying on it.

## Noise

`noise_std` is in the units of your cost function's measurement residuals:
metres for positions, unless the robot's `cost_function` weights them. When
it is `null`, FIGAROH fits the identifiable parameters freely and uses
√(RSS / (m − n)). That estimate includes unmodelled effects (backlash,
deflection), which is usually what you want: it keeps the data from looking
more precise than it is.

## Outputs

- `calib_config["estimation_report"]` (also `calibrator.estimation_report`):
  the method, candidates, exactly dependent parameters, the noise and its
  source, the priors, and per method the removal order with ratios, the
  cross-validation curve and the chosen factor or size.
- `calibrator.std_dev`: for `map`/`map_cv`, posterior standard deviations;
  a value close to the prior means the data did not inform that parameter.
- `redistribute_parameters()` and the PAL export: for the non-structural
  methods, the fitted joint parameters directly (candidates left out of the
  fit are reported at 0).

## Limits

- Not supported with `include_non_geometric` (elastic parameters); use
  `structural`.
- The robot's `cost_function` must keep the parameter layout: base frame
  first, joint parameters, tool point last (as the TIAGo, UR10 and TALOS
  examples do). Classes with extra parameters (e.g. table-contact planes)
  are not supported.
- Keep `regularization_coefficient: 0` with `map`/`map_cv`; otherwise both
  regularisations apply.
