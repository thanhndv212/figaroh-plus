# Choosing what to estimate in a calibration

A calibration's result depends on **which parameters you estimate and how**
as much as on the measurements. FIGAROH does not make that choice for you:
it provides several methods, diagnostics to compare them, and this guide.
The methods themselves (what each does, what it needs, pros and cons) are
described in the reference page
[Calibration estimation methods](../concepts/calibration_estimation.md).

This guide is the workflow: start from a baseline, decide what you know,
configure, read the diagnostics, and check the choice on data the fit did not
see.

## 1. Know why the choice matters

With an estimated base frame and tool point, some joint errors are not
identifiable at all (the frames absorb them), and others are identifiable
but barely excited by your postures. Estimating the latter freely fits noise:
the training residual drops, prediction on new postures gets worse.

On the TIAGo mocap reference (37 training postures, position only), a free
`full_params` fit with 31 parameters overfits even a synthetic robot whose
true errors are exactly of that form; at realistic noise it predicts new
postures two to three times worse than the methods that account for
excitation. See the table in the
[reference page](../concepts/calibration_estimation.md#what-a-known-truth-shows).

## 2. Run the baseline first

Keep the default (`structural`) and fit once. It is the behaviour of earlier
FIGAROH versions and the reference every other choice is compared with.

```python
calibrator = TiagoCalibration(robot, "config/tiago_unified_config.yaml")
calibrator.initialize()
calibrator.solve(plotting=False)
print(len(calibrator.calib_config["param_name"]), "parameters")
print(calibrator.calib_config.get("absorbed_param_name"))  # dropped: frames absorb them
```

Note the number of parameters against the number of measurements
(`NbSample` × measured components). Few measurements per parameter is the
first sign that the method matters.

## 3. Decide what you know

| You know… | Consider | Because |
|---|---|---|
| Nothing beyond the URDF | `map_cv`, `cv_subset` | They choose the prior scale or the set size by cross-validation on your postures |
| Typical error sizes (vendor tolerances, earlier calibrations of the same model) | `map`, `excitation` | They use those sizes directly; `map` for prediction, `excitation` for a short list |
| You need a short, named list of corrections (for review, a URDF diff, a runtime file) | `excitation`, `cv_subset` | They drop parameters instead of shrinking them |
| You need results comparable with an earlier run | `structural` | Unchanged behaviour |

None of these is right in general. If unsure, run several (step 6).

## 4. Configure

In the calibration task of the unified config:

```yaml
parameters:
  calibration_level: full_params
  regularization_coefficient: 0.0     # keep 0 with map / map_cv
  estimation:
    method: map_cv
```

Optional keys and their defaults are listed in the
[reference page](../concepts/calibration_estimation.md). The ones you are most
likely to set:

- `priors` — expected error sizes per group (`translation` m, `rotation` rad,
  `joint_offset` rad, `prismatic_offset` m). Used by `map` and `excitation`;
  `map_cv` only uses their ratios.
- `noise_std` — measurement noise in the units of your cost function's
  residuals. Leave `null` to estimate it from the data (recommended unless you
  have a reason to trust a sensor specification more than your residuals).
- `cv_folds`, `cv_seed` — cross-validation over your training postures.
- `cv_rule: one_se` — with `cv_subset`, prefer the smallest set within one
  standard error of the best.

The same settings can be set in Python before `initialize()`:

```python
calibrator.calib_config["estimation"] = {"method": "excitation", "excitation_k": 1.0}
```

## 5. Read the diagnostics

Selection happens in `initialize()`; its record is
`calibrator.estimation_report` (also `calib_config["estimation_report"]`):

```python
r = calibrator.estimation_report
print(r["method"], r["noise_std"], r["noise_source"])
print("not identifiable here:", r["dependent"])
for name, ratio in r.get("removal_order", []):   # excitation, cv_subset
    print(f"removed {name}: predicted SE {ratio:.1f} x expected size")
for point in r.get("cv_curve", []):              # map_cv, cv_subset
    print(point)
print(r.get("prior_scale"), r.get("n_joint_chosen"))
```

After `solve()`, for `map` and `map_cv`, compare each parameter's posterior
standard deviation with its prior: close to the prior means the data did not
inform it, and its value is essentially the nominal one.

```python
from figaroh.calibration import estimation

names = calibrator.calib_config["param_name"]
prior = estimation.prior_std(names, calibrator.model, estimation.settings(calibrator.calib_config)["priors"])
for n, s, p in zip(names, calibrator.std_dev, prior):
    if p != float("inf"):
        print(f"{n:28s} posterior/prior {s / p:.2f}")
```

Warning signs:

- the cross-validation curve is flat, or its minimum is at the edge of
  `cv_multipliers` / `cv_sizes` (extend the grid);
- `cv_subset` chooses very different sizes for different `cv_seed` values
  (too few postures for a stable choice);
- the estimated noise is much larger than the sensor's (unmodelled effects
  such as backlash or deflection dominate; no method fixes that).

## 6. Compare candidates on data the fit did not see

Cross-validation inside `map_cv` and `cv_subset` uses training postures only.
Judge the final choice on postures from another session (`validation_data_file`)
or a held-out part of the data, with the same frames and processing.

```python
results = {}
for method in ("structural", "excitation", "map_cv", "cv_subset"):
    c = TiagoCalibration(robot, config_path)
    c.calib_config["estimation"] = {"method": method}
    c.initialize()
    c.solve(plotting=False)
    results[method] = c.evaluation_metrics  # includes validation metrics when configured
```

Decide which comparison counts *before* looking at it, and look at the
held-out set once. Repeatedly tuning the method on the same held-out set
turns it into training data. figaroh-examples'
[TIAGo held-out protocol](https://github.com/thanhndv212/figaroh-examples/blob/main/docs/development/tiago-mocap-heldout-protocol.md)
shows one way to freeze sets and rules.

## 7. Check against a known truth when you can

A synthetic truth at **your** postures answers questions real data cannot: do
these postures support this many parameters, and does the method recover
what it should? figaroh-examples'
[TIAGo truth fixture](https://github.com/thanhndv212/figaroh-examples/blob/main/docs/development/tiago-calibration-synthetic-truth.md)
(`examples/tiago/calibration_truth.py`) draws errors of a chosen size,
simulates your postures with noise, fits, and compares with the truth. Adapt
it to your robot's postures and expected error sizes. It has no model
mismatch, so it complements real held-out data rather than replacing it.

## 8. Export and report

- Write the same joint corrections to the URDF and to PAL's
  `geometric_calibration`:

  ```python
  from figaroh.tools.urdf_exporter import export_urdf

  export_urdf("urdf/robot.urdf", calibrator.joint_corrections(), output_path="urdf/robot_calibrated.urdf")
  ```

  For `structural`, `joint_corrections()` lifts the fitted base parameters
  onto every joint with your expected error sizes, so a joint whose
  correction was represented by another parameter in the fit gets its share
  (`joint_corrections(lift=False)` returns the representatives only). For the
  other methods it returns the fitted joint parameters; those left out stay
  at nominal. Reload the written URDF and compare its forward kinematics with
  the fit before deploying it.
- Frames (base, tool point) are measurement-setup quantities and are not
  written into the robot URDF.
- Record in your report: the method and every non-default setting, the
  estimated or given noise, the priors, the selection record (removal order or
  cross-validation curve and choice), the parameter count, and the held-out
  result with its evidence type (training only, temporal block, other session,
  synthetic truth).

## See also

- [Calibration estimation methods](../concepts/calibration_estimation.md) — reference
- [Calibration walkthrough](calibration_walkthrough.md)
- [Plan, fit and validate](../example_workflow.md) — the general decision process
- [Optimal experiment design](optimal_design.md) — postures that excite more parameters
- [Reporting & verification](../reporting_and_verification.md)
