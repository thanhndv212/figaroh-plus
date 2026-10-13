# Calibrate a robot

Correct joint offsets and frame transforms from measured end-effector poses. Five steps, each with the choices you make and how you know it worked.

Use the toggle below to read the steps for the bundled UR10 example or for your own robot. Each choice card lists the options FIGAROH supports, and the badge marks the usual starting point.

## 1. Plan

Decide what you want to correct and check that you have the data for it.

Example robot

The UR10 example ships everything: a URDF in `urdf/`, measured poses in `data/calibration.csv`, and a config that extends the manipulator template. Clone the examples repository and skip to [Configure](#configure) if you only want to see a run.

```
git clone https://github.com/thanhndv212/figaroh-examples
cd figaroh-examples/examples/ur10
```

Your own robot

You need:

- a URDF whose joint names match your logs;
- a CSV with one row per measured pose: the marker position (and orientation, if your sensor measures it) followed by one column per active joint;
- a second CSV, recorded separately, to validate on.

If your robot has no folder under `figaroh-examples/examples/` yet, do [Add your robot](https://thanhndv212.github.io/figaroh-plus/workflows/new-robot/index.md) first.

**What to estimate** tasks.calibration.parameters.calibration_level

Estimates joint offsets and the full geometric parameters of each link, plus base and tool frames. Use it when link lengths or orientations may differ from the CAD values. It needs more poses than `joint_offset`.

Estimates only the encoder zero of each joint, plus base and tool frames. Use it when the geometry is trusted and only the zeros drift, for example after re-mounting motors. It works with fewer poses.

**How to weigh prior knowledge** tasks.calibration.parameters.estimation.method

Plain least squares on the identifiable parameters. It reproduces earlier FIGAROH results and needs no extra information. Start here.

Maximum a posteriori: pulls each parameter towards its nominal value, weighted by `priors`. Use it when you know manufacturing tolerances. It gives the best prediction and reports an uncertainty for every parameter.

Like `map`, but picks the prior strength by cross-validation (`cv_folds`, `cv_multipliers`, `cv_rule: min | one_se`). Use it when you have no tolerance information.

Keeps only the parameters your poses excite, judged against `priors` and `excitation_k`. Use it when you know tolerances and want a short, readable parameter list.

Fits a named subset of parameters sized by cross-validation (`cv_sizes`). Use it when you want to compare a few candidate parameter sets.

Default priors are 1 mm for translations, 2 mrad for rotations, 20 mrad for joint offsets and 2 mm for prismatic offsets. Priors that are wrong by a factor of 10 lose most of the gain, so set them from your hardware. `noise_std: null` estimates the sensor noise. See [Calibration estimation methods](https://thanhndv212.github.io/figaroh-plus/concepts/calibration_estimation/index.md) for the maths and [Choosing what to estimate](https://thanhndv212.github.io/figaroh-plus/tutorials/calibration_estimation_guide/index.md) for a worked comparison.

**Done when** you can name the parameters you will estimate, the method, and the file you will validate on.

## 2. Design

Choose the poses to measure. Poses that move every joint through its range make every parameter identifiable. Poses clustered in one region leave some parameters undetermined.

Example robot

The bundled data is already well designed, so you can skip this step. To see how the poses were chosen, run:

```
MPLBACKEND=Agg python optimal_config.py
```

Your own robot

Generate a candidate pool within joint limits, then let FIGAROH select the subset that maximises information. Write your robot's `optimal_config.py` by copying the UR10 one, and record the selected poses on the robot.

**How many poses to keep** calculate_optimal_configurations(selection_method=…)

Keeps every pose whose optimal weight is above a threshold. It is simple and predictable.

Adds poses until the information gain stops improving: the relative change must stay below `iroc_rel_tol` (1e-3) for `iroc_patience` (3) steps. Use it to measure as few poses as possible.

Measure more poses than you need and keep a separate session aside for validation. See [Experiment design](https://thanhndv212.github.io/figaroh-plus/tutorials/optimal_design/index.md).

**Done when** the design report shows a condition number you accept and you have recorded both a calibration set and a validation set.

## 3. Configure

Write one YAML file that extends a shared template and only overrides what differs.

**Which template to extend** extends:

For UR10, Stäubli TX40, SO-101 and similar arms. It assumes a 6-DOF marker on the wrist.

For TIAGo, TALOS and other robots with a mobile or floating base. It turns external forces on, uses 1000 Hz sampling and a 3-DOF hand marker.

The root of both templates. Extend it only when neither fits.

Example robot

Open `config/ur10_unified_config.yaml`. The calibration block is short:

```
extends: "../../templates/manipulator_robot.yaml"
tasks:
  calibration:
    parameters:
      calibration_level: "full_params"
      outlier_threshold: 0.02          # metres
    kinematics:
      base_frame: "universe"
      tool_frame: "wrist_3_link"
    measurements:
      markers:
        - reference_joint: "wrist_3_joint"
          measurable_dof: [true, true, true, true, true, true]
    data:
      source_file: "data/calibration.csv"
      validation_data_file: ""         # set this to validate on held-out data
      number_of_samples: 29
```

Your own robot

Start from the block above and change, in order:

1. `kinematics.tool_frame` (required): the frame your marker is attached to. Set `base_frame` if your measurements are not in the `universe` frame, and `free_flying_base: true` for a floating base.
1. `measurements.markers[0].measurable_dof`: which of x, y, z, rx, ry, rz your sensor measures. The default is position only, `[true, true, true, false, false, false]`. Only the first marker is read.
1. `poses.base_pose` and `poses.tool_pose`: rough initial guesses for the sensor base and marker offset. Base and tool are always estimated freely.
1. `data.source_file`, `data.validation_data_file` and `data.number_of_samples`.

**Options to know before the first run**

Samples whose worst marker position error exceeds this many metres are dropped (default 0.05). With several markers the whole sample goes if one is an outlier. Lower it once the fit is stable.

Adds non-geometric terms. Default `false`; not supported with `map`, `map_cv` or `cv_subset`.

Set `eye_hand.enabled: true` with `camera_frame` and `reference_frame` when the sensor is a camera with an unknown 6-DOF mount. Use it with `map` and designed poses.

Must stay `0`. Use `map` or `map_cv` instead.

All keys: [Config keys](https://thanhndv212.github.io/figaroh-plus/concepts/config_parameters/index.md). How inheritance works: [Configuration system](https://thanhndv212.github.io/figaroh-plus/concepts/configuration/index.md).

**Done when** the config loads without errors and `number_of_samples` matches the rows in your CSV.

## 4. Fit

Run the solver and read the terminal report before trusting any number.

Example robot

```
python calibration.py --calibrate-only --no-plot
```

Without `--calibrate-only`, the script also exports an updated URDF and opens a viewer.

Your own robot

```
from figaroh.tools.robot import load_robot
from examples.my_robot.utils.my_robot_tools import MyRobotCalibration

robot = load_robot("urdf/my_robot.urdf", load_by_urdf=True)
cal = MyRobotCalibration(robot, "config/my_robot_config.yaml")
cal.initialize()
cal.solve(html_report=True, save_results=True)
```

`solve()` runs Levenberg–Marquardt (`method="lm"`) for up to `max_iterations=3` outer passes, removing outliers between passes. Pass `outlier_threshold=` to override the config for one run.

Read the report in this order:

1. **Condition number.** A very large value means some parameters are not identifiable from your poses. Go back to [Design](#design) or estimate fewer parameters.
1. **RMSE before and after.** It should drop substantially. If it does not, check the frames and the column order in your CSV.
1. **Outliers.** More than a few percent usually means a units or frame mistake, not bad data.
1. **Parameter values.** Offsets of several degrees or several centimetres are suspicious. With `map` methods, compare each value with its reported uncertainty.

**Done when** the RMSE drops, few samples are dropped as outliers, and every parameter value is physically plausible.

## 5. Verify

A good fit on the training data is not proof. Check prediction on poses the solver never saw.

**What the verdict covers** verify(scope=…)

Requires a held-out set (`data.validation_data_file`) and explicit limits that you set. It checks `position_rmse_mm`, plus `orientation_rmse_deg` when rotation is measured. FIGAROH sets no default limit, because acceptable error depends on your application.

Only checks that the fit produced finite numbers. Use it in CI or as a smoke test. It says nothing about accuracy.

```
verdict = cal.verify(
    thresholds={"position_rmse_mm": {"threshold": 1.0, "comparison": "max"}},
)
print(verdict.passed)
cal.export_verification_report()   # results/calibration_verification.json
```

Without validation data, FIGAROH falls back to the calibration data and prefixes the metrics with `training_`, labelling them `calibration_data_fallback`. A fallback can never pass a prediction check. A missing required metric reads `not_evaluated`, not pass.

Example robot

The UR10 config has no validation file, so its prediction verdict reads `not_evaluated`. Run `python calibration.py` without flags to export the updated URDF and check its forward kinematics.

Your own robot

Export the corrected URDF once the verdict passes, then reload it and compare predicted poses on a fresh session. The [Reports & verdicts](https://thanhndv212.github.io/figaroh-plus/reporting_and_verification/index.md) page covers the HTML report and the two-run compare page.

**Done when** `verify()` passes in prediction scope on held-out data with limits you chose, and the exported URDF reproduces the validation poses.
