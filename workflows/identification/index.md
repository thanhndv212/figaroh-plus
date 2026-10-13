# Identify dynamics

Fit masses, inertias, friction and actuator inertia from recorded motion and joint torques. Five steps, each with the choices you make and how you know it worked.

Use the toggle below to read the steps for the bundled UR10 example or for your own robot. Each choice card lists the options FIGAROH supports, and the badge marks the usual starting point.

## 1. Plan

FIGAROH writes joint torque as a linear function of the inertial parameters and estimates the **base parameters**: the combinations of masses and inertias your data can actually determine. Decide which extra effects to model and check that you have the data for them.

Example robot

The UR10 example ships a training trajectory and a separate validation trajectory (`data/truth/`), recorded at 100 Hz.

```
git clone https://github.com/thanhndv212/figaroh-examples
cd figaroh-examples/examples/ur10
```

Your own robot

You need:

- a URDF with reasonable nominal inertias (used as the starting point and for reconstruction);
- joint positions and joint torques (or motor currents with a known torque constant) from an exciting motion;
- a **separate** recording, in the same layout, to validate on;
- the true sampling rate, read from the timestamps.

If your robot has no folder under `figaroh-examples/examples/` yet, do [Add your robot](https://thanhndv212.github.io/figaroh-plus/workflows/new-robot/index.md) first.

**What to model** tasks.identification.problem.model_components

All components `false`. Fits masses, first moments and inertias. Get this working before adding terms.

Adds viscous and Coulomb friction per joint. Almost always needed on geared robots. Turn it on when torque residuals change sign with velocity.

Adds rotor inertia reflected through the gearbox. Use it on high-ratio drives when residuals track acceleration.

Adds a constant torque offset per joint, for sensor bias.

**Which torque signal** tasks.identification.problem

Fits measured joint torques. `torque_fit_joints` restricts the fit to a list of joints.

Fits base wrenches from force/torque sensors (`force_torque_sensors`, `external_wrench_offsets`). Use it for floating-base robots. The humanoid template turns it on.

**Done when** you can name the model components, the torque source, and the file you will validate on.

## 2. Design

A slow or repetitive motion leaves most parameters undetermined. An exciting trajectory moves every joint with varied velocity and acceleration within its limits.

Example robot

The bundled trajectory is already exciting, so you can skip this step. To see how it was generated:

```
MPLBACKEND=Agg python optimal_trajectory.py
```

Your own robot

Copy the UR10 `optimal_trajectory.py` and its `tasks.optimal_trajectory` block (waypoints, segment duration, joint box, velocity and acceleration scaling). Execute the result on the robot and record a second, different motion for validation.

`OptimalTrajectory.solve(stack_reps=2)` minimises the condition number of the base regressor. It needs cyipopt, which installs from conda only (`conda env create -f environment.yml`). See [Experiment design](https://thanhndv212.github.io/figaroh-plus/tutorials/optimal_design/index.md).

**Done when** the trajectory stays within limits, its condition number is acceptable, and you have recorded both a training and a validation motion.

## 3. Configure

Write one YAML file that extends a shared template. Identification settings sit under `tasks.identification`.

Example robot

```
extends: "../../templates/manipulator_robot.yaml"
tasks:
  identification:
    problem:
      use_joint_torques: true
      model_components:
        friction: false
        actuator_inertia: false
        joint_offset: false
    signal_processing:
      sampling_frequency: 100.0   # Hz, matches the recorded clock
      cutoff_frequency: 10.0      # Hz, below Nyquist (50 Hz)
      filter_type: "butterworth"
      filter_order: 4
    data:
      validation_data_file: "data/truth/validation.csv"
```

Your own robot

Pick the template (`manipulator_robot.yaml` for fixed-base arms, `humanoid_robot.yaml` for mobile or floating bases), then set:

1. `signal_processing.sampling_frequency` from your timestamps. The template defaults (500 Hz, 1000 Hz for humanoids) are rarely right for your logger.
1. `cutoff_frequency` above the motion content and below half the sampling rate.
1. `data.validation_data_file` to the separate recording.
1. `coupling.has_coupled_wrist` to `false` unless your wrist joints are mechanically coupled.

**Velocity and acceleration from positions** signal_processing.differentiation_method

Second-order central differences in the interior, one-sided at the ends. Good for most logs.

Central differences. Similar to `gradient`.

One-sided differences. Use them only to match another tool's processing.

**Beyond base parameters** tasks.identification

No extra keys. You get the identifiable combinations, which predict torque but are not individual link masses.

`reconstruction: {enabled: true, method: nullspace}` recovers a full per-link parameter set closest to the URDF. `method: sdp` enforces physical consistency (needs picos); `auto` tries `sdp` and falls back to `nullspace`.

`physical_consistency: {enabled: true}` checks or enforces positive masses and positive-definite inertias (`mass_min`, `psd_eig_tol`, `solver: cvxopt`).

Chooses which result is reported and exported: `fit`, `reconstruction` or `physical_fit`. `physical_fit` is a research path.

If the log warns that the regressor is ill-conditioned, set `problem.qr_relative_tolerance: 1e-4`. All keys: [Config keys](https://thanhndv212.github.io/figaroh-plus/concepts/config_parameters/index.md).

**Done when** the config loads, the sampling rate matches your timestamps, and the cutoff is below Nyquist.

## 4. Fit

Example robot

```
MPLBACKEND=Agg python identification.py
```

The script writes an HTML report and runs `verify()` by default. Add `--wls` to refine with weighted least squares.

Your own robot

```
from figaroh.tools.robot import load_robot
from examples.my_robot.utils.my_robot_tools import MyRobotIdentification

robot = load_robot("urdf/my_robot.urdf", load_by_urdf=True)
ident = MyRobotIdentification(robot, "config/my_robot_config.yaml")
ident.initialize()
ident.solve(plotting=False, html_report=True, save_results=True)
```

**Which estimator** solve(…)

`solve()` reduces the regressor to base parameters by QR decomposition and solves ordinary least squares. It decimates by `decimation_factor=10` first to cut noise and size.

`solve(wls=True)` reweights each joint by its residual variance (iteratively reweighted least squares). Use it when joints have very different torque noise. The `problem.wls` config key is not read; pass the argument.

`solve_with_custom_solver(method=…)` takes one of `lstsq`, `qr`, `svd`, `ridge`, `lasso`, `elastic_net`, `tikhonov`, `constrained`, `robust`, `weighted`. Use it for regularisation or bounds.

Read the report in this order:

1. **Condition number of the base regressor.** A very large value means your motion does not excite some parameters. Go back to [Design](#design).
1. **Torque fit per joint.** Residuals should look like noise. Structure that follows velocity points to friction, and structure that follows acceleration points to actuator inertia.
1. **Relative standard deviation per parameter.** Large values mark parameters the data barely determines.

**Done when** the conditioning is acceptable, residuals look like noise on every joint, and well-determined parameters have small relative deviation.

## 5. Verify

Check torque prediction on the separate recording.

**What the verdict covers** verify(scope=…)

Requires the validation recording and a `validation_rmse:<joint>` limit in N·m for **every** active joint. Optional checks: `validation_abs_bias`, `validation_peak_error`. FIGAROH sets no default limits.

Only checks that the fit produced finite, consistent outputs. Use it in CI or as a smoke test.

```
limits = {f"validation_rmse:{j}": {"threshold": 2.0, "comparison": "max"}
          for j in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]}
verdict = ident.verify(thresholds=limits)
print(verdict.passed)
ident.export_verification_report()   # results/identification_verification.json
```

From the example scripts, pass the same limits on the command line:

```
python identification.py --verification-scope prediction \
  --acceptance-profile '{"validation_rmse:elbow_joint": {"threshold": 2.0, "comparison": "max"}, ...}'
```

Without validation data, metrics fall back to the training data, are labelled `identification_data_fallback`, and cannot pass a prediction check.

Your own robot

To compare two runs, for example with and without friction, build a static compare page with `figaroh.tools.compare_report.generate_compare_page`. To write identified inertias back to a URDF, every exported link needs all ten inertial values, and physically infeasible sets are refused. Reload the written URDF and check its predicted torques before using it. [Reports & verdicts](https://thanhndv212.github.io/figaroh-plus/reporting_and_verification/index.md) covers the HTML report and the stage records.

**Done when** `verify()` passes in prediction scope on the separate recording, with a limit you chose for every joint.
