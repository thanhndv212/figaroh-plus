---
hide: [toc]
---

# Identify dynamics

<p class="fg-lead">Fit masses, inertias, friction and actuator inertia from recorded motion and joint torques. Five steps, each with the choices you make and how you know it worked.</p>

Use the toggle below to read the steps for the bundled UR10 example or for your own robot. Each choice card lists the options FIGAROH supports, and the badge marks the usual starting point.

<div class="fg-guide" markdown>

<div class="fg-step" id="plan" data-title="Plan" markdown>

## 1. Plan

FIGAROH writes joint torque as a linear function of the inertial parameters and estimates the **base parameters**: the combinations of masses and inertias your data can actually determine. Decide which extra effects to model and check that you have the data for them.

<div data-fg-path="example" markdown>
<span class="fg-path-tag">Example robot</span>

The UR10 example ships a training trajectory and a separate validation trajectory (`data/truth/`), recorded at 100 Hz.

```bash
git clone https://github.com/thanhndv212/figaroh-examples
cd figaroh-examples/examples/ur10
```
</div>

<div data-fg-path="own" markdown>
<span class="fg-path-tag">Your own robot</span>

You need:

- a URDF with reasonable nominal inertias (used as the starting point and for reconstruction);
- joint positions and joint torques (or motor currents with a known torque constant) from an exciting motion;
- a **separate** recording, in the same layout, to validate on;
- the true sampling rate, read from the timestamps.

If your robot has no folder under `figaroh-examples/examples/` yet, do [Add your robot](new-robot.md) first.
</div>

<div class="fg-choice" markdown>
**What to model** <span class="fg-choice__key">tasks.identification.problem.model_components</span>

=== "Rigid body only <span class='fg-rec'>Start here</span>"

    All components `false`. Fits masses, first moments and inertias. Get this working before adding terms.

=== "friction"

    Adds viscous and Coulomb friction per joint. Almost always needed on geared robots. Turn it on when torque residuals change sign with velocity.

=== "actuator_inertia"

    Adds rotor inertia reflected through the gearbox. Use it on high-ratio drives when residuals track acceleration.

=== "joint_offset"

    Adds a constant torque offset per joint, for sensor bias.
</div>

<div class="fg-choice" markdown>
**Which torque signal** <span class="fg-choice__key">tasks.identification.problem</span>

=== "use_joint_torques <span class='fg-rec'>Default</span>"

    Fits measured joint torques. `torque_fit_joints` restricts the fit to a list of joints.

=== "include_external_forces"

    Fits base wrenches from force/torque sensors (`force_torque_sensors`, `external_wrench_offsets`). Use it for floating-base robots. The humanoid template turns it on.
</div>

<div class="fg-done" markdown>
**Done when** you can name the model components, the torque source, and the file you will validate on.
</div>

</div>

<div class="fg-step" id="design" data-title="Design" markdown>

## 2. Design

A slow or repetitive motion leaves most parameters undetermined. An exciting trajectory moves every joint with varied velocity and acceleration within its limits.

<div data-fg-path="example" markdown>
<span class="fg-path-tag">Example robot</span>

The bundled trajectory is already exciting, so you can skip this step. To see how it was generated:

```bash
MPLBACKEND=Agg python optimal_trajectory.py
```
</div>

<div data-fg-path="own" markdown>
<span class="fg-path-tag">Your own robot</span>

Copy the UR10 `optimal_trajectory.py` and its `tasks.optimal_trajectory` block (waypoints, segment duration, joint box, velocity and acceleration scaling). Execute the result on the robot and record a second, different motion for validation.
</div>

`OptimalTrajectory.solve(stack_reps=2)` minimises the condition number of the base regressor. It needs cyipopt, which installs from conda only (`conda env create -f environment.yml`). See [Experiment design](../tutorials/optimal_design.md).

<div class="fg-done" markdown>
**Done when** the trajectory stays within limits, its condition number is acceptable, and you have recorded both a training and a validation motion.
</div>

</div>

<div class="fg-step" id="configure" data-title="Configure" markdown>

## 3. Configure

Write one YAML file that extends a shared template. Identification settings sit under `tasks.identification`.

<div data-fg-path="example" markdown>
<span class="fg-path-tag">Example robot</span>

```yaml
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
</div>

<div data-fg-path="own" markdown>
<span class="fg-path-tag">Your own robot</span>

Pick the template (`manipulator_robot.yaml` for fixed-base arms, `humanoid_robot.yaml` for mobile or floating bases), then set:

1. `signal_processing.sampling_frequency` from your timestamps. The template defaults (500 Hz, 1000 Hz for humanoids) are rarely right for your logger.
2. `cutoff_frequency` above the motion content and below half the sampling rate.
3. `data.validation_data_file` to the separate recording.
4. `coupling.has_coupled_wrist` to `false` unless your wrist joints are mechanically coupled.
</div>

<div class="fg-choice" markdown>
**Velocity and acceleration from positions** <span class="fg-choice__key">signal_processing.differentiation_method</span>

=== "gradient <span class='fg-rec'>Default</span>"

    Second-order central differences in the interior, one-sided at the ends. Good for most logs.

=== "central"

    Central differences. Similar to `gradient`.

=== "forward / backward"

    One-sided differences. Use them only to match another tool's processing.
</div>

<div class="fg-choice" markdown>
**Beyond base parameters** <span class="fg-choice__key">tasks.identification</span>

=== "Base parameters only <span class='fg-rec'>Default</span>"

    No extra keys. You get the identifiable combinations, which predict torque but are not individual link masses.

=== "reconstruction"

    `reconstruction: {enabled: true, method: nullspace}` recovers a full per-link parameter set closest to the URDF. `method: sdp` enforces physical consistency (needs picos); `auto` tries `sdp` and falls back to `nullspace`.

=== "physical_consistency"

    `physical_consistency: {enabled: true}` checks or enforces positive masses and positive-definite inertias (`mass_min`, `psd_eig_tol`, `solver: cvxopt`).

=== "select_stage"

    Chooses which result is reported and exported: `fit`, `reconstruction` or `physical_fit`. `physical_fit` is a research path.
</div>

If the log warns that the regressor is ill-conditioned, set `problem.qr_relative_tolerance: 1e-4`. All keys: [Config keys](../concepts/config_parameters.md).

<div class="fg-done" markdown>
**Done when** the config loads, the sampling rate matches your timestamps, and the cutoff is below Nyquist.
</div>

</div>

<div class="fg-step" id="fit" data-title="Fit" markdown>

## 4. Fit

<div data-fg-path="example" markdown>
<span class="fg-path-tag">Example robot</span>

```bash
MPLBACKEND=Agg python identification.py
```

The script writes an HTML report and runs `verify()` by default. Add `--wls` to refine with weighted least squares.
</div>

<div data-fg-path="own" markdown>
<span class="fg-path-tag">Your own robot</span>

```python
from figaroh.tools.robot import load_robot
from examples.my_robot.utils.my_robot_tools import MyRobotIdentification

robot = load_robot("urdf/my_robot.urdf", load_by_urdf=True)
ident = MyRobotIdentification(robot, "config/my_robot_config.yaml")
ident.initialize()
ident.solve(plotting=False, html_report=True, save_results=True)
```
</div>

<div class="fg-choice" markdown>
**Which estimator** <span class="fg-choice__key">solve(…)</span>

=== "OLS <span class='fg-rec'>Default</span>"

    `solve()` reduces the regressor to base parameters by QR decomposition and solves ordinary least squares. It decimates by `decimation_factor=10` first to cut noise and size.

=== "WLS"

    `solve(wls=True)` reweights each joint by its residual variance (iteratively reweighted least squares). Use it when joints have very different torque noise. The `problem.wls` config key is not read; pass the argument.

=== "Custom solver"

    `solve_with_custom_solver(method=…)` takes one of `lstsq`, `qr`, `svd`, `ridge`, `lasso`, `elastic_net`, `tikhonov`, `constrained`, `robust`, `weighted`. Use it for regularisation or bounds.
</div>

Read the report in this order:

1. **Condition number of the base regressor.** A very large value means your motion does not excite some parameters. Go back to [Design](#design).
2. **Torque fit per joint.** Residuals should look like noise. Structure that follows velocity points to friction, and structure that follows acceleration points to actuator inertia.
3. **Relative standard deviation per parameter.** Large values mark parameters the data barely determines.

<div class="fg-done" markdown>
**Done when** the conditioning is acceptable, residuals look like noise on every joint, and well-determined parameters have small relative deviation.
</div>

</div>

<div class="fg-step" id="verify" data-title="Verify" markdown>

## 5. Verify

Check torque prediction on the separate recording.

<div class="fg-choice" markdown>
**What the verdict covers** <span class="fg-choice__key">verify(scope=…)</span>

=== "prediction <span class='fg-rec'>Default</span>"

    Requires the validation recording and a `validation_rmse:<joint>` limit in N·m for **every** active joint. Optional checks: `validation_abs_bias`, `validation_peak_error`. FIGAROH sets no default limits.

=== "execution"

    Only checks that the fit produced finite, consistent outputs. Use it in CI or as a smoke test.
</div>

```python
limits = {f"validation_rmse:{j}": {"threshold": 2.0, "comparison": "max"}
          for j in ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
                    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]}
verdict = ident.verify(thresholds=limits)
print(verdict.passed)
ident.export_verification_report()   # results/identification_verification.json
```

From the example scripts, pass the same limits on the command line:

```bash
python identification.py --verification-scope prediction \
  --acceptance-profile '{"validation_rmse:elbow_joint": {"threshold": 2.0, "comparison": "max"}, ...}'
```

Without validation data, metrics fall back to the training data, are labelled `identification_data_fallback`, and cannot pass a prediction check.

<div data-fg-path="own" markdown>
<span class="fg-path-tag">Your own robot</span>

To compare two runs, for example with and without friction, build a static compare page with `figaroh.tools.compare_report.generate_compare_page`. To write identified inertias back to a URDF, every exported link needs all ten inertial values, and physically infeasible sets are refused. Reload the written URDF and check its predicted torques before using it. [Reports & verdicts](../reporting_and_verification.md) covers the HTML report and the stage records.
</div>

<div class="fg-done" markdown>
**Done when** `verify()` passes in prediction scope on the separate recording, with a limit you chose for every joint.
</div>

</div>

</div>
