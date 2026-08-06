# Calibration Walkthrough

## The problem

Manufacturing tolerances and assembly errors give every physical robot
systematic geometric errors — joint zero-offsets, link length/orientation
deviations from CAD, tool-center-point misalignment — typically 5-10mm at
the end-effector. That's enough to make precision tasks (contact-rich
manipulation, vision-guided grasping, multi-robot coordination) unreliable.

Kinematic calibration corrects this using external measurements (motion
capture, a tracked marker, a camera) of the end-effector at a set of known
joint configurations.

## The model

```
P_measured = forward_kinematics(q, θ_nominal + Δθ) + ε
```

- `q` — commanded joint angles (known)
- `θ_nominal` — nominal kinematic parameters from the URDF
- `Δθ` — parameter corrections FIGAROH solves for
- `P_measured` — external sensor measurement of the end-effector (or
  another tracked frame) at each `q`
- `ε` — measurement noise

FIGAROH minimizes the residual between `P_measured` and the predicted pose
over `Δθ` using Levenberg-Marquardt, then reports which corrections are
well- vs. poorly-observed given your measurement set.

## Running it

Every robot in [figaroh-examples](https://github.com/thanhndv212/figaroh-examples)
provides a `BaseCalibration` subclass (e.g. `TiagoCalibration`) that
supplies the robot-specific frame/marker configuration; the shape is the
same for every robot:

```python
from examples.tiago.utils.tiago_tools import TiagoCalibration
from figaroh.tools.robot import load_robot

robot = load_robot("path/to/robot.urdf", load_by_urdf=True)
calibrator = TiagoCalibration(robot, "config/tiago_unified_config.yaml")
calibrator.initialize()
result = calibrator.solve(plotting=False, html_report=True)
```

`solve()` always prints a terminal quality report (condition number,
per-DOF residual RMSE, outlier count). `html_report=True` additionally
writes a self-contained diagnostic page — see
[Reporting & Verification](../reporting_and_verification.md).

From the command line, the TIAGo example wraps this in one script:

```bash
cd examples/tiago
python calibration.py                 # full pipeline: calibrate → plot → save → export → viser viz
python calibration.py --calibrate-only  # just solve, skip URDF export
python calibration.py --update-model    # export a corrected URDF from a saved result
```

## What you configure

The `calibration:` section of your [unified config](../concepts/configuration.md)
declares:

- `base_frame` / `tool_frame` — the kinematic chain endpoints
- `markers` — which frame(s) are measured, and which of their 6 DOF are
  observable (`measure: [x, y, z, roll, pitch, yaw]`) — a wrist-mounted
  6-DOF tracker measures all six; a hand-mounted marker seen by a fixed
  camera might only give 3-DOF position
- `nb_sample` — number of configurations in your measurement set (more
  isn't always better — see [Optimal Experiment Design](optimal_design.md)
  for how to pick the *best* set instead of the largest)

## Expected results

A well-conditioned calibration (condition number well below the default
1000.0 threshold — see [verification thresholds](../reporting_and_verification.md))
typically takes end-effector position error from 5-10mm down to under 1mm,
validated on a held-out set of configurations not used to fit `Δθ`.

## Next steps

- [Identification Walkthrough](identification_walkthrough.md) — once
  kinematics are corrected, identify the *dynamic* parameters (mass,
  inertia, friction) for control and simulation.
- [Reporting & Verification](../reporting_and_verification.md) — turn
  this run into a shareable report and a CI-gateable pass/fail check.
- [Examples Gallery](../examples/index.md) — complete calibration scripts
  for UR10, TIAGo, and TALOS.
