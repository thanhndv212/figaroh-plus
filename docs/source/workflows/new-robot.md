# Add your robot

<p class="fg-lead">Give your robot a folder in figaroh-examples with a URDF, a config and a tools subclass, so the calibration and identification guides work for it.</p>

Only do this when `figaroh-examples/examples/` has no folder for your robot yet. Otherwise go straight to [Calibrate a robot](calibration.md) or [Identify dynamics](identification.md).

<div class="fg-choice" markdown>
**Which example to copy from**

=== "UR10 <span class='fg-rec'>Fixed-base arms</span>"

    `examples/ur10/` with `templates/manipulator_robot.yaml`. It is the most complete reference: calibration, identification, both design scripts and verification flags. Use it for UR, KUKA, Stäubli and similar arms.

=== "TIAGo / TALOS"

    `examples/tiago/` or `examples/talos/` with `templates/humanoid_robot.yaml`. Use them for mobile manipulators, humanoids, floating bases and coupled wrists.
</div>

## 1. Place the description package

If your URDF references meshes with `package://<robot>_description/...`, put that package at `figaroh-examples/models/<robot>_description/`. Add to `models/`; do not edit what is already there. Skip this if the URDF has no mesh references.

## 2. Scaffold the folder

```bash
cd figaroh-examples/examples
./create_example.sh my_robot
```

This creates `examples/my_robot/` with `config/`, `data/`, `urdf/`, `utils/` and `docs/`. The generated `config/my_robot_unified_config.yaml` is a real config with TODO markers; edit it in place. The four generated scripts (`calibration.py`, `identification.py`, `optimal_config.py`, `optimal_trajectory.py`) are placeholders that point to their UR10 counterparts.

## 3. Add the URDF and prove it loads

Put the file at `examples/my_robot/urdf/my_robot.urdf`, then print its joints and frames:

```bash
cd figaroh-examples/examples/my_robot
python -c "
from figaroh.tools.robot import load_robot
r = load_robot('urdf/my_robot.urdf', package_dirs='../../models', load_by_urdf=True)
print('joints', list(r.model.names))
print('frames', [f.name for f in r.model.frames][:40])
"
```

Copy names from this output. They are the exact strings your config's `active_joints`, your CSV headers, and `base_frame` / `tool_frame` must use.

## 4. Fill in the config

```yaml
extends: "../../templates/manipulator_robot.yaml"   # or humanoid_robot.yaml
robot:
  name: "my_robot"
  properties:
    joints:
      active_joints: ["joint_1", "joint_2"]   # exact names from step 3
      joint_limits:
        position: [...]       # rad, min/max per joint, flattened
        velocity: [...]       # rad/s
        acceleration: [...]   # rad/s²
        torque: [...]         # N·m
tasks:
  calibration:    { enabled: true }
  identification: { enabled: true }
```

- Use `extends:`. The key `inherit_from` is not read.
- Robot-wide facts go under `robot.properties`, task settings under `tasks.<task>`. Task settings placed under `robot.properties` are silently ignored.
- Enable only the tasks you have scripts for.

## 5. Write the tools subclass and scripts

Replace `utils/my_robot_tools.py` with a copy of `examples/ur10/utils/ur10_tools.py` and adapt it:

| Task | Base class | You implement |
|---|---|---|
| Calibration | `BaseCalibration` | data and marker handling for your sensor |
| Identification | `BaseIdentification` | `load_trajectory_data()` |
| Design | `BaseOptimalCalibration`, `BaseOptimalTrajectory` | your constraints |

Then copy the entry scripts from `examples/ur10/` and change the class import, the `--config` and `--urdf` defaults, and the data paths. Keep the command-line flags the same as UR10's so `validate.py` can drive your robot.

## 6. Add data and run

Put your CSVs under `data/` and follow the Configure step of the [calibration](calibration.md#configure) or [identification](identification.md#configure) guide.

<div class="fg-done" markdown>
**Done when** `python validate.py --robot my_robot` exits 0 from the `figaroh-examples` root, and a run archive appears under `examples/my_robot/results/runs/`.
</div>

!!! tip "Let your agent do it"
    The `figaroh-setup-new-robot` skill walks a coding agent through these steps, including registering the robot in `validate.py`. See [AI agents](../agents/index.md).
