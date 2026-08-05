# Configuration System

FIGAROH uses a flexible YAML-based configuration system that supports both
a modern unified format (with template inheritance) and a legacy flat
format. Every robot example in
[figaroh-examples](https://github.com/thanhndv212/figaroh-examples) ships a
config under `examples/<robot>/config/`; this page explains the format
those files follow.

## Unified configuration format

The modern unified format provides better organization and template
inheritance:

```yaml
# modern_config.yaml
inherit_from: "templates/base_robot.yaml"

robot:
  name: "tiago"
  urdf_path: "urdf/tiago.urdf"

calibration:
  method: "full_params"
  sensor_type: "camera"

  markers:
    - ref_joint: "wrist_3_joint"
      position: [0.1, 0.0, 0.05]
      measure: [true, true, true, true, true, true]

identification:
  mechanics:
    friction_coefficients:
      viscous: [0.01, 0.02, 0.015]
      static: [0.001, 0.002, 0.0015]
    actuator_inertias: [0.1, 0.15, 0.12]

  signal_processing:
    sampling_frequency: 5000.0
    cutoff_frequency: 100.0
```

## Legacy format support

Existing configurations continue to work without modification:

```yaml
# legacy_config.yaml
calibration:
  calib_level: full_params
  markers:
    - ref_joint: wrist_3_joint
      measure: [True, True, True, True, True, True]

identification:
  robot_params:
    - fv: [0.01, 0.02, 0.015]
      fs: [0.001, 0.002, 0.0015]
  processing_params:
    - ts: 0.0002
      cut_off_frequency_butterworth: 100.0
```

## Parsing a config

```python
from figaroh.utils.config_parser import UnifiedConfigParser

# Parse any configuration format
parser = UnifiedConfigParser("config/robot_config.yaml")
config = parser.parse()

# Create task-specific configuration
calib_config = parser.create_task_config(robot, config, "calibration")
identif_config = parser.create_task_config(robot, config, "identification")
```

## Template inheritance

The unified format's real power is `extends:` — a robot config file
references a shared template so it only needs to specify what's actually
robot-specific. Templates and real examples live in
[figaroh-examples/examples/templates](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/templates):

| Template | Extends | For |
|---|---|---|
| `base_robot_config.yaml` | — (root of the chain) | Defines the full schema (`meta`, `robot`, `tasks`, `environment`, `custom`); every other template extends this |
| `manipulator_robot.yaml` | `base_robot_config.yaml` | Fixed-base serial arms (UR10, Staubli TX40, KUKA) — no mobile base, wrist-mounted 6-DOF marker default |
| `humanoid_robot.yaml` | `base_robot_config.yaml` | Humanoid / mobile manipulators (TIAGo, TALOS) — coupled wrists, mobile base, head camera + F/T sensors, 1000 Hz default sampling |

A robot config points at the template that matches its kinematic shape:

```yaml
# examples/ur10/config/ur10_unified_config.yaml
extends: "../../templates/manipulator_robot.yaml"

robot:
  name: "ur10"
  properties:
    joints:
      active_joints: ["shoulder_pan_joint", "shoulder_lift_joint", "..."]
    mechanics:
      reduction_ratios: [50.0, 50.0, 50.0, 50.0, 50.0, 50.0]
```

`UnifiedConfigParser` resolves the chain (robot config → template →
`base_robot_config.yaml`) and deep-merges it:

- **Scalars** (strings, numbers, booleans): the more specific file wins.
- **Dictionaries**: merged recursively — keys only in the template are kept,
  keys only in the robot config are added.
- **Lists**: replaced entirely, not merged — a robot's `markers:` list
  replaces the template's, it doesn't append to it.

### Creating a new robot config

1. Pick the template that matches your robot: `manipulator_robot.yaml` for a
   fixed-base arm, `humanoid_robot.yaml` for a humanoid or mobile
   manipulator, `base_robot_config.yaml` directly for anything else.
2. Point a new config at it with `extends:` and fill in only what differs
   (joint names, limits, friction coefficients, marker configuration).
3. Set `tasks.<task>.enabled: true` for each task your robot supports.

`examples/create_example.sh <robot_name>` in figaroh-examples scaffolds this
structure for you from the TIAGo layout — see the
[Examples Gallery](../examples/templates.md) for the full inheritance rules
and worked examples.

## Next steps

- [Configuration Parameter Reference](config_parameters.md) — what each
  `calibration:`/`identification:` field actually means, its units, and
  which pipeline step consumes it.
- [Tutorials](../tutorials/index.md) walk through a full calibration and
  identification run using these configs.
- [Examples Gallery → Config Templates](../examples/templates.md) has the
  complete override rules and best practices for template authors.
