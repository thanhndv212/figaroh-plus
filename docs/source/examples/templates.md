# Config Templates

Reusable YAML starting points for robot configs — every example in this
gallery extends one of these instead of writing its config from scratch.
[→ source folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/templates)

See [Configuration System](../concepts/configuration.md) for how `extends:`
resolution and merging work; this page covers the templates themselves and
how to author a new robot config against them.

## Available templates

- **`base_robot_config.yaml`** — root of the inheritance chain. Defines the
  full schema (`meta`, `robot.properties`, `tasks`, `environment`,
  `custom`) with every task type present as a documented stub.
- **`manipulator_robot.yaml`** (extends the base) — fixed-base serial arms.
  `robot.type: "manipulator"`, no mobile base, calibration + identification
  enabled by default, wrist-mounted 6-DOF marker default. Used by UR10,
  Staubli TX40.
- **`humanoid_robot.yaml`** (extends the base) — humanoid / mobile
  manipulators. `robot.type: "humanoid"`, coupled wrists, mobile base
  (differential drive), head camera + F/T sensors, 1000 Hz default sampling,
  eye-hand calibration section, hand-mounted 3-DOF (position-only) marker
  default. Used by TIAGo, TALOS.

## Writing a new robot config

```yaml
# examples/my_robot/config/my_robot_unified_config.yaml
extends: "../../templates/manipulator_robot.yaml"

robot:
  name: "my_robot"
  properties:
    joints:
      active_joints: ["joint_1", "joint_2", "..."]
  # override only what differs from the template
```

1. Pick `manipulator_robot.yaml` or `humanoid_robot.yaml` (or
   `base_robot_config.yaml` directly if neither fits).
2. Set `tasks.<task>.enabled: true` for each task your robot supports.
3. Fill in joint names, limits, friction, marker configuration — everything
   else inherits the template default.

## Best practices

- Keep robot specifics under `robot.properties.*`; task specifics under
  `tasks.<task>.*`; environment/execution settings under `environment.*`.
- **Don't** put `extends:` inside a template — templates are the root of
  the chain, not participants in it.
- **Don't** put robot-specific joint names/limits in a template — that
  defeats the point of sharing it.
- **Don't** delete a template key from your robot config to "remove" it —
  override it or leave the default.
- Adding a new default parameter to a template is safe (existing configs
  just inherit it); removing or renaming one breaks every config that
  extends it.

## Verifying a config parses

```python
from figaroh.utils.config_parser import UnifiedConfigParser
parser = UnifiedConfigParser("config/my_robot_unified_config.yaml")
config = parser.parse()
print(config["robot"]["name"])
```

## See also

- [Configuration System](../concepts/configuration.md) — the conceptual
  model (unified vs. legacy format, `extends:` merge rules).
- `examples/create_example.sh` — scaffolds a new robot folder (config
  included) from the TIAGo layout.
- Real-world configs extending these templates:
  [`ur10_unified_config.yaml`](https://github.com/thanhndv212/figaroh-examples/blob/main/examples/ur10/config/ur10_unified_config.yaml),
  [`tiago_unified_config.yaml`](https://github.com/thanhndv212/figaroh-examples/blob/main/examples/tiago/config/tiago_unified_config.yaml),
  [`staubli_tx40_unified_config.yaml`](https://github.com/thanhndv212/figaroh-examples/blob/main/examples/staubli_tx40/config/staubli_tx40_unified_config.yaml)
