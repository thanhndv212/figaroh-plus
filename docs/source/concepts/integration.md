# Integration API

`figaroh.integration` wraps the [backend abstraction](backends.md) and the
`BaseIdentification` workflow class into a one-line, script-friendly API —
for when you want a quick identification run without writing a
robot-specific `BaseIdentification` subclass (see the
[tutorials](../tutorials/index.md) and
[Examples Gallery](../examples/index.md) for the subclassing pattern the
full examples use instead).

```python
from figaroh.integration import RobotIdentificationSystem

# Create from URDF, defaulting to the Pinocchio backend
system = RobotIdentificationSystem.from_urdf("robot.urdf")

# Or pick a backend explicitly
system = RobotIdentificationSystem.from_urdf("robot.urdf", backend="mujoco")

# Run identification against CSV trajectory data
results = system.identify_parameters(
    config="config/robot_config.yaml",
    data_dir="data/",
)

print(f"Identified {len(results.phi_base)} base parameters")
print(f"RMS error: {results.rms_error}")
```

## `RobotIdentificationSystem`

- `from_urdf(urdf_path, backend="pinocchio", package_dirs=None, free_flyer=False, **kwargs)`
  — construct from a URDF, loading it through `figaroh.tools.load_robot`.
- `from_mjcf(mjcf_path, ...)` — reserved for MJCF-native workflows; not yet
  implemented (`BaseIdentification` currently requires a URDF-based `Robot`
  object; raises `NotImplementedError` until that's refactored).
- `identify_parameters(config, data_dir=None, decimate=True, decimation_factor=10, zero_tolerance=0.001, plotting=False, save_results=False, **kwargs)`
  — loads `q`/`dq`/`ddq`/`tau` CSVs from `data_dir` (or `data_files` in the
  config), builds the regressor, solves for base parameters, and returns an
  `IdentificationResult`.

## `IdentificationResult`

A plain dataclass: `phi_base`, `params_base`, `phi_standard` (if
physical-consistency reconstruction is enabled), `rms_error`, `correlation`,
`backend`, `model_path`, `config`, and `raw` (the full result dict from
`BaseIdentification`, for anything not surfaced on the dataclass directly).

## When to use this vs. a full example

`RobotIdentificationSystem` is the fast path: one call, CSV-in the four
required columns (`q.csv`, `dq.csv`, `ddq.csv`, `tau.csv`), a result object
out. The [Examples Gallery](../examples/index.md) robots
(UR10, TIAGo, TALOS, Staubli TX40) instead subclass `BaseCalibration` /
`BaseIdentification` directly — reach for that pattern once you need
robot-specific cost functions, custom data loaders, or the full reporting/
verification suite wired into a CLI script.

See [API Reference → Integration](../api/integration.md) for the complete
signatures.
