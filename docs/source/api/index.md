# API Reference

Generated from docstrings in `src/figaroh/` via
[mkdocstrings](https://mkdocstrings.github.io/). One page per top-level
subpackage:

| Module | Contains |
|---|---|
| [Calibration](calibration.md) | `BaseCalibration`, calibration tools, config, parameter, data loading |
| [Identification](identification.md) | `BaseIdentification`, identification tools, config, parameter |
| [Optimal](optimal.md) | `BaseOptimalCalibration`, `BaseOptimalTrajectory` |
| [Backends](backends.md) | `DynamicsBackend` interface + Pinocchio/MuJoCo implementations |
| [Integration](integration.md) | `RobotIdentificationSystem`, the one-line workflow API |
| [Measurements](measurements.md) | Measurement data structures |
| [Tools](tools.md) | Reporting/verification, provenance & run archiving, linear solver, robot management, regressor builder, visualization, collisions, QR decomposition, IPOPT wrapper, URDF export |
| [Utils](utils.md) | Config parser, cubic spline, results manager, error handling, legacy-to-unified config migration |
| [Visualisation](visualisation.md) | Plotting and viser-based visualization helpers |

New to the library? Start with [Concepts](../concepts/architecture.md) for
the architectural picture these modules fit into, or the
[Tutorials](../tutorials/index.md) for a task-driven walkthrough.
