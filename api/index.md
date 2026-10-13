# API Reference

Generated from docstrings in `src/figaroh/` via [mkdocstrings](https://mkdocstrings.github.io/). One page per top-level subpackage:

| Module                                                                                   | Contains                                                                                                                                                                        |
| ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [Calibration](https://thanhndv212.github.io/figaroh-plus/api/calibration/index.md)       | `BaseCalibration`, calibration tools, config, parameter, data loading                                                                                                           |
| [Identification](https://thanhndv212.github.io/figaroh-plus/api/identification/index.md) | `BaseIdentification`, identification tools, config, parameter                                                                                                                   |
| [Optimal](https://thanhndv212.github.io/figaroh-plus/api/optimal/index.md)               | `BaseOptimalCalibration`, `BaseOptimalTrajectory`                                                                                                                               |
| [Backends](https://thanhndv212.github.io/figaroh-plus/api/backends/index.md)             | `DynamicsBackend` interface + Pinocchio/MuJoCo implementations                                                                                                                  |
| [Integration](https://thanhndv212.github.io/figaroh-plus/api/integration/index.md)       | `RobotIdentificationSystem`, the one-line workflow API                                                                                                                          |
| [Data contract](https://thanhndv212.github.io/figaroh-plus/api/data/index.md)            | `TrajectoryData`, `PoseObservations`, `DataSource`, `Protocol` (#55)                                                                                                            |
| [Measurements](https://thanhndv212.github.io/figaroh-plus/api/measurements/index.md)     | Measurement data structures                                                                                                                                                     |
| [Tools](https://thanhndv212.github.io/figaroh-plus/api/tools/index.md)                   | Reporting/verification, provenance & run archiving, linear solver, robot management, regressor builder, visualization, collisions, QR decomposition, IPOPT wrapper, URDF export |
| [Utils](https://thanhndv212.github.io/figaroh-plus/api/utils/index.md)                   | Config parser, cubic spline, results manager, error handling, legacy-to-unified config migration                                                                                |
| [Visualisation](https://thanhndv212.github.io/figaroh-plus/api/visualisation/index.md)   | Plotting and viser-based visualization helpers                                                                                                                                  |

The supported names, experimental areas and the deprecation policy are on [Public API and stability](https://thanhndv212.github.io/figaroh-plus/api/stability/index.md).

New to the library? Start with [Concepts](https://thanhndv212.github.io/figaroh-plus/concepts/architecture/index.md) for the architectural picture these modules fit into, or the [Tutorials](https://thanhndv212.github.io/figaroh-plus/tutorials/index.md) for a task-driven walkthrough.
