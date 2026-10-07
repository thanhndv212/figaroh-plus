# FIGAROH Architecture

**Source audit: 2026-09-28.** This document describes the current source tree.
Future capabilities belong in the [roadmap](https://github.com/thanhndv212/figaroh-plus/blob/main/ROADMAP.md) and
[design decisions](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/README.md). The docs site embeds this file;
API details are generated from docstrings under `docs/source/api/`.

## System boundaries

FIGAROH is a Python library for dynamic identification, geometric calibration
and experiment design. The installable package is `src/figaroh/`; robot-specific
scripts, datasets and models live in the separate
[figaroh-examples repository](https://github.com/thanhndv212/figaroh-examples).

The principal user entry points are subclasses of `BaseIdentification`,
`BaseCalibration`, `BaseOptimalTrajectory` and `BaseOptimalCalibration`.
`RobotIdentificationSystem` provides a smaller CSV/URDF identification wrapper.
The source currently imports several subpackages eagerly, so installing only a
single numerical dependency is not sufficient to import the package.

```text
Robot-specific script + YAML + measurements + URDF
                     |
     Workflow classes / integration wrapper
                     |
  Regressors, solvers, robot model, result processing
                     |
     Backend-aware operations + direct Pinocchio use
                     |
       Identified/calibrated results
                     |
   Verification / reports / archives / explicit export
```

## Module responsibilities

| Module | Owns | Reference |
|---|---|---|
| `calibration/` | Config, measured poses, kinematic residuals, fitting and validation | [BaseCalibration](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/calibration/base_calibration.py) |
| `identification/` | Signal processing, dynamic regressors, identifiable parameters, optional physical projection/reconstruction | [BaseIdentification](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/identification/base_identification.py) |
| `optimal/` | Excitation trajectories and calibration posture selection | [trajectory workflow](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/optimal/base_optimal_trajectory.py), [configuration workflow](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/optimal/base_optimal_calibration.py) |
| `backends/` | Common dynamics operations and Pinocchio/MuJoCo implementations | [DynamicsBackend](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/backends/base.py) |
| `tools/` | Robot loading, QR/linear solvers, collisions, export, reports and archives | [Robot](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/tools/robot.py), [regressor](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/tools/regressor.py), [QR](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/tools/qrdecomposition.py) |
| `utils/` | Config inheritance/migration, result handling and error types | [config parser](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/utils/config_parser.py), [results](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/utils/results_manager.py) |
| `integration/` | Convenience wrapper around the identification workflow | [API](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/integration/api.py) |
| `measurements/`, `visualisation/` | Measurement and viewer utilities | Generated API reference |

Generic algorithms stay in core; robot paths, data conventions and experiment
recipes stay in examples. A change spanning both repos records the tested
revision pair and expected core dependency in the examples PR.

## Backend boundary

- [PinocchioBackend](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/backends/pinocchio.py) exists and implements
  dynamics/kinematics operations. `Robot.backend` lazily wraps the robot's current
  Pinocchio model and data through `PinocchioBackend.from_model()`.
- [MuJoCoBackend](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/backends/mujoco.py) exists. Its analytical regressor
  delegates to a Pinocchio model loaded from the model path; its Coriolis matrix
  uses a finite-difference construction. A MuJoCo operation being available does
  not imply the complete identification/calibration workflow is portable.
- Regressor construction and several calibration helpers use a backend when
  provided. Direct Pinocchio model/frame/inertia operations remain, especially
  model mutation during calibration and visualization/collision integration.
- Genesis and IsaacSim implementations are absent. They are roadmap proposals.
- `RobotIdentificationSystem.from_mjcf()` raises `NotImplementedError`.
  `from_urdf(backend=...)` loads the usual Robot and stores the requested name;
  that name alone does not switch the robot's executing backend. Treat the
  convenience identification path as Pinocchio/URDF-based until selection is
  implemented and verified.

Backend and cross-backend tests live in
[`test_backends.py`](https://github.com/thanhndv212/figaroh-plus/blob/main/tests/unit/test_backends.py) and
[`test_cross_backend.py`](https://github.com/thanhndv212/figaroh-plus/blob/main/tests/integration/test_cross_backend.py).
Optional dependency skips are reported as missing coverage.

## Identification data flow

1. Load the robot and YAML configuration. A subclass supplies trajectory data;
   the convenience wrapper reads position, velocity, acceleration and torque CSVs.
2. Process signals and build configurations, then construct the dynamic regressor
   and the nominal CAD reference torque.
3. Eliminate zero columns, optionally decimate, and use QR to obtain an
   identifiable parameter basis. Solve and compute fitting metrics.
4. Apply configured weighting, physical-consistency projection and reconstruction
   where enabled. These are distinct result stages.
5. Evaluate validation data when available, generate verification/report output,
   and explicitly choose the result stage to export or archive.

Important contracts:

- Joint order, units and the provenance of torque measurements belong to the data
  boundary. The convenience CSV loader creates sample-index timestamps; it is
  not a universal timestamped log-ingestion API.
- Regressor rows are joint-major. Enabled friction, actuator-inertia and offset
  blocks must match parameter key order; see the
  [identification regressions](https://github.com/thanhndv212/figaroh-plus/blob/main/tests/unit/test_identification_regressions.py).
- Pinocchio's dynamic parameter order is
  `[m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`; FIGAROH's standard ordering
  differs. Use the conversion helpers in
  [`identification/parameter.py`](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/identification/parameter.py).
- Projection is default-off. `result["physical consistency"]` records status,
  feasibility information, and, after projection, `raw_parameters` and
  `projected_parameters`. Reconstruction has its own `result["reconstruction"]`.
  Callers must not equate a fitted base vector with a complete physical model.
- Unified signal-processing settings feed the filter parameters; explicit filter
  overrides take precedence. `qr_relative_tolerance` is optional, preserving the
  default rank-selection behavior when unset.

## Calibration and experiment design

Calibration fits geometric parameters through the workflow and helpers in
`calibration/`. Model edits still use Pinocchio types. A held-out measurement
comparison assesses fit quality; an exported-model FK consistency check assesses
whether a file represents the intended calibrated model. They answer different
questions and are reported separately.

Optimal trajectory generation uses the IPOPT wrapper in
[`tools/robotipopt.py`](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/tools/robotipopt.py). Calibration posture
selection lives in `BaseOptimalCalibration`. Both provide result saving;
a common report format across every optimal workflow remains roadmap work.
Use the `figaroh-dev` conda environment for development, including `cyipopt`.
Physical projection is opt-in at runtime, although `picos` is currently a
package dependency in `pyproject.toml`.

## Export and reporting

[`tools/urdf_exporter.py`](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/tools/urdf_exporter.py) maps parameter names
to update handlers. Joint placement/offset, mass, friction, armature and elasticity
handlers exist. Identified inertial parameters are written as complete per-link
sets (#60): inputs follow Pinocchio `toDynamicParameters()` order, with first
moments and inertia about the link-frame origin. The exporter writes the CoM
origin and the CoM-frame tensor, keeping an existing inertial rotation. A link
target or a moving joint (resolved to its child link) is accepted. Targets
with massive links attached by fixed joints (merged by Pinocchio), partial
sets and `m <= 0` are refused; physically infeasible sets are refused unless
`allow_infeasible=True`. Tests check
reloaded-model RNEA parity against known feasible inputs; round-trip parity
shows file consistency, not hardware accuracy.

Metrology base/measurement-frame parameters are described for the caller instead
of automatically applied to robot geometry. PAL runtime correction YAML uses
[`geometric_calibration_export.py`](https://github.com/thanhndv212/figaroh-plus/blob/main/src/figaroh/tools/geometric_calibration_export.py).

Calibration and identification HTML reports, two-run comparison, provenance and
run archives live in `tools/`; workflow verification methods attach verdicts to
results. Example CLIs explicitly verify numerical execution; library verification defaults
to prediction acceptance. Independent prediction
acceptance requires explicit per-output error limits and separate validation.
Physical/export and general acquisition-provenance certification remain separate;
missing required evidence is not a successful check. Use the [reporting and verification guide](docs/source/reporting_and_verification.md)
for user-facing behavior. Store metrics with the model/config/data provenance;
report whether validation is held-out, training-only, simulated or physical.

## Configuration and extension rules

`UnifiedConfigParser` supports YAML inheritance. The workflow-specific config
adapters translate unified task config into internal dictionaries; legacy config
remains supported with deprecation warnings and a migration command. The
[configuration reference](docs/source/concepts/configuration.md) owns the schema.

For a new interface, dependency, result schema or module boundary, add a decision
record first or in the same PR. Proposed `ResidualTerm`/`LinearRegressorTerm`,
full MJCF workflows and deployment/control layers are not current API contracts.
For a new backend, implement the supported operations, document unsupported ones,
and add numerical comparisons before claiming workflow parity.

The [contribution workflow](https://github.com/thanhndv212/figaroh-plus/blob/main/CONTRIBUTING.md) owns branch/release policy; the
[validation guide](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/development/validation.md) owns commands and evidence
requirements. Keep numerical claims linked to tests or archived runs and update
this overview when boundaries or result contracts change.
