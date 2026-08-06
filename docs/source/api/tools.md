# Tools

## Reporting & Verification

The [reporting & verification suite](../reporting_and_verification.md)'s
HTML report generators and the static two-run compare page.

### report

::: figaroh.tools.report
    options:
      show_root_heading: false

### identification_report

::: figaroh.tools.identification_report
    options:
      show_root_heading: false

### compare_report

::: figaroh.tools.compare_report
    options:
      show_root_heading: false

### _report_common

Shared HTML/CSS primitives and the `VerificationVerdict`/`ThresholdCheck`
dataclasses used by both report generators.

::: figaroh.tools._report_common
    options:
      show_root_heading: false

### provenance

Run provenance metadata — git commit, config hash, timestamps, robot/asset
identity — attached to every calibration/identification run.

::: figaroh.tools.provenance
    options:
      show_root_heading: false

### run_archive

Writes each run to a timestamped `results/runs/<robot>/<task>/<timestamp>/`
directory (config snapshot, provenance JSON, report) instead of overwriting
a single `results/` path.

::: figaroh.tools.run_archive
    options:
      show_root_heading: false

## Linear Solver

::: figaroh.tools.solver
    options:
      show_root_heading: false

The linear solver provides comprehensive solving methods for robot
parameter identification:

- **Basic methods**: lstsq, QR decomposition, SVD
- **Regularized methods**: Ridge (L2), Lasso (L1), Elastic Net, Tikhonov
- **Advanced methods**: Constrained optimization, robust regression,
  weighted least squares
- **Constraint support**: Box constraints, linear equality/inequality
  constraints
- **Quality metrics**: RMSE, R², condition number, residual analysis

## Robot Management

::: figaroh.tools.robot
    options:
      show_root_heading: false

## Regressor Builder

::: figaroh.tools.regressor
    options:
      show_root_heading: false

## Robot Visualization

::: figaroh.tools.robotvisualization
    options:
      show_root_heading: false

## Robot Collisions

::: figaroh.tools.robotcollisions
    options:
      show_root_heading: false

## QR Decomposition

::: figaroh.tools.qrdecomposition
    options:
      show_root_heading: false

## Robot IPOPT

::: figaroh.tools.robotipopt
    options:
      show_root_heading: false

## Random Data Generation

::: figaroh.tools.randomdata
    options:
      show_root_heading: false

## URDF Export

Applies calibrated joint parameters back onto a URDF file.

::: figaroh.tools.urdf_exporter
    options:
      show_root_heading: false

## Export Validation

FK consistency checks and interactive viser visualization for a
calibration-exported URDF.

::: figaroh.tools.export_validation
    options:
      show_root_heading: false
