# Getting Started

## Installation

Install FIGAROH from PyPI:

```bash
pip install figaroh
```

For development with all dependencies:

```bash
conda env create -f environment.yml
conda activate figaroh-dev
pip install -e .
```

## Configuration System

FIGAROH uses a flexible YAML-based configuration system that supports both
a modern unified format and a legacy format.

### Unified Configuration Format

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

### Legacy Format Support

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

## Quick Start Examples

`BaseCalibration` and `BaseIdentification` are abstract base classes — each
robot example provides a concrete subclass (e.g. `TiagoCalibration`,
`UR10Identification`) that implements the robot-specific cost function. The
[examples repository](https://github.com/thanhndv212/figaroh-examples) has a
complete, runnable subclass per supported robot; the snippets below show the
shape every subclass follows.

### Basic Calibration

```python
from examples.tiago.utils.tiago_tools import TiagoCalibration
from figaroh.tools.robot import load_robot

robot = load_robot("path/to/robot.urdf", load_by_urdf=True)
calibrator = TiagoCalibration(robot, "config/calibration_config.yaml")
calibrator.initialize()
result = calibrator.solve(plotting=False, html_report=True)
```

`solve(html_report=True)` also writes a self-contained HTML diagnostic
report alongside the terminal quality report that's always printed — see
[Reporting & Verification](guides/reporting_and_verification.md) for the
full report/verdict/compare-page suite.

### Basic Identification

```python
from examples.tiago.utils.tiago_tools import TiagoIdentification
from figaroh.tools.robot import load_robot

robot = load_robot("path/to/robot.urdf", load_by_urdf=True)
identifier = TiagoIdentification(robot, "config/identification_config.yaml")
identifier.initialize()
result = identifier.solve(decimate=True, html_report=True)

verdict = identifier.verify()
print("PASS" if verdict.passed else "FAIL")
```

### Advanced Regressor Building

```python
from figaroh.tools.regressor import RegressorBuilder, RegressorConfig

# Configure regressor
config = RegressorConfig(
    has_friction=True,
    has_actuator_inertia=True,
    is_joint_torques=True
)

# Build regressor matrix
builder = RegressorBuilder(robot, config)
W = builder.build_basic_regressor(q, dq, ddq)
```

### Configuration Management

```python
from figaroh.utils.config_parser import UnifiedConfigParser

# Parse any configuration format
parser = UnifiedConfigParser("config/robot_config.yaml")
config = parser.parse()

# Create task-specific configuration
calib_config = parser.create_task_config(robot, config, "calibration")
identif_config = parser.create_task_config(robot, config, "identification")
```

### Advanced Linear Solver

```python
from figaroh.tools.solver import LinearSolver, solve_linear_system

# Basic usage with convenience function
result = solve_linear_system(
    A, b,
    method='ridge',
    alpha=0.01
)
x = result['solution']

# Advanced usage with constraints
solver = LinearSolver()
result = solver.solve(
    A, b,
    method='constrained',
    bounds=(0, None),  # Positive constraints
    A_eq=A_eq, b_eq=b_eq,  # Equality constraints
    alpha=0.1  # Regularization
)

# Access quality metrics
print(f"RMSE: {result['rmse']:.4f}")
print(f"R²: {result['r_squared']:.4f}")
print(f"Condition number: {result['condition_number']:.2e}")

# Use in identification workflow
params = identifier.solve_with_custom_solver(
    method='elastic_net',
    alpha=0.01,
    l1_ratio=0.5,
    bounds=(0, None)  # Physical constraints
)
```

## Next Steps

- Read [Reporting & Verification](guides/reporting_and_verification.md) to
  turn a solved calibration/identification into a shareable report and a
  CI-gateable pass/fail check.
- Explore the [Examples Repository](https://github.com/thanhndv212/figaroh-examples)
  for complete, runnable workflows per robot.
- Check the Core Modules / Tools & Utilities API reference for detailed
  module information.
- Review the configuration templates for your specific robot type.
