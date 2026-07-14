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

FIGAROH's configuration system (unified vs. legacy YAML format, template
inheritance) is covered separately in
[Configuration System](concepts/configuration.md) — read that before writing
your first robot config.

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
[Reporting & Verification](reporting_and_verification.md) for the
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

- Follow the [Tutorials](tutorials/index.md) for an end-to-end walkthrough of
  calibration, identification, and optimal experiment design.
- Read [Reporting & Verification](reporting_and_verification.md) to
  turn a solved calibration/identification into a shareable report and a
  CI-gateable pass/fail check.
- Browse the [Examples Gallery](examples/index.md) for complete, runnable
  workflows per robot (UR10, TIAGo, TALOS, Staubli TX40).
- Check the API Reference for detailed module information.
- Set up your own robot config from a [template](examples/templates.md).
