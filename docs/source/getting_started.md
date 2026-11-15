# Getting Started

## Installation

Install FIGAROH from PyPI:

```bash
pip install figaroh
```

Optional parts are installed as extras:

| Extra | Adds | Needed for |
|---|---|---|
| `figaroh[sklearn]` | scikit-learn | `LinearSolver` methods `lasso` and `elastic_net` |
| `figaroh[mujoco]` | MuJoCo 3.x | the MuJoCo backend (4.x is untested) |

IPOPT (`cyipopt`), which optimal trajectory generation needs, has no extra:
PyPI carries no wheels for it. Install it from conda-forge, as the
development environment below does.

For development with all dependencies:

```bash
conda env create -f environment.yml
conda activate figaroh-dev
pip install -e .
```

FIGAROH's configuration system (the unified YAML format, template
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

# Convenience function: returns the solution and the solver's metrics
x, info = solve_linear_system(A, b, method='ridge', alpha=0.01)

# Constrained least squares: bounds and linear constraints
solver = LinearSolver(
    method='constrained',
    bounds=[(0, None)] * A.shape[1],  # one (lower, upper) pair per parameter
    constraints={'A_eq': A_eq, 'b_eq': b_eq},
)
x = solver.solve(A, b)  # raises RuntimeError if the solve does not converge

# Quality metrics
info = solver.solver_info
print(f"RMSE: {info['rmse']:.4f}")
print(f"R²: {info['r_squared']:.4f}")
print(f"Condition number: {info['condition_number']:.2e}")

# Use in identification workflow
params = identifier.solve_with_custom_solver(
    method='elastic_net',
    alpha=0.01,
    l1_ratio=0.5,
)
```

Each method reads only its own options: `bounds` and linear `constraints`
belong to `constrained`, `alpha` to `ridge`, `lasso`, `elastic_net` and
`tikhonov`, `weights` to `weighted`. An option the selected method does not
use raises `ValueError` instead of being ignored.

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
