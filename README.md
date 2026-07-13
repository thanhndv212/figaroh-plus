# FIGAROH
**F**ree dynamics **I**dentification and **G**eometrical c**A**libration of **RO**bot and **H**uman

<p align="center">
  <a href="https://www.apache.org/licenses/LICENSE-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License"/></a>
  <a href="https://pypi.org/project/figaroh/"><img src="https://badge.fury.io/py/figaroh.svg" alt="PyPI version" height="20"/></a>
  <a href="https://thanhndv212.github.io/figaroh-plus/"><img src="https://img.shields.io/badge/docs-online-brightgreen" alt="Documentation"/></a>
  <a href="https://deepwiki.com/thanhndv212/figaroh-plus"><img src="https://deepwiki.com/badge.svg" alt="Ask DeepWiki"/></a>
</p>

FIGAROH is a Python toolbox providing efficient and highly flexible frameworks for dynamics identification and geometric calibration of rigid multi-body systems based on the URDF modeling convention. It supports both serial (industrial manipulators) and tree-structure systems (humanoids, mobile manipulators).

**📦 Available on PyPI:** `pip install figaroh`
**📖 Version:** 0.4.5

> Note: This repo is a fork from [gitlab repo](https://gitlab.laas.fr/gepetto/figaroh) of which the author is no longer a contributor.

---

## Installation

### Quick Installation (Recommended)

Install the core FIGAROH package with all dependencies (except for cyipopt):

```bash
pip install figaroh
```

### Development Installation

For development or local installation from source, choose one of these methods:

**Method 1: Direct pip installation (Simple)**
```bash
git clone https://github.com/thanhndv212/figaroh-plus.git
cd figaroh
pip install -e .
```

**Method 2: Conda environment (Recommended for the use of cyipopt)**
```bash
git clone https://github.com/thanhndv212/figaroh-plus.git
cd figaroh
# Create conda environment with optimization libraries
conda env create -f environment.yml
conda activate figaroh-dev
```

### Examples Repository
```bash
git clone https://github.com/thanhndv212/figaroh-examples.git
cd figaroh-examples && pip install -r requirements.txt
```

| Robot | Tasks |
|-------|-------|
| **Staubli TX40** | Dynamic identification |
| **Universal UR10** | Geometric calibration (RealSense camera) |
| **TIAGo** | Full workflow: identification + calibration |
| **TALOS Humanoid** | Torso-arm calibration, whole-body calibration (to be released) |

---

## Key Features

### 🔧 Dynamic Identification
- Extended dynamic models: friction, actuator inertia, joint offsets
- Optimal exciting trajectory generation (IPOPT)
- Multiple parameter estimation algorithms
- Physically consistent parameters for URDF updates

### 📐 Geometric Calibration
- Full kinematic parameter estimation (6 DOF per joint)
- Optimal posture selection via combinatorial optimization
- Support for cameras, motion capture, planar constraints
- Direct URDF model updates

### ⚙️ Configuration System
- **Unified YAML format** with template inheritance
- **Automatic format detection** (legacy compatibility)
- **Variable expansion** and validation
- **Task-specific configs**: calibration, identification, optimal trajectory

### 🛠️ Modern Architecture
- **Proper logging** (NullHandler pattern for libraries)
- **Abstract base classes** for extensibility
- **Pinocchio 3.x compatibility**
- **Cross-platform**: Linux, macOS, Windows

### 📊 Reporting & Verification (V&V)
- **Self-contained HTML diagnostic reports** with an interactive before/after
  chart — `solve(html_report=True)` / `export_html_report()`
- **Machine-readable pass/fail verdicts** for CI — `verify()` /
  `export_verification_report()`, with overridable quality thresholds
- **Static two-run compare page** — `generate_compare_page()` diffs two
  exported runs offline, with a mandatory compatibility check before
  overlaying them
- See the [Reporting & Verification guide](https://thanhndv212.github.io/figaroh-plus/guides/reporting_and_verification/)
  for the full walkthrough
---

## Core Modules (See more at [ARCHITECTURE](ARCHITECTURE))

### `figaroh.calibration` — Geometric Calibration

**BaseCalibration** provides a complete framework for kinematic parameter calibration:

- **Automatic parameter identification** using QR decomposition
- **Robust optimization** with iterative outlier removal (Levenberg-Marquardt)
- **Unit-aware weighting** for position/orientation measurements
- **Multiple calibration models**: full kinematic parameters, joint offsets
- **Sensor support**: cameras, motion capture, planar constraints

### `figaroh.identification` — Dynamic Identification

**BaseIdentification** implements the complete dynamic parameter identification workflow:

- **Standard + extended parameters**: inertial parameters, friction (viscous/Coulomb), actuator inertia, joint offsets
- **Regressor-based identification** with base parameter reduction
- **Multiple solvers**: Least Squares, Weighted LS, Ridge, Lasso, Elastic Net
- **Decimation and filtering** for signal processing
- **Quality metrics**: RMSE, correlation, condition number

#### Physical Consistency (optional, default-off)

FIGAROH can optionally project per-joint inertial parameters onto a physically
consistent set using a convex SDP/LMI based on Pinocchio pseudo-inertia.

- Enable it in config via `identification.physical_consistency.enabled: true`.
- Requires optional dependencies: `picos` and an SDP solver backend (e.g. `cvxopt`).

### `figaroh.optimal` — Trajectory & Configuration Optimization

**BaseOptimalTrajectory** generates exciting trajectories for dynamic identification:

- **IPOPT-based nonlinear optimization** with cyipopt
- **Cubic spline parameterization** for C² continuous trajectories
- **Constraint handling**: joint limits, velocity limits, torque limits, self-collision
- **Cost functions**: condition number minimization, excitation maximization

**BaseOptimalCalibration** selects optimal calibration configurations:

- **Combinatorial optimization** from feasible posture pool
- **Observability-based selection** for maximum information gain

### `figaroh.tools` — Robotics Utilities

| Class | Description |
|-------|-------------|
| `RegressorBuilder` | Object-oriented regressor computation with configurable parameters |
| `LinearSolver` | Advanced solver supporting 10+ methods (lstsq, QR, SVD, Ridge, Lasso, etc.) |
| `QRDecomposer` | QR decomposition with column pivoting for base parameter identification |
| `CollisionManager` | Pinocchio-based collision detection with visualization |
| `RobotIPOPTSolver` | High-level IPOPT interface with automatic differentiation |
| `generate_calibration_report` / `generate_identification_report` | Self-contained HTML diagnostic reports (`tools/report.py`, `tools/identification_report.py`) |
| `generate_compare_page` | Static, offline two-run compare page (`tools/compare_report.py`) |

### `figaroh.utils` — Configuration & Results

| Class | Description |
|-------|-------------|
| `UnifiedConfigParser` | YAML parsing with template inheritance and variable expansion |
| `ResultsManager` | Unified plotting for calibration/identification results |
| `CubicSpline` | C² continuous spline trajectory generation |

---

## Methodology

FIGAROH implements a systematic workflow for robot calibration and identification:

### Step 1: Configuration Setup
Define robot parameters, sensor configurations, and task-specific settings in YAML:

```yaml
# config/robot_config.yaml
robot:
  name: "my_robot"
  urdf_path: "models/robot.urdf"

calibration:
  start_frame: "base_link"
  end_frame: "tool0"
  method: "full_params"

identification:
  has_friction: true
  has_actuator_inertia: true
  active_joints: ["joint1", "joint2", "joint3"]

  physical_consistency:
    enabled: false
    solver: "cvxopt"
    mass_min: 1e-6
    psd_eig_tol: -1e-10
    skip_if_feasible: true
```

### Step 2: Optimal Experiment Design
Generate exciting trajectories or calibration postures:

- **For identification**: Solve IPOPT optimization to find trajectories maximizing regressor condition
- **For calibration**: Combinatorial selection of postures maximizing observability

### Step 3: Data Collection & Processing
`initialize()` loads and validates experimental data (paths come from the
YAML config, e.g. `measurement_file`):

```python
calibrator = MyCalibration(robot, "config/robot_config.yaml")
calibrator.initialize()
```

### Step 4: Parameter Estimation
Run identification/calibration, then get a quality report, an optional
HTML report, and a pass/fail verdict for the same run:

```python
# Calibration
calibrator.solve(html_report=True)      # prints + writes results/calibration_report.html
print(f"RMSE: {calibrator.evaluation_metrics['rmse']:.6f}")
verdict = calibrator.verify()           # pass/fail against quality thresholds

# Identification
identifier.solve(decimate=True, decimation_factor=10, html_report=True)
print(f"Correlation: {identifier.correlation:.4f}")
verdict = identifier.verify()
identifier.export_verification_report()  # results/identification_verification.json
```

### Step 5: Model Update
Export calibrated/identified parameters to URDF or YAML.

---

## Dependencies

| Category | Packages |
|----------|----------|
| **Scientific** | numpy, scipy, matplotlib, pandas, numdifftools |
| **Robotics** | pinocchio (pin), ndcurves, meshcat |
| **Config** | pyyaml, rospkg |
| **Optimization** | cyipopt (conda), picos |

---

## Citations

If you use FIGAROH in your research, please cite the following papers:

### Main Reference
```bibtex
@inproceedings{nguyen2023figaroh,
  title={FIGAROH: a Python toolbox for dynamic identification and geometric calibration of robots and humans},
  author={Nguyen, Dinh Vinh Thanh and Bonnet, Vincent and Maxime, Sabbah and Gautier, Maxime and Fernbach, Pierre and others},
  booktitle={IEEE-RAS International Conference on Humanoid Robots},
  pages={1--8},
  year={2023},
  address={Austin, TX, United States},
  doi={10.1109/Humanoids57100.2023.10375232},
  url={https://hal.science/hal-04234676v2}
}
```

### Related Work
```bibtex
@inproceedings{nguyen2024improving,
  title={Improving Operational Accuracy of a Mobile Manipulator by Modeling Geometric and Non-Geometric Parameters},
  author={Nguyen, Thanh D. V. and Bonnet, V. and Fernbach, P. and Flayols, T. and Lamiraux, F.},
  booktitle={2024 IEEE-RAS 23rd International Conference on Humanoid Robots (Humanoids)},
  pages={965--972},
  year={2024},
  address={Nancy, France},
  doi={10.1109/Humanoids58906.2024.10769790}
}

@techreport{nguyen2025humanoid,
  title={Humanoid Robot Whole-body Geometric Calibration with Embedded Sensors and a Single Plane},
  author={Nguyen, Thanh D V and Bonnet, Vincent and Fernbach, Pierre and Daney, David and Lamiraux, Florent},
  year={2025},
  institution={HAL},
  url={https://hal.science/hal-05169055}
}
```

## License

Please refer to the [LICENSE](LICENSE) file for licensing information.
