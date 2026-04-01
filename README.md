# FIGAROH
**F**ree dynamics **I**dentification and **G**eometrical c**A**libration of **RO**bot and **H**uman

FIGAROH is a Python toolbox providing efficient and highly flexible frameworks for dynamics identification and geometric calibration of rigid multi-body systems based on the URDF modeling convention. It supports both serial (industrial manipulators) and tree-structure systems (humanoids, mobile manipulators).

**📦 Available on PyPI:** `pip install figaroh`  
**📖 Version:** 0.3.1

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

---

## Package Structure

```
figaroh/
├── calibration/          # Geometric calibration framework
│   ├── BaseCalibration       # Abstract base class for kinematic calibration
│   ├── calibration_tools     # Parameter parsing, regressor computation
│   ├── config                # Configuration loading and validation
│   ├── data_loader           # CSV data loading utilities
│   └── parameter             # Kinematic parameter management
│
├── identification/       # Dynamic parameter identification
│   ├── BaseIdentification    # Abstract base class for dynamic identification
│   ├── identification_tools  # Regressor utilities, parameter extraction
│   ├── config                # Identification configuration parsing
│   └── parameter             # Inertial parameter management (friction, inertia)
│
├── optimal/              # Optimization-based trajectory & configuration
│   ├── BaseOptimalTrajectory     # IPOPT-based trajectory optimization
│   ├── BaseOptimalCalibration    # Optimal calibration posture selection
│   ├── BaseParameterComputer     # Base parameter computation utilities
│   ├── TrajectoryConstraintManager # Constraint handling for optimization
│   └── config                    # Optimization configuration management
│
├── tools/                # Core robotics utilities
│   ├── RegressorBuilder      # Object-oriented regressor computation
│   ├── LinearSolver          # Advanced linear solver (LS, Ridge, Lasso, etc.)
│   ├── QRDecomposer          # QR decomposition for base parameters
│   ├── CollisionManager      # Collision detection and visualization
│   ├── RobotIPOPTSolver      # IPOPT optimization wrapper
│   └── CubicSpline           # Trajectory interpolation utilities
│
├── utils/                # Helper utilities
│   ├── UnifiedConfigParser   # YAML config with inheritance support
│   ├── ResultsManager        # Unified plotting and result export
│   ├── error_handling        # Custom exceptions and validation
│   └── cubic_spline          # Spline trajectory generation
│
├── measurements/         # Data acquisition and processing
└── visualisation/        # Meshcat-based 3D visualization
```

---

## Core Modules

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

### `figaroh.utils` — Configuration & Results

| Class | Description |
|-------|-------------|
| `UnifiedConfigParser` | YAML parsing with template inheritance and variable expansion |
| `ResultsManager` | Unified plotting for calibration/identification results |
| `CubicSpline` | C² continuous spline trajectory generation |

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
Load experimental data with automatic validation:

```python
from figaroh.calibration import BaseCalibration

calibrator = MyCalibration(robot, "config/robot_config.yaml")
calibrator.load_data("data/measurements.csv")
```

### Step 4: Parameter Estimation
Run identification/calibration with quality metrics:

```python
# Calibration
calibrator.solve()
print(f"RMSE: {calibrator.evaluation_metrics['rmse']:.6f}")

# Identification  
identifier.solve(decimate=True, decimation_factor=10)
print(f"Correlation: {identifier.correlation:.4f}")
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

## Examples

Complete working examples are available in [figaroh-examples](https://github.com/thanhndv212/figaroh-examples):

| Robot | Tasks |
|-------|-------|
| **Staubli TX40** | Dynamic identification |
| **Universal UR10** | Geometric calibration (RealSense camera) |
| **TIAGo** | Full workflow: identification + calibration |
| **TALOS Humanoid** | Torso-arm calibration, whole-body calibration |

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
