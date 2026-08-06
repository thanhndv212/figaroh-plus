# FIGAROH Architecture Documentation

**Version:** 3.2
**Date:** July 12, 2026
**Status:** v0.4.4 released — calibration validation, quality report, URDF export, interactive
visualization, plus the reporting & verification (V&V) suite (HTML diagnostic reports,
machine-readable pass/fail verdicts, before/after panel, static two-run compare page —
see [§4.6](#4-module-details))

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Layered Architecture](#layered-architecture)
4. [Module Details](#module-details)
5. [Pinocchio Integration Map](#pinocchio-integration-map)
6. [Data Flow](#data-flow)
7. [Backend Architecture](#backend-architecture)
8. [Extension Points](#extension-points)
9. [Performance Considerations](#performance-considerations)

---

## 1. Executive Summary

FIGAROH (**F**ree dynamics **I**dentification and **G**eometrical c**A**libration of **RO**bot and **H**uman) is a modular framework for robot calibration and parameter identification. The architecture follows a **three-layer design**:

1. **Backend Layer** - Pluggable dynamics computation (Pinocchio, MuJoCo, Genesis, Isaac Sim)
2. **Tools Layer** - Core algorithms (regressor computation, solvers, parameter handling)
3. **Workflow Layer** - High-level orchestration (calibration, identification, optimal trajectory)

### Key Design Principles
- **Simulator Agnostic**: Backend abstraction enables seamless simulator switching
- **Algorithm Consistency**: Same identification/calibration algorithms across all backends
- **Modular & Extensible**: Clear separation of concerns enables easy extension
- **Production Ready**: Comprehensive error handling, validation, and documentation

### Package Structure

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

## 2. System Overview

> **📝 Note**: This document contains Mermaid diagrams. If they don't render:
> 1. Install the "Markdown Preview Mermaid Support" extension in VS Code
> 2. View the file on GitHub (renders Mermaid natively)
> 3. Use the online viewer: https://mermaid.live
>
> Or see the ASCII version below.

### ASCII Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        FIGAROH ARCHITECTURE                      │
└─────────────────────────────────────────────────────────────────┘

                    User Scripts/Examples
                    YAML Configuration
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │         WORKFLOW LAYER                    │
        │  BaseCalibration   BaseIdentification     │
        │  BaseOptimalTrajectory  BaseOptimalCalib  │
        └───────────────────┬───────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │           TOOLS LAYER                     │
        │  Robot  RegressorBuilder  LinearSolver    │
        │  QRDecomposer  CollisionManager           │
        │  URDFExporter  URDFComparison             │
        │  ExportValidation  RobotVisualization     │
        └───────────────────┬───────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────────┐
        │         BACKEND LAYER                     │
        │    DynamicsBackend (Abstract)             │
        │      │       │        │         │         │
        │  Pinocchio MuJoCo Genesis IsaacSim        │
        └───────────────────┬───────────────────────┘
                            │
                            ▼
                    URDF/MJCF/USD Files
                    Measurement Data
                    Parameter Files
```

**Key Relationships:**
- Workflow → Tools: Use core algorithms
- Tools → Backend: Delegate dynamics computation
- Backend → Data: Load robot models and measurements

### Detailed Mermaid Diagram

```mermaid
graph TB
    subgraph UserInterface["User Interface"]
        UI[User Scripts/Examples]
        Config[YAML Configuration]
    end

    subgraph WorkflowLayer["Workflow Layer"]
        BC[BaseCalibration]
        BI[BaseIdentification]
        BOT[BaseOptimalTrajectory]
        BOC[BaseOptimalCalibration]
    end

    subgraph ToolsLayer["Tools Layer"]
        Robot[Robot Model]
        Regressor[RegressorBuilder]
        Solver[LinearSolver/IPOPTSolver]
        QR[QRDecomposer]
        CM[CollisionManager]
        RM[ResultsManager]
    end

    subgraph BackendLayer["Backend Layer"]
        Backend[DynamicsBackend Abstract Interface]
        Pin[PinocchioBackend]
        Muj[MuJoCoBackend]
        Gen[GenesisBackend]
        Isaac[IsaacSimBackend]
    end

    subgraph DataLayer["Data Layer"]
        URDF[URDF Files]
        Meas[Measurement Data]
        Params[Parameter Files]
    end

    UI -->|configure| Config
    Config -->|parse| BC
    Config -->|parse| BI
    Config -->|parse| BOT
    Config -->|parse| BOC

    BC -->|use| Robot
    BC -->|use| Regressor
    BC -->|use| Solver
    BI -->|use| Robot
    BI -->|use| Regressor
    BI -->|use| Solver
    BI -->|use| QR
    BOT -->|use| Robot
    BOT -->|use| Solver
    BOC -->|use| Robot
    BOC -->|use| Solver

    Robot -->|delegates to| Backend
    Regressor -->|requests from| Backend

    Backend -.->|implements| Pin
    Backend -.->|implements| Muj
    Backend -.->|implements| Gen
    Backend -.->|implements| Isaac

    Pin -->|load| URDF
    Muj -->|load| URDF
    Gen -->|load| URDF
    Isaac -->|load| URDF

    BC -->|load| Meas
    BI -->|load| Meas
    BC -->|load| Params
    BI -->|load| Params

    BC -->|save| RM
    BI -->|save| RM
    BOT -->|save| RM
    BOC -->|save| RM

    style Backend fill:#ffeb3b,stroke:#333,stroke-width:3px
    style BC fill:#4caf50,stroke:#333,stroke-width:2px
    style BI fill:#4caf50,stroke:#333,stroke-width:2px
    style BOT fill:#4caf50,stroke:#333,stroke-width:2px
    style BOC fill:#4caf50,stroke:#333,stroke-width:2px
```

---

## 3. Layered Architecture

### Layer 1: Backend Layer (`src/figaroh/backends/`)

**Purpose:** Abstract dynamics computation across simulators

**Components:**
- `base.py` - `DynamicsBackend` abstract interface
- `pinocchio.py` - Pinocchio implementation (default)
- `mujoco.py` - MuJoCo implementation (high-performance)
- `genesis.py` - Genesis implementation (GPU-accelerated) *[Future]*
- `isaacsim.py` - Isaac Sim implementation (sim-to-real) *[Future]*

**Key Interface Methods:**
```python
class DynamicsBackend(ABC):
    @abstractmethod
    def compute_mass_matrix(q: ndarray) -> ndarray
    @abstractmethod
    def compute_coriolis_matrix(q: ndarray, v: ndarray) -> ndarray
    @abstractmethod
    def compute_gravity_vector(q: ndarray) -> ndarray
    @abstractmethod
    def compute_forward_kinematics(q: ndarray) -> dict
    @abstractmethod
    def compute_jacobian(q: ndarray, frame: str) -> ndarray
    @abstractmethod
    def compute_regressor(q: ndarray, v: ndarray, a: ndarray) -> ndarray
```

### Layer 2: Tools Layer (`src/figaroh/tools/`)

**Purpose:** Core algorithms independent of backend

**Components:**
- `robot.py` - Enhanced Robot wrapper with free-flyer support
- `regressor.py` - Regressor matrix computation
- `solver.py` - Linear/nonlinear solvers with regularization
- `qr_decomposer.py` - QR decomposition for base parameters
- `robotcollisions.py` - Collision detection and management
- `load_robot.py` - Multi-backend robot loading
- `urdf_exporter.py` - Apply calibrated parameters to URDF files
- `export_validation.py` - FK comparison and interactive viser visualization

**Key Classes:**
```python
class RegressorBuilder:
    """Build observation regressor W(q,v,a) for identification"""
    def build_basic_regressor(q, v, a) -> ndarray
    def build_reduced_regressor(W, zero_tolerance) -> ndarray

class LinearSolver:
    """Solve Ax=b with regularization and constraints"""
    METHODS = ["lstsq", "qr", "svd", "ridge", "lasso", ...]
    def solve(A, b) -> ndarray
```

### Layer 3: Workflow Layer

#### Calibration (`src/figaroh/calibration/`)

**Purpose:** Kinematic calibration and geometric parameter identification

**Key Files:**
- `base_calibration.py` - Template method pattern for calibration
- `calibration_tools.py` - Forward kinematics, Jacobian, regressor utilities
- `parameter.py` - Parameter management and frame handling
- `config.py` - Unified/legacy config parsing

**Workflow (ASCII):**
```
User
  │
  ├─► initialize(config) ──► BaseCalibration
  │                              │
  │                              ├─► load_data() ──► CalibrationTools
  │                              │
  │                              ├─► [_load_validation_data(path)]  ← optional
  │                              │
  │                              ├─► calculate_base_kinematics_regressor()
  │                              │        │
  │                              │        ├─► get frames ──► Robot ──► Backend
  │                              │        │                              │
  │                              │        ◄── frame poses ──────────────┘
  │                              │        │
  │                              │        ◄── Jacobian ──► Backend
  │                              │        │
  │                              │        └─► regressor W_base
  │                              │
  ├─► solve() ──────────────────► Optimization Loop
  │                              │
  │                              ├─► cost_function(var)
  │                              │     │
  │                              │     ├─► calc_updated_fkm()
  │                              │     │     │
  │                              │     │     └─► compute_forward_kinematics() ──► Backend
  │                              │     │           │
  │                              │     │           ◄── updated poses
  │                              │     │
  │                              │     └─► residuals via _compute_logmap_residuals()
  │                              │           (SE3 log-map, geometrically correct)
  │                              │
  │                              └─► optimize(least_squares)
  │                                    │
  ├─► _evaluate_solution() ──────► per-DOF stats, condition number, correlation
  │                                    │
  ├─► _compute_validation_metrics()  ← if validation data loaded
  │     │                              │
  │     ├─► nominal FK (zeros) ───► calc_updated_fkm() ──► Backend
  │     ├─► calibrated FK ────────► calc_updated_fkm() ──► Backend
  │     └─► compare vs ground truth → improvement %
  │
  └─► print_quality_report() ────► formatted terminal output
                                     (convergence, per-DOF residuals,
                                      condition number, validation,
                                      param uncertainty, correlations)
```

**Workflow (Mermaid Sequence Diagram):**
```mermaid
sequenceDiagram
    participant User
    participant BaseCalibration
    participant CalibrationTools
    participant Robot
    participant Backend

    User->>BaseCalibration: initialize(config)
    BaseCalibration->>CalibrationTools: load_data()
    BaseCalibration->>CalibrationTools: calculate_base_kinematics_regressor()
    CalibrationTools->>Robot: get frames
    Robot->>Backend: compute_forward_kinematics()
    Backend-->>Robot: frame poses
    Robot-->>CalibrationTools: frame data
    CalibrationTools->>Backend: compute_jacobian()
    Backend-->>CalibrationTools: Jacobian matrix
    CalibrationTools-->>BaseCalibration: regressor W_base

    User->>BaseCalibration: solve()
    BaseCalibration->>BaseCalibration: cost_function(var)
    BaseCalibration->>CalibrationTools: calc_updated_fkm()
    CalibrationTools->>Backend: compute_forward_kinematics()
    Backend-->>CalibrationTools: updated poses
    CalibrationTools-->>BaseCalibration: residuals
    BaseCalibration->>BaseCalibration: optimize (least_squares)
    BaseCalibration-->>User: calibrated parameters
```

#### Identification (`src/figaroh/identification/`)

**Purpose:** Dynamic parameter identification (inertial parameters)

**Key Files:**
- `base_identification.py` - Main identification workflow
- `identification_tools.py` - Utility functions for identification
- `parameter.py` - Parameter ordering and management
- `physical_consistency.py` - Feasibility checking
- `cad_constraints.py` - CAD-based parameter bounds
- `reconstruction.py` - Parameter reconstruction from base parameters

**Workflow (ASCII):**
```
User
  │
  ├─► initialize() ──────────────► BaseIdentification
  │                                    │
  │                                    ├─► process_data()
  │                                    │    (filter, differentiate)
  │                                    │
  │                                    ├─► build_basic_regressor(q,v,a)
  │                                    │        │
  │                                    │        └─► compute_regressor() ──► Backend
  │                                    │              │
  │                                    │              ◄── W (N×10nv)
  │                                    │
  │                                    └─► initialize_standard_parameters()
  │
  ├─► solve() ───────────────────────► Main Solving Loop
  │                                    │
  │                                    ├─► eliminate_zero_columns()
  │                                    │     │
  │                                    │     └─► W_reduced
  │                                    │
  │                                    ├─► apply_decimation() [optional]
  │                                    │     │
  │                                    │     └─► tau_processed, W_processed
  │                                    │
  │                                    ├─► qr_decomposition(W)
  │                                    │     │
  │                                    │     └─► W_base, elimination_matrix
  │                                    │
  │                                    ├─► solve(W_base, tau) ──► LinearSolver
  │                                    │     │
  │                                    │     └─► phi_base
  │                                    │
  │                                    ├─► check_feasibility(phi) ──► PhysicalConsistency
  │                                    │     │
  │                                    │     └─► validation result
  │                                    │
  │                                    └─► compute_quality_metrics()
  │                                          │
  ◄──────── identified parameters + metrics ┘
```

**Workflow (Mermaid Sequence Diagram):**
```mermaid
sequenceDiagram
    participant User
    participant BaseIdentification
    participant RegressorBuilder
    participant Backend
    participant Solver
    participant QRDecomposer
    participant PhysicalConsistency

    User->>BaseIdentification: initialize()
    BaseIdentification->>BaseIdentification: process_data()
    BaseIdentification->>RegressorBuilder: build_basic_regressor(q,v,a)
    RegressorBuilder->>Backend: compute_regressor()
    Backend-->>RegressorBuilder: W (N×10nv)
    RegressorBuilder-->>BaseIdentification: full regressor

    User->>BaseIdentification: solve()
    BaseIdentification->>BaseIdentification: eliminate_zero_columns()
    BaseIdentification->>BaseIdentification: apply_decimation() [optional]
    BaseIdentification->>QRDecomposer: qr_decomposition(W)
    QRDecomposer-->>BaseIdentification: W_base, elimination matrix

    BaseIdentification->>Solver: solve(W_base, tau)
    Solver-->>BaseIdentification: phi_base

    BaseIdentification->>PhysicalConsistency: check_feasibility(phi)
    PhysicalConsistency-->>BaseIdentification: validation result

    BaseIdentification-->>User: identified parameters + metrics
```

#### Optimal Trajectory (`src/figaroh/optimal/`)

**Purpose:** Optimal trajectory generation for identification/calibration

**Key Files:**
- `base_optimal_trajectory.py` - Trajectory optimization workflow
- `base_optimal_calibration.py` - Optimal calibration trajectories
- `contraints.py` - Joint/torque/collision constraints
- `base_parameter.py` - OED parameter handling

---

## 4. Module Details

### 1. Backend Layer

#### PinocchioBackend (`backends/pinocchio.py`)

**Status:** ✅ Implemented (Default)

**Key Features:**
- Excellent URDF support
- CPU-optimized dense operations
- Mature and stable
- Full SE(3) Lie group support

**Pinocchio Functions Used:**
- Model loading: `buildModelsFromUrdf()`, `JointModelFreeFlyer()`
- Dynamics: `crba()`, `nonLinearEffects()`, `rnea()`
- Kinematics: `framesForwardKinematics()`, `updateFramePlacements()`
- Jacobians: `computeFrameJacobian()`, `computeFrameKinematicRegressor()`
- Regressors: `computeJointTorqueRegressor()`
- Transformations: `SE3()`, `Quaternion()`, `rpy.rpyToMatrix()`

#### MuJoCoBackend (`backends/mujoco.py`)

**Status:** ✅ Implemented (Phase 1)

**Key Features:**
- High-performance sparse operations
- Built-in URDF → MJCF converter
- Contact dynamics support
- 2-3x faster than Pinocchio for large robots

**MuJoCo Functions Used:**
- Model loading: `MjModel.from_xml_path()`
- Dynamics: `mj_forward()`, `mj_fullM()`, `mj_inverse()`
- Kinematics: `mj_kinematics()`, `mj_comPos()`
- Jacobians: `mj_jac()`, `mj_jacBody()`

### 2. Tools Layer

#### Robot (`tools/robot.py`)

**Class:** `Robot(RobotWrapper)`

**Purpose:** Enhanced robot model with free-flyer support

**Key Methods:**
```python
def __init__(robot_urdf, package_dirs, isFext, freeflyer_ori, freeflyer_limits)
def _configure_freeflyer(freeflyer_ori)
def _set_freeflyer_limits()
def display_q0(visualizer_type, q)
```

**Pinocchio Dependencies:**
- Inherits from `pinocchio.robot_wrapper.RobotWrapper`
- Uses `pin.JointModelFreeFlyer()` for floating base
- Uses `pin.buildModelsFromUrdf()` indirectly

**Free-Flyer Configuration:**
- Position limits: configurable (default: [-1, 1])
- Quaternion normalization: automatic
- Orientation matrix: customizable 3×3 rotation

#### RegressorBuilder (`tools/regressor.py`)

**Class:** `RegressorBuilder`

**Purpose:** Compute observation regressor matrix W for identification

**Key Methods:**
```python
def build_basic_regressor(q, v, a, identif_config) -> W
def _build_joint_torque_regressor(Q, V, A, N) -> W
def _build_external_wrench_regressor(Q, V, A, N) -> W
```

**Regressor Structure:**
- **Joint torque mode:** W ∈ ℝ^(N×nv) × ℝ^(10nv)
- **External wrench mode:** W ∈ ℝ^(N×6) × ℝ^(10×nb_bodies)
- **Parameters per body:** 10 (m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz)
- **Additional parameters:** friction (fv, fs), actuator inertia (ia), joint offset

**Pinocchio Dependencies:**
- `pin.computeJointTorqueRegressor(model, data, q, v, a)` - Core regressor computation

**Backend Integration:**
```python
# Through backend abstraction:
W = backend.compute_regressor(q, v, a)
```

#### LinearSolver (`tools/solver.py`)

**Class:** `LinearSolver`

**Purpose:** Solve overdetermined system Ax = b with regularization

**Methods:**
```python
METHODS = ["lstsq", "qr", "svd", "ridge", "lasso", "elastic_net",
           "tikhonov", "constrained", "robust", "weighted"]

def solve(A, b) -> x
def _solve_lstsq(A, b) -> x
def _solve_ridge(A, b) -> x
def _solve_constrained(A, b) -> x  # with linear constraints
```

**Optimization Features:**
- Regularization: L1, L2, elastic net, Tikhonov
- Constraints: linear equality/inequality, box bounds
- Robust methods: iterative reweighting
- Condition number monitoring

**No Pinocchio Dependencies** - Pure linear algebra

### 3. Calibration Layer

#### BaseCalibration (`calibration/base_calibration.py`)

**Class:** `BaseCalibration(ABC)`

**Purpose:** Template method pattern for kinematic calibration

**Workflow Methods:**
```python
def __init__(robot, config_file, del_list)
def initialize() -> None
def solve() -> None                    # Main optimization + quality report
def _evaluate_solution(result, ...)    # → dict with per-DOF stats, condition number
def _compute_per_dof_stats(...)        # → mean, std, RMSE, max, R² per DOF
def _compute_condition_number(...)     # → cond(J) with well/moderate/ill label
def _compute_parameter_correlation()   # → pairs with |ρ| > 0.8
def _load_validation_data(path)        # → load separate held-out measurements
def _compute_validation_metrics()      # → nominal vs calibrated FK vs ground truth
def print_quality_report()             # → formatted terminal output
def plot_results() -> None
```

**Cost Function (uses SE3 log-map, geometrically correct):**
```python
@abstractmethod
def cost_function(var) -> residuals:
    """Robot-specific implementation"""
    PEEe = calc_updated_fkm(model, data, var, q_measured, calib_config)
    residuals = _compute_logmap_residuals(PEE_measured, PEEe,
                                          position_frame="body"|"world")
    return apply_measurement_weighting(residuals)
```

**Validation Data Support:**
Separate validation measurements can be provided via config (`validation_data_file`)
or CLI (`--validation-data`). After calibration, FK is computed with both nominal
(zero params) and calibrated parameters on the validation set, and errors are
compared against ground truth to measure improvement.

**Quality Report (printed automatically after solve()):**
```
  CALIBRATION QUALITY REPORT
  Convergence: ✓ converged    Iterations: 47    Cost: 0.02345
  Outliers:     3 / 500 (0.6%)
  Condition:    42.1 (well-conditioned)
  Per-DOF Residuals (training set, n=500)
    DOF        Mean      Std       RMSE      Max       R²
    X (mm)      0.12     0.85      0.86      3.21     0.9876
    Y (mm)     -0.34     0.62      0.71      2.45     0.9765
    Z (mm)      0.08     0.45      0.46      1.89     0.9934
    rx (deg)    0.002    0.015     0.015     0.034    0.9991
    ry (deg)   -0.001    0.012     0.012     0.028    0.9987
    rz (deg)    0.003    0.018     0.018     0.031    0.9989
  Overall
    Position RMSE: 0.68 mm    Orientation RMSE: 0.015 deg
  Validation (separate set, n=100)
    Metric               Nominal     Calibrated    Improvement
    Position RMSE        12.34 mm     0.71 mm        94.2% ↓
    Orientation RMSE      2.15 deg     0.016 deg      99.3% ↓
  Parameter Uncertainty (top 5)
  Correlated pairs (|ρ| > 0.8): base_x ↔ pEE_x  ρ = +0.93
```

**Calibration Models:**
- **full_params:** All geometric parameters (DH, joint offsets, etc.)
- **joint_offset:** Joint encoder offsets only

**Key Configuration Keys:**
- `calibration_type`: "full_params" or "joint_offset"
- `calib_params`: List of parameter names to calibrate
- `base_frame`, `end_frame`: Kinematic chain definition
- `measurement_file`: Path to calibration data
- `validation_data_file`: Path to separate validation data (optional)
- `position_frame`: "body" or "world" — position error reference frame

#### URDF Exporter (`tools/urdf_exporter.py`)

**Key Function:**
```python
def export_urdf(
    urdf_path: str,
    params: Dict[str, float],
    output_path: str,
    verbose: bool = False,
) -> str:
    """Apply calibrated joint parameters to a URDF file.

    Handles 12 joint-level categories: xyz offsets, rpy orientations,
    and joint-axis corrections. Automatically separates metrology
    frame parameters (base/pEE/phiEE) which are NOT applied to URDF.

    Returns path to the modified URDF.
    """
```

**Parameter categories auto-detected via registry:**
- Joint: `{joint}_x`, `{joint}_y`, `{joint}_z`, `{joint}_rx`, `{joint}_ry`, `{joint}_rz`
- Axis: `{joint}_ax`, `{joint}_ay`, `{joint}_az`, `{joint}_phix`, `{joint}_phiy`, `{joint}_phiz`
- Metrology (not applied): `base_`, `pEE`, `phiEE`

#### Export Validation (`tools/export_validation.py`)

**Key Classes:**

```python
class URDFComparison(nominal_urdf, modified_urdf):
    """Compare two URDF models via FK + viser visualization."""
    def fk_consistency_check(n_samples=100, seed=42) -> FkConsistencyResult
    def static_poses(poses=None) -> List[PoseError]
    def show_interactive_validation(n_trajectory=50, port=8080)
    def show_trajectory_animation(n_configs=50, ...)
    def show_static_grid(poses=None, ...)
    def show_overlay(server=None, port=8080, ...)

class FkConsistencyResult:
    """Aggregated FK consistency check."""
    rmse_position: float         # meters
    rmse_orientation: float      # radians
    max_position: float
    max_orientation: float
    per_sample: List[PoseDelta]
```

`show_interactive_validation()` renders a combined viser scene with:
- Trajectory animation (line-segment path traces, replay button, speed slider)
- Static pose comparison grid (opacity slider for modified model)
- Error plots (uplot) — position (mm) and orientation (deg) over the trajectory
- Pose selector to inspect individual configuration errors

The comparison loads URDFs via `yourdfpy` with a custom filename handler that
resolves `package://` mesh URIs by searching `ROS_PACKAGE_PATH` and auto-discovered
`models/` directories. Pre-loaded `yourdfpy.URDF` objects are passed directly to
`ViserUrdf` to avoid double-loading with the wrong handler.

#### CalibrationTools (`calibration/calibration_tools.py`)

**Key Functions:**

**Forward Kinematics Update:**
```python
def calc_updated_fkm(model, data, var, q_measured, calib_config) -> poses:
    """Update FK with calibration parameters. Composes wMf = wMo * oMee * eeMf:
    wMo (world/measurement frame -> chain start; unknown-baseframe or
    known camera/ref-frame anchor), oMee (geometric joint-placement errors,
    optionally with per-joint gravity-torque-driven elasticity when
    calib_config["non_geom"] is set), eeMf (end frame -> marker frame)."""
    # 1. Update joint placements (full_params / joint_offset)
    update_joint_placement(model, var, calib_config)
    # 2. Compute forward kinematics (re-run after any elastic perturbation --
    #    get_rel_transform reads the cached data.oMf, not live state)
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    # 3. Extract end-effector poses, composed with wMo and eeMf
    return get_frame_poses(data, frame_ids)
```

This is the single FK-update function for calibration — it previously had a
sibling, `update_forward_kinematics`, which supported elasticity and
camera-ref-frame composition but had accumulated several correctness
regressions and had no callers anywhere in `figaroh`/`figaroh-examples`; its
capabilities were merged into `calc_updated_fkm` and the dead function was
removed.

**Jacobian & Regressor:**
```python
def get_rel_jac(model, data, q, start_frameId, end_frameId) -> J_rel:
    """Compute relative Jacobian"""
    J_start = pin.computeFrameJacobian(model, data, q, start_frameId, pin.LOCAL)
    J_end = pin.computeFrameJacobian(model, data, q, end_frameId, pin.LOCAL)
    return J_end - J_start

def calculate_base_kinematics_regressor(robot, q_data, calib_config) -> W_base:
    """Calculate base kinematic regressor via QR"""
    W = calculate_kinematics_model(robot, q_data, calib_config)
    Q, R, P = qr_decomposition(W)
    return extract_base_regressor(Q, R, P)
```

**Pinocchio Functions Used (15+ functions):**
- FK: `framesForwardKinematics()`, `updateFramePlacements()`
- Jacobian: `computeFrameJacobian()`, `computeFrameKinematicRegressor()`
- Transforms: `SE3()`, `SE3.Identity()`, `rpy.rpyToMatrix()`, `rpy.matrixToRpy()`
- Dynamics: `computeGeneralizedGravity()` (for elasticity compensation)
- Config: `randomConfiguration()` (for identifiability analysis)

### 4. Identification Layer

#### BaseIdentification (`identification/base_identification.py`)

**Class:** `BaseIdentification(ABC)`

**Purpose:** Dynamic parameter identification workflow

**Main Workflow:**
```python
def initialize(truncate=None):
    self.process_data(truncate)
    self.calculate_full_regressor()
    self.initialize_standard_parameters()
    self.compute_reference_torque()

def solve(decimate=True, decimation_factor=10, zero_tolerance=0.001):
    # 1. Eliminate zero columns
    regressor_reduced, active_params = self._eliminate_zero_columns()

    # 2. Apply decimation (optional)
    if decimate:
        tau_processed, W_processed = self._apply_decimation(regressor_reduced)

    # 3. QR decomposition
    W_base, elimination_matrix = qr_decompose(W_processed)

    # 4. Solve for base parameters
    phi_base = self.solver.solve(W_base, tau_processed)

    # 5. Validate physical consistency
    validate_parameters(phi_base)

    # 6. Compute quality metrics
    self._compute_quality_metrics()

    return phi_base
```

**Data Processing:**
```python
def process_data(truncate=None):
    """Load and filter trajectory data"""
    # Load: q, v, a, tau from files
    # Filter: Butterworth, median filter
    # Differentiate: gradient or finite difference
    # Validate: check shapes, remove outliers
```

**Quality Metrics:**
- RMS error
- Relative RMS error
- Condition number
- Parameter correlation matrix
- Standard deviations

#### Parameter Handling (`identification/parameter.py`)

**Key Functions:**

**Parameter Reordering:**
```python
def reorder_inertial_parameters(p10: ndarray) -> ndarray:
    """Reorder from Pinocchio format to standard format

    Pinocchio: [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
    Standard:  [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, m]
    """
    param_order = [4, 5, 7, 6, 8, 9, 1, 2, 3, 0]
    return p10[param_order]
```

**Additional Parameters:**
```python
def add_standard_additional_parameters(phi_std, identif_config) -> phi_full:
    """Add friction, actuator inertia, joint offset"""
    # Append: fv (viscous friction)
    # Append: fs (static friction)
    # Append: ia (actuator inertia)
    # Append: offset (joint zero offset)
    return phi_full
```

#### Physical Consistency (`identification/physical_consistency.py`)

**Key Functions:**

**Feasibility Checking:**
```python
def check_p10_feasibility(p10: ndarray) -> bool:
    """Check if inertial parameters are physically feasible

    Conditions:
    1. Mass m > 0
    2. Pseudo-inertia matrix P is positive semi-definite
    3. Triangle inequalities for inertia tensor
    """
    P = pseudo_inertia_matrix_from_p10(p10)
    eigenvalues = np.linalg.eigvalsh(P)
    return all(eigenvalues >= -1e-10)  # Allow small numerical errors

def pseudo_inertia_matrix_from_p10(p10: ndarray) -> ndarray:
    """Build 4×4 pseudo-inertia matrix from 10D parameters"""
    # Use Pinocchio's PseudoInertia class
    pseudo_inertia = pin.PseudoInertia.FromDynamicParameters(p10)
    return pseudo_inertia.toMatrix()
```

**Pinocchio Dependency:**
- `pin.PseudoInertia.FromDynamicParameters(p10)` - Convert to pseudo-inertia

### 5. Optimal Layer

#### BaseOptimalTrajectory (`optimal/base_optimal_trajectory.py`)

**Purpose:** Generate optimal exciting trajectories for identification

**Optimization Objective:**
```
minimize: condition_number(W(q,v,a))
subject to:
    - Joint limits: q_min ≤ q ≤ q_max
    - Velocity limits: v_min ≤ v ≤ v_max
    - Acceleration limits: a_min ≤ a ≤ a_max
    - Torque limits: τ_min ≤ τ(q,v,a) ≤ τ_max
    - Collision avoidance: distance > safety_margin
    - Duration: t_final = T
```

**Parameterization:**
- Fourier series: `q(t) = q₀ + Σ aₖsin(ωₖt) + bₖcos(ωₖt)`
- B-splines: cubic/quintic splines with waypoints
- Polynomial: 5th order trajectories

### 6. Reporting & Verification (V&V) Suite

**Purpose:** Turn the metrics `solve()` already computes into four
consumable artifacts — a terminal report (human, right now), a
self-contained HTML report (human, shareable), a machine-readable verdict
(CI/scripts), and a static two-run compare page (offline, no backend). See
[`docs/source/reporting_and_verification.md`](docs/source/reporting_and_verification.md)
for usage; this section covers structure only.

**Shared kernel (`tools/_report_common.py`):** HTML/CSS primitives
(`_STYLE`, `_esc`, `_insights_section`, `_param_uncertainty_section`,
`_correlation_section`, `_series_panel_section`) and the verdict data model
used by both domains:

```python
@dataclass
class ThresholdCheck:
    name: str; value: float; threshold: float
    comparison: str  # "max" or "min"
    passed: bool

@dataclass
class VerificationVerdict:
    passed: bool
    checks: List[ThresholdCheck]
    metrics: Dict[str, float]
    insights: List[str]
    metadata: Dict[str, Any]   # provenance: git commit, config hash, timestamp
    series: Dict[str, Any]     # before/after time series, empty if no validation data
    compat: Dict[str, Any]     # domain/joint-names/decimate/sample_count, for cross-run compare
```

**Per-domain adapters** — structurally parallel, not shared, because
calibration (iterative nonlinear fit, per-DOF pose error) and
identification (one-shot linear QR solve, per-joint torque error) produce
different diagnostics:

| | `tools/report.py` (calibration) | `tools/identification_report.py` (identification) |
|---|---|---|
| HTML generator | `generate_calibration_report(calibrator, ...)` | `generate_identification_report(identifier, ...)` |
| Called via | `BaseCalibration.export_html_report()` | `BaseIdentification.export_html_report()` |
| Opt-in flag | `solve(html_report=True)` | `solve(html_report=True)` |

**Verification methods, added to both `BaseCalibration` and
`BaseIdentification`:**

```python
def verify(thresholds: dict = None) -> VerificationVerdict:
    """Check self.evaluation_metrics / self.result against pass/fail
    thresholds (default: condition_number ≤ 1000, plus domain-specific
    validation-set thresholds). A threshold whose metric wasn't computed
    (no validation data configured) is skipped, not failed."""

def export_verification_report(
    output_path: str = None, output_dir: str = "results",
    thresholds: dict = None,
) -> str:
    """verify() + write the verdict as numpy-safe JSON
    ({output_dir}/{calibration,identification}_verification.json)."""
```

Called explicitly (not from inside `solve()`) — verification is opt-in
output computed from already-stored results, not a new gate that could
make a successful numerical solve raise.

**Cross-run comparison (`tools/compare_report.py`):** a static,
self-contained HTML shell (`generate_compare_page()`) with no run object of
its own — it loads two `export_verification_report()` JSON files
client-side (drag-and-drop or file picker) and, after a mandatory
compatibility check on their `compat` blocks (domain, joint/DOF names,
`decimate`, sample count), renders a metric diff table and an overlaid
before/after chart entirely in the browser. No backend, no run history —
deliberately cut down from a broader "V&V dashboard" concept; see
`docs/decisions/roadmap-mujoco-sysid-inspired-features.md` (Feature 6) for
why.

---

## 5. Pinocchio Integration Map

### Critical Functions (Backend-abstracted ✅)

| Pinocchio Function | Purpose | Files Using | Backend Method | Migration Status |
|--------------------|---------|-------------|----------------|-----------------|
| `computeJointTorqueRegressor()` | Compute regressor W | regressor.py | `compute_regressor()` | ✅ Routed through `robot.backend` |
| `rnea()` | Compute joint torques | randomdata.py, cubic_spline.py | `compute_inverse_dynamics()` | ✅ randomdata.py migrated; cubic_spline.py deferred |
| `crba()` | Mass matrix M(q) | (in PinocchioBackend) | `compute_mass_matrix()` | ✅ Implemented |
| `aba()` | Forward dynamics | (in PinocchioBackend) | `compute_forward_dynamics()` | ✅ Implemented |
| `computeCoriolisMatrix()` | Coriolis matrix C(q,v) | (in PinocchioBackend) | `compute_coriolis_matrix()` | ✅ Implemented |
| `computeGeneralizedGravity()` | Gravity vector g(q) | calibration_tools.py | `compute_gravity_vector()` | ✅ Routed through backend |
| `framesForwardKinematics()` | Update all frames | calibration_tools.py (5 locations) | `compute_forward_kinematics()` | ✅ Routed through backend |
| `updateFramePlacements()` | Update frame placements | calibration_tools.py | `compute_forward_kinematics()` | ✅ Included in FK |
| `computeFrameJacobian()` | Frame Jacobian | calibration_tools.py | `compute_jacobian()` | ✅ Routed through backend |
| `difference()` | Manifold velocity | identification_tools.py | `compute_difference()` | ✅ Routed through backend |
| `randomConfiguration()` | Random config sampling | calibration_tools.py | `random_configuration()` | ✅ Routed through backend |

### Pinocchio-Specific Functions (Not abstracted ❌)

| Function | Purpose | Why Not Abstracted |
|----------|---------|-------------------|
| `computeFrameKinematicRegressor()` | Kinematic regressor for calibration | Pinocchio-only feature (no MuJoCo/Genesis equivalent); uses `get_model_object()` escape hatch |
| `model.jointPlacements[idx]` mutation | Modify joint frame placement | **Critical blocker**: MuJoCo/Genesis models are immutable after compilation. Calibration requires runtime model mutation. |
| `model.addFrame()` | Add custom frame at runtime | MuJoCo/Genesis don't support runtime frame addition |
| `SE3()`, `SE3.Identity()`, `SE3.actInv()` | Rigid body transform type | Pinocchio-specific type system; could use `scipy.spatial.transform` (future) |
| `Quaternion()` | Quaternion operations | Could use `scipy.spatial.transform.Rotation` (future) |
| `rpy.rpyToMatrix()`, `rpy.matrixToRpy()` | RPY ↔ rotation matrix | Could use `scipy.spatial.transform.Rotation` (future) |
| `Force()` | 6D wrench representation | Pinocchio-specific type |
| `Frame()`, `FrameType`, `Inertia.Zero()` | Frame/inertia construction | Pinocchio-specific model types |
| `computeCollisions()`, `computeDistance()` | Collision detection | Fundamentally simulator-specific (hppfcl vs MuJoCo vs PhysX) |
| `centerOfMass()`, `computeSubtreeMasses()` | COM computation | Visualization-specific, low priority |

### Backend Mapping Strategy

```mermaid
graph LR
    subgraph "FIGAROH Code"
        Tools[Tools Layer]
        Calib[Calibration]
        Identif[Identification]
    end

    subgraph "Backend Abstraction"
        API[Backend API]
    end

    subgraph "Implementations"
        Pin[Pinocchio]
        Muj[MuJoCo]
        Gen[Genesis]
    end

    Tools -->|compute_regressor| API
    Tools -->|compute_mass_matrix| API
    Tools -->|compute_jacobian| API
    Calib -->|compute_forward_kinematics| API
    Calib -->|compute_jacobian| API
    Identif -->|compute_regressor| API

    API -->|delegates| Pin
    API -->|delegates| Muj
    API -->|delegates| Gen

    style API fill:#ffeb3b,stroke:#333,stroke-width:3px
```

---

## 6. Data Flow

### Identification Data Flow

```mermaid
flowchart TD
    A[Robot Model URDF] --> B[Load Robot]
    C[Trajectory Data q v a tau] --> D[Process Data Filter Differentiate]

    B --> E[RegressorBuilder]
    D --> E

    E --> F[Full Regressor W N×10nv]

    F --> G[Eliminate Zero Columns W_reduced]

    G --> H{Decimate?}
    H -->|Yes| I[Decimation Reduce N]
    H -->|No| J[Skip]
    I --> K[QR Decomposition W_base]
    J --> K

    K --> L[Linear Solver Ax = b]

    L --> M[Base Parameters phi_base]

    M --> N[Physical Consistency Check]

    N --> O{Feasible?}
    O -->|Yes| P[Reconstruct Full Parameters]
    O -->|No| Q[Report Error]

    P --> R[Quality Metrics RMS Condition]

    R --> S[Save Results]

    style E fill:#4caf50,stroke:#333,stroke-width:2px
    style K fill:#2196f3,stroke:#333,stroke-width:2px
    style L fill:#ff9800,stroke:#333,stroke-width:2px
    style N fill:#f44336,stroke:#333,stroke-width:2px
```

### Calibration Data Flow

```mermaid
flowchart TD
    A[Robot Model URDF] --> B[Load Robot]
    C[Measurement Data q_meas poses_meas] --> D[Load Measurements]
    V[Validation Data CSV] -.->|optional| VAL[_load_validation_data]

    B --> E[Calculate Base Kinematic Regressor]
    D --> E

    E --> F[Base Regressor W_base Identifiable subset]

    F --> G[Initialize Variables var0]

    G --> H[Optimization Loop least_squares]

    H --> I[Cost Function residuals via SE3 log-map]

    I --> J[Update FK with calibration params]

    J --> K[Compute Predicted Poses]

    K --> L[Calculate Residuals measured vs predicted geometrically correct log-map]

    L --> M{Converged?}
    M -->|No| H
    M -->|Yes| N[Calibrated Parameters]

    N --> O[Evaluate Solution per-DOF stats condition number parameter correlation R2]

    VAL --> Q[Compute Validation Metrics nominal FK vs calibrated FK vs ground truth]

    O --> R[print_quality_report terminal output]
    Q -->|if validation data| R

    R --> S[Save and Visualize]

    style E fill:#4caf50,stroke:#333,stroke-width:2px
    style H fill:#2196f3,stroke:#333,stroke-width:2px
    style J fill:#ff9800,stroke:#333,stroke-width:2px
    style O fill:#9c27b0,stroke:#333,stroke-width:2px
    style R fill:#e91e63,stroke:#333,stroke-width:2px
```

---

## 7. Backend Architecture

### Backend Selection

```python
from figaroh.backends import get_backend

# Option 1: From URDF (auto-detect format)
backend = get_backend("pinocchio", model_path="robot.urdf")
backend = get_backend("mujoco", model_path="robot.urdf")  # Auto-converts to MJCF

# Option 2: Explicit format
backend = get_backend("mujoco", model_path="robot.xml", format="mjcf")

# Option 3: With options
backend = get_backend("pinocchio", model_path="robot.urdf",
                     package_dirs=["."], root_joint="free_flyer")
```

### Backend Interface Contract

```python
class DynamicsBackend(ABC):
    """Abstract interface for dynamics computation"""

    # === Core Dynamics (abstract) ===
    @abstractmethod
    def compute_mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """M(q) ∈ ℝ^(nv×nv)"""

    @abstractmethod
    def compute_coriolis_matrix(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """C(q,v) ∈ ℝ^(nv×nv)"""

    @abstractmethod
    def compute_gravity_vector(self, q: np.ndarray) -> np.ndarray:
        """g(q) ∈ ℝ^nv"""

    @abstractmethod
    def compute_inverse_dynamics(self, q: np.ndarray, v: np.ndarray,
                                 a: np.ndarray) -> np.ndarray:
        """τ = RNEA(q,v,a) ∈ ℝ^nv"""

    @abstractmethod
    def compute_forward_dynamics(self, q: np.ndarray, v: np.ndarray,
                                 tau: np.ndarray) -> np.ndarray:
        """qdd = ABA(q,v,τ) ∈ ℝ^nv"""

    # === Kinematics (abstract) ===
    @abstractmethod
    def compute_forward_kinematics(self, q: np.ndarray) -> Dict[str, Any]:
        """Returns: {frame_name: {position, orientation, transformation}}"""

    @abstractmethod
    def compute_jacobian(self, q: np.ndarray, frame: str) -> np.ndarray:
        """J(q, frame) ∈ ℝ^(6×nv)"""

    # === Identification (abstract) ===
    @abstractmethod
    def compute_regressor(self, q: np.ndarray, v: np.ndarray,
                         a: np.ndarray) -> np.ndarray:
        """W(q,v,a) ∈ ℝ^(nv × 10*nv)"""

    # === Properties (abstract) ===
    @property
    @abstractmethod
    def nq(self) -> int: ...
    @property
    @abstractmethod
    def nv(self) -> int: ...
    @property
    @abstractmethod
    def model_format(self) -> str: ...

    # === Optional methods (raise NotImplementedError by default) ===
    def get_joint_names(self) -> list: ...
    def get_frame_names(self) -> list: ...
    def get_inertias(self) -> list: ...
        """Per-body inertia objects (for nonzero-inertia filtering)"""
    def get_frame_id(self, frame: str) -> int: ...
    def compute_difference(self, q1, q2) -> np.ndarray: ...
        """Lie group difference (q2 ⊖ q1) → [nv]"""
    def compute_integrate(self, q, v) -> np.ndarray: ...
        """Lie group integration (q ⊕ v) → [nq]"""
    def random_configuration(self) -> np.ndarray: ...
    def get_model_object(self) -> Any: ...
        """Escape hatch: return underlying simulator model (pin.Model, mj.MjModel, ...)
        for advanced features not yet abstracted (PseudoInertia, collision model, etc.)"""
    def compute_dynamics_derivatives(self, q, v) -> Dict[str, np.ndarray]: ...
```

### Implementation Comparison

| Feature | Pinocchio | MuJoCo | Genesis | Isaac Sim |
|---------|-----------|--------|---------|-----------|
| **Format** | URDF | MJCF (auto-converts URDF) | URDF/MJCF/USD | USD |
| **Performance** | Fast (dense) | Faster (sparse) | Fastest (GPU) | Fast (GPU) |
| **Contact** | Limited | Excellent | Excellent | Excellent |
| **Visualization** | MeshCat | MuJoCo Viewer | Native | Native |
| **Maturity** | Mature | Mature | New | Mature |
| **Use Case** | Research, identification, calibration | Simulation, control | Large-scale, parallel | Sim-to-real |
| **Status** | ✅ Implemented (default) | ⚠️ Partial (regressor is placeholder) | ❌ Not started | ❌ Not started |
| **Model manipulation** | ✅ Runtime mutable | ❌ Immutable after compile | ❌ Immutable after build | ❌ Requires physics re-init |
| **Kinematic regressor** | ✅ Native (`computeFrameKinematicRegressor`) | ❌ No equivalent | ❌ No equivalent | ❌ No equivalent |

### Backend Migration Status

**Fully portable (identification path):**
- `RegressorBuilder` → `backend.compute_regressor()`
- `randomdata.py` → `backend.compute_inverse_dynamics()`
- `identification_tools.py` → `backend.compute_difference()`

**Partially portable (calibration path):**
- `calibration_tools.py` — 7 functions accept optional `backend` parameter for FK, Jacobian, gravity, random-configuration
- `computeFrameKinematicRegressor` uses `get_model_object()` escape hatch (no backend equivalent)
- Model manipulation (`model.jointPlacements[idx]` mutation + revert) remains Pinocchio-specific — see [Backend Migration Limitations](#backend-migration-limitations) below

**Not portable (by design):**
- `calibration/parameter.py` — `pin.SE3`, `pin.Frame`, `pin.Inertia.Zero()` data type construction
- `measurements/measurement.py` — `pin.SE3`, `pin.Force`, `pin.Frame` data types
- `tools/robotvisualization.py` — Meshcat/Gepetto visualization
- `tools/robotcollisions.py` — hppfcl/coal collision detection

---

## 8. Extension Points

### Adding a New Backend

1. **Create backend file:** `src/figaroh/backends/my_backend.py`

```python
from .base import DynamicsBackend
import my_simulator as sim

class MyBackend(DynamicsBackend):
    def __init__(self, model_path: str, **kwargs):
        self.model = sim.load_model(model_path)
        self.data = sim.create_data(self.model)

    def compute_mass_matrix(self, q):
        return sim.compute_M(self.model, self.data, q)

    # Implement all abstract methods...
```

2. **Register in `__init__.py`:**

```python
try:
    from .my_backend import MyBackend
    _AVAILABLE_BACKENDS["my_simulator"] = MyBackend
except ImportError:
    pass
```

3. **Add tests:** `tests/backends/test_my_backend.py`

4. **Update documentation**

### Adding a New Solver

1. **Add method to `LinearSolver`:**

```python
def _solve_my_method(self, A, b):
    """My custom solving method"""
    # Implementation
    return x
```

2. **Register in `METHODS` list**

3. **Add unit tests**

### Adding a New Calibration Type

1. **Extend `BaseCalibration`:**

```python
class MyRobotCalibration(BaseCalibration):
    def cost_function(self, var):
        # Robot-specific implementation
        return residuals
```

2. **Add configuration support in YAML**

3. **Create example in `figaroh-examples`**

---

## 9. Backend Migration Limitations

> **This section documents the fundamental barriers to fully migrating FIGAROH's
> calibration algorithms from Pinocchio to other simulator backends (MuJoCo, Genesis,
> IsaacSim). These are not implementation gaps — they are architectural constraints
> imposed by the simulators themselves.**

### The Core Problem: Model Manipulation

FIGAROH's geometric calibration algorithms work by **modifying joint frame placements
in-place, computing forward kinematics, then reverting the modifications** — repeated
hundreds of times per optimization iteration. This is the fundamental mechanism for
identifying kinematic parameters (joint offsets, link lengths, frame placements).

```python
# Current calibration pattern (calibration_tools.py):
model.jointPlacements[joint_idx].translation = updated_translation
model.jointPlacements[joint_idx].rotation = updated_rotation
# ... compute forward kinematics with modified model ...
pin.framesForwardKinematics(model, data, q)
# ... measure end-effector pose ...
# ... revert model to original ...
model.jointPlacements[joint_idx].translation = original_translation
model.jointPlacements[joint_idx].rotation = original_rotation
```

**Why this matters:** Calibration is not about finding a different configuration `q` —
it's about finding the correct **kinematic structure** itself (where joints are placed,
what their offsets are). The model *is* the parameter being optimized. This is
fundamentally different from identification (where `q`, `v`, `a` are the inputs and
inertial parameters are the outputs).

### Simulator Capabilities Matrix

| Capability | Pinocchio | MuJoCo | Genesis | IsaacSim |
|---|---|---|---|---|
| **Runtime model mutation** | ✅ `model.jointPlacements[idx]` directly mutable | ❌ Model immutable after `MjModel.from_xml_path()` | ❌ Model immutable after `scene.build()` | ❌ Requires physics re-initialization |
| **Runtime frame addition** | ✅ `model.addFrame()` | ❌ Fixed kinematic tree after compile | ❌ Frames defined at load time only | ❌ Complex USD prim manipulation |
| **Kinematic regressor** | ✅ `pin.computeFrameKinematicRegressor()` | ❌ No equivalent | ❌ No equivalent | ❌ No equivalent |
| **SE3/transform types** | ✅ `pin.SE3`, `pin.Quaternion`, `pin.rpy` | ❌ Raw arrays (`pos[3]` + `quat[4]`) | ❌ Raw arrays | ❌ `torch.Tensor` / USD transforms |

### Why MuJoCo/Genesis Can't Support This Pattern

MuJoCo and Genesis **compile** their models into optimized internal representations
(sparse matrices, JIT-compiled physics). After compilation, the kinematic tree is
fixed — you cannot change where a joint is placed without rebuilding the model from
XML/spec.

The workaround (reload model per iteration) is **prohibitively expensive**: a typical
calibration optimization runs hundreds of iterations with multiple samples per
iteration. Reloading a model from XML takes 10-100ms, compared to microseconds for
Pinocchio's in-place mutation.

### What This Means for the Backend Track

**Identification (dynamics) is fully portable.** The dynamics regressor, mass matrix,
Coriolis, gravity, inverse/forward dynamics are all abstracted in `DynamicsBackend`
and work across simulators. This is the high-value path for multi-simulator support.

**Calibration (kinematics) is Pinocchio-bound** for the foreseeable future. The
`calibration_tools.py` functions accept an optional `backend` parameter and route
dynamics calls (FK, Jacobian, gravity) through it when available — but the model
manipulation pattern (`update_joint_placement`) has no backend equivalent and remains
Pinocchio-specific via the `get_model_object()` escape hatch.

### Long-Term Paths to Full Portability

1. **Propose upstream changes to MuJoCo/Genesis** — request runtime model mutation
   support (modify body/joint transforms without full recompilation). This is the
   cleanest solution but depends on simulator maintainers' priorities. A PR to
   MuJoCo's `MjSpec` API enabling partial recompilation would be the most impactful.

2. **Reformulate calibration to avoid model mutation** — some calibration formulations
   can be rewritten to parameterize the kinematic error in configuration space rather
   than model space. This is a significant algorithmic change but would eliminate the
   model manipulation dependency entirely.

3. **Mutable model wrapper** — implement a wrapper that caches model modifications and
   reloads the simulator model only when needed (e.g., batch modifications, reload
   once per iteration instead of per sample). Reduces overhead but doesn't eliminate it.

4. **SE3/transform abstraction layer** — use `scipy.spatial.transform.Rotation` and
   `RigidTransform` as a simulator-agnostic replacement for `pin.SE3`/`pin.Quaternion`/
   `pin.rpy`. This would eliminate the data type coupling but not the model manipulation
   coupling. Feasible and low-risk, but only solves part of the problem.

5. **Finite-difference kinematic regressor** — implement `computeFrameKinematicRegressor`
   generically via finite differences of forward kinematics w.r.t. joint placement
   perturbations. Would work with any backend that supports FK, but requires model
   mutation (back to problem #1).

### Current Architecture Decision

The backend abstraction covers **dynamics computation** (identification + calibration
algorithm paths). It does **not** abstract:
- **Model manipulation** — Pinocchio-specific (runtime mutable model)
- **Data type construction** — `pin.SE3`, `pin.Quaternion`, `pin.Force`, `pin.Frame`
- **Kinematic regressor** — Pinocchio-only feature (no simulator equivalent)
- **Collision detection** — fundamentally simulator-specific (hppfcl vs MuJoCo vs PhysX)
- **Visualization** — already decoupled (figaroh-examples uses Viser, not Pinocchio viz)

This is a **deliberate architectural boundary**, not a gap. The backend track focuses on
making identification portable across simulators. Calibration remains Pinocchio-specific
until simulator APIs evolve to support runtime model mutation.

---

## 10. Performance Considerations

### Computational Bottlenecks

1. **Regressor Computation** (Most expensive)
   - `pin.computeJointTorqueRegressor()` called N times (once per sample)
   - **Optimization:** Batch computation, caching, sparse operations
   - **MuJoCo advantage:** 2-3x faster due to sparse representation

2. **QR Decomposition** (Medium)
   - QR on W_reduced (size depends on rank)
   - **Optimization:** Use LAPACK optimized QR, consider rank-revealing QR

3. **Forward Kinematics** (Low to Medium)
   - Called many times in calibration optimization
   - **Optimization:** Cache when parameters don't change

### Memory Considerations

**Regressor Matrix Size:**
- Full regressor: N × (10 × nv) where N = number of samples
- Example: 10,000 samples, 7 DOF → 10,000 × 70 = 700,000 elements (5.6 MB)
- **Large robots:** N=50,000, nv=50 → 50,000 × 500 = 25M elements (200 MB)

**Mitigation Strategies:**
1. **Decimation:** Reduce N by factor of 10-100 (configurable)
2. **Batching:** Process data in chunks
3. **Sparse representation:** For MuJoCo backend
4. **Out-of-core:** For very large datasets

### Numerical Stability

**Conditioning:**
- Monitor condition number: `κ(W) = σ_max / σ_min`
- Typical values: 10³-10⁶ (acceptable), >10⁸ (problematic)
- **Regularization:** Ridge, Tikhonov for ill-conditioned systems

**Scaling:**
- Normalize q, v, a to unit scale
- Scale parameters to similar magnitudes
- Use unit-aware weighting (position vs orientation)

**Manifold Handling:**
- Use `pin.difference()` for SE(3) velocity computation
- Avoid Euler angle singularities (use quaternions)
- Validate SE(3) transformations (orthogonality, det=1)

---

## Summary

FIGAROH's architecture achieves simulator independence through a **three-layer design**:

1. **Backend Layer** abstracts dynamics computation
2. **Tools Layer** implements core algorithms (regressor, solver, QR decomposition)
3. **Workflow Layer** orchestrates high-level tasks (calibration, identification, optimal)

**Key Strengths:**
- Clean separation of concerns
- Pluggable backends enable simulator flexibility for identification
- Mature Pinocchio integration as reference implementation
- "Strangler fig" migration pattern — backward-compatible, opt-in backend routing

**Current State (v0.4.4, June 2026):**
- ✅ `DynamicsBackend` interface defined (9 abstract + 9 optional methods)
- ✅ `PinocchioBackend` implemented (default, 293 tests, numerical correctness verified)
- ✅ `MuJoCoBackend` implemented (dynamics, FK, Jacobian; regressor delegates to Pinocchio)
- ✅ Core algorithm path migrated (identification + calibration route through backend)
- ✅ URDF exporter with 12 joint-level categories + metrology registry
- ✅ Export validation with interactive viser visualization (trajectory + static + error plots)
- ✅ Validation data support — ground-truth FK comparison against held-out measurements
- ✅ Statistical quality report — per-DOF residuals (R²), condition number, parameter correlation
- ✅ SE3 log-map pose error replaces element-wise RPY subtraction
- ❌ Genesis / IsaacSim backends not started

**Key Limitation:**
Calibration algorithms require **runtime model manipulation** (modifying joint frame
placements in-place). MuJoCo, Genesis, and IsaacSim compile models to immutable internal
representations. This is the fundamental barrier to full cross-simulator calibration.
See [Backend Migration Limitations](#backend-migration-limitations) for details and
long-term paths to resolution.

**Next Steps:**
- Cross-backend validation suite (Pinocchio vs MuJoCo dynamics consistency)
- Genesis backend (GPU acceleration for identification)
- End-to-end example: identification → physical consistency projection → URDF export
- Inertia ellipsoid / CoM overlay visualization (v0.5)
- Investigate MuJoCo/Genesis upstream PRs for runtime model mutation support

---

**Document Version:** 3.2
**Last Updated:** July 12, 2026
**Maintained by:** FIGAROH Core Team
