# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.3] - Unreleased

### Added

- `feat(cad_constraints)`: New module `figaroh.identification.cad_constraints`
  with convex CAD-informed constraints for inertial parameter identification.
- `CADConstraints` dataclass — container for per-link mass bounds, first-moment
  (CoM) bounds, and symmetry equality constraints.
- `add_mass_bounds(cad, joint, *, m_min, m_max)` — add box constraint
  `m_min ≤ m_j ≤ m_max` for a link.
- `add_com_bounds(cad, joint, *, axis, h_min, h_max)` — add first-moment
  (linear) box constraint `h_min ≤ h_k ≤ h_max` for axis `k ∈ {x,y,z}`.
- `add_symmetry_constraints(cad, joint_a, joint_b, *, keys)` — add equality
  constraints between inertial parameters of two symmetric links.
- `bounds_from_urdf(model, ...)` — derive mass and CoM bounds automatically
  from Pinocchio model URDF inertials (CAD data source strategy).
- `build_cad_constraints_from_config(cfg, *, model)` — build `CADConstraints`
  from a YAML config sub-dict; returns `None` when empty (safe default-off).
- `apply_cad_constraints_to_problem(problem, theta, params_r, cad)` — inject
  CAD constraints into a picos `Problem` for the full-robot SDP.
- `feat(physical_consistency)`: `project_p10_lmi` gains optional `mass_bounds`
  and `com_bounds` kwargs — per-link box constraints on the SDP.
- `feat(physical_consistency)`: `project_robot_p10_lmi` gains optional
  `cad_constraints` kwarg — passes per-link bounds to `project_p10_lmi`.
- `feat(reconstruction)`: `_reconstruct_sdp` and `reconstruct_full_parameters`
  gain optional `cad_constraints` kwarg — injects CAD constraints after the
  per-joint LMI loop in the full-robot SDP.
- `feat(base_identification)`: Both `_apply_physical_consistency_if_enabled`
  and `_apply_reconstruction_if_enabled` hooks parse `cad_constraints` from
  config and pass the result to the underlying solvers.
- Example YAML comment block for `cad_constraints` in
  `figaroh-examples/examples/templates/manipulator_robot.yaml`.

### Changed

- All new parameters are optional with safe defaults — fully backward-compatible.

## [0.4.2] - Unreleased

### Added

- `feat(reconstruction)`: `ReconstructionResult.status` — outcome code (`"ok"`,
  `"solver_missing"`, `"error"`) on every result
- `feat(reconstruction)`: `ReconstructionResult.base_residual_norm` — L2 norm
  of the base constraint residual `M θ − φ_base` (scalar, always set)
- `feat(reconstruction)`: `ReconstructionResult.objective` — SDP objective value
  for Option B (`None` for nullspace)
- `feat(reconstruction)`: `BaseResult` frozen dataclass — structured container
  for `(M, phi_base, params_r)` as input to `reconstruct_full_parameters()`
- `feat(reconstruction)`: `reconstruct_full_parameters()` — unified entry point
  with `method="nullspace" | "sdp" | "auto"` dispatch
- `feat(reconstruction)`: Option B SDP (`method="sdp"`) — minimises
  `‖diag(w)(θ−θ₀)‖²` subject to `M θ = φ_base` and per-joint pseudo-inertia
  LMI `P_j ≽ 0` via Schur-complement epigraph (requires picos + cvxopt/mosek)
- `feat(reconstruction)`: `method="auto"` — silently falls back to nullspace
  when picos is not installed; `status="solver_missing"` when picos unavailable
  and `method="sdp"` was requested explicitly
- `feat(reconstruction)`: `_load_prior_from_urdf()` — builds flat prior dict
  from Pinocchio model inertias (`prior_source="urdf"`)
- `feat(reconstruction)`: `_load_prior_from_yaml()` — loads prior from a flat
  YAML file (`prior_source="yaml"`)
- `feat(identification)`: `_apply_reconstruction_if_enabled()` hook in
  `BaseIdentification._store_results()` — called after physical consistency;
  stores result under `self.result["reconstruction"]`
- `feat(identification)`: `_calculate_base_parameters()` now uses
  `QRDecomposer` directly to expose `self._M_matrix` / `self._params_r_for_recon`
  for downstream reconstruction; result dict includes `"M"` and `"params_r"` keys
- `feat(config)`: `reconstruction` block parsed from both legacy YAML
  (`get_param_from_yaml`) and unified config (`unified_to_legacy_identif_config`)
- `feat(identification/__init__)`: `BaseResult` and `reconstruct_full_parameters`
  added to public exports and `__all__`
- `tests`: 13 new unit tests in `tests/unit/test_reconstruction.py` covering
  new fields, BaseResult, prior helpers, `_p10_indices_for_joints`, nullspace
  end-to-end, auto fallback, YAML prior, and unsupported method error

### Fixed

- `fix(reconstruction)`: alternation loop in `run_reconstruction` now correctly
  unpacks `project_robot_p10_lmi()` as `(projected_p10_dict, robot_report)`;
  removes `AttributeError: tuple has no attribute 'p10_by_joint'`
- `fix(identification)`: `"projected parameters"` key (space) renamed to
  `"projected_parameters"` (underscore) in `_apply_physical_consistency_if_enabled`
  result dict for consistency with `"raw_parameters"` and Python conventions
- `fix(tools/robotcollisions)`: `print_collision_pairs()` uses `print()` instead
  of `logger.info()` so output is visible in interactive use and captured by tests
- `fix(tools/solver)`: `LinearSolver._print_solution_info()` uses `print()`
  instead of `logger.info()` so verbose output is visible in interactive use
  and captured by tests

## [0.4.1] - Unreleased

### Added

- `feat(physical-consistency)`: `is_feasible_link()` — public alias for
  `check_p10_feasibility()` matching roadmap spec naming
- `feat(physical-consistency)`: `project_link()` — public alias for
  `project_p10_lmi()` matching roadmap spec naming
- `feat(physical-consistency)`: `ProjectionReport.runtime` — per-link solve
  time (seconds) recorded via `time.perf_counter()` around `problem.solve()`
- `feat(physical-consistency)`: `weights` keyword argument to
  `project_robot_p10_lmi()` for passing a 10-element weight vector to all
  per-link projection calls
- `feat(config)`: `weights.mode: "auto" | "manual"` parsed from the
  `physical_consistency` YAML block in `_apply_physical_consistency_if_enabled`
- `feat(config)`: `weights.manual.{m, h, Sigma}` per-group manual weights built
  into a 10-element array and forwarded to `project_robot_p10_lmi`
- `feat(identification)`: `physical consistency` result dict now stores both
  `raw_parameters` (pre-projection) and `projected_parameters` (post-projection)
  as separate keys, preserving the original identified values for comparison
- `tests`: 28 unit tests in `tests/unit/test_physical_consistency.py` covering
  TC-1 through TC-12 (projection correctness, aliases, runtime, robot aggregation)
  plus 3 config-wiring tests for weights and raw/projected separation

### Fixed

- `fix(physical-consistency)`: SDP formulation in `project_p10_lmi` no longer
  uses the picos 2.x-incompatible `pc.vstack`, `pc.multiply`, or `pc.sum_squares`
  functions; replaced with element-wise objective using `pc.SquaredNorm` and
  explicit sigma-entry loop
- `fix(physical-consistency)`: solver keyword `verbose=` replaced with
  `verbosity=` (picos 2.x API); eliminates `DeprecationWarning` on every call
- `fix(physical-consistency)`: feasibility check after projection uses a
  relaxed `psd_eig_tol=-1e-8` tolerance to absorb inevitable SDP solver
  numerical noise, preventing valid projections from being reported as
  `"infeasible"` due to tiny eigenvalue violations (~1e-9)

## [0.3.1] - 2026-04-01

### Added
- `feat(physical-consistency)`: optional projection of inertial parameters onto
  a physically consistent set (`identification/physical_consistency.py`)
- `feat(reconstruction)`: reconstruction utilities — `run_reconstruction` and
  `run_option_a_reconstruction` entry points (`identification/reconstruction.py`)
- `feat(qrdecomposition)`: `get_M` / `get_M_labels` methods for retrieving the
  stored base mapping matrix after decomposition
- `feat(qrdecomposition)`: `QRResult` dataclass for structured, full-precision
  decomposition output (rank, base_indices, W_b, beta, M, phi_b, diag_R,
  cond_R1, method, …)
- `feat(qrdecomposition)`: `relative_tolerance` constructor parameter for
  scale-invariant rank detection (threshold relative to largest pivot)
- `feat(qrdecomposition)`: `get_diagnostics()` method returning rank, diag_R,
  cond_R1, and method after every decomposition
- `feat(qrdecomposition)`: `decompose()` unified entry point returning a
  `QRResult`; delegates to pivoting or double path based on `method` argument
- `feat(identification)`: enhanced parameter management with standard and
  custom parameter support in `BaseIdentification`

### Fixed
- `fix(qrdecomposition)`: `_find_rank` now correctly returns 0 for zero/empty
  matrices (previously returned row count, causing phantom base parameters)
- `fix(reconstruction)`: add missing `run_option_a_reconstruction` alias that
  blocked package-level import

### Changed
- `refactor(qrdecomposition)`: replace non-pivoted `numpy.linalg.qr` with
  `scipy.linalg.qr(..., pivoting=True)` in `_identify_base_parameters` —
  permutation-stable, deterministic basis selection
- `refactor(qrdecomposition)`: remove premature `np.around` from `beta` in all
  code paths; rounding deferred to display-only `_build_parameter_expressions`
- `refactor(qrdecomposition)`: remove fragile `assert np.allclose(W_base, W_b)`
- `refactor(qrdecomposition)`: enhanced docstrings and nominal-parameter handling
- `refactor(identification)`: patch filter configuration management in
  `BaseIdentification`
- `refactor(calibration)`: update deprecated `Frame.parent` →
  `Frame.parentJoint` for Pinocchio 3.x compatibility
- `refactor(logging)`: replace print statements with `logging` calls across
  calibration and identification modules
- `refactor(config)`: rename `load_from_yaml` → `load_param` for clarity

### Tests
- `test(qrdecomposition)`: `TestNumericalImprovements` suite — 14 new tests,
  23/23 total pass (rank-zero, relative tolerance, permutation stability ×20,
  mapping matrix property, column space, diagnostics, QRResult structure,
  full-precision beta)
- `test(reconstruction)`: unit tests for reconstruction utilities

---

## [0.3.0] - 2025-12-09

### Added
- **Advanced Linear Solver (`figaroh.tools.solver`)**: Comprehensive multivariate linear solver for robot parameter identification
  - Multiple solving methods: lstsq, QR, SVD, Ridge, Lasso, Elastic Net, Tikhonov, constrained, robust, weighted
  - Regularization support: L1 (Lasso), L2 (Ridge), Elastic Net, custom Tikhonov
  - Constraint handling: Box constraints (bounds), linear equality/inequality constraints
  - Robust regression with iterative reweighting for outlier resistance
  - Comprehensive solution quality metrics (RMSE, R², condition number, residuals)
  - Optimized for dense, large, thin matrices typical in robot dynamics
  - Full unit test coverage (18 tests) with robot identification scenarios

- **Module Reorganization**: Better code organization and separation of concerns
  - **Calibration module restructuring**:
    - `calibration/config.py`: Configuration parsing and YAML handling (624 lines)
    - `calibration/parameter.py`: Parameter management utilities (240 lines)
    - `calibration/data_loader.py`: Data loading and I/O operations (160 lines)
  - **Identification module restructuring**:
    - `identification/config.py`: Configuration parsing for identification (334 lines)
    - `identification/parameter.py`: Parameter management for identification (388 lines)
  - Maintains 100% backward compatibility through re-exports

- **BaseIdentification Enhancement**:
  - `solve_with_custom_solver()`: New method using advanced linear solver with regularization and constraints
  - Flexible solving with multiple methods and custom constraints
  - Support for physical parameter bounds (e.g., positive masses/inertias)

### Improved
- **Parameter Ordering**: Changed to Pinocchio dynamic parameter ordering for consistency
  - New order: [Ixx, Ixy, Ixz, Iyy, Iyz, Izz, mx, my, mz, m]
  - Previous order: [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
  
- **Regressor Module**: Cleaned up build_basic_regressor methods
  - Removed unused `tau` parameter for better API clarity
  - Improved method signatures and documentation

- **Code Quality**: Significant reduction in code duplication
  - `calibration_tools.py`: Reduced from ~1500 to ~630 lines (-58%)
  - `identification_tools.py`: Reduced from ~900 to ~295 lines (-67%)
  - Modular design with clear single responsibilities

### Fixed
- Parameter naming: Changed from numbered indices to parent joint names for clarity
- Regressor handling: Better support for additional columns in regressor matrices
- Results manager imports and formatting issues

### Technical Details
- **Files Changed**: 21 files
- **Lines Added**: +3,372
- **Lines Removed**: -1,604
- **Net Change**: +1,768 lines
- **Test Coverage**: All 18 new solver tests passing

## [0.2.4] - 2025-09-08

### Changed
- **Optional Dependencies**: Removed `cyipopt` from required dependencies
  - cyipopt is now truly optional and loaded only when IPOPT optimization is used
  - Users can install without cyipopt and still use all other features
  - Install cyipopt separately when needed: `pip install cyipopt` or via conda environment

### Improved
- **Installation Flexibility**: Package now installs without requiring heavy optimization dependencies
- **Error Handling**: Better error messages when optional dependencies are missing

## [0.2.3] - 2025-09-08

### Added
- **Streamlined Dependencies**: All core dependencies now available via PyPI with automatic installation
- **Lazy Loading**: Optional dependencies (cyipopt) now loaded only when needed to improve startup time
- **Enhanced Installation Notes**: Clear documentation of simplified dependency management

### Improved
- **Dependency Management**: Complete cleanup and optimization of package dependencies
  - Removed redundant `requirements.txt` and `setup.py` files
  - Consolidated all dependencies in `pyproject.toml`
  - Updated to use PyPI versions of robotics libraries (`pin` for Pinocchio)
- **Installation Process**: Significantly simplified installation with better cross-platform compatibility
- **Documentation**: Comprehensive README updates reflecting modern packaging standards
  - Combined development installation methods for clarity
  - Added official Pinocchio repository reference
  - Updated dependency documentation with descriptions
- **Performance**: Faster module loading through localized imports
- **Environment Setup**: Streamlined conda environment with minimal dependencies

### Enhanced
- **Import Strategy**: Localized cyipopt import to specific functions for better error handling
- **Error Messages**: More informative import error messages with installation instructions
- **Package Structure**: Modern Python packaging standards with pyproject.toml-only approach

### Removed
- **Redundant Files**: Eliminated `requirements.txt` and `setup.py` in favor of modern `pyproject.toml`
- **Unnecessary Dependencies**: Cleaned up unused dependencies for leaner installation

### Fixed
- **Package Name**: Corrected dependency references (e.g., proper use of `pin` for Pinocchio PyPI version)
- **Installation Conflicts**: Resolved potential conflicts between conda and pip installations

## [0.2.0] - 2025-09-05

### Added
- **Unified Configuration System**: Complete overhaul of configuration management
  - New `UnifiedConfigParser` with YAML template inheritance
  - Automatic format detection for seamless legacy compatibility  
  - Advanced parameter mapping between configuration formats
  - Comprehensive configuration validation with helpful error messages

- **Enhanced Base Classes**: Modern object-oriented workflow management
  - `BaseCalibration`: Standardized calibration workflow with unified config support
  - `BaseIdentification`: Standardized identification workflow with unified config support  
  - Automatic configuration format detection and conversion

- **Advanced Regressor Builder**: Complete redesign of regressor computation
  - `RegressorBuilder`: Object-oriented, extensible regressor construction
  - `RegressorConfig`: Configuration dataclass for regressor parameters
  - Enhanced input validation and error handling
  - Support for joint torque and external wrench modes

- **Configuration Format Mapping**: Seamless format conversion utilities
  - `unified_to_legacy_config`: Calibration parameter mapping function
  - `unified_to_legacy_identif_config`: Identification parameter mapping function
  - Perfect compatibility with existing legacy configurations

### Improved  
- **Parameter Processing**: Enhanced parameter handling with better defaults
- **Error Messages**: More informative validation and error reporting
- **Documentation**: Comprehensive updates to README and module documentation
- **Code Organization**: Better structured modules with clear responsibilities
- **Type Safety**: Added type hints throughout the codebase

### Enhanced
- **Cross-Platform Support**: Improved compatibility across operating systems
- **Input Validation**: Robust parameter validation and type checking  
- **Template System**: Flexible configuration template inheritance
- **Backward Compatibility**: Full support for existing legacy configurations

### Removed
- **quadprog dependency**: Removed unused quadprog dependency to reduce package size

### Fixed
- **Configuration Parsing**: Resolved edge cases in YAML parsing
- **Parameter Mapping**: Accurate conversion between configuration formats
- **Validation Logic**: Improved configuration validation accuracy
- **Error Handling**: Better error recovery and user feedback

### Technical Improvements
- Modern Python practices with dataclasses and type hints
- Enhanced error handling with custom exception classes
- Improved testing framework with comprehensive validation
- Better code documentation and examples

### Documentation
- Updated README with modern API examples
- Enhanced configuration system documentation  
- New API usage patterns and best practices
- Comprehensive module documentation updates

## [0.1.0] - 2025-01-25

### Added
- Initial release of FIGAROH package
- Dynamic identification algorithms for rigid multi-body systems
- Geometric calibration algorithms for serial and tree-structure robots
- Support for URDF modeling convention
- Optimal trajectory generation for dynamic identification
- Optimal posture generation for geometric calibration
- Integration with Pinocchio for efficient computations
- Support for various optimization algorithms
- Data filtering and pre-processing utilities
- Model parameter update utilities

### Features
- **Dynamic Identification**:
  - Dynamic model including friction, actuator inertia, and joint torque offset
  - Continuous optimal exciting trajectory generation
  - Multiple parameter estimation algorithms
  - Physically consistent standard inertial parameters calculation

- **Geometric Calibration**:
  - Full kinematic parameter calibration
  - Optimal calibration posture generation via combinatorial optimization
  - Support for external sensors (cameras, motion capture)
  - Non-external methods (planar constraints)

### Dependencies
- Core scientific computing: numpy, scipy, matplotlib, pandas
- Robotics: pinocchio (via conda)
- Optimization: cyipopt (via conda)
- Visualization: meshcat
- Additional: numdifftools, ndcurves, rospkg

### Documentation
- Comprehensive README with installation and usage instructions
- Examples moved to separate repository (figaroh-examples)
- API documentation structure prepared

### Notes
- Examples and URDF models moved to separate repository for clean package distribution
- Package optimized for PyPI distribution
- Supports Python 3.8+
