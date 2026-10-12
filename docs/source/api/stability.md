# Public API and stability

This page is the contract for 1.0. It states which names you can build on,
which are experimental, which are internal, and how a public name is retired.
The `__all__` lists in the source match it; a unit test
(`tests/unit/test_public_api.py`) fails if they drift apart.

## What "public" means

A name is public when it appears in the `__all__` of its package or of a
module this page lists. Anything else is not covered, even when it is
importable and even when [the API reference](index.md) renders it:

- a name that starts with an underscore;
- a module that starts with an underscore;
- a module or name that is not listed here, such as
  `figaroh.calibration.config` or `figaroh.identification.selection`.

Public names follow [semantic versioning](https://semver.org/): a breaking
change to a supported name needs a new major version. Experimental names may
change in any minor release.

## Top-level packages

`import figaroh` imports every package below eagerly, so
`figaroh.tools.robot` and friends work after a bare import. The status column
is the status of the package as a whole; the sections below give the detail.

| Package | Status |
|---|---|
| `figaroh.tools` | Supported (`robotvisualization` is experimental) |
| `figaroh.calibration` | Supported |
| `figaroh.identification` | Supported |
| `figaroh.measurements` | Experimental |
| `figaroh.utils` | Supported |
| `figaroh.visualisation` | Experimental |
| `figaroh.optimal` | Supported |
| `figaroh.data` | Supported |
| `figaroh.backends` | Supported (Pinocchio); MuJoCo is experimental |
| `figaroh.integration` | Experimental |

## Supported entry points

Each line names a package or module and the complete list of its public
names. For a package, a name that is itself a module (such as `calibration_tools`)
means that module is public too and is listed on its own line.

- `figaroh.calibration`: `BaseCalibration`, `calibration_tools`, `data_loader`, `parameter`
- `figaroh.calibration.calibration_tools`: `get_param_from_yaml`, `unified_to_legacy_config`, `get_sup_joints`, `get_joint_offset`, `get_fullparam_offset`, `add_base_name`, `add_pee_name`, `add_eemarker_frame`, `read_config_data`, `load_data`, `get_idxq_from_jname`, `cartesian_to_SE3`, `xyzquat_to_SE3`, `get_rel_transform`, `get_rel_kinreg`, `get_rel_jac`, `initialize_variables`, `calc_updated_fkm`, `update_joint_placement`, `apply_joint_offset`, `random_joint_configuration`, `estimate_frames_closed_form`, `measurement_jacobian`, `select_identifiable_parameters`, `drop_calibration_parameters`, `calculate_kinematics_model`, `calculate_identifiable_kinematics_model`, `calculate_base_kinematics_regressor`
- `figaroh.calibration.parameter`: `get_joint_offset`, `get_fullparam_offset`, `add_base_name`, `add_pee_name`, `add_eemarker_frame`, `FULL_PARAMTPL`, `JOINT_OFFSETTPL`, `ELAS_TPL`, `EE_TPL`, `BASE_TPL`
- `figaroh.calibration.data_loader`: `read_config_data`, `load_data`, `get_idxq_from_jname`

- `figaroh.identification`: `identification_tools`, `parameter`, `physical_fit`, `BaseIdentification`, `BaseResult`, `ReconstructionResult`, `prior_vector_from_dict`, `reconstruct_from_base`, `reconstruct_full_parameters`, `reconstruct_theta_r`, `run_reconstruction`, `run_option_a_reconstruction`, `ProjectionReport`, `RobotProjectionReport`, `check_p10_feasibility`, `is_feasible_link`, `p10_by_joint_from_param_dict`, `param_dict_with_p10_by_joint`, `project_link`, `project_p10_lmi`, `project_robot_p10_lmi`, `pseudo_inertia_matrix_from_p10`, `CADConstraints`, `add_mass_bounds`, `add_com_bounds`, `add_symmetry_constraints`, `bounds_from_urdf`, `build_cad_constraints_from_config`
- `figaroh.identification.identification_tools`: `base_param_from_standard`, `relative_stdev`, `index_in_base_params`, `weigthed_least_squares`, `calculate_first_second_order_differentiation`, `low_pass_filter_data`, `get_param_from_yaml`
- `figaroh.identification.parameter`: `reorder_inertial_parameters`, `add_standard_additional_parameters`, `add_custom_parameters`, `get_standard_parameters`, `get_parameter_info`
- `figaroh.identification.physical_fit`: `FixedExtras`, `PhysicalPolicy`, `ComparatorProblem`, `SolveRecord`, `build_problem`, `solve_exact_reconstruction`, `solve_direct_effort_fit`, `solve_per_link_projection`, `diagnose_exact`

- `figaroh.optimal`: `BaseOptimalCalibration`, `BaseOptimalTrajectory`, `BaseTrajectoryIPOPTProblem`

- `figaroh.data`: `DataSource`, `Session`, `file_sha256`, `Protocol`, `ProtocolSession`, `TrajectoryData`, `PoseObservations`, `JOINT_TORQUE`, `JOINT_FORCE`, `MOTOR_CURRENT`, `MOTOR_TORQUE`, `LOAD_FRACTION`, `DRIVE_KINDS`

- `figaroh.backends`: `DynamicsBackend`, `get_backend`, `list_backends`, `get_backend_info`
- `figaroh.backends.pinocchio`: `PinocchioBackend`

- `figaroh.tools`: `robot`, `randomdata`, `regressor`, `qrdecomposition`, `robotvisualization`, `robotcollisions`, `robotipopt`, `urdf_exporter`, `geometric_calibration_export`, `export_validation`, `report`, `identification_report`, `compare_report`, `run_archive`, `stages`, `solver`
- `figaroh.tools.robot`: `Robot`, `create_robot`, `load_robot`, `get_available_loaders`, `list_available_robots`
- `figaroh.tools.regressor`: `RegressorConfig`, `RegressorBuilder`, `build_regressor_basic`, `eliminate_non_dynaffect`, `get_index_eliminate`, `build_regressor_reduced`, `build_total_regressor_current`, `build_total_regressor_wrench`
- `figaroh.tools.qrdecomposition`: `QRDecomposer`, `QRResult`, `QR_pivoting`, `double_QR`, `get_baseParams`, `get_baseIndex`, `build_baseRegressor`, `cond_num`, `redistribute_min_norm`, `propagate_covariance_min_norm`
- `figaroh.tools.solver`: `LinearSolver`, `solve_linear_system`
- `figaroh.tools.urdf_exporter`: `export_urdf`, `frame_settings_doc`
- `figaroh.tools.geometric_calibration_export`: `build_geometric_calibration`, `export_geometric_calibration_yaml`
- `figaroh.tools.export_validation`: `PoseDelta`, `FkConsistencyResult`, `PoseError`, `URDFComparison`
- `figaroh.tools.run_archive`: `compute_run_dir`, `archive_run`
- `figaroh.tools.stages`: `StageResult`, `record_stage`, `stages_as_dicts`
- `figaroh.tools.report`: `generate_calibration_report`
- `figaroh.tools.identification_report`: `generate_identification_report`
- `figaroh.tools.compare_report`: `generate_compare_page`
- `figaroh.tools.randomdata`: `generate_waypoints`, `generate_waypoints_fext`, `get_torque_rand`
- `figaroh.tools.robotcollisions`: `add_non_adjacent_pairs`, `CollisionManager`, `CollisionWrapper`
- `figaroh.tools.robotipopt`: `IPOPTConfig`, `BaseOptimizationProblem`, `RobotIPOPTSolver`, `TrajectoryOptimizationProblem`, `create_trajectory_solver`

- `figaroh.utils`: `config_parser`, `error_handling`, `ResultsManager`, `FigarohExampleError`, `CalibrationError`, `IdentificationError`, `handle_calibration_errors`, `handle_identification_errors`, `validate_input_data`, `setup_example_logging`
- `figaroh.utils.config_parser`: `UnifiedConfigParser`, `create_task_config`, `parse_configuration`, `get_param_from_yaml`, `is_unified_config`, `ConfigurationError`
- `figaroh.utils.error_handling`: `FigarohExampleError`, `RobotInitializationError`, `ConfigurationError`, `DataProcessingError`, `CalibrationError`, `IdentificationError`, `ValidationError`, `validate_robot_config`, `validate_trajectory_data`, `validate_numeric_range`, `validate_robot_initialization`, `validate_input_data`, `handle_calibration_errors`, `handle_identification_errors`, `safe_execute`, `setup_example_logging`, `ErrorContext`

Notes on judgement calls:

- `figaroh.calibration.calibration_tools` is public because the examples
  build their calibration pipelines on it. Its `get_param_from_yaml` accepts
  a unified calibration config; the legacy flat format it also reads is
  deprecated (see below), the function is not.
- `figaroh.identification.physical_fit` is the module behind
  `select_stage: physical_fit`. Its functions are public; its private
  helpers are not.
- The `figaroh.tools` modules `randomdata`, `robotcollisions` and
  `robotipopt` are public helpers kept for the optimal-design classes.

## Experimental areas

These are importable and tested, but they may change or move in a minor
release, and a call can raise `NotImplementedError`.

- **MuJoCo backend and `from_mjcf`.** `MuJoCoBackend` needs the optional
  `mujoco` package. `RobotIdentificationSystem.from_mjcf` raises
  `NotImplementedError`: MJCF loading is not supported yet. Genesis and
  Isaac Sim backends are not shipped; `get_backend` only names them.
- **`figaroh.integration`.** The one-line `RobotIdentificationSystem`
  workflow.
- **`figaroh.visualisation` and `figaroh.tools.robotvisualization`.**
  Meshcat and plotting helpers.
- **`figaroh.measurements`.** A measurement container that no shipped
  workflow uses.
- **Log-Cholesky fits (LC1).** The log-Cholesky physical fit is not part of
  the library: the feasibility spike was a no-go for 0.7.0 (D5, issue 155).
  If it ships, it ships as an opt-in, experimental method.

Names in these modules, as they stand today:

- `figaroh.integration`: `RobotIdentificationSystem`, `IdentificationResult`
- `figaroh.visualisation`: `colors`, `MeshcatVisualizer`
- `figaroh.visualisation.colors`: `rgb2int`, `material`, `red`, `blue`, `green`, `yellow`, `magenta`, `cyan`, `white`, `black`, `grey`, `colormap`
- `figaroh.backends.mujoco`: `MuJoCoBackend`
- `figaroh.tools.robotvisualization`: `VisualizationConfig`, `RobotVisualizer`, `place`, `display_COM`, `display_axes`, `rotation_matrix_from_vectors`, `display_force`, `display_bounding_boxes`, `display_joints`
- `figaroh.measurements`: `Measurement`

## Internal modules

These support the public API. Do not import them from outside figaroh; they
may change or disappear in any release.

- Any module or name that starts with an underscore, for example
  `figaroh.identification._physical_comparator` and
  `figaroh.tools._report_common`.
- `figaroh.tools.load_robot` (use `figaroh.tools.robot`, which re-exports
  `load_robot`), `figaroh.tools.provenance`.
- `figaroh.calibration.config`, `figaroh.calibration.estimation` and
  `figaroh.identification.config`, `figaroh.identification.selection`,
  `figaroh.identification.physical_consistency`,
  `figaroh.identification.cad_constraints`,
  `figaroh.identification.reconstruction` and the other modules not listed
  above. Their main names are re-exported by the packages that list them.
- `figaroh.optimal.config`, `figaroh.optimal.base_parameter`,
  `figaroh.optimal.contraints`.
- `figaroh.utils.cubic_spline`, `figaroh.utils.config_migration` (run it as
  `python -m figaroh.utils.config_migration`; its functions are internal).
- `figaroh.data.protocol`, `figaroh.data.source`,
  `figaroh.data.observations` and `figaroh.data.trajectory`: import their
  names from `figaroh.data`.

## Deprecated

These still work, warn, and are scheduled for removal in 1.0 (issue 185):

- the legacy flat configuration format (since 0.4.6); migrate with
  `python -m figaroh.utils.config_migration`;
- `get_param_from_yaml_with_warning` in the calibration and identification
  config modules (since 0.2.0);
- `figaroh.utils.config_parser.get_calibration_param_from_yaml` and
  `get_identification_param_from_yaml` (since 0.2.0): use
  `get_param_from_yaml` with `task_type`;
- a non-zero `regularization_coefficient` / `coeff_regularize` (since 0.6.0).

## Deprecation policy

- A public name is deprecated by emitting a `DeprecationWarning` that names
  the replacement. It keeps working for at least one minor release before
  removal.
- After 1.0, a public name is removed only in a major release.
- Every deprecation and removal has a migration note in the
  [CHANGELOG](../further_reading/changelog.md).
- Experimental names and internal modules are outside this policy.

## Python versions

`requires-python` is `>=3.10`, following Pinocchio's own support window.
CI tests Python 3.10 (lowest), 3.12 (main) and 3.14 (highest).
