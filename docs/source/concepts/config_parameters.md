# Configuration Parameter Reference

[Configuration System](configuration.md) explains the two config *formats*
(unified vs. legacy) and how they're parsed. This page is the field-by-field
glossary for what each parameter in the `calibration:` and `identification:`
sections actually *means* — units, which pipeline step consumes it, and its
unified-format equivalent where one exists.

Field names below match the **legacy flat format**, since that's what you
see verbatim in a config file (`get_param_from_yaml` /
`identification.config.get_param_from_yaml`). The **Unified path** column
gives the equivalent key under `tasks.calibration.*` /
`tasks.identification.*` in the unified format — `unified_to_legacy_config`
and `unified_to_legacy_identif_config` (both in `figaroh/src/figaroh/*/config.py`)
convert one to the other, so the two formats agree on meaning even though
the layout differs.

## `calibration:` section

| Field | Meaning | Unified path |
|---|---|---|
| `calib_level` | Kinematic-error model: `joint_offset` (one rotational offset per joint) or `full_params` (full 6-DOF xyz+rpy DH-style placement error per joint) | `parameters.calibration_level` |
| `non_geom` | Also identify joint elasticity (gravity-torque-driven compliance, one term per active joint) on top of the geometric model | `parameters.include_non_geometric` |
| `base_frame` | URDF frame name at the **start** of the kinematic chain | `kinematics.base_frame` |
| `tool_frame` | URDF frame name at the **end** of the chain (usually the gripper/tool mount) | `kinematics.tool_frame` |
| `base_to_ref_frame` | *(optional, eye-hand/camera mode only)* A frame reachable from `base_frame` via a **known** transform (e.g. a camera housing anchor) | `eye_hand.camera_frame` (requires `eye_hand.enabled: true`) |
| `ref_frame` | *(optional, eye-hand/camera mode only)* The frame the chain restarts from after the known `base_to_ref_frame` segment; the *unknown* camera/anchor pose being estimated sits between `ref_frame` and the true world frame | `eye_hand.reference_frame` (requires `eye_hand.enabled: true`) |
| `markers[].ref_joint` | Joint the external marker is rigidly mounted near; its offset from `tool_frame` is what gets estimated (`pEEx/y/z`, `phiEEx/y/z`) | `measurements.markers[].reference_joint` — written by convention, but **not currently read back** by `_extract_marker_info` (only `measurable_dof`, from the first marker, is consumed) |
| `markers[].measure` | 6-DOF boolean mask `[x, y, z, roll, pitch, yaw]` — which axes the external sensor (mocap/vision/contact) actually reports for that marker | `measurements.markers[].measurable_dof` |
| `free_flyer` | `True` if the robot's base itself is unconstrained/floating in the model (mobile-base robots with an unknown per-sample base pose) | `kinematics.free_flying_base` |
| `base_pose` | Initial guess `[x, y, z, roll, pitch, yaw]` (m, rad) for the `base_frame` → world (or anchor → camera, in eye-hand mode) transform being estimated | `measurements.poses.base_pose` |
| `tip_pose` | Initial guess `[x, y, z, roll, pitch, yaw]` (m, rad) for the `tool_frame` → marker transform being estimated | `measurements.poses.tool_pose` |
| `coeff_regularize` | L2 regularization weight applied to the non-base/non-tip parameters in the least-squares cost, to keep identified offsets small and the problem well-conditioned | `parameters.regularization_coefficient` |
| `outlier_eps` | Residual distance (**meters**) above which a sample is treated as an outlier and iteratively dropped before refitting | `parameters.outlier_threshold` |
| `data_file` | Path to the CSV of recorded `(joint configuration, measured marker pose)` samples | `data.source_file` |
| `sample_configs_file` | Optional: path cross-referencing which planned/optimal configuration each `data_file` row corresponds to. Traceability only — not required for the fit itself | `data.sample_configurations_file` |
| `nb_sample` | Number of samples expected from `data_file` | `data.number_of_samples` |

!!! info "Why the camera pose is a free parameter, not a small offset"
    Everything else in `full_params`/`joint_offset` calibration treats the
    URDF as *approximately* right — each joint gets a small correction
    around its known nominal placement. An add-on eye-in-hand/eye-to-hand
    camera is different: its housing-to-robot mounting pose usually isn't
    known precisely from CAD/assembly at all, so it can't be treated as a
    small perturbation. `base_to_ref_frame`/`ref_frame` handle this by
    inserting the camera as a genuine 6-DOF unknown *in the middle of the
    chain* (`base_pose` is a real initial guess, not a near-zero one)
    rather than folding it into the same small-offset parameters as every
    other joint.

    That's also what makes eye-hand calibration a specialized, harder case:
    this unknown camera-pose block is estimated *simultaneously*, from the
    same measurements, as the per-joint offsets of every joint the camera
    observes through — "camera is off by X" and "joint N is off by Y" can
    produce very similar marker-pose residuals, which can leave the problem
    under-determined or poorly conditioned. `coeff_regularize` and
    deliberately exciting poses that vary joint loading/configuration
    widely (`sample_configs_file`, D-optimal design a la
    `tiago_pro/generate_optimal_configs.py`) are the two practical levers
    against this. A plain single-anchor unknown base frame — `base_frame`
    set to `universe`/`base_footprint` with no `base_to_ref_frame`, as in
    TiagoPro's `tiago_pro_calibration_config.yaml` (`known_baseframe=False`
    in `run_calibration.py`) — doesn't have this coupling, because nothing
    downstream of that single anchor is *itself* another unknown sensor
    mount; it's one 6-DOF unknown at the root, not one inserted mid-chain
    and re-estimated jointly with everything past it.

!!! note "Eye-hand / camera calibration now has a unified-format equivalent"
    `base_to_ref_frame`/`ref_frame` map to
    `tasks.calibration.eye_hand.{camera_frame,reference_frame}` (with
    `eye_hand.enabled: true`) — the unified schema template
    (`templates/base_robot_config.yaml`) already reserved this section;
    `unified_to_legacy_config` just didn't read it. Fixed, so eye-hand
    configs like `examples/tiago/config/tiago_config_hey5.yaml` are no
    longer stuck in the legacy format — see
    [Migrating from legacy to unified format](#migrating-from-legacy-to-unified-format)
    below. (`free_flyer`/`kinematics.free_flying_base` had a similar
    wrong-key bug — the converter read `parameters.free_flyer`, a key the
    template never defines — fixed the same way.)

## `identification:` section

### `robot_params`

| Field | Meaning | Unified path |
|---|---|---|
| `q_lim_def`, `dq_lim_def`, `ddq_lim_def`, `tau_lim_def` | Joint position/velocity/acceleration/torque limit margins | `joints.joint_limits.{position,velocity,acceleration,torque}` — **parsed but not currently consumed anywhere in the identification pipeline** (kept for schema/future use) |
| `fv` | Viscous friction coefficient per joint (N·m·s/rad) | `mechanics.friction_coefficients.viscous` — used only if `has_friction` |
| `fs` | Coulomb/static friction coefficient per joint (N·m) | `mechanics.friction_coefficients.static` — used only if `has_friction` |
| `Ia` | Reflected actuator (motor + gearbox) inertia per joint (kg·m²) | `mechanics.actuator_inertias` — used only if `has_actuator_inertia` |
| `offset` | Constant joint torque offset per joint (N·m) — electrical/sensor bias term | `mechanics.joint_offsets` — used only if `has_joint_offset` |
| `Iam6`, `fvm6`, `fsm6` | Same three quantities as `Ia`/`fv`/`fs`, but for the shared "6th motor" of a differential-coupled wrist (two motors jointly drive the last two wrist axes, e.g. Staubli TX40) | `coupling.{Iam6,fvm6,fsm6}` — used only if `has_coupled_wrist` |
| `reduction_ratio` | Per-joint gear reduction ratio (motor turns per joint turn); sign encodes actuation direction | `mechanics.reduction_ratios` |
| `ratio_essential` | Threshold (%) for the "essential parameters" selection technique (Pham et al., 1995) | `mechanics.ratio_essential` — **documented intent only; not currently wired into any solver in this codebase** |

### `problem_params`

| Field | Meaning | Unified path |
|---|---|---|
| `is_external_wrench` | Include a measured force/torque sensor wrench as a regressor input | `problem.include_external_forces` |
| `is_joint_torques` | Regress against joint torque signals (standard inverse-dynamics identification; almost always `True`) | `problem.use_joint_torques` |
| `force_torque` | Which F/T channels to use (e.g. `['Fx','Fy','Fz']` or `['All']`) — only applied if `is_external_wrench` is `True` | `problem.force_torque_sensors` |
| `external_wrench_offsets` | Also identify a constant bias offset on the F/T sensor reading | `problem.external_wrench_offsets` |
| `has_friction` | Include the `fv`/`fs` columns in the regressor | `problem.model_components.friction` |
| `has_actuator_inertia` | Include the `Ia` column in the regressor | `problem.model_components.actuator_inertia` |
| `has_joint_offset` | Include the `offset` column in the regressor | `problem.model_components.joint_offset` |
| `is_static_regressor` | Include the friction + offset ("static") regressor block | *(no unified equivalent found — always derived)* |
| `is_inertia_regressor` | Include the rigid-body inertial-parameter regressor block | *(no unified equivalent found — always derived)* |
| `has_coupled_wrist` | Last two wrist joints share differential actuation → adds the `Iam6`/`fvm6`/`fsm6` terms and a coupling transform to the regressor | `coupling.has_coupled_wrist` |
| `embedded_forces` | Express the external wrench in a body-embedded (tool) frame rather than a fixed/world frame | *(no unified equivalent found)* |
| `active_joints` | *(optional)* Restrict identification to this explicit subset of joint names, instead of every active joint in the chain | `joints.active_joints` |

### `processing_params`

| Field | Meaning | Unified path |
|---|---|---|
| `cut_off_frequency_butterworth` | Low-pass Butterworth filter cutoff frequency (Hz) applied to position/velocity/acceleration/torque signals before regression | `signal_processing.cutoff_frequency` |
| `ts` | Sample period (**seconds**) of the recorded data | `signal_processing.sampling_frequency` *(unified format specifies the rate directly, in Hz, rather than the period)* |

!!! warning "`ts` → `nb_samples` is a rate, not a count"
    Despite the name, the legacy parser computes
    `identif_config["nb_samples"] = int(1 / ts)` — i.e. an implied
    samples-per-second rate, not the number of rows in your data file. Don't
    be misled by the field name; it isn't used as a row count anywhere
    downstream.

### `tls_params` (optional — known-payload identification)

| Field | Meaning |
|---|---|
| `mass_load` | Known mass (kg) of a test payload attached during a supplementary run, used to refine the identified inertial parameters via total-least-squares |
| `which_body_loaded` | Index of the link/body in the regressor that carries that payload |
| `sync_joint_motion` | **Parsed by nothing** — not even read into `identif_config` by `get_param_from_yaml`. Effectively decorative today. |

### `trajectory_params` (optional — used by `optimal_trajectory.py`, not `identification.py`)

| Field | Meaning |
|---|---|
| `n_wps` | Number of waypoints in the generated exciting trajectory |
| `freq` | Trajectory sample frequency (Hz) |
| `t_s` | Time window between waypoints (s) |
| `soft_lim` | Joint-limit safety margin/discount applied during trajectory search |
| `max_attempts` | Maximum attempts to find a feasible trajectory before giving up |

## Migrating from legacy to unified format

Both formats keep working side by side, but loading a legacy config now
emits a `DeprecationWarning` (plus a `logger.warning`) from
`figaroh.calibration.config.get_param_from_yaml` /
`figaroh.identification.config.get_param_from_yaml` — the single parser
every legacy-format caller funnels through, whether via
`BaseCalibration`/`BaseIdentification.load_param`, `figaroh.utils.config_parser`,
or a custom script calling the legacy parser directly.

`figaroh.utils.config_migration` converts a legacy config to unified format
automatically, using the exact field mapping in the tables above:

```bash
python -m figaroh.utils.config_migration \
    --input examples/tiago/config/tiago_config_hey5.yaml \
    --output examples/tiago/config/tiago_config_hey5_unified.yaml \
    --urdf urdf/tiago_48_hey5.urdf   # optional: runs a round-trip self-check
```

- `--template {base,manipulator,humanoid}` (default `base`) picks which
  template the output `extends:` — `base` (extending
  `base_robot_config.yaml` directly) is the safest default since it doesn't
  risk silently inheriting a specialized template's default the legacy file
  didn't actually have.
- Passing `--urdf` runs the output back through the real
  `unified_to_legacy_config`/`unified_to_legacy_identif_config` and diffs it
  against the original legacy values — the same self-check mechanism that
  caught the `free_flyer`/`eye_hand` bugs above. A migration that reports
  `MISMATCHES` should not be trusted without investigating why.
- Fields with no unified-format consumer today (`is_static_regressor`,
  `is_inertia_regressor`, `sync_joint_motion`, the unused joint-limit
  fields — see the tables above) are preserved under the output's `custom:`
  section rather than silently dropped, even though nothing reads them back
  yet.
- The tool only converts the file; it doesn't rewrite the example script
  that loads it, or verify the migrated config still produces the same
  calibration/identification *results* on real data — treat that as a
  required follow-up before switching an example over, the same way the
  `calc_updated_fkm` merge was validated against real TIAGo/TIAGo
  Pro/TALOS/UR10 data before landing.

## Next steps

- [Configuration System](configuration.md) — format structure and template
  inheritance.
- [Calibration Walkthrough](../tutorials/calibration_walkthrough.md) and
  [Identification Walkthrough](../tutorials/identification_walkthrough.md) —
  these fields in context, end to end.
