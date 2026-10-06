# Decision: Minimal additive data and result contract

- Status: Proposed
- Date: 2026-10-06
- Issue: [#54](https://github.com/thanhndv212/figaroh-plus/issues/54)
  (package [W3](https://github.com/thanhndv212/figaroh-plus/issues/35)).
  Implementation: [#55](https://github.com/thanhndv212/figaroh-plus/issues/55);
  adapter validation:
  [figaroh-examples#17](https://github.com/thanhndv212/figaroh-examples/issues/17).
- Supersedes / superseded by: None

## Context

Data enter FIGAROH as plain dictionaries and arrays, and results leave as
string-keyed dictionaries. The meaning of each array is decided in different
places, and is mostly not written down where the data travel:
- joint order and frames;
- units and clocks;
- whether an effort was measured or converted;
- which samples are valid;
- which session a sample belongs to;
- which stage produced a number, and whether it failed.

The C1/C2/D2 audits found defects of exactly this kind:
- a 3.9 s clock offset (examples#67);
- a velocity channel lagging positions (examples#20);
- a torque vector flattened sample-major against a joint-major regressor
  (comment at `base_identification.py:1115`);
- validation data overwriting the training sample count (#105).

This section inventories what exists. It is based on `devel` `efd9f69` and
figaroh-examples `main` `2c75f50`.

### 1. Dynamic inputs (identification)

The core interface is `BaseIdentification.load_trajectory_data()`. It returns
a dictionary with these keys:
- `timestamps`: (n, 1);
- `positions`, `velocities`, `accelerations`, `torques`: (n, n_joints),
  where `velocities` and `accelerations` may be `None`.

`process_data()` then fills `processed_data` with filtered and
differentiated signals. `process_torque_data()`, overridden per robot, turns
the recorded effort into joint torque.

| Adapter | Clock | Effort source → joint torque | Joint order from | Provenance kept |
|---|---|---|---|---|
| TIAGo (`tiago_tools.py`) | recorded `t`, checked against the filter rate | motor effort × `reduction_ratio` × `kmotor` (+ torso gravity term) | `active_joints`, exact column names | `trajectory_provenance` dict: files, rate, velocity lag, dropped rows, zero/duplicate effort fractions |
| UR10 (`ur10_tools.py`) | **assumed** `ts` (files have no time) | simulated torque, used as is | `q0..`, `tau1..` column names | `trajectory_provenance`: source counts, trim |
| Stäubli TX40 (`staubli_tx40_tools.py`) | **assumed** `ts`: `linspace(0, n·ts, n)` | motor current → torque through a coupled reduction matrix (wrist 5–6) | file column order | none |
| SO-101 (`so101_tools.py`) | recorded `t` (`q.csv`) | servo current or load % × `nm_per_unit`; dry-run logs detected | channel names | log metadata, not kept |
| Integration API (`integration/api.py`) | `q.csv`, `tau.csv` | torque as is | file column order | none |

Effort semantics are implicit. `raw_data["torques"]` holds whatever was
recorded (a current, a motor effort, a load percentage, or a torque), and
`processed_data["torques"]` holds the joint torque. Neither carries a unit
or an origin label. The torque vector the solver sees is stacked joint-major
(`tau.T.flatten()`), a convention stated in one code comment.

### 2. Geometric inputs (calibration)

The core loader is `calibration.data_loader.load_data(path, model,
calib_config, del_list)`. It reads a CSV with columns
`x1, y1, z1, phix1, …` per marker (only the measured DOF) and one column
per active joint, named as in the model. It returns:
- `PEE_measured`: flat, component-major
  (`x1` all samples, then `y1`, …);
- `q_measured`: (n, nq), full configurations.

Implicit conventions in this path:
- **Frame:** the measured points are in the frame the base parameters
  register (TIAGo: the Qualisys `base_frame` body).
- **Orientation:** the orientation convention of `phi*` is not stated.
- **Units:** m and rad.
- **Sample count:** written into `calib_config["NbSample"]` (#105 came from
  this).
- **Removed samples:** `del_list` removes them silently.
- **Missing column:** logged as a warning, then a `KeyError` follows.

| Adapter | Observation | Sessions / splits | Extra fields ignored by core |
|---|---|---|---|
| TIAGo mocap (`mocap_extraction.py` → CSV) | 4 points (BL, BR, TR, TL) per posture, `base_frame` body; core reads point 1 | one file per session; roles (training/validation/confirmation) live in a test and a doc | `t_start_robot`, `t_end_robot`, `marker_std_mm` |
| TALOS table contact (`talos_table_tools.py`) | no measurement: the "observation" is a zero contact gap (`PEE_measured = 0`) | `session_id` column → one table plane per session | — |
| UR10, TALOS upper body | marker positions in a world frame | single file | — |

### 3. Results

| Producer | Container | Notes |
|---|---|---|
| `BaseIdentification` | `self.result` dict, plus attributes (`phi_base`, `params_base`, `tau_identif`, `tau_noised`, `std_relative`, …) | Keys are free text with units in the name; `"rmse norm (N/m)"` is labelled N/m for a torque RMSE in N·m. Optional `"physical consistency"`, `"reconstruction"`, `"validation_metrics"` |
| `BaseCalibration` | `self.results_data` dict, `evaluation_metrics`, `LM_result`, `var_`, `std_dev` | `"PEE measured (2D array)"`, `"outlier indices"`, `"calibration config"` (the live config) |
| Both | `verify()` → `VerificationVerdict` (`_report_common.py`) | Scoped (`execution` / `prediction`), with `not_evaluated` distinct from fail; carries `metadata` = provenance, `series`, `compat` |
| `provenance.collect_run_provenance` | dict | nominal model, curated config keys, software versions, data files (by config key), asset, timestamps |
| `ResultsManager` | yaml / csv / json / npz of the result dict | serialises whatever keys exist |
| `run_archive` | `results/runs/<asset>/<task>/<stamp>_<commit>/` | `provenance.json`, `config.snapshot.yaml`, `parameters.csv`, `verdict.json`, `index.jsonl` |

Persisted-format consumers:
- `compare_report.py` (two `*_verification.json` files; checks domain,
  joint names and sample counts before comparing);
- the HTML reports;
- in examples: the golden-output hook (records `param_name`, `x`, fit RMS,
  `phi_base`, torque RMSE from live objects),
  `test_export_report.py`, the held-out protocol and the export check (live
  objects).

### 4. Where each convention is decided today

| Convention | Decided in | Carried with the data? |
|---|---|---|
| Joint order | adapter column names / `active_joints` / `actJoint_idx` | no (array position only) |
| Frames | adapter (mocap body, base frame), config `base_frame`/`tool_frame` | no |
| Units | adapter code, `kmotor`, `nm_per_unit` | no |
| Clock | adapter (recorded vs assumed `ts`), lag estimation (TIAGo) | TIAGo/UR10 `trajectory_provenance` only |
| Effort provenance | `process_torque_data` overrides | no |
| Masks | `del_list` (calibration), truncation, decimation, outlier indices | outlier indices only (as a result) |
| Splits | file per session, `validation_data_file`, `session_id` (TALOS) | no |
| Result stages | key presence (`"reconstruction"`, `"validation_metrics"`), `verify(scope=)` | partly (verdict scope) |
| Failure semantics | exceptions, warnings, silent fallbacks (validation → training data, with a warning) | verdict `not_evaluated` only |

### 5. Defects found by the inventory

- **`q0` aliasing (#125).** `load_data` and the structural selection in
  `calculate_identifiable_kinematics_model` write into `calib_config["q0"]`.
  That is the same array as `robot.q0`. A calibration therefore leaves
  `robot.q0` set to the last sample's active joints. Fits are unaffected,
  because every sample overwrites those joints. Later users of `robot.q0` in
  the same process are affected.
- **Wrong unit label.** `"rmse norm (N/m)"` labels a torque RMSE in N·m.
- **Missing CSV column.** `load_data` logs it as a warning and then fails on
  a `KeyError` with less context.
- **TX40 sample spacing.** The adapter's `linspace(0, n·ts, n)` spaces
  samples `n·ts/(n−1)` apart, not `ts`: a 1/(n−1) relative error in every
  derivative. A recorded or stated clock in `TrajectoryData` makes this
  visible.

## Decision

Add two input types and one result record, alongside the existing
dictionaries. Nothing existing changes meaning, and every current adapter
and result consumer keeps working. Dynamic and geometric inputs stay
separate types: they share only a small provenance record, not a base
class with optional fields.

### `DataSource` (shared, small)

- `files`: path → sha256.
- `adapter`: name and version.
- `split`: one of `training`, `validation`, `confirmation` or a free
  label, plus a session id.
- `notes`: free text, e.g. "velocity shifted 3 samples earlier".

### `TrajectoryData` (dynamic)

| Field | Content |
|---|---|
| `t` | (n,) s; `clock`: `recorded` or `assumed` (with the rate) |
| `joint_names` | model joint names; the order of every column below |
| `q`, `dq`, `ddq` | (n, n_j) rad or m; per signal an origin: `measured`, `derived:<method>` or `absent` |
| `effort` | (n, n_j) joint torque or force, N·m or N |
| `effort_origin` | `measured_torque`, `converted` (from current / motor effort / load, with the conversion recorded), `simulated` |
| `effort_raw` | optional (n, n_j) recorded signal and its unit, kept when `converted` |
| `mask` | (n,) bool, valid samples; removed samples stay visible |
| `source` | `DataSource` |

Rules:
- **Validation:** checked at construction: shapes, finite values, strictly
  increasing `t`, joint names present in the model.
- **Filtering stays where it is:** the type records what was done, it does
  not do it.
- **Stacking:** the joint-major stacking the solver uses becomes a method
  (`stacked_effort()`), not a convention.

### `PoseObservations` (geometric)

| Field | Content |
|---|---|
| `joint_names`, `q` | (n, n_j) active joints, rad or m |
| `points` | (n, n_points, 3) m, or `poses` (n, n_points, 6) with the orientation convention named |
| `point_names` | e.g. BL, BR, TR, TL; core may use a subset (#119) |
| `measurability` | per point, which DOF are observed |
| `frame` | the frame the observations are expressed in (e.g. `mocap:base_frame`), and the robot frame it is registered to |
| `mask`, `session` | (n,) bool and (n,) labels; replaces silent `del_list` and ad-hoc `session_id` |
| `source` | `DataSource` |

A constraint-only dataset such as TALOS contact is a `PoseObservations` with
no `points`, plus a declared constraint kind. It is not a zero
"measurement".

### `StageResult` and the run record

A result is a list of stages, each with:
- `stage`: `data`, `fit`, `validation`, `physical`, `export`;
- `status`: `ok`, `failed`, `fallback`, `not_run`;
- `reason`;
- `metrics`: name → value with a unit;
- `artifacts`: paths.

`verify()` already separates `execution` from `prediction` and
`not_evaluated` from fail; stages make the same distinction for every step,
which S1 (#63) needs.

The run record is `provenance + stages + parameters`. It is persisted as
the existing archive files plus `stages.json`, and `verdict.json` gains a
`schema_version`.

### Compatibility

- **`load_trajectory_data()`** may return a `TrajectoryData` or the old
  dictionary. The base class converts with `TrajectoryData.from_legacy()` /
  `.to_legacy()`, so existing subclasses are untouched.
- **Calibration:** `PoseObservations.from_csv()` reads today's CSV layout,
  and `.to_legacy()` returns `(PEE_measured, q_measured)` exactly as
  `load_data` does. `NbSample` is derived from it, never written back by a
  loader.
- **Result dicts:** `self.result` and `self.results_data` keep every key;
  `stages` is added. Old archives without `stages.json` read as stages
  `unknown`.
- **Unsupported:**
  - channels at different rates (the adapter resamples);
  - streaming data;
  - more than one robot per dataset;
  - unit conversion inside core (adapters convert and record it).

## Alternatives and consequences

| Option | Why not chosen |
|---|---|
| One `Dataset` type for both workflows | Dynamic and geometric inputs share almost no fields; one type would be mostly optional fields, the case the issue rules out. |
| pandas DataFrame with `attrs` | `attrs` are dropped by most operations; column naming would again be the contract. |
| xarray | New dependency for labelled arrays only. |
| Documented conventions only | Cheapest. But it doesn't stop silent misalignment; the defects above were all documented somewhere. |
| `TypedDict` | Names the keys but checks nothing at run time. |

**Cost:**
- **Maintenance:** two small dataclasses, a converter each way, and a
  stage list to maintain.
- **Subclasses:** adapters adopt them one at a time.
- **Runtime:** construction checks cost one pass over the data.

**Benefit:**
- **Conventions travel with the data:** joint order, units, effort origin,
  clock, mask and split are carried by it and validated once.
- **Reports and archives:** they can say which stage produced which
  number.

## Validation and implementation

Two concrete consumers, both in figaroh-examples (examples#17):
1. **TIAGo identification adapter (dynamic).**
   - Clock: `recorded`.
   - Velocity: `measured`, shifted, with the lag in `notes`.
   - Effort: `converted` from motor effort via `reduction_ratio`·`kmotor`,
     with `effort_raw` kept.
   - Zero-effort and duplicate channels: reported as today, via the
     `DataSource` notes.
2. **TIAGo mocap calibration adapter (geometric).**
   - Points: the four points named BL, BR, TR, TL, with `frame` =
     `qualisys:base_frame`.
   - Sessions: the held-out protocol's four, labelled with their roles, so
     the split lives in the data rather than in a test file.
   - `.to_legacy()` reproduces today's `PEE_measured` and `q_measured`
     byte for byte.

Acceptance for the implementation (#55):
- **Legacy parity:** each converter round-trips the legacy form exactly.
- **Unchanged outputs:** the golden outputs in examples are unchanged.
- **Opt-in:** an adapter that keeps returning a dictionary runs as before.
- **Stages:**
  - a validation fallback to training data is recorded as stage
    `validation` `fallback`, not only as a log warning;
  - an export rejection (#62) as `export` `failed`.
- **Fixed with it:** the `q0` aliasing (#125) and the N/m label.

Not verified yet:
- that the field list covers TALOS upper-body and UR10 calibration without
  extra fields;
- that S1's report needs nothing beyond stages and units.

Both are checked during #55 and examples#17.

### Open questions for the maintainer

1. **Effort in N·m only?** Yes, with `effort_raw` for the recorded signal.
   Or allow raw units in `effort`, with a flag?
2. **`mask` vs removal:** keep removed samples visible (proposed), or keep
   `del_list` deleting rows?
3. **Where splits live:** in the data (`DataSource.split`, proposed), or
   only in run configuration?
