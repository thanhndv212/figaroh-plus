---
name: figaroh-setup-calibration
description: 'Set up and run a FIGAROH geometric (kinematic) calibration task end to end — correcting joint offsets and frame transforms from measured end-effector poses. Use when the user wants to calibrate a robot, has mocap/laser-tracker measurements to fit, or asks which config keys and CSV columns a calibration needs. Covers both existing example robots and bringing your own data.'
---

# Set Up a Geometric Calibration Task

_Arguments: robot name (or "my robot"), and where the measurement data lives._

> **Paths.** `$FIGAROH_WS` is the workspace root — the directory holding the
> `figaroh` and `figaroh-examples` repos side by side. Set it once per shell:
> `export FIGAROH_WS="$(cd /path/to/your/workspace && pwd)"`.

**Prerequisite:** `figaroh-setup-env` exits 0. **Working directory for every command
below:** `$FIGAROH_WS/figaroh-examples/examples/<robot>/`.

## What this task does

Fits geometric parameters (joint offsets, link frame transforms, and the base/tool
metrology frames) so forward kinematics reproduces externally measured end-effector
poses. Inputs: a URDF + a CSV of measured poses paired with joint configurations.
Outputs: a parameter vector, a modified URDF, and a run archive.

## Step 1 — Pick the path

| Situation | Do this |
|---|---|
| Robot already under `examples/` with a `calibration.py` | Go to Step 2. Robots with calibration: `ur10`, `tiago`, `talos` (`calibration_upperbody.py`), `talos_table_contact` (`run_calibration.py`), `tiago_pro` (`run_calibration.py`). |
| Robot exists but no calibration entry point | Copy `examples/ur10/calibration.py` and `utils/ur10_tools.py`'s `UR10Calibration` as the reference; add a `tasks.calibration` block to its unified config. |
| No folder for this robot yet | Stop — run **`figaroh-setup-new-robot`** first, then come back. |

Verify before assuming:

```bash
ls examples/<robot>/{calibration.py,config,urdf,data} 2>&1
```

## Step 2 — Assemble the three required inputs

**a) URDF + meshes.** URDF at `examples/<robot>/urdf/<robot>.urdf`; its
`package://` mesh references must resolve under `figaroh-examples/models/`. Loaded as:

```python
from figaroh.tools.robot import load_robot
robot = load_robot("urdf/<robot>.urdf", package_dirs="../../models", load_by_urdf=True)
```

`../../models` is only correct when CWD is `examples/<robot>/`.

**b) Measurement CSV.** This is the input people get wrong most often. Columns are
derived from the config, not fixed — `figaroh/src/figaroh/calibration/data_loader.py`
builds them like this:

- **Marker columns**, per marker `i` (1-indexed), emitted **only for DOFs flagged
  `true` in that marker's `measurable_dof`**, in the order
  `x, y, z, phix, phiy, phiz`:
  - 6-DOF marker (`[true]*6`) → `x1,y1,z1,phix1,phiy1,phiz1` (UR10)
  - position-only (`[true,true,true,false,false,false]`) → `x1,y1,z1` (TIAGo)
  - two markers → repeat the block as `…2` columns
- **Joint columns**: one per active joint, named **exactly** as in the URDF /
  `model.names` — e.g. `shoulder_pan_joint,…,wrist_3_joint`. Not `q0..q5`.

One row = one pose. Real header from `examples/ur10/data/calibration.csv`:

```
x1,y1,z1,phix1,phiy1,phiz1,shoulder_pan_joint,shoulder_lift_joint,elbow_joint,wrist_1_joint,wrist_2_joint,wrist_3_joint
```

Positions in metres, orientations in radians (log-map / rotation-vector convention).

**c) Config.** `config/<robot>_unified_config.yaml`, unified format, with `extends:`
pointing at `../../templates/manipulator_robot.yaml` (fixed base) or
`humanoid_robot.yaml` (mobile/floating base).

## Step 3 — Fill in `tasks.calibration`

```yaml
tasks:
  calibration:
    enabled: true

    parameters:
      calibration_level: "full_params"   # or "joint_offset" for offsets only
      include_non_geometric: false
      regularization_coefficient: 0.001
      outlier_threshold: 0.02            # metres; reject residuals above this

    kinematics:
      base_frame: "universe"             # start of the calibrated chain
      tool_frame: "wrist_3_link"         # end of it — must exist in the URDF

    measurements:
      markers:
        - name: "wrist_marker"
          reference_joint: "wrist_3_joint"   # joint the marker is rigidly attached to
          measurable_dof: [true, true, true, true, true, true]  # drives CSV columns
          sensor_type: "mocap"
      poses:
        base_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # metrology → robot base guess
        tool_pose: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # flange → marker guess

    data:
      source_file: "data/calibration.csv"    # relative to examples/<robot>/
      validation_data_file: ""               # held-out set; see note below
      number_of_samples: 29
```

Key coupling to keep straight:

- `measurable_dof` ⇄ CSV columns ⇄ `reference_joint` must all agree. Changing
  `measurable_dof` changes which columns are read.
- `base_pose` / `tool_pose` are *initial guesses* for the metrology frames, in
  `[x,y,z,roll,pitch,yaw]`. UR10's script deliberately sets
  `calib_config["known_baseframe"] = False` / `known_tipframe = False` so both are
  estimated; if your setup measures them independently, leave them known.
- **`validation_data_file` must be genuinely independent data** — a separate
  acquisition, never a split of the training file. Leaving it empty falls back to the
  calibration data with a visible warning; that verdict is not an independent
  validation, so do not report it as one.

## Step 4 — Run

```bash
conda activate figaroh-dev
cd $FIGAROH_WS/figaroh-examples/examples/<robot>

python calibration.py --calibrate-only --no-plot     # fit only — start here
python calibration.py                                # full: fit → export URDF → verify → viz
python calibration.py --update-model                 # saved .npz → modified URDF → FK check
python calibration.py --viz-validation               # visually diff nominal vs modified
python calibration.py --interactive                  # choose steps
```

Useful flags (UR10 exposes all of them; check `--help` per robot):

| Flag | Effect |
|---|---|
| `--config`, `--urdf` | Override the defaults instead of editing files |
| `--validation-data <csv>` | Override `validation_data_file` for one run |
| `--no-plot` | No matplotlib windows — required for headless/CI |
| `--html-report` / `--no-html-report` | Self-contained HTML diagnostic (on by default) |
| `--geometric-calibration-yaml` | PAL `robot_state_publisher` deploy YAML, full + `>=2σ` conservative |
| `--archive` / `--no-archive` | Timestamped run archive (on by default) |
| `--asset-id`, `--operator` | Provenance for the specific physical unit |
| `--verbose` / `-v` | INFO logging (default is WARNING) |

## Step 5 — Read the outputs

```
data/calibration/calibration_results_<timestamp>.npz   # result vector + param_names
urdf/<stem>_modified_<timestamp>.urdf                  # exported calibrated model
results/runs/<asset>/calibration/<timestamp>/           # verified contents:
    config.snapshot.yaml                               #   exact config used
    provenance.json                                    #   git commit, config hash, times
    parameters.csv                                     #   identified parameters
    report.html                                        #   HTML diagnostic
    master_calibration.yaml                            #   PAL deploy YAML (full)
    master_calibration_conservative.yaml               #   same, >=2 sigma only
results/runs/index.jsonl                               # one summary line per run
```

`<asset>` comes from `robot.instance.asset_id` / `--asset-id`; with neither, runs land
under `<robot>-unspecified` (e.g. `ur10e-unspecified`). Nothing is overwritten — every
run is timestamped and archived separately. `results/` and `calibration_results_*.npz`
are gitignored, so running a task never dirties the tree.

Judge the fit by the printed block:

```
Position RMSE:    <mm>       ← the number that matters for a manipulator
Orientation RMSE: <deg>
Position MAE / Orientation MAE
Overall RMSE/MAE             ← mixes metres and radians into one norm;
                               kept for .npz backward-compat, do not quote it alone
```

Then the URDF export consistency check re-runs FK on the exported URDF over 200
samples. Position RMSE there should be **small relative to the calibration residual** —
if the export check is large, the export lost parameters, not the fit.

**Metrology frame parameters are printed but NOT written into the URDF** (`base_*`,
`pEE*`, `phiEE*`). They describe your measurement setup and must be applied in the
controller or pipeline. `frame_settings_doc()` explains the conventions.

## Step 6 — Verify

```bash
python calibration.py --calibrate-only --no-plot     # residuals sane?
python calibration.py --update-model                 # FK consistency check passes?
cd ../.. && python validate.py --robot <robot>       # nothing else regressed
```

## Failure → cause → fix

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: 'x1'` / `'<joint>'` from pandas | CSV column names do not match what the config implies | Regenerate headers per Step 2b — check `measurable_dof` and use exact URDF joint names |
| `"<name> does not exist in the file"` warning, then a crash | Same, caught one step earlier | Same fix |
| Residuals huge (metres) but solver converged | Wrong `base_pose`/`tool_pose` guess, or marker attached to a different joint than `reference_joint` says | Fix the guesses; confirm the marker's parent joint |
| Residuals fine, FK export check huge | `tool_frame` not the frame the markers actually measure | Set `tool_frame` to the real measured link |
| Config/URDF "file not found" | Ran from repo root | `cd examples/<robot>/` |
| `package://` mesh errors | Wrong CWD, so `../../models` misses | Same |
| Validation metrics suspiciously equal to training | `validation_data_file` empty → documented fallback | Supply an independent CSV, or stop calling it validation |
| Plot window blocks | Interactive backend | `--no-plot` or `MPLBACKEND=Agg` |

Anything not in this table: Re-run with `--verbose`/`-v` for INFO logging and work from the traceback; each repo's `AGENTS.md` lists the known gotchas.

## Related

`figaroh-setup-optimal` generates the measurement configurations you should collect
*before* acquiring calibration data. `figaroh-setup-identification` is the dynamics
counterpart. Library-side changes belong in `figaroh/src/figaroh/calibration/`.
