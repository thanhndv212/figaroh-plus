# TIAGo

Mobile manipulator (PAL Robotics TIAGo). Same four workflows as
[UR10](ur10.md), adapted for a coupled-wrist end-effector, a mobile base,
and (optionally) camera-based eye-hand calibration.
[→ source folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/tiago)

## What's included

- `calibration.py` — kinematic calibration (full pipeline flags identical
  to UR10's)
- `identification.py` — dynamic parameter identification
- `optimal_config.py` — D-optimal calibration configuration generation
  (supports both Hey5 and Schunk end-effectors via `--end-effector`)
- `optimal_trajectory.py` — exciting trajectory generation (uses IPOPT)
- `utils/tiago_tools.py` — `TiagoCalibration` / `TiagoIdentification` /
  `TiagoOptimalCalibration` / `TiagoOptimalTrajectory`
- `utils/simplified_collision_model.py` — collision geometry for
  self-collision constraints during trajectory optimization
- `config/tiago_unified_config.yaml` — extends
  [`humanoid_robot.yaml`](templates.md) (default); legacy
  `tiago_config.yaml` / `tiago_config_hey5.yaml` also present under
  `config/archive/`

This is the example new robots are scaffolded from
(`examples/create_example.sh`) — the reference to copy when adding support
for a new robot.

## Run

```bash
cd examples/tiago

# 1. Optimal calibration configurations
python optimal_config.py --end-effector hey5

# 2. Kinematic calibration (full pipeline: calibrate → plot → save → export → viser viz)
python calibration.py
python calibration.py --viz-validation --model urdf/tiago_48_schunk_modified_20260127.urdf

# 3. Optimal identification trajectory (requires IPOPT)
python optimal_trajectory.py

# 4. Dynamic identification (html-report + verify on by default)
python identification.py
```

All saved files are timestamped: `data/calibration/calibration_results_{ts}.npz`,
`urdf/tiago_48_schunk_modified_{ts}.urdf`.

## Outputs

Archived under `results/runs/tiago-<asset>/{calibration,identification}/<timestamp>/`
by default (or `results/` with `--no-archive`) — HTML report, verification
JSON, and (for calibration) a `compare.html` for offline before/after
comparison. See [Reporting & Verification](../reporting_and_verification.md).

## See also

- [Calibration Walkthrough](../tutorials/calibration_walkthrough.md) /
  [Identification Walkthrough](../tutorials/identification_walkthrough.md)
- Full README with class hierarchy, troubleshooting, and expected accuracy:
  [figaroh-examples/examples/tiago/README.md](https://github.com/thanhndv212/figaroh-examples/blob/main/examples/tiago/README.md)
