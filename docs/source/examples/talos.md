# TALOS

Humanoid robot (PAL Robotics TALOS) — kinematic calibration of the
torso/arm chain.
[→ source folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/talos)

## What's included

- `calibration_upperbody.py` — all-in-one entry point: calibration, URDF
  export, and visual validation, selected by flag
- `update_model.py` — thin shim delegating to
  `calibration_upperbody.py --update-model`
- `utils/talos_tools.py` — `TALOSCalibration`, a
  `figaroh.calibration.base_calibration.BaseCalibration` specialization for
  TALOS's torso/arm kinematic chain
- `config/talos_unified_config.yaml` — extends
  [`humanoid_robot.yaml`](templates.md)

## Run

```bash
cd examples/talos

python calibration_upperbody.py                # full pipeline
python calibration_upperbody.py --calibrate-only
python calibration_upperbody.py --update-model
python calibration_upperbody.py --viz-validation
python calibration_upperbody.py --interactive
```

## Outputs

`results/calibration_report.html`; timestamped calibration results under
`data/calibration_results_*.npz` and modified URDFs under `urdf/`.

## See also

- [Calibration Walkthrough](../tutorials/calibration_walkthrough.md)
- Full README:
  [figaroh-examples/examples/talos/README.md](https://github.com/thanhndv212/figaroh-examples/blob/main/examples/talos/README.md)
