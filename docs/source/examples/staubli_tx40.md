# Staubli TX40

Fixed-base manipulator — dynamic parameter identification only (no
calibration or optimal-design scripts in this example).
[→ source folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/staubli_tx40)

## What's included

- `identification.py` — loads the TX40 URDF, initializes identification,
  runs the solve step
- `utils/staubli_tx40_tools.py` — `TX40Identification` (active joints,
  filtering/decimation choices, reporting)
- `config/staubli_tx40_unified_config.yaml` — extends
  [`manipulator_robot.yaml`](templates.md)
- `data/pos_read_data.csv`, `data/curr_data.csv` — input logs

## Run

```bash
cd examples/staubli_tx40
python identification.py
```

`--verify` and `--html-report` are on by default — a plain run already
prints a terminal quality report, writes a self-contained HTML diagnostic
report with an interactive before/after chart, and writes a machine-readable
pass/fail verdict.

Plotting and result-saving are controlled by arguments passed to
`TX40Identification.solve(...)` inside `identification.py` (e.g.
`plotting=True`, `save_results=False`, `wls=True`) rather than CLI flags —
edit the script directly to change them.

## See also

- [Identification Walkthrough](../tutorials/identification_walkthrough.md)
- Full README:
  [figaroh-examples/examples/staubli_tx40/README.md](https://github.com/thanhndv212/figaroh-examples/blob/main/examples/staubli_tx40/README.md)
