---
name: figaroh-setup-env
description: 'Bootstrap and verify a working FIGAROH environment before any task runs. Use on a fresh machine, at the start of a session, or whenever "import figaroh" fails, cyipopt/picos is missing, meshes do not load, or library edits appear to have no effect. Ships a read-only doctor script that names the exact remedy for each failure.'
---

# FIGAROH Environment Setup

_Arguments: none needed. Optionally pass a workspace root if it is not
`$FIGAROH_WS`._

> **Paths.** `$FIGAROH_WS` is the workspace root — the directory holding the
> `figaroh` and `figaroh-examples` repos side by side. Set it once per shell:
> `export FIGAROH_WS="$(cd /path/to/your/workspace && pwd)"`.

## When to Use

- First contact with this workspace, or a new machine.
- `ModuleNotFoundError: figaroh`, `No module named 'cyipopt'`, mesh/package-dir errors.
- Library edits in `figaroh/src/` seem to have no effect (→ shadowed by a
  site-packages copy).
- Before invoking `figaroh-setup-calibration`, `-identification`, `-optimal`, or
  `-new-robot`.

## Step 1 — Run the doctor (always start here)

```bash
bash "$FIGAROH_WS"/figaroh/skills/figaroh-setup-env/scripts/doctor.sh
```

Read-only; mutates nothing. Exit 0 = ready, 1 = blocked. It checks workspace layout,
the conda env, `import figaroh` **and whether it resolves to the local source tree**,
core deps, each optional dep (with the single task it blocks), and that the UR10
reference example is intact. Every FAIL prints its own `fix:` line — apply those, then
re-run until exit 0. Do not proceed to a task skill while it exits 1.

`--ws <root>` and `--env <name>` override the defaults.

## Step 2 — The environment contract

**`figaroh-dev` is the only supported environment.** Python 3.12, conda-forge, defined
by `figaroh/environment.yml`:

```bash
conda env create -f $FIGAROH_WS/figaroh/environment.yml
conda activate figaroh-dev
```

`environment.yml` already does `pip install -e .`, so creating the env gives you the
editable install of the local library. To repair or refresh it:

```bash
conda run -n figaroh-dev pip install -e $FIGAROH_WS/figaroh
conda env update -n figaroh-dev -f $FIGAROH_WS/figaroh/environment.yml
```

Two ways to run commands — both fine, be consistent within a session:

```bash
conda activate figaroh-dev && python calibration.py     # interactive
conda run -n figaroh-dev python calibration.py          # scripted / CI
```

**Trap:** `figaroh-examples/environment.yml` declares an env named `figaroh-examples`.
It is stale and unused — ignore it. The real env lives in the sibling `figaroh/` repo.

## Step 3 — Optional dependencies, and exactly what each one gates

Install these only when the task needs them; none of them block calibration or
identification.

| Package | Gates | Install |
|---|---|---|
| `cyipopt` | `figaroh.optimal`, `tools/robotipopt.py`, `optimal_trajectory.py` | `conda install -n figaroh-dev -c conda-forge cyipopt` — **conda only**, not pip-installable in practice. This is the entire reason the conda env exists. |
| `picos` + `cvxopt` | `identification.physical_consistency` (SDP projection, default-off) | `conda run -n figaroh-dev pip install picos cvxopt` |
| `viser` | `--viz-validation`, interactive URDF comparison | `conda run -n figaroh-dev pip install viser` |
| `meshcat` | 3D preview helpers | already a core dependency |

## Step 4 — Confirm with the reference example

The install is not proven until a real task runs. Use UR10 as the bisect point:

```bash
conda activate figaroh-dev
cd $FIGAROH_WS/figaroh-examples/examples/ur10
python calibration.py --calibrate-only --no-plot
```

Expect post-calibration position/orientation RMSE and a saved
`data/calibration/calibration_results_<timestamp>.npz`. **If this passes, the
environment is not the problem** — route any later failure to the user's model,
config, or data.

Broader gate, when you want the whole suite:

```bash
cd $FIGAROH_WS/figaroh-examples
python validate.py --quick      # skips IPOPT-heavy scripts
python validate.py              # full: pytest + every example script
```

## Known-benign noise — do not chase these

- `RuntimeWarning: to-Python converter for std::__1::shared_ptr<ndcurves::curve_abc…>
  already registered; second conversion method ignored` on `import figaroh`. Harmless
  Boost.Python double-registration from `ndcurves`/`pinocchio`.
- `conda run` appends a trailing blank line to captured stdout — `... | tail -1`
  silently yields the empty line instead of your value. Grep for a sentinel instead
  (this is why `doctor.sh` has a `probe()` helper).
- IPOPT timeouts in `validate.py` are reported separately from failures and are
  expected, not bugs.
- Pre-existing test failures in `figaroh/`: `test_print_collision_pairs*`,
  `test_verbose_output` (logging-related). Not regressions.

## Failure → cause → fix

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: figaroh` | env not active, or no editable install | `conda activate figaroh-dev`; `pip install -e <ws>/figaroh` |
| Library edits have no effect | `figaroh` resolving to site-packages, shadowing `src/` | doctor Step 3 flags this; `pip install -e <ws>/figaroh` |
| `No module named 'cyipopt'` | pip-installed attempt | conda-forge only (table above) |
| Mesh / `package://` resolution errors | wrong CWD | `cd examples/<robot>/`; `package_dirs="../../models"` only resolves from there |
| Config or URDF "file not found" | ran from repo root | `cd examples/<robot>/` — scripts use relative paths |
| Plot windows block a headless/CI run | matplotlib interactive backend | `export MPLBACKEND=Agg`, or pass `--no-plot` (`validate.py` sets this itself) |
| Scripts silent / no INFO output | default log level is WARNING | add `--verbose` / `-v` |

## Next

Environment green → return to `figaroh-start`'s Dispatch table, or go straight to
`figaroh-setup-calibration`, `figaroh-setup-identification`, `figaroh-setup-optimal`,
or `figaroh-setup-new-robot`.
