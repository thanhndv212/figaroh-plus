---
name: figaroh-start
description: 'Entry point and router for the figaroh-ws workspace. Use FIRST whenever the user wants to "use FIGAROH", run a calibration/identification/optimal task, onboard a robot, or asks where something lives — it maps the two repos, names the exact package and directory for the job, and dispatches to the right figaroh-setup-* skill.'
---

# FIGAROH Workspace — Start Here

_Arguments: say what you want to do (calibrate a robot, identify dynamics, generate
optimal configs/trajectories, onboard a new robot, just get it running) and, if you
have one, the robot name._

> **Paths.** `$FIGAROH_WS` is the workspace root — the directory holding the
> `figaroh` and `figaroh-examples` repos side by side. Set it once per shell:
> `export FIGAROH_WS="$(cd /path/to/your/workspace && pwd)"`.

## When to Use

- The user says "set up FIGAROH", "how do I run this", "where do I start".
- A task is named (calibration / identification / optimal / new robot) but the
  package, directory, config, or env is not yet established.
- You are about to `cd` or `import` and are not certain which of the two repos owns
  the thing you need.

Run this skill's **Orientation** section, then hand off to the setup skill named in
**Dispatch**. Do not start editing files before Dispatch — the most common failure in
this workspace is doing correct work in the wrong repo.

## Orientation — the two repos

Workspace root: `$FIGAROH_WS/`

| Directory | What it is | Installable? | You work here when… |
|---|---|---|---|
| `figaroh/` | **The toolbox.** Python library `figaroh`, src layout at `figaroh/src/figaroh/`, PyPI name `figaroh` (v0.4.7). Owns the algorithms. | Yes — `pip install -e .` | Changing library behaviour: solvers, regressors, base classes, reports. |
| `figaroh-examples/` | **Where users actually run things.** Robot example scripts, YAML configs, CSV data, URDFs, and shared `models/`. Explicitly **not** an installable package. | No | Running a task, adding a robot, authoring configs, integration tests. |

These two are the whole picture. If the workspace directory happens to hold other
checkouts, they are unrelated to FIGAROH — this skill set assumes and touches only
`figaroh/` and `figaroh-examples/`.

**The two-repo rule.** FIGAROH is one library plus one example repo, side by side.
`figaroh-examples` imports `figaroh` from `../figaroh` via an editable install — there
is no vendored copy. After changing library code, re-run `pip install -e .` in
`figaroh/` (only needed if metadata/entry points changed; editable src picks up edits
live).

## Orientation — directories inside `figaroh-examples`

```
figaroh-examples/
  examples/<robot>/        # ← run scripts from HERE, never from repo root
    calibration.py         # entry points (not every robot has all of them)
    identification.py
    optimal_config.py
    optimal_trajectory.py
    update_model.py
    config/                # *_unified_config.yaml (current) + legacy *_config.yaml
    data/                  # CSV measurements in, .npz results out
    urdf/                  # the robot's URDF(s)
    utils/<robot>_tools.py # robot-specific subclasses of figaroh Base* classes
    results/runs/…         # timestamped archived runs
  models/                  # shared ROS description packages (meshes) — read-only
  examples/templates/      # base/manipulator/humanoid config templates
  validate.py              # the quality gate: tests + all example scripts
```

Two path facts every script depends on, and that break everything when violated:

1. Scripts use **relative** paths (`config/x.yaml`, `urdf/x.urdf`), so the working
   directory **must** be `examples/<robot>/`.
2. Meshes resolve via `package_dirs="../../models"`, which is only correct from
   `examples/<robot>/`.

## Orientation — what exists today

| Robot | Directory | Calibration | Identification | Optimal | Config |
|---|---|---|---|---|---|
| UR10 | `examples/ur10/` | `calibration.py` | `identification.py` | `optimal_config.py`, `optimal_trajectory.py` | `config/ur10_unified_config.yaml` |
| TIAGo | `examples/tiago/` | `calibration.py` | `identification.py` | both | `config/tiago_unified_config.yaml` |
| TALOS | `examples/talos/` | `calibration_upperbody.py` | — | — | `config/talos_unified_config.yaml` |
| TALOS table-contact | `examples/talos_table_contact/` | `run_calibration.py` | — | — | `config/talos_table_{left,right}_config.yaml` |
| Staubli TX40 | `examples/staubli_tx40/` | — | `identification.py` | — | `config/staubli_tx40_unified_config.yaml` |
| TIAGo Pro | `examples/tiago_pro/` | `run_calibration.py` | — | `generate_optimal_configs.py` | `tiago_pro_calibration_config.yaml` (**at robot root**, no `config/` dir) |

**UR10 is the reference implementation.** When you need a working example of anything,
read `examples/ur10/` first; `examples/tiago/` second (it has the most scripts).

## Dispatch

| User wants | Invoke | Working directory |
|---|---|---|
| Anything, first time on this machine / import fails / `cyipopt` missing | **`figaroh-setup-env`** | anywhere |
| Geometric (kinematic) calibration — correct joint offsets & frames from measured poses | **`figaroh-setup-calibration`** | `figaroh-examples/examples/<robot>/` |
| Dynamic identification — masses, inertias, friction from motion + torques | **`figaroh-setup-identification`** | `figaroh-examples/examples/<robot>/` |
| Optimal measurement configs or exciting trajectories (OED) | **`figaroh-setup-optimal`** | `figaroh-examples/examples/<robot>/` |
| A robot that has no folder under `examples/` yet | **`figaroh-setup-new-robot`** | `figaroh-examples/` |

Beyond setup, this skill set hands off to the repos themselves: re-run with
`--verbose` and work from the traceback when something breaks mid-run; library
algorithms live in `figaroh/src/figaroh/`; the example suite and integration tests run
from `figaroh-examples/validate.py`; packaging, CI, and docs live in `figaroh/`. Each
repo's `AGENTS.md` and `README.md` carry the conventions for that work.

Always run `figaroh-setup-env` before any of the task setup skills unless a `figaroh`
import has already succeeded in this session.

## Fastest path to a working run (verify the toolbox before touching the user's robot)

```bash
conda activate figaroh-dev
cd $FIGAROH_WS/figaroh-examples/examples/ur10
python calibration.py --calibrate-only --no-plot
```

If that prints post-calibration RMSE, the toolbox is healthy and any later failure is
in the user's model, config, or data — not the install. Use this as the bisect point.

## Hard rules

- **Conda env `figaroh-dev` is mandatory** for every command in `figaroh/` and
  `figaroh-examples/`. It is defined by `figaroh/environment.yml` (Python 3.12) and is
  the only supported source of `cyipopt`. Ignore `figaroh-examples/environment.yml` —
  it names a `figaroh-examples` env that nobody uses.
- **Never run example scripts from the repo root.** `cd examples/<robot>` first.
- **Never edit `figaroh-examples/models/`.** Those are upstream ROS description
  packages, mesh assets only.
- **Write new configs in the unified format** (`tasks.<task>.*` with `extends:`), not
  the legacy flat format. Legacy is auto-detected for backward compatibility only.
- **`examples/shared/` does not exist** and is gitignored. Never import
  `examples.shared.*`.
- The library uses the NullHandler logging pattern — never add `print`/root logging to
  `figaroh/src/`. Example scripts may print freely.
