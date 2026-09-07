---
name: figaroh-setup-new-robot
description: 'Onboard a robot that has no folder under figaroh-examples/examples/ yet — scaffold the directory, place the URDF and description package, author a unified config from a template, write the robot-specific tools subclass, and wire it into validate.py. Use when the user names a robot FIGAROH does not know, or asks how to add their own robot to the toolbox.'
---

# Onboard a New Robot

_Arguments: robot name, where its URDF/description package lives, and which tasks it
needs (calibration / identification / optimal)._

> **Paths.** `$FIGAROH_WS` is the workspace root — the directory holding the
> `figaroh` and `figaroh-examples` repos side by side. Set it once per shell:
> `export FIGAROH_WS="$(cd /path/to/your/workspace && pwd)"`.

**Prerequisite:** `figaroh-setup-env` exits 0. **Working directory:**
`$FIGAROH_WS/figaroh-examples/`.

## Step 0 — Confirm this is actually needed

```bash
ls $FIGAROH_WS/figaroh-examples/examples/
```

If a folder already exists, this is the wrong skill — go to `figaroh-setup-calibration`
/ `-identification` / `-optimal` instead. Only proceed when there is no folder.

Then pick a reference by kinematic class, because the config template and the base
classes differ:

| Robot class | Reference example | Template |
|---|---|---|
| Fixed-base serial manipulator (UR, KUKA, Staubli) | `examples/ur10/` | `templates/manipulator_robot.yaml` |
| Mobile manipulator / humanoid (floating or mobile base, coupled wrists) | `examples/tiago/`, `examples/talos/` | `templates/humanoid_robot.yaml` |

**UR10 is the most complete reference. Copy from a real example, not from the
scaffold's placeholders.**

## Step 1 — Place the description package (meshes)

```bash
ls figaroh-examples/models/     # ur_description, tiago_description, talos_description, …
```

Drop the robot's ROS description package in as `models/<robot>_description/` so
`package://<robot>_description/...` URIs in the URDF resolve. `models/` is shared,
upstream content — **add to it, never edit what is there.** Meshes are large; check
`.gitignore` and the 100 MB pre-commit limit before committing them.

If the robot has no meshes, skip this — `package_dirs` is unused when the URDF has no
mesh references.

## Step 2 — Scaffold the folder

```bash
cd $FIGAROH_WS/figaroh-examples/examples
./create_example.sh <robot_name>
```

Creates `examples/<robot_name>/` with the standard tree (`config/`, `data/calibration/
mocap`, `data/identification/dynamic`, `data/optimal_configurations`, `docs/`, `urdf/`,
`utils/`, `tmp/`) and TIAGo-derived name substitution (acronyms like `UR10`/`TX40` are
preserved, not title-cased).

It also stubs `README.md`, `SETUP_GUIDE.md`, `data/*/README.md`, `utils/__init__.py`,
`utils/<robot>_tools.py`, and `utils/simplified_collision_model.py`.

What it gives you, verified by running it:

- **All four generated scripts are placeholders** (`calibration.py`,
  `identification.py`, `optimal_config.py`, `optimal_trajectory.py`). Each prints
  "not yet implemented" and exits, and points you at the corresponding
  `examples/ur10/` file. Replace them wholesale in Step 5.
- **The generated config is a real unified config** — `config/<robot>_unified_config.yaml`
  with `extends: "../../templates/manipulator_robot.yaml"` and all four task blocks,
  pre-filled with TODO markers and the URDF-probe command from Step 3. Edit it in
  place rather than starting over; swap the template for `humanoid_robot.yaml` if the
  robot has a mobile or floating base.

Or do it by hand:

```bash
mkdir -p examples/<robot>/{config,data,urdf,utils,results}
touch examples/<robot>/utils/__init__.py
```

## Step 3 — Add the URDF

Put it at `examples/<robot>/urdf/<robot>.urdf`, then prove it loads before writing any
task code:

```bash
cd $FIGAROH_WS/figaroh-examples/examples/<robot>
conda run -n figaroh-dev python -c "
from figaroh.tools.robot import load_robot
r = load_robot('urdf/<robot>.urdf', package_dirs='../../models', load_by_urdf=True)
print('nq', r.model.nq, 'nv', r.model.nv)
print('joints', [n for n in r.model.names])
print('frames', [f.name for f in r.model.frames][:40])
"
```

**Do not skip this.** The printed joint names are the exact strings your config's
`active_joints` and your calibration CSV headers must use, and the frame names are what
`base_frame`/`tool_frame` must match. Copy them from this output rather than typing
them.

## Step 4 — Author the unified config

Create `config/<robot>_unified_config.yaml`. Start from the template with `extends:`
(the keyword is `extends`, **not** `inherit_from`) and override only what differs:

```yaml
extends: "../../templates/manipulator_robot.yaml"   # or humanoid_robot.yaml

robot:
  name: "<robot>"
  description: "<one line>"

  properties:
    joints:
      active_joints: ["joint_1", "joint_2"]   # EXACT names from Step 3
      joint_limits:
        position: [...]      # rad, [min,max] per joint, flattened
        velocity: [...]      # rad/s
        acceleration: [...]  # rad/s^2
        torque: [...]        # Nm
    mechanics:
      reduction_ratios: [...]
      friction_coefficients:
        viscous: [...]       # 0 = to be identified
        static: [...]
      actuator_inertias: [...]
      joint_offsets: [...]   # 0 = to be calibrated

tasks:
  calibration:      { enabled: true }   # fill per figaroh-setup-calibration
  identification:   { enabled: true }   # fill per figaroh-setup-identification
  optimal_configuration: { enabled: false }
  optimal_trajectory:    { enabled: false }

environment:
  working_directory: "."
  data_directory: "data"
  results_directory: "results"
```

Rules:

- **Unified format only.** `tasks.<task>.*` with `extends:`. The legacy flat format is
  auto-detected for backward compatibility — never write a new config in it.
- Robot-wide facts go under `robot.properties.*`; task-specific ones under
  `tasks.<task>.*`. Putting task settings in `robot.properties` silently does nothing.
- Enable only the tasks you are actually implementing. An `enabled: true` task with no
  entry-point script is a trap for the next reader.
- Optionally add a `robot.instance` block (`asset_id`, `serial_number`, `site`,
  `operator`) for per-unit provenance. Prefer a separate overlay file per physical unit
  (see `examples/ur10/config/UR10-007.yaml`) over editing the shared config, so
  onboarding a second unit cannot disturb the first.

## Step 5 — Write the tools subclass and entry points

`utils/<robot>_tools.py` holds the robot-specific subclasses of the FIGAROH base
classes:

| Task | Base class | Must implement |
|---|---|---|
| Calibration | `figaroh.calibration.base_calibration.BaseCalibration` | robot-specific data/marker handling |
| Identification | `figaroh.identification.base_identification.BaseIdentification` | **`load_trajectory_data()`** — the abstract method |
| Optimal | `figaroh.optimal.base_optimal_calibration.BaseOptimalCalibration`, `…base_optimal_trajectory.BaseOptimalTrajectory` | problem-specific constraints |

The scaffold leaves a `utils/<robot>_tools.py` stub; overwrite it — copy
`examples/ur10/utils/ur10_tools.py` and adapt. Then copy the corresponding entry
points (`calibration.py`, `identification.py`, …) from `examples/ur10/`, replacing:

- the import of `UR10Calibration` with your class,
- the argparse `--config` / `--urdf` defaults,
- `URDF_STEM` and `DATA_DIR` constants,
- the `package_dirs="../../models"` call (keep the value — only the URDF path changes).

Keep the argparse surface (`--verbose`, `--html-report`, `--verify`, `--archive`, …)
identical to UR10's so `validate.py` and CI can drive your robot the same way.

Path convention every entry script follows — scripts use **relative** paths and are run
from `examples/<robot>/`:

```python
project_root = Path(__file__).parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from examples.<robot>.utils.<robot>_tools import <Robot>Calibration
```

**Never import `examples.shared.*`** — that package does not exist and is gitignored;
`examples/__init__.py` swallows the ImportError silently.

## Step 6 — Add data

Layout and column contracts are task-specific and non-obvious — follow
`figaroh-setup-calibration` Step 2 (marker + joint-name columns, driven by
`measurable_dof`) and `figaroh-setup-identification` Step 2 (q/tau CSVs, per-robot
filenames). Get these wrong and the failure surfaces as a pandas `KeyError` or a shape
mismatch, not a helpful message.

## Step 7 — Wire into validation and docs

Add the robot to `EXAMPLE_SCRIPTS` in `figaroh-examples/validate.py` — entries are
`(script_name, timeout_seconds, is_slow[, extra_args])`:

```python
EXAMPLE_SCRIPTS = {
    "<robot>": [
        ("calibration.py", 120, False,
         ["--calibrate-only", "--no-plot", "--html-report"]),
        ("identification.py", 120, False, ["--verify", "--html-report"]),
        ("optimal_trajectory.py", 600, True),     # IPOPT → slow
    ],
}
```

Mark anything IPOPT-driven `is_slow=True` with a 600 s timeout so `--quick` skips it.
Then write `examples/<robot>/README.md` (copy UR10's shape: what the robot is, which
scripts exist, which flags they support, expected data files) and add a bullet to
`figaroh-examples/README.md`'s Examples list.

## Step 8 — Verify

```bash
conda activate figaroh-dev
cd $FIGAROH_WS/figaroh-examples

python validate.py --robot <robot>     # the new robot's scripts + tests
python validate.py --quick             # nothing else regressed
pre-commit run --all-files             # the real quality gate; there is no lint CI
```

Done means: `validate.py --robot <robot>` exits 0, and a run archive appeared under
`examples/<robot>/results/runs/`.

## Checklist

- [ ] `models/<robot>_description/` present (or URDF has no meshes)
- [ ] URDF loads; joint and frame names captured from Step 3's output
- [ ] `config/<robot>_unified_config.yaml` uses `extends:`, unified format, real joint names
- [ ] `utils/<robot>_tools.py` subclasses the right `Base*` classes
- [ ] Entry scripts copied from UR10, argparse surface preserved
- [ ] Data files match the task's column contract
- [ ] `validate.py` `EXAMPLE_SCRIPTS` entry added, slow scripts marked
- [ ] `README.md` written and linked from the repo README
- [ ] `validate.py --robot <robot>` exits 0

## Gotchas

- `tiago_pro` keeps its config at the robot root, not in `config/` — an exception, not
  the pattern to copy.
- `create_example.sh` derives from TIAGo and substitutes `tiago` → `<robot>`
  throughout, so grep generated files for a stray `tiago` before trusting them.
  Pointers to the reference example are protected by a `REF_EXAMPLE` token and
  correctly resolve to `examples/ur10/`.
- Pre-commit enforces a 100 MB file limit (`models/` excluded). Large meshes elsewhere
  will be rejected.
- `devel` is the development branch, `main` the release branch — matching the core
  `figaroh/` repo convention. PRs target `main`.
