# AGENTS.md

Compact guidance for OpenCode sessions working in this repo. Read before editing.
Every line here is something an agent would likely get wrong without it.

## What this is

FIGAROH — Python toolbox for robot/human dynamics identification and geometric
calibration, URDF-based, built on Pinocchio. Published to PyPI as `figaroh` (0.4.3).
Forked from the LAAS gitlab repo. Examples and robot models live in a **separate**
repo (`figaroh-examples`) and are excluded from this package's sdist.

## Setup

- **Core dev install:** `pip install -e .` (src layout, hatchling backend). Required
  before any `import figaroh` works.
- **Need IPOPT?** Use conda: `conda env create -f environment.yml` then
  `conda activate figaroh-dev`. This is the only supported way to get `cyipopt`
  (Python 3.12, conda-forge), which `figaroh.optimal` / `tools/robotipopt.py` require
  for trajectory optimization. `cyipopt` is not pip-installable in practice — that is
  the whole reason the conda env exists.
- `requires-python >= 3.8`, but the dev/conda env pins 3.12.

## Commands

- **Tests:** `pytest` from repo root. There is **no pytest config** (no `[tool.pytest]`,
  no `pytest.ini`/`setup.cfg`/`tox.ini`) — defaults apply and discovery picks up `tests/`.
  - Single file: `pytest tests/unit/test_solver.py`
  - Single test: `pytest tests/unit/test_solver.py::TestLinearSolver::test_name`
  - Shared fixtures in `tests/conftest.py`: `temp_urdf`, `sample_robot_data`,
    `regressor_params`.
  - Tests import submodules directly, e.g. `from figaroh.tools.solver import LinearSolver`.
    Tests needing Pinocchio/meshcat/robot models (collisions, ipopt, visualization) will
    fail or skip without those deps + model files.
- **Lint/format:** `pre-commit run --all-files` — this is the real quality gate.
- **Docs:** `cd docs && make html` (Sphinx).

## Quality gate = pre-commit, not CI

- CI (`.github/workflows/docs.yml`) **only builds docs** on push/PR to `main`. There is
  **no test or lint CI workflow** — do not assume CI will catch type/test errors.
- Pre-commit hooks: `flake8` (ignores E501 line length), `clang-format --style=Google`,
  `trailing-whitespace`, `check-yaml`, `check-ast`, `check-merge-conflict`,
  `check-added-large-files` (100 MB, excludes `models/`).
- `pre-commit` `ci.autoupdate_branch` is `devel` → **`devel` is the dev branch, `main`
  is the release branch.** PRs target `main`.

## Architecture — trust the code, not ARCHITECTURE.md

`ARCHITECTURE.md` is partly aspirational and out of date. Verify against the actual tree:

- **The "Pinocchio backend" does not exist as a backend module.** `src/figaroh/backends/`
  contains only `base.py` (abstract `DynamicsBackend`) and `mujoco.py`.
  `backends/__init__.py` tries to import `pinocchio.py`, `genesis.py`, `isaacsim.py` —
  those files do not exist, so the imports silently fail and `get_backend("pinocchio")`
  raises `ValueError: ... not available`. Ignore ARCHITECTURE.md's "✅ Implemented
  (Default)" claim for PinocchioBackend.
- **Pinocchio is used directly**, imported as `import pinocchio as pin` throughout
  `tools/`, `calibration/`, `identification/` (e.g. `pin.computeJointTorqueRegressor`,
  `RobotWrapper`). It is **not** routed through the backend abstraction.
- **Real `tools/` filenames** (ARCHITECTURE.md gets some wrong): `qrdecomposition.py`
  (not `qr_decomposer.py`), `robotcollisions.py`, `robotipopt.py`,
  `robotvisualization.py`, `randomdata.py`, `load_robot.py`, `regressor.py`, `robot.py`,
  `solver.py`.
- `tools/__init__.py` eagerly imports `robot, randomdata, regressor, qrdecomposition,
  robotvisualization, robotcollisions, robotipopt` — but **not** `solver` or
  `load_robot`. Import those by full path (`figaroh.tools.solver`).
- `figaroh/__init__.py` eagerly imports all subpackages, so `import figaroh` pulls
  Pinocchio immediately.

## Three layers (real)

1. **Workflow** (`calibration/`, `identification/`, `optimal/`) — `Base*` abstract
   classes; users subclass and implement `cost_function` etc.
2. **Tools** (`tools/`) — `RegressorBuilder`, `LinearSolver` (10+ methods: lstsq, qr,
   svd, ridge, lasso, elastic_net, tikhonov, constrained, robust, weighted),
   `QRDecomposer`, `CollisionManager`, `RobotIPOPTSolver`.
3. **Backends** (`backends/`) — abstract `DynamicsBackend`; only MuJoCo implemented.
   Treat as WIP.

## Conventions / gotchas

- **Inertial parameter ordering differs** between Pinocchio
  (`[m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`) and the "standard" format used by
  solvers. Use `reorder_inertial_parameters` in `identification/parameter.py` — do not
  hand-roll the permutation.
- **Physical consistency (SDP projection) is optional and default-off.** Enable via
  `identification.physical_consistency.enabled: true`. Needs `picos` + an SDP solver
  (e.g. `cvxopt`).
- **Config:** YAML with template inheritance via `utils.UnifiedConfigParser`; legacy
  formats are auto-detected.
- **Logging:** library uses the NullHandler pattern — do not add `print` or root logging
  in library code.
- `.github/skills/` and `ROADMAP_PRIVATE.md` are **gitignored** — local-only, not
  shipped. Do not treat them as committed repo content.
