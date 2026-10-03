# Pinocchio compatibility audit — 2026-10-02

Issue: [#28](https://github.com/thanhndv212/figaroh-plus/issues/28).
The change preserves the Pinocchio 3.7 baseline and adds a tested 4.1 profile.
It is independent of the pseudo-inertia correction and log-Cholesky solver work.

## Environments and dependency resolution

Both local profiles use Python 3.12 on macOS. The original `figaroh-dev`
remains on Pinocchio 3.7.0. A separate environment at
`/tmp/figaroh-pin41/figaroh-dev` validates 4.1.0.

| Component | Retained baseline | New profile |
| --- | --- | --- |
| pin | 3.7.0 | 4.1.0 |
| ndcurves | 2.0.0.1 | 2.3.0 |
| cmeel-assimp | 5.4.3.1 | 6.0.5 |
| cmeel-urdfdom | 4.0.1 | 6.0.0 |
| cmeel-tinyxml2 | 10.0.0 | 11.0.0 |

Native Pinocchio and ndcurves imports and `python -m pip check` passed in
both environments. Resolver dry runs for `pip install -e '.[dev]'
pytest-timeout --ignore-installed --dry-run` passed with each profile's
`PIP_CONSTRAINT`. CI applies the constraints before environment creation.
Updating Pinocchio alone can retain incompatible native libraries; these
checks validate imports in addition to package metadata.

## Core validation

Run from the independent compatibility worktree:

```bash
PYTHONPATH="$PWD/src" conda run -n figaroh-dev python -m pytest -q -rs --tb=short
PYTHONPATH="$PWD/src" conda run -p /tmp/figaroh-pin41/figaroh-dev python -m pytest -q -rs --tb=short
```

Both runs: **532 passed, 6 skipped**. The skips are four opt-in visual tests
and two existing regressor mock/signature cases. No new compatibility skips
were introduced. This local suite includes the installed optional backend;
CI separately runs core-only and MuJoCo profiles.

The new regressions exercise real frame access, relative kinematic
regressors, COM visualization calculations without a display, and three
seeded log-Cholesky conversions with analytic Jacobians checked against
central differences. Deprecated frame access raises an error in these tests.
Changed-file pre-commit hooks, critical flake8 checks and MkDocs build passed.

## Representative examples

Examples revision: `3a2c8e9b07e10b397cda79dbb04a5e88469bcc22`.
Run UR10 examples from separate temporary copies with their models and
templates available, using the compatibility worktree's `src` on
`PYTHONPATH` and `MPLBACKEND=Agg`:

```bash
python identification.py --no-html-report --no-archive --no-verify
python calibration.py --calibrate-only --no-plot --no-html-report --no-archive
```

Each command completed successfully on both versions. Reported metrics:

| Metric | Pinocchio 3.7 | Pinocchio 4.1 |
| --- | --- | --- |
| Identification torque RMSE | 0.0666 | 0.0666 |
| Identification nominal torque RMSE | 0.0725 | 0.0725 |
| Identification correlation | 1.0000 | 1.0000 |
| Calibration position RMSE | 2.95 mm | 2.95 mm |
| Calibration orientation RMSE | 0.3332 deg | 0.3332 deg |
| Calibration overall RMSE | 0.006522 | 0.006522 |

These are matching printed metrics on the existing dataset, not a claim of
hardware validation or improved held-out model quality. Identification's
verification gate and calibration URDF export/FK verification were not run.
The source example datasets were unchanged; generated outputs stayed in
temporary copies.

## Coverage and evidence

Local logs: `/tmp/figaroh-pincompat-{37,41}-suite.log`,
`/tmp/figaroh-pincompat-{37,41}-{identification,calibration}.log`, and
`/tmp/figaroh-pincompat-docs.log`. They are session artifacts, not durable
repository fixtures. Hosted CI publishes installed versions, constraints,
fixture revision and test XML for each profile.

Only 3.7.0 and 4.1.0 are directly validated. Package range `pin>=3.7,<5`
allows other compatible releases without claiming every release, operating
system or Python version was tested. Hosted checks must pass before merge.
