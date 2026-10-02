# Issue #32: tangent differentiation validation — 2026-10-02

## Scope and numerical policy

Baseline core: `90c8431` (devel). Examples: `edef318` (documentation-only
changes above main; unchanged UR10 scripts, inputs and config). The issue fixes
configuration differentiation, not signal filtering, simulated torque generation
or the estimator. No source datasets/models were changed.

The helper takes `nq` configuration columns and uses model/backend differences
to produce `nv` tangent components. All components are differentiated regardless
of effort flags. `dt` is a scalar interval or exactly one positive finite interval
per sample pair, never an absolute timestamp vector. Irregular-time acceleration
uses the times at which interval velocities are centered. Invalid intervals,
configuration shape/nonfinite values and fewer than three samples fail explicitly.

The historical output alignment is retained: positions `q[:-2]`, the first
`n_samples-2` forward interval velocities and their gradients. Positions and
interval velocities are not collocated in time. Both removed position samples
are trailing samples, contrary to the old docstring. Manifold configurations
must already be valid. The routine differentiates returned tangent components;
it does not perform moving-frame transport or interpolate them to position times.

## Independent regression evidence

Quadratic trajectories independently supply every coordinate's acceleration and
interval-midpoint velocity. Tests cover constant and uneven intervals, all effort
flag combinations, a model-free backend adapter call, quaternion free-flyer and
continuous joints with `nq != nv`, an angle-wrap crossing, minimum sample count
and malformed/nonpositive/nonfinite timing/configuration inputs.

Before the change, the initial 33-test regression set failed on the baseline.
The minimal six-joint example `q[:, -1] = 0.5*t**2` returned zero acceleration
for every last-joint sample instead of 1 rad/s². With the correction, both native
Pinocchio 3.7.0 and 4.1.0 profiles pass the regression suite. The angle-wrap case
was added afterward to independently check Lie-group behavior at the coordinate
representation boundary; final counts are recorded in the PR.

Final regression count: **34 passed** per native profile. Final full suite:
**573 passed, 6 skipped** on Pinocchio 3.7.0 and **573 passed, 6 skipped** on
Pinocchio 4.1.0. Critical lint, changed-file hooks, diff check and MkDocs build
pass. Existing API docstring warnings remain.

## Unchanged UR10 workflow: compatibility evidence

Run copied robot/templates folders under `/tmp/figaroh-issue32-ur10-validation/`
with documented model packages, preserving the original checkout and recordings.
For each source version, from the corresponding copied `examples/ur10/`:

```bash
conda run --no-capture-output -n figaroh-dev env MPLBACKEND=Agg \
  PYTHONPATH=<baseline-or-fixed-core>/src python identification.py \
  --no-html-report --no-archive --verify
```

| Metric | Baseline | Corrected helper |
| --- | --- | --- |
| Loaded / processed samples | 500 / 498 | 500 / 498 |
| Printed aggregate fit RMSE | 0.066635 | 0.066744 |
| Condition number | 20617.776354 | 21302.771354 |
| Nominal-fit improvement (%) | 8.033 | 7.297 |
| Correlation gate | Pass | Pass |
| Condition / improvement gates | Fail / Fail | Fail / Fail |
| Exit code | 1 | 1 |

Both runs complete fitting and fail the **existing** quality gates (condition
<=1000, improvement >=50%); no thresholds are changed. The current entry point
warns that it uses **training fallback** for validation. These values are not
independent held-out performance or evidence that stored simulation torques were
generated with correct accelerations. The existing printed RMSE label is not
corrected in this issue; per-joint effort conventions remain an adapter concern.

An audit of the current UR10 source finds consumers of the simulation CSV names,
but no traced generator recording the accelerations used for their torques.
The loader additionally filters/labels time at 100 Hz while configuration timing
requires its separate audit. Old scientific comparisons remain preprocessing-
limited; do not rerun or rank physical estimators on these CSVs as validated
simulation truth. D2/D3 and examples signal/truth issues own that follow-up.

## Input identity

| File relative to examples repository | SHA256 |
| --- | --- |
| `examples/ur10/utils/ur10_tools.py` | `d5f8fe1f1feb44bc7978792db027f9d16ba68b177fc5e9ce0361cae6f82ea865` |
| `examples/ur10/config/ur10_unified_config.yaml` | `bd093e400cd8400d283fb883f49a4a76d02bbdc9d8b523728b6b69dcb3a6f883` |
| `examples/ur10/urdf/ur10_robot.urdf` | `1da0c0de1909bbf6bb5ea9449ee07ccf88e0b3456edca2955e23cc4770027ecd` |
| `examples/ur10/data/identification_q_simulation.csv` | `cee459266b276c995529a2a979a72a5edc05c3d86fd4c67857888b4d063718b6` |
| `examples/ur10/data/identification_tau_simulation.csv` | `95443e9f749afa51db1f89a230c630e1c4e30c3eccee88fc522913c04a860828` |

## Validation commands and retained evidence

All tests explicitly set `PYTHONPATH` to this issue worktree's `src`, including
the isolated Pinocchio 4.1 environment `/tmp/figaroh-pin41/figaroh-dev`.

```bash
conda run --no-capture-output -n figaroh-dev env PYTHONPATH=<issue-worktree>/src \
  python -m pytest -q -rs
conda run --no-capture-output -p /tmp/figaroh-pin41/figaroh-dev \
  env PYTHONPATH=<issue-worktree>/src python -m pytest -q -rs
conda run --no-capture-output -n figaroh-dev \
  python -m flake8 src tests --select=E9,F63,F7,F82 --show-source --statistics
```

Changed-file hooks, diff checks and the docs build accompany the PR. Existing
skips are four opt-in visualization cases and two incompatible legacy regressor
mocks; no new skips hide the differentiation regression. Local evidence logs:
`/tmp/figaroh-issue32-before.log`, `core37.log`, `core41.log`,
`ur10-before.log`, `ur10-after.log` (the latter names share the
`/tmp/figaroh-issue32-` prefix). Hosted checks must be assessed separately.

This completes the focused implementation evidence, not maintainer acceptance
or D1 milestone closure. Merge and closure require explicit maintainer approval.
