# Workflow rollout audit — 2026-09-28

Source baseline: `c1d39cd` on `devel` (includes identification fixes #11–#14).
Examples inspected at `3a2c8e9`. Local environment: macOS arm64, Python 3.12,
`figaroh-dev`; `cyipopt` 1.6.1, PICOS 2.6.1, CVXOPT 1.3.2.
This record is a dated observation, not a continuously updated status dashboard.

## Source/document discrepancies resolved

| Old claim | Observed source and correction |
|---|---|
| Pinocchio backend absent (`AGENTS.md`) | `backends/pinocchio.py` exists; `Robot.backend` creates it lazily from the current model/data |
| Docs built with Sphinx | `mkdocs.yml` and the existing workflow use MkDocs |
| Core PRs target main while devel is the dev branch | CONTRIBUTING now defines normal PRs to devel; releases/hotfixes to main and a back-merge |
| Projection overwrites raw parameters | BaseIdentification already stores raw/projected dictionaries separately |
| Solver timeout never forwarded | Config adapters pass the physical-consistency/reconstruction dictionaries through; consumers read max_seconds |
| High-level API is fully backend-switchable/MJCF-capable | from_mjcf raises; from_urdf records a backend label but loads the ordinary Robot |
| Generic inertial export is complete | First-moment and inertia handlers are debug-only stubs |
| One timeless test-health number | Replace conflicting old snapshots with dated commands, results and skip reasons |

The archived v2 roadmap preserves original track detail. Historical design
checklists remain explicitly historical; current tasks belong in issues.

## Validation observed locally

| Command / environment | Result |
|---|---|
| `python -m pytest -q -rs` before changes, no MuJoCo | 499 passed, 20 skipped |
| Existing backend/cross-backend tests with MuJoCo 3.14.0 before fix | 56 passed, 3 failed: missing `MjData.qM` |
| `python -m pytest -q -rs` after fix + new regression, MuJoCo 3.14.0 | 527 passed, 6 skipped |
| Same full suite with MuJoCo 3.9.0 | 527 passed, 6 skipped |
| `python -m flake8 src tests --select=E9,F63,F7,F82 --show-source --statistics` | Passed |
| Full pre-commit hook set on changed files | Passed |
| `python -m mkdocs build` | Passed; existing docstring warnings remain; no link warnings on the rewritten roadmap/architecture/contributor pages |
| `actionlint` 1.7.12 | Passed for both workflows |
| `git diff --check` | Passed |

The six skips with MuJoCo installed are four interactive visualization checks
and two regressor tests with obsolete mocks that catch `TypeError` and skip.
They remain visible follow-up work; no skips or test exclusions were added.
The new regression skips only when the optional MuJoCo dependency is absent.

Full-tree pre-commit on the original clean baseline failed on formatting,
trailing whitespace in four URDF fixtures, unused imports/variables and other
legacy findings. Its automatic edits were reverted before implementation.
The initial gate checks changed files strictly and critical errors globally;
the full-tree audit remains an explicitly advisory artifact. A cleanup issue
must remove the debt before that audit can be promoted to required.

## Pilot fix: MuJoCo mass-matrix API compatibility

**Reproducer:** install MuJoCo 3.14.0 in `figaroh-dev`, then run
`python -m pytest tests/unit/test_backends.py tests/integration/test_cross_backend.py -q -rs`.
The original backend accessed removed `data.qM`; three existing tests failed.

**Cause:** the upstream [MuJoCo API change](https://mujoco.readthedocs.io/en/latest/changelog.html)
changed `mj_fullM(model, destination, qM)` to `mj_fullM(model, data, destination)`.
The backend now tries the new binding and falls back on its argument-type
rejection for older MuJoCo. Both real versions are tested; no version is skipped
to avoid the failure.

**Acceptance:** mass matrices match Pinocchio on a two-joint model at two
configurations, remain positive definite and return independent result arrays;
existing backend tests and the full suite pass on both API generations.

The core CI matrix pins MuJoCo 3.9.0 and 3.14.0 to retain this coverage. Bump
these deliberately with numerical evidence rather than silently following an
unbounded backend dependency.

## CI rollout and limits

The first hosted Linux run passed lint and docs but found two previously hidden
test-portability problems: TIAGo mesh symlinks require the sibling examples repo,
and the QR precision assertion used NumPy's default relative tolerance, masking
six-decimal rounding differences for order-one entries. CI now fetches the pinned
mesh subtree; the precision assertion uses `rtol=0, atol=1e-12`. The empty-matrix
test also asserts its returned shape/mapping instead of discarding the result.
No numerical algorithm or failing-test selection was changed to obtain green.

The added workflows declare tests, changed-file hooks, critical lint, an
advisory full-tree lint audit, docs build and artifacts. Solver imports are
checked before tests so solver coverage cannot silently disappear. Docs build
and deployment are separate jobs; only a push to main can deploy. The old
unconditional PR deployment path and ignored package-install failures are removed.

## Hosted validation

Commit `556f070124ff6f3fb5dea81265cb7a96ec6b24ce` passed the
[Linux Core CI run](https://github.com/thanhndv212/figaroh-plus/actions/runs/36441865400)
and [docs run](https://github.com/thanhndv212/figaroh-plus/actions/runs/36441865407)
on 2026-09-28, using Python 3.12 and the examples revision pinned above.

| Hosted check | Result |
|---|---|
| Core full suite, without MuJoCo | 499 passed, 21 skipped |
| MuJoCo 3.9.0 full suite | 527 passed, 6 skipped |
| MuJoCo 3.14.0 full suite | 527 passed, 6 skipped |
| Critical lint and all changed-file hooks | Passed |
| Docs build | Passed; deployment correctly skipped for the PR |
| Full-tree lint backlog | Existing debt reported in the advisory artifact |

The extra core-only skips cover absent MuJoCo, including the new regression
module. With MuJoCo installed, the remaining six skips are the same four GUI
checks and two obsolete regressor mocks recorded above. Each test job archives
JUnit, installed dependency versions and the exact examples revision.

M0's implementation is validated in draft PRs #17–#19 and awaits integration
into `devel`. Server-side branch protection is not established by these runs.
No release, PyPI publication or physical hardware validation is part of this
change; broader lint and warning cleanup remains follow-up work.
