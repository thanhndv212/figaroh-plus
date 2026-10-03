# Validation guide

Run local commands from the core repository in `figaroh-dev` (Python 3.12).
Tests should establish numerical or user-visible behavior rather than merely
mirror implementation. Dataset/model provenance is part of validation evidence.

The TIAGo fixture mesh symlinks require the sibling `figaroh-examples` checkout.
CI fetches only its mesh subtree at commit
`3a2c8e9b07e10b397cda79dbb04a5e88469bcc22` and archives the revision. For a new
local workspace, clone that repository next to `figaroh` before the full suite;
keep an existing examples checkout intact and record its revision. Kinematic
models/data are already tracked in core. Do not replace missing geometry with
empty directories or skip its existence test.

## Levels

| Level | Required for | Evidence |
|---|---|---|
| V0 — local checks | Every PR | Changed-file hooks, critical lint, diff check; docs build for documentation changes |
| V1 — core suite | Core code, dependencies, CI, and release changes | Full pytest result including skip reasons; regression tests for fixes |
| V2 — example workflow | Changes to data/config, numerical pipelines or public APIs | Affected `figaroh-examples` workflow, exact core/examples revisions, before/after metrics with the same data/environment |
| V3 — milestone/release | Milestone closure and release preparation | Exit criteria from ROADMAP; fit and held-out metrics, physical verdict and export/reload invariants where applicable, plus V0–V2 |

Docs-only PRs use V0 and the normal hosted jobs. State which levels were run;
list omitted evidence explicitly. A synthetic test, simulation example and
physical experiment establish different things.

## Local commands

```bash
conda activate figaroh-dev
python -m pip install -e '.[dev,docs]'
python -m pytest -q -rs
python -m flake8 src tests --select=E9,F63,F7,F82 --show-source --statistics
pre-commit run --all-files
python -m mkdocs build
git diff --check
```

For optional backend coverage:

```bash
python -m pip install mujoco
python -m pytest tests/unit/test_backends.py tests/integration/test_cross_backend.py -q -rs
```

Physical projection/reconstruction and IPOPT coverage run in the core suite.
CI verifies that `cyipopt`, `picos` and `cvxopt` actually import before running it;
the MuJoCo jobs add versions 3.9.0 and 3.14.0 respectively and execute the
same full suite. These pins cover both mass-matrix APIs; update them deliberately
when refreshing backend support. The `core` and `mujoco-3.9` profiles use
Pinocchio 3.7.0 / ndcurves 2.0.0.1; `pinocchio-4.1` and `mujoco-current`
use Pinocchio 4.1.0 / ndcurves 2.3.0. CI writes `ci-constraints.txt` and
sets `PIP_CONSTRAINT` before creating `figaroh-dev`, preventing an initial
unconstrained install followed by incompatible native-package downgrades.
Native imports, exact robotics versions and `pip check` are required.
See the [support decision](../decisions/pinocchio-version-support.md).
Package
metadata currently installs `picos`, so a no-PICOS install matrix needs a separate
packaging decision; runtime missing-dependency behavior has mocked unit coverage.

Run example scripts from their robot directory, matching their relative model,
config and data paths. Set `MPLBACKEND=Agg` for headless runs. Preserve original
datasets and record the output directory. For identification changes, inspect
held-out torque error, rank/condition, parameter ordering and physical validity;
for calibration, inspect held-out pose error and exported-model FK consistency.

## Lint gate

The blocking `Lint` job runs:

- Critical Python checks across **all** source and tests (`E9,F63,F7,F82`).
- Every configured pre-commit hook on the **full tree**
  (`pre-commit run --all-files`).

Until 2026-10-03 the full tree carried legacy debt (see the
[2026-09-28 audit](workflow-audit-2026-09-28.md)), so `Lint` gated only changed
files and an advisory `Lint backlog` job audited the full tree. The debt was
inventoried (below) and cleared in focused PRs, after which the full hook suite
became blocking and the advisory job was removed. No tests were excluded or
skipped to get there.

### Full-tree debt inventory (2026-10-03)

`pre-commit run --all-files` on `devel` `cc262d8` (`figaroh-dev`), each hook
run on its own on a pristine tree (#57). The critical selection
(`E9,F63,F7,F82`) is clean; every other hook not listed here passes.

| Hook | Findings | Cleanup issue |
|---|---|---|
| `black` | 10 files would be reformatted | #81 |
| `trailing-whitespace` | 4 TIAGo fixture URDFs | #81 |
| `flake8` F401 in package `__init__.py` (public re-exports) | 22 | #82 |
| `flake8` F401 / E402 / E131 in other modules | 32 / 18 / 2 | #83 |
| `flake8` F841 / E722 in `src/` (possible dead logic) | 8 / 2 | #84 |
| `flake8` F841 in `tests/` (possible missing assertions) | 6 | #85 |

The 90 flake8 findings are split by the kind of review they need, not by
module. Each issue lists its exact findings and is fixed in its own PR.
Re-exports must not be deleted (#82). Unused locals in #84 and #85 may be
logic or assertions that were meant to take effect, so they are not
mechanical deletions.

**Cleared.** All six rows were fixed in #87, #88, #89, #91 and #92 (rebase-merged
into `devel`). `pre-commit run --all-files` passes on `devel` `a2caeea`, locally in
`figaroh-dev` and in the hosted `Lint backlog` audit of #92 (audit step
`success`), and `flake8 src tests` reports no findings. Two findings turned out
to be real defects and were fixed with regression tests (`frame_settings_doc`
hint, #84; two unasserted test computations, #85); two pre-existing algorithm
defects found on the way are tracked in #90.

Documentation currently has legacy mkdocstrings/link warnings. The build fails
on errors; warning cleanup is a separate task before enabling global strict mode.
The workflow must not suppress installation or build errors. Docs deployment is
restricted to successful builds on **pushes to `main`**, never pull requests.

## Evidence record

A PR or checked-in milestone result records:

- Core and examples commit IDs, Python/platform and relevant solver versions.
- Commands, exit status, pass/fail/skip counts and reasons for missing coverage.
- Data/model/config identifiers, random seeds, units and train/validation split.
- Comparable before/after metrics and any changed tolerances, with rationale.
- Artifact locations: exported model, report/verdict, logs and reproducible input.

Keep large run artifacts outside source control; link durable CI artifacts or
archive locations. A summary committed under `docs/development/` can reference
those artifacts without duplicating generated data. Never promote a passing
file-write test into a claim of physical-model or hardware correctness.
