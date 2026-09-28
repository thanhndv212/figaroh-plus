# Validation guide

Run local commands from the core repository in `figaroh-dev` (Python 3.12).
Tests should establish numerical or user-visible behavior rather than merely
mirror implementation. Dataset/model provenance is part of validation evidence.

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
pre-commit run --files <changed-files>
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
when refreshing backend support. Package
metadata currently installs `picos`, so a no-PICOS install matrix needs a separate
packaging decision; runtime missing-dependency behavior has mocked unit coverage.

Run example scripts from their robot directory, matching their relative model,
config and data paths. Set `MPLBACKEND=Agg` for headless runs. Preserve original
datasets and record the output directory. For identification changes, inspect
held-out torque error, rank/condition, parameter ordering and physical validity;
for calibration, inspect held-out pose error and exported-model FK consistency.

## Initial lint gate and existing debt

The 2026-09-28 full-tree pre-commit baseline fails on existing formatting,
unused imports/variables and other style findings. See the [audit](workflow-audit-2026-09-28.md).
The initial blocking `Lint` job therefore runs:

- Critical Python checks across **all** source and tests (`E9,F63,F7,F82`).
- Every configured pre-commit hook on **all changed files** relative to the PR
  base, or the preceding push commit. A new branch/manual run compares to `devel`.

The separate `Lint backlog` job runs the complete hook suite on the full tree,
uploads its output and explicitly warns on failure; it is advisory. No tests are
excluded to make this rollout pass. Clean existing debt in focused follow-ups,
then make the full hook suite blocking. Changed-file checks can require cleanup
of an existing file; they do not exempt old violations in a touched file.

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
