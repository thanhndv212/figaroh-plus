# Agent guidance for FIGAROH

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the contribution/branch/release
workflow. Root workspace instructions also apply. This file holds execution
constraints and pitfalls; current architecture belongs in
[ARCHITECTURE.md](ARCHITECTURE.md), priorities in [ROADMAP.md](ROADMAP.md).

## Environment and commands

- All FIGAROH development, tests, lint and docs builds use `figaroh-dev`:
  `conda activate figaroh-dev` or `conda run -n figaroh-dev ...`.
- Create it with `conda env create -f environment.yml`; install contributor tools
  using `python -m pip install -e '.[dev,docs]'`. Use conda for `cyipopt`/IPOPT.
- Python development/initial CI baseline is 3.12. Package metadata still declares
  Python >=3.8; this is not a claim that every supported version is CI-tested.
- Tests: `python -m pytest -q -rs`; one file:
  `python -m pytest tests/unit/test_identification_regressions.py -q -rs`.
- Hooks: `pre-commit run --all-files` (what the blocking `Lint` job runs); for a
  quick check, `pre-commit run --files <changed-files>` (new files must be staged
  for hooks to see them).
  Hooks can modify files. Inspect and rerun after formatting.
- Docs: `python -m mkdocs build` from the repository root. This is MkDocs,
  not Sphinx. Generated `site/` is ignored.
- Consult [validation.md](docs/development/validation.md) for the critical lint
  command, known legacy debt, optional backend checks and evidence requirements.

## Source navigation

When `.codegraph/` exists, use `codegraph explore` or `codegraph node` before
text searches for code understanding. It may not index Markdown/config files;
read those directly. Re-check source when a document claims a feature is absent
or complete. Do not use an old test count as present-day validation evidence.

## Repository boundaries

- `src/figaroh/` is the installable library. Robots, models and user scripts live
  in sibling `figaroh-examples`; keep robot-specific logic there.
- Normal core PRs target `devel`; release PRs target `main`. Examples PRs target
  that repository's `main`. See CONTRIBUTING for hotfixes and back-merges.
- Core CI now has tests and lint in addition to docs. Hosted check results and
  branch protection must be verified independently; a workflow file is not proof
  of a green run or an enforced rule.
- Complete and validate one issue, then commit/push its focused branch and open
  a PR for maintainer review. Keep independent issues in separate PRs. Merge
  only after explicit maintainer approval and required checks pass on the
  current head; implementation authorization or green CI alone is insufficient.

## Implementation pitfalls

- `backends/pinocchio.py` and `backends/mujoco.py` both exist. `Robot.backend`
  lazily wraps its Pinocchio model; direct Pinocchio calls and model mutation
  still exist. Full backend independence is not implemented.
- `RobotIdentificationSystem.from_mjcf()` raises `NotImplementedError`.
  The `backend=` name in `from_urdf()` does not itself switch the Robot backend.
- Use the parameter conversion helpers: Pinocchio and FIGAROH inertial parameter
  ordering differs. Regressor rows are joint-major; enabled extra blocks and
  parameter keys must stay aligned.
- Projection/reconstruction are opt-in at runtime. Projection keeps raw and
  projected result dictionaries; choose the intended result stage explicitly.
  `picos` is currently a package dependency even when projection is disabled.
- URDF first-moment/inertia handlers are currently stubs. A successful write is
  not proof of a physically complete exported model; require reload checks.
- Library logging uses module loggers/NullHandler. Do not introduce root logging
  configuration or `print` into library workflows.
- Tests needing GUI or optional dependencies can skip. Report why; do not hide
  a failing regression by turning it into a skip.

## Documentation updates

Update canonical root documents; docs-site roadmap/architecture pages embed
those files. `docs/decisions/` owns design rationale and is linked from the site.
Keep acceptance criteria/task status in issues and release history in CHANGELOG.
Check references when files move; update the roadmap only when its outcome or
priority changes. Preserve historical decisions with explicit status/supersession.

`.github/skills/` and `ROADMAP_PRIVATE.md` are ignored local material.
`site/`, `dist/` and `.codegraph/` are generated; do not commit them.
