# Contributing to FIGAROH

Use this workflow for features, bug fixes, documentation and maintenance.
Development commands run in the **`figaroh-dev` conda environment**. Start with
[ARCHITECTURE.md](https://github.com/thanhndv212/figaroh-plus/blob/main/ARCHITECTURE.md), [ROADMAP.md](https://github.com/thanhndv212/figaroh-plus/blob/main/ROADMAP.md), and
[validation.md](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/development/validation.md).

## One owner for each kind of information

| Information | Authoritative location |
|---|---|
| Product outcomes, priorities and milestone exit criteria | `ROADMAP.md` |
| Current module boundaries, data flow and limitations | `ARCHITECTURE.md` |
| Design choices, alternatives and research | `docs/decisions/` |
| Task status, bugs, acceptance criteria and dependencies | GitHub issues |
| What changed in a release | `CHANGELOG.md` |
| User instructions and API reference | `docs/source/` and source docstrings |
| Contribution and release procedure | This file |
| Agent environment, commands and implementation pitfalls | `AGENTS.md` |

The site includes root roadmap/architecture documents directly. Update their
canonical files rather than creating another copy. Keep detailed historical
planning in the archive with an explicit historical banner.

## Planning discussion

The [identification/calibration delivery plan](docs/source/further_reading/plans.md)
is a **draft for discussion**, not an implementation mandate. Its proposed
workflow amendments are reviewed alongside the roadmap before being adopted.
Current issue/PR/merge rules below continue to apply. Keep new API contracts
in focused decision records and avoid placing unreviewed proposals in the
current architecture description.

## Issue → branch → PR

1. Select an existing issue or open one using the feature/bug template. Link its
   milestone/track, define testable acceptance criteria and list dependencies.
   Bugs include a minimal reproducer, expected/actual results, environment and
   data provenance. Distinguish wrong numerical results, crashes, unsupported
   behavior and missing dependencies. For docs/CI work, a focused PR can carry
   the problem and acceptance criteria directly.
2. For a new public interface, schema, dependency or module boundary, add a
   [decision record](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/README.md) before or with the implementation.
   A design proposal is not evidence that the feature exists.
3. Branch from current `devel`: `feature/<issue>-<slug>`, `fix/<issue>-<slug>`,
   `docs/<slug>` or `chore/<slug>`. Keep a PR focused on one outcome.
4. Implement the acceptance criteria. Fixes include a regression test that
   exposes the original failure; features include meaningful behavioral tests.
   Preserve unrelated changes and avoid bundling numerical changes with broad
   formatting. For existing lint debt, clean the touched files or separate the
   necessary cleanup into a reviewable preliminary commit.
5. Run the [required validation](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/development/validation.md). Record commands,
   environment, pass/fail/skip counts and relevant domain metrics. Fix failures;
   do not hide them with test exclusions or new skip markers.
6. When one issue's implementation and local validation are complete, commit
   and push its focused branch and open a PR **against `devel`** using the template.
   Keep each completed issue in its own PR; independent work may continue on a
   separate branch. Include a changelog entry
   under `[Unreleased]` (including documentation reorganizations), user docs for
   changed behavior, and architecture/roadmap updates when their facts change.
7. The maintainer reviews each PR. Agents must wait for the maintainer's explicit
   approval before merging; passing checks or a request to implement/open a PR
   does not authorize a merge. Address review comments and rerun affected checks.
   After approval and required checks pass on the current head, squash-merge the
   focused feature/fix PR. If the approved scope changes materially, obtain a
   renewed review before merging. Use a
   Conventional Commit title, e.g. `fix(identification): preserve joint ordering`.
8. Close the issue with the merged PR linked. Because `main` is the default
   branch, `Closes #N` may not close it when merging into `devel`. Mark roadmap
   outcomes complete only after their exit criteria pass.

For triage, use one type (`bug`, `feature`, `docs`, `maintenance`), an area
(`identification`, `calibration`, `optimal`, `backends`, `examples`, `tooling`)
and a priority. Put only actionable, sufficiently specified issues into `Ready`;
then use `In progress` → `In review` → `Done`. These are recommended tracker
fields, not a claim that a GitHub Project or branch protection is configured.

## Repository and branch boundaries

| Branch/repository | Role |
|---|---|
| `figaroh-plus/devel` | Core integration branch; normal PR target |
| `figaroh-plus/main` | Stable release branch; release and urgent hotfix PRs |
| `figaroh-examples/main` | Separate examples integration branch; no `devel` branch is assumed |

Cross-repository work uses linked PRs and records the exact core/examples
revisions validated together. Land a compatible core API before examples depend
on it; document a minimum release or temporary commit dependency. Do not copy
robot-specific code into core just to avoid a second PR.

## Merge and release gates

Required core checks are `Tests (core)`, `Tests (pinocchio-4.1)`, `Tests (mujoco-3.9)`,
`Tests (mujoco-current)`, `Lint`, and `Docs`.
`Lint backlog` reports legacy full-tree debt and is advisory until that debt is
resolved. These names describe workflow jobs; a maintainer must select them in
GitHub branch protection/rulesets for server-enforced merge blocking.

Release sequence:

1. Confirm the selected roadmap exit criteria and linked validation evidence on
   `devel`. Run full tests, package build/metadata checks and the affected example
   workflows. Record unsupported or untested platforms explicitly.
2. Prepare the version and dated changelog on `devel` through a PR. Keep
   `pyproject.toml` and `src/figaroh/__init__.py` version strings aligned.
3. Open `devel` → `main` and merge with a **merge commit**, preserving shared
   history. Tag/release publishing is a separate maintainer action after the
   release checks; this repository's docs workflow does not publish to PyPI.
4. Merge `main` back into `devel` with a merge commit so it includes the release
   merge. This avoids repeated divergence at the next release.

For an urgent released-code bug, branch `fix/<issue>-<slug>` from `main`, target
`main`, validate and make a patch release, then back-merge into `devel`.
For normal fixes, use `devel` and the next planned release.

## Definition of done

- Acceptance criteria are met with evidence, including regression coverage.
- Required checks pass on the final PR head; skips and limitations are explained.
- Changelog and affected docs agree with the implementation.
- Architecture decisions are recorded when applicable.
- PR is merged, linked issue is closed, and milestone status reflects its gate.

A locally prepared branch or a passing local suite alone is not a completed
release or evidence of passing hosted CI.
