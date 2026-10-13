# Contributing

The contributor workflow is maintained alongside the code in [`CONTRIBUTING.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/CONTRIBUTING.md).

## Contributing to FIGAROH

Use this workflow for features, bug fixes, documentation and maintenance. Development commands run in the **`figaroh-dev` conda environment**. Start with [ARCHITECTURE.md](https://github.com/thanhndv212/figaroh-plus/blob/main/ARCHITECTURE.md), [ROADMAP.md](https://github.com/thanhndv212/figaroh-plus/blob/main/ROADMAP.md), and [validation.md](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/development/validation.md).

### One owner for each kind of information

| Information                                              | Authoritative location               |
| -------------------------------------------------------- | ------------------------------------ |
| Product outcomes, priorities and milestone exit criteria | `ROADMAP.md`                         |
| Current module boundaries, data flow and limitations     | `ARCHITECTURE.md`                    |
| Design choices, alternatives and research                | `docs/decisions/`                    |
| Task status, bugs, acceptance criteria and dependencies  | GitHub issues                        |
| What changed in a release                                | `CHANGELOG.md`                       |
| User instructions and API reference                      | `docs/source/` and source docstrings |
| Contribution and release procedure                       | This file                            |
| Agent environment, commands and implementation pitfalls  | `AGENTS.md`                          |

The site includes root roadmap/architecture documents directly. Update their canonical files rather than creating another copy. Keep detailed historical planning in the archive with an explicit historical banner.

### Planning discussion

The [identification/calibration delivery plan](https://thanhndv212.github.io/figaroh-plus/further_reading/plans/index.md) is a **draft for discussion**, not an implementation mandate. Its proposed workflow amendments are reviewed alongside the roadmap before being adopted. Current issue/PR/merge rules below continue to apply. Keep new API contracts in focused decision records and avoid placing unreviewed proposals in the current architecture description.

### Issue → branch → PR

1. Select an existing issue or open one with the *Feature / work item* or *Bug report* template (the same forms are used in figaroh-examples). Write it for a reader without local context: link its delivery package, define testable acceptance criteria and list dependencies as `#N` (same repository) or `thanhndv212/figaroh-examples#N` (the other one), which GitHub links. See [how to read delivery issues](https://github.com/thanhndv212/figaroh-plus/blob/devel/docs/plans/README.md#how-to-read-delivery-issues). Bugs include a minimal reproducer, expected/actual results, environment and data provenance. Distinguish wrong numerical results, crashes, unsupported behavior and missing dependencies. For docs/CI work, a focused PR can carry the problem and acceptance criteria directly.
1. For a new public interface, schema, dependency or module boundary, add a [decision record](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/README.md) before or with the implementation. A design proposal is not evidence that the feature exists.
1. Branch from current `devel`: `feature/<issue>-<slug>`, `fix/<issue>-<slug>`, `docs/<slug>` or `chore/<slug>`. Keep a PR focused on one outcome.
1. Implement the acceptance criteria. Fixes include a regression test that exposes the original failure; features include meaningful behavioral tests. Preserve unrelated changes and avoid bundling numerical changes with broad formatting. For existing lint debt, clean the touched files or separate the necessary cleanup into a reviewable preliminary commit.
1. Run the [required validation](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/development/validation.md). Record commands, environment, pass/fail/skip counts and relevant domain metrics. Fix failures; do not hide them with test exclusions or new skip markers.
1. When one issue's implementation and local validation are complete, commit and push its focused branch and open a PR **against `devel`** using the template. Keep each completed issue in its own PR; independent work may continue on a separate branch. Include a changelog entry under `[Unreleased]` (including documentation reorganizations), user docs for changed behavior, and architecture/roadmap updates when their facts change.
1. The maintainer reviews each PR. Agents must wait for the maintainer's explicit approval before merging; passing checks or a request to implement/open a PR does not authorize a merge. Address review comments and rerun affected checks. After approval and required checks pass on the current head, merge the focused feature/fix PR with a **merge commit** (not squash or rebase), so the PR's individual commits stay in `devel`'s history. If the approved scope changes materially, obtain a renewed review before merging. Use a Conventional Commit title, e.g. `fix(identification): preserve joint ordering`.
1. Close the issue with the merged PR linked. Because `main` is the default branch, `Closes #N` may not close it when merging into `devel`. Mark roadmap outcomes complete only after their exit criteria pass.

For triage, use one type (`bug`, `feature`, `docs`, `maintenance`), an area (`identification`, `calibration`, `optimal`, `backends`, `examples`, `tooling`) and a priority. Put only actionable, sufficiently specified issues into `Ready`; then use `In progress` → `In review` → `Done`. These are recommended tracker fields, not a claim that a GitHub Project or branch protection is configured.

### Repository and branch boundaries

| Branch/repository       | Role                                                               |
| ----------------------- | ------------------------------------------------------------------ |
| `figaroh-plus/devel`    | Core integration branch; normal PR target                          |
| `figaroh-plus/main`     | Stable release branch; release and urgent hotfix PRs               |
| `figaroh-examples/main` | Separate examples integration branch; no `devel` branch is assumed |

Cross-repository work uses linked PRs and records the exact core/examples revisions validated together. Land a compatible core API before examples depend on it; document a minimum release or temporary commit dependency. Do not copy robot-specific code into core just to avoid a second PR.

### Merge and release gates

Required core checks are `Tests (core)`, `Tests (pinocchio-4.1)`, `Tests (mujoco-3.9)`, `Tests (mujoco-current)`, `Lint`, and `Docs`. `Lint` runs every pre-commit hook on the full tree (the legacy debt was cleared in #81-#85). These names describe workflow jobs; a maintainer must select them in GitHub branch protection/rulesets for server-enforced merge blocking.

Release sequence:

1. Confirm the selected roadmap exit criteria and linked validation evidence on `devel`. Run full tests, package build/metadata checks and the affected example workflows. Record unsupported or untested platforms explicitly.

1. Prepare the version and dated changelog on `devel` through a PR. Keep `pyproject.toml` and `src/figaroh/__init__.py` version strings aligned. In the same PR, revise `README.md` against the release: rewrite "What's new", rerun the Quickstart commands, check every claim (features, supported platforms, links) against the code and CI, and remove anything stale. Do the same for the figaroh-examples README (compatibility row, robot table, `requirements.txt` pin) in a paired examples PR. Also check that the maintainer's copyright notices, both the source file headers and `LICENSE`, end at the release year. In both repositories this command must print nothing:

   ```
   git grep -nE "Copyright (\[2021-[0-9]{4}\] Thanh Nguyen|[0-9-]+ Thanh D\. V\. Nguyen)" \
     | grep -v "$(date +%Y)"
   ```

   Third-party notices (CNRS/Toward SAS, PAL Robotics fixtures) keep their own years.

1. Open `devel` → `main` and merge with a **merge commit**, preserving shared history. Tag/release publishing is a separate maintainer action after the release checks; this repository's docs workflow does not publish to PyPI.

1. Merge `main` back into `devel` with a merge commit so it includes the release merge. This avoids repeated divergence at the next release.

For an urgent released-code bug, branch `fix/<issue>-<slug>` from `main`, target `main`, validate and make a patch release, then back-merge into `devel`. For normal fixes, use `devel` and the next planned release.

#### Release candidate procedure (1.0.0)

The 1.0.0 soak uses release candidates `v1.0.0rc1`, `v1.0.0rc2`, and so on.

> **Waived for 1.0.0.** On 2026-10-12 the maintainer released 1.0.0 directly through the release sequence above, without a candidate (#192). The evidence of step 3 was still recorded on the release pair. The procedure is kept for a future major release. Steps 1–4 of the release sequence above apply unchanged to each candidate; the steps below add only what is specific to the RC.

1. **Freeze `devel`.** Announce the freeze on the release tracker issue. From then on only fixes for soak blockers land on `devel`, each as a focused PR.

1. **Open the release PR into `main`.** Set the version to `1.0.0rc1` in `pyproject.toml` and `src/figaroh/__init__.py`, add the dated changelog section, and update the README as in release sequence step 2. Merge `devel` → `main` with a merge commit, then tag `v1.0.0rc1` on that merge commit.

1. **Collect evidence on one core/examples pair.** Record both exact commit SHAs in the release PR and tracker issue, then record command, environment, and pass/fail/skip counts for each item:

1. core `python -m pytest -q -rs`;

1. core hosted CI green on the tagged commit, and examples CI green on both Pinocchio profiles (3.7.0 and 4.1.0) against that core revision;

1. examples `MPLBACKEND=Agg python validate.py` against that core revision;

1. core `python -m build`, then `twine check dist/*`, then install of the wheel into a clean environment (not the editable `figaroh-dev` install) and an import check;

1. `python -m mkdocs build` from the repository root.

1. **Build once and publish back to back.** Build the sdist and wheel once from the `v1.0.0rc1` tag. The maintainer runs `twine upload` personally; upload the same files to PyPI and immediately attach them to the GitHub pre-release:

   ```
   gh release create v1.0.0rc1 --prerelease --target <full 40-character commit SHA> dist/*
   ```

   Use the full commit SHA for `--target`, not a branch or short SHA. Do not rebuild between the two uploads.

1. **Verify checksums.** Compare the sha256 of the PyPI files with the sha256 of the GitHub assets (`gh release download v1.0.0rc1`) and with the local `dist/` files. All three must match before the RC is announced.

1. **Pin examples to the RC.** During the RC, the figaroh-examples dependency pin (`requirements.txt`, README compatibility row) is `figaroh>=1.0.0rc1,<1.1`, merged to that repository's `main` through its own PR.

1. **Soak with fixes only.** Only soak-blocking fixes land on frozen `devel`. Each fix is released as the next candidate (`v1.0.0rc2`, …) and repeats step 3 in full and steps 4–5 for the new tag.

1. **End the soak** when both conditions hold: no new blocking issue has been opened for the period the maintainer sets for this RC (`<N days>`, recorded on the tracker issue), and the examples CI passes against the RC installed from PyPI.

1. **Release 1.0.0.** Repeat the release sequence for version `1.0.0` (steps 1–3, same build-once and checksum rules, without `--prerelease`) and tag `v1.0.0`. Tag the paired figaroh-examples revision `v1.0.0`. The existing `>=1.0.0rc1,<1.1` pin already admits 1.0.0, so no pin change is needed unless the maintainer decides otherwise. Finish with the back-merge in release sequence step 4.

### Definition of done

- Acceptance criteria are met with evidence, including regression coverage.
- Required checks pass on the final PR head; skips and limitations are explained.
- Changelog and affected docs agree with the implementation.
- Architecture decisions are recorded when applicable.
- PR is merged, linked issue is closed, and milestone status reflects its gate.

A locally prepared branch or a passing local suite alone is not a completed release or evidence of passing hosted CI.
