<!-- Base: `devel` for normal work; `main` only for a release PR from devel or a hotfix.
     Title: Conventional Commits, e.g. "fix(identification): preserve joint ordering".
     It becomes the squash commit's subject. -->

Closes #
<!-- Merging into devel does not auto-close: close the issue after merge. Paired change? Link thanhndv212/figaroh-examples#… -->

## Why

<!-- The problem, with evidence (numbers, logs, a failing test). -->

## What

<!-- The change, by module. Call out behaviour or default changes explicitly. -->

## Validation

<!-- Levels from docs/development/validation.md. Tick what you ran and paste results.
     For a level you skipped, say which and why. -->

- [ ] V0 local: changed-file hooks, critical lint (+ `mkdocs build` for docs)
- [ ] V1 core suite: pass/fail/skip counts, regression test for a fix
- [ ] V2 example workflow: figaroh-examples commit, command, before/after metrics on the same data
- [ ] V3 milestone/release: exit criteria from ROADMAP
- [ ] CI green on the latest commit

## Docs and changelog

- [ ] `CHANGELOG.md` `[Unreleased]` entry, or not user-visible (reason: …)
- [ ] Docs / ARCHITECTURE / ROADMAP / decision record updated, or not needed

## Found along the way / limits

<!-- Issues noticed but not fixed here (open issues for them), and what this does not establish. -->

- [ ] Maintainer approval received before merging
