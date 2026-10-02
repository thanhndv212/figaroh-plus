## Problem and resulting behavior

Link the issue and milestone/track. Describe the concrete before/after outcome.
Normal core PRs target `devel`; release/hotfix PRs target `main`.

## Validation evidence

Record commands, environment/revisions, pass/fail/skip counts and relevant
before/after metrics. State the V0–V3 levels run and explain missing coverage.
For core/examples changes, identify the tested revision pair and artifacts.

## Review checklist

- [ ] Acceptance criteria met; regression or behavioral coverage included
- [ ] Required checks pass on the current head; skips explained
- [ ] CHANGELOG entry and affected user docs updated
- [ ] Architecture/roadmap/decision record updated when their facts change
- [ ] No unrelated code or generated artifacts included
- [ ] Maintainer review and explicit approval received before merging

After merging into `devel`, close the linked issue explicitly if GitHub does not
close it automatically; update milestone status only after its exit gate passes.
