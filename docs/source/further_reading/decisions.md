# Design Decisions

`docs/decisions/` in the repository holds design-rationale documents —
implementation plans, roadmaps, and comparison research. They're
intentionally **not** part of this built site (kept as a sibling of
`docs/source/`, which is the `docs_dir` MkDocs actually reads) since they're
working documents for contributors, not user-facing reference — but they're
worth knowing about if you want the reasoning behind a feature, not just
its usage docs.

| Document | What it covers |
|---|---|
| [`implementation-plan-calibration-validation-quality-report.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/implementation-plan-calibration-validation-quality-report.md) | FK validation + statistical quality matrix implementation plan |
| [`roadmap-mujoco-sysid-inspired-features.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/roadmap-mujoco-sysid-inspired-features.md) | Quality & reporting infrastructure roadmap — the design rationale behind the [Reporting & Verification](../reporting_and_verification.md) suite, including why the compare page is a static two-run artifact rather than a run history/dashboard |
| [`sysid-comparison-mujoco-vs-figaroh.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/sysid-comparison-mujoco-vs-figaroh.md) | Research note comparing MuJoCo's `sysid` tooling against FIGAROH's approach |

See also the top-level [`ARCHITECTURE.md`](../concepts/architecture.md) and
[`ROADMAP.md`](roadmap.md), which are user-facing and embedded directly in
this site.
