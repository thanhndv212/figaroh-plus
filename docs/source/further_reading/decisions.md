# Design Decisions

`docs/decisions/` in the repository holds design-rationale documents —
implementation plans, roadmaps, and comparison research. They're
intentionally **not** part of this built site (kept as a sibling of
`docs/source/`, which is the `docs_dir` MkDocs actually reads) since they're
working documents for contributors, not user-facing reference — but they're
worth knowing about if you want the reasoning behind a feature, not just
its usage docs.

As of 2026-08-31 this directory holds eight documents. Several previously
separate files (a `robot_calibration` comparison, a MuJoCo `sysid`
comparison, the quality/reporting roadmap that comparison produced, a TIAGo
calibration analysis, and a TIAGo port review) were merged into two combined
documents rather than kept as cross-linked siblings; a later TIAGo
suspension/backlash port-review follow-up was split back into two focused
documents (one done, one still proposed) rather than left as one growing
combined file — the table below reflects the current state, not the
historical filenames referenced by older commits.

| Document | What it covers |
|---|---|
| [`external-tool-comparisons.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/external-tool-comparisons.md) | Three parts in one file: **Part A** — comparison against `robot_calibration` (ROS 2 kinematic/sensor calibration) and the resulting calibration-composability adaptation roadmap. **Part B** — comparison against MuJoCo's `sysid` module. **Part C** — the full quality & reporting infrastructure roadmap Part B's comparison produced (the design rationale behind the [Reporting & Verification](../reporting_and_verification.md) suite, including why the compare page is a static two-run artifact rather than a run history/dashboard). Each part carries a verified implementation-status table. |
| [`tiago-calibration-and-port-review.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/tiago-calibration-and-port-review.md) | Two parts: **Part A** — deep structural/statistical analysis of the TIAGo/TIAGo Pro mocap calibration pipelines, including the base-parameter redistribution work now shipped in core. **Part B** — review of porting eye-hand calibration (still not started) and mobile-base/suspension identification (superseded by `tiago-suspension-backlash-examples.md` below, now done) from old branches. |
| [`tiago-suspension-backlash-examples.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/tiago-suspension-backlash-examples.md) | ✅ **Done.** TIAGo generalized-base suspension identification and empirical backlash-surface examples, shipped as standalone research examples in `figaroh-examples` — what was ported, corrections made versus the historical prototype (gravity torque feature, polynomial degree, structurally-zero column handling), and known remaining gaps (no held-out train/validation split yet). |
| [`modular-linear-residual-terms-plan.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/modular-linear-residual-terms-plan.md) | 🔬 **Proposed, not started.** Generic `LinearRegressorTerm`/`ResidualTerm`/`WeightPolicy` composition architecture for the identification layer (motivated by, but not required by, the suspension/backlash examples above), plus the still-unbuilt physical/stateful backlash model and its core-promotion gates. |
| [`figaroh-examples-improvement_plan.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/figaroh-examples-improvement_plan.md) | Cross-example audit of UR10/TIAGo/TALOS/Staubli TX40 + shared infrastructure — 35 of 39 items closed, 4 open (TALOS/Staubli script parity) |
| [`urdf_exporter.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/urdf_exporter.md) | URDF exporter plan & parameter-name registry — implemented with a few documented deviations from the original spec (function-based API, metrology-frame params not auto-applied, inertia-tensor handler still a stub) |
| [`validation-quality-report.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/validation-quality-report.md) | FK validation + statistical quality matrix implementation plan — fully implemented, effectively superseded by the reporting suite `external-tool-comparisons.md` Part C tracks |
| [`sim2real-modelbased-deployment.md`](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/sim2real-modelbased-deployment.md) | 🔬 **Ongoing research, not yet committed.** Positioning FIGAROH as a sim-to-real deployment/model-based-control toolbox for RL/IL policies and motion retargeting — a 6-phase plan, research citations, and open decisions (including how it should integrate with tools like mjlab and motion-retargeting libraries) |

See also the top-level [`ARCHITECTURE.md`](../concepts/architecture.md) and
[`ROADMAP.md`](roadmap.md), which are user-facing and embedded directly in
this site. `ROADMAP.md` §15 (References) links back to every document
above from the specific roadmap track it informs.
