# FIGAROH Roadmap

**Evidence refresh: 2026-10-02; delivery status refreshed 2026-10-06. Priority changes below are a tracker scope approved; execution details under review, revision 4.**

Review the [detailed delivery plan](docs/source/further_reading/plans.md)
before implementation. The user requested calibration improvements in parallel
with dynamic identification, with TIAGo mocap as the first calibration reference
and TALOS contact as the second consumer/regression target. Proposed work is not an accepted API, completed
feature or authorization to implement/merge.

FIGAROH turns robot measurements into calibrated, identified models with
explicit validation and export evidence. The proposed next delivery cycle advances two parallel reference workflows in
`figaroh-examples`: dynamic identification → physical model → held-out effort
verification → inertial export, and geometric calibration → identifiable
corrections → held-out pose/contact verification → geometric export.
Both must archive reproducible evidence and validate reloaded models.

This file owns priorities and milestone exit criteria. [GitHub issues](https://github.com/thanhndv212/figaroh-plus/issues)
own individual work items; [CHANGELOG.md](https://github.com/thanhndv212/figaroh-plus/blob/main/CHANGELOG.md) owns release history;
[ARCHITECTURE.md](https://github.com/thanhndv212/figaroh-plus/blob/main/ARCHITECTURE.md) describes current implementation; and
[design decisions](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/README.md) explain proposals and trade-offs.
The [contribution workflow](https://github.com/thanhndv212/figaroh-plus/blob/main/CONTRIBUTING.md) defines how work reaches `devel`
and then `main`. The docs site embeds this file directly.

## Current evidence and unresolved gates

- Pinocchio and MuJoCo backend modules exist. The high-level identification
  workflow still requires a URDF-based robot; `from_mjcf()` raises
  `NotImplementedError`. See the [backend boundary](https://github.com/thanhndv212/figaroh-plus/blob/main/ARCHITECTURE.md#backend-boundary).
- Projection and reconstruction utilities exist. Projection results already
  retain both `raw_parameters` and `projected_parameters`; their config
  dictionaries, including `max_seconds`, are passed through by the parser.
  The old roadmap's claims that raw parameters are overwritten and that this
  field is never forwarded are obsolete.
- General URDF export now writes identified first moments and inertia
  tensors and verifies them by reload ([#60](https://github.com/thanhndv212/figaroh-plus/issues/60),
  PR #138, merged 2026-10-06). D6 cleared on 2026-10-06, so the dynamic side
  of the M1.3 export gate is met; the whole gate closes at review.
- The September identification fixes (#11–#14: CAD inertia lookup, optional
  regressor blocks, filtering configuration, relative QR threshold) are in
  `[Unreleased]`. See the changelog for their exact scope.
- The local baseline before this workflow change was **499 passed, 20 skipped**
  in `figaroh-dev` (Python 3.12). This is dated evidence, not a permanent test
  count or a claim about hosted CI. Details and skipped coverage are in the
  [audit record](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/development/workflow-audit-2026-09-28.md).

### October findings

- The reproduced final-joint acceleration defect ([core #32](https://github.com/thanhndv212/figaroh-plus/issues/32),
  package D1) is fixed by [PR #69](https://github.com/thanhndv212/figaroh-plus/pull/69),
  merged into `devel` on 2026-10-02. UR10/TIAGo signal-rate assumptions still
  require separate examples audits (D2) before new scientific benchmark claims.
- [Core PR #31](https://github.com/thanhndv212/figaroh-plus/pull/31) is open with
  passing hosted checks and a **revise** feasibility decision. The private
  experiment is not a production solver. [#30](https://github.com/thanhndv212/figaroh-plus/issues/30)
  must resolve convergence/scaling before #23–#25 proceed.
- [Examples PR #13](https://github.com/thanhndv212/figaroh-examples/pull/13) is still draft.
  The Hey5 geometry failure was fixed by W1 (every robot loads from a clean
  checkout) and the 3.7 TALOS regression ([examples #14](https://github.com/thanhndv212/figaroh-examples/issues/14))
  was explained and fixed by W2. Completion of a dataset runner is not a
  passing physical-model workflow.
- Local supplemental comparisons distinguish exact base-preserving SDP
  reconstruction from per-link projection; exact reconstruction yields no
  accepted model and nonlinear dataset candidates exhaust 200 evaluations.
  A direct constrained SDP effort-fit comparator remains unmeasured.
- Existing calibration redistribution, report/archive and geometric export
  helpers provide a starting point for a parallel calibration reference.
  New data/result types or composition interfaces require an ADR first.

### Delivery progress (2026-10-06)

**Cleared:** W1, W2, W4 (M0 delivery roster); D1, D2, C1 (M1.1 inputs);
C2 (calibration side of M1.2); C3 and D6 (M1.3 export); W3 data/result
contracts and S1 minimum reporting (M1.4 groundwork); **C4 calibration
reference** (M1.4). The calibration workstream is complete through M1.4: one
command (`examples/tiago/reference_run.py`) fits TIAGo mocap, reports
held-out marker RMSE of 3.8–4.2 mm on three unused sessions, and exports a
URDF and PAL file that reload within 2.1e-12 m. The dynamic workstream is
now the critical path: D3 (UR10 truth fixture, frozen protocol v1) and D6 have
cleared; D4 is next, then D7 and U1 acceptance. Per-package evidence is in the
[implementation status](docs/source/further_reading/plans.md#implementation-status).

These are dated observations; verify heads and status before beginning work.
The [delivery plan](docs/source/further_reading/plans.md) maps
concrete work to existing modules and both repositories. Numerical limitations
must be reproduced and classified rather than waived to obtain a green check.

## Delivery order

Milestone IDs below are stable references for issues. `Now` means the next
integration priority; `Next` means queued behind its dependencies; `Later`
and `Research` are unscheduled. Completion requires the exit criteria and
linked validation evidence, not just the presence of a module.

| Milestone | Priority | Outcome | Exit criteria |
|---|---|---|---|
| M0 — Engineering baseline | Now, continuous | Reproducible review and validation in both repositories | Existing core gates retained; examples assets/dependencies reproducible; numerical regressions explained; review/merge rules and cleanup applied; exact tested commit pairs |
| M1 — Trustworthy model workflows | Now, proposed parallel tracks | Dynamic and geometric reference workflows with held-out verification and faithful export | Correct signals/frames/units; explicit fit and validation splits; identifiable/physical parameter diagnostics; selected output stage; exported model reload parity; archived report and provenance |
| M2 — Composable calibration and identification | Next, after reference contracts | Reusable data/diagnostic primitives and selective regressor/residual composition | Two concrete consumers; accepted narrow ADR; legacy/default numerical parity; additive result compatibility; no premature broad workflow rewrite |
| M3 — Backend parity and broader examples | Later, after reference acceptance | Supported backend operations have measured parity | Actual backend selection or rejection; capability matrix; same-input accuracy/runtime comparisons; supported export/validation path; unsupported capabilities explicit |

M0's delivery roster for this cycle (W1, W2, W4) has cleared: examples assets
and hosted CI are reproducible on both Pinocchio profiles, the TALOS regression
is explained, legacy lint debt is removed and the full pre-commit suite is
blocking. The M0 gate review itself is still to be recorded. M1 is the proposed
next integration priority. Calibration reliability improvements run alongside
dynamic identification; M2 refers to the broader composition architecture,
not a requirement to postpone all calibration work until dynamics finishes.
Versions follow the [release plan](#release-plan): each minor release ships one
accepted outcome; no release date is promised.

### Delivery milestones beneath roadmap outcomes

M0–M3 describe strategic outcomes. Work packages are the delivery milestones
that achieve them; issues are their implementation units. Their canonical
scope, dependencies and closure policy are in the
[work-package milestone rules](docs/source/further_reading/plans.md#work-package-milestone-rules).

| Roadmap outcome / gate | Required delivery milestones in this draft | Status at 2026-10-06 (✅ = cleared) |
| --- | --- | --- |
| M0 reviewed delivery scope | W1 fixtures, W2 regression diagnosis, W4 engineering process | ✅ **W1, W2, W4 all cleared** |
| M1.1 inputs | D1–D2 dynamic signals; C1 calibration frames/data | ✅ **D1, D2, C1 all cleared** |
| M1.2 estimation | D3–D4 dynamic truth/comparison; C2 calibration truth/holdout | ✅ **C2, D3**; D4 ready, partly in review (examples #12) |
| M1.3 export | D6 inertial export; C3 geometric export | ✅ **C3, D6** both cleared |
| M1.4 integrated workflows | W3 minimum contracts, S1 minimum reports, D7 dynamic reference, C4 calibration reference, U1 verified onboarding | ✅ **W3, S1, C4**; U1 in progress; D7 blocked |
| M2 follow-up | S2 uncertainty, S3 experiment-design improvements, S4 selective composition; W3 is prerequisite groundwork | ✅ W3 groundwork done; S2 in progress; S3, S4 blocked |
| M3 backend parity | B1 capabilities and measured parity; broader ports need separately reviewed packages | Blocked on M1 |
| Optional nonlinear research | D5 and linked production #23–#25 when that path is approved; excluded from baseline M1 closure | D5 ready (#30, PR #31 open) |

A delivery milestone is cleared only when **all included issues meet their
acceptance criteria**, required changes are reviewed/merged, and milestone-level
integration evidence is accepted. Deferred or administratively closed issues
are not completed work; scope changes require review and preserve the original
roster/history. Each roadmap gate closes only when all required delivery
milestones and its own exit criteria pass. M0's delivery roster is scoped per
cycle; ongoing maintenance does not prevent acceptance of that reviewed cycle.

GitHub milestones and canonical issue rosters are available in the
[delivery register](docs/source/further_reading/plans.md#github-delivery-register).
The optional LC1 tracker reuses Feature 3 #20 and remains outside baseline M1.

Track the chain in both directions: roadmap gate → delivery milestone → owning
issues → reviewed PRs → acceptance report. Issue/project records own current
status and evidence links; this roadmap does not duplicate live checklists.

### M1 review gates

| Gate | Dynamic workstream | Calibration workstream |
| --- | --- | --- |
| M1.1 — Inputs and reproducibility | Correct tangent derivatives; audited rates/conversions; synchronized channels; independent preprocessing | Audited observation frames, units and timestamps; reproducible TIAGo/TALOS baseline; explicit gauge/measurement assumptions |
| M1.2 — Verified estimation protocol | Fresh UR10 truth fixture; distinct base/SDP/projection/nonlinear objectives; frozen comparison and solver status | Synthetic correction truth in identifiable coordinates; genuine held-out postures; solver/gauge/regularization diagnostics |
| M1.3 — Faithful model export | Complete first-moment/tensor semantics; physical verdict and reloaded-model RNEA parity | Redistribution semantics; exported/reloaded FK parity; runtime correction output where relevant |
| M1.4 — Accepted reference workflow | Supported physical baseline → held-out effort report → export/archive; nonlinear integration only after go | Geometric fit → held-out pose/contact report → export/archive; contact-only evidence explicitly labeled |

The calibration and dynamic gates are reviewed independently. M1 closes when
all required delivery milestones above and both reference outcomes pass. A supported physical dynamic baseline can
ship without waiting for log-Cholesky go; the nonlinear method remains opt-in
and conditional. A failed/fallback model cannot silently become an exportable
production result. Future uncertainty and experiment-design work follows stable
reference baselines, with backend expansion queued behind measured parity.

## Release plan

Versions follow roadmap outcomes, not commit counts. A release ships what has
been accepted: its delivery packages are cleared, each with a closure record
and a core/examples revision pair, and those records are its evidence.

| Bump | When |
|---|---|
| **0.MINOR** (before 1.0) | One roadmap outcome or gate side is accepted |
| **0.x.PATCH** | Wrong-result or crash fixes with no behaviour change, from a `main` hotfix branch, then merged back into `devel` |
| **1.0.0** | M1 accepted (both reference workflows), public API declared, deprecated features removed; a release candidate first |
| **After 1.0** | M2 additions are minor releases; removing a deprecated feature is a major release |

A deprecated feature warns for at least one minor release before it is removed;
removals are batched into 1.0. Optional research (D5/LC1) ships as an opt-in
feature in the minor release after its go decision and never holds a release.

| Version | Outcome | Delivery packages | Status |
|---|---|---|---|
| 0.5.0 | M0 engineering baseline | W1, W2, W4, D1 | Released 2026-10-03 |
| 0.6.0 | Calibration reference workflow (M1.1–M1.4, calibration side) and shared foundations | D2, C1–C4, W3, S1; D6 inertial export and D3 fixture fixes, both accepted | Released 2026-10-07 ([release](https://github.com/thanhndv212/figaroh-plus/releases/tag/v0.6.0), [tracker #146](https://github.com/thanhndv212/figaroh-plus/issues/146)); examples tag `v0.6.0` |
| 0.7.0 | Dynamic reference workflow (M1.2–M1.4, dynamic side) | D4, D7; D5 if it gets a go decision | Next; D3 and D6 already shipped in 0.6.0 |
| 1.0.0 | M1 accepted | U1 acceptance, M1 review, deprecations removed | After 0.7.0 |
| 1.1, 1.2 | M2 composable calibration and identification | S2, S3, S4 | Later |
| 1.x or 2.0 | M3 backend parity | B1 | Later; 2.0 only if the backend API breaks |

Each release is tracked by a core release issue. It lists the packages in scope
and what is excluded, freezes `devel`, and records the release-candidate
evidence on one core/examples pair: core tests, hosted CI on both Pinocchio
profiles, examples `validate.py`, package build with `twine check` and a clean
wheel install, and the docs build. The release PR bumps both version strings
and dates the changelog with a scope paragraph, behaviour changes, migration
notes and limitations. The [release sequence](https://github.com/thanhndv212/figaroh-plus/blob/devel/CONTRIBUTING.md#merge-and-release-gates)
then merges `devel` into `main` with a merge commit (branch deletion off); one
build from the `vX.Y.Z` tag goes to PyPI and the GitHub release, with matching
sha256. The examples repository pins `figaroh>=X.Y,<X.(Y+1)` and is tagged
`vX.Y.Z` as the paired revision. Finally `main` is merged back into `devel`.

## Work to split into issues

Create focused issues from these candidates as they enter `Ready`; keep their
acceptance criteria and changing status in the tracker. For work spanning both
repositories, use a parent issue in `figaroh-plus` and linked implementation
issues/PRs in `figaroh-examples`, with the tested commit pair recorded.

| Candidate | Milestone | Evidence or acceptance gate |
|---|---|---|
| ✅ Legacy lint and formatting cleanup | M0 follow-up | **Done in W4** (#81–#85 via #87–#92); full pre-commit is a blocking gate (#93) |
| ✅ Regressor tests that skip on incompatible mocks | M0 follow-up | **Done in W4** (#58 via #80): obsolete mocks replaced with real-model coverage |
| ✅ Signal-rate/provenance audits and fresh UR10 truth data | M1.1/M1.2 | ✅ Audits **done in D2**; fresh UR10 truth data **done in D3** (examples #89, frozen protocol v1). Original gate: independent analytic truth and validation trajectories |
| Physical estimator comparison and convergence revision | M1.2 | Reuse #30; distinguish exact reconstruction, direct constrained effort fitting, per-link projection and log-Cholesky; freeze metrics/extra-column policies before measurement |
| ✅ Calibration held-out reference and export parity | Parallel M1.1–M1.4 | ✅ TALOS #14 diagnosed (W2), held-out reference (C2) and reloaded-model FK parity (C3) **done**; integrated TIAGo reference **done in C4** (examples #29 via examples #88) |
| General robot/dataset onboarding guide | Parallel M1 / proposed U1 | Data/model inventory → method choice → acquisition/processing → fit interpretation → held-out validation/export; reusable experiment brief; worked dynamic and TIAGo mocap cases; missing-data limitations explicit |
| ✅ Narrow data/result contract | M1/M2 | **Done in W3** (core #126/#128/#130, examples #84). Original gate: Inventory current adapters; ADR; two consumers; legacy dictionaries/configs remain compatible; proposed types not presented as existing API |
| ✅ Full inertial URDF export | M1 correctness | **Merged in D6** (#60, PR #138). Original gate: first moments and inertia tensors survive export/reload; mass/CoM conventions and inertia reference frames are tested |
| End-to-end reference example | M1 | Calibration side ✅ done in C4; dynamic side remains in D7. Connect the existing projection, reconstruction, verification, export and archive APIs; demonstrate improvement on held-out data |
| Backend selection in `RobotIdentificationSystem` | M3 correctness | `from_urdf(backend=...)` currently stores a name while loading the usual Robot; selection must change the executing backend or reject unsupported choices |
| Optional backend/solver coverage | M0/M3 | Record tested dependency versions and skips; broaden OS/Python coverage after the initial Python 3.12 gate |

Bug severity takes precedence over milestone order. A reproducible wrong-result
bug can be fixed immediately without starting the rest of its milestone.

## Existing tracks and design material

The active [delivery plan](docs/source/further_reading/plans.md)
contains work packages, dependency order and proposed workflow amendments.
Milestone/issue setup is authorized and linked from the plan; implementation
proceeds one ready issue at a time after its unresolved decisions are reviewed.
Track letters from the previous roadmap remain useful labels for issues.
Detailed historical tasks are retained in the [archived v2 roadmap](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/plans/archive/roadmap-v2.md);
its old dates and completion claims are not current commitments.

| Track | Retained scope | Delivery home / design reference |
|---|---|---|
| A — Algorithmic core | Physical consistency, reconstruction, CAD constraints, modular pipeline, model visualization; later online ID, friction models and FIM-based experiment design | M1/M2; [linear terms proposal](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/modular-linear-residual-terms-plan.md) |
| B — Backends | Pinocchio/MuJoCo parity, explicit API capabilities, benchmarks and CLI; later Genesis/IsaacSim | M3; [current backend limits](https://github.com/thanhndv212/figaroh-plus/blob/main/ARCHITECTURE.md#backend-boundary) |
| C — Reporting and quality | Verification, reports/archives; remaining optimal-task reporting and report-schema decisions | M0/M1; [comparison and reporting rationale](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/external-tool-comparisons.md) |
| D — Calibration composability | Residuals, staged solving, priors, camera intrinsics, sensor input and camera-YAML export | M2; [calibration design material](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/external-tool-comparisons.md) |
| E — Examples and robot ports | TALOS/TX40 script parity, TIAGo eye-hand work, suspension/backlash examples and export integration | M1/M3; [example audit](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/figaroh-examples-improvement_plan.md), [TIAGo review](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/tiago-calibration-and-port-review.md) |
| F — Deployment and sim-to-real | Data adapters, model-based control, rollout refinement and retargeting | Research; [deployment proposal](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/sim2real-modelbased-deployment.md) |

The SO-101 example added to `figaroh-examples` is useful simulation evidence for
gravity/friction identification. It does not establish complete inertial export
or physical-hardware validation for M1. Existing TIAGo suspension/backlash work
remains a research example; generic core promotion has separate gates in the
[composition proposal](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/modular-linear-residual-terms-plan.md).

## Review and completion

At milestone review, record the core/examples revisions, environment, commands,
pass/fail/skip counts and domain metrics using the [validation guide](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/development/validation.md).
A numerical improvement needs a comparable dataset and metric; a successful
URDF write needs a reload/round-trip check; a simulated result is labeled as such.

Update this roadmap when a milestone's outcome or priority changes. Update the
issue for task progress, the architecture for changed contracts, and the changelog
for shipped behavior. Research enters the delivery queue only after its scope,
dependencies and measurable exit criteria are accepted.
