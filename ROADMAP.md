# FIGAROH Roadmap

**Evidence refresh: 2026-10-02. Priority changes below are a tracker scope approved; execution details under review, revision 4.**

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
- General URDF export supports several calibration/dynamics fields, but the
  first-moment and inertia-tensor handlers are still stubs. The physical-model
  export gate below is therefore open.
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
- [Examples PR #13](https://github.com/thanhndv212/figaroh-examples/pull/13) is draft.
  Its offline UR10/TX40/TIAGo runs complete on both Pinocchio profiles, while
  overall hosted jobs remain failed on missing Hey5 geometry and the 3.7 TALOS
  regression tracked in [examples #14](https://github.com/thanhndv212/figaroh-examples/issues/14).
  Completion of a dataset runner is not a passing physical-model workflow.
- Local supplemental comparisons distinguish exact base-preserving SDP
  reconstruction from per-link projection; exact reconstruction yields no
  accepted model and nonlinear dataset candidates exhaust 200 evaluations.
  A direct constrained SDP effort-fit comparator remains unmeasured.
- Existing calibration redistribution, report/archive and geometric export
  helpers provide a starting point for a parallel calibration reference.
  New data/result types or composition interfaces require an ADR first.

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

M0's contributor/CI foundations already exist; remaining example failures and
legacy lint/documentation debt keep its follow-up gates open. M1 is the proposed
next integration priority. Calibration reliability improvements run alongside
dynamic identification; M2 refers to the broader composition architecture,
not a requirement to postpone all calibration work until dynamics finishes.
No version or release date is assigned before scope and compatibility are reviewed.

### Delivery milestones beneath roadmap outcomes

M0–M3 describe strategic outcomes. Work packages are the delivery milestones
that achieve them; issues are their implementation units. Their canonical
scope, dependencies and closure policy are in the
[work-package milestone rules](docs/source/further_reading/plans.md#work-package-milestone-rules).

| Roadmap outcome / gate | Required delivery milestones in this draft |
| --- | --- |
| M0 reviewed delivery scope | W1 fixtures, W2 regression diagnosis, W4 engineering process |
| M1.1 inputs | D1–D2 dynamic signals; C1 calibration frames/data |
| M1.2 estimation | D3–D4 dynamic truth/comparison; C2 calibration truth/holdout |
| M1.3 export | D6 inertial export; C3 geometric export |
| M1.4 integrated workflows | W3 minimum contracts, S1 minimum reports, D7 dynamic reference, C4 calibration reference, U1 verified onboarding |
| M2 follow-up | S2 uncertainty, S3 experiment-design improvements, S4 selective composition; W3 is prerequisite groundwork |
| M3 backend parity | B1 capabilities and measured parity; broader ports need separately reviewed packages |
| Optional nonlinear research | D5 and linked production #23–#25 when that path is approved; excluded from baseline M1 closure |

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

## Work to split into issues

Create focused issues from these candidates as they enter `Ready`; keep their
acceptance criteria and changing status in the tracker. For work spanning both
repositories, use a parent issue in `figaroh-plus` and linked implementation
issues/PRs in `figaroh-examples`, with the tested commit pair recorded.

| Candidate | Milestone | Evidence or acceptance gate |
|---|---|---|
| Legacy lint and formatting cleanup | M0 follow-up | Full pre-commit currently fails; remove the recorded debt by module before promoting the advisory full-tree audit to a required gate |
| Regressor tests that skip on incompatible mocks | M0 follow-up | Replace obsolete fixtures with meaningful assertions; two baseline tests currently skip on `TypeError` |
| Signal-rate/provenance audits and fresh UR10 truth data | M1.1/M1.2 | Validate timestamps, filters and derivative source; preserve raw files; independent analytic truth and validation trajectories |
| Physical estimator comparison and convergence revision | M1.2 | Reuse #30; distinguish exact reconstruction, direct constrained effort fitting, per-link projection and log-Cholesky; freeze metrics/extra-column policies before measurement |
| Calibration held-out reference and export parity | Parallel M1.1–M1.4 | Reuse TIAGo redistribution/export; diagnose TALOS #14; independent posture validation and reloaded-model FK checks |
| General robot/dataset onboarding guide | Parallel M1 / proposed U1 | Data/model inventory → method choice → acquisition/processing → fit interpretation → held-out validation/export; reusable experiment brief; worked dynamic and TIAGo mocap cases; missing-data limitations explicit |
| Narrow data/result contract | M1/M2 | Inventory current adapters; ADR; two consumers; legacy dictionaries/configs remain compatible; proposed types not presented as existing API |
| Full inertial URDF export | M1 correctness | First moments and inertia tensors survive export/reload; mass/CoM conventions and inertia reference frames are tested |
| End-to-end reference example | M1 | Connect the existing projection, reconstruction, verification, export and archive APIs; demonstrate improvement on held-out data |
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
