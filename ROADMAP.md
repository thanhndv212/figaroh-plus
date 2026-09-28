# FIGAROH Roadmap

**Source audit: 2026-09-28. Latest version in the package: 0.4.8.**

FIGAROH turns robot measurements into calibrated, identified models with
explicit validation and export evidence. The next delivery priority is a
complete identification → physical model → held-out verification → export
workflow in `figaroh-examples`.

This file owns priorities and milestone exit criteria. [GitHub issues](https://github.com/thanhndv212/figaroh-plus/issues)
own individual work items; [CHANGELOG.md](https://github.com/thanhndv212/figaroh-plus/blob/main/CHANGELOG.md) owns release history;
[ARCHITECTURE.md](https://github.com/thanhndv212/figaroh-plus/blob/main/ARCHITECTURE.md) describes current implementation; and
[design decisions](https://github.com/thanhndv212/figaroh-plus/blob/main/docs/decisions/README.md) explain proposals and trade-offs.
The [contribution workflow](https://github.com/thanhndv212/figaroh-plus/blob/main/CONTRIBUTING.md) defines how work reaches `devel`
and then `main`. The docs site embeds this file directly.

## Current evidence

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

## Delivery order

Milestone IDs below are stable references for issues. `Now` means the next
integration priority; `Next` means queued behind its dependencies; `Later`
and `Research` are unscheduled. Completion requires the exit criteria and
linked validation evidence, not just the presence of a module.

| Milestone | Priority | Outcome | Exit criteria |
|---|---|---|---|
| M0 — Engineering baseline | Now | Consistent docs, branch rules, and automated quality checks | Source audit reconciled; contributor/templates in place; core tests, changed-file hooks, critical lint and docs checks pass on the PR; deploy only from `main`; lint debt recorded |
| M1 — Trustworthy physical-model workflow | Next | One reproducible example from measurements through deployable inertials | Named joints/units and torque provenance; separate fit/validation data; explicit raw/projected/reconstructed outputs; meaningful physical verdict; exported URDF reloads and reproduces intended dynamics; archived metrics and report |
| M2 — Composable calibration and identification | Later, after M1 | Smaller pipeline stages and reusable residual/regressor terms | One accepted interface decision; existing examples retain numerical behavior; one new composed example passes held-out validation |
| M3 — Backend parity and broader examples | Later, after M1 | Supported backend operations have measured parity | Capability matrix backed by tests; public API selects the actual backend; representative examples and same-environment accuracy/runtime comparisons; unsupported paths fail explicitly |

M0's implementation has passed hosted checks in the draft PR stack and awaits
integration into `devel`; see the audit record for dated results. M1–M3 describe
delivery outcomes, not promised release versions or dates.
Select a version when the scope and compatibility impact are known.

## Work to split into issues

Create focused issues from these candidates as they enter `Ready`; keep their
acceptance criteria and changing status in the tracker. For work spanning both
repositories, use a parent issue in `figaroh-plus` and linked implementation
issues/PRs in `figaroh-examples`, with the tested commit pair recorded.

| Candidate | Milestone | Evidence or acceptance gate |
|---|---|---|
| Legacy lint and formatting cleanup | M0 follow-up | Full pre-commit currently fails; remove the recorded debt by module before promoting the advisory full-tree audit to a required gate |
| Regressor tests that skip on incompatible mocks | M0 follow-up | Replace obsolete fixtures with meaningful assertions; two baseline tests currently skip on `TypeError` |
| Differentiate every active velocity coordinate | M1 correctness | `calculate_first_second_order_differentiation()` still loops over `range(nq - 1)`; reproduce for a fixed-base final joint and assess `nq != nv` and variable sample times before fixing |
| Full inertial URDF export | M1 correctness | First moments and inertia tensors survive export/reload; mass/CoM conventions and inertia reference frames are tested |
| End-to-end reference example | M1 | Connect the existing projection, reconstruction, verification, export and archive APIs; demonstrate improvement on held-out data |
| Backend selection in `RobotIdentificationSystem` | M3 correctness | `from_urdf(backend=...)` currently stores a name while loading the usual Robot; selection must change the executing backend or reject unsupported choices |
| Optional backend/solver coverage | M0/M3 | Record tested dependency versions and skips; broaden OS/Python coverage after the initial Python 3.12 gate |

Bug severity takes precedence over milestone order. A reproducible wrong-result
bug can be fixed immediately without starting the rest of its milestone.

## Existing tracks and design material

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
