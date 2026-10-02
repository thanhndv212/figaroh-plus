# Delivery plans

The [roadmap](../../ROADMAP.md) owns priorities and milestone outcomes.
Active plans describe proposed execution order and review gates; task status
belongs in GitHub issues and design contracts in `docs/decisions/`.

| Plan | Status |
| --- | --- |
| [Identification and calibration delivery](identification-calibration-delivery.md) | Revision 4, 2026-10-02; tracker setup approved; execution details under review |
| [Archived roadmap v2](archive/roadmap-v2.md) | Historical; dates/completion claims are not current commitments |

## How to read delivery issues

FIGAROH work spans two repositories, [figaroh-plus](https://github.com/thanhndv212/figaroh-plus/issues)
(core library) and [figaroh-examples](https://github.com/thanhndv212/figaroh-examples/issues)
(robot examples and datasets). Issue numbers repeat across them, so a reference to
the other repository is written `thanhndv212/figaroh-examples#N` or
`thanhndv212/figaroh-plus#N`; plain `#N` means the same repository. GitHub links
both forms, and `Closes …` only closes issues written this way.

**Roadmap milestones** (M0–M3) are the outcomes in the [roadmap](../../ROADMAP.md):
M0 engineering baseline, M1 trustworthy physical-model workflow, M2 composable
calibration and identification, M3 backend parity. Gates such as M1.3 are steps
within M1 defined in the [delivery plan](identification-calibration-delivery.md).

**Delivery packages** group the work toward a milestone. Each has a stable ID, one
*tracker* issue in figaroh-plus (`type:tracking`) and a GitHub milestone of the same
name in each repository that has work for it.

| Prefix | Area | Example |
| --- | --- | --- |
| W | Workflow foundations: fixtures, regressions, data contracts, process | W3 — data/result contracts |
| D | Dynamic identification | D6 — inertial URDF export |
| C | Geometric calibration | C2 — calibration held-out reference |
| S | Shared reporting, diagnostics, experiment design, composition | S1 — minimum reporting contract |
| U | User onboarding | U1 — general robot/dataset guide |
| B | Backends | B1 — backend capability review |
| LC | Conditional log-Cholesky production (outside baseline M1) | LC1 |

**Work issues** are the units a PR closes. They are sub-issues of their package
tracker and state Problem, Goal, Acceptance criteria, Out of scope and Dependencies
(the *Feature / work item* template). A tracker closes last, under the
[package closure rules](identification-calibration-delivery.md#work-package-milestone-rules).

**Labels**: `type:bug`, `type:feature`, `type:tracking`; `delivery` marks planned
work; status is one of `status:planned` (scoped, not yet startable),
`status:ready`, `status:blocked` (see the issue's "blocked by"), `status:in-review`
(a PR is open). Unscheduled issues carry no package.
