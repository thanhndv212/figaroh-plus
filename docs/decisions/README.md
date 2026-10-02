# Design decisions and proposals

These documents preserve rationale, alternatives and experiments. The current
implementation is described in [ARCHITECTURE.md](../../ARCHITECTURE.md); priorities
are in [ROADMAP.md](../../ROADMAP.md), task status in linked issues.
Historical checklists and counts inside older documents are dated evidence.

## Index

| Document | Kind / scope |
|---|---|
| [Pinocchio version support](pinocchio-version-support.md) | Accepted dependency range and explicit 3.7/4.1 compatibility profiles; native dependency alignment and validation policy. |
| [External tool comparisons](external-tool-comparisons.md) | Research and mixed implementation history: calibration composition, dynamics refinement and reporting. Reporting code exists; individual remaining proposals need issue-level acceptance criteria. |
| [TIAGo calibration and port review](tiago-calibration-and-port-review.md) | Historical analysis and port proposals; some redistribution/export work shipped. Suspension follow-up below supersedes that portion of the port plan. |
| [TIAGo suspension/backlash examples](tiago-suspension-backlash-examples.md) | Implemented example design and limitations; generic core promotion remains separate. |
| [Modular linear/residual terms](modular-linear-residual-terms-plan.md) | Proposed interface and promotion gates; not a current API contract. |
| [Examples improvement plan](figaroh-examples-improvement_plan.md) | Historical implementation checklist; re-check the examples revision before selecting remaining work. |
| [URDF exporter](urdf_exporter.md) | Accepted design with partial implementation and documented deviations; first-moment/inertia support is still an open correctness gate. |
| [Validation quality report](validation-quality-report.md) | Historical implementation plan; current usage is documented in the reporting guide and implementation history in the changelog. |
| [Sim-to-real deployment](sim2real-modelbased-deployment.md) | Research proposal, not a committed delivery milestone. |

## New records

Use the [template](template.md) for a new public interface, dependency, module
boundary or result-schema decision. Record `Proposed`, `Accepted`, `Rejected`
or `Superseded`, a date, the related issue and the scope of acceptance. **Accepted
means the decision was chosen; it does not mean the implementation is complete.**
Track implementation in issues and completion in the changelog/roadmap gate.

Use stable filenames. If a decision changes, add a superseding record and link
both directions; preserve the old reasoning. Small bug fixes normally need the
issue's cause/reproducer/regression evidence rather than a new design document.
