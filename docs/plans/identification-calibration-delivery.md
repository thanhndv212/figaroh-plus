# Identification and calibration delivery plan

**Status: Tracker setup approved; execution details under review — revision 4, 2026-10-02.
Implementation status refreshed 2026-10-09 (after D4 closure and the #61 merge)** (see [implementation status](#implementation-status)).
Confirmed user preferences: advance calibration alongside dynamic identification;
TIAGo mocap is the first calibration reference, with TALOS contact as the
second consumer/regression target. The user authorized GitHub milestone/issue creation on 2026-10-02 and requested
one-by-one delivery. Tracker setup does not accept unresolved public interfaces,
change solver defaults, or approve automatic implementation/merges. Review issue
readiness before selecting implementation work. Existing approved work and open PRs
retain their own review gates.

## Purpose and scope

Deliver two trustworthy workflows within the existing package structure:

1. Measurements → dynamic identification → physical model → held-out effort
   verification → inertial URDF export/reload → archived report.
2. Measurements → geometric calibration → identifiable corrections → held-out
   pose/contact verification → geometric export/reload → archived report.

The workstreams proceed in parallel through focused PRs, not simultaneous
edits to shared workflow classes. Their shared foundations are provenance,
units/frames, explicit result stages, validation and reproducible artifacts.
Dynamic and geometric solvers remain mathematically distinct.

The [roadmap](../source/further_reading/roadmap.md) owns priority and milestone outcomes.
This document owns the proposed work breakdown, dependencies, alternatives
and review sequence. Once approved, GitHub issues own changing task status
and detailed acceptance criteria. [Architecture](../source/concepts/architecture.md) continues
to describe existing code, not these proposed contracts. Existing design
records remain authoritative for their scope; this plan does not supersede
the modular-terms, Pinocchio-support or log-Cholesky decisions.

## Evidence motivating the plan

| Observation at 2026-10-02 | Consequence |
| --- | --- |
| Core #32 reproduces missing final-joint acceleration | Signal correctness precedes new dataset-based numerical claims |
| UR10/TIAGo sampling/filter assumptions disagree | Add explicit timing and derivative provenance before refitting |
| Exact base-preserving SDP returns no accepted model on the three robot datasets | Diagnose constraints/scaling; distinguish reconstruction from direct constrained estimation |
| All dataset log-Cholesky fits exhaust 200 evaluations | Keep #23–#25 conditional; investigate #30 rather than promote finite candidates |
| Per-link projection succeeds on UR10/TX40 but changes the base fit | Report it as a separate method and select exported stages explicitly |
| Inertial URDF first-moment/tensor handlers are stubs | A file-write success cannot close physical-model delivery |
| Examples CI fails on missing Hey5 geometry and TALOS 3.7 held-out regression | Make assets reproducible and retain numerical regression gates |
| Current examples have robot-specific loaders and existing reporting/export helpers | Improve adapters and reuse primitives before proposing a new framework |

Sources: core issues #20/#22–#25/#30/#32, core PR #31, examples issues
#11/#12/#14 and draft PR #13. The examples dynamic report and supplemental
analysis are still local additions on the PR branch. Reports include failed
solver statuses and preprocessing limitations; they are not scientific go
approval. Current issue/PR status must be checked when work starts.

## Ownership within the current repositories

| Concern | Existing owner | Proposed change boundary |
| --- | --- | --- |
| Dynamics preprocessing and regression | `figaroh/identification/{identification_tools,base_identification,parameter}.py` and `tools/regressor.py` | Fix numerical contracts and expose diagnostics through existing workflow |
| Physical checks, projection, reconstruction | `identification/{physical_consistency,reconstruction}.py` | Keep distinct operations; add a research constrained-fit comparator before production design |
| Geometric preprocessing/solving | `calibration/{base_calibration,calibration_tools,data_loader,config}.py` | Fix proven problems; retain frame conventions and existing subclasses |
| Export and export verification | `tools/{urdf_exporter,export_validation,geometric_calibration_export}.py` | Reuse handlers/checks; complete missing inertial semantics |
| Reports and archives | `tools/{identification_report,report,compare_report,run_archive}.py`, `utils/results_manager.py` | Reuse output machinery; add explicit stage/provenance fields through an approved contract |
| Robot I/O and conversions | `figaroh-examples/examples/<robot>/utils/` | CSV adapters, current-to-effort mapping and robot-specific assumptions stay here |
| Robot models/config/data | `examples/<robot>/{urdf,config,data}` and shared `models/` | Preserve originals; document sources, frames, sample rates and redistribution constraints |
| Research comparisons | Core private `docs/development/spikes/`; examples `benchmarks/` | Explicit source/commit dependency; never present as supported package API |
| Reference user workflows | Existing robot entry points under `examples/<robot>/` | Integrate validated primitives after research acceptance; avoid a competing CLI |

No `TrajectoryData` class was found in the current source audit. A typed data
contract is a proposal, not an existing API. First inventory existing
trajectory dictionaries and calibration measurement structures. Extend those
or introduce one narrow type only after an ADR demonstrates two consumers,
conversion rules and backward-compatible adapters. Keep dynamic trajectory
samples and geometric pose/contact observations as separate payloads.

## Dependency structure

```mermaid
flowchart TD
    R["Scope and contract review"]
    W1["<b>W1: Fixtures</b> · #33 ✅<br/>✓ ex#15 ex#16 ex#43 ex#59"]
    W2["<b>W2: TALOS regression</b> · #34 ✅<br/>✓ ex#14 ex#48"]
    W3["<b>W3: Data/result contracts</b> · #35 ✅<br/>✓ #54 #55 #125 #131<br/>✓ ex#17"]
    W4["<b>W4: Engineering process</b> · #50 ✅<br/>✓ #56 #57 #58 #73 #81–#85<br/>✓ ex#18 ex#40 ex#45 ex#50<br/>✓ ex#56 ex#60 ex#76"]
    D1["<b>D1: Acceleration correctness</b> · #36 ✅<br/>✓ #32"]
    D2["<b>D2: Signal audit</b> · #37 ✅<br/>✓ ex#19 ex#20 ex#51"]
    D3["<b>D3: UR10 truth fixture</b> · #38 ✅<br/>✓ #142 #143<br/>✓ ex#21"]
    D4["<b>D4: Physical comparison</b> · #39 ✅<br/>✓ #59 ex#12 ex#22"]
    D5["<b>D5: Convergence research</b> · #40<br/>✓ #22 (revise) #30 (no-go)<br/>#155"]
    D6["<b>D6: Inertial export</b> · #41 ✅<br/>✓ #60 (PR #138)"]
    D7["<b>D7: Dynamic reference</b> · #42<br/>#61 (code merged, PR #165) #116<br/>ex#23 ex#68 ex#69"]
    C1["<b>C1: Frames/data audit</b> · #43 ✅<br/>✓ ex#24 ex#25"]
    C2["<b>C2: Calibration truth/holdout</b> · #44 ✅<br/>✓ #101 #102 #105 #110 #113<br/>✓ ex#26 ex#27 ex#67"]
    C3["<b>C3: Geometric export</b> · #45 ✅<br/>✓ #62 #111 #114 #123<br/>✓ ex#28"]
    C4["<b>C4: Calibration reference</b> · #46 ✅<br/>✓ #97 #98 #99 #119 #120<br/>✓ ex#29"]
    C5["<b>C5: Calibration studies</b> · #167<br/>ex#96–#101 ex#70 ex#71"]
    D8["<b>D8: Identification studies</b> (research)<br/>#157–#161"]
    S1["<b>S1: Minimum reporting contract</b> · #47 ✅<br/>✓ #63 #70 #100 #103<br/>✓ ex#30 ex#36"]
    U1["<b>U1: Onboarding acceptance</b> · #51<br/>✓ #66<br/>ex#33"]
    A["M1 acceptance review"]
    S2["<b>S2: Sensitivity/uncertainty</b> · #48<br/>✓ #107<br/>open: #64 ex#31"]
    S3["<b>S3: Improved experiment design</b> · #49<br/>#65 #90<br/>ex#32"]
    S4["<b>S4: Selective composition</b> · #52<br/>#67"]
    B1["<b>B1: Backend parity</b> · #53<br/>#68<br/>ex#34"]
    LC1["<b>LC1: Log-Cholesky production</b> · #20<br/>#23 #24 #25<br/>ex#11"]

    R --> W1
    R --> W2
    R --> W3
    R --> W4
    W1 --> D2
    D1 --> D3
    D2 --> D3
    D3 --> D4
    D4 --> D5
    D6 --> D7
    D4 --> D7
    D5 -. optional nonlinear path .-> D7
    D5 -. go decision .-> LC1
    W1 --> C1
    W2 --> C1
    C1 --> C2
    C2 --> C3
    C3 --> C4
    C4 --> C5
    D4 --> D8
    C5 -. more worked studies .-> U1
    W3 --> S1
    S1 --> D7
    S1 --> C4
    D7 --> U1
    C4 --> U1
    D7 --> A
    C4 --> A
    U1 --> A
    A --> S2
    S2 --> S3
    A --> S4
    W3 --> S4
    A --> B1

    classDef cleared fill:#c8e6c9,stroke:#2e7d32,stroke-width:2px,color:#1b5e20
    classDef ready fill:#fff3cd,stroke:#b8860b,color:#5c4400
    class R,W1,W2,W3,W4,D1,D2,C1,C2,C3,S1,D6,C4,D3,D4 cleared
    class D5 ready
    class D7 active
    classDef active fill:#fff3cd,stroke:#b8860b,stroke-width:3px,stroke-dasharray:6 3,color:#5c4400
```

Legend: **green ✅ = tracker cleared** (closure record confirmed); **amber =
ready**, all prerequisites cleared (dashed amber border = work in progress or
in review); uncoloured = blocked on an upstream package.
Each box lists its tracker number, then the issues in its milestone (core
first, then `ex#` = figaroh-examples). Issues marked ✓ are closed; unmarked
issues are open. Two open examples issues have no package yet: ex#70 (TIAGo
suspension real-data fit) and ex#71 (TIAGo backlash surface); both are
research examples outside M1.

### Implementation status

Status as of 2026-10-06, taken from the GitHub trackers. The trackers stay
authoritative; refresh this table when a tracker closes.

| Package | Status | Closed | Delivered by / remaining |
| --- | --- | --- | --- |
| **W1** fixtures | ✅ **Cleared** | 2026-10-03 | examples #44, #57, #61, #62: hosted examples CI on both Pinocchio profiles; every robot loads from a clean checkout |
| **W2** TALOS regression | ✅ **Cleared** | 2026-10-03 | examples #46, #54: examples #14 explained; deterministic TALOS fixtures and noise-floor held-out checks |
| **W4** engineering process | ✅ **Cleared** | 2026-10-03 | core #71, #74, #80, #86–#93; examples #37, #40, #47: templates, real-model regressor tests, lint debt removed, full pre-commit blocking |
| **D1** acceleration | ✅ **Cleared** | 2026-10-02 | core #69 (fixes #32) |
| **D2** signal audit | ✅ **Cleared** | 2026-10-03 | examples #19/#20 UR10 and TIAGo audits, examples #35 |
| **C1** frames/data audit | ✅ **Cleared** | 2026-10-03 | examples #24/#25 TIAGo and TALOS audits; its follow-ups #97–#99 were moved to C4 and closed there |
| **C2** calibration truth/holdout | ✅ **Cleared** | 2026-10-05 | accepted on core `a991bf3` + examples `c791ecd`; deferred items moved to #119/#120 (C4) |
| **C3** geometric export | ✅ **Cleared** | 2026-10-06 | accepted on core `efd9f69` + examples `2c75f50` |
| **W3** data/result contracts | ✅ **Cleared** | 2026-10-06 | core #126 (ADR, #54), #128/#130 (contract, #55), #132 (#131), #127 (#125); examples #84 (#17); accepted on core `4ec06d4` + examples `9a9e6d7` |
| **D3** UR10 truth fixture | ✅ **Cleared** | 2026-10-07 | examples #21 by examples #89/#91 (fixture, frozen protocol v1, UR10 example moved onto it); #142 by #144; #143 by #145 + examples #92 (both added 2026-10-07); accepted on core `2260786` + examples `b23b836` (core 846 passed; examples `validate.py` 16/16, pytest 255 passed) |
| D5 convergence research | Ready (optional) | — | #22 done via PR #31 (revise); #30 done via #154 (no-go: 8/24 gated fits exhaust 2000 evaluations); #155 (method change for reliable termination) ready, tracker #40 in review |
| **D6** inertial export | ✅ **Cleared** | 2026-10-06 | #60 by PR #138; accepted on core `0e6a78c` + examples `a3381bc` (core 817 passed; examples `validate.py` 16/16, pytest 237 passed); merged-body export left to D7 (#61) |
| **S1** reporting contract | ✅ **Cleared** | 2026-10-06 | core #63 (PR #134), #70, #100, #103 (PR #135); examples #30 (examples PR #85), #36 |
| **C4** calibration reference | ✅ **Cleared** | 2026-10-06 | examples #29 by examples #88 (`reference_run.py`); #119 by #141 (core support; TIAGo protocol keeps one marker); #120 by #140 + examples #87 (`map` replaces the coefficient); #97–#99 by #127, #137, #139 + examples #86; accepted on core `c80b5fc` + examples `88e2fc2`. Held-out marker RMSE 3.8–4.2 mm; export parity ≤ 2.1e-12 m |
| S2 uncertainty | In progress (M2) | — | core #107 calibration standard errors fixed; #64 open |
| U1 onboarding | In progress | — | core #66 guideline published; the TIAGo mocap walkthrough can use C4 now; acceptance still waits on D7 |
| **D4** physical comparison | ✅ **Cleared** | 2026-10-09 | #59 by PR #153 (2026-10-08); examples #12 by examples PR #13; examples #22 by examples PRs #94 and #95 |
| D7 dynamic reference | **In progress** | — | Library side merged 2026-10-09: #61 by PR #165 (`select_stage`, public `physical_fit`, `export_urdf`, `merged_bodies="subtract_fixed"`; core 926 passed, 6 skipped). #61 stays open until V2 evidence exists. #116 merged by PR #169 and examples #69 by PR #103 (2026-10-10). Open: examples #68, #23 (see [path forward](#path-forward)) |
| S3, B1, LC1 | Blocked | — | waiting on the upstream packages shown in the graph; S3 and B1 wait on D7 and S2 |
| S4 composition (M2) | Blocked | — | W3 cleared; still waits on accepted M1 references |

**Gate summary:** M0's delivery roster (W1, W2, W4) and M1.1 (D1, D2, C1)
are cleared. **The calibration workstream is complete through M1.4** (C2, C3,
C4). On the dynamic side D3, D4 and D6 have cleared. In M1.4, W3, S1 and C4 have
cleared and D7 is the only package on the critical path: its library side is
merged (#61, PR #165) and its example evidence is open. U1 acceptance waits on D7.

Housekeeping: the examples milestones for W1, D2, C1 and W4 have no open issues
but are still open on GitHub, so they can be closed. The `status:blocked` labels
on core #61 and tracker #42 were stale after the D4 closure and were set to
`status:ready` on 2026-10-09. Examples #23 stays `status:blocked` until examples
#68 and #69 land; its remaining blockers are listed in the [path forward](#path-forward).

Log-Cholesky #30 can use the independent analytic core fixture. D1 is fixed
(#69, merged 2026-10-02) and the D2 audits are complete, so real-dataset work
can start with D3. Inertial export
can be tested with known feasible parameters independently of solver success.
Calibration need not wait for dynamic optimization, but changes to shared
export/report code require paired regression coverage and serial integration.

### Path forward

Status as of 2026-10-09. The trackers stay authoritative.

**Critical path (D7 → U1 → M1 acceptance).** All upstream packages have cleared
(D3, D4, D6, S1, W3). What remains is example evidence and one reproducibility
bug.

| Step | Issue | State | What it needs |
| --- | --- | --- | --- |
| 1 | examples #69 — ship the `calibration_slow` and `calibration_weight` runs, default `validation_data_file`, payload scale check | **done** (PR #103) | The original 2021-07 bags and the exporter that reproduces the shipped run byte for byte. Do this first: #68's acceptance bar is held-out RMSE on `calibration_slow` no worse than 1.21 |
| 2 | examples #68 — velocity filter model, torso/wrist/arm_1 effort handling, Hey5 URDF, constants in config | ready | The same bags. Seven checklist items, each ships its derived data and analysis in the repository |
| 3 | core #116 — deterministic base-parameter selection | **done** (PR #169) | Independent of 1–2; can run in parallel. Lets the golden tests drop the basis-dependent masks added in PR #165 |
| 4 | examples #23 — headless reference command (UR10 truth protocol with `physical_fit` and export; TIAGo rejected-stage and `arm_7` merged-body cases) | blocked on 1–2 for TIAGo; UR10 part can start now | Needs #61's library side (merged). Acceptance is V3: full `validate.py` |
| 5 | core #61 — close with the V2 run linked | open | Library side done in PR #165; its acceptance criteria are met, its validation level (V2) waits on step 4 |
| 6 | D7 tracker #42 — closing review | open | Closes last, after steps 1–5 and the maintainer accepts the integration evidence |
| 7 | U1 acceptance (examples #33), then M1 acceptance review | blocked on D7 | Worked briefs checked against the accepted evidence |

**After M1 acceptance (M2/M3):** S2 sensitivity (core #64, examples #31), then S3
experiment design (core #65, #90, examples #32), S4 composition (core #67) and
B1 backend parity (core #68, examples #34).

**Found along the way (unscheduled, `delivery` label, no package).** These do not
block D7.

| Issue | Defect |
| --- | --- |
| core #163 | The physical-consistency projection projects the nominal model, not the fit. PR #165 leaves it unchanged, cannot select it as a stage, and reports relabel it |
| core #164 | The SDP reconstruction and LMI projection pass `max_seconds` to picos, which only knows `timelimit`. Fixed for `physical_fit` only |

**Identification study expansion (package D8, research, `status:planned`, outside
baseline M1).** D8 groups core #157–#161 and examples #115 under milestone D8 (no tracker yet).
Each extends the D4 physical comparison and none changes the baseline:

| Issue | Study |
| --- | --- |
| core #155 | Change the log-Cholesky fitting method for reliable termination (belongs to D5, not D8; `delivery`, ready) |
| core #157 | Geometric (log-det) regularization for the physical direct fit |
| core #158 | Geometry-derived physical constraints (mass, CoM, bounding ellipsoid) |
| core #159 | Manifold-optimization alternative to log-Cholesky |
| core #160 | Noise-bias estimators (IDIM-IV, DIDIM, TLS) in the physical comparison |
| core #161 | Bayesian physically consistent identification |
| examples #115 | TX40: trace the identification CSVs to the original recording and ship the author's 2021 reference identification (data for #160 and the D4 comparison) |
| examples #70, #71 | TIAGo suspension and backlash research examples; now Phase 3 of C5 (below) |

**Calibration study expansion (package C5, tracker core #167, `status:planned`).** The
[calibration studies plan](calibration-studies-plan.md) (proposed 2026-10-09)
brings further studies to the TIAGo motion-capture standard. Phase 0 first,
because every study uses its tools:

| Phase | Issue | Study |
| --- | --- | --- |
| 0 | examples #96 | Shared simulation and held-out harness (confirms the identifiable-set fixes #99 and #113 hold) |
| 1 | examples #97 | TIAGo Pro motion capture |
| 1 | examples #98 | TALOS table contact |
| 1 | examples #99 | UR10 hand-eye (k-fold within one session) |
| 2 | examples #100 | TIAGo motion capture across hardware (audit first) |
| 2 | examples #101 | TIAGo head camera, chessboard on hand (audit first) |
| 3 | examples #71, #70 | TIAGo backlash surface and suspension base (existing issues, to be attached to C5) |

C5 is separate from M1: no study depends on D7, and the TIAGo motion-capture
reference numbers are its regression check. Phase 2 needs the original 2023
recordings from the maintainer.

## Work-package milestones and PR boundaries

### Work-package milestone rules

Each work package is a **delivery milestone** beneath a roadmap outcome M0–M3.
M1.1–M1.4 are acceptance gates, not additional issue containers. Package IDs are
stable package identifiers, distinct from GitHub issue numbers. GitHub
milestones use these IDs; existing issues are reused and missing tasks have
explicit owning packages. The package table below defines their roadmap mapping.

A package can be marked **Cleared** only when:

1. Its reviewed issue roster is complete, including linked core and examples
   issues. Assign each issue one owning package; link dependencies elsewhere.
2. **Every included issue is resolved with its acceptance criteria met**, its
   required PRs reviewed/merged, and evidence linked. Closing an issue as a
   duplicate, deferred or “not planned” does not satisfy that obligation.
3. Package-level integration/acceptance evidence also passes on the declared
   revision pair. All issue closures alone are insufficient if the combined
   workflow still fails its gate.
4. A maintainer reviews the closure record and confirms the accepted outcome.

Before Ready, record owner, roadmap gate, issue roster, dependencies, acceptance
criteria and evidence location. At closure, retain issue/PR links, tested
revisions, commands, report and any limitations. Status is Draft → Ready → In
progress → In review → Cleared; blocked work retains its unmet dependencies.
An empty draft roster cannot be treated as a completed milestone.

Deferral requires an explicit reviewed scope change: preserve the original
issue and reason, move it to a named future package, and update the roadmap
mapping if its exit criteria change. Do not silently remove unresolved issues
to clear a package. Cross-repository milestones need one canonical roster
(parent issue/project record) linking both repositories, since their individual
GitHub milestone views do not provide a combined closure record.

D5 is a separate research milestone. Its issues must all meet the agreed
research criteria to clear D5, but D5 is not required for the supported baseline
M1 scope. Promoting the nonlinear path adds D5 and #23–#25 to that path's reviewed
roster; none are silently treated as completed by D7. Basic acquisition/design
guidance is part of U1; S3 is the later algorithmic improvement milestone.

| ID / canonical tracker | Roadmap outcome / gate | Outcome / justification | Repository and dependencies | Acceptance evidence |
| --- | --- | --- | --- | --- |
| [W1](https://github.com/thanhndv212/figaroh-plus/issues/33) | M0; enables M1.1 | Reproducible examples environment and fixtures | Examples; existing PR #13 plus focused asset follow-up | Clean checkout resolves required geometry; pinned sources/licenses; explicit dependency failures; no reliance on a developer ROS search path |
| [W2](https://github.com/thanhndv212/figaroh-plus/issues/34) | M0; enables calibration M1.1 | Explain TALOS Linux/3.7 regression | Examples #14, core companion only if cause is generic | Capture solver status, seeds, native versions and both chains; reproduce before/after; retain current improvement gate until independently justified |
| [W3](https://github.com/thanhndv212/figaroh-plus/issues/35) | M1 shared contract; M2 prerequisite | Working data/result contracts | Proposed core ADR after inventory of both workflows | Joint/order/frame/unit/timestamp provenance, observed vs reconstructed effort, masks, split IDs and optional fields; documented adapters preserve old users |
| [D1](https://github.com/thanhndv212/figaroh-plus/issues/36) | M1.1 dynamic | Correct every tangent acceleration coordinate | Core #32 | Analytic final-joint regression, constant/variable dt, invalid timestamps and documented `nq != nv` policy; affected examples rechecked |
| [D2](https://github.com/thanhndv212/figaroh-plus/issues/37) | M1.1 dynamic | Audit robot signal processing | Separate examples issues for UR10 and TIAGo | Recorded vs inferred timestamps; explicit filter rates/cutoffs/order; torque/current units/signs; trim/decimation index provenance; immutable raw files |
| [D3](https://github.com/thanhndv212/figaroh-plus/issues/38) | M1.2 dynamic | Fresh UR10 truth fixture and benchmark protocol | Examples; D1/D2 | Save verified true inertias, analytic q/dq/ddq, independent train/validation trajectories and noise seeds; no ground-truth parameter claims from old CSVs |
| [D4](https://github.com/thanhndv212/figaroh-plus/issues/39) | M1.2 dynamic | Fair physical-estimation comparator | Private core spike + linked examples experiment; D3 | Base OLS, exact reconstruction, direct LMI-constrained effort fit, per-link projection and log-Cholesky; common inputs; comparable objectives and extras policy; separate failures |
| [D5](https://github.com/thanhndv212/figaroh-plus/issues/40) | M1.2 optional nonlinear research | Convergence/scaling revision | Core #30 (no-go), #155; analytic fixture now, robot evaluation after D3/D4 | Objective/gradient histories, scaling/bounds/prior ablations, multiple starts and justified budget; preserve earlier protocol; the #30 go gates, carried unchanged into #155, govern go |
| [D6](https://github.com/thanhndv212/figaroh-plus/issues/41) | M1.3 dynamic | Complete inertial export | Core; prior mapping review and known-parameter fixture | Mass/first moments/CoM, origin vs CoM tensor and inertial rotation handled; export/reload matches intended RNEA and physical verdict; unsupported targets fail explicitly |
| [D7](https://github.com/thanhndv212/figaroh-plus/issues/42) | M1.4 dynamic | Accepted dynamic reference workflow | Core integration + examples; D4/D6/S1; D5 only for optional nonlinear path | Explicit selected stage, fit and genuine validation splits, per-joint units, physical/solver verdict, exported model and archive; successful supported baseline can ship without nonlinear go |
| [C1](https://github.com/thanhndv212/figaroh-plus/issues/43) | M1.1 calibration | Audit geometric data/frames and reproduction | Examples with focused core fixes; W1/W2 | Named observation frames, translation/rotation units, timestamps, measurement source and split policy; TIAGo and TALOS current behavior captured |
| [C2](https://github.com/thanhndv212/figaroh-plus/issues/44) | M1.2 calibration | Calibration synthetic truth + real held-out reference | Examples; C1 | Known synthetic corrections recovered in identifiable coordinates; held-out pose/contact errors on unused postures; diagnose gauge, priors and solver failures |
| [C3](https://github.com/thanhndv212/figaroh-plus/issues/45) | M1.3 calibration | Validate redistribution and geometric export | Existing core QR/export helpers + examples; C2 | Calibrated-model FK matches exported/reloaded model; parameter/frame semantics retained; PAL runtime YAML checked where relevant; metrology frames are not silently embedded |
| [C4](https://github.com/thanhndv212/figaroh-plus/issues/46) | M1.4 calibration | One accepted calibration reference workflow | Examples TIAGo mocap selected; C2/C3/S1 | One command, training/held-out report, identifiable vs redistributed corrections, export and archive; contact-only evidence labeled separately from full pose metrology |
| [S1](https://github.com/thanhndv212/figaroh-plus/issues/47) | M1.4 shared reporting prerequisite | Minimum reporting contract | Existing core report/archive modules; W3; enables D7/C4 | Separate solver/data/fit/physical/export verdicts, explicit failed/fallback stages, paired revisions/hashes, old result compatibility |
| [S2](https://github.com/thanhndv212/figaroh-plus/issues/48) | M2 follow-up diagnostics | Sensitivity and uncertainty | Core diagnostics + examples; stable D7/C4 | Noise/initialization trials, singular spectrum and parameter correlations; observable combinations reported; no unjustified full-parameter certainty |
| [S3](https://github.com/thanhndv212/figaroh-plus/issues/49) | M2 follow-up experiment design | Improved experiment design | Existing `optimal/` + examples; S2 | Compare excitation/posture selection with a frozen baseline under declared constraints; improve independent validation evidence rather than only matrix condition |
| [W4](https://github.com/thanhndv212/figaroh-plus/issues/50) | M0 | Engineering process and regression coverage | Both repositories; existing validation/contribution rules | Reviewed workflow changes, focused legacy lint/mock coverage issues, reproducible checks; claimed CI automation verified on hosted runs |
| [U1](https://github.com/thanhndv212/figaroh-plus/issues/51) | M1.4 shared onboarding | General robot/dataset guide and worked briefs | Core docs + examples; draft now, acceptance uses D7/C4 | Fresh dynamic and TIAGo mocap walkthroughs plus missing-data case; supported commands, methods and limitations verified |
| [S4](https://github.com/thanhndv212/figaroh-plus/issues/52) | M2 composition | Selective reusable regressor/residual composition | Core + two examples consumers; W3 and accepted M1 references | Narrow ADR, two consumers, legacy numerical parity and additive compatibility; no broad refactor prerequisite |
| [B1](https://github.com/thanhndv212/figaroh-plus/issues/53) | M3 | Backend capability/parity review | Core `backends/`/`integration/`; accepted reference workflow | Actual backend selection or explicit rejection, supported operations matrix, same model/state comparisons and export/validation parity |

Potential new work must not duplicate #20/#22–#25/#30/#32 or examples
#11/#12/#14. If one package contains independent defects, split them into
individual issues/PRs rather than closing a broad issue with partial evidence.

## GitHub delivery register

Milestones and their canonical rosters were created on 2026-10-02. Open each
tracker for its complete issue roster, dependency links and acceptance evidence.
Core trackers combine both repositories; GitHub milestone percentages are local
to a repository and do not supersede the closure review. No due dates or
automatic merging are configured by this setup.

| Package | Core milestone | Examples milestone |
| --- | --- | --- |
| [W1](https://github.com/thanhndv212/figaroh-plus/issues/33) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/1) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/1) |
| [W2](https://github.com/thanhndv212/figaroh-plus/issues/34) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/2) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/2) |
| [W3](https://github.com/thanhndv212/figaroh-plus/issues/35) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/3) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/3) |
| [D1](https://github.com/thanhndv212/figaroh-plus/issues/36) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/4) | No examples-owned issues in this roster |
| [D2](https://github.com/thanhndv212/figaroh-plus/issues/37) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/5) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/4) |
| [D3](https://github.com/thanhndv212/figaroh-plus/issues/38) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/6) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/5) |
| [D4](https://github.com/thanhndv212/figaroh-plus/issues/39) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/7) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/6) |
| [D5](https://github.com/thanhndv212/figaroh-plus/issues/40) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/8) | No examples-owned issues in this roster |
| [D6](https://github.com/thanhndv212/figaroh-plus/issues/41) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/9) | No examples-owned issues in this roster |
| [D7](https://github.com/thanhndv212/figaroh-plus/issues/42) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/10) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/7) |
| [C1](https://github.com/thanhndv212/figaroh-plus/issues/43) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/11) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/8) |
| [C2](https://github.com/thanhndv212/figaroh-plus/issues/44) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/12) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/9) |
| [C3](https://github.com/thanhndv212/figaroh-plus/issues/45) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/13) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/10) |
| [C4](https://github.com/thanhndv212/figaroh-plus/issues/46) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/14) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/11) |
| [S1](https://github.com/thanhndv212/figaroh-plus/issues/47) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/15) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/12) |
| [S2](https://github.com/thanhndv212/figaroh-plus/issues/48) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/16) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/13) |
| [S3](https://github.com/thanhndv212/figaroh-plus/issues/49) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/17) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/14) |
| [W4](https://github.com/thanhndv212/figaroh-plus/issues/50) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/18) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/15) |
| [U1](https://github.com/thanhndv212/figaroh-plus/issues/51) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/19) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/16) |
| [S4](https://github.com/thanhndv212/figaroh-plus/issues/52) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/20) | No examples-owned issues in this roster |
| [B1](https://github.com/thanhndv212/figaroh-plus/issues/53) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/21) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/17) |
| [C5](https://github.com/thanhndv212/figaroh-plus/issues/167) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/23) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/19) |
| D8 (no tracker yet) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/24) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/20) |
| [LC1](https://github.com/thanhndv212/figaroh-plus/issues/20) | [Open milestone](https://github.com/thanhndv212/figaroh-plus/milestone/22) | [Open milestone](https://github.com/thanhndv212/figaroh-examples/milestone/18) |

**C5** (calibration studies, tracker core #167) and **D8** (identification studies,
core #157–#161) were added on 2026-10-09 beneath M2 and are outside baseline M1
closure. Their milestones were created on 2026-10-10. Each data transfer is its
own examples issue: C5 examples #104–#114 (sub-issues of their studies), D8
examples #115 (sub-issue of core #160).

**LC1** uses existing core #20 as the canonical conditional-production tracker
for core #23–#25 and examples #11. D5 owns research #22/#30/#155; its accepted go
decision is a dependency, not completion of those production issues. LC1 is
outside baseline M1 closure and introduces no additional production scope.

Initial queue (2026-10-02): D1 / core #32 was the first ready correctness fix,
delivered by #69. W1, W2, W3, W4, D1, D2, D3, D6, C1, C2, C3, C4 and S1 have since cleared
(see [implementation status](#implementation-status)). Current queue
(2026-10-09): calibration is done through M1.4, and D3 and D4 have cleared. The
critical path is D7 only: examples #69 and #68 (TIAGo inputs and held-out runs),
then examples #23 (reference command), then closing #61, #42 and U1 acceptance;
core #116 runs in parallel (see the [path forward](#path-forward)). On the optional path, D5 #30 recorded no-go and #155 (method change) is ready.
Select one issue, validate its focused change, open its PR, then await
review/merge approval.

## Experimental design decisions to review before coding

### Dynamic identification

Use two comparison levels. The first holds identical training-estimated extras
fixed to isolate inertial estimation; the second jointly estimates physical
extras where supported. Do not rank a joint solver against a frozen-extra
solver as though only parameterization changed. Define per-joint residual
scaling from training noise/units if whitening is proposed, and first retain
an unweighted compatibility baseline. TIAGo force and torque cannot become
one unlabeled physical RMSE.

Full SDP equality reconstruction preserves an estimated base vector; direct
constrained effort fitting may change it; per-link projection may change both
base fit and effort. Each needs its own objective, success definition and
stage label. Boundary solutions and massless fixed links need explicit policy.
No API is chosen just because one research script already contains a function.

Freeze ranks/tolerances, bounds, noise, initialization, iteration/evaluation
budget, timeout and quality metrics before measuring. Preserve the old
protocol. The existing #30 criteria (carried into #155) are not relaxed by this plan; any revised
criterion requires a reasoned decision before new measurements.

### Calibration

TIAGo mocap is the user-selected first real reference because existing redistribution
and runtime-export support can be reused. TALOS table-contact remains a second
consumer and a regression target, not a substitute for full 6D ground truth.
Use held-out postures/recordings with no fit-stage leakage. Address gauge/frame
ambiguity before interpreting individual geometric corrections or covariance.
Assess translated and rotational residuals separately unless a justified
measurement covariance defines a common whitened residual.

New camera/eye-hand, multi-marker or suspension features are candidates for a
separate reviewed scope after the reference baseline; they are not implicitly
included in this iteration. Historical port documents must be re-audited.

## General onboarding pipeline alongside the references

The two reference workflows are demonstrations, not the only supported route
for users. Add a reusable decision pipeline for any new robot/dataset without
creating a new runtime orchestration API in this planning phase:

1. Inventory available models, sensors, timestamps, measured/derived channels
   and independent validation; record missing information.
2. Define the application, observation model, parameter blocks, fixed quantities,
   gauge and limits on what the data can identify.
3. Select supported baseline/candidate methods by objective, constraints and
   noise/prior assumptions; label research paths separately.
4. Design acquisition and final validation together, reviewing excitation,
   observation coverage, feasibility and collection constraints.
5. Preserve raw measurements; audit clocks, units, frames, synchronization,
   derivatives/filter rates and raw-to-processed indices.
6. Fit under a frozen protocol; interpret training fit, observability, physical
   feasibility and solver termination separately.
7. Evaluate unused data, inspect computed/skipped verification checks, validate
   exported-model parity and archive reproducible evidence.

Core `docs/source/example_workflow.md` owns this scientific decision guide.
Existing task tutorials link to it and retain task-specific runnable material.
Examples `docs/new-example-guide.md` owns repository integration and
`docs/experiment-brief-template.md` provides a human-readable record copied into
`examples/<robot>/EXPERIMENT.md`. This is not a new YAML schema or an enforced
runtime contract. Robot README/data notes own actual measurements and commands;
CONTRIBUTING owns contributor/review rules.

**Work-package milestone U1 — general user onboarding** runs alongside D7/C4.
Draft the guide now; validate it later by walking through one fresh dynamic
example and TIAGo mocap, plus a missing-data case. Acceptance means the brief
leads to an explicit model/method choice, reproducible processing and applicable
validation without relying on undocumented reference assumptions. Check that
supported commands/links match the accepted core/examples revisions; label
unimplemented methods and missing evidence. Do not add an interactive wizard,
new config keys or generated acquisition code without separate review.

Round 2 reviews the brief's questions and evidence requirements alongside the
data/result contracts. Reference delivery PRs provide worked briefs and expose
where the generic guide needs correction. U1 does not wait for backend expansion
or a successful research-only log-Cholesky integration.

## Repository and documentation organization

Preserve existing directories and user entry points. Do not move datasets,
robot helpers or broad packages as a prerequisite to fixing known defects.

- Core `ROADMAP.md`: milestone order, priorities and exit criteria; one canonical
  file embedded by the docs site. Stable M0–M3 IDs remain.
- Core `docs/plans/`: this active execution proposal; `archive/` keeps old plans.
  Add an index with explicit Draft/Accepted/Historical status.
- Core `docs/decisions/`: narrow ADRs for new schemas/interfaces/objectives.
  Link to existing decisions rather than duplicating their reasoning.
- Core `docs/development/`: dated audits and validation evidence; private spikes
  remain marked experimental. Architecture changes only after facts change.
- Examples `benchmarks/`: research protocol, comparisons, curated small result
  records and dated reports. Preserve old protocols and source hashes.
- Examples `examples/<robot>/`: runnable supported recipes and robot data/config
  notes. Large generated runs remain outside source control or in CI artifacts.
- Each repository `CONTRIBUTING.md`: its execution/PR instructions. Core owns the
  shared rules; examples documents exact commands, dependency pairing, fixtures
  and its `main` PR target rather than copying the whole core workflow.

Existing examples `.github/` ignore rules and local-only CI material need a
focused tracked-file audit before adding more automation. A local file or
passing local test is not proof that it is tracked or executed remotely.

## Proposed development workflow improvements

These amendments are for review; CONTRIBUTING/AGENTS are not silently replaced
by this draft. Retain the established one-issue/one-PR rule and explicit merge
approval. Suggested additions:

1. **Discussion before implementation.** Agree on scope and acceptance evidence;
   approve public contracts through an ADR. Separate “approve plan,” “implement,”
   and “merge.” A research revise decision cannot be interpreted as API approval.
2. **Issue readiness.** Classify numerical bug, asset/dependency problem, research,
   API feature or maintenance. State core/examples owner, dependencies, current
   reproducer, immutable inputs, proposed tests and out-of-scope work. Avoid
   creating all speculative issues at once.
3. **Focused branches.** Core branches from current `devel`; examples from `main`.
   Keep discussion/report changes out of unrelated numerical PRs. Record tested
   core/examples commit pairs. Prefer isolated worktrees when an open PR is
   already using the main checkout.
4. **Before/after evidence.** Run a reproducer on baseline and changed revisions.
   Report command/exit code and classify data validity, solver convergence,
   physical consistency, held-out quality and export fidelity separately.
5. **Validation tiers.** V0 hooks/docs for documentation; V1 core tests for code;
   V2 affected examples for data/numerics/API; V3 both reference workflows and
   export invariants for milestone/release. Examples retains full `validate.py`
   phase checks and known-failure evidence. A failure cannot be hidden by a skip.
6. **CI organization.** Fast core checks plus Pinocchio 3.7/4.1 profiles; examples
   deterministic fixtures/smokes and dataset jobs with reproducible assets.
   Slow IPOPT/full workflow checks can be separately scheduled/manual, retaining
   their failure/timeout statuses. Collect research artifacts on failure without
   marking a failed regression green. This is a proposed CI split, not deployed.
7. **Review and merge.** Open each completed issue's focused PR. Maintainer reviews;
   merge only after explicit approval and required checks on the current head.
   Material scope changes require renewed review. No automatic merging.
8. **Post-merge cleanup.** Synchronize base branches; remove that PR's merged
   remote/local feature branch and worktree after checking for uncommitted or
   unique commits. Keep main/devel/releases and unrelated ancient branches.
9. **Release evidence.** An accepted method and passing CI are insufficient:
   verify supported examples, exported model reloads, archived reports, version
   metadata and dependency pair. Version/tag/publishing remains separate review.

Recommended issue metadata: type, area, priority, milestone, dependency links,
research/production designation and acceptance evidence. Recommended states:
Draft → Ready → In progress → In review → Done, with Blocked carrying its
actual dependency. Readiness labels and milestone/issue rosters are established by the approved
tracker setup. They are maintained manually; GitHub Project automation and
server-enforced branch rules are not implied.

## Review rounds and bounded implementation batches

### Round 1 — this proposal

Review the two parallel outcomes, package ownership, calibration reference,
method comparison boundaries and proposed sequence. Revise roadmap/plan locally.
The subsequent user instruction explicitly authorized tracker creation. No
code changes, commits/pushes or merges follow merely from accepting the plan.

### Round 2 — contracts and frozen protocol

Inventory current data/results and propose the smallest compatible extension.
Review each numerical objective, extras/weighting policy, exports and failure
semantics. Decide what metrics and datasets can actually establish. Update
narrow ADRs and the benchmark protocol before approving implementation.

### Round 3 — first delivery batch

Select only ready work: existing core #32 (delivered by #69); reproducible example asset fixes;
existing examples #14; separate robot sampling audits. Calibration frame audit
can proceed in parallel. **Done (2026-10-03):** this batch cleared as D1, W1, W2,
D2 and C1, together with W4. Open focused issues for the approved missing work.
Each completed issue produces a PR for review; dependent work waits or uses an
explicit tested commit pair. Review evidence before expanding the batch.

### Later batches

Fresh UR10 truth fixture and calibration truth/held-out fixture; physical solver
comparison and #30; independent inertial/geometric export gates; then dynamic
and calibration reference integration. Each batch ends with an evidence review,
not a guessed calendar deadline. Estimates follow a scoped issue audit; no
release date or package version is promised in this draft. Progress at
2026-10-06: the calibration batches (C2, C3, C4) and inertial export (D6)
have cleared; the UR10 truth fixture (D3) cleared on 2026-10-07;
the physical comparison (D4) cleared on 2026-10-09.

## Decisions still open

Calibration order is settled for this draft: TIAGo mocap first, TALOS contact
second. The following choices still need review:

1. Should the first joint physical estimator include bounded/positive friction
   and actuator parameters, or limit the initial comparison to fixed extras?
2. Which independent real recordings are available for final acceptance?
3. What is the minimum supported result/data extension after adapter inventory?
4. What measured physical boundary policy and export failure behavior should
   the public API expose?

Proposed remaining defaults: fixed-extra comparison followed by a separate
joint-estimation comparison; no claims of independent real validation until
recordings exist; additive compatibility adapters; failed/nonphysical stages
cannot silently become the selected exported model.
