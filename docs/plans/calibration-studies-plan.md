# Calibration studies plan

**Status:** In progress. Proposed 2026-10-09; tracker core #167 (package C5) with
examples #96–#101; backlash (#71) and suspension (#70) are existing issues.
Phases 0–3 are done: every study issue (#96–#101, #70, #71, #172) and every
data issue (#104–#114) is closed. The last study, TALOS motion capture
(examples [#172](https://github.com/thanhndv212/figaroh-examples/issues/172)),
closed 2026-10-26 by examples PR #173. Remaining: the C5 closing review on
core #167.

**Scope:** improve the existing real-data calibration and identification
studies in `figaroh-examples` so that each meets the standard set by the TIAGo
motion-capture reference (C2/C4). No new calibration features are added to
FIGAROH core. Every study gets a simulated fixture with known truth.

## Studies

| Study | Robot | Measurement | Data in figaroh-examples |
| --- | --- | --- | --- |
| Motion-capture reference | TIAGo | Qualisys marker position | 4 frozen sessions (2021-11) |
| Motion capture | TIAGo Pro | Qualisys marker pose | 3 sessions: 2026-07-01 (44), 07-02 (94), 08-05 (48) |
| Table contact | TALOS | Flush contact with one table (height, roll, pitch) | Left 21 + 9, right 29 + 9 postures (2022-10/11); provenance and contact evidence: [#113](https://github.com/thanhndv212/figaroh-examples/issues/113) |
| Hand-eye | UR10 | Flange camera observing a fixed chessboard (6D) | `ur10/data/calibration.csv`, 23 postures, one session; frozen (examples #99, PR #140) |
| Motion capture across hardware | TIAGo | Qualisys, OptiTrack, Vicon | Qualisys 2021-11 and 2021-10-26 ([#104](https://github.com/thanhndv212/figaroh-examples/issues/104), PR #139), OptiTrack 2023-09-12 ([#105](https://github.com/thanhndv212/figaroh-examples/issues/105), PR #144), Vicon 2023-09-20 runs 1508 and 1607 ([#106](https://github.com/thanhndv212/figaroh-examples/issues/106), PR #145), seven square-motion validation runs ([#107](https://github.com/thanhndv212/figaroh-examples/issues/107), PR #146); all frozen |
| Head camera, chessboard on hand | TIAGo | Head camera observing a hand-held chessboard | Sessions 2023-10-27 to 2023-11-27, two robots ([#108](https://github.com/thanhndv212/figaroh-examples/issues/108), PRs #150, #151; PAL tool CSVs are the camera source; 10-24 and 10-25 not usable); frozen |
| Backlash surface | TIAGo | Absolute minus relative encoder | 2023-07-24 trajectory (training) and 2023-03-09 session ([#109](https://github.com/thanhndv212/figaroh-examples/issues/109), PR #154; validation); arm_5/arm_6 joint inspections ([#110](https://github.com/thanhndv212/figaroh-examples/issues/110), PR #157; diagnostic); frozen |
| Suspension base | TIAGo | Base motion and force-plate wrench | Eleven CREPS Vicon runs 2023-09-20 ([#111](https://github.com/thanhndv212/figaroh-examples/issues/111), PR #158); 22 OptiTrack oscillation runs 2023-07/09 without force plate ([#112](https://github.com/thanhndv212/figaroh-examples/issues/112), PR #171); frozen |
| Motion capture | TALOS | Qualisys, 3 bodies (waist, upper body, end effector) | Left 56 training, right 39 validation, right 23 confirmation static postures, 2022-03-15 ([#114](https://github.com/thanhndv212/figaroh-examples/issues/114), PR #147; body attachments unknown); frozen. Study: [#172](https://github.com/thanhndv212/figaroh-examples/issues/172) |

TIAGo Pro's three sessions are all the data that exists. UR10 has one session.

## The study standard

Each study meets all eight requirements. The TIAGo motion-capture reference
already does.

1. **Frozen data.** Derived data shipped in `figaroh-examples`, sha256 recorded
   in a `protocol.yaml`, and a data README covering sensor, frame convention,
   dates, posture counts and known defects. Shipped files contain derived data
   and code only.
2. **Unified config** with an explicit `parameters.estimation.method`.
3. **Simulated fixture.** A known truth drawn at the study's real postures,
   using the study's own measurement model, a noise grid and seeds 0–4. It is
   judged on held-out postures against the noise-free truth and reports
   parameter recovery as z-scores (template: `examples/tiago/calibration_truth.py`).
4. **Held-out protocol.** Training, validation and confirmation sets fixed
   before any fit; posture groups repeated, new and out of range; methods never
   chosen on the confirmation sets (template:
   `docs/development/tiago-mocap-heldout-protocol.md` in figaroh-examples).
   A single-session study uses k-fold within the session and says so.
5. **Baseline** that fits only the unknown frames (registration, or table and
   contact offset), never the raw nominal model.
6. **Run provenance** with `report.html` under `results/runs/`.
7. **Tests** on simulated and real data, plus a golden output.
8. **Development note** `docs/development/<study>.md` covering design,
   results, limits and reproduction.

## Current state

✓ done, ~ partial, ✗ missing. Partial data means shipped but not frozen, or
only some sessions shipped. Rows for finished studies show their state at
closure. The suspension study writes
goldens and its note but no `results/runs/` record or `report.html`; the
maintainer accepted this as a disclosed limit on 2026-10-25 (negative result).

| Study | 1 Data | 2 Config | 3 Simulated | 4 Held-out | 5 Baseline | 6 Provenance | 7 Tests | 8 Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TIAGo mocap (reference) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TIAGo Pro mocap (done: examples #97, PR #135) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TALOS table contact (done: examples #98, PR #138) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| UR10 hand-eye (done: examples #99, PR #140; k-fold within one session) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TIAGo multi-hardware (done: examples #100, PRs #148, #149) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TIAGo head camera (done: examples #101, PRs #152, #153) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TIAGo backlash surface (done: examples #71, PRs #159, #160; validation-level) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TIAGo suspension (done: examples #70, PRs #161–#164; no suspension parameter identified) | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ |
| TALOS mocap (done: examples #172, PR #173; disagrees with table contact) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

## Phases

| Phase | Work | Why this order |
| --- | --- | --- |
| 0 — Shared tools | Generic simulation harness (measurement model, truth sampler, noise model) generalized from `calibration_truth.py`; generic held-out runner driven by `protocol.yaml`; fix #99 (identifiable set depends on random draws) and #116 (platform-dependent QR ties) | Every later headline number moves when the identifiable set changes; the TALOS two-chain held-out height error already ranges 4.7–12.3 mm across draws |
| 1 — Data already shipped | TIAGo Pro, TALOS table contact, UR10 hand-eye | Most studies brought to standard for least effort |
| 2 — Data to import | TIAGo multi-hardware, TIAGo head camera, TALOS motion capture | Each starts with a data audit that can end in "not usable" |
| 3 — Beyond geometry | Backlash surface, suspension base | Both depend on the accepted geometric reference and on Phase 2 OptiTrack data |

Each data import is its own figaroh-examples issue, attached as a sub-issue of
its study (or of the tracker for a new study), and meets the bar set by the
TIAGo identification shipment (examples #69): source recordings named by
sha256 and not distributed, a typed extractor that re-creates the shipped files
byte for byte, frozen roles and hashes in `protocol.yaml`, a data README and an
inventory entry. Data issues: [#104](https://github.com/thanhndv212/figaroh-examples/issues/104)–[#114](https://github.com/thanhndv212/figaroh-examples/issues/114). Milestones: C5 in
[figaroh-plus](https://github.com/thanhndv212/figaroh-plus/milestone/23) and
[figaroh-examples](https://github.com/thanhndv212/figaroh-examples/milestone/19).

Phase 0 changes shared code: plan it with the `architect` agent first and keep
the TIAGo reference numbers as a regression check. Phase 1 studies can go to
the `implementer` agent.

## Study by study

### TIAGo Pro motion capture (Phase 1)

- **Have:** three Qualisys sessions. A fit on 48 samples gives 7.1 mm and
  1.9° RMSE; 22 of 38 parameters significant at 3σ.
- **Do:** fix the split now — train 2026-07-02 (94), validate 07-01 (44),
  confirm 08-05 (48). Freeze, unified config, registration baseline, golden
  output, note. Decide on the flagged outlier before fitting.
- **Simulated fixture:** TIAGo truth model on TIAGo Pro's chain and the 94
  training postures; pose measurement with noise estimated from Qualisys
  plateaus.
- **Question:** does the TIAGo pattern repeat on a second robot (free full
  parameters overfit, regularized methods help, real data favours more
  parameters than the simulation)?

### TALOS table contact (Phase 1)

- **Have:** real held-out by day; synthetic fixture at one noise setting;
  real-data tests. Held-out height error 5.8 mm (left), 8.0 mm (right).
- **Do:** unified config with `estimation.method`; baseline fitting only the
  table and contact offset; `report.html` in provenance; golden output; report
  the spread across random draws until #99 is fixed.
- **Simulated fixture:** extend `generate_synthetic_data.py` to the 5-seed
  noise grid (encoder noise in degrees) with every estimation method; add a
  slightly tilted or curved table as a model-mismatch case.
- **Question:** 57 parameters from 21–29 contacts gives 0.48 mm on training
  against 5.76 mm held out. How much of that gap do `map` and `map_cv` close,
  and what is the real gain over the table-only baseline?

### UR10 hand-eye (Phase 1)

- **Have:** 23 postures, one session, 6D chessboard pose plus 6 joint angles;
  unified config, golden output and provenance. Sensor not documented.
- **Do:** document sensor and frame convention, freeze. Held-out is k-fold
  within the session, labelled as such. Baseline: camera-in-flange and
  board-in-base only.
- **Simulated fixture:** truth with joint errors, camera-in-flange and
  board-in-base; anisotropic chessboard pose noise (larger along camera depth
  and in rotation about the board plane); the 23 real postures.
- **Question:** with 23 postures and 6D measurements, which estimation method
  is reliable? A second recorded session would make this a full held-out study.

### TIAGo motion capture across hardware (Phase 2)

- **Have:** Qualisys 2021 (the reference). OptiTrack 2023-09-12 and Vicon
  2023-09-20 recordings not imported. The Vicon calibration bag needs
  `rosbag reindex`; the OptiTrack rigid body was redefined mid-session, so
  each run needs its own tool registration.
- **Do:** audit, extract derived posture files, freeze. Fit each system;
  then apply the 2021 Qualisys model to the 2023 data. Check clock lag on any
  continuous-motion runs.
- **Simulated fixture:** one truth seen through each system's noise level,
  tool registration error and clock lag, to set how large a cross-system
  difference hardware alone produces.
- **Question:** is the calibration independent of the measurement system?
  Hardware and date are confounded (arm_5 encoder shifted ~90 mrad between
  2021 and 2023), so the simulation sets the hardware-only expectation first.

### TALOS motion capture (Phase 2)

- **Have:** three torso-arm sessions from 2022-03-15, shipped and frozen as
  static postures ([#114](https://github.com/thanhndv212/figaroh-examples/issues/114),
  PR #147): left 56 (training), right 39 (validation), right 23 (confirmation).
  Body attachment links, base pose and sensor hardware are unknown.
- **Do:** study issue [#172](https://github.com/thanhndv212/figaroh-examples/issues/172):
  resolve or disclose the unknown attachments, pre-register (roles span two
  arms, so only the shared torso transfers to held-out data), simulated
  fixture, frames-only baseline, comparison with table contact.
- **Question:** does a measured TALOS calibration agree with the table-contact
  result, whose observations are assumed rather than measured?
- **Result** (examples PR #173): no. `map` was selected by k-fold (4.58 mm
  against 8.04 mm for the registration-only baseline). Held-out validation and
  confirmation are neutral: the shared torso parameters stay at their prior.
  Table-contact and mocap fits predict poses 15.3 mm apart. Disclosed limit:
  every model has about 37.6 mm error on the confirmation session, cause
  unexplained.

### TIAGo head camera, chessboard on hand (Phase 2)

- **Have:** sessions 2023-10-24 to 2023-11-27 and PAL's reference
  calibration, not imported. The 10-24 and 10-25 bags hold joints only; the
  11-07 OptiTrack runs have chessboard flips; the head chain and the camera
  pose absorb each other (gauge problem).
- **Do:** audit, drop or repair flipped samples, extract camera-observed
  chessboard poses, freeze. Fix the gauge (hold the head chain, or estimate it
  relative to the camera). PAL's calibration as a second baseline.
- **Simulated fixture:** truth with head chain, arm chain, camera-in-head and
  board-on-hand. It must show the gauge problem and confirm the fix before any
  real fit.
- **Question:** can the head camera replace motion capture for calibrating the
  arm? Sessions with both systems compare them on the same postures.

### TIAGo backlash surface (Phase 3)

- **Have:** one trajectory, per-joint R² 0.93–0.996 on the encoder
  difference. figaroh-examples#71: sign label reversed, no held-out run, does
  not transfer on arm_5 and arm_6.
- **Do:** fix the sign label; hold out a second trajectory; choose polynomial
  degree and switch steepness by cross-validation; then the end-effector test —
  geometric calibration with and without the surface on the held-out
  motion-capture sessions.
- **Simulated fixture:** a physical deadband and hysteresis model as truth.
  Does the surface recover it, transfer to new motions, and do full geometric
  parameters absorb it when it is left out?
- **Question:** does the surface lower the 2–3 mm floor on held-out sessions
  and stop the full geometric model from absorbing backlash?

### TIAGo suspension base (Phase 3)

- **Have:** one Vicon and force-plate log. OptiTrack oscillation sessions
  (2023-07, 2023-09, with and without weight) and a second Vicon session not
  imported. figaroh-examples#70: the fit explains only the My moment.
- **Do:** import a training session and held-out sessions with and without
  weight; config, provenance, golden output, note. Baseline: rigid base.
- **Simulated fixture:** known 12 spring-damper parameters with wrench noise
  at the real base motions, plus one misspecified truth (coupled axes or a
  nonlinear spring) to see whether it reproduces the "only My" signature.
- **Question:** is a single 6-DoF spring-damper enough?

## Decisions

- Improve existing studies; no new FIGAROH calibration features for now.
- UR10 `calibration.csv` is the hand-eye data, from one session.
- The three TIAGo Pro sessions are all the data that exists.
- Every study gets a simulated fixture.
- Shipped files contain derived data and code only.

## Risks

- Phase 0 changes shared code; the TIAGo reference numbers are the regression
  check.
- Single-session studies (UR10) can only use k-fold within the session; their
  notes must say their conclusions are weaker.
- Raw-data defects (Vicon bag index, OptiTrack flips, redefined rigid body,
  unsynchronized clocks) can each block a Phase 2 import.

## Next steps

1. ~~Open one tracker in figaroh-plus with one issue per study (in
   figaroh-examples) and one for Phase 0.~~ Done: core #167, examples #96–#101.
1. ~~One issue per data transfer and the C5 milestones.~~ Done 2026-10-10:
   examples [#104](https://github.com/thanhndv212/figaroh-examples/issues/104)–[#114](https://github.com/thanhndv212/figaroh-examples/issues/114).
1. ~~Phase 0 shared harness.~~ Done 2026-10-14: examples #96 by examples PR #133.
1. ~~Phase 1 TIAGo Pro and TALOS table contact.~~ Done: examples #97 by
   examples PR #135 (2026-10-15); examples #113 by examples PR #134 and
   examples #98 by examples PR #138 (2026-10-18).
1. ~~Phase 1 UR10 hand-eye.~~ Done 2026-10-19: examples #99 by examples
   PR #140. First Phase 2 import done the same day: examples #104 by
   examples PR #139.
1. ~~Phase 2 data imports and studies.~~ Done: #105–#107 (PRs #144–#146),
   then study #100 (PRs #148, #149); #114 (PR #147); #108 (PRs #150, #151),
   then study #101 (PRs #152, #153).
1. ~~Phase 3.~~ Done: data #109–#111 (PRs #154, #157, #158), study #71 (PRs
   #159, #160), study #70 (PRs #161–#164), data #112 (PR #171, 2026-10-25).
1. ~~TALOS motion-capture study.~~ Done 2026-10-26: examples #172 by
   examples PR #173.
2. C5 closing review on core #167: integration evidence in the closing
   comment, maintainer acceptance, then the tracker closes.
