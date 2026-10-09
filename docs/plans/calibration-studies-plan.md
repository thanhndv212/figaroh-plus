# Calibration studies plan

**Status:** Proposed, 2026-10-09. Tracker and issues not opened yet.

**Scope:** improve the existing real-data calibration and identification
studies in `figaroh-examples` so that each meets the standard set by the TIAGo
motion-capture reference (C2/C4). No new calibration features are added to
FIGAROH core. Every study gets a simulated fixture with known truth.

## Studies

| Study | Robot | Measurement | Data in figaroh-examples |
| --- | --- | --- | --- |
| Motion-capture reference | TIAGo | Qualisys marker position | 4 frozen sessions (2021-11) |
| Motion capture | TIAGo Pro | Qualisys marker pose | 3 sessions: 2026-07-01 (44), 07-02 (94), 08-05 (48) |
| Table contact | TALOS | Flush contact with one table (height, roll, pitch) | Left 21 + 9, right 29 + 9 postures (2022-10/11) |
| Hand-eye | UR10 | Flange camera observing a fixed chessboard (6D) | `ur10/data/calibration.csv`, 23 postures, one session |
| Motion capture across hardware | TIAGo | Qualisys, OptiTrack, Vicon | Qualisys only; OptiTrack (2023-09-12) and Vicon (2023-09-20) not imported |
| Head camera, chessboard on hand | TIAGo | Head camera observing a hand-held chessboard | Not imported (sessions 2023-10-24 to 2023-11-27) |
| Backlash surface | TIAGo | Absolute minus relative encoder | One trajectory (2023-07-24) |
| Suspension base | TIAGo | Base motion and force-plate wrench | One Vicon log; OptiTrack 2023-07 and 2023-09 sessions not imported |

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
only some sessions shipped.

| Study | 1 Data | 2 Config | 3 Simulated | 4 Held-out | 5 Baseline | 6 Provenance | 7 Tests | 8 Note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TIAGo mocap (reference) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TIAGo Pro mocap | ~ | ✗ | ✗ | ✗ | ✗ | ~ | ~ | ~ |
| TALOS table contact | ~ | ✗ | ~ | ✓ | ✗ | ~ | ~ | ~ |
| UR10 hand-eye | ~ | ~ | ✗ | ✗ | ✗ | ✓ | ~ | ✗ |
| TIAGo multi-hardware | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| TIAGo head camera | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ | ✗ |
| TIAGo backlash surface | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ~ |
| TIAGo suspension | ~ | ✗ | ✗ | ✗ | ✗ | ✗ | ~ | ~ |

## Phases

| Phase | Work | Why this order |
| --- | --- | --- |
| 0 — Shared tools | Generic simulation harness (measurement model, truth sampler, noise model) generalized from `calibration_truth.py`; generic held-out runner driven by `protocol.yaml`; fix #99 (identifiable set depends on random draws) and #116 (platform-dependent QR ties) | Every later headline number moves when the identifiable set changes; the TALOS two-chain held-out height error already ranges 4.7–12.3 mm across draws |
| 1 — Data already shipped | TIAGo Pro, TALOS table contact, UR10 hand-eye | Most studies brought to standard for least effort |
| 2 — Data to import | TIAGo multi-hardware, TIAGo head camera | Each starts with a data audit that can end in "not usable" |
| 3 — Beyond geometry | Backlash surface, suspension base | Both depend on the accepted geometric reference and on Phase 2 OptiTrack data |

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

1. Open one tracker in figaroh-plus with one issue per study (in
   figaroh-examples) and one for Phase 0; acceptance criteria are requirements
   1–8.
2. Plan Phase 0 with the `architect` agent; fix the TIAGo Pro split in
   parallel.
