---
name: figaroh-setup-identification
description: 'Set up and run a FIGAROH dynamic parameter identification task end to end — fitting masses, inertias, friction and actuator inertias from recorded motion and joint torques. Use when the user wants to identify robot dynamics, has q/tau logs to fit, asks about base parameters, regressor conditioning, WLS refinement, or physical consistency.'
---

# Set Up a Dynamic Identification Task

_Arguments: robot name, and where the motion + torque logs live._

> **Paths.** `$FIGAROH_WS` is the workspace root — the directory holding the
> `figaroh` and `figaroh-examples` repos side by side. Set it once per shell:
> `export FIGAROH_WS="$(cd /path/to/your/workspace && pwd)"`.

**Prerequisite:** `figaroh-setup-env` exits 0. **Working directory for every command
below:** `$FIGAROH_WS/figaroh-examples/examples/<robot>/`.

## What this task does

Builds the dynamic regressor from measured trajectories, reduces it to an identifiable
**base parameter** set by QR decomposition, and solves for those parameters from
measured joint torques. Optionally refines with weighted least squares, reconstructs
standard parameters, and projects onto the physically-consistent set.

## Step 1 — Pick the path

| Situation | Do this |
|---|---|
| Robot has `identification.py` | Go to Step 2. Today: `ur10`, `tiago`, `staubli_tx40`. |
| Robot exists, calibration only | Add `tasks.identification` to its config and an `identification.py` + a `BaseIdentification` subclass modelled on `examples/ur10/`. |
| No folder for this robot | Run **`figaroh-setup-new-robot`** first. |

Staubli TX40 is the identification-focused reference (it is the only robot whose
config turns `wls` on by default); UR10 is the most complete script.

## Step 2 — Understand the data contract (this is the step people get wrong)

Unlike calibration, **there is no single fixed CSV schema.** `load_trajectory_data()`
is an abstract method on `BaseIdentification`, implemented per robot in
`examples/<robot>/utils/<robot>_tools.py`. It must return positions, velocities,
accelerations, and torques. Two conventions exist in this codebase:

**a) Example-script convention (what `examples/` actually uses).** Two CSVs in
`data/`, robot-specific filenames hardcoded in the subclass. UR10:

```
data/identification_q_simulation.csv     header: q0,q1,q2,q3,q4,q5        (rad)
data/identification_tau_simulation.csv   header: tau1,tau2,…,tau6         (Nm)
```

Rows are time-ordered samples at the config's `sampling_frequency`. Velocities and
accelerations are **derived by numerical differentiation and filtering**, not read from
disk (`calculate_first_second_order_differentiation`). Column *names* here are
positional-only — the loader takes the whole frame — but column *count* must equal the
active joint count, and row counts must match between the two files.

**b) Integration-API convention** (`figaroh.integration.api.RobotIdentificationSystem`,
for a one-line API rather than an example script): four files in `data_dir` —
`q.csv`, `dq.csv`, `ddq.csv`, `tau.csv`.

**When wiring your own robot, follow (a)** and mirror UR10's `load_trajectory_data`.
`data_source` is an optional directory override — the same filenames read from a
different directory, which is how `validation_data_file` supplies a held-out set.

## Step 3 — Fill in `tasks.identification`

```yaml
tasks:
  identification:
    enabled: true

    problem:
      include_external_forces: false   # true only with a real F/T sensor
      use_joint_torques: true
      wls: false                       # WLS refinement of the OLS estimate

      model_components:
        friction: false                # viscous + Coulomb per joint
        joint_offset: false
        actuator_inertia: false
        static_regressor: true
        inertia_regressor: true

    signal_processing:
      sampling_frequency: 500.0        # Hz — must match how the data was logged
      cutoff_frequency: 50.0           # Hz — low-pass before differentiation
      filter_type: "butterworth"
      filter_order: 4

    data:
      validation_data_file: ""         # DIRECTORY holding the same filenames,
                                       # genuinely different data. Empty = disabled.
```

Notes that change results:

- **`sampling_frequency` must be the true log rate.** It scales the derived velocities
  and accelerations; wrong here means wrong inertias, with no error raised.
- `cutoff_frequency` too high → differentiation amplifies noise; too low → real
  dynamics smoothed away. Start at ~1/10 of sampling and inspect residuals.
- Turn on `model_components` one at a time. Each adds columns to the regressor and can
  worsen conditioning; enabling everything at once makes a bad condition number
  impossible to attribute.
- `wls` refines OLS with iteratively-weighted least squares (Gautier, 1997), which
  helps when joints have very different torque scales. Staubli TX40 defaults it on;
  UR10/TIAGo default off. Override per run with `--wls` / `--no-wls`.
- **`validation_data_file` points at a directory**, not a file, and must be an
  independent acquisition — never a split of the training log. Empty falls back to the
  training data with a warning; that verdict is not independent validation.

## Step 4 — Run

```bash
conda activate figaroh-dev
cd $FIGAROH_WS/figaroh-examples/examples/<robot>

python identification.py                          # verify + HTML report + archive (all default on)
python identification.py --verify --html-report    # explicit, what validate.py runs
python identification.py --wls                     # force WLS refinement for this run
python identification.py --no-verify -v            # exploratory, INFO logging
```

| Flag | Effect |
|---|---|
| `--config`, `--urdf` | Override defaults without editing files |
| `--verify` / `--no-verify` | Check condition number + validation correlation/improvement against thresholds, write a JSON verdict, **`exit(1)` on failure**. On by default — this is the CI gate. |
| `--html-report` / `--no-html-report` | Self-contained HTML diagnostic (on by default) |
| `--wls` / `--no-wls` | Override `identification.problem.wls` |
| `--archive` / `--no-archive` | Timestamped run archive (on by default) |
| `--asset-id`, `--operator` | Provenance for the specific physical unit |
| `--verbose` / `-v` | INFO logging (default WARNING) |

Because `--verify` exits nonzero on a failed threshold, `python identification.py` is
directly usable as a CI gate — and a nonzero exit is a *verdict*, not a crash.

## Step 5 — Read the outputs

```
results/runs/<asset>/identification/<timestamp>/
    report.html            # diagnostics: torque fit, per-parameter stats
    verdict.json           # machine-readable pass/fail (from --verify)
    provenance + config snapshot + parameters
results/runs/index.jsonl   # one summary line per run
```

What to look at, in order:

1. **Base-parameter condition number.** The headline health metric. A large value means
   the trajectory did not excite the parameters — fix the *data*, not the solver. That
   is what `figaroh-setup-optimal` (exciting trajectories) is for.
2. **Torque correlation / RMS error** between measured and reconstructed torques.
3. **Validation correlation and improvement**, if you supplied independent data.
4. **Per-parameter standard deviations** — a parameter with huge relative σ is not
   identifiable from this dataset; treat its value as meaningless rather than reporting it.

## Gotchas specific to identification

- **Only base parameters are identifiable.** Standard parameters are reconstructed
  under extra assumptions; do not present reconstructed standard inertias with the same
  confidence as base parameters.
- **Inertial parameter ordering differs between Pinocchio and the solver's "standard"
  format.** Pinocchio uses `[m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`. Use
  `reorder_inertial_parameters` in `figaroh/identification/parameter.py` — never
  hand-roll the permutation.
- **Physical consistency (SDP projection) is default-off** and needs `picos` + an SDP
  solver: `identification.physical_consistency.enabled: true`.
- A bad condition number is almost always insufficiently exciting data, not a solver
  bug. `figaroh.tools.solver.LinearSolver` offers ridge/tikhonov/robust variants, but
  regularising a badly-excited problem hides the problem rather than fixing it.

## Step 6 — Verify

```bash
python identification.py --verify --html-report      # exit 0?
cd ../.. && python validate.py --robot <robot>       # nothing regressed
```

## Failure → cause → fix

| Symptom | Cause | Fix |
|---|---|---|
| Shape mismatch between regressor and torque vector | q and tau logs have different row counts, or column count ≠ active joint count | Align the two CSVs; check the active-joint list in the config |
| Condition number enormous | Trajectory does not excite the parameters | Generate an exciting trajectory — `figaroh-setup-optimal` |
| Identified masses/inertias physically absurd (negative mass) | Under-excited data, or physical consistency off | Fix excitation first; then enable `physical_consistency` (needs `picos`) |
| Torque fit good, validation poor | Overfitting an under-excited dataset | More/better data; check validation data is genuinely independent |
| Inertias systematically off by a constant factor | `sampling_frequency` does not match the real log rate | Correct it — nothing errors when it is wrong |
| Noisy accelerations, unstable fit | `cutoff_frequency` too high | Lower it; inspect filtered signals before fitting |
| `ModuleNotFoundError: picos` | Physical consistency enabled without deps | `pip install picos cvxopt`, or disable it |
| `exit(1)` with a verdict file | `--verify` threshold failure — working as designed | Read `verdict.json`; fix data or thresholds, not the exit code |

Anything else: Re-run with `--verbose`/`-v` for INFO logging and work from the traceback; each repo's `AGENTS.md` lists the known gotchas.

## Related

`figaroh-setup-optimal` produces the exciting trajectories that make identification
well-conditioned — run it *before* collecting data. `figaroh-setup-calibration` is the
geometric counterpart. Solver and regressor work belongs in
`figaroh/src/figaroh/tools/` and `figaroh/src/figaroh/identification/`.
