# FAQ & Troubleshooting

## Installation

### `pip install figaroh` fails to build, or optimization scripts complain about missing solvers

Optimal configuration/trajectory generation uses `cyipopt` (a wrapper around the IPOPT solver), which needs system libraries that aren't pulled in by pip alone:

```
# Debian/Ubuntu
sudo apt-get install -y pkg-config coinor-libipopt-dev libblas-dev liblapack-dev
pip install cyipopt picos cvxpy
```

If you use conda, some robotics dependencies are easier to get from conda-forge:

```
conda install -c conda-forge pinocchio cyipopt
```

### Do I need MuJoCo, Genesis, or Isaac Sim installed?

No — Pinocchio is the only always-available backend and FIGAROH's core algorithms don't require the others. Install `mujoco`, `genesis-world`, or Isaac Sim only if you actually want that [backend](https://thanhndv212.github.io/figaroh-plus/concepts/backends/index.md); `figaroh.backends.list_backends()` reports what's currently available in your environment.

## Configuration

### What happened to the legacy flat format?

It was removed in 1.0; use the [unified format](https://thanhndv212.github.io/figaroh-plus/concepts/configuration/index.md) — it supports `extends:` template inheritance, so a robot config only states what's actually robot-specific. Loading an old flat config raises an error naming the converter, `python -m figaroh.utils.config_migration`. The examples in [figaroh-examples](https://github.com/thanhndv212/figaroh-examples) all use the unified format.

### My robot config isn't picking up template defaults

Check that `extends:` is a path relative to the config file itself, not the working directory — e.g. `extends: "../../templates/manipulator_robot.yaml"` from `examples/my_robot/config/`. Also remember list-valued keys (like `markers`) are *replaced*, not merged, when both the template and the robot config define them — see [override rules](https://thanhndv212.github.io/figaroh-plus/concepts/configuration/index.md).

## Calibration & Identification

### Calibration/identification isn't converging

Check your initial parameter guesses and joint limits in the config first — most non-convergence traces back to a bad initial guess or an over-constrained problem, not the solver itself.

### My calibration fits the training postures well but predicts new postures badly

Usually too many parameters for what the postures excite: estimating weakly excited corrections freely fits noise. Check the number of measurements per estimated parameter, then try an estimation method that accounts for excitation (`excitation`, `map`, `map_cv`, `cv_subset`) and compare them on held-out postures. See [Choosing what to estimate](https://thanhndv212.github.io/figaroh-plus/tutorials/calibration_estimation_guide/index.md).

### The calibrated parameter set differs between my laptop and CI

With the default `structural` method, ties in the structural QR are broken by floating-point details, so the selected parameters can differ between platforms (figaroh-plus#113). The non-structural methods select on the measured postures in a fixed order; `map` and `map_cv` select nothing. See [Calibration estimation methods](https://thanhndv212.github.io/figaroh-plus/concepts/calibration_estimation/index.md).

### Identification results have high uncertainty or fail `verify()`

This is almost always a data-quality problem, not an algorithm problem:

- Was the trajectory actually "exciting"? See [Optimal Experiment Design](https://thanhndv212.github.io/figaroh-plus/tutorials/optimal_design/index.md) — an ill-conditioned trajectory will identify poorly no matter how good the solver is.
- Is the signal-to-noise ratio adequate? Try enabling `decimate=True` or tightening the Butterworth cutoff frequency in `signal_processing`.
- Do you have a genuinely held-out `validation_data_file` configured? A run with no validation data skips (not fails) the validation-based checks in `verify()` — see [thresholds](https://thanhndv212.github.io/figaroh-plus/reporting_and_verification/index.md).

### Why did a metric get *skipped* instead of *failed* in my verification report?

A threshold whose metric can't be computed — typically because no `validation_data_file` is configured — is skipped rather than counted as a failure. An empty check list counts as passed since there's nothing to fail on. See [Reporting & Verification](https://thanhndv212.github.io/figaroh-plus/reporting_and_verification/index.md).

## Examples

### Where are the runnable robot examples?

In the separate [figaroh-examples](https://github.com/thanhndv212/figaroh-examples) repository, not in this one — see the [Examples Gallery](https://thanhndv212.github.io/figaroh-plus/examples/index.md). figaroh (this repo) is the library; figaroh-examples has the data, URDFs, and per-robot scripts.

### How do I add support for a new robot?

Scaffold it from the TIAGo example:

```
cd figaroh-examples/examples
./create_example.sh <robot_name>
```

Then implement a `BaseCalibration`/`BaseIdentification` subclass under `utils/`, and point its config at the matching [template](https://thanhndv212.github.io/figaroh-plus/examples/templates/index.md).
