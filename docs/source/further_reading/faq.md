# FAQ & Troubleshooting

## Installation

### `pip install figaroh` fails to build, or optimization scripts complain about missing solvers

Optimal configuration/trajectory generation uses `cyipopt` (a wrapper
around the IPOPT solver), which needs system libraries that aren't pulled
in by pip alone:

```bash
# Debian/Ubuntu
sudo apt-get install -y pkg-config coinor-libipopt-dev libblas-dev liblapack-dev
pip install cyipopt picos cvxpy
```

If you use conda, some robotics dependencies are easier to get from
conda-forge:

```bash
conda install -c conda-forge pinocchio cyipopt
```

### Do I need MuJoCo, Genesis, or Isaac Sim installed?

No — Pinocchio is the only always-available backend and FIGAROH's core
algorithms don't require the others. Install `mujoco`, `genesis-world`, or
Isaac Sim only if you actually want that
[backend](../concepts/backends.md); `figaroh.backends.list_backends()`
reports what's currently available in your environment.

## Configuration

### Should I use the unified format or the legacy format?

Use the [unified format](../concepts/configuration.md) for anything new —
it supports `extends:` template inheritance, so a robot config only states
what's actually robot-specific. The legacy flat format still works and
existing configs don't need migrating, but new examples in
[figaroh-examples](https://github.com/thanhndv212/figaroh-examples) all use
the unified format.

### My robot config isn't picking up template defaults

Check that `extends:` is a path relative to the config file itself, not the
working directory — e.g. `extends: "../../templates/manipulator_robot.yaml"`
from `examples/my_robot/config/`. Also remember list-valued keys (like
`markers`) are *replaced*, not merged, when both the template and the robot
config define them — see [override rules](../concepts/configuration.md).

## Calibration & Identification

### Calibration/identification isn't converging

Check your initial parameter guesses and joint limits in the config first —
most non-convergence traces back to a bad initial guess or an
over-constrained problem, not the solver itself.

### Identification results have high uncertainty or fail `verify()`

This is almost always a data-quality problem, not an algorithm problem:

- Was the trajectory actually "exciting"? See
  [Optimal Experiment Design](../tutorials/optimal_design.md) — an
  ill-conditioned trajectory will identify poorly no matter how good the
  solver is.
- Is the signal-to-noise ratio adequate? Try enabling `decimate=True` or
  tightening the Butterworth cutoff frequency in `signal_processing`.
- Do you have a genuinely held-out `validation_data_file` configured? A run
  with no validation data skips (not fails) the validation-based checks in
  `verify()` — see [thresholds](../reporting_and_verification.md).

### Why did a metric get *skipped* instead of *failed* in my verification report?

A threshold whose metric can't be computed — typically because no
`validation_data_file` is configured — is skipped rather than counted as a
failure. An empty check list counts as passed since there's nothing to
fail on. See [Reporting & Verification](../reporting_and_verification.md).

## Examples

### Where are the runnable robot examples?

In the separate [figaroh-examples](https://github.com/thanhndv212/figaroh-examples)
repository, not in this one — see the [Examples Gallery](../examples/index.md).
figaroh (this repo) is the library; figaroh-examples has the data, URDFs,
and per-robot scripts.

### How do I add support for a new robot?

Scaffold it from the TIAGo example:

```bash
cd figaroh-examples/examples
./create_example.sh <robot_name>
```

Then implement a `BaseCalibration`/`BaseIdentification` subclass under
`utils/`, and point its config at the matching
[template](../examples/templates.md).
