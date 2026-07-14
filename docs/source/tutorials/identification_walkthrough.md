# Identification Walkthrough

## The problem

Dynamic models generated from CAD have 20-50% error in mass, inertia, and
friction parameters — CAD doesn't know about cables, connectors, paint, or
assembly tolerances. Accurate dynamic parameters matter for feedforward
torque control, energy-efficient trajectory planning, collision detection,
and any digital-twin/simulation use of the model.

Dynamic parameter identification recovers the true parameters from logged
motion + torque data.

## The model

The robot's equation of motion:

```
τ = M(q) q̈ + C(q, q̇) q̇ + G(q) + F(q̇)
```

is linear in the *dynamic parameters* `φ` (masses, first mass moments,
inertia tensor entries, friction coefficients), so it can be rewritten as:

```
τ = W(q, q̇, q̈) · φ
```

`W` is the **regressor matrix** — computed purely from motion (`q`, `q̇`,
`q̈`), independent of the unknown `φ`. Given enough logged `(q, q̇, q̈, τ)`
samples from a sufficiently "exciting" trajectory, `φ` is recovered by
linear least squares.

## The pipeline

1. **Base parameter analysis** — QR decomposition of `W` finds the minimal
   set of *identifiable* linear combinations of `φ` (many individual
   parameters aren't observable in isolation; only combinations of them
   are).
2. **Signal processing** — filter noisy position/torque logs (median filter
   for outliers, Butterworth for noise) and estimate velocity/acceleration
   if not measured directly.
3. **Regressor construction** — build `W` from the processed motion data.
4. **Parameter estimation** — solve the linear least-squares problem for
   the base parameters (FIGAROH's [solver](../api/tools.md) supports OLS,
   WLS, ridge, and several constrained/robust variants).
5. **Validation** — compare predicted vs. measured torques on held-out data.

## Running it

```python
from examples.tiago.utils.tiago_tools import TiagoIdentification
from figaroh.tools.robot import load_robot

robot = load_robot("path/to/robot.urdf", load_by_urdf=True)
identifier = TiagoIdentification(robot, "config/tiago_unified_config.yaml")
identifier.initialize()
result = identifier.solve(decimate=True, html_report=True)

verdict = identifier.verify()
print("PASS" if verdict.passed else "FAIL")
```

`decimate=True` downsamples the regressor to reduce redundant, highly
correlated rows before the solve — cheaper and often better-conditioned
than fitting on every raw sample.

From the command line:

```bash
cd examples/tiago
python identification.py                          # html-report + verify on by default
python identification.py --no-html-report --no-verify
```

For a one-line version without a robot-specific subclass, see
[Integration API](../concepts/integration.md).

## What you configure

The `identification:` section of your
[unified config](../concepts/configuration.md) sets `has_friction`,
`has_actuator_inertia`, `has_external_forces`, signal-processing
(`sampling_frequency`, `cutoff_frequency`), and joint/velocity/torque
limits used to sanity-check the logged data.

## Expected results

- Base parameters identified with under ~5% uncertainty
- Torque prediction accuracy above ~95% (`validation_correlation` above the
  default 0.9 threshold) on held-out trajectories
- A condition number well under the default 1000.0 threshold — see
  [verification thresholds](../reporting_and_verification.md)

The quality of all three depends heavily on the trajectory used to collect
data — see [Optimal Experiment Design](optimal_design.md).

## Next steps

- [Reporting & Verification](../reporting_and_verification.md) — the
  full report/verdict/compare-page suite, and how to wire `--verify` into CI.
- [Examples Gallery](../examples/index.md) — complete identification scripts
  for UR10, TIAGo, and Staubli TX40.
