# Identification Walkthrough

## The problem

Nominal models can omit cables, connectors, payloads, friction and assembly
differences. The resulting parameter and prediction errors depend on the robot
and the operating conditions. Accurate dynamic parameters matter for feedforward
torque control, energy-efficient trajectory planning, collision detection,
and any digital-twin/simulation use of the model.

Dynamic identification estimates observable parameter combinations from
motion and effort data. Individual physical link parameters require additional
constraints and are not necessarily uniquely recoverable.

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

1. **Partition and signal processing** — declare training/validation data,
   check rates and synchronization, then process each partition independently;
   estimate velocity/acceleration if not measured directly.
2. **Regressor construction** — build `W` from checked motion and effort data.
3. **Base parameter analysis** — analyze identifiable combinations and dataset
   excitation; distinguish structural dependencies from weak sample coverage.
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

## Interpreting results

Report per-joint training and held-out effort errors against the nominal model,
rank/conditioning under declared scaling, and uncertainty assumptions. Default
verification thresholds are configurable checks, not guarantees of percentage
accuracy or full-parameter recovery. Record which checks were skipped and whether
the adapter used genuine held-out data.

If reconstructing or projecting full inertias, report solver and physical
verdicts separately and recompute prediction for the selected stage. The private
log-Cholesky benchmark remains experimental pending convergence review. Follow
[Plan, Fit and Validate](../example_workflow.md) for method selection, comparison
policy, parameter interpretation and export checks.

## Next steps

- [Reporting & Verification](../reporting_and_verification.md) — the
  full report/verdict/compare-page suite, and how to wire `--verify` into CI.
- [Examples Gallery](../examples/index.md) — complete identification scripts
  for UR10, TIAGo, and Staubli TX40.
