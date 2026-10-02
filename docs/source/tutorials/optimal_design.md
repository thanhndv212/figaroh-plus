# Optimal Experiment Design

Both [calibration](calibration_walkthrough.md) and
[identification](identification_walkthrough.md) need measurement data — but
*which* poses to measure and *which* trajectory to execute has a huge
effect on how much data you need and how good the resulting parameters are.
Measurement count alone does not establish observability. FIGAROH provides
optimization-based configuration and trajectory design tools; their benefit
must be assessed for the selected parameter scope, constraints and sensor model.
Use [Plan, Fit and Validate](../example_workflow.md) to plan acquisition and
independent validation together.

## Optimal configuration generation (for calibration)

**Goal:** pick the smallest set of robot poses that maximizes observability
of the kinematic parameters.

Uses **D-optimal design**: select configurations that maximize the
determinant of the Fisher information matrix built from the kinematic
regressor `R`:

```
maximize:   det(Σ w_i · Rᵢᵀ Rᵢ)^(1/n)
subject to: Σ w_i ≤ 1,  w_i ≥ 0,  kinematic feasibility
```

Solved as a Second-Order Cone Program (SOCP) over candidate weights `w_i`
for a pool of feasible candidate poses. Inspect the selected set's rank and
information spectrum under the chosen
model. Numerical design quality does not guarantee real measurement accuracy
or recovery of parameters outside the observable subspace.

```bash
cd examples/tiago
python optimal_config.py --end-effector hey5
# → results/tiago_optimal_configurations_hey5.yaml
```

## Optimal trajectory generation (for identification)

**Goal:** generate a single smooth motion that "excites" all the dynamic
coupling effects (inertial, Coriolis, gravitational, friction) needed to
observe the dynamic parameters, while respecting every physical limit.

Formulated as a constrained optimization over cubic-spline waypoints:

```
minimize:   condition_number(W_base(trajectory))
subject to: joint position/velocity/acceleration limits
            joint torque/power limits
            self-collision avoidance
            C² trajectory smoothness
```

Solved with IPOPT (interior-point). Record convergence and verify which
constraints the selected implementation enforces. Compare the candidate with a
feasible baseline under the same scaling and assess the acquired data's held-out
prediction; improved conditioning alone is insufficient evidence.

```bash
cd examples/tiago
python optimal_trajectory.py
# → candidate trajectory; check actual robot/acquisition constraints before execution
```

## Why this matters

Both problems reduce to the same idea: the [regressor](identification_walkthrough.md)
or parameter Jacobian determines which directions the sampled model observes,
under the declared measurement assumptions. A poorly chosen measurement set can remain rank-deficient despite many
samples. Inspect conditioning alongside rank, scaling and the chosen design
objective; D-optimal information and condition number are different criteria.
The condition-number check in [`verify()`](../reporting_and_verification.md)
is one diagnostic, not a complete experiment-quality or accuracy certificate.

## Next steps

- Run the [Calibration](calibration_walkthrough.md) or
  [Identification](identification_walkthrough.md) walkthrough using the
  configurations/trajectory generated here.
- [Examples Gallery](../examples/index.md) — `optimal_config.py` /
  `optimal_trajectory.py` are shipped for UR10 and TIAGo.
