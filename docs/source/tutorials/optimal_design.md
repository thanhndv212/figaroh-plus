# Optimal Experiment Design

Both [calibration](calibration_walkthrough.md) and
[identification](identification_walkthrough.md) need measurement data — but
*which* poses to measure and *which* trajectory to execute has a huge
effect on how much data you need and how good the resulting parameters are.
Ad-hoc configurations (grid patterns, random poses, engineering intuition)
typically need 100-500 measurements and still leave some parameters poorly
observed. FIGAROH replaces guesswork with two optimization-based design
steps.

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
for a large pool (1,000-10,000) of feasible candidate poses. In practice
this cuts required measurements from 100+ down to 15-30, with a
mathematical observability guarantee instead of a hope.

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

Solved with IPOPT (interior-point). A well-optimized trajectory typically
improves regressor conditioning by 10-100x over a hand-designed one, which
directly tightens the uncertainty on the identified parameters.

```bash
cd examples/tiago
python optimal_trajectory.py
# → an exciting joint-space trajectory, ready to execute and log
```

## Why this matters

Both problems reduce to the same idea: the [regressor](identification_walkthrough.md)
`W` (kinematic or dynamic) determines how much information a measurement
set carries about the parameters you want. A poorly chosen measurement set
can be numerically singular even with hundreds of samples; a well-designed
one is well-conditioned with a fraction of the data. FIGAROH's condition
number check in [`verify()`](../guides/reporting_and_verification.md)
(default threshold: 1000.0) is the same metric these two steps are
optimizing — it exists precisely to catch a bad experiment design *before*
you trust the identified parameters.

## Next steps

- Run the [Calibration](calibration_walkthrough.md) or
  [Identification](identification_walkthrough.md) walkthrough using the
  configurations/trajectory generated here.
- [Examples Gallery](../examples/index.md) — `optimal_config.py` /
  `optimal_trajectory.py` are shipped for UR10 and TIAGo.
