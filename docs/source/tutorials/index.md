# Tutorials

These walkthroughs explain the *why* and *how* behind FIGAROH's four core
workflows, using TIAGo (a mobile manipulator) as the running example. Each
one is backed by a real, runnable script in
[figaroh-examples](https://github.com/thanhndv212/figaroh-examples) — see
the [Examples Gallery](../examples/index.md) for the complete per-robot
reference implementations (UR10, TIAGo, TALOS, Staubli TX40) once you've
read the walkthrough for the workflow you need.

| Tutorial | Workflow | Answers |
|---|---|---|
| [Calibration Walkthrough](calibration_walkthrough.md) | Kinematic calibration | Why do robots need calibrating, and how does FIGAROH solve for the correction? |
| [Choosing What to Estimate](calibration_estimation_guide.md) | Kinematic calibration | Which corrections should I estimate, and how, for my robot and postures? |
| [Identification Walkthrough](identification_walkthrough.md) | Dynamic parameter identification | How does FIGAROH turn a torque/motion log into a validated dynamic model? |
| [Optimal Experiment Design](optimal_design.md) | Optimal configurations & trajectories | How does FIGAROH decide *which* poses/motions to measure, instead of guessing? |

## Starting with a new robot or dataset

Begin with [Plan, Fit and Validate](../example_workflow.md) to inventory data,
choose the model/methods, design acquisition, process measurements and establish
independent validation. These task walkthroughs start after those decisions;
the examples repository supplies a reusable experiment brief and integration guide.

## Prerequisites

- FIGAROH installed (see [Getting Started](../getting_started.md))
- A robot URDF and a [unified config](../concepts/configuration.md) for it
  — or clone [figaroh-examples](https://github.com/thanhndv212/figaroh-examples)
  and use one of the shipped robot folders directly
- Basic familiarity with the linear-in-parameters formulation
  `τ = W(q, q̇, q̈) · φ` for dynamics; geometric calibration instead fits
  pose/contact residuals and analyzes their parameter Jacobian

## The four workflows, at a glance

```
Optimal Configuration Generation ──▶ (collect calibration data) ──▶ Kinematic Calibration
Optimal Trajectory Generation    ──▶ (collect identification data) ──▶ Dynamic Identification
```

The two design steps are optional tools for choosing informative, feasible
experiments (see [Optimal Experiment Design](optimal_design.md)). Acquisition
coverage, synchronization and sensor quality still limit the resulting fit.

Use verification and HTML reports where supported by the selected workflow —
see [Reporting & Verification](../reporting_and_verification.md). Inspect skipped
checks and actual held-out data; a passing verdict alone is not proof of a
validated model.
