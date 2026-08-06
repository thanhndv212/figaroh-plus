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
| [Identification Walkthrough](identification_walkthrough.md) | Dynamic parameter identification | How does FIGAROH turn a torque/motion log into a validated dynamic model? |
| [Optimal Experiment Design](optimal_design.md) | Optimal configurations & trajectories | How does FIGAROH decide *which* poses/motions to measure, instead of guessing? |

## Prerequisites

- FIGAROH installed (see [Getting Started](../getting_started.md))
- A robot URDF and a [unified config](../concepts/configuration.md) for it
  — or clone [figaroh-examples](https://github.com/thanhndv212/figaroh-examples)
  and use one of the shipped robot folders directly
- Basic familiarity with the linear-in-parameters formulation
  `τ = W(q, q̇, q̈) · φ` used throughout (both calibration and identification
  reduce to a regressor + a least-squares solve over this equation)

## The four workflows, at a glance

```
Optimal Configuration Generation ──▶ (collect calibration data) ──▶ Kinematic Calibration
Optimal Trajectory Generation    ──▶ (collect identification data) ──▶ Dynamic Identification
```

The two "optimal" steps are optional but recommended — they replace ad-hoc
pose/trajectory selection with a mathematically justified minimum-data
design (see [Optimal Experiment Design](optimal_design.md)). Once you have
data, calibration and identification are independent of *how* the data was
collected.

Every workflow ends the same way: call `.verify()` for a pass/fail verdict
and `.export_html_report()` for a shareable diagnostic — see
[Reporting & Verification](../reporting_and_verification.md).
