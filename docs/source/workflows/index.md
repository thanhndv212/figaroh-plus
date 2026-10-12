# Workflows

<p class="fg-lead">Calibration and identification follow the same five steps. Each guide explains the choices you make at every step and how you know it worked.</p>

| Step | You decide | You check |
|---|---|---|
| **1. Plan** | What to estimate and which data you have | You have a separate validation set |
| **2. Design** | Which postures or trajectories to record | The design is well conditioned |
| **3. Configure** | Which template to extend and which keys to override | The config loads and matches your data |
| **4. Fit** | Which estimator to run | Conditioning, residuals and parameter values |
| **5. Verify** | Which limits your application needs | `verify()` passes on held-out data |

<div class="grid cards" markdown>

-   **Calibrate a robot**

    ---

    Joint offsets, link geometry, base and tool frames from measured poses. Covers the five estimation methods and when to use each.

    [:octicons-arrow-right-24: Calibration guide](calibration.md)

-   **Identify dynamics**

    ---

    Base parameters, friction and actuator inertia from motion and torque. Covers filtering, estimators and reconstruction.

    [:octicons-arrow-right-24: Identification guide](identification.md)

-   **Add your robot**

    ---

    Scaffold a robot folder before either guide applies to your hardware.

    [:octicons-arrow-right-24: New-robot guide](new-robot.md)

-   **Reports & verdicts**

    ---

    The HTML report, the JSON verdict and the two-run compare page.

    [:octicons-arrow-right-24: Reporting & verification](../reporting_and_verification.md)

</div>

For a single narrative walkthrough on TIAGo, read the [tutorials](../tutorials/index.md). For a full plan-to-verdict example in code, read [Plan, fit & verify](../example_workflow.md).
