# Workflows

Calibration and identification follow the same five steps. Each guide explains the choices you make at every step and how you know it worked.

| Step             | You decide                                          | You check                                    |
| ---------------- | --------------------------------------------------- | -------------------------------------------- |
| **1. Plan**      | What to estimate and which data you have            | You have a separate validation set           |
| **2. Design**    | Which postures or trajectories to record            | The design is well conditioned               |
| **3. Configure** | Which template to extend and which keys to override | The config loads and matches your data       |
| **4. Fit**       | Which estimator to run                              | Conditioning, residuals and parameter values |
| **5. Verify**    | Which limits your application needs                 | `verify()` passes on held-out data           |

- **Calibrate a robot**

  ______________________________________________________________________

  Joint offsets, link geometry, base and tool frames from measured poses. Covers the five estimation methods and when to use each.

  [Calibration guide](https://thanhndv212.github.io/figaroh-plus/workflows/calibration/index.md)

- **Identify dynamics**

  ______________________________________________________________________

  Base parameters, friction and actuator inertia from motion and torque. Covers filtering, estimators and reconstruction.

  [Identification guide](https://thanhndv212.github.io/figaroh-plus/workflows/identification/index.md)

- **Add your robot**

  ______________________________________________________________________

  Scaffold a robot folder before either guide applies to your hardware.

  [New-robot guide](https://thanhndv212.github.io/figaroh-plus/workflows/new-robot/index.md)

- **Reports & verdicts**

  ______________________________________________________________________

  The HTML report, the JSON verdict and the two-run compare page.

  [Reporting & verification](https://thanhndv212.github.io/figaroh-plus/reporting_and_verification/index.md)

For a single narrative walkthrough on TIAGo, read the [tutorials](https://thanhndv212.github.io/figaroh-plus/tutorials/index.md). For a full plan-to-verdict example in code, read [Plan, fit & verify](https://thanhndv212.github.io/figaroh-plus/example_workflow/index.md).
