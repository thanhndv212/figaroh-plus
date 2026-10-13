---
hide: [toc]
---

# Quickstart

<p class="fg-lead">Install FIGAROH, run a real calibration on a bundled robot, and read its report. About five minutes.</p>

## 1. Install

=== "pip"

    ```bash
    pip install figaroh
    ```

    Enough for calibration and identification. Trajectory design needs cyipopt, which installs from conda only.

=== "conda (everything)"

    ```bash
    git clone https://github.com/thanhndv212/figaroh-plus figaroh
    cd figaroh
    conda env create -f environment.yml
    conda activate figaroh-dev
    pip install -e .
    ```

    Includes cyipopt for trajectory design and the tools to build these docs.

## 2. Get the examples

Robots, data and runnable scripts live in a separate repository:

```bash
git clone https://github.com/thanhndv212/figaroh-examples
cd figaroh-examples/examples/ur10
```

## 3. Run a calibration

```bash
python calibration.py --calibrate-only --no-plot
```

FIGAROH loads the UR10 URDF and 29 measured poses, fits joint offsets and link geometry, and prints a quality report: condition number, RMSE before and after, and outliers.

## 4. Read the result

| Output | Where | What to look for |
|---|---|---|
| Terminal report | your shell | RMSE should drop; few or no outliers |
| Saved results | `data/calibration/calibration_results_<timestamp>.npz` | input for `--update-model`, which exports a corrected URDF |
| HTML report | `results/calibration_report.html` (with `--html-report`) | residual plots and parameter table, shareable as one file |

<div class="fg-done" markdown>
**Done when** the terminal shows post-calibration RMSE below the pre-calibration value.
</div>

## Where next

<div class="grid cards" markdown>

-   **Calibrate your robot**

    ---

    Plan, design, configure, fit and verify, with the choices at each step.

    [:octicons-arrow-right-24: Calibration guide](../workflows/calibration.md)

-   **Identify dynamics**

    ---

    Masses, inertias and friction from motion and torque logs.

    [:octicons-arrow-right-24: Identification guide](../workflows/identification.md)

-   **Add your robot**

    ---

    Scaffold a folder with URDF, config and a tools subclass.

    [:octicons-arrow-right-24: New-robot guide](../workflows/new-robot.md)

-   **Use an AI agent**

    ---

    Let a coding agent set up and run the task with FIGAROH's skills.

    [:octicons-arrow-right-24: Agent setup](../agents/index.md)

</div>
