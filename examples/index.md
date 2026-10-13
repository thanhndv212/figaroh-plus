# Examples Gallery

Complete, runnable reference implementations live in the companion [figaroh-examples](https://github.com/thanhndv212/figaroh-examples) repository — one folder per robot, each with its own config, data, and results. This gallery is the index into that repository: what each example demonstrates, how to run it, and where its outputs land. For the theory behind what these scripts do, read the [Tutorials](https://thanhndv212.github.io/figaroh-plus/tutorials/index.md) first.

| Robot                                                                                      | Type                   | Demonstrates                                                         |                                                                                             |
| ------------------------------------------------------------------------------------------ | ---------------------- | -------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| [UR10](https://thanhndv212.github.io/figaroh-plus/examples/ur10/index.md)                  | Fixed-base manipulator | Calibration, identification, optimal config + trajectory             | [→ folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/ur10)         |
| [TIAGo](https://thanhndv212.github.io/figaroh-plus/examples/tiago/index.md)                | Mobile manipulator     | Calibration, identification, optimal config + trajectory             | [→ folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/tiago)        |
| [TALOS](https://thanhndv212.github.io/figaroh-plus/examples/talos/index.md)                | Humanoid (torso/arm)   | Calibration                                                          | [→ folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/talos)        |
| [Staubli TX40](https://thanhndv212.github.io/figaroh-plus/examples/staubli_tx40/index.md)  | Fixed-base manipulator | Identification                                                       | [→ folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/staubli_tx40) |
| [Config Templates](https://thanhndv212.github.io/figaroh-plus/examples/templates/index.md) | —                      | The `extends:` template-inheritance system every example config uses | [→ folder](https://github.com/thanhndv212/figaroh-examples/tree/main/examples/templates)    |

## Installing figaroh-examples

```
git clone https://github.com/thanhndv212/figaroh-examples
cd figaroh-examples
pip install figaroh
pip install -r requirements.txt
```

Each script assumes you run it from inside its robot folder:

```
cd examples/ur10
python calibration.py
```

## Common layout

Every robot folder follows the same shape:

```
{robot}/
  calibration.py            # kinematic calibration (if present)
  identification.py         # dynamic identification (if present)
  optimal_config.py         # optimal measurement configurations (if present)
  optimal_trajectory.py     # exciting trajectories for identification (if present)
  config/                   # YAML configs (unified format, extends a template)
  data/                     # CSV logs / measurement data
  urdf/                     # robot URDF(s)
  utils/                    # robot-specific BaseCalibration/BaseIdentification subclass
  results/                  # generated HTML reports, verification JSON, plots
```

## Creating a new example

Scaffold a new robot folder from the TIAGo template:

```
cd examples
./create_example.sh <robot_name>
```

The generated scripts are placeholders pointing back at the TIAGo example as the reference implementation — fill in the robot-specific class in `utils/`, then point its config at the matching [template](https://thanhndv212.github.io/figaroh-plus/examples/templates/index.md) (`manipulator_robot.yaml` or `humanoid_robot.yaml`).
