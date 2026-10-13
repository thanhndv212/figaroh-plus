# Skills

<p class="fg-lead">Each skill is a Markdown file under <code>figaroh/skills/</code> that tells an agent how to do one job, which files to touch and how to know it worked.</p>

`figaroh-start` is the entry point. It maps the two repositories and hands the task to one of the setup skills.

| Skill | Use it when | It ends with |
|---|---|---|
| `figaroh-start` | Any FIGAROH request. Routes to the skills below. | The right skill loaded, with the robot folder and task named |
| `figaroh-setup-env` | A fresh machine, `import figaroh` fails, cyipopt or picos is missing, meshes do not load, or library edits have no effect | `doctor.sh` passes and the UR10 example runs |
| `figaroh-setup-calibration` | Correcting joint offsets and frame transforms from measured poses | A calibration run, an HTML report and a verdict |
| `figaroh-setup-identification` | Fitting masses, inertias, friction and actuator inertia from q/τ logs | An identification run with held-out torque error per joint |
| `figaroh-setup-optimal` | Badly conditioned results, or before collecting data on a new robot | Designed postures or an exciting trajectory, with its condition number |
| `figaroh-setup-new-robot` | A robot with no folder under `figaroh-examples/examples/` | A scaffolded robot folder that `validate.py --robot <robot>` runs |

## How routing works

``` text
you ──► figaroh-start ─┬─► figaroh-setup-env            (always first on a new machine)
                       ├─► figaroh-setup-calibration
                       ├─► figaroh-setup-identification
                       ├─► figaroh-setup-optimal
                       └─► figaroh-setup-new-robot ──► calibration / identification
```

The skills follow the same five steps as the [workflow guides](../workflows/index.md): Plan, Design, Configure, Fit, Verify. If the agent proposes something the guides warn against, such as validating on a split of the training file, point it at the relevant step.

## Read a skill

The skills are plain Markdown. Reading the one your agent loaded shows you what it will do:

- [`skills/figaroh-start/SKILL.md`](https://github.com/thanhndv212/figaroh-plus/tree/main/skills/figaroh-start)
- [`skills/`](https://github.com/thanhndv212/figaroh-plus/tree/main/skills) for the full set
