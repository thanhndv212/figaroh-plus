# Use FIGAROH with an AI agent

FIGAROH ships agent skills: step-by-step instructions a coding agent follows to check your environment, pick a config template, run a task and read the verdict back to you.

## 1. Set up your agent

Pick the agent you use. Each tab is a complete setup.

Clone both repositories side by side, copy the skills into your workspace, and tell the skills where the workspace is:

```
git clone https://github.com/thanhndv212/figaroh-plus figaroh
git clone https://github.com/thanhndv212/figaroh-examples
mkdir -p .claude/skills && cp -r figaroh/skills/* .claude/skills/
export FIGAROH_WS="$PWD"
```

Claude Code discovers the skills automatically. Start a session in the workspace and describe your task.

Clone both repositories as above, then add one line to your workspace's `AGENTS.md` (or the instructions file your agent reads):

```
For FIGAROH tasks, read figaroh/skills/figaroh-start/SKILL.md first.
```

`figaroh-start` routes the agent to the right task skill. Set `FIGAROH_WS` to the workspace root, as for Claude Code.

Without a coding agent, give the assistant the whole documentation as one file:

```
https://thanhndv212.github.io/figaroh-plus/llms-full.txt
```

Paste the link, or download the file and attach it. Every docs page also has a **Copy as Markdown** button at the top right.

## 2. Check the environment

Ask the agent to run the read-only doctor, or run it yourself. It names the exact fix for each failure (missing cyipopt, an import from the wrong checkout, meshes that do not load).

```
bash figaroh/skills/figaroh-setup-env/scripts/doctor.sh
```

**Done when** the doctor reports no failures and `python -c "import figaroh"` works in the environment the agent uses.

## 3. Ask for a task

Good prompts name the robot, the data, and what "done" means. Copy one and edit the paths.

All Setup Calibration Identification Design New robot

figaroh-setup-envCopy

Check that my FIGAROH environment works, fix anything the doctor reports, then run the UR10 calibration example as a smoke test.

Ends with a run that proves the fix.

figaroh-setup-calibrationCopy

Calibrate my UR10 with the mocap poses in data/poses.csv, validating on data/poses_validation.csv. Use a 1 mm position RMSE limit and tell me whether the verdict passes and which parameters changed most.

Names the robot, the data, the held-out set and the limit.

figaroh-setup-calibrationCopy

I know my arm's tolerances: about 0.5 mm on link lengths and 0.1° on joint zeros. Recalibrate with the map method using those priors and compare held-out error with the structural method.

Gives the priors the method needs and asks for a comparison.

figaroh-setup-identificationCopy

Identify the dynamics of TIAGo's arm from q and τ in data/arm_log.csv. Validate on the separate recording data/arm_log_validation.csv and report the held-out torque RMSE per joint, with and without friction.

Asks for held-out validation, not just a fit.

figaroh-setup-optimalCopy

My identification is badly conditioned. Generate an exciting trajectory for the UR10 within its joint limits and compare its condition number with my current motion.

States the problem the design should fix.

figaroh-setup-new-robotCopy

Add my robot to figaroh-examples. The URDF is in ~/robots/my_arm/urdf/my_arm.urdf and the joints I move are joint_1 to joint_6. Start with calibration.

Gives the URDF path, the joints and the first task.

## Tips

- **Say what done means.** "Passes prediction verification with a 2 N·m limit per joint" is checkable; "looks good" is not.
- **Ask for held-out validation.** A fit on training data always looks better than it is. Give the agent a separate recording.
- **Read the verdict yourself.** The agent writes `results/*_verification.json`. Check `passed` and any `not_evaluated` entries before you trust a summary.
- **Keep runs headless.** Plot windows block agents. Set `MPLBACKEND=Agg`, or pass `--no-plot` where a script supports it.

## llms.txt

The site publishes machine-readable copies for assistants and tools:

| File                                                                        | Contains                                                        |
| --------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [`llms.txt`](https://thanhndv212.github.io/figaroh-plus/llms.txt)           | An index of the documentation with one link per page            |
| [`llms-full.txt`](https://thanhndv212.github.io/figaroh-plus/llms-full.txt) | Every page in one Markdown file                                 |
| `<page>/index.md`                                                           | The Markdown of a single page (the **Copy as Markdown** button) |

Next: [what each skill does](https://thanhndv212.github.io/figaroh-plus/agents/skills/index.md).
