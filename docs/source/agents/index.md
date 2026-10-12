---
hide: [toc]
---

# Use FIGAROH with an AI agent

<p class="fg-lead">FIGAROH ships agent skills: step-by-step instructions a coding agent follows to check your environment, pick a config template, run a task and read the verdict back to you.</p>

## 1. Set up your agent

Pick the agent you use. Each tab is a complete setup.

=== "Claude Code"

    Clone both repositories side by side, copy the skills into your workspace, and tell the skills where the workspace is:

    ```bash
    git clone https://github.com/thanhndv212/figaroh-plus figaroh
    git clone https://github.com/thanhndv212/figaroh-examples
    mkdir -p .claude/skills && cp -r figaroh/skills/* .claude/skills/
    export FIGAROH_WS="$PWD"
    ```

    Claude Code discovers the skills automatically. Start a session in the workspace and describe your task.

=== "Other coding agents"

    Clone both repositories as above, then add one line to your workspace's `AGENTS.md` (or the instructions file your agent reads):

    ```markdown
    For FIGAROH tasks, read figaroh/skills/figaroh-start/SKILL.md first.
    ```

    `figaroh-start` routes the agent to the right task skill. Set `FIGAROH_WS` to the workspace root, as for Claude Code.

=== "Chat only"

    Without a coding agent, give the assistant the whole documentation as one file:

    ```text
    https://thanhndv212.github.io/figaroh-plus/llms-full.txt
    ```

    Paste the link, or download the file and attach it. Every docs page also has a **Copy as Markdown** button at the top right.

## 2. Check the environment

Ask the agent to run the read-only doctor, or run it yourself. It names the exact fix for each failure (missing cyipopt, an import from the wrong checkout, meshes that do not load).

```bash
bash figaroh/skills/figaroh-setup-env/scripts/doctor.sh
```

<div class="fg-done" markdown>
**Done when** the doctor reports no failures and `python -c "import figaroh"` works in the environment the agent uses.
</div>

## 3. Ask for a task

Good prompts name the robot, the data, and what "done" means. Copy one and edit the paths.

<div class="fg-filters" data-target="prompts" role="group" aria-label="Filter prompts">
  <button type="button" data-filter="all" aria-pressed="true">All</button>
  <button type="button" data-filter="setup" aria-pressed="false">Setup</button>
  <button type="button" data-filter="calibration" aria-pressed="false">Calibration</button>
  <button type="button" data-filter="identification" aria-pressed="false">Identification</button>
  <button type="button" data-filter="design" aria-pressed="false">Design</button>
  <button type="button" data-filter="robot" aria-pressed="false">New robot</button>
</div>

<div class="fg-prompts" id="prompts">
  <div class="fg-prompt" data-cat="setup">
    <div class="fg-prompt__head"><span class="fg-prompt__skill">figaroh-setup-env</span><button type="button" class="fg-copy">Copy</button></div>
    <p class="fg-prompt__text">Check that my FIGAROH environment works, fix anything the doctor reports, then run the UR10 calibration example as a smoke test.</p>
    <p class="fg-prompt__why">Ends with a run that proves the fix.</p>
  </div>
  <div class="fg-prompt" data-cat="calibration">
    <div class="fg-prompt__head"><span class="fg-prompt__skill">figaroh-setup-calibration</span><button type="button" class="fg-copy">Copy</button></div>
    <p class="fg-prompt__text">Calibrate my UR10 with the mocap poses in data/poses.csv, validating on data/poses_validation.csv. Use a 1 mm position RMSE limit and tell me whether the verdict passes and which parameters changed most.</p>
    <p class="fg-prompt__why">Names the robot, the data, the held-out set and the limit.</p>
  </div>
  <div class="fg-prompt" data-cat="calibration">
    <div class="fg-prompt__head"><span class="fg-prompt__skill">figaroh-setup-calibration</span><button type="button" class="fg-copy">Copy</button></div>
    <p class="fg-prompt__text">I know my arm's tolerances: about 0.5 mm on link lengths and 0.1° on joint zeros. Recalibrate with the map method using those priors and compare held-out error with the structural method.</p>
    <p class="fg-prompt__why">Gives the priors the method needs and asks for a comparison.</p>
  </div>
  <div class="fg-prompt" data-cat="identification">
    <div class="fg-prompt__head"><span class="fg-prompt__skill">figaroh-setup-identification</span><button type="button" class="fg-copy">Copy</button></div>
    <p class="fg-prompt__text">Identify the dynamics of TIAGo's arm from q and τ in data/arm_log.csv. Validate on the separate recording data/arm_log_validation.csv and report the held-out torque RMSE per joint, with and without friction.</p>
    <p class="fg-prompt__why">Asks for held-out validation, not just a fit.</p>
  </div>
  <div class="fg-prompt" data-cat="design">
    <div class="fg-prompt__head"><span class="fg-prompt__skill">figaroh-setup-optimal</span><button type="button" class="fg-copy">Copy</button></div>
    <p class="fg-prompt__text">My identification is badly conditioned. Generate an exciting trajectory for the UR10 within its joint limits and compare its condition number with my current motion.</p>
    <p class="fg-prompt__why">States the problem the design should fix.</p>
  </div>
  <div class="fg-prompt" data-cat="robot">
    <div class="fg-prompt__head"><span class="fg-prompt__skill">figaroh-setup-new-robot</span><button type="button" class="fg-copy">Copy</button></div>
    <p class="fg-prompt__text">Add my robot to figaroh-examples. The URDF is in ~/robots/my_arm/urdf/my_arm.urdf and the joints I move are joint_1 to joint_6. Start with calibration.</p>
    <p class="fg-prompt__why">Gives the URDF path, the joints and the first task.</p>
  </div>
</div>

## Tips

- **Say what done means.** "Passes prediction verification with a 2 N·m limit per joint" is checkable; "looks good" is not.
- **Ask for held-out validation.** A fit on training data always looks better than it is. Give the agent a separate recording.
- **Read the verdict yourself.** The agent writes `results/*_verification.json`. Check `passed` and any `not_evaluated` entries before you trust a summary.
- **Keep runs headless.** Plot windows block agents. Set `MPLBACKEND=Agg`, or pass `--no-plot` where a script supports it.

## llms.txt

The site publishes machine-readable copies for assistants and tools:

| File | Contains |
|---|---|
| [`llms.txt`](https://thanhndv212.github.io/figaroh-plus/llms.txt) | An index of the documentation with one link per page |
| [`llms-full.txt`](https://thanhndv212.github.io/figaroh-plus/llms-full.txt) | Every page in one Markdown file |
| `<page>/index.md` | The Markdown of a single page (the **Copy as Markdown** button) |

Next: [what each skill does](skills.md).
