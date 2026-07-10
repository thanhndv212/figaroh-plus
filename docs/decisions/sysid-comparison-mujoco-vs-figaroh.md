# System Identification: MuJoCo `sysid` vs. FIGAROH

## Date
2026-07-10

## Status
Reference (research note, not a decision)

## Summary

MuJoCo's `sysid` module and FIGAROH solve the same problem — turning logged
motion into trustworthy model parameters — from opposite ends of the method
space: one treats the simulator as a black box and fits a trajectory
rollout, the other treats the robot as a linear-in-parameters equation and
solves it in closed form.

Sources:
`python/mujoco/sysid/{README.md, _src/*.py, report/*}` in the MuJoCo repo
(`robot-irl/mujoco`), and
`figaroh/src/figaroh/{identification,calibration,tools,optimal}` plus
`figaroh-examples/examples/*/utils/*_tools.py` in this workspace.

## 1. At a glance

| Dimension | MuJoCo sysid | FIGAROH |
|---|---|---|
| Paradigm | Black-box, simulate-and-compare | Analytic, regressor-based least squares |
| Objective built from | Full forward-dynamics rollout vs. measured sensors | Linear regressor `W(q,q̇,q̈)·φ = τ` from RNEA |
| Solver | Nonlinear least squares (Gauss-Newton / trust-region) | Closed-form / IRLS linear least squares |
| Gradients | Finite differences, batched across threads | Not needed — regressor is exact and linear |
| Physical consistency | By construction (log-Cholesky pseudo-inertia) | Convex SDP/LMI projection onto PSD cone (optional) |
| Excitation design | Not addressed — user supplies trajectories | IPOPT trajectory optimization minimizing regressor condition number |
| Kinematic calibration | Out of scope | Separate nonlinear LM pipeline on SE3 pose error |
| What it can identify | Anything settable on an `MjSpec` (inertia, friction, gains, sensor delay/bias…) | Standard/base inertial parameters, joint friction, actuator inertia, geometric calibration |

## 2. Pipeline shape

The regressor approach is a straight line from data to answer; the
simulation approach is a loop that re-runs the physics engine every
iteration.

**MuJoCo sysid:**
`Log data (control + sensor TimeSeries)` → `Guess φ (modifier rebuilds MjModel)` → `Roll out (mujoco.rollout, full sim)` → `Compare (resample + weighted residual)` → `Step φ (FD Jacobian, GN/LM update)` → loop back to rollout.

**FIGAROH:**
`Log data (q, τ, filter + differentiate)` → `Build W (RNEA regressor, per sample)` → `Reduce (QR → base parameter set)` → `Solve ((weighted) least squares, closed form)` → `Project (SDP onto physical-consistency cone)`.

## 3. Where they diverge

### 3.1 What can actually be identified

**MuJoCo — anything the spec exposes.** A `Parameter` is generic: a value
plus a `modifier(MjSpec, Parameter)` callback, so anything settable on the
MJCF spec is fair game — contact friction, joint damping, `armature`, PD
gains, even sensor delay/gain/bias applied post-rollout. Rigid-body inertia
gets special treatment via `InertiaType.Pseudo`: a 10-D pseudo-inertia is
parameterized through the Rucker & Wensing log-Cholesky encoding, so every
candidate decodes back to a valid `m, h, I` automatically.

**FIGAROH — the classical 10+ per link.** Standard inertial parameters per
link (`m, mx, my, mz, Ixx, Ixy, Ixz, Iyy, Iyz, Izz`), reordered to Pinocchio
convention, plus optional per-joint viscous/Coulomb friction, rotor inertia,
and torque offset columns. Because the full standard set is usually
structurally unidentifiable, a QR-based base parameter extraction
(Gautier/Khalil style) reduces it to a minimal, full-rank basis before
solving — identical in spirit to MuJoCo's identifiability handling in
`optimize.py`, but performed analytically up front rather than diagnosed
after the fact via `cond(JᵀJ)`.

### 3.2 How the objective function is built

**MuJoCo** (`_src/residual.py`, `_src/trajectory.py`): `model_residual`
rebuilds a full `MjModel` for every candidate parameter vector and calls
`sysid_rollout`, which wraps MuJoCo's batched forward-dynamics `rollout` —
a genuine multi-step simulation from a shared initial state. Predicted
sensor traces are resampled to measurement timestamps, differenced
(measured − predicted), and normalized per-sensor so mixed units contribute
comparably.

**FIGAROH** (`tools/regressor.py`): `RegressorBuilder` calls Pinocchio's
`computeJointTorqueRegressor` (RNEA-derived) at each sample to build
`W(q,q̇,q̈)`, giving an exact linear relationship `τ = W·φ` — no
integration, no simulation, no iteration needed to evaluate the model at a
candidate. The residual is simply `τ_measured − W·φ`, which is why the
whole dynamic-identification problem collapses to one linear solve.

### 3.3 Optimization mechanics

**MuJoCo — nonlinear, derivative-free.** `optimize()` dispatches to
`mujoco.minimize.least_squares` (trust-region Gauss-Newton/LM), or SciPy's
trust-region-reflective solver, box-constrained by parameter bounds. No
autodiff, no MJX/JAX anywhere in the module — gradients come from
finite-difference Jacobians, one rollout per perturbed parameter,
parallelized across threads rather than through the physics itself.

**FIGAROH — linear, closed-form.** Ordinary or iteratively-reweighted least
squares (Gautier 1997) on the base regressor — solved via QR/pseudo-inverse,
not iterated against a simulator. A pluggable `LinearSolver` also supports
ridge/lasso/constrained variants. Kinematic (geometric) calibration is the
one place FIGAROH goes nonlinear: SE3 pose residuals solved with
Levenberg-Marquardt plus an iterative outlier-removal loop —
methodologically closer to what MuJoCo sysid does everywhere.

### 3.4 Physical consistency & uncertainty

**MuJoCo — consistency by parameterization.** The log-Cholesky pseudo-inertia
encoding makes `J = UUᵀ ⪰ 0` true by construction — no separate feasibility
constraint is needed for mass/inertia validity, only box bounds around the
unconstrained encoding. Uncertainty comes from the classical linearized
covariance `Σ = s²(JᵀJ)⁻¹` at the optimum, with an eigen-based pseudo-inverse
that suppresses unobservable directions instead of blowing up.

**FIGAROH — consistency by projection.** An optional convex SDP/LMI step
(`identification/physical_consistency.py`, via `picos`) projects each link's
pseudo-inertia matrix onto the PSD cone — a proper Traversaro/Wensing-style
physical-consistency guarantee, applied as post-processing after the linear
solve, optionally bounded by CAD priors. Uncertainty via `relative_stdev`
(Pressé & Gautier 1991): %-relative parameter std-dev from residual
covariance `σ²(WᵀW)⁻¹` — the regressor-based sibling of MuJoCo's
rollout-Jacobian covariance.

## 4. Full comparison matrix

| Dimension | MuJoCo sysid | FIGAROH |
|---|---|---|
| Model source | Compiled `MjModel` from `MjSpec`, rebuilt every iteration | Pinocchio rigid-body model, static across the solve |
| Dynamics engine | MuJoCo C engine (forward dynamics + integration) | Pinocchio RNEA (analytic inverse dynamics) |
| Data needed | Control + sensor time series, any sensors defined in the model | Joint position (+ torque); velocity/acceleration numerically differentiated if absent |
| Signal processing | Resampling, delay/gain/bias as optimizable `SignalTransform`s | Butterworth low-pass + median filtering, central-difference differentiation, decimation |
| Excitation trajectories | Not generated by the tool | IPOPT + cubic B-splines, minimizing base-regressor condition number under joint/torque limits |
| Friction modeling | Whatever the MJCF exposes (joint/contact friction as spec parameters) | Explicit viscous + Coulomb columns appended to the regressor |
| Kinematic calibration | Not a focus | Dedicated module: DH-like geometric errors via LM on SE3 log-map pose error |
| Reporting | Interactive HTML: covariance heatmap, optimization trace, bound-hit insights, rollout video | Base-parameter table (symbolic combinations), RMSE, condition number, relative std-dev%, consistency report |
| Differentiation method | Finite differences over rollouts (no autodiff / no MJX) | Not applicable — regressor is symbolic/analytic, exact |
| Cost per iteration | High — one (or many, batched) full trajectory simulation | Low — one linear solve, or none after the regressor is built |

## 5. Which one fits the job

Reach for **FIGAROH** when the model is linear-in-parameters and the plant
is a serial manipulator. If the goal is classical inertial/friction
identification on an articulated robot arm with joint torque sensing,
FIGAROH's regressor approach is faster, cheaper, and gives closed-form
uncertainty and a physically-consistent result with an SDP guarantee — plus
it can design the excitation trajectory that makes the identification
well-conditioned in the first place, which MuJoCo sysid leaves entirely to
the user.

Reach for **MuJoCo sysid** when the thing you're fitting doesn't have a
clean analytic regressor: contact/friction parameters in a manipulation
task, actuator gains, sensor calibration, or any quantity that only shows
up through the simulator's forward dynamics and contact solver. Its cost is
a full rollout per evaluation; its payoff is that it can identify literally
anything expressible in an MJCF, with no need to derive a regressor by
hand.

The two are complementary rather than competing: a regressor-based
base-parameter identification (FIGAROH) is a natural warm start for a
black-box rollout refinement (MuJoCo sysid) when contact or actuator
nonlinearities need to be folded in afterward.
