# Log-Cholesky feasibility: revise before production

- Status: Accepted — **revise**, not go
- Date: 2026-10-02
- Issue: [#22](https://github.com/thanhndv212/figaroh-plus/issues/22)
- Parent: [#20](https://github.com/thanhndv212/figaroh-plus/issues/20)
- Follow-up: [#30](https://github.com/thanhndv212/figaroh-plus/issues/30)
- Scope: private synthetic feasibility experiment; no production API approval

## Decision and measured rationale

Retain linear identification and corrected post-fit SDP as the defaults.
Log-Cholesky torque fitting has a useful accuracy signal near the physical
boundary, but **19 of 20 runs exhaust the frozen 200-evaluation budget**.
Only the noisy repaired-OLS start converges. Nominal starts fail termination
in all four cases, even when their torque errors are small. This repeats on
Pinocchio 3.7.0 and 4.1.0 with identical generated dataset hashes and numerical
results. Do not publish a solver that treats these candidates as success.

Proceed with the private convergence/scaling follow-up, not production
issues #23–#25 yet. This closes the first spike's question with a revise
outcome; it does not complete Feature 3 or authorize raising budgets after
looking at results and relabeling the original experiment as a pass.

## Frozen protocol and independent checks

[Private script](../development/spikes/log_cholesky_feasibility.py) contains
all constants selected before measuring. It generates a deterministic six
joint manipulator with nonzero first moments and off-diagonal inertias,
120 training samples and 160 independent held-out samples. Fourier positions,
velocities and accelerations are analytic; there is no numerical
differentiation, filtering, measured torque or physical robot in this spike.
Noise is Gaussian, 0.03 Nm on training torque only; validation torque is
noise-free RNEA. Seeds are 2201–2206. Four scenarios share the same reference
model: clean, noisy, weak excitation (only joint 1 moves in training), and
near-boundary ground-truth inertias with noisy training.

Rows are explicitly joint-major. Each joint's ten columns use Pinocchio
order `[m,mx,my,mz,Ixx,Ixy,Iyy,Ixz,Iyz,Izz]`. Units are SI. Torque residual
weights are one for all rows. The OLS null-space component is anchored to
nominal parameters by solving for a delta from nominal; its predictions are
ordinary OLS, not a claim of uniquely recovered full inertials. SDP projects
each OLS link in dynamic-parameter coordinates using the same fixed scale
weights: mass 1 kg, first moments 0.2 kg m, inertia entries 0.2 kg m².

Nonlinear residuals concatenate torque error and
`sqrt(1e-6) * (p - nominal) / scale`. The dynamic-parameter prior is explicit,
is held fixed across starts and does not consume validation data. It differs
from unregularized OLS; no claim of identical optimization objectives is made.
TRF uses the Pinocchio analytic Jacobian, `x_scale='jac'`, scalar coordinate
bounds ±12, tolerances `ftol=xtol=gtol=1e-10`, 200 function evaluations and a
20-second guard checked at each residual call. Jacobian calls, evaluations,
objective, status, runtime and initialization repairs are recorded.

Starts: nominal; OLS repaired by independently eigendecomposing pseudo-inertia
and flooring eigenvalues at 1e-6 (changes reported per link); nominal with
three log diagonals reduced by 3; and two seeded perturbations. The boundary
truth reduces one diagonal log by 4. Repair does not invoke SDP. A repaired
OLS start near the boundary explicitly exercises invalid OLS inputs.

The oracle constructs `P=[[0.5 tr(I_O)I-I_O,h],[h.T,m]]` independently of
FIGAROH/Pinocchio conversion. It also checks positive mass and principal
triangle margins of centre-of-mass inertia. Feasibility tolerance is 1e-8;
PSD baseline candidates are allowed within that tolerance. Exact-arithmetic
PD encoding does not imply a numerical margin: some final candidates have
eigenvalues near 1e-14 and need explicit finite-precision checks in production.
Before fitting, the script checks inverse-coordinate round trips,
independent block conversion, analytic Jacobian against central differences,
and torque regressor times full inertials against RNEA on both trajectories.
Maximum Jacobian absolute discrepancy: **2.066e-10**.

## Results

The following nominal-start errors are diagnostics of **unsuccessful** fits,
not accepted solver outputs. Nm, aggregate over all six joints:

| Case | OLS train / held-out | OLS+SDP train / held-out | Log-Cholesky train / held-out |
| --- | --- | --- | --- |
| Clean | <1e-12 / <1e-12 | 8.365e-5 / 9.343e-5 | 2.779e-6 / 2.983e-6 |
| Noisy | 0.029028 / 0.044712 | 0.029028 / 0.044729 | 0.029028 / 0.044712 |
| Weak excitation | 0.029553 / 0.517970 | 0.029553 / 0.517943 | 0.029553 / 0.465411 |
| Boundary truth | 0.029028 / 0.044712 | 0.135214 / 0.150022 | 0.029028 / 0.044716 |

Training regressor rank is **38 of 60**, or **18 of 60** under weak excitation;
condition number on retained singular directions is 456 or 383 respectively
(rank cutoff 1e-10 relative). The full regressor is rank deficient in every
case. Physical consistency cannot resolve its unidentifiable directions.
Weak-excitation nonlinear held-out errors range from approximately 0.430 to
1.349 Nm across starts: accuracy is initialization/prior dependent.

All 20 final nonlinear candidates pass the independent feasibility verdict,
as do all SDP results. Boundary-truth OLS has minimum eigenvalue -0.03158;
its SDP projection repairs feasibility but worsens held-out prediction.
No infeasible candidate is hidden. Per-link masses, minimum eigenvalues,
triangle margins, initial physical verdicts and per-joint torque errors are
in the machine-readable records below.

Local timings with one BLAS thread: OLS <0.002 s; SDP approximately
0.04–0.21 s; nonlinear approximately 0.25–0.46 s. These are reference-machine
observations, not transferable performance guarantees. Runtime passes the
budget but reliable termination does not. SDP introduces a small clean-case
prediction change through numerical solve tolerance, even when OLS is feasible.
Fit minimizes torque residuals plus prior; SDP minimizes weighted inertial
parameter distance. They are not mathematically interchangeable.

## Proposed contract and production gates

These design choices guide the follow-up; acceptance of this record does not
approve their production implementation:

- Reuse `pin.LogCholeskyParameters` in 3.7 and 4.1; do not invent a second
  encoding. Order is `[alpha,d1,d2,d3,s12,s23,s13,t1,t2,t3]`, with
  `P=exp(2 alpha) U U.T` and
  `U=[[exp(d1),s12,s13,t1],[0,exp(d2),s23,t2],
  [0,0,exp(d3),t3],[0,0,0,1]]` in the link frame. Interpret numerical values
  in kg/metres; reference units/scales must be explicit. The inverse uses
  reverse Cholesky; derivative interface is `dp10/dz` (10×10), composed
  with the explicit full-column map and regressor for torque fitting.
  See the [upstream implementation](https://github.com/stack-of-tasks/pinocchio/blob/v3.7.0/include/pinocchio/spatial/inertia.hpp).
- Require positive-definite movable-link input for inversion. Reject singular
  PSD input by default; explicit opt-in interior repair reports its size and
  preserves the original. Keep massless/fixed links outside optimized blocks;
  do not assign them fictitious mass. This spike has no massless links.
- Validate shapes, symmetry, finiteness, reconstructed eigenvalues and mass.
  Production exponential guards must reject overflow/underflow before native
  conversion and check decoded outputs; ±12 here is an experiment bound,
  not a universal admissible range or a CAD constraint.
- Start from nominal or complete reconstructed per-link inertials; a QR base
  vector is not an independent full inertia estimate. Repair is a separate,
  reported stage. Rank deficiency requires an explicit prior/fixed-variable
  policy and separate identifiability diagnostics. Revise scaling/convergence
  before selecting final production defaults.
- This spike has no friction/actuator/offset extras. The first implementation
  may keep extras fixed with explicit metadata, rejecting requests to optimize
  them; active extras must retain their separate regressor columns. CAD/mass/
  CoM bounds are not implied by PD. Reject unsupported CAD constraints before
  solving; adding their enforcement is a separate validated extension.
- Keep SDP projection, reconstruction and nonlinear torque fitting distinctly
  named. Select an explicit final stage; later projection/reconstruction must
  not silently overwrite it. Failure preserves the prior valid stage. The
  nonlinear runtime need not call SDP, but PICOS stays a package dependency
  until a separate packaging decision. No Feature 2 or MuJoCo dependency.

The preselected accuracy/runtime gates are clean held-out RMSE ≤1e-5 Nm,
noisy full-excitation held-out RMSE ≤1.05× corrected OLS+SDP, and ≤20 s/fit
on the reference machine. A future go additionally requires **all nominal
and repaired-OLS starts to converge and be feasible** under a newly frozen,
justified budget. Weak excitation remains a diagnostic, not a promise of
full-inertial recovery. Boundary/perturbed failures need explicit failure
status and must not replace a valid earlier stage. Preserve this first
experiment and record any revised protocol before remeasurement.

## Reproduction and evidence

From the core repository in `figaroh-dev`:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python docs/development/spikes/log_cholesky_feasibility.py --output results.json
```

The script is private research support under docs, not an installed public
entry point. No production numerical behavior changes in this issue.

- [Pinocchio 3.7 results](../development/spikes/results/log-cholesky-pin37.json)
- [Pinocchio 4.1 results](../development/spikes/results/log-cholesky-pin41.json)

Records include the script SHA256, core base revision `90c8431`, complete
model placements/inertials, seeds/config, data hashes, dependency versions,
all statuses and parameters. Both use Python 3.12.11, NumPy 2.3.2,
SciPy 1.16.1, PICOS 2.6.1 and CVXOPT 1.3.2 on macOS. Original `figaroh-dev`
uses 3.7; an isolated clone at `/tmp/figaroh-pin41/figaroh-dev` uses 4.1.
Data hashes match between profiles. Solver timings need not reproduce bitwise.
Generated model and synthetic observations are entirely in core; no examples
revision or hardware result is involved. Real-robot held-out benchmarking
remains examples issue #11 after the production go gate.
