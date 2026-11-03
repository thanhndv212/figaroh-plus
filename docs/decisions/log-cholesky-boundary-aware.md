# Log-Cholesky boundary-aware fit: method change (frozen before confirmation)

- Status: Proposed — protocol frozen (2026-11-03); confirmation not yet run
- Issue: [#155](https://github.com/thanhndv212/figaroh-plus/issues/155)
- Parent: [#20 (LC1)](https://github.com/thanhndv212/figaroh-plus/issues/20)
- Follows the **no-go** of [log-Cholesky convergence](log-cholesky-convergence.md)
  (#30), whose "next revision" proposal this record implements. The #22 and
  #30 protocols, scripts and results stay unchanged.
- Scope: private synthetic research spike; no production API approval

The protocol sections below are committed before any confirmation-seed run.
Results will be added in a separate later commit, so the history shows what
was fixed in advance.

## What changes

#30 kept the method and raised the budget. Its diagnosis found two causes of
non-termination: the constrained infimum often lies on the boundary of the
positive-definite set, which log-Cholesky coordinates reach only at minus
infinity, and the solver drifts slowly along the regressor null space. This
revision changes what the solver is asked to do.

The objective is the #22 objective, unchanged:
`f(p) = 0.5 ||Y p - tau||^2 + 0.5 * 1e-6 * ||(p - prior) / scale||^2`.
It is convex in the dynamic parameters `p`; only the physical constraint is
not. The method returns the minimiser of `f` subject to every link's
pseudo-inertia satisfying `P_j >= 1e-6 I`, in three steps.

1. **Detect.** Solve `f` without the physical constraint in closed form
   (SVD of `Y diag(scale)`, relative rank cutoff 1e-10). If the minimiser's
   smallest pseudo-inertia eigenvalue over all links is at least the margin
   1e-6, it is the exact optimum of the log-Cholesky problem: the branch is
   **interior**. Otherwise the branch is **boundary**.
2. **Interior: log-Cholesky fit with the null space pinned.** Fit in
   log-Cholesky coordinates from the given start, adding residual rows
   `sigma_max * N.T q` to the #22 residual, where `q = (p - prior) / scale`,
   `N` is the null-space basis of `Y diag(scale)` and `sigma_max` its largest
   singular value. The unconstrained minimiser has no null-space component in
   `q`, so the extra term vanishes there and the minimiser is unchanged; it
   only makes the unidentifiable directions as stiff as the best-identified
   one. Solver: SciPy `least_squares`, **dogbox**, analytic Jacobian,
   `x_scale='jac'`, ±12 coordinate bounds, `ftol=xtol=gtol=1e-10`, 2000
   evaluations and the 20-second guard (budget and guard unchanged from #30).
3. **Boundary: SDP on the torque objective.** Solve `f` subject to
   `P_j >= 1e-6 I` for every link as one convex problem (PICOS 2.6 with
   CVXOPT, a second-order cone for the objective after QR of the stacked
   residual, absolute/relative feasibility and optimality tolerances 1e-9).
   No log coordinates are used, so nothing can diverge. The start does not
   enter this branch.

The margin 1e-6 is the #22 repair eigenvalue floor; using it for both
detection and the SDP constraint makes the branches consistent: the interior
branch is taken exactly when the margin constraint would be inactive.

### Stopping rule and convergence

The interior fit stops, through SciPy's callback, as soon as the
**identifiable residual distance** to the detected optimum is at most
**1e-7 Nm**:

`||M (q - q_star)|| / sqrt(n_tau) <= 1e-7`, with
`M = [Y diag(scale); sqrt(1e-6) I; sigma_max N.T]`.

Because the pinned objective is quadratic in `q` with Hessian `M.T M`, this
is the square root of twice the cost excess over the optimum, per training
torque sample: an RMS torque difference. 1e-7 Nm is 100 times below the clean
held-out gate. A fit **converges** when it ends without an exception and
satisfies this rule, whatever SciPy's status; budget exhaustion without the
rule is non-convergence. A boundary fit converges when the solver claims
status `optimal`. In both branches the independent #22 feasibility oracle
(tolerance 1e-8) is a separate gate, not part of convergence.

## Development and peeking disclosure

The method, the pin weight, the stopping rule and the solver were chosen on
**development sets that were seen before this record**: the #22 exploration
seeds (2201–2206) and the three #30 confirmation sets (3001–3206). Those are
the fixtures that the #30 no-go was measured on, so they are not counted.

- 20 development fits take the interior branch (15 on 2201, 5 on 3201).
  With the pinned objective, TRF used 133–2000 evaluations on them and hit
  the budget on 4 of 20; LM (no bounds) hit it on 3 and stopped short of the
  stopping rule on 3 more. **Dogbox converged all 20 in 9–53 evaluations.**
- Without the pin, dogbox terminates away from the optimum on 8 of the 20
  interior fits, which fail the stopping rule; the pin is needed.
- On the #30 confirmation seeds, 11 of 12 fixtures (55 of 60 fits) take
  the boundary branch, in line with #30's observation that the ridge optimum is usually infeasible
  there. All four development sets pass every gate on Pinocchio 3.7 and 4.1,
  with identical branches, evaluation counts and data hashes; held-out errors
  agree to 5e-8 relative.
- On every #30 fixture, the boundary SDP objective is at most 8.6e-7 relative
  above the lowest objective that any #30 log-Cholesky fit reached. The excess
  is the 1e-6 margin: with margin 1e-8 or 0 the SDP is below all #30 fits.
- Running the SDP alone on the interior fixtures gives the same held-out
  predictions as the interior branch to 2e-9 Nm. For this linear-in-parameter
  objective the convex problem alone determines the answer; the interior
  log-Cholesky branch is a representation that must reach it reliably, not a
  different estimator.
- The margin, the 1e-7 Nm rule and the 1e-9 SDP tolerances were set before
  the first development run and not changed afterwards; only the solver
  (TRF → dogbox) changed after seeing development results.

## Fresh seeds (declared here, never run before this commit)

Seed order is model, training, held-out, noise, perturbed start 1, perturbed
start 2.

| Set | Seeds | Role |
| --- | --- | --- |
| development_2201 | 2201–2206 | #22 exploration; not counted |
| development_3001/3101/3201 | 3001–3206 | #30 confirmation; not counted |
| confirmation_a | 4001–4006 | Gated |
| confirmation_b | 4101–4106 | Gated |
| confirmation_c | 4201–4206 | Gated |

## Gates (unchanged thresholds from #30)

A **go** requires every confirmation set, on both Pinocchio 3.7 and 4.1, to pass:

- every nominal and repaired-OLS fit in all four cases converges (as defined
  above), is feasible on every link and runs in ≤20 s;
- clean held-out RMSE ≤1e-5 Nm for both of those starts;
- noisy held-out RMSE ≤1.05 × the corrected OLS+SDP held-out RMSE for both
  starts, with SDP itself successful.

Weak excitation is gated on convergence and feasibility only. Near-boundary
and perturbed starts are reported, not gated. No threshold changes. The
convergence definition changes as stated above: the #30 definition (SciPy
status 1, 2 or 4) did not measure distance to the optimum, which is what
termination is meant to certify.

## Failure semantics

Unchanged from #30: a non-converged fit keeps its record with
`converged: false`, its branch and status. In a production pipeline built
after a go it must return an explicit failure and leave the previous valid
stage as the result.

## Reproduction

From the core repository, once per Pinocchio profile:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python docs/development/spikes/log_cholesky_boundary_aware.py --output results.json
```

[Script](../development/spikes/log_cholesky_boundary_aware.py) imports the
#22 spike and the #30 diagnosis unchanged, asserts that it rebuilds their
fixtures bit for bit, and records the SHA256 of all three scripts.
