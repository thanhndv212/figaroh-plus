# Log-Cholesky boundary-aware fit: method change (frozen before confirmation)

- Status: Accepted — **go** for the boundary-aware method (2026-11-03);
  protocol frozen the same day in a separate earlier commit
- Amended 2026-11-06 ([#234](https://github.com/thanhndv212/figaroh-plus/issues/234)):
  production computes the detected optimum on the kept rank only; see
  [Amendment](#amendment-2026-11-06-null-space-component-of-the-optimum)
- Issue: [#155](https://github.com/thanhndv212/figaroh-plus/issues/155)
- Parent: [#20 (LC1)](https://github.com/thanhndv212/figaroh-plus/issues/20)
- Follows the **no-go** of [log-Cholesky convergence](log-cholesky-convergence.md)
  (#30), whose "next revision" proposal this record implements. The #22 and
  #30 protocols, scripts and results stay unchanged.
- Scope: private synthetic research spike; no production API approval

The protocol sections below were committed before any confirmation-seed run.
The decision and results sections were added afterwards in a separate commit,
so the history shows what was fixed in advance.

## Decision

**Go**, on the private synthetic spike. On the fresh seeds, all 24 gated fits
converge, are feasible and pass every accuracy and runtime gate, on both
Pinocchio 3.7 and 4.1; so do all 36 reported near-boundary and perturbed
starts. #23 is unblocked. The go covers termination of this method on these
fixtures; it does not approve a production API, and the shape of #23–#25
should take the consequences below into account.

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

## Results (2026-11-03)

Run at core `6127b63` (protocol commit). Python 3.12.11, SciPy 1.16.1, PICOS
2.6.1, CVXOPT 1.3.2, NumPy 2.3.4; Pinocchio 3.7.0 (`figaroh-dev`) and 4.1.0
(a pip virtual environment with `ci/pinocchio-4.1.0.txt`). The 3.7 output
lists NumPy 2.3.2 because `figaroh-dev` contains a stale `numpy-2.3.2`
metadata directory next to the imported 2.3.4. Both profiles give the same
branch, evaluation count, convergence and data hash for every fit; held-out
errors agree to 5e-8 relative.

| Set | Branch (clean, noisy, weak, near-boundary) | Converged (all starts) | Max runtime |
| --- | --- | --- | --- |
| confirmation_a | boundary, boundary, boundary, boundary | 20 / 20 | 0.04 s |
| confirmation_b | boundary, boundary, boundary, boundary | 20 / 20 | 0.02 s |
| confirmation_c | boundary, boundary, interior, boundary | 20 / 20 | 0.05 s |

The unconstrained optimum is infeasible on 1–6 links in 11 of 12 fixtures, as
on the #30 seeds, so 22 of the 24 gated fits are decided by the boundary SDP,
all with status `optimal`. The one interior fixture (confirmation_c, weak
excitation) converges from all five starts in 9–33 evaluations, each stopped
by the identifiable-residual rule. Every result is feasible on every link;
the smallest pseudo-inertia eigenvalue is the 1e-6 margin.

| Set | Clean held-out, nominal / repaired OLS (Nm) | Noisy held-out, method / OLS+SDP (Nm) |
| --- | --- | --- |
| confirmation_a | 3.40e-6 / 3.40e-6 | 0.0163 / 0.8155 |
| confirmation_b | 3.84e-7 / 3.84e-7 | 0.0254 / 0.0595 |
| confirmation_c | 5.74e-7 / 5.74e-7 | 0.0119 / 0.0567 |

Nominal and repaired-OLS starts give the same result because both branches
return the same constrained optimum; in the boundary branch the start is not
used at all.

## Consequences for #23–#25 (not frozen)

- The boundary is the common case, not an edge case: on 15 of 16 fixtures
  drawn after #22 the constrained optimum lies on the margin. A log-Cholesky
  solver alone cannot return it; any production fit needs the detection step
  and a convex boundary solver.
- For the linear-in-parameter torque objective, the margin SDP alone returns
  the same answer as both branches (development check above). The interior
  log-Cholesky branch adds a representation that is feasible by construction,
  not a different estimator. #23–#25 should state what that representation is
  for (for example nonlinear terms, or objectives that are not convex in `p`)
  before committing to it as the user-facing solver. The boundary branch is
  the same kind of problem as the direct LMI effort fit that D4 (#39)
  selected for the D7 reference, so the two should share one implementation.
- The null-space pin uses the training regressor's null space. Production
  needs the same explicit rank-cutoff and prior policy, and must report which
  directions were pinned.
- The SDP depends on PICOS/CVXOPT, which are already package dependencies.

## Amendment 2026-11-06: null-space component of the optimum

Step 2 says the unconstrained minimiser has no null-space component, so the
pin vanishes there. That holds in exact arithmetic, but the spike and #24
computed it as `V diag(S / (S**2 + lam)) U.T e` over **all** singular values.
Values below the rank cutoff are numerical noise (about 1e-16 relative), and
the ridge factor amplifies them by `1 / lam` into a null-space component
whenever the torques have a part the regressor cannot explain. The pin weight
`sigma_max` then turns that into a distance the stop rule never reaches.

On `buildSampleModelManipulator` with the #25 integration data (400 samples,
`sigma_max` 7268), this component was 2.4e-6 in `q`, so the interior fit stopped
at identifiable RMS 3.5e-4 Nm and was rejected. The torque part of that
distance was 4e-13 Nm, so the estimate itself was already correct. On the
synthetic spike fixtures the component stays at round-off level, so the results
above are unaffected.

Production (`figaroh.identification.log_cholesky_fit`) now restricts the
minimiser to the kept directions, which makes the statement in step 2 true by
construction, and also covers `prior_weight = 0`, where the old factor becomes
`1 / S`. The method, pin, stopping rule, thresholds and solver are unchanged.
The spike script stays frozen as the record of what was run.

## Reproduction

From the core repository, once per Pinocchio profile:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python docs/development/spikes/log_cholesky_boundary_aware.py --output results.json
```

[Script](../development/spikes/log_cholesky_boundary_aware.py) imports the
#22 spike and the #30 diagnosis unchanged, asserts that it rebuilds their
fixtures bit for bit, and records the SHA256 of all three scripts.
