# Log-Cholesky convergence: revised protocol (frozen before confirmation)

- Status: Accepted — **no-go** for the revised protocol (2026-10-08); protocol
  frozen the same day in a separate earlier commit
- Issue: [#30](https://github.com/thanhndv212/figaroh-plus/issues/30)
- Parent: [#40 (D5)](https://github.com/thanhndv212/figaroh-plus/issues/40)
- Supersedes the measurement protocol of
  [log-Cholesky feasibility](log-cholesky-feasibility.md) (#22). That record,
  its script and its results stay unchanged as the first experiment.
- Scope: private synthetic research spike; no production API approval

The protocol sections below were committed before any confirmation-seed run.
The results section was added afterwards in a separate commit, so the history
shows what was fixed in advance.

## Decision

**No-go.** Raising the budget alone does not make termination reliable. On the
fresh seeds, 8 of 24 gated fits run out of 2000 evaluations, on both Pinocchio
3.7 and 4.1. Every accuracy, feasibility and runtime gate passes, so the
remaining problem is termination, as in #22. Production issues #23–#25 stay
blocked. A next revision should change the method, not the budget again (see
the end of this record).

## Diagnosis: why 19 of 20 fits stopped

[Diagnosis script](../development/spikes/log_cholesky_convergence_diagnosis.py)
imports the #22 spike unchanged, rebuilds its fixtures, reproduces the recorded
evaluation counts and objectives for all eight nominal and repaired-OLS fits,
then traces each fit and runs sensitivity probes. Output:
[diagnosis results](../development/spikes/results/log-cholesky-diagnosis-pin37.json)
(Pinocchio 3.7).

- **The 200-evaluation budget is the proximate cause.** With every other
  setting frozen, a budget of 1000, 2000 or 20000 terminates all eight fits by
  `ftol` (status 2) in 197–598 evaluations, all feasible, clean held-out RMSE
  1.28e-6 Nm, at most 0.9 s per fit.
- **The remaining work is slow drift along unidentifiable directions.** At the
  200-evaluation stop, the error in the identifiable subspace of the regressor
  is about 2e-6 but 0.07–1.3 in its null space. The gradient has almost no
  null-space component (fraction about 1e-5), while the Gauss-Newton step is
  entirely in the null space and much longer than the accepted steps (0.25
  against about 5e-3 for clean nominal): the trust region limits progress. The
  null space rotates 17–67 degrees along the path because the log-Cholesky map
  is nonlinear. Jacobian condition number is about 2e6–4e6 (4e13 under weak
  excitation). The trust-region mechanism is inferred, not proven.
- **`gtol` never triggers under TRF.** Over each run the infinity-norm gradient
  stays 1.7e6–2.6e9 times above `gtol=1e-10`, because it scales with the squared
  largest singular value times the residual error. `ftol` is the criterion that
  terminates. Roundoff is not the barrier: dogbox reaches `gtol` in 7–264
  evaluations.
- **Scaling changes do not help.** `x_scale='jac'` lowers the condition number
  only from about 2e6 to 5.8e5; `x_scale=1`, scaling from the initial Jacobian
  and residual scaling by 1/σ or 1/√N all terminate at a 2000 budget with
  similar counts.
- **Prior strength trades accuracy for speed.** Prior 1e-4 terminates in 68–139
  evaluations but gives clean held-out 1.3e-4 Nm; 1e-2 gives 9e-3 Nm. Both fail
  the pre-declared 1e-5 Nm clean gate. Prior 1e-8 is slower (one fit reaches
  2000).
- **Near-boundary truth has no attainable optimum.** Its ridge minimiser is
  infeasible on four links (minimum pseudo-inertia eigenvalue -1.7e-2), so the
  infimum lies on the boundary of the positive-definite set. Fits stop by `ftol`
  with relative cost excess 6.6e-7.

## Revised protocol

Everything in the #22 protocol stays fixed except the items below: model
builder and generation, four cases, 120 training and 160 held-out samples,
0.03 Nm training noise, five starts per case, fixed scales, prior 1e-6, TRF with
analytic Jacobian and `x_scale='jac'`, `ftol=xtol=gtol=1e-10`, ±12 coordinate
bounds, 20-second guard, independent feasibility oracle with tolerance 1e-8,
and the accuracy/runtime gates.

| Item | #22 | Revised | Justification |
| --- | --- | --- | --- |
| Evaluation budget | 200 | **2000** | 3.3 × the worst diagnosed gated count (598); about 3 s per fit, well inside the 20 s guard. |
| Convergence | `result.success` was recorded but not gated | **status 1, 2 or 4** (gradient, cost or step tolerance) | Status 0 means the budget ran out; 3 (`xtol` alone) is not counted. |
| Seeds | one set (2201–2206) | **three fresh sets** declared below | One model draw cannot separate the protocol from the fixture. |
| Reported only | — | bound activity; relative cost excess over the ridge optimum | Diagnostics, not gates, so no new thresholds are introduced. |

Seed order is model, training, held-out, noise, perturbed start 1, perturbed
start 2.

| Set | Seeds | Role |
| --- | --- | --- |
| exploration | 2201–2206 | Replication of #22; not counted |
| confirmation_a | 3001–3006 | Gated |
| confirmation_b | 3101–3106 | Gated |
| confirmation_c | 3201–3206 | Gated |

## Gates (unchanged thresholds)

A **go** requires every confirmation set, on both Pinocchio 3.7 and 4.1, to pass:

- every nominal and repaired-OLS fit in all four cases converges (as defined
  above), is feasible on every link and runs in ≤20 s;
- clean held-out RMSE ≤1e-5 Nm for both of those starts;
- noisy held-out RMSE ≤1.05 × the corrected OLS+SDP held-out RMSE for both
  starts, with SDP itself successful.

Weak excitation is gated on convergence and feasibility only; its accuracy is
reported as a spread, not a claim of full inertial recovery. Near-boundary and
perturbed starts are reported, not gated.

## Failure semantics

A fit that does not converge keeps its record with `converged: false`, its
status and its evaluation count. Its parameters are diagnostics only. In a
production pipeline built after a go, such a fit must return an explicit
failure and leave the previous valid stage (OLS or SDP) as the result; it
must not replace that stage or be relabelled as success.

## Peeking disclosure

- The 2000 budget and the decision to keep prior 1e-6 were chosen with the
  exploration seeds, which the diagnosis used. Rejecting priors 1e-4 and 1e-2
  follows from the existing clean gate, not from tuning.
- Before freezing, the confirmation script was checked on the exploration set
  only. That replication passes the gates. It also showed that 6 of the 12
  near-boundary and perturbed starts still reach 2000 evaluations (status 0).
  These starts were already outside the gates in #22, and the budget was not
  changed after seeing this.
- No confirmation seed was run before this record was committed.

## Results (2026-10-08)

Run at core `3578b63` (protocol commit). Python 3.12.11, SciPy 1.16.1, PICOS
2.6.1, CVXOPT 1.3.2; Pinocchio 3.7.0 and 4.1.0. Both profiles give the same
status, evaluation count and held-out error for every fit. The exploration
replication passes the gates. Gated fits (nominal and repaired-OLS starts):

| Set | Converged | Not converged (status 0 at 2000) |
| --- | --- | --- |
| confirmation_a | 5 / 8 | clean nominal, weak-excitation nominal, near-boundary repaired-OLS |
| confirmation_b | 4 / 8 | clean nominal and repaired-OLS, near-boundary nominal and repaired-OLS |
| confirmation_c | 7 / 8 | clean nominal |

Converged gated fits used 209–1875 evaluations; three needed more than 1700.
All 24 gated fits are feasible on every link and take at most 3.9 s.

Accuracy passes everywhere, including in non-converged fits:

| Set | Clean held-out (Nm) | Noisy held-out, log-Cholesky / OLS+SDP (Nm) |
| --- | --- | --- |
| confirmation_a | 1.24e-6, 1.26e-6 | 0.0164 / 0.0303 |
| confirmation_b | 1.7e-7, 9.8e-8 | 0.0093 / 2.19 |
| confirmation_c | 9.5e-7, 9.5e-7 | 0.0112 / 0.223 |

OLS+SDP held-out error on the fresh seeds is much larger than on the
exploration seeds; per-link projection of OLS changes the predictions far more
there. The log-Cholesky candidates beat it by 1.8× to 235×, but the go decision
depends on termination, which fails.

**Observed pattern (not proven).** On the fresh seeds, the ridge optimum (the
unconstrained minimiser of the same objective) is physically infeasible in
11 of 12 cases, on 1–5 links. On the exploration seeds it was infeasible only
in the near-boundary case. When the ridge optimum is infeasible, the
constrained infimum lies on the boundary of the positive-definite set, which
log-Cholesky coordinates reach only as some coordinate goes to minus infinity.
Every non-converged gated fit is in such a case. Not every such case fails,
so infeasibility is consistent with, but not sufficient for, non-termination.
The clean cases show that the identifiable part of the solution is accurate
long before the solver stops (relative cost excess over the ridge optimum
7e-4 to 5e-2, against an unattainable reference).

The exploration seeds therefore underrepresented the boundary case that
dominates the fresh draws. That is why the budget chosen from them did not
transfer.

## Next revision (proposal, not frozen)

Change what the solver is asked to do rather than how long it runs:

- Treat a boundary infimum explicitly: detect an infeasible ridge optimum
  first, and in that case fit with a barrier or margin on the pseudo-inertia
  eigenvalues (or the SDP directly on the torque objective) instead of letting
  log coordinates diverge.
- Remove the unidentifiable drift: fix the regressor null-space component to
  the prior (fit only identifiable combinations plus the feasibility margin),
  or test dogbox/LM, which terminated quickly in the diagnosis.
- Declare a stopping rule on the identifiable residual, not only on SciPy's
  status, and freeze it with new fresh seeds before measuring.

## Reproduction

From the core repository, once per Pinocchio profile:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src \
  python docs/development/spikes/log_cholesky_convergence.py --output results.json
```

[Confirmation script](../development/spikes/log_cholesky_convergence.py)
imports the #22 spike and the diagnosis module unchanged, overrides only the
seeds and budget, and records the SHA256 of all three scripts.
