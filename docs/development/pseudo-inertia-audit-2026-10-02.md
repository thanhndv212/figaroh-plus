# Pseudo-inertia convention audit — 2026-10-02

Local implementation and validation for [issue #21](https://github.com/thanhndv212/figaroh-plus/issues/21),
the correctness prerequisite for the Feature 3 feasibility benchmark.
This records local evidence; issue closure and integration require the normal
review/merge workflow. Log-Cholesky fitting is not implemented by this change.

## Findings and corrected behavior

1. The NumPy fallback used rotational inertia as the pseudo-inertia upper
   block. It now uses the second moment
   `Sigma = 0.5*trace(I_O)*eye(3) - I_O`, matching Pinocchio.
2. SDP projection optimized and returned entries of `Sigma` as though they
   were entries of `I_O`. Its objective now compares the original dynamic
   parameters, and extraction uses `I_O = trace(Sigma)*eye(3) - Sigma`.
3. First-moment weighting used a PICOS vector multiplication that penalized
   a weighted sum rather than each component. An explicit diagonal weight
   matrix now preserves feasible nonzero-CoM inputs.
4. SDP reconstruction constrained `[[I_O, h], [h.T, m]]`. Its LMI now uses
   `Sigma`, enforcing the rotational-inertia triangle inequalities while
   retaining the base-parameter equalities.
5. URDF reconstruction priors read Pinocchio's CoM tensor into link-origin
   dynamic parameters. `toDynamicParameters()` now includes the required
   parallel-axis contribution and preserves named/reordered subsets.

CAD mass bounds and the existing first-moment semantics of `com_bounds`
remain supported. No dependency, configuration selector or result-stage
contract changed.

## Environment and revisions

- Core base: `f4a5b589a64dd0b28fa30e7c072f2c271a83bc99`, plus the local issue-21 diff.
- Examples: `3a2c8e9b07e10b397cda79dbb04a5e88469bcc22`; no example source changes.
- Environment: `figaroh-dev`, Python 3.12.11, macOS 26.6.2 arm64.
- NumPy 2.3.4, SciPy 1.16.1, Pinocchio 3.7.0, PICOS 2.6.1, CVXOPT 1.3.2.

## V0/V1 checks

Commands run from the core checkout, with `conda activate figaroh-dev`:

```bash
python -m pytest tests/unit/test_pseudo_inertia_conventions.py -q -rs
python -m pytest -q -rs --tb=short
python -m flake8 src tests --select=E9,F63,F7,F82 --show-source --statistics
pre-commit run --files src/figaroh/identification/physical_consistency.py \
  src/figaroh/identification/reconstruction.py \
  tests/unit/test_pseudo_inertia_conventions.py CHANGELOG.md \
  docs/source/api/identification.md docs/development/pseudo-inertia-audit-2026-10-02.md
python -m mkdocs build
git diff --check
```

- Seven new convention regressions pass. Before the fix, five of the initial
  six regressions failed; the URDF prior regression was added during the audit.
- Full suite: **534 passed, 6 skipped**. Four skips require visual-test opt-in;
  two existing regressor tests use incompatible mocks. No new skip was added
  to suppress a failure.
- Critical lint and changed-file hooks pass. Black initially reformatted the
  touched Python files; the rerun passed.
- Docs build passes with existing docstring/link warnings.

Independent checks include an explicit diagonal second moment, a translated
body with off-diagonal inertia, and Pinocchio's canonical conversions. The
weighted triangle-inequality regression has the analytic projection
`(Ixx, Iyy, Izz) = (13/9, 13/9, 26/9)` from `(1, 1, 3)` when the `Izz`
weight is two and other weights are one. Reconstruction fixes the first nine
dynamic entries and checks that `Izz` is reduced from three to two.

## V2 UR10 evidence

A temporary headless validation harness ran from
`figaroh-examples/examples/ur10` using `MPLBACKEND=Agg`. It follows the example
CLI's active-joint setup, enables physical projection with
`skip_if_feasible=False`, and runs `solve(decimate=False, plotting=False,
save_results=False, html_report=False)` with the existing independent
`data/validation` trajectory. It also exports an HTML report outside the repo.

Projection completed with status `ok` for all six links. The normal example
produced 398 validation samples, identified torque RMSE 0.0723657 N*m versus
nominal RMSE 0.0414295 N*m, and correlation 0.9999988. The fit therefore
**worsened held-out RMSE** despite high correlation. These pipeline numbers
establish execution coverage, not an identification-quality improvement.
Existing example preprocessing limitations remain; they must be handled
explicitly before the Feature 3 feasibility benchmark.

Separately, the harness projects the canonical URDF inertials with the old
module from `f4a5b58` and the corrected module, using identical auto weights.
It evaluates inverse dynamics against the original model at 50 deterministic
states (NumPy seed 21, q uniform in [-1, 1] rad, dq/ddq standard normal in
rad/s and rad/s²). This comparison uses no finite-difference preprocessing.

| Metric | Before | Corrected |
|---|---:|---:|
| Projection status | partial | ok |
| Minimum canonical pseudo-inertia eigenvalue | -0.096201 | 0.00135095 |
| Maximum absolute dynamic-parameter change | 3.62015 | 0.0000277446 |
| Inverse-dynamics RMSE versus original URDF (N*m) | 16.4120 | 0.000110604 |

The absolute parameter maximum combines entries with different units and is
only a round-trip diagnostic; torque RMSE is the comparable dynamics metric.
The corrected projection preserves the nominal physical model up to solver
precision. This is simulation/model evidence, with no physical-hardware or
URDF-export claim.

Local artifacts from this run (temporary, not published CI artifacts):

- `/tmp/figaroh-issue21-ur10-check.py`: validation harness.
- `/tmp/figaroh-issue21-validation/metrics.json`: full numerical output.
- `/tmp/figaroh-issue21-validation/ur10-report.html`: workflow report.
- `/tmp/figaroh-issue21-ur10.log`, `/tmp/figaroh-issue21-pytest.log`,
  `/tmp/figaroh-issue21-mkdocs.log`: command output.

V3 release/milestone checks and hosted checks have not been run. The next
delivery item is [issue #22](https://github.com/thanhndv212/figaroh-plus/issues/22):
compare corrected SDP projection against a log-Cholesky prototype and record
the go/no-go decision before production implementation.
