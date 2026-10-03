# Scoped acceptance policy

Status: Accepted on 2026-10-03 following the maintainer's acceptance review.
Implementation/review: core #70 and examples #36. Broader stage-aware result
integration remains #55/#63; this bounded correction does not clear S1.

Universal 50% improvement, 0.9 correlation, raw conditioning 1000, 2 mm position
and 0.1 degree orientation gates have no application-independent justification.
Preserve these metrics as diagnostics; callers may impose explicit reviewed
limits, but the library does not choose scientific thresholds for them.

Keep the existing verdict boolean for compatibility and add scope/status/stage
and applied-policy evidence. Unscoped library verification defaults to prediction acceptance, preventing old
boolean-only consumers from receiving a looser scientific PASS. Example CLIs
explicitly request and name numerical execution.
Prediction additionally requires a loaded separate validation set and explicit
per-output error limits. Missing required evidence is not evaluated; nonfinite
computed evidence fails. Empty threshold policies cannot establish acceptance.
Advisory checks are explicit and cannot silently substitute for required ones.

Physical, export and general data-provenance certification are not implemented
by this bounded verifier. They stay not evaluated rather than being inferred from
a finite base fit. Prediction scope checks quantitative errors under a declared
split; the experiment must still establish independent acquisition and coverage.

The examples must name the scope in CLI output, preserve complete subprocess
logs and actual exit/timeout status, and reject required timeouts. Historical
results are preserved. A revised execution PASS is a policy change, not an
improvement to fitting, physical feasibility or hardware performance.

No automatic train/validation split, new estimator or universal RMSE/NRMSE gate
is introduced. Future per-joint normalized errors, uncertainty/scaled singular
values and physical/export integration retain their existing delivery owners.
