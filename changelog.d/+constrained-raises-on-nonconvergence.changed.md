`LinearSolver(method="constrained")` raises `RuntimeError` when the
optimizer does not converge, instead of returning its last iterate.
`solver_info` still holds `success` and `message`.
