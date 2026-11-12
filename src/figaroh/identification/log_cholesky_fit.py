"""Opt-in, boundary-aware log-Cholesky physical fit (LC1, #24). Experimental.

:func:`solve_log_cholesky_fit` minimises the objective of
:func:`~figaroh.identification.physical_fit.solve_direct_effort_fit` on the
same :class:`~figaroh.identification.physical_fit.ComparatorProblem`::

    ||W (Y theta - tau)||^2 + lambda ||D^-1 (theta - theta0)||^2

subject to every link being physically consistent. ``W`` is the row weight,
``D`` the coordinate scale, ``theta0`` the prior and ``lambda`` the policy's
``prior_weight``. The method is the one accepted in
``docs/decisions/log-cholesky-boundary-aware.md``:

1. **Detect.** Compute the unconstrained (ridge) minimiser in closed form.
   It is the constrained optimum when every link has
   ``min eig(P) >= eigenvalue_margin``, ``m >= mass_min`` and its
   ``mass_ratio_bounds``; then the *interior* branch runs. Otherwise the
   optimum is on the boundary, which log-Cholesky coordinates cannot reach,
   and the *boundary* branch runs.
2. **Interior.** Fit in log-Cholesky coordinates (one 10-vector per link,
   :mod:`~figaroh.identification.inertial_parameterization`) plus the
   remaining linear columns, with the regressor null space pinned to the
   prior, and stop once the identifiable residual distance to the detected
   optimum is below ``stop_rms``.
3. **Boundary.** Run the direct LMI effort fit itself (PICOS/CVXOPT).

For this linear objective both branches return the convex optimum; the
interior branch is the feasible-by-construction path that objectives which
are not convex in ``theta`` need, and the record names the branch used. It
is a local method: no global-optimum claim is made for other objectives.

Every outcome is a :class:`~figaroh.identification.physical_fit.SolveRecord`
with an independent per-link feasibility check. ``theta`` is ``None``
whenever the fit did not converge (``invalid_start``, ``max_nfev``,
``time_budget``, ``nonfinite_trial``, ``not_converged`` or an SDP failure);
callers keep their previous valid stage.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.optimize import least_squares

from .inertial_parameterization import (
    MAX_LOG_SCALE,
    log_cholesky_from_p10,
    p10_from_log_cholesky,
    p10_jacobian_from_log_cholesky,
    pseudo_inertia_from_p10,
    repair_p10,
)
from .physical_consistency import mass_bound_tolerance
from .physical_fit import (
    ComparatorProblem,
    SolveRecord,
    _link_indices,
    _make_record,
    _solve_core,
)
from .reconstruction import _INERTIAL_KEYS

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = ["LogCholeskyOptions", "solve_log_cholesky_fit"]

_BRANCHES = ("auto", "interior", "boundary")
_METHODS = ("dogbox", "trf")


@dataclasses.dataclass(frozen=True)
class LogCholeskyOptions:
    """Options for :func:`solve_log_cholesky_fit` (defaults: #155 protocol).

    ``rank_rtol``: singular values below ``rank_rtol * s_max`` of the scaled
    weighted regressor span the pinned null space. ``eigenvalue_margin``:
    minimum pseudo-inertia eigenvalue for the interior branch.
    ``stop_rms``: interior stop on ``||M (q - q*)|| / sqrt(rows)``, in
    weighted-effort units (N*m for unit weights). ``max_nfev`` and
    ``max_seconds`` bound the interior fit. ``branch`` forces a branch
    (``"auto"`` detects). ``repair_start`` allows :func:`repair_p10` on
    start links that cannot be encoded.
    """

    rank_rtol: float = 1e-10
    eigenvalue_margin: float = 1e-6
    stop_rms: float = 1e-7
    max_nfev: int = 2000
    max_seconds: float = 60.0
    method: str = "dogbox"
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    max_log_scale: float = MAX_LOG_SCALE
    repair_start: bool = False
    branch: str = "auto"
    sdp_solver: str = "cvxopt"

    def __post_init__(self):
        if self.branch not in _BRANCHES:
            raise ValueError(f"branch must be one of {_BRANCHES}, got {self.branch!r}")
        if self.method not in _METHODS:
            raise ValueError(
                f"method must be one of {_METHODS} (bounded least squares), "
                f"got {self.method!r}"
            )
        for name in ("rank_rtol", "stop_rms", "max_seconds", "max_log_scale"):
            value = getattr(self, name)
            if not (np.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be positive and finite, got {value}")
        if not (np.isfinite(self.eigenvalue_margin) and self.eigenvalue_margin >= 0):
            raise ValueError("eigenvalue_margin must be nonnegative and finite")
        if int(self.max_nfev) < 1:
            raise ValueError(f"max_nfev must be >= 1, got {self.max_nfev}")


class _Stop(Exception):
    def __init__(self, reason: str, x: Optional[np.ndarray] = None):
        super().__init__(reason)
        self.reason = reason
        self.x = x


class _Objective:
    """The direct-fit objective in scaled deviations ``q = D^-1 (theta - theta0)``.

    Uses the problem's QR (``W Y = Q R``), so residuals have ``n`` rows instead
    of one per sample; the dropped orthogonal part is a constant.
    """

    def __init__(self, p: ComparatorProblem, rank_rtol: float):
        self.p = p
        self.D = np.asarray(p.coord_scale, dtype=float)
        self.lam = max(float(p.policy.prior_weight), 0.0)
        self.RD = p.R * self.D[None, :]
        self.c = p.Qt_tau - p.R @ p.theta_prior
        n = self.D.size
        U, S, Vt = np.linalg.svd(self.RD, full_matrices=True)
        s_max = float(S[0]) if S.size else 0.0
        self.rank = int(np.sum(S > s_max * rank_rtol)) if s_max > 0 else 0
        self.s_max = s_max
        r = self.rank
        # Ridge minimiser on the kept directions only. Singular values below
        # the cutoff are numerical noise; S / (S**2 + lam) would amplify them
        # by 1 / lam into a spurious null-space component (#234).
        self.q_star = Vt[:r].T @ (
            S[:r] / (S[:r] ** 2 + self.lam) * (U[:, :r].T @ self.c)
        )
        self.null = Vt[r:].T  # (n, n - rank)
        # Pinning the null component adds a term that vanishes at q_star (it
        # has none by construction), so it removes flat directions only.
        self.pin = max(s_max, 1.0)
        self.M = np.vstack(
            [self.RD, np.sqrt(self.lam) * np.eye(n), self.pin * self.null.T]
        )
        self.rhs = np.r_[self.c, np.zeros(n), np.zeros(n - self.rank)]
        self.rows = max(int(p.tau.size), 1)
        self.theta_star = p.theta_prior + self.D * self.q_star

    def q(self, theta: np.ndarray) -> np.ndarray:
        return (theta - self.p.theta_prior) / self.D

    def value(self, theta: np.ndarray) -> float:
        """Same convention as the direct SDP's objective value."""
        q = self.q(theta)
        r = self.RD @ q - self.c
        return float(r @ r + self.lam * q @ q)

    def identifiable_rms(self, q: np.ndarray) -> float:
        return float(np.linalg.norm(self.M @ (q - self.q_star)) / np.sqrt(self.rows))


def _detect(
    p: ComparatorProblem, obj: _Objective, margin: float, rank_rtol: float
) -> Dict[str, Any]:
    links = {}
    lo_tol = mass_bound_tolerance(p.policy.mass_min)
    for joint, idx in _link_indices(p).items():
        p10 = np.array([obj.theta_star[idx[k]] for k in _INERTIAL_KEYS])
        min_eig = float(np.linalg.eigvalsh(pseudo_inertia_from_p10(p10))[0])
        reasons = []
        if min_eig < margin:
            reasons.append("pseudo_inertia_margin")
        if p10[0] < p.policy.mass_min - lo_tol:
            reasons.append("mass_min")
        if p.policy.mass_ratio_bounds is not None:
            lo, hi = p.policy.mass_ratio_bounds
            m0 = float(p.theta_prior[idx["m"]])
            if not (lo * m0 - lo_tol <= p10[0] <= hi * m0 + lo_tol):
                reasons.append("mass_ratio_bounds")
        links[joint] = {"min_eig": min_eig, "mass": float(p10[0]), "active": reasons}
    active = [j for j, v in links.items() if v["active"]]
    return {
        "branch": "boundary" if active else "interior",
        "active_links": active,
        "unconstrained_min_eig": min(
            (v["min_eig"] for v in links.values()), default=None
        ),
        "per_link": links,
        "rank": obj.rank,
        "n_params": int(obj.D.size),
        "null_space_dim": int(obj.D.size - obj.rank),
        "rank_rtol": rank_rtol,
        "eigenvalue_margin": margin,
    }


class _Layout:
    """Variables ``x = [z_link1, ..., z_linkL, q_free]`` and their maps."""

    def __init__(self, p: ComparatorProblem):
        self.links = _link_indices(p)
        if not self.links:
            raise ValueError(
                "problem has no link with all ten inertial parameters; "
                "nothing to fit in log-Cholesky coordinates"
            )
        self.cols: List[List[int]] = [
            [idx[k] for k in _INERTIAL_KEYS] for idx in self.links.values()
        ]
        used = {c for cols in self.cols for c in cols}
        self.free = [i for i in range(len(p.params_std)) if i not in used]
        self.nz = 10 * len(self.cols)
        self.size = self.nz + len(self.free)


def _start_coordinates(
    p: ComparatorProblem,
    layout: _Layout,
    obj: _Objective,
    theta_start: np.ndarray,
    opts: LogCholeskyOptions,
):
    x0 = np.empty(layout.size)
    repairs: Dict[str, Dict[str, Any]] = {}
    for b, (joint, cols) in enumerate(zip(layout.links, layout.cols)):
        p10 = theta_start[cols]
        try:
            z = log_cholesky_from_p10(p10, max_log_scale=opts.max_log_scale)
        except ValueError as exc:
            if not opts.repair_start:
                return None, repairs, f"link {joint!r}: {exc}"
            try:
                report = repair_p10(p10)
                z = log_cholesky_from_p10(report.p10, max_log_scale=opts.max_log_scale)
            except ValueError as exc2:
                return None, repairs, f"link {joint!r} (after repair): {exc2}"
            repairs[joint] = {
                "min_eig_before": report.min_eig_before,
                "min_eig_after": report.min_eig_after,
                "p10_change_norm": report.p10_change_norm,
                "p10_relative_change": report.p10_relative_change,
            }
        x0[10 * b : 10 * b + 10] = z
    x0[layout.nz :] = obj.q(theta_start)[layout.free]
    return x0, repairs, None


def _fit_interior(p, obj, layout, x0, opts):
    begin = time.perf_counter()
    n = obj.D.size
    counts = {"nfev": 0, "njev": 0}

    def theta_of(x):
        theta = p.theta_prior.copy()
        for b, cols in enumerate(layout.cols):
            theta[cols] = p10_from_log_cholesky(
                x[10 * b : 10 * b + 10], max_log_scale=opts.max_log_scale
            )
        theta[layout.free] = (
            p.theta_prior[layout.free] + obj.D[layout.free] * x[layout.nz :]
        )
        return theta

    def residual(x):
        counts["nfev"] += 1
        if time.perf_counter() - begin > opts.max_seconds:
            raise _Stop("time_budget")
        try:
            q = obj.q(theta_of(x))
        except ValueError:
            raise _Stop("nonfinite_trial")
        r = obj.M @ q - obj.rhs
        if not np.all(np.isfinite(r)):
            raise _Stop("nonfinite_trial")
        if obj.identifiable_rms(q) <= opts.stop_rms:
            raise _Stop("converged", x.copy())
        return r

    def jacobian(x):
        counts["njev"] += 1
        dq = np.zeros((n, layout.size))
        for b, cols in enumerate(layout.cols):
            J = p10_jacobian_from_log_cholesky(
                x[10 * b : 10 * b + 10], max_log_scale=opts.max_log_scale
            )
            dq[cols, 10 * b : 10 * b + 10] = J / obj.D[cols][:, None]
        dq[layout.free, layout.nz + np.arange(len(layout.free))] = 1.0
        return obj.M @ dq

    lb = np.full(layout.size, -np.inf)
    ub = np.full(layout.size, np.inf)
    for b in range(len(layout.cols)):
        lb[10 * b : 10 * b + 4] = -opts.max_log_scale
        ub[10 * b : 10 * b + 4] = opts.max_log_scale

    info: Dict[str, Any] = {}
    x_final = None
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            result = least_squares(
                residual,
                x0,
                jac=jacobian,
                bounds=(lb, ub),
                method=opts.method,
                x_scale="jac",
                ftol=opts.ftol,
                xtol=opts.xtol,
                gtol=opts.gtol,
                max_nfev=int(opts.max_nfev),
            )
        info["scipy_status"] = int(result.status)
        info["scipy_message"] = str(result.message)
        q = obj.q(theta_of(result.x))
        if obj.identifiable_rms(q) <= opts.stop_rms:
            reason, x_final = "converged", result.x
        elif result.status == 0:
            reason = "max_nfev"
        else:
            reason = "not_converged"
        info["identifiable_rms"] = obj.identifiable_rms(q)
        info["max_abs_log_scale"] = float(
            max(
                np.abs(result.x[10 * b : 10 * b + 4]).max()
                for b in range(len(layout.cols))
            )
        )
    except _Stop as stop:
        reason, x_final = stop.reason, stop.x
        if x_final is not None:
            info["identifiable_rms"] = obj.identifiable_rms(obj.q(theta_of(x_final)))
    info.update(counts)
    info["stop_reason"] = reason
    info["runtime_s"] = time.perf_counter() - begin
    theta = theta_of(x_final) if reason == "converged" else None
    return theta, info


def solve_log_cholesky_fit(
    p: ComparatorProblem,
    *,
    start: Optional[np.ndarray] = None,
    options: Optional[LogCholeskyOptions] = None,
) -> SolveRecord:
    """Boundary-aware log-Cholesky fit on a problem from ``build_problem``.

    ``start`` is a full vector over ``p.params_std`` (default: the prior);
    only the interior branch uses it. Returns a ``SolveRecord`` with
    ``objective="log_cholesky_fit"``; ``notes`` holds the branch, detection
    (rank, pinned null-space dimension, per-link verdicts), start repairs and
    interior counters. Experimental; the module is not re-exported from
    ``figaroh.identification``.
    """
    opts = options or LogCholeskyOptions()
    t0 = time.perf_counter()
    obj = _Objective(p, opts.rank_rtol)
    layout = _Layout(p)
    detection = _detect(p, obj, opts.eigenvalue_margin, opts.rank_rtol)
    branch = detection["branch"] if opts.branch == "auto" else opts.branch
    notes: Dict[str, Any] = {
        "method": "boundary_aware_log_cholesky",
        "branch": branch,
        "branch_forced": opts.branch != "auto",
        "detection": detection,
        "options": dataclasses.asdict(opts),
    }

    if branch == "boundary":
        raw = _solve_core(p, "direct", solver=opts.sdp_solver)
        notes["start"] = {"used": False}
        raw["runtime_s"] = time.perf_counter() - t0
        record = _make_record(
            p, "log_cholesky_fit", solver=opts.sdp_solver, raw=raw, notes=notes
        )
        logger.info("log-Cholesky fit: boundary branch, status %s", raw["status"])
        return record

    theta_start = (
        p.theta_prior.copy()
        if start is None
        else np.asarray(start, dtype=float).reshape(len(p.params_std))
    )
    if not np.all(np.isfinite(theta_start)):
        raise ValueError("start has nonfinite entries")
    x0, repairs, error = _start_coordinates(p, layout, obj, theta_start, opts)
    notes["start"] = {
        "used": True,
        "source": "prior" if start is None else "given",
        "repairs": repairs,
    }
    raw: Dict[str, Any] = {
        "status": "invalid_start",
        "picos_status": None,
        "exception": None,
        "objective_value": None,
        "theta": None,
    }
    if error is not None:
        raw["exception"] = (
            f"{error}. Pass a physically consistent start or set "
            "LogCholeskyOptions(repair_start=True)"
        )
    else:
        theta, info = _fit_interior(p, obj, layout, x0, opts)
        notes["interior"] = info
        raw["status"] = "optimal" if theta is not None else info["stop_reason"]
        raw["theta"] = theta
        if theta is not None:
            raw["objective_value"] = obj.value(theta)
    raw["runtime_s"] = time.perf_counter() - t0
    logger.info("log-Cholesky fit: interior branch, status %s", raw["status"])
    return _make_record(p, "log_cholesky_fit", solver="scipy", raw=raw, notes=notes)
