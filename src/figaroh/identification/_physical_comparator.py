# Copyright [2021-2025] Thanh Nguyen
# Copyright [2022-2023] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Private comparator for physically consistent parameter recovery.

Three *different* objectives are compared on one shared problem; none of
them silently falls back to another:

1. ``solve_exact_reconstruction`` - closest-to-prior theta that reproduces
   the OLS base parameters exactly (``M_full theta = phi_ols``) with a
   pseudo-inertia LMI per link.
2. ``solve_direct_effort_fit`` - regularised effort fit with the same LMIs
   and bounds, no base equality.
3. ``solve_per_link_projection`` - nullspace representative of
   ``phi_ols`` followed by an independent LMI projection per link.

``diagnose_exact`` runs staged diagnostics (D0..D3) to explain why
objective 1 fails on a given problem.

This module is private (not exported from ``figaroh.identification``) and
carries no production API or configuration keys.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from figaroh.identification.physical_consistency import (
    check_p10_feasibility,
    project_p10_lmi,
    pseudo_inertia_matrix_from_p10,  # noqa: F401  (documented dependency)
)
from figaroh.identification.reconstruction import (
    _p10_indices_for_joints,
    reconstruct_full_parameters,
)
from figaroh.tools.qrdecomposition import QRDecomposer

_P10_KEYS = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedExtras:
    """Parameters held fixed (e.g. friction); their effort is subtracted.

    ``names`` must be column names of ``Y_std``; ``source`` records where
    ``values`` came from (``truth``, ``zero`` or ``declared``).
    """

    names: Tuple[str, ...] = ()
    values: Tuple[float, ...] = ()
    source: str = "zero"

    def __post_init__(self):
        if self.source not in ("truth", "zero", "declared"):
            raise ValueError(f"invalid FixedExtras.source={self.source!r}")
        if len(self.names) != len(self.values):
            raise ValueError("FixedExtras names/values length mismatch")


@dataclass(frozen=True)
class PhysicalPolicy:
    mass_min: float = 1e-6
    mass_ratio_bounds: Optional[Tuple[float, float]] = None
    feas_tol: float = -1e-8
    base_rel_tol: float = 1e-6
    prior_weight: float = 1e-6


@dataclass(frozen=True)
class ComparatorProblem:
    params_std: Tuple[str, ...]
    joint_names: Tuple[str, ...]
    Y: np.ndarray
    tau: np.ndarray
    row_weight: np.ndarray
    theta_prior: np.ndarray
    coord_scale: np.ndarray
    M_full: np.ndarray
    phi_ols: np.ndarray
    R: np.ndarray
    Qt_tau: np.ndarray
    extras: FixedExtras
    policy: PhysicalPolicy
    theta_truth: Optional[np.ndarray] = None
    base_indices: Tuple[int, ...] = ()
    removed_columns: Tuple[int, ...] = ()


@dataclass
class SolveRecord:
    objective: str
    solver: str
    solver_status: str
    picos_status: Optional[str]
    exception: Optional[str]
    runtime_s: float
    objective_value: Optional[float]
    theta: Optional[np.ndarray]
    fallback_used: bool
    fallback_candidate: Optional[np.ndarray]
    base_residual: Dict[str, Optional[float]]
    effort_rmse_fit_per_joint: Dict[str, float]
    feasibility: Dict[str, Dict[str, Any]]
    accepted: bool
    truth_error: Optional[Dict[str, float]]
    inputs_hash: str
    notes: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        def arr(x):
            return None if x is None else [float(v) for v in np.asarray(x).ravel()]

        d = {
            "objective": self.objective,
            "solver": self.solver,
            "solver_status": self.solver_status,
            "picos_status": self.picos_status,
            "exception": self.exception,
            "runtime_s": _f(self.runtime_s),
            "objective_value": _f(self.objective_value),
            "theta": arr(self.theta),
            "fallback_used": False,
            "fallback_candidate": arr(self.fallback_candidate),
            "base_residual": {k: _f(v) for k, v in self.base_residual.items()},
            "effort_rmse_fit_per_joint": {
                k: _f(v) for k, v in self.effort_rmse_fit_per_joint.items()
            },
            "feasibility": {
                k: {
                    "mass": _f(v["mass"]),
                    "min_eig": _f(v["min_eig"]),
                    "ok": bool(v["ok"]),
                }
                for k, v in self.feasibility.items()
            },
            "accepted": bool(self.accepted),
            "truth_error": (
                None
                if self.truth_error is None
                else {k: _f(v) for k, v in self.truth_error.items()}
            ),
            "inputs_hash": self.inputs_hash,
        }
        if self.notes:
            d["notes"] = json.loads(json.dumps(self.notes, default=str))
        return d


def _f(x: Optional[float]) -> Optional[float]:
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


# ---------------------------------------------------------------------------
# Problem construction
# ---------------------------------------------------------------------------


def _coord_scale(params: Sequence[str], joints: Sequence[str], prior: np.ndarray):
    scale = np.maximum(np.abs(prior), 1.0)
    for idx in _p10_indices_for_joints(params, joints).values():
        p = np.array([prior[idx[k]] for k in _P10_KEYS])
        ms = max(abs(p[0]), 1e-3)
        hs = max(float(np.linalg.norm(p[1:4])), 1e-3)
        is_ = max(float(np.linalg.norm(p[4:10])), 1e-3)
        for k in _P10_KEYS:
            scale[idx[k]] = (
                ms if k == "m" else hs if k in ("mx", "my", "mz") else is_
            )
    return scale


def build_problem(
    Y_std: np.ndarray,
    tau: np.ndarray,
    params_std: Sequence[str],
    joint_names: Sequence[str],
    *,
    prior: np.ndarray,
    extras: Optional[FixedExtras] = None,
    policy: Optional[PhysicalPolicy] = None,
    row_weight: Optional[np.ndarray] = None,
    qr_tol: float = 1e-6,
    theta_truth: Optional[np.ndarray] = None,
) -> ComparatorProblem:
    """Assemble the shared problem.

    ``prior``/``theta_truth`` are given over ``params_std``. Columns named in
    ``extras`` are removed from the unknowns and their effort is subtracted
    from ``tau``. Columns removed by the QR step stay in ``M_full`` as zeros
    so every link keeps its LMI.
    """
    extras = extras or FixedExtras()
    policy = policy or PhysicalPolicy()
    Y_std = np.asarray(Y_std, dtype=float)
    tau = np.asarray(tau, dtype=float).reshape(-1)
    names = list(params_std)
    prior = np.asarray(prior, dtype=float).reshape(len(names))
    truth = (
        None
        if theta_truth is None
        else np.asarray(theta_truth, dtype=float).reshape(len(names))
    )
    if Y_std.shape != (tau.size, len(names)):
        raise ValueError(
            f"Y_std shape {Y_std.shape} inconsistent with tau {tau.shape} "
            f"and {len(names)} parameters"
        )

    # Freeze extras
    tau = tau.copy()
    keep = np.ones(len(names), dtype=bool)
    for nm, val in zip(extras.names, extras.values):
        j = names.index(nm)
        tau = tau - Y_std[:, j] * float(val)
        keep[j] = False
    names = [n for n, k in zip(names, keep) if k]
    Y = Y_std[:, keep]
    prior = prior[keep]
    truth = None if truth is None else truth[keep]

    w = (
        np.ones(tau.size)
        if row_weight is None
        else np.asarray(row_weight, dtype=float).reshape(tau.size)
    )
    Yw = w[:, None] * Y
    tw = w * tau
    n = len(names)

    # Core QR decomposer on the non-zero columns; removed columns stay zero.
    colnorm = np.linalg.norm(Yw, axis=0)
    thresh = qr_tol * max(float(colnorm.max()), 1e-300)
    kept = [i for i in range(n) if colnorm[i] > thresh]
    removed = tuple(i for i in range(n) if i not in kept)
    params_r = [names[i] for i in kept]
    dec = QRDecomposer(tolerance=qr_tol)
    M_r, _, base_r, _ = dec.get_base_mapping_matrix_double(Yw[:, kept], params_r)
    M_full = QRDecomposer.expand_mapping_matrix_to_full(M_r, params_r, names)
    base_full = tuple(kept[i] for i in base_r)
    phi_ols, *_ = np.linalg.lstsq(Yw[:, list(base_full)], tw, rcond=None)

    Q, R = np.linalg.qr(Yw, mode="reduced")
    return ComparatorProblem(
        params_std=tuple(names),
        joint_names=tuple(joint_names),
        Y=Y,
        tau=tau,
        row_weight=w,
        theta_prior=prior,
        coord_scale=_coord_scale(names, joint_names, prior),
        M_full=M_full,
        phi_ols=phi_ols,
        R=R,
        Qt_tau=Q.T @ tw,
        extras=extras,
        policy=policy,
        theta_truth=truth,
        base_indices=base_full,
        removed_columns=removed,
    )


def _inputs_hash(p: ComparatorProblem) -> str:
    h = hashlib.sha256()
    for a in (p.Y, p.tau, p.row_weight, p.theta_prior, p.M_full, p.phi_ols):
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    h.update(json.dumps(list(p.params_std)).encode())
    h.update(repr(p.policy).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Record assembly (independent checks)
# ---------------------------------------------------------------------------


def _link_indices(p: ComparatorProblem) -> Dict[str, Dict[str, int]]:
    return _p10_indices_for_joints(list(p.params_std), list(p.joint_names))


def _feasibility(p: ComparatorProblem, theta: np.ndarray) -> Dict[str, Dict[str, Any]]:
    out = {}
    for j, idx in _link_indices(p).items():
        p10 = np.array([theta[idx[k]] for k in _P10_KEYS])
        rep = check_p10_feasibility(
            p10, mass_min=p.policy.mass_min, psd_eig_tol=p.policy.feas_tol
        )
        out[j] = {
            "mass": rep.mass,
            "min_eig": rep.min_eig,
            "ok": rep.status == "feasible",
        }
    return out


def _base_residual(p: ComparatorProblem, theta: np.ndarray) -> Dict[str, float]:
    res = p.M_full @ theta - p.phi_ols
    ab = float(np.linalg.norm(res))
    rel = ab / max(float(np.linalg.norm(p.phi_ols)), 1e-300)
    b = list(p.base_indices)
    Yb = p.row_weight[:, None] * p.Y[:, b]
    eff = float(np.sqrt(np.mean((Yb @ res) ** 2)))
    return {"abs": ab, "rel": rel, "effort_rms": eff}


def _per_joint_rmse(p: ComparatorProblem, theta: np.ndarray) -> Dict[str, float]:
    r = p.Y @ theta - p.tau
    nj = len(p.joint_names)
    if nj == 0 or r.size % nj:
        return {"all": float(np.sqrt(np.mean(r**2)))}
    r = r.reshape(nj, -1)
    return {
        j: float(np.sqrt(np.mean(r[k] ** 2))) for k, j in enumerate(p.joint_names)
    }


def _make_record(
    p: ComparatorProblem,
    objective: str,
    *,
    solver: str,
    raw: Dict[str, Any],
    exact: bool = False,
    notes: Optional[Dict[str, Any]] = None,
) -> SolveRecord:
    theta = raw.get("theta")
    status = raw["status"]
    if status != "optimal":
        theta = None
    base = {"abs": None, "rel": None, "effort_rms": None}
    rmse: Dict[str, float] = {}
    feas: Dict[str, Dict[str, Any]] = {}
    terr = None
    accepted = False
    if theta is not None:
        theta = np.asarray(theta, dtype=float).reshape(-1)
        base = _base_residual(p, theta)
        rmse = _per_joint_rmse(p, theta)
        feas = _feasibility(p, theta)
        accepted = bool(feas) and all(v["ok"] for v in feas.values())
        if exact:
            accepted = accepted and base["rel"] <= p.policy.base_rel_tol
        if p.theta_truth is not None:
            d = theta - p.theta_truth
            terr = {
                "abs": float(np.linalg.norm(d)),
                "rel": float(
                    np.linalg.norm(d) / max(np.linalg.norm(p.theta_truth), 1e-300)
                ),
            }
    cand = raw.get("fallback_candidate")
    return SolveRecord(
        objective=objective,
        solver=solver,
        solver_status=status,
        picos_status=raw.get("picos_status"),
        exception=raw.get("exception"),
        runtime_s=float(raw.get("runtime_s", 0.0)),
        objective_value=raw.get("objective_value"),
        theta=theta,
        fallback_used=False,
        fallback_candidate=None if cand is None else np.asarray(cand, dtype=float),
        base_residual=base,
        effort_rmse_fit_per_joint=rmse,
        feasibility=feas,
        accepted=accepted,
        truth_error=terr,
        inputs_hash=_inputs_hash(p),
        notes=notes or {},
    )


# ---------------------------------------------------------------------------
# SDP core
# ---------------------------------------------------------------------------


def _solve_core(
    p: ComparatorProblem,
    mode: str,
    *,
    solver: str = "cvxopt",
    norm: str = "cone",
    scaling: bool = True,
    variables: str = "full",
    bounds: bool = True,
    max_seconds: Optional[float] = None,
) -> Dict[str, Any]:
    """Build and solve one SDP. Never raises: errors become status 'error'.

    mode: ``exact`` | ``direct`` | ``relaxed`` | ``phase1``.
    """
    t0 = time.perf_counter()
    raw: Dict[str, Any] = {
        "status": "error",
        "picos_status": None,
        "exception": None,
        "objective_value": None,
        "theta": None,
    }
    try:
        import picos as pc

        n = len(p.params_std)
        D = np.diag(p.coord_scale) if scaling else np.eye(n)
        if variables == "params_r":
            # Removed columns stay at the prior (their LMI is then constant).
            keep = [i for i in range(n) if i not in p.removed_columns]
            D = D[:, keep]
        elif variables != "full":
            raise ValueError(f"unknown variables={variables!r}")
        nz = D.shape[1]
        prob = pc.Problem()
        z = pc.RealVariable("z", (nz, 1))
        th0 = pc.Constant("theta0", p.theta_prior.reshape(n, 1))
        theta = th0 + pc.Constant("D", D) * z

        s = None
        if mode == "phase1":
            s = pc.RealVariable("s", 1)
            prob.add_constraint(s <= 1)
        I4 = pc.Constant("I4", np.eye(4))
        for j, idx in _link_indices(p).items():
            m = theta[idx["m"]]
            if bounds:
                prob.add_constraint(m >= p.policy.mass_min)
                if p.policy.mass_ratio_bounds is not None:
                    lo, hi = p.policy.mass_ratio_bounds
                    m0 = float(p.theta_prior[idx["m"]])
                    prob.add_constraint(m >= lo * m0)
                    prob.add_constraint(m <= hi * m0)
            Ixx, Ixy, Ixz = theta[idx["Ixx"]], theta[idx["Ixy"]], theta[idx["Ixz"]]
            Iyy, Iyz, Izz = theta[idx["Iyy"]], theta[idx["Iyz"]], theta[idx["Izz"]]
            ht = 0.5 * (Ixx + Iyy + Izz)
            mx, my, mz = theta[idx["mx"]], theta[idx["my"]], theta[idx["mz"]]
            P = pc.block(
                [
                    [ht - Ixx, -Ixy, -Ixz, mx],
                    [-Ixy, ht - Iyy, -Iyz, my],
                    [-Ixz, -Iyz, ht - Izz, mz],
                    [mx, my, mz, m],
                ]
            )
            prob.add_constraint((P - s * I4 if s is not None else P) >> 0)

        Mz = pc.Constant("M", p.M_full) * theta - pc.Constant(
            "phi", p.phi_ols.reshape(-1, 1)
        )
        if mode in ("exact", "phase1"):
            prob.add_constraint(Mz == 0)
        if mode == "exact":
            if norm == "cone":
                prob.minimize = abs(z)
            elif norm == "schur":
                t = pc.RealVariable("t", 1)
                prob.add_constraint(
                    pc.block([[t, z.T], [z, pc.Constant("Inz", np.eye(nz))]]) >> 0
                )
                prob.minimize = t
            else:
                raise ValueError(f"unknown norm={norm!r}")
        elif mode == "direct":
            res = pc.Constant("R", p.R) * theta - pc.Constant(
                "Qt", p.Qt_tau.reshape(-1, 1)
            )
            lam = np.sqrt(max(p.policy.prior_weight, 0.0))
            prob.minimize = abs(res) ** 2 + (lam**2) * (abs(z) ** 2)
        elif mode == "relaxed":
            Rb = pc.Constant("Rb", p.R[:, list(p.base_indices)])
            lam = np.sqrt(max(p.policy.prior_weight, 0.0))
            prob.minimize = abs(Rb * Mz) ** 2 + (lam**2) * (abs(z) ** 2)
        elif mode == "phase1":
            prob.maximize = s
        else:
            raise ValueError(f"unknown mode={mode!r}")

        kw: Dict[str, Any] = {"solver": solver, "verbosity": 0}
        if max_seconds is not None:
            kw["max_seconds"] = float(max_seconds)
        prob.solve(**kw)
        raw["picos_status"] = str(prob.status)
        raw["status"] = "optimal" if prob.status == "optimal" else str(prob.status)
        if z.value is not None and prob.status == "optimal":
            zv = np.asarray(z.value, dtype=float).reshape(nz)
            raw["theta"] = p.theta_prior + D @ zv
            raw["objective_value"] = float(prob.value)
            if s is not None:
                raw["s_star"] = float(s.value)
    except Exception as exc:  # no silent fallback: report as error
        raw["status"] = "error"
        raw["exception"] = f"{type(exc).__name__}: {exc}"
        raw["theta"] = None
    raw["runtime_s"] = time.perf_counter() - t0
    return raw


# ---------------------------------------------------------------------------
# The three objectives
# ---------------------------------------------------------------------------


def solve_exact_reconstruction(
    p: ComparatorProblem, *, solver: str = "cvxopt", **opts: Any
) -> SolveRecord:
    """Objective 1: min ||D(theta-theta0)|| s.t. M_full theta = phi_ols, LMIs."""
    raw = _solve_core(p, "exact", solver=solver, **opts)
    return _make_record(p, "exact_reconstruction", solver=solver, raw=raw, exact=True)


def solve_direct_effort_fit(
    p: ComparatorProblem, *, solver: str = "cvxopt", **opts: Any
) -> SolveRecord:
    """Objective 2: regularised effort fit with the same LMIs, no equality."""
    raw = _solve_core(p, "direct", solver=solver, **opts)
    return _make_record(p, "direct_effort_fit", solver=solver, raw=raw)


def solve_per_link_projection(
    p: ComparatorProblem, *, solver: str = "cvxopt"
) -> SolveRecord:
    """Objective 3: nullspace representative, then per-link LMI projection."""
    t0 = time.perf_counter()
    raw: Dict[str, Any] = {
        "status": "error",
        "picos_status": None,
        "exception": None,
        "objective_value": None,
        "theta": None,
    }
    try:
        D = p.coord_scale
        A = p.M_full * D[None, :]
        z = np.linalg.pinv(A) @ (p.phi_ols - p.M_full @ p.theta_prior)
        theta = p.theta_prior + D * z
        total = 0.0
        for j, idx in _link_indices(p).items():
            cols = [idx[k] for k in _P10_KEYS]
            p10, rep = project_p10_lmi(
                theta[cols],
                mass_min=p.policy.mass_min,
                psd_eig_tol=p.policy.feas_tol,
                solver=solver,
                mass_bounds=_mass_bounds(p, idx),
            )
            if rep.status == "error":
                raise RuntimeError(f"link {j}: {rep.message}")
            theta[cols] = p10
            total += float(rep.objective or 0.0)
        raw.update(status="optimal", picos_status="optimal", theta=theta)
        raw["objective_value"] = total
    except Exception as exc:
        raw["exception"] = f"{type(exc).__name__}: {exc}"
        raw["theta"] = None
    raw["runtime_s"] = time.perf_counter() - t0
    return _make_record(p, "per_link_projection", solver=solver, raw=raw)


def _mass_bounds(p: ComparatorProblem, idx) -> Optional[Tuple[float, float]]:
    if p.policy.mass_ratio_bounds is None:
        return None
    m0 = float(p.theta_prior[idx["m"]])
    lo, hi = p.policy.mass_ratio_bounds
    return (lo * m0, hi * m0)


# ---------------------------------------------------------------------------
# Diagnostics D0..D3
# ---------------------------------------------------------------------------


def diagnose_exact(
    p: ComparatorProblem, *, solver: str = "cvxopt"
) -> List[SolveRecord]:
    """Staged diagnostics for objective 1 (D0 reproduce, D1 phase-I,
    D2 one-change variants, D3 relaxed equality)."""
    records: List[SolveRecord] = []

    # D0: reproduce with the production reconstruction entry point.
    t0 = time.perf_counter()
    raw: Dict[str, Any] = {
        "status": "error",
        "picos_status": None,
        "exception": None,
        "theta": None,
    }
    try:
        res = reconstruct_full_parameters(
            (p.M_full, p.phi_ols, list(p.params_std)),
            method="sdp",
            theta0=p.theta_prior,
            weights=1.0 / p.coord_scale,
            joint_names=list(p.joint_names),
            mass_min=p.policy.mass_min,
            solver=solver,
        )
        raw["objective_value"] = res.objective
        if res.status == "ok":
            raw.update(status="optimal", theta=res.theta_r)
        else:
            # Legacy code silently fell back; keep it only as a candidate.
            raw["status"] = "error"
            raw["exception"] = f"reconstruct_full_parameters status={res.status}"
            raw["fallback_candidate"] = res.theta_r
    except Exception as exc:
        raw["exception"] = f"{type(exc).__name__}: {exc}"
    raw["runtime_s"] = time.perf_counter() - t0
    records.append(
        _make_record(
            p, "D0:reconstruct_full_parameters_sdp", solver=solver, raw=raw, exact=True
        )
    )

    # D1: phase-I feasibility.
    raw = _solve_core(p, "phase1", solver=solver, bounds=False)
    s_star = raw.get("s_star")
    notes: Dict[str, Any] = {"s_star": s_star}
    if raw["status"] == "optimal" and s_star is not None:
        notes["label"] = "infeasible (certificate s*<0)" if s_star < 0 else "feasible"
        if s_star < 0:
            raw["theta"] = None  # a strictly infeasible point is not a solution
    else:
        notes["label"] = "solver failure"
    rec = _make_record(p, "D1:phase1", solver=solver, raw=raw, exact=True, notes=notes)
    rec.objective_value = s_star
    rec.accepted = False
    records.append(rec)

    # D2: one change at a time.
    variants = {
        "schur_norm": {"norm": "schur"},
        "scaling_off": {"scaling": False},
        "params_r_only": {"variables": "params_r"},
        "bounds_off": {"bounds": False},
    }
    for name, opts in variants.items():
        raw = _solve_core(p, "exact", solver=solver, **opts)
        records.append(
            _make_record(p, f"D2:{name}", solver=solver, raw=raw, exact=True)
        )

    # D3: relaxed equality.
    raw = _solve_core(p, "relaxed", solver=solver)
    records.append(_make_record(p, "D3:relaxed_equality", solver=solver, raw=raw))
    return records
