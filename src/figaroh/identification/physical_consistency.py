"""Physical-consistency utilities for inertial parameters.

v0.4.1 scope: project per-link inertial parameters onto the physically
consistent set via an SDP/LMI using Pinocchio's pseudo-inertia structure.

This module is intentionally opt-in and does not change identification
defaults.

Pinocchio reference (v3.7.x): PseudoInertia / LogCholeskyParameters.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclasses.dataclass(frozen=True)
class ProjectionReport:
    status: str
    mass: float
    min_eig: float
    objective: Optional[float] = None
    solver: Optional[str] = None
    message: Optional[str] = None
    runtime: Optional[float] = None


@dataclasses.dataclass(frozen=True)
class RobotProjectionReport:
    status: str
    projected_links: int
    failed_links: int
    per_link: Dict[str, ProjectionReport]


def _sigma_from_p10(p10: np.ndarray) -> np.ndarray:
    # p10 order (Pinocchio dynamic parameters):
    # [m, mcx, mcy, mcz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]
    Ixx, Ixy, Iyy = p10[4], p10[5], p10[6]
    Ixz, Iyz, Izz = p10[7], p10[8], p10[9]
    return np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]], dtype=float)


def pseudo_inertia_matrix_from_p10(p10: np.ndarray) -> np.ndarray:
    """Build the 4x4 pseudo-inertia matrix P from Pinocchio dynamic params.

    P = [[sigma, h], [h^T, m]]

    where sigma is the 3x3 matrix block and h = m*c is the first moment.
    """
    p10 = np.asarray(p10, dtype=float).reshape(10)

    # Prefer Pinocchio's canonical conversion when available. This helps
    # avoid convention mismatches and matches Pinocchio's own definition.
    try:
        import pinocchio as pin  # type: ignore

        P = pin.PseudoInertia.FromDynamicParameters(p10).toMatrix()
        return np.asarray(P, dtype=float)
    except Exception as e:
        logger.debug(
            "Pinocchio pseudo-inertia conversion unavailable; "
            "falling back to explicit construction (%s)",
            e,
        )

    # Fallback: explicit block construction P = [[sigma, h], [h^T, m]].
    m = float(p10[0])
    h = np.asarray(p10[1:4], dtype=float).reshape(3)
    sigma = _sigma_from_p10(p10)

    P = np.zeros((4, 4), dtype=float)
    P[:3, :3] = sigma
    P[:3, 3] = h
    P[3, :3] = h
    P[3, 3] = m
    return P


def check_p10_feasibility(
    p10: np.ndarray,
    *,
    mass_min: float = 0.0,
    psd_eig_tol: float = -1e-10,
) -> ProjectionReport:
    p10 = np.asarray(p10, dtype=float).reshape(10)
    m = float(p10[0])
    if m < mass_min:
        return ProjectionReport(
            status="infeasible",
            mass=m,
            min_eig=float("-inf"),
            message=f"mass {m} < mass_min {mass_min}",
        )

    P = pseudo_inertia_matrix_from_p10(p10)
    # Symmetrize for numerical robustness
    P = 0.5 * (P + P.T)
    min_eig = float(np.linalg.eigvalsh(P).min())
    status = "feasible" if min_eig >= psd_eig_tol else "infeasible"
    return ProjectionReport(status=status, mass=m, min_eig=min_eig)


def _auto_weights(p10_hat: np.ndarray) -> np.ndarray:
    p10_hat = np.asarray(p10_hat, dtype=float).reshape(10)

    m_scale = max(abs(float(p10_hat[0])), 1e-6)
    h_scale = max(float(np.linalg.norm(p10_hat[1:4])), 1e-6)
    sigma_scale = max(float(np.linalg.norm(p10_hat[4:10])), 1e-6)

    w = np.ones(10, dtype=float)
    w[0] = 1.0 / m_scale
    w[1:4] = 1.0 / h_scale
    w[4:10] = 1.0 / sigma_scale
    return w


def project_p10_lmi(
    p10_hat: np.ndarray,
    *,
    mass_min: float = 1e-6,
    psd_eig_tol: float = -1e-10,
    weights: Optional[np.ndarray] = None,
    solver: str = "cvxopt",
    verbose: bool = False,
    max_seconds: Optional[float] = None,
    mass_bounds: Optional[Tuple[float, float]] = None,
    com_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
) -> Tuple[np.ndarray, ProjectionReport]:
    """Project a 10D inertial parameter vector.

    Projects onto the physically feasible set.

    Solves a convex SDP:
      min ||W (p - p_hat)||^2
      s.t. m >= mass_min
           P(p) >= 0  (4x4 pseudo-inertia PSD)

    Returns projected p10 and a report.
    """

    p10_hat = np.asarray(p10_hat, dtype=float).reshape(10)
    if weights is None:
        w = _auto_weights(p10_hat)
    else:
        w = np.asarray(weights, dtype=float).reshape(10)

    try:
        import picos as pc
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "Physical-consistency projection requires 'picos'. "
            "Install it (and a solver backend like cvxopt/mosek) to enable."
        ) from e

    problem = pc.Problem()

    m = pc.RealVariable("m", 1)
    h = pc.RealVariable("h", 3)  # column vector (3x1)
    sigma = pc.SymmetricVariable("sigma", 3)

    # 4x4 pseudo-inertia matrix P = [[sigma, h], [h^T, m]]
    P = pc.block([[sigma, h], [h.T, m]])

    # Constraints
    problem.add_constraint(m >= mass_min)
    problem.add_constraint(P >> 0)

    # Optional CAD-informed per-link box constraints
    if mass_bounds is not None:
        problem.add_constraint(m >= mass_bounds[0])
        problem.add_constraint(m <= mass_bounds[1])
    if com_bounds:
        _axis_idx = {"x": 0, "y": 1, "z": 2}
        for _axis, (_h_lo, _h_hi) in com_bounds.items():
            _k = _axis_idx.get(_axis)
            if _k is not None:
                problem.add_constraint(h[_k] >= _h_lo)
                problem.add_constraint(h[_k] <= _h_hi)

    # Objective: minimize ||W(p - p_hat)||^2  (element-wise, picos 2.x API)
    # Mass term (1-element)
    h_hat_c = pc.Constant("h_hat", p10_hat[1:4].reshape(3, 1))
    dm = float(w[0]) * (m - float(p10_hat[0]))
    dh = w[1:4].reshape(3, 1) * (h - h_hat_c)
    obj = pc.SquaredNorm(dm) + pc.SquaredNorm(dh)
    # Sigma terms: iterate over 6 unique upper-triangle entries to avoid
    # Frobenius double-counting.  p10 order: Ixx=4, Ixy=5, Iyy=6, Ixz=7, Iyz=8, Izz=9
    for _r, _c, _idx in [
        (0, 0, 4),
        (0, 1, 5),
        (1, 1, 6),
        (0, 2, 7),
        (1, 2, 8),
        (2, 2, 9),
    ]:
        obj = obj + (float(w[_idx]) * (sigma[_r, _c] - float(p10_hat[_idx]))) ** 2
    problem.minimize = obj

    # Solve — use verbosity (picos 2.x) instead of the deprecated verbose kwarg
    solve_kwargs: Dict[str, Any] = {
        "solver": solver,
        "verbosity": int(verbose),
    }
    if max_seconds is not None:
        solve_kwargs["max_seconds"] = float(max_seconds)

    t_start = time.perf_counter()
    try:
        problem.solve(**solve_kwargs)
    except Exception as e:
        report = ProjectionReport(
            status="error",
            mass=float(p10_hat[0]),
            min_eig=float("nan"),
            solver=solver,
            message=str(e),
            runtime=time.perf_counter() - t_start,
        )
        return p10_hat.copy(), report
    t_elapsed = time.perf_counter() - t_start

    # Extract solution
    m_val = float(m.value)
    h_val = np.asarray(h.value).reshape(3)
    sigma_val = np.asarray(sigma.value).reshape(3, 3)

    p10_proj = np.zeros(10, dtype=float)
    p10_proj[0] = m_val
    p10_proj[1:4] = h_val
    p10_proj[4] = sigma_val[0, 0]
    p10_proj[5] = sigma_val[0, 1]
    p10_proj[6] = sigma_val[1, 1]
    p10_proj[7] = sigma_val[0, 2]
    p10_proj[8] = sigma_val[1, 2]
    p10_proj[9] = sigma_val[2, 2]

    # Verify with a relaxed tolerance to absorb solver numerical noise
    # (SDP solvers satisfy P >> 0 up to ~1e-8 relative error).
    _verify_tol = min(psd_eig_tol, -1e-8)
    feas = check_p10_feasibility(p10_proj, mass_min=mass_min, psd_eig_tol=_verify_tol)
    report = dataclasses.replace(
        feas,
        status="projected" if feas.status == "feasible" else "infeasible",
        solver=solver,
        objective=float(problem.value) if problem.value is not None else None,
        runtime=t_elapsed,
    )
    return p10_proj, report


def project_robot_p10_lmi(
    p10_by_link: Mapping[str, np.ndarray],
    *,
    mass_min: float = 1e-6,
    psd_eig_tol: float = -1e-10,
    weights: Optional[np.ndarray] = None,
    solver: str = "cvxopt",
    verbose: bool = False,
    max_seconds: Optional[float] = None,
    cad_constraints: Optional[Any] = None,
) -> Tuple[Dict[str, np.ndarray], RobotProjectionReport]:
    projected: Dict[str, np.ndarray] = {}
    reports: Dict[str, ProjectionReport] = {}

    failed = 0
    for link, p10 in p10_by_link.items():
        _mb = cad_constraints.mass_bounds.get(link) if cad_constraints else None
        _cb = cad_constraints.com_bounds.get(link, {}) if cad_constraints else {}
        p_proj, rep = project_p10_lmi(
            p10,
            mass_min=mass_min,
            psd_eig_tol=psd_eig_tol,
            weights=weights,
            solver=solver,
            verbose=verbose,
            max_seconds=max_seconds,
            mass_bounds=_mb,
            com_bounds=_cb or None,
        )
        projected[link] = p_proj
        reports[link] = rep
        if rep.status in {"error", "infeasible"}:
            failed += 1

    status = "ok" if failed == 0 else "partial"
    return projected, RobotProjectionReport(
        status=status,
        projected_links=len(projected),
        failed_links=failed,
        per_link=reports,
    )


def p10_by_joint_from_param_dict(
    *,
    parameter_dict: Mapping[str, float],
    joint_names: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Build per-joint Pinocchio 10D inertial vectors from a flat param dict.

    Expects keys like `m_<joint>`, `mx_<joint>`, ..., `Izz_<joint>`.
    """

    inertial_keys = [
        "m",
        "mx",
        "my",
        "mz",
        "Ixx",
        "Ixy",
        "Iyy",
        "Ixz",
        "Iyz",
        "Izz",
    ]
    out: Dict[str, np.ndarray] = {}

    for j in joint_names:
        values = []
        for k in inertial_keys:
            name = f"{k}_{j}"
            if name not in parameter_dict:
                raise KeyError(f"Missing inertial parameter '{name}'")
            values.append(float(parameter_dict[name]))
        out[j] = np.asarray(values, dtype=float)

    return out


def param_dict_with_p10_by_joint(
    *,
    parameter_dict: Dict[str, float],
    p10_by_joint: Mapping[str, np.ndarray],
) -> Dict[str, float]:
    """Return a copy of parameter_dict.

    The returned dict has inertials overwritten by the projected p10 vectors.
    """

    inertial_keys = [
        "m",
        "mx",
        "my",
        "mz",
        "Ixx",
        "Ixy",
        "Iyy",
        "Ixz",
        "Iyz",
        "Izz",
    ]
    updated = dict(parameter_dict)

    for joint, p10 in p10_by_joint.items():
        p10 = np.asarray(p10, dtype=float).reshape(10)
        for idx, key in enumerate(inertial_keys):
            updated[f"{key}_{joint}"] = float(p10[idx])

    return updated


# ---------------------------------------------------------------------------
# Public API aliases (roadmap spec names)
# ---------------------------------------------------------------------------


def is_feasible_link(
    p10: np.ndarray,
    *,
    mass_min: float = 0.0,
    psd_eig_tol: float = -1e-10,
) -> ProjectionReport:
    """Check physical feasibility of a single link's inertial parameters.

    Alias for :func:`check_p10_feasibility` using the roadmap-specified name.
    """
    return check_p10_feasibility(p10, mass_min=mass_min, psd_eig_tol=psd_eig_tol)


def project_link(
    p10: np.ndarray,
    *,
    mass_min: float = 1e-6,
    psd_eig_tol: float = -1e-10,
    weights: Optional[np.ndarray] = None,
    solver: str = "cvxopt",
    verbose: bool = False,
    max_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, ProjectionReport]:
    """Project a single link's inertial parameters onto the feasible set.

    Alias for :func:`project_p10_lmi` using the roadmap-specified name.
    """
    return project_p10_lmi(
        p10,
        mass_min=mass_min,
        psd_eig_tol=psd_eig_tol,
        weights=weights,
        solver=solver,
        verbose=verbose,
        max_seconds=max_seconds,
    )
