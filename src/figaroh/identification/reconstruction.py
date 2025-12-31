"""Reconstruction utilities for base-parameter identification.

This module provides helpers to reconstruct a parameter vector ``theta_r``
(ordered as ``params_r``) from identified base parameters.

Given a base mapping matrix ``M`` and a base vector ``phi_base``:

    phi_base = M @ theta_r

If ``M`` is rank-deficient (typical when ``theta_r`` contains unobservable
parameters), there are infinitely many solutions. We pick one by projecting a
prior ``theta0`` onto the affine constraint set using a weighted least-squares
metric.

The default reconstruction solves:

    minimize    ||W^{1/2} (theta - theta0)||_2^2
    subject to  M theta = phi_base

where W is diagonal (weights).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ReconstructionResult:
    """Result of reconstructing ``theta_r`` from base parameters."""

    theta_r: np.ndarray
    params_r: List[str]
    residual: np.ndarray

    def as_dict(self) -> Dict[str, float]:
        return {
            name: float(self.theta_r[i])
            for i, name in enumerate(self.params_r)
        }


def prior_vector_from_dict(
    params_r: Sequence[str],
    params_std: Optional[Dict[str, float]],
    *,
    default: float = 0.0,
) -> np.ndarray:
    """Build a prior vector aligned with ``params_r`` from a dict.

    Missing entries fall back to ``default``.
    """

    if params_std is None:
        return np.full(len(params_r), float(default))

    return np.array(
        [float(params_std.get(name, default)) for name in params_r]
    )


def reconstruct_theta_r(
    M: np.ndarray,
    phi_base: np.ndarray,
    *,
    theta0: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    rcond: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reconstruct ``theta_r`` from the base constraint ``M theta = phi_base``.

    Args:
        M: Base mapping matrix with shape (r, n).
        phi_base: Base vector with shape (r,).
        theta0: Prior vector with shape (n,). Defaults to zeros.
        weights: Optional diagonal weights with shape (n,). Larger values
            keep ``theta`` closer to the prior in that coordinate.
        rcond: Cutoff for least-squares fallback.

    Returns:
        (theta, residual) where residual is ``M @ theta - phi_base``.

    Raises:
        ValueError: if shapes are incompatible.
    """

    M = np.asarray(M, dtype=float)
    phi_base = np.asarray(phi_base, dtype=float).reshape(-1)

    if M.ndim != 2:
        raise ValueError("M must be 2D")

    r, n = M.shape
    if phi_base.shape != (r,):
        raise ValueError(
            f"phi_base must have shape ({r},), got {phi_base.shape}"
        )

    if theta0 is None:
        theta0 = np.zeros(n, dtype=float)
    else:
        theta0 = np.asarray(theta0, dtype=float).reshape(-1)
        if theta0.shape != (n,):
            raise ValueError(
                f"theta0 must have shape ({n},), got {theta0.shape}"
            )

    if weights is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape != (n,):
            raise ValueError(f"weights must have shape ({n},), got {w.shape}")
        if np.any(w <= 0):
            raise ValueError("weights must be strictly positive")

    # Weighted projection of theta0 onto {theta | M theta = phi_base}.
    # W = diag(w), so W^{-1} = diag(1/w).
    w_inv = 1.0 / w

    # A = M W^{-1} M^T
    MW_inv = M * w_inv[np.newaxis, :]
    A = MW_inv @ M.T
    rhs = M @ theta0 - phi_base

    try:
        lam = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        lam, *_ = np.linalg.lstsq(A, rhs, rcond=rcond)

    theta = theta0 - (w_inv * (M.T @ lam))
    residual = M @ theta - phi_base
    return theta, residual


def reconstruct_from_base(
    M: np.ndarray,
    phi_base: np.ndarray,
    params_r: Sequence[str],
    *,
    params_std_prior: Optional[Dict[str, float]] = None,
    default_prior: float = 0.0,
    weights: Optional[np.ndarray] = None,
) -> ReconstructionResult:
    """Convenience wrapper that returns a labeled result.

    Args:
        M: Base mapping matrix, shape (r, n).
        phi_base: Base vector, shape (r,).
        params_r: Names of the remaining parameters, length n.
        params_std_prior: Optional dict used as prior for ``theta0``.
        default_prior: Default value for missing prior entries.
        weights: Optional diagonal weights, shape (n,).

    Returns:
        ReconstructionResult with ``theta_r`` aligned to ``params_r``.
    """

    theta0 = prior_vector_from_dict(
        params_r,
        params_std_prior,
        default=default_prior,
    )
    theta, residual = reconstruct_theta_r(
        M,
        phi_base,
        theta0=theta0,
        weights=weights,
    )
    return ReconstructionResult(
        theta_r=theta,
        params_r=list(params_r),
        residual=residual,
    )


def run_reconstruction(
    M: np.ndarray,
    phi_base: np.ndarray,
    params_r: Sequence[str],
    *,
    params_std_prior: Optional[Dict[str, float]] = None,
    recon_cfg: Optional[Mapping[str, Any]] = None,
    joint_names: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Any], Dict[str, float], bool]:
    """Run reconstruction Option A and optionally alternate physical consistency.

    This mirrors the pipeline behavior used in `BaseIdentification`, but is
    callable as a standalone utility.

    Returns:
        (reconstruction_payload, full_parameter_dict, pc_already_applied)
    """

    cfg: Mapping[str, Any] = recon_cfg or {}

    mode = str(cfg.get("mode", "A")).strip().upper()
    if mode not in {"A"}:
        raise ValueError(
            f"Unsupported reconstruction mode={mode!r}. Only 'A' is implemented."
        )

    default_prior = float(cfg.get("default_prior", 0.0))
    weights = cfg.get("weights", None)

    recon = reconstruct_from_base(
        M,
        phi_base,
        params_r,
        params_std_prior=params_std_prior,
        default_prior=default_prior,
        weights=weights,
    )

    full_param_dict: Dict[str, float] = {}
    if isinstance(params_std_prior, dict):
        full_param_dict.update(params_std_prior)
    full_param_dict.update(recon.as_dict())

    alt_enabled = bool(cfg.get("alternate_physical_consistency", False))
    max_iters = int(cfg.get("max_iters", 1))
    if max_iters < 1:
        max_iters = 1

    pc_alt_reports = []
    final_feas = None
    pc_already_applied = False

    if alt_enabled and max_iters > 1:
        if not joint_names:
            raise ValueError(
                "joint_names must be provided when alternation is enabled"
            )

        pc_cfg = cfg.get("physical_consistency", {})
        if not isinstance(pc_cfg, dict):
            pc_cfg = {}

        mass_min = float(pc_cfg.get("mass_min", 1e-6))
        psd_eig_tol = float(pc_cfg.get("psd_eig_tol", -1e-10))
        solver = str(pc_cfg.get("solver", "cvxopt"))
        verbose = bool(pc_cfg.get("verbose", False))
        max_seconds = pc_cfg.get("max_seconds", None)
        if max_seconds is not None:
            max_seconds = float(max_seconds)

        from figaroh.identification.physical_consistency import (
            check_p10_feasibility,
            param_dict_with_p10_by_joint,
            p10_by_joint_from_param_dict,
            project_robot_p10_lmi,
        )

        for _ in range(max_iters - 1):
            p10_by_joint = p10_by_joint_from_param_dict(
                parameter_dict=full_param_dict,
                joint_names=list(joint_names),
            )
            proj_report = project_robot_p10_lmi(
                p10_by_joint,
                mass_min=mass_min,
                psd_eig_tol=psd_eig_tol,
                solver=solver,
                verbose=verbose,
                max_seconds=max_seconds,
            )
            pc_alt_reports.append(proj_report)

            full_param_dict = param_dict_with_p10_by_joint(
                parameter_dict=full_param_dict,
                p10_by_joint=proj_report.p10_by_joint,
            )

            theta0 = np.array(
                [full_param_dict.get(k, default_prior) for k in params_r],
                dtype=float,
            )
            theta_r, residual = reconstruct_theta_r(
                M,
                phi_base,
                theta0=theta0,
                weights=weights,
            )
            recon = ReconstructionResult(
                theta_r=theta_r,
                params_r=list(params_r),
                residual=residual,
            )
            full_param_dict.update(recon.as_dict())

        try:
            final_p10 = p10_by_joint_from_param_dict(
                parameter_dict=full_param_dict,
                joint_names=list(joint_names),
            )
            final_feas = {
                j: check_p10_feasibility(
                    p10,
                    mass_min=mass_min,
                    psd_eig_tol=psd_eig_tol,
                )
                for j, p10 in final_p10.items()
            }
        except Exception:
            final_feas = None

        pc_already_applied = True

    payload: Dict[str, Any] = {
        "mode": mode,
        "params_r": recon.params_r,
        "theta_r": recon.theta_r,
        "theta_r_dict": recon.as_dict(),
        "residual": recon.residual,
        "parameter_dict": full_param_dict,
        "alternate_physical_consistency": alt_enabled,
        "alternation_iters": max_iters,
        "final_feasibility": final_feas,
        "pc_reports": pc_alt_reports,
    }

    return payload, full_param_dict, pc_already_applied
