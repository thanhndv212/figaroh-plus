"""Reconstruction utilities for base-parameter identification.

This module provides helpers to reconstruct a parameter vector ``theta_r``
(ordered as ``params_r``) from identified base parameters.

Given a base mapping matrix ``M`` and a base vector ``phi_base``:

    phi_base = M @ theta_r

If ``M`` is rank-deficient (typical when ``theta_r`` contains unobservable
parameters), there are infinitely many solutions. We pick one by projecting a
prior ``theta0`` onto the affine constraint set using a weighted least-squares
metric.

The default reconstruction (Option A / "nullspace") solves:

    minimize    ||W^{1/2} (theta - theta0)||_2^2
    subject to  M theta = phi_base

where W is diagonal (weights).

Option B ("sdp") solves the same problem but adds physical-consistency LMI
constraints (P_j ≽ 0, m_j ≥ mass_min per joint), requiring picos + a
compatible SDP solver backend (cvxopt by default).

Public API
----------
reconstruct_theta_r        — core nullspace projection
reconstruct_from_base      — labeled convenience wrapper
run_reconstruction         — pipeline entry point (mode A + alternation loop)
run_option_a_reconstruction — alias for run_reconstruction (backward compat)
reconstruct_full_parameters — unified entry point (nullspace | sdp | auto)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Inertial parameter keys in FIGAROH order
_INERTIAL_KEYS = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]


@dataclass(frozen=True)
class ReconstructionResult:
    """Result of reconstructing ``theta_r`` from base parameters.

    Attributes:
        theta_r: Reconstructed parameter vector, shape (n,).
        params_r: Parameter names aligned to ``theta_r``, length n.
        residual: Base constraint residual ``M @ theta_r - phi_base``, shape (r,).
        status: Outcome code — ``"ok"``, ``"infeasible"``, ``"solver_missing"``,
            ``"error"``.
        base_residual_norm: L2 norm of the residual (scalar); always set when
            the solve completes.
        objective: Objective value from the SDP solver (Option B only);
            ``None`` for nullspace reconstruction.
    """

    theta_r: np.ndarray
    params_r: List[str]
    residual: np.ndarray
    status: str = "ok"
    base_residual_norm: Optional[float] = None
    objective: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        """Return {param_name: value} mapping aligned to ``params_r``."""
        return {
            name: float(self.theta_r[i])
            for i, name in enumerate(self.params_r)
        }


@dataclass(frozen=True)
class BaseResult:
    """Structured representation of QR base-parameter output.

    Wraps the mapping matrix *M*, identified base vector *phi_base*, and
    parameter list *params_r* so that callers of
    :func:`reconstruct_full_parameters` do not need to assemble them manually.

    Attributes:
        M: Base mapping matrix satisfying ``phi_base = M @ theta_r``,
            shape (r, n).
        phi_base: Identified base-parameter values, shape (r,).
        params_r: Names of the remaining parameters (columns of *M*), length n.
        phi_base_dict: Optional dict mapping base-expression strings to their
            identified values.
    """

    M: np.ndarray
    phi_base: np.ndarray
    params_r: List[str]
    phi_base_dict: Optional[Dict[str, float]] = None


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
        status="ok",
        base_residual_norm=float(np.linalg.norm(residual)),
    )


# ---------------------------------------------------------------------------
# Prior source helpers
# ---------------------------------------------------------------------------


def _load_prior_from_urdf(
    model: Any,
    params_r: Sequence[str],
    *,
    default: float = 0.0,
) -> Dict[str, float]:
    """Load a flat parameter prior from Pinocchio model inertias.

    Args:
        model: A ``pinocchio.Model`` instance.
        params_r: Parameter names in the FIGAROH convention ``"<key>_<joint>"``.
        default: Fallback for any parameter not found in the model.

    Returns:
        Dict mapping parameter names to float values.

    Raises:
        ImportError: If pinocchio is not installed.
    """
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Loading priors from URDF requires pinocchio. "
            "Install it with: conda install -c conda-forge pinocchio"
        ) from exc

    # Build a name→joint-index map for model joints (skip 'universe' at index 0)
    name_to_idx = {str(model.names[i]): i for i in range(1, model.njoints)}

    prior: Dict[str, float] = {}
    for p_name in params_r:
        found = False
        for key in _INERTIAL_KEYS:
            prefix = key + "_"
            if p_name.startswith(prefix):
                joint_name = p_name[len(prefix):]
                idx = name_to_idx.get(joint_name)
                if idx is not None:
                    inertia = model.inertias[idx]
                    cx, cy, cz = float(inertia.lever[0]), float(inertia.lever[1]), float(inertia.lever[2])
                    m_val = float(inertia.mass)
                    I = inertia.inertia  # 3×3
                    flat = {
                        f"m_{joint_name}": m_val,
                        f"mx_{joint_name}": m_val * cx,
                        f"my_{joint_name}": m_val * cy,
                        f"mz_{joint_name}": m_val * cz,
                        f"Ixx_{joint_name}": float(I[0, 0]),
                        f"Ixy_{joint_name}": float(I[0, 1]),
                        f"Iyy_{joint_name}": float(I[1, 1]),
                        f"Ixz_{joint_name}": float(I[0, 2]),
                        f"Iyz_{joint_name}": float(I[1, 2]),
                        f"Izz_{joint_name}": float(I[2, 2]),
                    }
                    prior[p_name] = flat.get(p_name, default)
                    found = True
                break
        if not found:
            prior[p_name] = default
    return prior


def _load_prior_from_yaml(
    yaml_path: str,
    params_r: Sequence[str],
    *,
    default: float = 0.0,
) -> Dict[str, float]:
    """Load a flat parameter prior from a YAML file.

    The YAML file must map parameter names (strings) to numeric values at the
    top level, e.g.::

        m_joint1: 2.5
        mx_joint1: 0.0
        ...

    Args:
        yaml_path: Path to the YAML file.
        params_r: Parameter names to extract.
        default: Fallback for parameters not in the file.

    Returns:
        Dict mapping parameter names to float values.
    """
    import yaml

    with open(yaml_path, "r") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(
            f"Prior YAML file must contain a top-level mapping, got {type(raw)}"
        )
    flat = {str(k): float(v) for k, v in raw.items() if isinstance(v, (int, float))}
    return {p: flat.get(p, default) for p in params_r}


# ---------------------------------------------------------------------------
# Option B — SDP reconstruction
# ---------------------------------------------------------------------------


def _p10_indices_for_joints(
    params_r: Sequence[str],
    joint_names: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    """Return {joint_name: {inertial_key: index_into_params_r}}.

    Only joints that have *all* 10 inertial keys in ``params_r`` are included.
    """
    name_list = list(params_r)
    result: Dict[str, Dict[str, int]] = {}
    for j in joint_names:
        indices: Dict[str, int] = {}
        for key in _INERTIAL_KEYS:
            p = f"{key}_{j}"
            try:
                indices[key] = name_list.index(p)
            except ValueError:
                break  # joint incomplete → skip
        else:
            result[j] = indices
    return result


def _reconstruct_sdp(
    M: np.ndarray,
    phi_base: np.ndarray,
    params_r: Sequence[str],
    *,
    theta0: np.ndarray,
    w: np.ndarray,
    joint_names: Sequence[str],
    mass_min: float = 1e-6,
    psd_eig_tol: float = -1e-10,
    solver: str = "cvxopt",
    max_seconds: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Project theta onto {M theta = phi_base} with per-joint LMI constraints.

    Solves (pure SDP — Schur-complement epigraph)::

        minimise   t
        subject to [[t, d^T], [d, I_n]] ≽ 0   (⟺  t ≥ ‖d‖²)
                   M theta = phi_base
                   P_j(theta) ≽ 0   ∀ joint j with full p10 in params_r
                   theta[m_j] ≥ mass_min   ∀ such joint j

    where d = diag(w) (theta − theta0).

    Requires picos (lazy import).

    Returns:
        (theta_r_array, objective_value)

    Raises:
        ImportError: if picos is not installed.
    """
    try:
        import picos as pc
    except Exception as exc:
        raise ImportError(
            "Option B (SDP) reconstruction requires 'picos'. "
            "Install it (and a solver backend like cvxopt/mosek) to use method='sdp'."
        ) from exc

    n = len(params_r)
    r = M.shape[0]

    theta = pc.RealVariable("theta", (n, 1))
    t_var = pc.RealVariable("t", 1)

    problem = pc.Problem()

    # Equality: M theta = phi_base
    M_c = pc.Constant("M", M)
    phi_c = pc.Constant("phi", phi_base.reshape(r, 1))
    problem.add_constraint(M_c * theta == phi_c)

    # Per-joint LMI and mass constraints
    joint_map = _p10_indices_for_joints(params_r, joint_names)
    for j_name, idx in joint_map.items():
        problem.add_constraint(theta[idx["m"]] >= mass_min)

        Ixx = theta[idx["Ixx"]]; Ixy = theta[idx["Ixy"]]; Ixz = theta[idx["Ixz"]]
        Iyy = theta[idx["Iyy"]]; Iyz = theta[idx["Iyz"]]; Izz = theta[idx["Izz"]]
        mx  = theta[idx["mx"]];  my  = theta[idx["my"]];  mz  = theta[idx["mz"]]
        m   = theta[idx["m"]]

        # 4×4 pseudo-inertia P_j
        P_j = pc.block([
            [Ixx, Ixy, Ixz, mx],
            [Ixy, Iyy, Iyz, my],
            [Ixz, Iyz, Izz, mz],
            [mx,  my,  mz,  m ],
        ])
        problem.add_constraint(P_j >> 0)

    # Quadratic objective via Schur complement:
    # minimise t s.t. [[t, diff^T], [diff, I_n]] >> 0
    # where diff = diag(w) (theta - theta0)
    theta0_c = pc.Constant("theta0", theta0.reshape(n, 1))
    W_c = pc.Constant("W", np.diag(w))
    diff = W_c * (theta - theta0_c)       # (n, 1)
    I_n  = pc.Constant("In", np.eye(n))
    schur = pc.block([[t_var, diff.T], [diff, I_n]])
    problem.add_constraint(schur >> 0)
    problem.minimize = t_var

    solve_kwargs: Dict[str, Any] = {"solver": solver, "verbosity": 0}
    if max_seconds is not None:
        solve_kwargs["max_seconds"] = float(max_seconds)

    problem.solve(**solve_kwargs)

    theta_r_val = np.asarray(theta.value, dtype=float).reshape(n)
    obj_val = float(t_var.value)
    return theta_r_val, obj_val


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------


def reconstruct_full_parameters(
    base_result: Union[
        "BaseResult",
        Tuple[np.ndarray, np.ndarray, Sequence[str]],
    ],
    *,
    method: str = "nullspace",
    theta0: Optional[np.ndarray] = None,
    params_std_prior: Optional[Dict[str, float]] = None,
    prior_source: str = "dict",
    prior_yaml_path: Optional[str] = None,
    model: Optional[Any] = None,
    weights: Optional[np.ndarray] = None,
    joint_names: Optional[Sequence[str]] = None,
    mass_min: float = 1e-6,
    psd_eig_tol: float = -1e-10,
    solver: str = "cvxopt",
    max_seconds: Optional[float] = None,
) -> ReconstructionResult:
    """Unified entry point for base → full parameter reconstruction.

    Args:
        base_result: Either a :class:`BaseResult` or a ``(M, phi_base, params_r)``
            tuple.
        method: ``"nullspace"`` (default), ``"sdp"`` (requires picos), or
            ``"auto"`` (try sdp; fall back silently to nullspace if picos is
            unavailable).
        theta0: Explicit prior vector (n,). Overrides ``params_std_prior``.
        params_std_prior: Dict from which the prior is extracted when
            ``prior_source="dict"`` and ``theta0`` is not given.
        prior_source: How to load the prior: ``"dict"``, ``"urdf"``, or
            ``"yaml"``.
        prior_yaml_path: Path to YAML prior file (required when
            ``prior_source="yaml"``).
        model: Pinocchio model (required when ``prior_source="urdf"``).
        weights: Diagonal weight vector (n,) for the reconstruction metric.
        joint_names: Joint names for per-joint LMI constraints (required when
            ``method="sdp"``).
        mass_min: Lower bound on link mass for SDP constraints.
        psd_eig_tol: Eigenvalue tolerance for pseudo-inertia PSD check.
        solver: SDP/LP solver backend (e.g. ``"cvxopt"``, ``"mosek"``).
        max_seconds: Time limit for SDP solver (``None`` = no limit).

    Returns:
        :class:`ReconstructionResult` with ``status``, ``base_residual_norm``,
        and ``objective`` always set.
    """
    # Normalise base_result
    if isinstance(base_result, BaseResult):
        M_in = np.asarray(base_result.M, dtype=float)
        phi_in = np.asarray(base_result.phi_base, dtype=float).reshape(-1)
        params_r = list(base_result.params_r)
    else:
        M_raw, phi_raw, params_r_raw = base_result
        M_in = np.asarray(M_raw, dtype=float)
        phi_in = np.asarray(phi_raw, dtype=float).reshape(-1)
        params_r = list(params_r_raw)

    n = len(params_r)

    # Resolve prior
    if theta0 is not None:
        prior_vec = np.asarray(theta0, dtype=float).reshape(n)
    elif prior_source == "urdf":
        prior_dict = _load_prior_from_urdf(model, params_r)
        prior_vec = prior_vector_from_dict(params_r, prior_dict)
    elif prior_source == "yaml":
        if prior_yaml_path is None:
            raise ValueError("prior_yaml_path must be set when prior_source='yaml'")
        prior_dict = _load_prior_from_yaml(prior_yaml_path, params_r)
        prior_vec = prior_vector_from_dict(params_r, prior_dict)
    else:  # "dict"
        prior_vec = prior_vector_from_dict(params_r, params_std_prior)

    # Resolve weights
    w = np.ones(n, dtype=float) if weights is None else np.asarray(weights, dtype=float).reshape(n)

    # Resolve effective method
    effective_method = method.lower().strip()
    if effective_method == "auto":
        try:
            import picos  # noqa: F401
            effective_method = "sdp"
        except Exception:
            effective_method = "nullspace"

    # Reconstruction
    status = "ok"
    objective: Optional[float] = None

    if effective_method == "sdp":
        if joint_names is None:
            raise ValueError("joint_names must be provided when method='sdp'")
        try:
            theta_r, objective = _reconstruct_sdp(
                M_in, phi_in, params_r,
                theta0=prior_vec, w=w,
                joint_names=joint_names,
                mass_min=mass_min, psd_eig_tol=psd_eig_tol,
                solver=solver, max_seconds=max_seconds,
            )
        except ImportError:
            status = "solver_missing"
            theta_r, residual_fall = reconstruct_theta_r(
                M_in, phi_in, theta0=prior_vec, weights=weights
            )
            return ReconstructionResult(
                theta_r=theta_r, params_r=params_r,
                residual=residual_fall, status=status,
                base_residual_norm=float(np.linalg.norm(residual_fall)),
            )
        except Exception as exc:
            status = "error"
            logger.warning("SDP reconstruction failed: %s", exc)
            theta_r, residual_fall = reconstruct_theta_r(
                M_in, phi_in, theta0=prior_vec, weights=weights
            )
            return ReconstructionResult(
                theta_r=theta_r, params_r=params_r,
                residual=residual_fall, status=status,
                base_residual_norm=float(np.linalg.norm(residual_fall)),
            )
    elif effective_method == "nullspace":
        theta_r, _ = reconstruct_theta_r(
            M_in, phi_in, theta0=prior_vec, weights=weights
        )
    else:
        raise ValueError(
            f"Unsupported method={method!r}. "
            "Choose 'nullspace', 'sdp', or 'auto'."
        )

    residual = M_in @ theta_r - phi_in
    return ReconstructionResult(
        theta_r=theta_r, params_r=params_r, residual=residual,
        status=status,
        base_residual_norm=float(np.linalg.norm(residual)),
        objective=objective,
    )


# ---------------------------------------------------------------------------
# Legacy pipeline entry point
# ---------------------------------------------------------------------------


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
            # project_robot_p10_lmi returns (projected_p10_dict, robot_report)
            projected_p10_dict, robot_report = project_robot_p10_lmi(
                p10_by_joint,
                mass_min=mass_min,
                psd_eig_tol=psd_eig_tol,
                solver=solver,
                verbose=verbose,
                max_seconds=max_seconds,
            )
            pc_alt_reports.append(robot_report)

            full_param_dict = param_dict_with_p10_by_joint(
                parameter_dict=full_param_dict,
                p10_by_joint=projected_p10_dict,
            )

            theta0_alt = np.array(
                [full_param_dict.get(k, default_prior) for k in params_r],
                dtype=float,
            )
            theta_r, residual = reconstruct_theta_r(
                M,
                phi_base,
                theta0=theta0_alt,
                weights=weights,
            )
            recon = ReconstructionResult(
                theta_r=theta_r,
                params_r=list(params_r),
                residual=residual,
                status="ok",
                base_residual_norm=float(np.linalg.norm(residual)),
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
        "base_residual_norm": recon.base_residual_norm,
        "parameter_dict": full_param_dict,
        "alternate_physical_consistency": alt_enabled,
        "alternation_iters": max_iters,
        "final_feasibility": final_feas,
        "pc_reports": pc_alt_reports,
    }

    return payload, full_param_dict, pc_already_applied


# Backward-compat alias: run_reconstruction already implements mode-A only
run_option_a_reconstruction = run_reconstruction
