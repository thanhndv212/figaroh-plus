"""Log-Cholesky coordinates for per-link inertial parameters.

This module maps between three representations of one rigid body's
inertial parameters and gives the derivative of the map. It is a
representation, not an estimator: nothing here fits data, solves an SDP or
imports PICOS/CVXOPT (decision record
``docs/decisions/log-cholesky-boundary-aware.md``).

Conventions
-----------
``p10`` (Pinocchio dynamic-parameter order, SI units)::

    [m, mx, my, mz, Ixx, Ixy, Iyy, Ixz, Iyz, Izz]

``m`` in kg; ``h = (mx, my, mz) = m*c`` in kg*m, with ``c`` the centre of
mass in the link frame; the six tensor entries in kg*m^2 are the rotational
inertia ``I_O`` about the link-frame origin, not about the centre of mass.
FIGAROH's regressor order differs; convert with the parameter helpers.

Pseudo-inertia ``P`` (4x4, symmetric)::

    P = [[Sigma, h], [h.T, m]],   Sigma = 0.5*trace(I_O)*eye(3) - I_O

A body is physically consistent when ``P`` is positive semidefinite; these
mappings need it strictly positive definite.

Log-Cholesky ``z`` (Pinocchio ``LogCholeskyParameters`` order)::

    [alpha, d1, d2, d3, s12, s23, s13, t1, t2, t3]

with ``P = exp(2*alpha) * U @ U.T`` and the *upper*-triangular factor::

    U = [[exp(d1), s12,     s13,     t1],
         [0,       exp(d2), s23,     t2],
         [0,       0,       exp(d3), t3],
         [0,       0,       0,       1 ]]

So ``m = exp(2*alpha)`` and ``(t1, t2, t3) = c``. ``alpha`` is
``0.5*ln(m / 1 kg)``; ``exp(d_i)``, ``s_ij`` and ``t_i`` are in metres, so
``d_i`` are logarithms of lengths in metres. ``U`` comes from a Cholesky
factorisation of ``P`` with rows and columns reversed. Every finite ``z``
decodes to a positive definite ``P``, and
``log(det(P)) = 8*alpha + 2*(d1 + d2 + d3)`` is linear in ``z``.

Policies
--------
- Massless or fixed links (``m <= 0``) have no encoding. Leave them out of
  the encoded blocks; the dictionary helpers can skip them on request and
  leave their parameters untouched.
- Singular or indefinite input is rejected. :func:`repair_p10` is the
  opt-in alternative: it lifts the eigenvalues of ``P`` to a floor, returns
  a new vector and reports how far it moved.
- ``alpha`` and ``d_i`` must stay within ``max_log_scale`` (default
  :data:`MAX_LOG_SCALE`) so that ``exp`` neither overflows nor flushes a
  factor to zero. An encoding outside it means a near-singular body.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, Iterable, Mapping, Optional

import numpy as np

from .physical_consistency import (
    p10_by_joint_from_param_dict,
    param_dict_with_p10_by_joint,
)

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

__all__ = [
    "P10_ORDER",
    "LOG_CHOLESKY_ORDER",
    "MAX_LOG_SCALE",
    "PD_RTOL",
    "RepairReport",
    "pseudo_inertia_from_p10",
    "p10_from_pseudo_inertia",
    "log_cholesky_from_pseudo_inertia",
    "pseudo_inertia_from_log_cholesky",
    "log_cholesky_from_p10",
    "p10_from_log_cholesky",
    "p10_jacobian_from_log_cholesky",
    "repair_p10",
    "log_cholesky_by_joint_from_param_dict",
    "param_dict_with_log_cholesky_by_joint",
]

P10_ORDER = ("m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz")
LOG_CHOLESKY_ORDER = (
    "alpha",
    "d1",
    "d2",
    "d3",
    "s12",
    "s23",
    "s13",
    "t1",
    "t2",
    "t3",
)

MAX_LOG_SCALE = 30.0
"""Default bound on ``|alpha|`` and ``|d_i|``.

``exp(2*30)`` is about ``1e26``: far outside any robot link in SI units
(``m`` from 1e-6 to 1e4 kg gives ``|alpha| < 7``; lengths from 1e-6 to
1e3 m give ``|d_i| < 14``), yet ``P`` stays finite with headroom. It guards
against overflow and against near-singular bodies whose factor would round
to zero; it is not a physical bound.
"""

PD_RTOL = 1e-12
"""Default relative floor on ``min eig(P)`` / ``max eig(P)`` for encoding."""

# Positions of (row, col) in U for z[4:] (s12, s23, s13, t1, t2, t3).
_OFFDIAG = ((0, 1), (1, 2), (0, 2), (0, 3), (1, 3), (2, 3))


@dataclasses.dataclass(frozen=True)
class RepairReport:
    """Result of :func:`repair_p10`. ``p10_original`` is a copy."""

    p10_original: np.ndarray
    p10: np.ndarray
    repaired: bool
    min_eig_before: float
    min_eig_after: float
    p10_change_norm: float
    p10_relative_change: float
    pseudo_inertia_change_norm: float


def _as_vector(x, name: str) -> np.ndarray:
    arr = np.array(x, dtype=float, copy=True)
    if arr.shape != (10,):
        raise ValueError(
            f"{name} must have shape (10,), got {arr.shape}; pass one link at "
            "a time or use the per-joint dictionary helpers"
        )
    if not np.all(np.isfinite(arr)):
        bad = [i for i in range(10) if not np.isfinite(arr[i])]
        raise ValueError(f"{name} has nonfinite entries at indices {bad}")
    return arr


def _as_pseudo_inertia(P, *, sym_rtol: float = 1e-12) -> np.ndarray:
    arr = np.array(P, dtype=float, copy=True)
    if arr.shape != (4, 4):
        raise ValueError(f"pseudo-inertia must have shape (4, 4), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("pseudo-inertia has nonfinite entries")
    asym = float(np.max(np.abs(arr - arr.T)))
    scale = max(float(np.max(np.abs(arr))), np.finfo(float).tiny)
    if asym > sym_rtol * scale:
        raise ValueError(
            f"pseudo-inertia is not symmetric (max |P - P.T| = {asym:.3e}, "
            f"relative {asym / scale:.3e} > {sym_rtol:.0e})"
        )
    return 0.5 * (arr + arr.T)


def pseudo_inertia_from_p10(p10) -> np.ndarray:
    """Return ``P = [[Sigma, h], [h.T, m]]`` for one link's ``p10``."""
    p = _as_vector(p10, "p10")
    m, h = p[0], p[1:4]
    Ixx, Ixy, Iyy, Ixz, Iyz, Izz = p[4:10]
    inertia = np.array([[Ixx, Ixy, Ixz], [Ixy, Iyy, Iyz], [Ixz, Iyz, Izz]])
    P = np.empty((4, 4))
    P[:3, :3] = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    P[:3, 3] = h
    P[3, :3] = h
    P[3, 3] = m
    return P


def p10_from_pseudo_inertia(P) -> np.ndarray:
    """Inverse of :func:`pseudo_inertia_from_p10`; ``P`` must be symmetric."""
    P = _as_pseudo_inertia(P)
    return _p10_from_symmetric(P)


def _p10_from_symmetric(P: np.ndarray) -> np.ndarray:
    sigma = P[:3, :3]
    inertia = np.trace(sigma) * np.eye(3) - sigma
    return np.array(
        [
            P[3, 3],
            P[0, 3],
            P[1, 3],
            P[2, 3],
            inertia[0, 0],
            inertia[0, 1],
            inertia[1, 1],
            inertia[0, 2],
            inertia[1, 2],
            inertia[2, 2],
        ]
    )


def _check_log_scales(z: np.ndarray, max_log_scale: float, what: str) -> None:
    scales = z[:4]
    worst = int(np.argmax(np.abs(scales)))
    if abs(scales[worst]) > max_log_scale:
        raise ValueError(
            f"{what}: |{LOG_CHOLESKY_ORDER[worst]}| = {abs(scales[worst]):.3g} "
            f"exceeds max_log_scale = {max_log_scale:g} (exp would overflow "
            "or flush to zero). Check units (SI expected); a near-singular "
            "body cannot be encoded reliably, see repair_p10"
        )


def log_cholesky_from_pseudo_inertia(
    P,
    *,
    rtol: float = PD_RTOL,
    max_log_scale: float = MAX_LOG_SCALE,
) -> np.ndarray:
    """Encode a strictly positive definite pseudo-inertia as ``z``.

    Raises ``ValueError`` for a wrong shape, nonfinite or nonsymmetric
    matrix, ``P[3, 3] <= 0``, ``min eig(P) <= rtol * max eig(P)``, or an
    encoding outside ``max_log_scale``.
    """
    P = _as_pseudo_inertia(P)
    if P[3, 3] <= 0.0:
        raise ValueError(
            f"mass P[3, 3] = {P[3, 3]:.6g} <= 0: a massless or fixed link has "
            "no log-Cholesky encoding; keep it out of the encoded blocks"
        )
    eig = np.linalg.eigvalsh(P)
    if eig[0] <= rtol * eig[-1]:
        raise ValueError(
            f"pseudo-inertia is not positive definite: min eig {eig[0]:.3e}, "
            f"max eig {eig[-1]:.3e} (relative floor {rtol:.0e}). The body is "
            "physically inconsistent or singular (e.g. a point mass or thin "
            "rod); use repair_p10 to lift it explicitly"
        )
    try:
        # Reverse Cholesky: P = U U^T with U upper triangular.
        U = np.linalg.cholesky(P[::-1, ::-1])[::-1, ::-1]
    except np.linalg.LinAlgError as exc:  # pragma: no cover - guarded above
        raise ValueError(f"pseudo-inertia Cholesky failed: {exc}") from exc
    u44 = U[3, 3]
    U = U / u44
    z = np.empty(10)
    z[0] = np.log(u44)
    z[1:4] = np.log(np.diag(U)[:3])
    for k, (i, j) in enumerate(_OFFDIAG):
        z[4 + k] = U[i, j]
    _check_log_scales(z, max_log_scale, "encoding")
    return z


def _factor(z: np.ndarray) -> np.ndarray:
    U = np.eye(4)
    U[0, 0], U[1, 1], U[2, 2] = np.exp(z[1:4])
    for k, (i, j) in enumerate(_OFFDIAG):
        U[i, j] = z[4 + k]
    return U


def pseudo_inertia_from_log_cholesky(
    z, *, max_log_scale: float = MAX_LOG_SCALE
) -> np.ndarray:
    """Decode ``z`` to ``P = exp(2*alpha) * U @ U.T``."""
    z = _as_vector(z, "log-Cholesky z")
    _check_log_scales(z, max_log_scale, "decoding")
    U = _factor(z)
    with np.errstate(over="ignore", invalid="ignore"):
        P = np.exp(2.0 * z[0]) * (U @ U.T)
    if not np.all(np.isfinite(P)):
        raise ValueError(
            "decoded pseudo-inertia overflowed; off-diagonal coordinates "
            f"are too large (max |s, t| = {np.max(np.abs(z[4:])):.3g})"
        )
    return 0.5 * (P + P.T)


def log_cholesky_from_p10(
    p10,
    *,
    rtol: float = PD_RTOL,
    max_log_scale: float = MAX_LOG_SCALE,
) -> np.ndarray:
    """Encode one link's ``p10`` as log-Cholesky ``z``.

    Rejects massless, singular or physically inconsistent links; see
    :func:`log_cholesky_from_pseudo_inertia`.
    """
    return log_cholesky_from_pseudo_inertia(
        pseudo_inertia_from_p10(p10), rtol=rtol, max_log_scale=max_log_scale
    )


def p10_from_log_cholesky(z, *, max_log_scale: float = MAX_LOG_SCALE) -> np.ndarray:
    """Decode ``z`` to one link's ``p10``; always physically consistent."""
    return _p10_from_symmetric(
        pseudo_inertia_from_log_cholesky(z, max_log_scale=max_log_scale)
    )


def p10_jacobian_from_log_cholesky(
    z, *, max_log_scale: float = MAX_LOG_SCALE
) -> np.ndarray:
    """Return ``d p10 / d z`` (10x10; rows in :data:`P10_ORDER`, columns in
    :data:`LOG_CHOLESKY_ORDER`).

    Matches Pinocchio's ``LogCholeskyParameters.calculateJacobian``. Chain a
    regressor in Pinocchio order through it: ``d tau / d z = W @ J``.
    """
    z = _as_vector(z, "log-Cholesky z")
    _check_log_scales(z, max_log_scale, "Jacobian")
    U = _factor(z)
    scale = np.exp(2.0 * z[0])
    J = np.empty((10, 10))
    J[:, 0] = _p10_from_symmetric(2.0 * scale * (U @ U.T))
    for k in range(1, 10):
        E = np.zeros((4, 4))
        if k < 4:
            E[k - 1, k - 1] = U[k - 1, k - 1]
        else:
            i, j = _OFFDIAG[k - 4]
            E[i, j] = 1.0
        dP = scale * (E @ U.T + U @ E.T)
        J[:, k] = _p10_from_symmetric(dP)
    return J


def repair_p10(
    p10,
    *,
    min_eig: Optional[float] = None,
    rtol: float = 1e-8,
) -> RepairReport:
    """Opt-in repair: lift ``eig(P)`` to a floor and report the change.

    The floor is ``min_eig`` if given, else ``rtol * max(max eig(P), m)``.
    Eigenvalues below it are raised to it (the Frobenius-nearest matrix on
    ``P`` with that floor, not a weighted or torque-aware projection; use
    the LMI projection or the SDP fit for those). The input is copied, never
    modified; a body already above the floor is returned unchanged with
    ``repaired=False``. A nonpositive mass is not repaired.
    """
    original = _as_vector(p10, "p10")
    if original[0] <= 0.0:
        raise ValueError(
            f"mass {original[0]:.6g} <= 0: repair does not invent mass for a "
            "massless or fixed link; keep it out of the encoded blocks"
        )
    P = pseudo_inertia_from_p10(original)
    eigval, eigvec = np.linalg.eigh(P)
    floor = (
        float(min_eig)
        if min_eig is not None
        else rtol * max(float(eigval[-1]), float(original[0]))
    )
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"repair floor must be positive and finite, got {floor}")
    if eigval[0] >= floor:
        repaired_p10, P_new, did = original.copy(), P, False
    else:
        P_new = (eigvec * np.maximum(eigval, floor)) @ eigvec.T
        P_new = 0.5 * (P_new + P_new.T)
        repaired_p10, did = _p10_from_symmetric(P_new), True
    change = float(np.linalg.norm(repaired_p10 - original))
    report = RepairReport(
        p10_original=original,
        p10=repaired_p10,
        repaired=did,
        min_eig_before=float(eigval[0]),
        min_eig_after=float(np.linalg.eigvalsh(P_new)[0]),
        p10_change_norm=change,
        p10_relative_change=change / max(float(np.linalg.norm(original)), 1e-300),
        pseudo_inertia_change_norm=float(np.linalg.norm(P_new - P)),
    )
    if did:
        logger.info(
            "repaired p10: min eig %.3e -> %.3e, |dp10| = %.3e (relative %.3e)",
            report.min_eig_before,
            report.min_eig_after,
            change,
            report.p10_relative_change,
        )
    return report


def log_cholesky_by_joint_from_param_dict(
    *,
    parameter_dict: Mapping[str, float],
    joint_names: Iterable[str],
    massless: str = "raise",
    rtol: float = PD_RTOL,
    max_log_scale: float = MAX_LOG_SCALE,
) -> Dict[str, np.ndarray]:
    """Encode the links named by ``joint_names`` from a flat parameter dict.

    Keys follow ``m_<joint>``, ``mx_<joint>``, ..., ``Izz_<joint>``; other
    keys (friction, offsets, ...) are ignored and the input is not
    modified. ``massless="skip"`` leaves links with ``m <= 0`` out of the
    result (so their parameters stay as they are on decode); the default
    ``"raise"`` rejects them. Errors name the offending joint.
    """
    if massless not in ("raise", "skip"):
        raise ValueError(f"massless must be 'raise' or 'skip', got {massless!r}")
    p10_by_joint = p10_by_joint_from_param_dict(
        parameter_dict=parameter_dict, joint_names=list(joint_names)
    )
    out: Dict[str, np.ndarray] = {}
    for joint, p10 in p10_by_joint.items():
        if massless == "skip" and p10[0] <= 0.0:
            logger.debug("skipping massless link %s (m = %g)", joint, p10[0])
            continue
        try:
            out[joint] = log_cholesky_from_p10(
                p10, rtol=rtol, max_log_scale=max_log_scale
            )
        except ValueError as exc:
            raise ValueError(f"joint {joint!r}: {exc}") from exc
    return out


def param_dict_with_log_cholesky_by_joint(
    *,
    parameter_dict: Mapping[str, float],
    log_cholesky_by_joint: Mapping[str, np.ndarray],
    max_log_scale: float = MAX_LOG_SCALE,
) -> Dict[str, float]:
    """Return a copy of ``parameter_dict`` with the encoded links decoded.

    Only the ten inertial keys of each joint in ``log_cholesky_by_joint``
    change; every other key, and key order, is preserved.
    """
    p10_by_joint = {}
    for joint, z in log_cholesky_by_joint.items():
        try:
            p10_by_joint[joint] = p10_from_log_cholesky(z, max_log_scale=max_log_scale)
        except ValueError as exc:
            raise ValueError(f"joint {joint!r}: {exc}") from exc
    return param_dict_with_p10_by_joint(
        parameter_dict=dict(parameter_dict), p10_by_joint=p10_by_joint
    )
