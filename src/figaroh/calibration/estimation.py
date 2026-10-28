# Copyright [2021-2026] Thanh Nguyen
# Copyright [2022-2024] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Choosing which calibration parameters to estimate, and how (#113).

``calib_config["estimation"]["method"]`` selects one of:

``structural`` (default)
    The structural base parameters from random configurations, then the
    data-level elimination of frame-absorbed parameters (#102). Unchanged
    behaviour. ``structural_selection`` picks how the independent columns
    are chosen among exactly dependent ones: ``lapack`` (default) takes the
    largest residual first, so the selected names follow the random draw
    (``random_seed``); ``ordered`` takes the first independent columns in
    joint order, which is the same for every draw (#204).
``excitation``
    Start from every joint parameter of the calibration level, drop exact
    dependencies on the measured postures, then remove, one at a time, the
    parameter whose predicted standard error is largest relative to its
    expected size, while that ratio exceeds ``excitation_k``. Kept
    parameters are estimated freely.
``map``
    Estimate every joint parameter with a zero-mean Gaussian prior of the
    given expected sizes (``priors``). Weakly excited directions stay near
    nominal; nothing is dropped.
``map_cv``
    ``map`` with the prior sizes scaled by a factor chosen by k-fold
    cross-validation over the training postures (optionally one factor per
    parameter group). Needs no robot-specific sizes.
``cv_subset``
    Nested parameter sets in the ``excitation`` removal order; the set size
    is chosen by k-fold cross-validation over the training postures.

Nuisance parameters are always estimated freely (no prior): by default the
base and tool frames, or what the calibrator's
:meth:`~BaseCalibration.nuisance_param_names` returns (#210). Noise is
``noise_std`` in the units of the cost function's measurement residuals
(metres for position unless the robot's cost function weights them), or,
when ``None``, estimated from a free fit of the identifiable parameters.
``noise_std`` may also be one value per measurement component, given or
estimated (``per_component``); the measurement residuals are then whitened
before the fit, the prior weighting and the covariance (#215).
That fit also gives ``excitation`` and ``cv_subset`` their Jacobian; it is
skipped for ``map`` and ``map_cv`` when ``noise_std`` is given, and a fit
that stops unconverged is logged and reported as ``prelim_converged`` (#211).
Selection runs once, in :meth:`BaseCalibration.create_param_list`; the
choices and diagnostics are stored in ``calib_config["estimation_report"]``.
See the guide ``docs/source/concepts/calibration_estimation.md``.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from .calibration_tools import (
    calc_updated_fkm,
    measurement_jacobian,
    select_identifiable_parameters,
)
from .parameter import BASE_TPL, get_fullparam_offset, get_joint_offset

logger = logging.getLogger(__name__)

METHODS = ("structural", "excitation", "map", "map_cv", "cv_subset")
#: ``noise_std`` value asking for one noise estimate per measurement component
PER_COMPONENT = "per_component"
STRUCTURAL_SELECTIONS = ("lapack", "ordered")
GROUPS = ("translation", "rotation", "joint_offset", "prismatic_offset")
DEFAULT_PRIORS = {
    "translation": 1e-3,  # m, placement translation
    "rotation": 2e-3,  # rad, placement rotation
    "joint_offset": 2e-2,  # rad, revolute joint-angle offset
    "prismatic_offset": 2e-3,  # m, prismatic joint offset
}
DEFAULTS = {
    "method": "structural",
    "priors": DEFAULT_PRIORS,
    "noise_std": None,
    "excitation_k": 1.0,
    "cv_folds": 5,
    "cv_seed": 0,
    "cv_multipliers": [0.1, 0.3, 1.0, 3.0, 10.0],
    "cv_per_group": False,
    "cv_sizes": None,
    "cv_rule": "min",
    "dependency_tol": 1e-7,
    "structural_selection": "lapack",
}
# evaluation cap of the free fit behind the noise estimate and the
# excitation Jacobian (#211)
PRELIM_MAX_NFEV = 2000
_BASE_MAPPING_KEYS = (
    "base_mapping_matrix",
    "base_mapping_param_names",
    "base_mapping_row_names",
    "base_mapping_slice",
)
# per-component whitening read by :func:`whiten` (#215)
_NOISE_KEYS = ("measurement_noise_std", "measurement_noise_ref")


class EstimationError(ValueError):
    """Invalid estimation settings or data for the chosen method."""


def settings(calib_config: dict) -> dict:
    """``calib_config["estimation"]`` merged over :data:`DEFAULTS`."""
    user = dict(calib_config.get("estimation") or {})
    out = dict(DEFAULTS, **user)
    out["priors"] = dict(DEFAULT_PRIORS, **(user.get("priors") or {}))
    if out["method"] not in METHODS:
        raise EstimationError(
            f"estimation.method '{out['method']}' not in {', '.join(METHODS)}"
        )
    unknown = set(out["priors"]) - set(GROUPS)
    if unknown:
        raise EstimationError(f"unknown prior groups {sorted(unknown)}")
    if out["structural_selection"] not in STRUCTURAL_SELECTIONS:
        raise EstimationError(
            "estimation.structural_selection must be one of "
            f"{', '.join(STRUCTURAL_SELECTIONS)}"
        )
    if out["cv_rule"] not in ("min", "one_se"):
        raise EstimationError("estimation.cv_rule must be 'min' or 'one_se'")
    _check_noise_setting(out["noise_std"])
    return out


def _check_noise_setting(noise) -> None:
    if noise is None or (isinstance(noise, str) and noise == PER_COMPONENT):
        return
    if isinstance(noise, str):
        raise EstimationError(
            f"estimation.noise_std must be a number, a list of numbers, null "
            f"or '{PER_COMPONENT}', not '{noise}'"
        )
    values = np.asarray(noise, dtype=float)
    if values.ndim > 1 or values.size == 0:
        raise EstimationError("estimation.noise_std must be a number or a flat list")
    if not np.all(np.isfinite(values) & (values > 0)):
        raise EstimationError("estimation.noise_std values must be positive")


# ── measurement noise per component (#215) ──────────────────────


def n_components(calib) -> int:
    """Measurement components per sample: ``len(PEE_measured) / NbSample``."""
    n = calib.calib_config["NbSample"]
    m = len(calib.PEE_measured)
    if n <= 0 or m % n:
        raise EstimationError(
            f"{m} measurements are not a whole number of components for " f"{n} samples"
        )
    return m // n


def _noise_rotation(calib, n_comp=None):
    """The calibrator's per-sample sensor-frame rotations, or None."""
    R = getattr(calib, "noise_rotation", None)
    if R is None:
        return None
    R = np.asarray(R, dtype=float)
    if n_comp is not None:
        n = calib.calib_config["NbSample"]
        if R.shape != (n, n_comp, n_comp):
            raise EstimationError(
                f"noise_rotation has shape {R.shape}, expected "
                f"({n}, {n_comp}, {n_comp}): one rotation per sample over "
                "the measurement components"
            )
        if not np.allclose(R @ np.swapaxes(R, 1, 2), np.eye(n_comp), atol=1e-9):
            raise EstimationError("noise_rotation matrices must be orthonormal")
    return R


def whiten(calib, rows, sigma=None, ref=None):
    """Measurement rows of ``rows`` (residuals or Jacobian) whitened.

    Off (``rows`` returned as is) unless ``configure`` set unequal
    per-component noise (``calib_config["measurement_noise_std"]``). The
    first ``len(PEE_measured)`` rows, DOF-major, are scaled by
    ``ref / sigma_c`` of their component, with ``ref`` the smallest sigma
    (``measurement_noise_ref``): the problem keeps one noise level ``ref``
    for the priors and the covariance. With per-sample rotations
    (``calib.noise_rotation``, R_i) the sensor-frame covariance
    Sigma = diag(sigma**2) becomes R_i Sigma R_i^T, and sample i's rows are
    multiplied by ref * R_i diag(1/sigma) R_i^T. Later rows are unchanged.
    ``sigma`` and ``ref`` override the configured values.
    """
    cfg = calib.calib_config
    if sigma is None:
        sigma = cfg.get("measurement_noise_std")
        ref = cfg.get("measurement_noise_ref")
    if sigma is None:
        return rows
    sigma = np.asarray(sigma, dtype=float)
    rows = np.array(rows, dtype=float)
    n, c = cfg["NbSample"], len(sigma)
    head = rows[: n * c].reshape((c, n) + rows.shape[1:])
    scale = (ref / sigma).reshape((c, 1) + (1,) * (head.ndim - 2))
    R = _noise_rotation(calib)
    if R is None:
        head = head * scale
    else:
        sensor = np.einsum("ikc,ki...->ci...", R, head)
        head = np.einsum("ikc,ci...->ki...", R, sensor * scale)
    rows[: n * c] = head.reshape((n * c,) + rows.shape[1:])
    return rows


def component_noise(calib, residuals, n_params) -> np.ndarray:
    """Noise per measurement component from free-fit residuals.

    Each component's rows share the ``n_params`` degrees of freedom equally,
    so ``sigma_c = sqrt(sum r_c**2 / (NbSample - n_params / n_components))``.
    With ``calib.noise_rotation`` the residuals are first expressed in the
    sensor frame. The residuals hold the unmodelled systematic error too: the
    estimate is noise plus model error.
    """
    n, c = calib.calib_config["NbSample"], n_components(calib)
    r = np.asarray(residuals, dtype=float)[: n * c].reshape(c, n)
    R = _noise_rotation(calib)
    if R is not None:
        r = np.einsum("ikc,ki->ci", R, r)
    dof = n - n_params / c
    if dof <= 0:
        raise EstimationError(
            "too few samples to estimate the noise per component; set "
            "estimation.noise_std"
        )
    return np.sqrt(np.sum(r**2, axis=1) / dof)


# ── candidates and priors ──────────────────────────────────────


def joint_candidates(model, calib_config: dict) -> List[str]:
    """Every joint parameter of the calibration level, in joint order."""
    act = [model.names[j] for j in calib_config["actJoint_idx"]]
    if calib_config["calib_model"] == "full_params":
        return list(get_fullparam_offset(act))
    names = list(get_joint_offset(model, act))  # covers every model joint
    by_joint = {n.split("_", 1)[1]: n for n in names}
    return [by_joint[j] for j in act if j in by_joint]


def _joint_axis(model, joint: str):
    """('R' or 'P', axis letter) for an axis-aligned 1-DoF joint, else None."""
    if model is None or not model.existJointName(joint):
        return None
    short = model.joints[model.getJointId(joint)].shortname()
    tail = short.replace("JointModel", "")
    if len(tail) == 2 and tail[0] in "RP" and tail[1] in "XYZ":
        return tail[0], tail[1].lower()
    return None


def prior_group(name: str, model) -> Optional[str]:
    """Prior group of a parameter name; ``None`` for frame parameters."""
    if name.startswith("base_") or "EE" in name:
        return None
    if name.startswith("d_"):
        _, kind, joint = name.split("_", 2)
        axis = _joint_axis(model, joint)
        if axis == ("R", kind[-1]) and kind.startswith("phi"):
            return "joint_offset"
        if axis == ("P", kind[-1]) and kind.startswith("p"):
            return "prismatic_offset"
        return "rotation" if kind.startswith("phi") else "translation"
    if name.startswith("offsetP"):
        return "prismatic_offset"
    return "joint_offset"


def prior_std(
    names: Sequence[str], model, priors: Dict[str, float], nuisance=()
) -> np.ndarray:
    """Expected size per parameter; ``inf`` for frame and nuisance parameters."""
    nuisance = set(nuisance)
    out = []
    for n in names:
        group = None if n in nuisance else prior_group(n, model)
        out.append(np.inf if group is None else float(priors[group]))
    return np.array(out)


# ── fitting helpers ─────────────────────────────────────────────


def _sample_rows(n: int, ndof: int, idx) -> np.ndarray:
    """Rows of a DOF-major measurement vector that belong to samples ``idx``."""
    idx = np.asarray(idx)
    return np.concatenate([d * n + idx for d in range(ndof)])


@contextmanager
def _samples(calib, idx):
    """Restrict the calibrator's measurements to sample indices ``idx``.

    Slices ``q_measured``, ``PEE_measured`` and every attribute named in
    ``calib.per_sample_attributes`` (first axis), and sets ``NbSample``.
    """
    cfg = calib.calib_config
    n = cfg["NbSample"]
    pee = np.asarray(calib.PEE_measured)
    ndof = pee.size // n
    extra = tuple(getattr(calib, "per_sample_attributes", ()))
    if getattr(calib, "noise_rotation", None) is not None and (
        "noise_rotation" not in extra
    ):
        extra += ("noise_rotation",)
    saved = (calib.q_measured, calib.PEE_measured, n)
    saved_extra = {a: getattr(calib, a) for a in extra}
    idx = np.asarray(idx)
    calib.q_measured = np.asarray(calib.q_measured)[idx]
    calib.PEE_measured = pee[_sample_rows(n, ndof, idx)]
    for a, value in saved_extra.items():
        setattr(calib, a, np.asarray(value)[idx])
    cfg["NbSample"] = len(idx)
    try:
        yield
    finally:
        calib.q_measured, calib.PEE_measured, cfg["NbSample"] = saved
        for a, value in saved_extra.items():
            setattr(calib, a, value)


@contextmanager
def _params(calib, names, weights=None):
    cfg = calib.calib_config
    saved = (cfg["param_name"], cfg.get("prior_weights"))
    cfg["param_name"] = list(names)
    cfg["prior_weights"] = weights
    try:
        yield
    finally:
        cfg["param_name"], cfg["prior_weights"] = saved


def _start(calib, names):
    guess = calib.initial_frame_guess()
    return np.array([guess.get(n, 0.0) for n in names])


def _fit(calib, names, weights=None, max_nfev=2000):
    """Least squares of the calibrator's objective over ``names``."""
    with _params(calib, names, weights):
        x0 = _start(calib, names)
        m = len(calib._objective(x0))
        method = "lm" if m >= len(names) else "trf"
        return least_squares(calib._objective, x0, method=method, max_nfev=max_nfev)


def _measurement_rms(calib, names, x) -> float:
    with _params(calib, names):
        r = np.asarray(whiten(calib, calib.cost_function(x)))
        r = r[: len(calib.PEE_measured)]
    return float(np.sqrt(np.mean(r**2)))


def _folds(n: int, k: int, seed: int) -> List[np.ndarray]:
    if k < 2 or k > n:
        raise EstimationError(f"cv_folds={k} needs 2 <= k <= {n} samples")
    perm = np.random.default_rng(seed).permutation(n)
    return [np.sort(f) for f in np.array_split(perm, k)]


def _cv_error(calib, names, weights_fn, folds) -> np.ndarray:
    """Held-out measurement RMS per fold; ``weights_fn(names)`` or None."""
    n = calib.calib_config["NbSample"]
    errors = []
    for test in folds:
        train = np.setdiff1d(np.arange(n), test)
        w = weights_fn(names) if weights_fn else None
        with _samples(calib, train):
            x = _fit(calib, names, w).x
        with _samples(calib, test):
            errors.append(_measurement_rms(calib, names, x))
    return np.array(errors)


# ── selection ───────────────────────────────────────────────────


def check_nuisance(J, names, nuisance, tol, where="on this data") -> None:
    """Raise when the nuisance columns of ``J`` are dependent among themselves.

    Nuisance parameters get no prior, so a direction they share would drift
    freely instead of being fixed by the data.
    """
    if not nuisance:
        return
    cols = [names.index(n) for n in nuisance]
    _, dependent = select_identifiable_parameters(J[:, cols], nuisance, tol=tol)
    if dependent:
        raise EstimationError(
            f"nuisance parameters are not identifiable {where}: fix a gauge "
            f"(dependent on the others: {dependent})"
        )


def identifiable(calib, names, nuisance, tol):
    """Drop exact dependencies on the measured postures (nuisance kept).

    Returns the kept names and the measurement Jacobian over ``names``
    (:meth:`BaseCalibration.measurement_jacobian`). Raises when the nuisance
    parameters are dependent among themselves (:func:`check_nuisance`).
    """
    x0 = _start(calib, names)
    J = calib.measurement_jacobian(names, x0)
    check_nuisance(J, names, nuisance, tol)
    kept, _ = select_identifiable_parameters(J, names, nuisance, tol=tol)
    return kept, J


def _check_folds(J, names, nuisance, folds, n, tol) -> None:
    """Every CV training set must determine the nuisance parameters.

    A fold whose training samples miss, for example, every posture of one
    session leaves that session's nuisance parameters undetermined.
    """
    ndof = J.shape[0] // n
    for k, test in enumerate(folds):
        train = np.setdiff1d(np.arange(n), test)
        check_nuisance(
            J[_sample_rows(n, ndof, train)],
            names,
            nuisance,
            tol,
            where=f"on the training samples of CV fold {k} (reduce cv_folds or "
            "add samples)",
        )


def excitation_order(J, names, frames, prior, noise, k=None):
    """Backward elimination by predicted standard error / expected size.

    ``frames`` are the nuisance parameters, never removed. Returns (kept,
    removed in order, ratio at removal). With ``k`` None the elimination runs
    until only ``frames`` remain.
    """
    active = list(names)
    removed, ratios = [], []
    while True:
        cols = [names.index(n) for n in active]
        Ja = J[:, cols]
        cov = noise**2 * np.linalg.inv(Ja.T @ Ja)
        ratio = np.sqrt(np.abs(np.diag(cov))) / prior[cols]
        joint = [i for i, n in enumerate(active) if n not in frames]
        if not joint:
            break
        worst = max(joint, key=lambda i: (ratio[i], -i))
        if k is not None and ratio[worst] <= k:
            break
        removed.append(active[worst])
        ratios.append(float(ratio[worst]))
        active.pop(worst)
    return active, removed, ratios


def _canonical(order, names):
    keep = set(names)
    return [n for n in order if n in keep]


def configure(calib) -> dict:
    """Set ``param_name`` and prior weights for a non-structural method.

    The candidates are every joint parameter of the calibration level plus
    the calibrator's nuisance parameters
    (:meth:`BaseCalibration.nuisance_param_names`, by default the unknown
    base and tool frames), which get no prior. A calibrator with its own
    measurement model (not an end-effector pose) works here when it keeps
    this contract (#210):

    - the first ``len(PEE_measured)`` entries of ``cost_function`` are the
      measurement residuals, laid out DOF-major (all samples of the first
      measured component, then the next), with no regularisation rows
      appended (use priors, #120); later rows are ignored;
    - parameters are read by name from ``calib_config["param_name"]``, which
      this function changes;
    - :meth:`BaseCalibration.measurement_jacobian` is the derivative of the
      quantity measured (by default a finite difference of
      :meth:`BaseCalibration.get_pose_from_measure`);
    - all per-sample state is ``q_measured``, ``PEE_measured`` or an
      attribute named in ``per_sample_attributes``, so that the
      cross-validation folds can slice it; ``NbSample`` follows the slice;
    - the nuisance parameters are identifiable on the data, and on every
      fold's training samples for ``map_cv`` and ``cv_subset``; otherwise
      :class:`EstimationError` asks for a gauge.
    """
    cfg = calib.calib_config
    est = settings(cfg)
    if cfg.get("non_geom"):
        raise EstimationError(
            f"estimation.method '{est['method']}' does not support "
            "include_non_geometric; use 'structural'"
        )
    if not (hasattr(calib, "q_measured") and hasattr(calib, "PEE_measured")):
        raise EstimationError(
            f"estimation.method '{est['method']}' selects on the measured "
            "postures: load the data before create_param_list()"
        )
    model = calib.model
    candidates = joint_candidates(model, cfg)
    nuisance = list(calib.nuisance_param_names())
    clash = sorted(set(nuisance) & set(candidates))
    if clash:
        raise EstimationError(f"nuisance parameters {clash} are joint parameters")
    # base frame first, the other nuisance parameters last: the layout the
    # robot cost functions read
    head = [n for n in nuisance if n in BASE_TPL]
    all_names = head + candidates + [n for n in nuisance if n not in head]
    cfg["param_name"] = list(all_names)
    for key in _BASE_MAPPING_KEYS + _NOISE_KEYS:
        cfg.pop(key, None)
    prior = prior_std(all_names, model, est["priors"], nuisance)
    prior_of = dict(zip(all_names, prior))

    ident, J_all = identifiable(calib, all_names, nuisance, est["dependency_tol"])
    method = est["method"]
    n_meas = len(calib.PEE_measured)
    n_comp = n_components(calib)
    rotation = _noise_rotation(calib, n_comp)
    noise, noise_source = est["noise_std"], "given"
    per_component = isinstance(noise, str)
    sigma = None  # one noise per component, when not one scalar
    if not per_component and np.ndim(noise) == 1:
        sigma = np.asarray(noise, dtype=float)
        if len(sigma) != n_comp:
            raise EstimationError(
                f"estimation.noise_std has {len(sigma)} values; the "
                f"measurements have {n_comp} components per sample"
            )
        noise = _set_component_noise(cfg, sigma)
    dof = n_meas - len(ident)
    if (noise is None or per_component) and dof <= 0:
        raise EstimationError(
            "too few measurements to estimate the noise; set estimation.noise_std"
        )
    # the free fit feeds only the noise estimate and the excitation
    # Jacobian; skip it when neither is needed (#211)
    prelim = None
    # per-component row weights of the free fit, hence of its Jacobian
    prelim_weight = _row_weight(cfg, n_comp)
    if noise is None or per_component or method in ("excitation", "cv_subset"):
        prelim = _fit(calib, ident, max_nfev=PRELIM_MAX_NFEV)
        if not prelim.success:
            logger.warning(
                "estimation '%s': the free fit of the %d identifiable "
                "parameters did not converge (%s, %d evaluations); its noise "
                "estimate and excitation Jacobian are unreliable. Set "
                "estimation.noise_std to skip it for map and map_cv.",
                method,
                len(ident),
                prelim.message,
                prelim.nfev,
            )
    if per_component:
        # the unweighted fit lets the noisiest component pull the others'
        # residuals: estimate again after one fit weighted by the first
        # estimate (further rounds change it by well under 1%)
        sigma = component_noise(calib, prelim.fun, len(ident))
        _set_component_noise(cfg, sigma)
        if cfg.get("measurement_noise_std") is not None:
            prelim = _fit(calib, ident, max_nfev=PRELIM_MAX_NFEV)
            with _params(calib, ident):
                raw = calib.cost_function(prelim.x)
            prelim_weight = _row_weight(cfg, n_comp)
            sigma = component_noise(calib, raw, len(ident))
            for key in _NOISE_KEYS:
                cfg.pop(key, None)
        noise = _set_component_noise(cfg, sigma)
        noise_source = (
            "estimated per component (free fit of the identifiable "
            "parameters): noise plus model error"
        )
    elif noise is None:
        noise = float(np.sqrt(np.sum(prelim.fun[:n_meas] ** 2) / dof))
        noise_source = "estimated (free fit of the identifiable parameters)"
    whitened = cfg.get("measurement_noise_std") is not None
    report = {
        "method": est["method"],
        "candidates": candidates,
        "nuisance": nuisance,
        "dependent": [n for n in all_names if n not in ident],
        # one value, or one per measurement component (then whitened
        # to the smallest, ``noise_std_ref``)
        "noise_std": [float(v) for v in sigma] if whitened else noise,
        "noise_source": noise_source,
        "priors": est["priors"],
        # None: the free fit did not run
        "prelim_converged": None if prelim is None else bool(prelim.success),
        "prelim_nfev": 0 if prelim is None else int(prelim.nfev),
    }
    if whitened:
        report["noise_std_ref"] = noise
        report["noise_rotated"] = rotation is not None

    def weights(names, scale=None):
        scale = scale or {}
        s = np.array(
            [prior_of[n] * scale.get(prior_group(n, model), 1.0) for n in names]
        )
        return np.where(np.isfinite(s), noise / s, 0.0)

    if method in ("map_cv", "cv_subset"):
        n = cfg["NbSample"]
        folds = _folds(n, est["cv_folds"], est["cv_seed"])
        _check_folds(J_all, all_names, nuisance, folds, n, est["dependency_tol"])
    final, w = None, None
    if method in ("excitation", "cv_subset"):
        # reweight the free fit's Jacobian to the final whitening
        ratio = _row_weight(cfg, n_comp) / prelim_weight
        J = prelim.jac[:n_meas]
        if np.any(ratio != 1.0):
            J = whiten(calib, J, sigma=1.0 / ratio, ref=1.0)
        prior_ident = np.array([prior_of[n] for n in ident])
        k = est["excitation_k"] if method == "excitation" else None
        kept, removed, ratios = excitation_order(
            J, ident, nuisance, prior_ident, noise, k
        )
        report["removal_order"] = list(zip(removed, ratios))
        if method == "excitation":
            report["excitation_k"] = est["excitation_k"]
            final = kept
        else:
            final = _choose_subset(calib, ident, removed, est, report)
    elif method == "map":
        final, w = all_names, weights(all_names)
    elif method == "map_cv":
        scale = _choose_scale(calib, all_names, weights, est, report, nuisance)
        report["prior_scale"] = scale
        final, w = all_names, weights(all_names, scale)

    cfg["param_name"] = _canonical(all_names, final)
    cfg["prior_weights"] = None if w is None else w
    cfg["prior_noise"] = None if w is None else noise
    cfg["absorbed_param_name"] = report["dependent"]
    report["param_name"] = list(cfg["param_name"])
    cfg["estimation_report"] = report
    calib.estimation_report = report
    logger.info(
        "estimation '%s': %d parameters (noise %.3g, %s)",
        method,
        len(cfg["param_name"]),
        noise,
        noise_source,
    )
    return report


def _row_weight(cfg, n_comp) -> np.ndarray:
    """Per-component weights :func:`whiten` applies (ones when off)."""
    sigma = cfg.get("measurement_noise_std")
    if sigma is None:
        return np.ones(n_comp)
    return cfg["measurement_noise_ref"] / np.asarray(sigma, dtype=float)


def _set_component_noise(cfg, sigma) -> float:
    """Turn on whitening for unequal ``sigma``; return the problem's noise.

    Equal values are one scalar noise: no whitening, so the scalar results
    are reproduced exactly (a rotation of an isotropic covariance is the
    same covariance).
    """
    sigma = np.asarray(sigma, dtype=float)
    ref = float(np.min(sigma))
    if np.all(sigma == ref):
        return ref
    cfg["measurement_noise_std"] = [float(v) for v in sigma]
    cfg["measurement_noise_ref"] = ref
    return ref


def _choose_scale(calib, names, weights, est, report, nuisance=()) -> Dict[str, float]:
    folds = _folds(calib.calib_config["NbSample"], est["cv_folds"], est["cv_seed"])
    grid = [float(m) for m in est["cv_multipliers"]]
    curve = {}

    def score(scale):
        key = tuple(sorted(scale.items()))
        if key not in curve:
            err = _cv_error(calib, names, lambda n: weights(n, scale), folds)
            curve[key] = (float(err.mean()), float(err.std(ddof=1) / np.sqrt(len(err))))
        return curve[key][0]

    best = min(grid, key=lambda m: score({g: m for g in GROUPS}))
    scale = {g: best for g in GROUPS}
    if est["cv_per_group"]:
        present = {prior_group(n, calib.model) for n in names if n not in nuisance} - {
            None
        }
        for _ in range(2):
            for g in GROUPS:
                if g not in present:
                    continue
                scale[g] = min(grid, key=lambda m: score(dict(scale, **{g: m})))
    report["cv_curve"] = [
        {"scale": dict(k), "rms": v[0], "se": v[1]} for k, v in curve.items()
    ]
    return scale


def _choose_subset(calib, ident, removed, est, report) -> List[str]:
    folds = _folds(calib.calib_config["NbSample"], est["cv_folds"], est["cv_seed"])
    n_joint = len(removed)
    sizes = est["cv_sizes"] or list(range(n_joint + 1))
    sizes = sorted({int(s) for s in sizes if 0 <= int(s) <= n_joint})
    curve = []
    for size in sizes:
        drop = set(removed[: n_joint - size])
        names = [n for n in ident if n not in drop]
        err = _cv_error(calib, names, None, folds)
        curve.append(
            {
                "n_joint": size,
                "rms": float(err.mean()),
                "se": float(err.std(ddof=1) / np.sqrt(len(err))),
            }
        )
    best = min(curve, key=lambda c: (c["rms"], c["n_joint"]))
    chosen = best
    if est["cv_rule"] == "one_se":
        limit = best["rms"] + best["se"]
        chosen = min(
            (c for c in curve if c["rms"] <= limit), key=lambda c: c["n_joint"]
        )
    report["cv_curve"] = curve
    report["cv_rule"] = est["cv_rule"]
    report["n_joint_chosen"] = chosen["n_joint"]
    drop = set(removed[: n_joint - chosen["n_joint"]])
    return [n for n in ident if n not in drop]


# ── lift of structural base parameters (#111) ───────────────────


def lift_structural(
    calib, priors: Optional[Dict[str, float]] = None, refine_iterations: int = 3
):
    """Weighted minimum-norm lift of fitted base parameters to joint corrections.

    The ``structural`` method estimates one representative per group of
    dependent joint parameters; which one is a numerical accident. Among all
    joint corrections that reproduce the fit, return the most plausible
    under the expected sizes ``priors`` (default: the estimation priors):

        theta = argmin theta' P^-1 theta  subject to  M_full theta = t

    ``M_full`` is the base mapping before any row was dropped; ``t`` holds
    the fitted value of every kept joint row, and 0 on the rows the base
    frame carries (``base_frame_row_names``) and on the rows the fit
    dropped (held at 0). The result does not depend on which
    representatives were chosen.

    ``M`` is a first-order (structural) map, so with corrections of tens of
    mrad the lifted model drifts from the fit by up to ~0.1 mm. When the
    measured postures are available, ``refine_iterations`` Gauss-Newton
    steps then make the lifted model reproduce the fitted predictions at
    those postures, each step taking the smallest correction in the same
    prior-weighted norm.

    Returns:
        (names, theta, cov): joint parameter names
        (``base_mapping_param_names``), lifted values, and their covariance
        conditional on this lift (propagated from the fitted base
        parameters; directions the data does not see have zero variance
        here, not their prior).
    """
    cfg = calib.calib_config
    M = np.asarray(cfg.get("base_mapping_matrix_full", cfg["base_mapping_matrix"]))
    rows_full = list(
        cfg.get("base_mapping_row_names_full", cfg["base_mapping_row_names"])
    )
    kept = list(cfg["base_mapping_row_names"])
    frame_rows = set(cfg.get("base_frame_row_names", []))
    names = list(cfg["base_mapping_param_names"])
    start, _ = cfg["base_mapping_slice"]
    x = np.asarray(calib.var_)
    C = np.asarray(calib._C_param)

    free = [
        (i, start + kept.index(r))
        for i, r in enumerate(rows_full)
        if r in kept and r not in frame_rows
    ]
    t = np.zeros(len(rows_full))
    for i, k in free:
        t[i] = x[k]

    sizes = (
        settings(cfg)["priors"] if priors is None else dict(DEFAULT_PRIORS, **priors)
    )
    P = prior_std(names, getattr(calib, "model", None), sizes) ** 2
    G = (P[:, None] * M.T) @ np.linalg.inv((M * P) @ M.T)
    theta = G @ t
    if refine_iterations and hasattr(calib, "q_measured"):
        frames = set(calib._frame_param_names())
        other = [
            n
            for n in cfg["param_name"]
            if n not in frames and not n.startswith(("d_", "offset"))
        ]
        if other:
            # e.g. contact-plane parameters: core's FK cannot evaluate the fit
            logger.info(
                "lift not refined (first order only): parameters %s are not "
                "joint or frame parameters",
                other[:3],
            )
        else:
            theta = _refine_lift(calib, names, theta, np.sqrt(P), refine_iterations)
    rows_i = [i for i, _ in free]
    var_k = [k for _, k in free]
    Gs = G[:, rows_i]
    cov = Gs @ C[np.ix_(var_k, var_k)] @ Gs.T
    return names, theta, cov


def _refine_lift(calib, names, theta, scale, iterations):
    """Make the lifted joint corrections reproduce the fit at the postures."""
    cfg = calib.calib_config
    model, data, q = calib.model, calib.data, calib.q_measured
    fitted = list(cfg["param_name"])
    x = np.asarray(calib.var_)
    frames = calib._frame_param_names()
    frame_values = np.array([x[fitted.index(n)] for n in frames])
    target = calc_updated_fkm(model, data, x, q, cfg)
    lifted_cfg = dict(cfg, param_name=frames + list(names))
    nf = len(frames)
    for _ in range(iterations):
        var = np.r_[frame_values, theta]
        r = calc_updated_fkm(model, data, var, q, lifted_cfg) - target
        J = measurement_jacobian(model, data, var, q, lifted_cfg)[:, nf:]
        # unobservable directions show up as finite-difference noise
        # (~1e-11 relative): truncate them instead of inverting them
        step = np.linalg.lstsq(J * scale, r, rcond=1e-6)[0]
        theta = theta - scale * step
    return theta
