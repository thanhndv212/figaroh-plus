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

"""Choosing which calibration parameters to estimate, and how (#113).

``calib_config["estimation"]["method"]`` selects one of:

``structural`` (default)
    The structural base parameters from random configurations, then the
    data-level elimination of frame-absorbed parameters (#102). Unchanged
    behaviour.
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

The base and tool frames are always estimated freely (no prior). Noise is
``noise_std`` in the units of the cost function's measurement residuals
(metres for position unless the robot's cost function weights them), or,
when ``None``, estimated from a free fit of the identifiable parameters.
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

from .calibration_tools import measurement_jacobian, select_identifiable_parameters
from .parameter import BASE_TPL, add_pee_name, get_fullparam_offset, get_joint_offset

logger = logging.getLogger(__name__)

METHODS = ("structural", "excitation", "map", "map_cv", "cv_subset")
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
}
_BASE_MAPPING_KEYS = (
    "base_mapping_matrix",
    "base_mapping_param_names",
    "base_mapping_row_names",
    "base_mapping_slice",
)


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
    if out["cv_rule"] not in ("min", "one_se"):
        raise EstimationError("estimation.cv_rule must be 'min' or 'one_se'")
    return out


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


def prior_std(names: Sequence[str], model, priors: Dict[str, float]) -> np.ndarray:
    """Expected size per parameter; ``inf`` for frame parameters."""
    out = []
    for n in names:
        group = prior_group(n, model)
        out.append(np.inf if group is None else float(priors[group]))
    return np.array(out)


# ── fitting helpers ─────────────────────────────────────────────


@contextmanager
def _samples(calib, idx):
    """Restrict the calibrator's measurements to sample indices ``idx``."""
    cfg = calib.calib_config
    n = cfg["NbSample"]
    pee = np.asarray(calib.PEE_measured)
    ndof = pee.size // n
    saved = (calib.q_measured, calib.PEE_measured, n)
    idx = np.asarray(idx)
    calib.q_measured = np.asarray(calib.q_measured)[idx]
    calib.PEE_measured = np.concatenate([pee[d * n + idx] for d in range(ndof)])
    cfg["NbSample"] = len(idx)
    try:
        yield
    finally:
        calib.q_measured, calib.PEE_measured, cfg["NbSample"] = saved


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


def _fit(calib, names, weights=None):
    """Least squares of the calibrator's objective over ``names``."""
    with _params(calib, names, weights):
        x0 = _start(calib, names)
        m = len(calib._objective(x0))
        method = "lm" if m >= len(names) else "trf"
        return least_squares(calib._objective, x0, method=method, max_nfev=2000)


def _measurement_rms(calib, names, x) -> float:
    with _params(calib, names):
        r = np.asarray(calib.cost_function(x))[: len(calib.PEE_measured)]
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


def identifiable(calib, names, frames, tol) -> List[str]:
    """Drop exact dependencies on the measured postures (frames kept)."""
    cfg = calib.calib_config
    x0 = _start(calib, names)
    J = measurement_jacobian(
        calib.model, calib.data, x0, calib.q_measured, dict(cfg, param_name=names)
    )
    kept, _ = select_identifiable_parameters(J, names, frames, tol=tol)
    return kept


def excitation_order(J, names, frames, prior, noise, k=None):
    """Backward elimination by predicted standard error / expected size.

    Returns (kept, removed in order, ratio at removal). With ``k`` None the
    elimination runs until only frames remain.
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
    """Set ``param_name`` and prior weights for a non-structural method."""
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
    base = [] if cfg["known_baseframe"] else list(BASE_TPL)
    cfg["param_name"] = base + candidates
    if not cfg["known_tipframe"]:
        add_pee_name(cfg)
    all_names = list(cfg["param_name"])
    for key in _BASE_MAPPING_KEYS:
        cfg.pop(key, None)
    frames = calib._frame_param_names()
    prior = prior_std(all_names, model, est["priors"])
    prior_of = dict(zip(all_names, prior))

    ident = identifiable(calib, all_names, frames, est["dependency_tol"])
    prelim = _fit(calib, ident)
    n_meas = len(calib.PEE_measured)
    noise, noise_source = est["noise_std"], "given"
    if noise is None:
        dof = n_meas - len(ident)
        if dof <= 0:
            raise EstimationError(
                "too few measurements to estimate the noise; set estimation.noise_std"
            )
        noise = float(np.sqrt(np.sum(prelim.fun[:n_meas] ** 2) / dof))
        noise_source = "estimated (free fit of the identifiable parameters)"
    report = {
        "method": est["method"],
        "candidates": candidates,
        "dependent": [n for n in all_names if n not in ident],
        "noise_std": noise,
        "noise_source": noise_source,
        "priors": est["priors"],
    }

    def weights(names, scale=None):
        scale = scale or {}
        s = np.array(
            [prior_of[n] * scale.get(prior_group(n, model), 1.0) for n in names]
        )
        return np.where(np.isfinite(s), noise / s, 0.0)

    method = est["method"]
    final, w = None, None
    if method in ("excitation", "cv_subset"):
        J = prelim.jac[:n_meas]
        prior_ident = np.array([prior_of[n] for n in ident])
        k = est["excitation_k"] if method == "excitation" else None
        kept, removed, ratios = excitation_order(
            J, ident, frames, prior_ident, noise, k
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
        scale = _choose_scale(calib, all_names, weights, est, report)
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


def _choose_scale(calib, names, weights, est, report) -> Dict[str, float]:
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
        present = {prior_group(n, calib.model) for n in names} - {None}
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
