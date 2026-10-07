"""Calibration estimation methods (#113).

Synthetic TIAGo measurements from known joint errors; each method must keep
the parameter layout the robot cost functions rely on (base frame first,
tool point last), estimate the frames freely, and behave as documented.
"""

import numpy as np
import pytest

try:
    import pinocchio as pin  # noqa: F401
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration import estimation
from figaroh.calibration.base_calibration import BaseCalibration
from figaroh.calibration.calibration_tools import (
    calc_updated_fkm,
    random_joint_configuration,
)
from figaroh.calibration.config import unified_to_legacy_config
from figaroh.calibration.parameter import BASE_TPL
from figaroh.utils.error_handling import CalibrationError

TIP = ["pEEx_1", "pEEy_1", "pEEz_1"]
FRAMES = {
    "base_px": 0.01,
    "base_py": 0.2,
    "base_pz": -0.3,
    "base_phix": 0.01,
    "base_phiy": 0.005,
    "base_phiz": -0.003,
    "pEEx_1": 0.06,
    "pEEy_1": -0.003,
    "pEEz_1": 0.07,
}


class _Robot:
    def __init__(self, model):
        self.model = model
        self.data = model.createData()
        self.q0 = pin.neutral(model)


class _Calib(BaseCalibration):
    def cost_function(self, var):
        pee = calc_updated_fkm(
            self.model, self.data, var, self.q_measured, self.calib_config
        )
        return pee - self.PEE_measured


def _calibrator(model, level, estimation_cfg, n=40, noise=2e-4, seed=0):
    unified = {
        "joints": {},
        "kinematics": {"base_frame": "universe", "tool_frame": "wrist_ft_tool_link"},
        "parameters": {"calibration_level": level, "estimation": estimation_cfg},
        "measurements": {
            "markers": [
                {
                    "reference_joint": "arm_7_joint",
                    "measurable_dof": [True] * 3 + [False] * 3,
                }
            ]
        },
        "data": {"source_file": "unused.csv"},
    }
    cfg = unified_to_legacy_config(_Robot(model), unified)
    cfg["known_baseframe"] = False
    cfg["known_tipframe"] = False
    calib = _Calib.__new__(_Calib)
    calib.model, calib.data, calib.calib_config = model, model.createData(), cfg
    rng = np.random.default_rng(seed)
    q = np.array([random_joint_configuration(model, rng) for _ in range(n)])
    cfg["NbSample"] = n
    calib.q_measured = q
    truth = dict(FRAMES)
    for name in estimation.joint_candidates(model, cfg):
        group = estimation.prior_group(name, model)
        truth[name] = rng.normal(0.0, estimation.DEFAULT_PRIORS[group])
    names = list(BASE_TPL) + estimation.joint_candidates(model, cfg) + TIP
    values = np.array([truth[k] for k in names])
    true_cfg = dict(cfg, param_name=names)
    pee = calc_updated_fkm(model, calib.data, values, q, true_cfg)
    calib.PEE_measured = pee + rng.normal(0.0, noise, pee.shape)
    calib.truth = truth
    calib.del_list_ = []
    return calib


def _layout_ok(names):
    assert names[:6] == list(BASE_TPL)
    assert names[-3:] == TIP


# ── settings and priors ─────────────────────────────────────────


def test_settings_merge_defaults_and_reject_unknown():
    s = estimation.settings(
        {"estimation": {"method": "map", "priors": {"rotation": 1e-3}}}
    )
    assert s["priors"]["rotation"] == 1e-3
    assert s["priors"]["translation"] == estimation.DEFAULT_PRIORS["translation"]
    assert estimation.settings({})["method"] == "structural"
    with pytest.raises(estimation.EstimationError):
        estimation.settings({"estimation": {"method": "lasso"}})
    with pytest.raises(estimation.EstimationError):
        estimation.settings({"estimation": {"priors": {"wrist": 1.0}}})


def test_prior_groups_follow_joint_axes(tiago_model):
    g = lambda n: estimation.prior_group(n, tiago_model)  # noqa: E731
    assert g("d_phiz_arm_2_joint") == "joint_offset"  # revolute about z
    assert g("d_phix_arm_2_joint") == "rotation"
    assert g("d_pz_torso_lift_joint") == "prismatic_offset"  # prismatic along z
    assert g("d_px_torso_lift_joint") == "translation"
    assert g("offsetRZ_arm_5_joint") == "joint_offset"
    assert g("offsetPZ_torso_lift_joint") == "prismatic_offset"
    assert g("base_px") is None and g("pEEx_1") is None


def test_candidates_cover_the_active_joints(tiago_model):
    calib = _calibrator(tiago_model, "joint_offset", {"method": "map"}, n=5)
    names = estimation.joint_candidates(tiago_model, calib.calib_config)
    assert len(names) == 8
    assert names[0] == "offsetPZ_torso_lift_joint"
    full = _calibrator(tiago_model, "full_params", {"method": "map"}, n=5)
    assert len(estimation.joint_candidates(tiago_model, full.calib_config)) == 48


# ── methods ─────────────────────────────────────────────────────


def test_excitation_keeps_frames_and_layout(tiago_model):
    calib = _calibrator(tiago_model, "full_params", {"method": "excitation"})
    calib.create_param_list()
    cfg = calib.calib_config
    _layout_ok(cfg["param_name"])
    report = cfg["estimation_report"]
    assert report["method"] == "excitation"
    # every removal was beyond k, every kept joint parameter within it
    assert all(r > 1.0 for _, r in report["removal_order"])
    kept_joints = [n for n in cfg["param_name"] if n.startswith("d_")]
    assert 0 < len(kept_joints) < 48
    assert set(report["dependent"]).isdisjoint(cfg["param_name"])


def test_map_estimates_everything_and_shrinks_unexcited(tiago_model):
    calib = _calibrator(tiago_model, "full_params", {"method": "map"})
    calib.create_param_list()
    cfg = calib.calib_config
    assert len(cfg["param_name"]) == 6 + 48 + 3
    w = cfg["prior_weights"]
    assert np.all(w[:6] == 0) and np.all(w[-3:] == 0) and np.all(w[6:-3] > 0)
    calib.solve(plotting=False, enable_logging=False)
    std = dict(zip(cfg["param_name"], calib.std_dev))

    def prior(n):
        return estimation.prior_std([n], tiago_model, estimation.DEFAULT_PRIORS)[0]

    # absorbed entirely by the free tool point: the data says nothing about
    # it, so its posterior equals its prior
    assert "d_pz_arm_7_joint" in cfg["estimation_report"]["dependent"]
    assert std["d_pz_arm_7_joint"] == pytest.approx(prior("d_pz_arm_7_joint"), rel=0.02)
    # no parameter is less certain than its prior
    for n in cfg["param_name"][6:-3]:
        assert std[n] <= prior(n) * 1.001
    redistributed = calib.redistribute_parameters()
    assert set(redistributed) == set(cfg["estimation_report"]["candidates"])


def test_map_prior_size_controls_shrinkage(tiago_model):
    tiny = {
        "translation": 1e-9,
        "rotation": 1e-9,
        "joint_offset": 1e-9,
        "prismatic_offset": 1e-9,
    }
    calib = _calibrator(tiago_model, "joint_offset", {"method": "map", "priors": tiny})
    calib.create_param_list()
    calib.solve(plotting=False, enable_logging=False)
    joints = np.asarray(calib.LM_result.x[6:-3])
    assert np.abs(joints).max() < 1e-6


def test_map_cv_is_deterministic_and_reports_its_curve(tiago_model):
    est = {"method": "map_cv", "cv_folds": 4, "cv_multipliers": [0.1, 1.0, 10.0]}
    a = _calibrator(tiago_model, "joint_offset", est)
    a.create_param_list()
    b = _calibrator(tiago_model, "joint_offset", est)
    b.create_param_list()
    ra, rb = a.calib_config["estimation_report"], b.calib_config["estimation_report"]
    assert ra["prior_scale"] == rb["prior_scale"]
    assert set(ra["prior_scale"].values()) <= {0.1, 1.0, 10.0}
    assert len(ra["cv_curve"]) == 3
    np.testing.assert_array_equal(
        a.calib_config["prior_weights"], b.calib_config["prior_weights"]
    )


def test_cv_subset_rules(tiago_model):
    est = {"method": "cv_subset", "cv_folds": 4}
    a = _calibrator(tiago_model, "joint_offset", est)
    a.create_param_list()
    b = _calibrator(tiago_model, "joint_offset", dict(est, cv_rule="one_se"))
    b.create_param_list()
    ra, rb = a.calib_config["estimation_report"], b.calib_config["estimation_report"]
    n_removable = len(ra["removal_order"])
    assert [c["n_joint"] for c in ra["cv_curve"]] == list(range(n_removable + 1))
    assert rb["n_joint_chosen"] <= ra["n_joint_chosen"]
    _layout_ok(a.calib_config["param_name"])


def test_given_noise_is_used(tiago_model):
    calib = _calibrator(
        tiago_model, "joint_offset", {"method": "map", "noise_std": 1e-3}
    )
    calib.create_param_list()
    report = calib.calib_config["estimation_report"]
    assert report["noise_std"] == 1e-3 and report["noise_source"] == "given"


def test_non_structural_methods_need_data(tiago_model):
    calib = _calibrator(tiago_model, "joint_offset", {"method": "map"})
    del calib.PEE_measured
    with pytest.raises(CalibrationError, match="load the data"):
        calib.create_param_list()


def test_structural_default_unchanged(tiago_model):
    calib = _calibrator(tiago_model, "joint_offset", {})
    calib.create_param_list()
    cfg = calib.calib_config
    assert "estimation_report" not in cfg
    assert cfg.get("prior_weights") is None
    assert "base_mapping_matrix" in cfg


# ── determinism of the selection (#113) ─────────────────────────


def _selected(model, perturb=0.0, order=None):
    calib = _calibrator(model, "full_params", {"method": "excitation"})
    if order is not None:
        n = calib.calib_config["NbSample"]
        pee = calib.PEE_measured.reshape(-1, n)
        calib.q_measured = calib.q_measured[order]
        calib.PEE_measured = pee[:, order].ravel()
    if perturb:
        rng = np.random.default_rng(7)
        calib.PEE_measured = calib.PEE_measured + perturb * rng.standard_normal(
            calib.PEE_measured.shape
        )
    calib.create_param_list()
    report = calib.calib_config["estimation_report"]
    return list(calib.calib_config["param_name"]), [
        n for n, _ in report["removal_order"]
    ]


def test_excitation_selection_is_stable_to_noise_and_sample_order(tiago_model):
    """The selected set must not hinge on floating-point details, unlike the
    structural QR's tie-breaks (figaroh-plus#113)."""
    ref = _selected(tiago_model)
    assert _selected(tiago_model, perturb=1e-12) == ref
    order = np.random.default_rng(3).permutation(40)
    assert _selected(tiago_model, order=order) == ref


def test_excitation_order_ignores_column_order(tiago_model):
    """Permuting the candidate columns gives the same removals and kept set."""
    calib = _calibrator(tiago_model, "joint_offset", {"method": "excitation"})
    calib.create_param_list()
    names = list(calib.calib_config["param_name"])
    frames = calib._frame_param_names()
    x0 = np.zeros(len(names))
    from figaroh.calibration.calibration_tools import measurement_jacobian

    J = measurement_jacobian(
        tiago_model, calib.data, x0, calib.q_measured, calib.calib_config
    )
    prior = estimation.prior_std(names, tiago_model, estimation.DEFAULT_PRIORS)
    kept, removed, _ = estimation.excitation_order(J, names, frames, prior, 1e-3, k=0.5)
    perm = np.random.default_rng(1).permutation(len(names))
    names_p = [names[i] for i in perm]
    kept_p, removed_p, _ = estimation.excitation_order(
        J[:, perm], names_p, frames, prior[perm], 1e-3, k=0.5
    )
    assert removed_p == removed
    assert set(kept_p) == set(kept)
