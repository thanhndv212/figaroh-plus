"""Several points of one rigid body per sample (#119).

Synthetic TIAGo measurements of four points on the tool frame. Each point
has its own offset; the points share the base frame and the kinematic
chain. Free point offsets absorb any rotation of the tool frame itself, so
four points identify the same joint parameters as one point; they measure
them more precisely.
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
    estimate_frames_closed_form,
    random_joint_configuration,
)
from figaroh.calibration.config import unified_to_legacy_config
from figaroh.calibration.parameter import BASE_TPL

# a 6 cm square, 7 cm along the tool axis (a mocap rigid body)
POINTS = np.array(
    [
        [0.03, 0.03, 0.07],
        [-0.03, 0.03, 0.07],
        [-0.03, -0.03, 0.07],
        [0.03, -0.03, 0.07],
    ]
)
BASE = [0.4, -0.2, 0.3, 0.02, -0.01, 0.5]
NOISE = 5e-4  # m, per coordinate


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


def _config(model, n_markers, level="joint_offset"):
    marker = {
        "reference_joint": "arm_7_joint",
        "measurable_dof": [True] * 3 + [False] * 3,
    }
    unified = {
        "joints": {},
        "kinematics": {"base_frame": "universe", "tool_frame": "wrist_ft_tool_link"},
        "parameters": {"calibration_level": level},
        "measurements": {"markers": [dict(marker) for _ in range(n_markers)]},
        "data": {"source_file": "unused.csv"},
    }
    cfg = unified_to_legacy_config(_Robot(model), unified)
    cfg.update(known_baseframe=False, known_tipframe=False)
    return cfg


def _tips(n_markers):
    return [f"pEE{a}_{k + 1}" for k in range(n_markers) for a in "xyz"]


def _truth(model, cfg, n_markers, seed=0):
    """Base frame, joint errors of the prior sizes, and the points."""
    rng = np.random.default_rng(seed)
    joints = estimation.joint_candidates(model, cfg)
    names = list(BASE_TPL) + joints + _tips(n_markers)
    values = list(BASE) + [
        rng.normal(0.0, estimation.DEFAULT_PRIORS[estimation.prior_group(n, model)])
        for n in joints
    ]
    values += list(POINTS[:n_markers].reshape(-1))
    return names, np.array(values)


def _postures(model, n=40, seed=1):
    rng = np.random.default_rng(seed)
    return np.array([random_joint_configuration(model, rng) for _ in range(n)])


def _fit(model, n_markers, q, seed=0, method=None):
    """Fit noisy synthetic measurements of ``n_markers`` points."""
    cfg = _config(model, n_markers)
    if method:
        cfg["estimation"] = {"method": method}
    names, truth = _truth(model, cfg, n_markers)
    cfg["NbSample"] = len(q)
    clean = calc_updated_fkm(
        model, model.createData(), truth, q, dict(cfg, param_name=names)
    )
    noise = np.random.default_rng(100 + seed).normal(0.0, NOISE, clean.shape)
    calib = _Calib.__new__(_Calib)
    calib.model, calib.data, calib.calib_config = model, model.createData(), cfg
    calib.q_measured, calib.PEE_measured, calib.del_list_ = q, clean + noise, []
    calib.create_param_list()
    calib.solve(plotting=False, enable_logging=False)
    return calib, dict(zip(names, truth))


def test_forward_model_stacks_points_marker_major(tiago_model):
    """Each point's rows are the one-marker FK with that point's offset."""
    model, data = tiago_model, tiago_model.createData()
    q = _postures(model, n=5)
    cfg = dict(_config(model, 4), NbSample=len(q))
    names = list(BASE_TPL) + _tips(4)
    var = np.r_[BASE, POINTS.reshape(-1)]
    stacked = calc_updated_fkm(model, data, var, q, dict(cfg, param_name=names))
    one = dict(_config(model, 1), NbSample=len(q), param_name=list(BASE_TPL) + _tips(1))
    for k in range(4):
        single = calc_updated_fkm(model, data, np.r_[BASE, POINTS[k]], q, one)
        np.testing.assert_allclose(stacked[k * 3 * 5 : (k + 1) * 3 * 5], single)


def test_closed_form_recovers_base_and_points(tiago_model):
    model, data = tiago_model, tiago_model.createData()
    q = _postures(model)
    cfg = dict(_config(model, 4), NbSample=len(q))
    cfg["param_name"] = list(BASE_TPL) + _tips(4)
    var = np.r_[BASE, POINTS.reshape(-1)]
    pee = calc_updated_fkm(model, data, var, q, cfg)
    guess = estimate_frames_closed_form(model, data, q, pee, cfg)
    np.testing.assert_allclose([guess[n] for n in cfg["param_name"]], var, atol=1e-8)


@pytest.fixture(scope="module")
def fits(tiago_model):
    q = _postures(tiago_model)
    return {n: _fit(tiago_model, n, q) for n in (1, 4)}


def test_four_points_keep_the_same_joint_parameters(fits):
    joint = {
        n: [p for p in c.calib_config["param_name"] if p.startswith("offset")]
        for n, (c, _) in fits.items()
    }
    assert joint[4] == joint[1]
    # the tool-axis joint is absorbed by the point offsets in both cases
    assert "offsetRZ_arm_7_joint" in fits[4][0].calib_config["absorbed_param_name"]


def test_four_points_recover_the_truth_more_precisely(fits):
    calib, truth = fits[4]
    names = calib.calib_config["param_name"]
    std = dict(zip(names, calib.std_dev))
    joint = [n for n in names if n.startswith("offset")]
    z = [(calib.var_[names.index(n)] - truth[n]) / std[n] for n in joint]
    assert np.max(np.abs(z)) < 4.0
    np.testing.assert_allclose(
        [calib.var_[names.index(n)] for n in _tips(4)], POINTS.reshape(-1), atol=5e-3
    )
    single, _ = fits[1]
    std1 = dict(zip(single.calib_config["param_name"], single.std_dev))
    ratio = np.array([std[n] / std1[n] for n in joint])
    assert np.all(ratio < 0.75), ratio  # ~0.5 with independent noise


def test_reports_errors_per_point(fits):
    calib, _ = fits[4]
    n = calib.calib_config["NbSample"]
    assert calib._PEE_dist.shape == (4, n)
    # metrics over every point of every sample, at the noise level
    assert 0.5 * NOISE * np.sqrt(3) < calib.evaluation_metrics["rmse"] < 2 * NOISE * 3
    val = calib.results_data["validation_metrics"]
    assert val["n_markers"] == 4
    assert len(val["pos_rmse_calibrated_per_point_mm"]) == 4


def test_map_handles_several_points(tiago_model):
    calib, _ = _fit(tiago_model, 4, _postures(tiago_model), method="map")
    names = calib.calib_config["param_name"]
    assert all(t in names for t in _tips(4))
    assert calib.LM_result.success


def test_reports_print_each_point(fits, capsys):
    from figaroh.tools.report import generate_calibration_report

    calib, _ = fits[4]
    calib.print_quality_report()
    out = capsys.readouterr().out
    assert all(f"point {k} RMSE" in out for k in range(1, 5))
    html = generate_calibration_report(calib)
    assert all(f"Position RMSE, point {k}" in html for k in range(1, 5))


def test_missing_point_is_rejected_with_its_rows(tiago_model, tmp_path):
    """An occluded point (empty cell) names its rows; del_list skips them."""
    import pandas as pd

    from figaroh.calibration.data_loader import load_data

    model = tiago_model
    cfg = _config(model, 2)
    q = _postures(model, n=4)
    act = [model.names[j] for j in cfg["actJoint_idx"]]
    df = pd.DataFrame({j: q[:, model.joints[model.getJointId(j)].idx_q] for j in act})
    for k in (1, 2):
        for a in "xyz":
            df[f"{a}{k}"] = 0.1 * k
    df.loc[2, "y2"] = np.nan
    path = tmp_path / "points.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"rows \[2\]"):
        load_data(str(path), model, cfg)
    pee, q_loaded = load_data(str(path), model, cfg, del_list=[2])
    assert q_loaded.shape[0] == 3 and pee.shape == (2 * 3 * 3,)
