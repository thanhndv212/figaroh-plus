"""Weighted minimum-norm lift of structural base parameters (#111).

Synthetic TIAGo measurements; the default ``structural`` method picks one
representative per dependent group, and which one depends on the random
configurations (``random_seed``). The lift must not: two seeds give the
same joint corrections, and a URDF written from them reloads to the
calibrated forward kinematics.
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
from figaroh.tools.urdf_exporter import export_urdf

TIP = ["pEEx_1", "pEEy_1", "pEEz_1"]


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


def _fitted(model, random_seed, n=40):
    unified = {
        "joints": {},
        "kinematics": {"base_frame": "universe", "tool_frame": "wrist_ft_tool_link"},
        "parameters": {"calibration_level": "full_params"},
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
    cfg.update(known_baseframe=False, known_tipframe=False, random_seed=random_seed)
    cfg["coeff_regularize"] = 0.0
    calib = _Calib.__new__(_Calib)
    calib.model, calib.data, calib.calib_config = model, model.createData(), cfg
    rng = np.random.default_rng(0)
    q = np.array([random_joint_configuration(model, rng) for _ in range(n)])
    cfg["NbSample"] = n
    calib.q_measured = q
    calib.del_list_ = []
    names = list(BASE_TPL) + estimation.joint_candidates(model, cfg) + TIP
    truth = rng.normal(0.0, 1.0, len(names))
    for i, name in enumerate(names):
        group = estimation.prior_group(name, model)
        truth[i] *= 0.05 if group is None else estimation.DEFAULT_PRIORS[group]
    calib.PEE_measured = calc_updated_fkm(
        model, calib.data, truth, q, dict(cfg, param_name=names)
    )
    calib.create_param_list()
    calib.solve(plotting=False, enable_logging=False)
    return calib


@pytest.fixture(scope="module")
def two_fits(tiago_model):
    # seeds 0 and 2 pick different representatives for the same model (same
    # predictions); other seeds can keep a different number of parameters,
    # i.e. a different model (figaroh-plus#113), which no lift can undo
    return _fitted(tiago_model, 0), _fitted(tiago_model, 2)


def test_representatives_differ_but_lift_does_not(two_fits, tiago_model):
    a, b = two_fits
    q = a.q_measured
    pa = calc_updated_fkm(
        tiago_model, tiago_model.createData(), a.var_, q, a.calib_config
    )
    pb = calc_updated_fkm(
        tiago_model, tiago_model.createData(), b.var_, q, b.calib_config
    )
    assert np.abs(pa - pb).max() < 1e-8  # same model
    rows_a = set(a.calib_config["base_mapping_row_names_full"])
    rows_b = set(b.calib_config["base_mapping_row_names_full"])
    assert rows_a != rows_b  # different representatives
    ja, jb = a.joint_corrections(lift=True), b.joint_corrections(lift=True)
    assert set(ja) == set(jb)
    diff = max(abs(ja[n] - jb[n]) for n in ja)
    assert diff < 1e-5  # m or rad
    ra, rb = a.joint_corrections(lift=False), b.joint_corrections(lift=False)
    raw = max(abs(ra.get(n, 0.0) - rb.get(n, 0.0)) for n in set(ra) | set(rb))
    assert raw > 100 * diff  # representatives alone depend on the choice


def test_lift_round_trips_through_the_full_mapping(two_fits):
    calib = two_fits[0]
    cfg = calib.calib_config
    # the linear step; refinement then corrects its first-order error
    names, theta, _ = estimation.lift_structural(calib, refine_iterations=0)
    M = cfg["base_mapping_matrix_full"]
    rows = cfg["base_mapping_row_names_full"]
    kept = cfg["base_mapping_row_names"]
    start = cfg["base_mapping_slice"][0]
    frames = set(cfg["base_frame_row_names"])
    target = [
        0.0 if r in frames or r not in kept else calib.var_[start + kept.index(r)]
        for r in rows
    ]
    np.testing.assert_allclose(M @ theta, target, atol=1e-12)
    assert len(frames) == 6


def test_lifted_urdf_reproduces_calibrated_fk(
    two_fits, tiago_model, tiago_urdf_path, tmp_path
):
    calib = two_fits[0]
    cfg = calib.calib_config
    out = tmp_path / "lifted.urdf"
    export_urdf(
        str(tiago_urdf_path), calib.joint_corrections(lift=True), output_path=str(out)
    )
    exported = pin.buildModelFromUrdf(str(out))
    frames = [n for n in cfg["param_name"] if n.startswith("base_") or "EE" in n]
    values = dict(zip(cfg["param_name"], calib.var_))
    q = calib.q_measured
    calibrated = calc_updated_fkm(
        tiago_model, tiago_model.createData(), calib.var_, q, cfg
    )
    reloaded = calc_updated_fkm(
        exported,
        exported.createData(),
        np.array([values[n] for n in frames]),
        q,
        dict(cfg, param_name=frames),
    )
    # refined lift; the exporter writes 6 significant digits
    assert np.abs(reloaded - calibrated).max() < 1e-5
