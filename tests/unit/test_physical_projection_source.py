"""Regression tests for #163 (projection source) and #164 (picos timelimit)."""

import os
import sys

import numpy as np
import pytest

from figaroh.identification.physical_consistency import project_p10_lmi
from figaroh.identification.reconstruction import reconstruct_full_parameters

INERTIAL = ("m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz")


# -- #164: max_seconds reaches picos as timelimit --


def test_lmi_projection_accepts_max_seconds():
    pytest.importorskip("picos")
    # Positive mass, indefinite rotational inertia: needs a real solve
    p10 = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, -1.0])
    _, report = project_p10_lmi(p10, max_seconds=60)
    assert report.status == "projected", report.message


def test_sdp_reconstruction_accepts_max_seconds():
    pytest.importorskip("picos")
    params_r = [f"{k}_j1" for k in INERTIAL]
    M = np.eye(10)[:4]  # mass and first moments fixed by the base fit
    phi = np.array([2.0, 0.1, 0.0, 0.0])
    prior = dict(zip(params_r, [1.0, 0, 0, 0, 1.0, 0, 1.0, 0, 0, 1.0]))
    res = reconstruct_full_parameters(
        (M, phi, params_r),
        method="sdp",
        params_std_prior=prior,
        joint_names=["j1"],
        max_seconds=60,
    )
    assert res.status == "ok"
    assert res.effective_method == "sdp"


# -- #163: the projection reads the fit, never the nominal prior --

try:
    import pinocchio as pin
except ImportError:  # pragma: no cover
    pin = None

needs_pin = pytest.mark.skipif(pin is None, reason="Pinocchio not available")


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


@pytest.fixture(scope="module")
def traj(model):
    sys.path.insert(0, os.path.dirname(__file__))
    from test_data_contract_wiring import _trajectory

    return _trajectory(model)


def _run(model, traj, **cfg):
    from test_data_contract_wiring import _Ident

    ident = _Ident(model)
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.identif_config.update(cfg)
    ident.solve(decimate=False, plotting=False)
    return ident


def _stage(ident, name):
    return next(s for s in ident.result["stages"] if s["stage"] == name)


@needs_pin
def test_without_reconstruction_projection_is_skipped(model, traj):
    ident = _run(model, traj, physical_consistency={"enabled": True})
    pc = ident.result["physical consistency"]
    assert pc["status"] == "skipped"
    assert "projected_parameters" not in pc
    assert _stage(ident, "physical")["status"] == "not_run"


@needs_pin
def test_projection_source_is_the_reconstructed_fit(model, traj):
    ident = _run(
        model,
        traj,
        reconstruction={"enabled": True, "method": "nullspace"},
        physical_consistency={"enabled": True},
    )
    pc = ident.result["physical consistency"]
    assert pc["source"] == "reconstruction"
    fit = ident.result["reconstruction"]["theta_r_dict"]
    for name, value in fit.items():
        assert pc["raw_parameters"][name] == pytest.approx(value)
    # The fit is not the prior, so the projection cannot be of the prior
    prior = ident.standard_parameter
    assert any(
        not np.isclose(pc["raw_parameters"][n], prior[n]) for n in fit if n in prior
    )
