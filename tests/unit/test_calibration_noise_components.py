"""Per-component measurement noise for the estimation methods (#215).

``estimation.noise_std`` may be one value per measurement component, given
or estimated (``per_component``). Unequal values whiten the measurement
rows, optionally in a per-sample sensor frame; a scalar, or equal values,
leave the problem exactly as before.
"""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import pinocchio as pin  # noqa: F401
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration import estimation
from figaroh.calibration.calibration_tools import calc_updated_fkm
from figaroh.utils.error_handling import CalibrationError

from test_calibration_estimation import _Calib, _calibrator

SIGMA = [2e-4, 5e-4, 2e-3]  # x, y, z


def _anisotropic(model, est, sigma=SIGMA, n=60, seed=0, cls=None):
    """TIAGo calibrator whose x/y/z measurement noise is ``sigma``."""
    calib = _calibrator(model, "joint_offset", est, n=n, noise=0.0, seed=seed, cls=cls)
    rng = np.random.default_rng(seed + 100)
    noise = rng.normal(size=(3, n)) * np.asarray(sigma)[:, None]
    calib.PEE_measured = calib.PEE_measured + noise.ravel()  # DOF-major
    return calib


def _rotations(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        q, r = np.linalg.qr(rng.normal(size=(3, 3)))
        out.append(q * np.sign(np.diag(r)))
    return np.array(out)


# ── settings ────────────────────────────────────────────────────


@pytest.mark.parametrize("bad", ["auto", [1e-3, -1e-3], [[1e-3]], []])
def test_invalid_noise_setting_is_refused(bad):
    with pytest.raises(estimation.EstimationError, match="noise_std"):
        estimation.settings({"estimation": {"method": "map", "noise_std": bad}})


def test_wrong_length_vector_is_refused(tiago_model):
    calib = _anisotropic(tiago_model, {"method": "map", "noise_std": [1e-3] * 6})
    with pytest.raises(CalibrationError, match="3 components"):
        calib.create_param_list()


# ── scalar results unchanged ────────────────────────────────────


def test_scalar_and_equal_vector_do_not_whiten(tiago_model):
    a = _anisotropic(tiago_model, {"method": "map", "noise_std": 1e-3})
    b = _anisotropic(tiago_model, {"method": "map", "noise_std": [1e-3] * 3})
    a.create_param_list()
    b.create_param_list()
    for calib in (a, b):
        assert "measurement_noise_std" not in calib.calib_config
        assert calib.calib_config["estimation_report"]["noise_std"] == 1e-3
    np.testing.assert_array_equal(
        a.calib_config["prior_weights"], b.calib_config["prior_weights"]
    )
    rows = np.arange(5.0)
    assert estimation.whiten(a, rows) is rows  # untouched, not a copy
    x = np.random.default_rng(1).normal(
        scale=1e-3, size=len(a.calib_config["param_name"])
    )
    np.testing.assert_array_equal(a._objective(x), b._objective(x))


# ── given vector: the explicitly whitened problem ───────────────


class _Whitened(_Calib):
    """The same measurements, rows scaled by hand to the smallest sigma."""

    def cost_function(self, var):
        pee = calc_updated_fkm(
            self.model, self.data, var, self.q_measured, self.calib_config
        )
        scale = np.repeat(min(SIGMA) / np.asarray(SIGMA), self.calib_config["NbSample"])
        return (pee - self.PEE_measured) * scale


@pytest.mark.parametrize("method", ["map", "excitation"])
def test_vector_noise_equals_the_whitened_problem(tiago_model, method):
    a = _anisotropic(tiago_model, {"method": method, "noise_std": SIGMA})
    b = _anisotropic(
        tiago_model, {"method": method, "noise_std": min(SIGMA)}, cls=_Whitened
    )
    a.create_param_list()
    b.create_param_list()
    ra, rb = a.calib_config["estimation_report"], b.calib_config["estimation_report"]
    assert ra["noise_std"] == SIGMA and ra["noise_std_ref"] == min(SIGMA)
    assert ra["noise_rotated"] is False
    assert ra["param_name"] == rb["param_name"]
    if method == "excitation":
        assert [n for n, _ in ra["removal_order"]] == [
            n for n, _ in rb["removal_order"]
        ]
        np.testing.assert_allclose(
            [r for _, r in ra["removal_order"]], [r for _, r in rb["removal_order"]]
        )
    else:
        np.testing.assert_array_equal(
            a.calib_config["prior_weights"], b.calib_config["prior_weights"]
        )
    x = np.random.default_rng(2).normal(scale=1e-3, size=len(ra["param_name"]))
    np.testing.assert_allclose(a._objective(x), b._objective(x), rtol=1e-12)


# ── estimated per component ─────────────────────────────────────


def test_component_noise_is_the_rms_per_component():
    n = 50
    r = np.random.default_rng(3).normal(size=3 * n) * np.repeat([1.0, 2.0, 5.0], n)
    calib = SimpleNamespace(calib_config={"NbSample": n}, PEE_measured=np.zeros(3 * n))
    rms = np.sqrt(np.mean(r.reshape(3, n) ** 2, axis=1))
    np.testing.assert_allclose(estimation.component_noise(calib, r, 0), rms)
    # the parameters' degrees of freedom are shared by the components
    np.testing.assert_allclose(
        estimation.component_noise(calib, r, 30), rms * np.sqrt(n / (n - 10))
    )


def test_per_component_estimate_recovers_the_noise(tiago_model):
    calib = _anisotropic(
        tiago_model, {"method": "map", "noise_std": "per_component"}, n=200
    )
    calib.create_param_list()
    report = calib.calib_config["estimation_report"]
    np.testing.assert_allclose(report["noise_std"], SIGMA, rtol=0.15)
    assert report["noise_std_ref"] == min(report["noise_std"])
    assert "noise plus model error" in report["noise_source"]
    assert calib.calib_config["prior_noise"] == report["noise_std_ref"]


def test_per_component_needs_the_free_fit(tiago_model):
    calib = _anisotropic(
        tiago_model, {"method": "map", "noise_std": "per_component"}, n=4
    )
    with pytest.raises(CalibrationError, match="noise"):
        calib.create_param_list()


# ── sensor-frame rotation ───────────────────────────────────────


def test_rotated_covariance_matches_the_hand_built_one():
    n = 7
    sigma = np.array(SIGMA)
    R = _rotations(n)
    calib = SimpleNamespace(
        calib_config={
            "NbSample": n,
            "measurement_noise_std": list(sigma),
            "measurement_noise_ref": sigma.min(),
        },
        PEE_measured=np.zeros(3 * n),
        noise_rotation=R,
    )
    r = np.random.default_rng(4).normal(size=3 * n)
    extra = np.array([7.0, 8.0])  # rows after the measurements are kept
    w = estimation.whiten(calib, np.r_[r, extra])
    np.testing.assert_array_equal(w[3 * n :], extra)
    per_sample_r, per_sample_w = r.reshape(3, n).T, w[: 3 * n].reshape(3, n).T
    for i in range(n):
        cov = R[i] @ np.diag(sigma**2) @ R[i].T
        expected = (
            sigma.min() ** 2 * per_sample_r[i] @ np.linalg.solve(cov, per_sample_r[i])
        )
        assert per_sample_w[i] @ per_sample_w[i] == pytest.approx(expected, rel=1e-12)
    # a Jacobian is whitened column by column
    J = np.random.default_rng(5).normal(size=(3 * n, 4))
    Jw = estimation.whiten(calib, J)
    for k in range(4):
        np.testing.assert_allclose(Jw[:, k], estimation.whiten(calib, J[:, k]))


def test_rotation_is_validated_and_sliced(tiago_model):
    calib = _anisotropic(tiago_model, {"method": "map", "noise_std": SIGMA}, n=20)
    calib.noise_rotation = _rotations(19)
    with pytest.raises(CalibrationError, match="noise_rotation"):
        calib.create_param_list()
    calib.noise_rotation = _rotations(20)
    with estimation._samples(calib, np.arange(5)):
        np.testing.assert_array_equal(calib.noise_rotation, _rotations(20)[:5])
    assert len(calib.noise_rotation) == 20
    calib.create_param_list()
    assert calib.calib_config["estimation_report"]["noise_rotated"] is True
