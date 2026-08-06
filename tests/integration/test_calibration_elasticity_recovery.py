"""Ground-truth recovery test for calc_updated_fkm's elasticity support.

Standard calibration-algorithm validation pattern: inject known per-joint
compliance parameters, generate synthetic marker measurements through
calc_updated_fkm itself, then run the same Levenberg-Marquardt pattern used
by TiagoProCalibration.solve (figaroh-examples/examples/tiago_pro/calibration.py)
to recover them from noiseless and lightly-noisy data. This is the strongest
evidence the merged elasticity math (calc_updated_fkm + non_geom=True) is
correct end-to-end, not just per-call, following the merge of the previously
dead-code `update_forward_kinematics` into `calc_updated_fkm`.
"""

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.calibration_tools import calc_updated_fkm
from scipy.optimize import least_squares


def _make_calib_config(model, param_name, n_samples):
    j1 = model.getJointId("joint1")
    j2 = model.getJointId("joint2")
    return {
        "calib_model": "full_params",
        "base_to_ref_frame": None,
        "ref_frame": None,
        "non_geom": True,
        "NbMarkers": 1,
        "NbSample": n_samples,
        "start_frame": "base_link",
        "end_frame": "link2",
        "actJoint_idx": [j1, j2],
        "measurability": [True] * 6,
        "calibration_index": 6,
        "param_name": param_name,
    }


def _synthetic_samples(n_samples, seed=0):
    rng = np.random.default_rng(seed)
    # both joints must vary for both gravity torques to be excited (a fixed
    # q2 puts every mass on joint1's own rotation axis -- see the unit-test
    # note in test_calibration_tools.py::test_many_samples_all_populated)
    q1 = rng.uniform(-1.2, 1.2, n_samples)
    q2 = rng.uniform(-1.2, 1.2, n_samples)
    return np.column_stack([q1, q2])


class TestElasticityRecovery:
    PARAM_NAME = ["k_RX_joint1", "k_RY_joint2"]
    TRUE_COMPLIANCE = np.array([0.04, -0.025])

    def _solve(self, model, data, calib_config, PEE_measured, q):
        def cost(var):
            return PEE_measured - calc_updated_fkm(model, data, var, q, calib_config)

        result = least_squares(cost, x0=np.zeros(2), method="lm")
        return result.x

    def test_recovers_exact_compliance_noiseless(self, two_joint_urdf):
        model = pin.buildModelFromUrdf(two_joint_urdf)
        data = model.createData()
        n_samples = 40
        q = _synthetic_samples(n_samples)
        calib_config = _make_calib_config(model, self.PARAM_NAME, n_samples)

        PEE_true = calc_updated_fkm(model, data, self.TRUE_COMPLIANCE, q, calib_config)
        recovered = self._solve(model, data, calib_config, PEE_true, q)

        assert recovered == pytest.approx(self.TRUE_COMPLIANCE, abs=1e-6)

    def test_recovers_compliance_within_tolerance_with_noise(self, two_joint_urdf):
        model = pin.buildModelFromUrdf(two_joint_urdf)
        data = model.createData()
        n_samples = 60
        q = _synthetic_samples(n_samples, seed=1)
        calib_config = _make_calib_config(model, self.PARAM_NAME, n_samples)

        PEE_true = calc_updated_fkm(model, data, self.TRUE_COMPLIANCE, q, calib_config)
        rng = np.random.default_rng(2)
        PEE_noisy = PEE_true + rng.normal(0, 1e-4, size=PEE_true.shape)

        recovered = self._solve(model, data, calib_config, PEE_noisy, q)

        assert recovered == pytest.approx(self.TRUE_COMPLIANCE, abs=5e-3)
