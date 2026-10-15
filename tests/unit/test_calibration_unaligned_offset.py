"""Joint offsets on unaligned revolute joints (#205).

``get_joint_offset`` renamed ``RevoluteUnaligned`` to ``RZ`` only for the
"canopies" model, so on any other robot the offset was never applied and
``calc_updated_fkm`` failed its "all parameters updated" check. An offset
must act about the joint's own axis: offset ``d`` is the configuration
``q + d``, also when the axis is a negative coordinate axis.
"""

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.calibration_tools import calc_updated_fkm
from figaroh.calibration.parameter import get_joint_offset


def _chain(joint_models):
    """Serial chain of joints ``j0``, ``j1``, ... with a ``tip`` frame."""
    model = pin.Model()
    model.name = "not_canopies"
    parent = 0
    for i, joint in enumerate(joint_models):
        placement = pin.SE3(np.eye(3), np.array([0.1, 0.0, 0.2]))
        parent = model.addJoint(parent, joint, placement, f"j{i}")
        model.appendBodyToJoint(
            parent, pin.Inertia.FromSphere(1.0, 0.05), pin.SE3.Identity()
        )
        model.addJointFrame(parent)
    model.addFrame(
        pin.Frame(
            "tip",
            parent,
            model.getFrameId(f"j{len(joint_models) - 1}"),
            pin.SE3(np.eye(3), np.array([0.3, 0.1, 0.0])),
            pin.FrameType.OP_FRAME,
        )
    )
    return model


def _config(model, param_name, n_samples):
    return {
        "calib_model": "joint_offset",
        "base_to_ref_frame": None,
        "ref_frame": None,
        "non_geom": False,
        "NbMarkers": 1,
        "NbSample": n_samples,
        "start_frame": "universe",
        "end_frame": "tip",
        "actJoint_idx": list(range(1, model.njoints)),
        "measurability": [True] * 6,
        "calibration_index": 6,
        "param_name": param_name,
    }


def _unaligned(axis):
    return pin.JointModelRevoluteUnaligned(np.array(axis, dtype=float))


class TestNaming:
    def test_coordinate_axis_is_named_after_it(self):
        model = _chain(
            [pin.JointModelRX(), _unaligned([0, 0, -1]), _unaligned([0, 1, 0])]
        )
        names = list(get_joint_offset(model, model.names[1:]))
        assert names == ["offsetRX_j0", "offsetRZ_j1", "offsetRY_j2"]

    def test_tilted_axis_keeps_its_name(self):
        model = _chain([_unaligned([0.6, 0.8, 0.0])])
        assert list(get_joint_offset(model, model.names[1:])) == [
            "offsetRevoluteUnaligned_j0"
        ]


class TestOffsetActsAboutTheJointAxis:
    @pytest.mark.parametrize("axis", [[0, 0, -1], [0, 0, 1], [0, -1, 0], [1, 0, 0]])
    def test_offset_equals_shifted_configuration(self, axis):
        model = _chain([pin.JointModelRY(), _unaligned(axis), pin.JointModelRZ()])
        names = list(get_joint_offset(model, model.names[1:]))
        offsets = np.array([0.03, -0.07, 0.02])
        rng = np.random.default_rng(205)
        q = rng.uniform(-1.0, 1.0, size=(4, model.nq))

        cfg = _config(model, names, len(q))
        data = model.createData()
        with_offset = calc_updated_fkm(model, data, offsets, q, cfg)

        nominal = _config(model, [], len(q))
        shifted = calc_updated_fkm(model, data, np.array([]), q + offsets, nominal)
        np.testing.assert_allclose(with_offset, shifted, atol=1e-12)

    def test_prismatic_offset_follows_a_negative_axis(self):
        slide = pin.JointModelPrismaticUnaligned(np.array([-1.0, 0.0, 0.0]))
        model = _chain([pin.JointModelRY(), slide])
        names = list(get_joint_offset(model, model.names[1:]))
        assert names == ["offsetRY_j0", "offsetPX_j1"]
        offsets = np.array([0.0, 0.05])
        q = np.array([[0.3, 0.1]])
        data = model.createData()
        with_offset = calc_updated_fkm(
            model, data, offsets, q, _config(model, names, 1)
        )
        shifted = calc_updated_fkm(
            model, data, np.array([]), q + offsets, _config(model, [], 1)
        )
        np.testing.assert_allclose(with_offset, shifted, atol=1e-12)

    def test_model_is_restored(self):
        model = _chain([_unaligned([0, 0, -1])])
        before = model.jointPlacements[1].copy()
        cfg = _config(model, ["offsetRZ_j0"], 1)
        calc_updated_fkm(
            model, model.createData(), np.array([0.1]), np.zeros((1, 1)), cfg
        )
        assert model.jointPlacements[1].isApprox(before)

    def test_tilted_axis_is_refused(self):
        model = _chain([_unaligned([0.6, 0.8, 0.0])])
        cfg = _config(model, ["offsetRevoluteUnaligned_j0"], 1)
        with pytest.raises(ValueError, match="not a coordinate axis"):
            calc_updated_fkm(
                model, model.createData(), np.array([0.1]), np.zeros((1, 1)), cfg
            )
