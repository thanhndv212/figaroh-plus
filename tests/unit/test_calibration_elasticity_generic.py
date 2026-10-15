"""Joint elasticity on robots other than the fixed-base test fixtures (#206).

The elastic path (``include_non_geometric``) was only exercised on chains
whose joints are aligned, fixed-base and distinctly named. These tests pin
the four robot-specific defects fixed in #206:

- ``add_base_name`` renamed ``param_name[0:6]``, the elastic ``k_*`` names,
  instead of the first structural parameters;
- the gravity torque was read at ``j_id - 1`` instead of the joint's velocity
  index (wrong after a free-flyer);
- parameter names were matched by substring (``joint1`` took ``joint10``'s);
- every unaligned revolute joint was given the ``RZ`` axis.
"""

from types import SimpleNamespace

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

from figaroh.calibration.calibration_tools import (
    apply_joint_offset,
    calc_updated_fkm,
    get_rel_transform,
    update_joint_placement,
)
from figaroh.calibration.config import (
    _build_elastic_param_names,
    _determine_axis_motion,
)
from figaroh.calibration.parameter import BASE_TPL, add_base_name


def _chain(names, joint_models=None, free_flyer=False):
    """Serial chain of massive links, one joint per name, with a ``tip``."""
    model = pin.Model()
    parent = 0
    if free_flyer:
        parent = model.addJoint(
            0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root"
        )
        model.appendBodyToJoint(
            parent, pin.Inertia.FromBox(2.0, 0.2, 0.2, 0.2), pin.SE3.Identity()
        )
    for i, name in enumerate(names):
        joint = joint_models[i] if joint_models else pin.JointModelRY()
        placement = pin.SE3(np.eye(3), np.array([0.0, 0.0, 0.2]))
        parent = model.addJoint(parent, joint, placement, name)
        com = pin.SE3(np.eye(3), np.array([0.15, 0.02, 0.1]))
        model.appendBodyToJoint(parent, pin.Inertia.FromSphere(1.0, 0.05), com)
        model.addJointFrame(parent)
    model.addFrame(
        pin.Frame(
            "tip",
            parent,
            model.getFrameId(names[-1]),
            pin.SE3(np.eye(3), np.array([0.3, 0.0, 0.0])),
            pin.FrameType.OP_FRAME,
        )
    )
    return model


def _config(model, param_name, joints, n_samples=1):
    return {
        "calib_model": "full_params",
        "base_to_ref_frame": None,
        "ref_frame": None,
        "non_geom": any(n.startswith("k_") for n in param_name),
        "NbMarkers": 1,
        "NbSample": n_samples,
        "start_frame": "universe",
        "end_frame": "tip",
        "actJoint_idx": [model.getJointId(j) for j in joints],
        "measurability": [True] * 6,
        "calibration_index": 6,
        "param_name": param_name,
    }


def _pose(model, q, start="universe", end="tip"):
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    T = get_rel_transform(model, data, start, end)
    return np.concatenate([T.translation, pin.rpy.matrixToRpy(T.rotation)])


class TestAddBaseNameWithElasticity:
    def test_renames_structural_parameters_not_compliances(self):
        k = ["k_RZ_a", "k_RY_b"]
        structural = [f"p{i}" for i in range(8)]
        cfg = {
            "calib_model": "full_params",
            "param_name": k + structural,
            "base_mapping_slice": (2, 10),
        }
        add_base_name(cfg)
        assert cfg["param_name"][:2] == k
        assert cfg["param_name"][2:8] == list(BASE_TPL)
        assert cfg["param_name"][8:] == structural[6:]

    def test_rigid_renames_from_the_start(self):
        cfg = {"calib_model": "full_params", "param_name": [f"p{i}" for i in range(8)]}
        add_base_name(cfg)
        assert cfg["param_name"] == list(BASE_TPL) + ["p6", "p7"]


class TestFreeFlyerTorque:
    def test_deflection_uses_the_joint_velocity_index(self):
        model = _chain(["joint1", "joint2"], free_flyer=True)
        j2 = model.getJointId("joint2")
        assert model.joints[j2].idx_v != j2 - 1  # the defect's precondition

        q = pin.neutral(model)
        q[[7, 8]] = [0.4, -0.7]
        compliance = 0.05
        cfg = _config(model, ["k_RY_joint2"], ["joint2"])
        pee = calc_updated_fkm(
            model, model.createData(), np.array([compliance]), q[None, :], cfg
        )

        tau = pin.computeGeneralizedGravity(model, model.createData(), q)
        deflected = apply_joint_offset(
            model.copy(),
            j2,
            np.r_[0, 0, 0, 0, compliance * tau[model.joints[j2].idx_v], 0],
        )
        assert pee == pytest.approx(_pose(deflected, q), abs=1e-9)
        assert pee != pytest.approx(_pose(model, q), abs=1e-6)


class TestExactNameMatching:
    """``joint1`` is a prefix of ``joint10``: neither may take the other's."""

    NAMES = ["joint1", "joint10"]

    def test_geometric_parameter_moves_only_its_joint(self):
        model = _chain(self.NAMES)
        j10 = model.getJointId("joint10")
        names = [f"d_{a}_{j}" for j in self.NAMES for a in ("px", "pz")]
        var = np.zeros(len(names))
        var[names.index("d_px_joint10")] = 0.01
        q = np.array([0.3, -0.5])
        pee = calc_updated_fkm(
            model,
            model.createData(),
            var,
            q[None, :],
            _config(model, names, self.NAMES),
        )
        moved = update_joint_placement(model.copy(), j10, np.r_[0.01, 0, 0, 0, 0, 0])
        assert pee == pytest.approx(_pose(moved, q), abs=1e-9)

    def test_compliance_deflects_only_its_joint(self):
        model = _chain(self.NAMES)
        j1, j10 = (model.getJointId(n) for n in self.NAMES)
        names = ["k_RY_joint1", "k_RY_joint10"]
        q = np.array([0.3, -0.5])
        pee = calc_updated_fkm(
            model,
            model.createData(),
            np.array([0.0, 0.05]),
            q[None, :],
            _config(model, names, self.NAMES),
        )
        tau = pin.computeGeneralizedGravity(model, model.createData(), q)
        deflected = apply_joint_offset(
            model.copy(), j10, np.r_[0, 0, 0, 0, 0.05 * tau[model.joints[j10].idx_v], 0]
        )
        assert pee == pytest.approx(_pose(deflected, q), abs=1e-9)
        assert j1 != j10


class TestElasticAxis:
    @pytest.mark.parametrize(
        "joint, expected",
        [
            (pin.JointModelRX(), "RX"),
            (pin.JointModelRUBZ(), "RZ"),
            (pin.JointModelPY(), "PY"),
            (pin.JointModelRevoluteUnaligned(np.array([0.0, 0.0, -1.0])), "RZ"),
            (pin.JointModelRevoluteUnaligned(np.array([0.0, 1.0, 0.0])), "RY"),
            (pin.JointModelPrismaticUnaligned(np.array([-1.0, 0.0, 0.0])), "PX"),
        ],
    )
    def test_coordinate_axes_are_named(self, joint, expected):
        assert _determine_axis_motion(joint) == expected

    @pytest.mark.parametrize(
        "joint",
        [
            pin.JointModelRevoluteUnaligned(np.array([0.6, 0.8, 0.0])),
            pin.JointModelSpherical(),
            pin.JointModelFreeFlyer(),
        ],
    )
    def test_other_joints_are_refused(self, joint):
        with pytest.raises(ValueError, match="joint elasticity"):
            _determine_axis_motion(joint)

    def test_only_active_joints_are_checked(self):
        tilted = pin.JointModelRevoluteUnaligned(np.array([0.6, 0.8, 0.0]))
        down = pin.JointModelRevoluteUnaligned(np.array([0.0, 0.0, -1.0]))
        model = _chain(["a", "b", "c"], [tilted, down, pin.JointModelRX()])
        robot = SimpleNamespace(model=model)
        active = [model.getJointId("b"), model.getJointId("c")]
        assert _build_elastic_param_names(robot, active) == ["k_RZ_b", "k_RX_c"]
        with pytest.raises(ValueError, match="not a coordinate axis"):
            _build_elastic_param_names(robot, [model.getJointId("a")])


class TestCompliancePhysicalSign:
    """k is the physical compliance: the joint moves q -> q + k * tau,
    whichever way its axis points in the joint frame."""

    HORIZONTAL_Z = pin.SE3(pin.rpy.rpyToMatrix(np.pi / 2, 0.0, 0.0), np.zeros(3))

    @pytest.mark.parametrize(
        "joint, placement, name",
        [
            (pin.JointModelRZ(), HORIZONTAL_Z, "k_RZ_j"),
            (
                pin.JointModelRevoluteUnaligned(np.array([0.0, 0.0, -1.0])),
                HORIZONTAL_Z,
                "k_RZ_j",
            ),
            (pin.JointModelRY(), pin.SE3.Identity(), "k_RY_j"),
            (
                pin.JointModelRevoluteUnaligned(np.array([0.0, -1.0, 0.0])),
                pin.SE3.Identity(),
                "k_RY_j",
            ),
            (pin.JointModelPZ(), pin.SE3.Identity(), "k_PZ_j"),
            (
                pin.JointModelPrismaticUnaligned(np.array([0.0, 0.0, -1.0])),
                pin.SE3.Identity(),
                "k_PZ_j",
            ),
        ],
    )
    def test_deflection_is_compliance_times_torque_along_the_axis(
        self, joint, placement, name
    ):
        model = pin.Model()
        j = model.addJoint(0, joint, placement, "j")
        com = pin.SE3(np.eye(3), np.array([0.2, 0.05, 0.1]))
        model.appendBodyToJoint(j, pin.Inertia.FromSphere(1.5, 0.05), com)
        model.addJointFrame(j)
        model.addFrame(
            pin.Frame(
                "tip",
                j,
                model.getFrameId("j"),
                pin.SE3(np.eye(3), np.array([0.3, 0.1, 0.0])),
                pin.FrameType.OP_FRAME,
            )
        )
        q = np.array([0.4])
        k = 0.02
        tau = pin.computeGeneralizedGravity(model, model.createData(), q)
        assert abs(tau[0]) > 0.1  # the joint is loaded

        pee = calc_updated_fkm(
            model,
            model.createData(),
            np.array([k]),
            q[None, :],
            _config(model, [name], ["j"]),
        )
        assert pee == pytest.approx(_pose(model, q + k * tau), abs=1e-9)

    def test_name_off_the_joint_axis_is_refused(self):
        model = _chain(
            ["j"], [pin.JointModelRevoluteUnaligned(np.array([0.0, 0.0, -1.0]))]
        )
        with pytest.raises(ValueError, match="not the axis"):
            calc_updated_fkm(
                model,
                model.createData(),
                np.array([0.01]),
                np.array([[0.3]]),
                _config(model, ["k_RX_j"], ["j"]),
            )
