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

"""PAL Robotics ``robot_state_publisher`` geometric-calibration deploy file.

Produces the runtime joint-correction YAML PAL robots (TIAGo, TIAGo Pro,
TALOS, ...) read at ``/etc/calibration/master_calibration.yaml``
(``robot_state_publisher: geometric_calibration: {<joint>_<axis>: value}``),
directly from a solved ``BaseCalibration``.

This is a different deploy target from
:mod:`figaroh.tools.urdf_exporter`: that module bakes corrections into a
*modified URDF file*; this one produces a small *runtime correction
overlay* PAL's ``robot_state_publisher`` applies on top of the original,
unmodified URDF at startup. Both read the same ``d_px_{joint}``-style
(``full_params``) and ``offsetRZ_{joint}``-style (``joint_offset``)
parameter names — this module reuses
:func:`figaroh.tools.urdf_exporter._parse_param_name` rather than
re-deriving that parsing. The keys are additive deltas on the URDF
``<origin>`` xyz and rpy as written; pass ``nominal_urdf`` so they are
exact (figaroh-plus#123).

The values come from
:meth:`~figaroh.calibration.base_calibration.BaseCalibration.redistribute_parameters`:
for the ``structural`` method, the weighted minimum-norm lift of the fitted
base parameters onto every joint (figaroh-plus#111), so a joint whose
correction was represented by another parameter in the fit still gets its
share; for the other estimation methods, the fitted joint parameters. Use
:meth:`~figaroh.calibration.base_calibration.BaseCalibration.joint_corrections`
to write the same values into a URDF.
"""

import logging
from typing import Dict, Optional

import numpy as np
import pinocchio as pin
import yaml

from figaroh.tools.urdf_exporter import _parse_param_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Order matches _parse_param_name's sub_idx (0..5), which in turn matches
# FULL_PARAMTPL = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"].
_AXIS_SUFFIX = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]

_JOINT_SUFFIX = "_joint"


def _urdf_origin(model, joint: str) -> pin.SE3:
    """The joint's URDF ``<origin>``, recovered from a Pinocchio model.

    Pinocchio merges the fixed joints between the parent moving joint and
    this one into ``jointPlacements``; the URDF origin is that placement
    with the prefix (the placement of the parent link's frame) removed
    (figaroh-plus#123).
    """
    jid = model.getJointId(joint)
    placement = model.jointPlacements[jid]
    if not model.existFrame(joint):
        return placement
    parent = model.frames[model.frames[model.getFrameId(joint)].parentFrame]
    if parent.parentJoint != model.parents[jid]:
        return placement
    return parent.placement.inverse() * placement


def _urdf_origins(nominal_urdf) -> Dict[str, tuple]:
    """``{joint: (xyz, rpy)}`` as written in the URDF's ``<origin>``.

    The PAL keys add to these numbers, so the rpy triplet as written matters:
    near pitch = +-pi/2, or with |pitch| > pi/2, it differs from the one
    Pinocchio decomposes from the rotation (figaroh-plus#123).
    """
    import xml.etree.ElementTree as ET

    origins = {}
    for joint in ET.parse(str(nominal_urdf)).getroot().findall("joint"):
        origin = joint.find("origin")
        get = (
            (lambda k: origin.get(k, "0 0 0"))
            if origin is not None
            else (lambda k: "0 0 0")
        )
        origins[joint.get("name")] = (
            np.array([float(v) for v in get("xyz").split()]),
            np.array([float(v) for v in get("rpy").split()]),
        )
    return origins


def _rpy_delta(rpy: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``d`` with ``rpyToMatrix(rpy + d) == target``, the smallest found.

    Gauss-Newton on the rotation error (local frame), so it is exact for
    the rpy triplet as written, from two starts: ``d = 0`` (a calibration
    correction is small) and the difference of decomposed angles. At pitch
    = +-pi/2 a rotation about the axis roll and yaw share has no small rpy
    change; only the second start reaches it, by trading roll against yaw
    (``droll`` and ``dyaw`` of +-pi/2 or +-pi, opposite signs, small
    ``dpitch``): the same small rotation, not a large correction
    (figaroh-plus#123).
    """
    decomposed = pin.rpy.matrixToRpy(target) - pin.rpy.matrixToRpy(
        pin.rpy.rpyToMatrix(rpy)
    )
    solutions = []
    for d in (np.zeros(3), (decomposed + np.pi) % (2 * np.pi) - np.pi):
        for _ in range(50):
            err = pin.log3(pin.rpy.rpyToMatrix(rpy + d).T @ target)
            if np.linalg.norm(err) < 1e-15:
                break
            jac = pin.rpy.computeRpyJacobian(rpy + d, pin.LOCAL)
            d = d + np.linalg.lstsq(jac, err, rcond=1e-10)[0]
        d = (d + np.pi) % (2 * np.pi) - np.pi
        err = pin.log3(pin.rpy.rpyToMatrix(rpy + d).T @ target)
        solutions.append((np.linalg.norm(err) > 1e-12, np.linalg.norm(d), d))
    _, _, d = min(solutions, key=lambda s: s[:2])
    if max(abs(d[0]), abs(d[2])) > np.pi / 4:
        logger.info(
            "rpy %s is at/near gimbal lock: the delta %s switches rpy branch "
            "(same small rotation)",
            np.round(rpy, 6),
            np.round(d, 6),
        )
    return d


def _origin_delta(
    placement: pin.SE3, xyz_rpy, offset: bool = False, rpy=None
) -> np.ndarray:
    """PAL deltas for one joint: change of the origin's xyz and RPY.

    ``placement`` is the joint's URDF origin and ``rpy`` its rpy as written
    (default: decomposed from ``placement``). FIGAROH's ``d_*`` placement
    error acts in the joint frame, ``origin * SE3(exp3(d_phi), d_p)``
    (figaroh-plus#110); a joint offset ``offsetR*``/``offsetP*`` is the same
    transform about/along one axis (``apply_joint_offset``, which composes
    rotations as RPY; ``offset`` selects that). The PAL keys are taken as
    additive deltas on the URDF origin ``xyz`` and ``rpy``, the meaning this
    module gave them before #110, so the corrected origin is converted back
    to those deltas: ``rpyToMatrix(rpy + d_rpy)`` is the corrected rotation
    exactly (:func:`_rpy_delta`).
    """
    xyz_rpy = np.asarray(xyz_rpy, dtype=float)
    rotation = (pin.rpy.rpyToMatrix if offset else pin.exp3)(xyz_rpy[3:6])
    corrected = placement * pin.SE3(rotation, xyz_rpy[0:3])
    d_xyz = corrected.translation - placement.translation
    if rpy is None:
        rpy = pin.rpy.matrixToRpy(placement.rotation)
    return np.r_[d_xyz, _rpy_delta(np.asarray(rpy, dtype=float), corrected.rotation)]


def _pal_joint_name(target: str) -> str:
    """Strip a trailing ``_joint`` (URDF/Pinocchio convention) — PAL's
    config keys each entry by the bare joint name, e.g. ``arm_right_2``,
    not ``arm_right_2_joint``."""
    if target.endswith(_JOINT_SUFFIX):
        return target[: -len(_JOINT_SUFFIX)]
    return target


def build_geometric_calibration(
    calibrator,
    *,
    min_sigma: Optional[float] = None,
    nominal_urdf: Optional[str] = None,
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build a PAL ``robot_state_publisher.geometric_calibration`` dict
    from a solved ``BaseCalibration`` instance's redistributed parameters.

    Per-joint kinematic corrections are included: ``full_params``
    placements (``d_px_{joint}`` etc.) and ``joint_offset`` offsets
    (``offsetRZ_{joint}`` etc., figaroh-plus#123), both converted to
    additive deltas on the joint's URDF origin. Marker/tip, base-frame and
    elasticity parameters are not kinematic joint corrections in
    :func:`~figaroh.tools.urdf_exporter._parse_param_name`'s registry and
    are dropped by the category check. The base frame needs no exclusion:
    the lift holds the base-frame rows at 0, so every lifted value is a joint
    correction.

    Args:
        calibrator: A solved ``BaseCalibration`` instance (``solve()``
            already called).
        min_sigma: If given, only include parameters with
            ``|value| / std_dev >= min_sigma``: a conservative deploy
            that leaves statistically insignificant corrections at
            nominal. ``None`` (default) includes every joint-placement
            parameter. Whether it generalizes better is robot- and
            data-specific; check on held-out postures.
        nominal_urdf: The URDF the robot runs, to which the deltas are
            added. Its ``<origin>`` values, as written, are the reference
            (figaroh-plus#123). Without it they are recovered from
            ``calibrator.model``, which is exact unless an origin's rpy is
            at or near pitch = +-pi/2 or has |pitch| > pi/2.

    Returns:
        ``{"robot_state_publisher": {"geometric_calibration": {key: value}}}``

    Raises:
        CalibrationError: Propagated from ``redistribute_parameters()`` if
            ``solve()``/``create_param_list()`` haven't run.
    """
    redistributed = calibrator.redistribute_parameters()

    corrections: Dict[str, np.ndarray] = {}
    offsets = set()
    for name, info in redistributed.items():
        parsed = _parse_param_name(name)
        # joint_offset: same joint-frame transform as the matching d_* (#123)
        if parsed is None or parsed[0] not in ("joint_placement", "joint_offset"):
            continue
        category, target, sub_idx, _ = parsed

        value, std_dev = info["value"], info["std_dev"]
        if min_sigma is not None:
            sigma = abs(value) / std_dev if std_dev > 0 else float("inf")
            if sigma < min_sigma:
                continue

        corrections.setdefault(target, np.zeros(6))[sub_idx] = value
        if category == "joint_offset":
            offsets.add(target)

    model = getattr(calibrator, "model", None)
    origins = _urdf_origins(nominal_urdf) if nominal_urdf is not None else {}
    geometric_calibration: Dict[str, float] = {}
    for target, xyz_rpy in corrections.items():
        placement, rpy = pin.SE3.Identity(), None
        if target in origins:
            xyz, rpy = origins[target]
            placement = pin.SE3(pin.rpy.rpyToMatrix(rpy), xyz)
        elif nominal_urdf is not None:
            raise ValueError(f"Joint '{target}' not found in {nominal_urdf}")
        elif model is not None and model.existJointName(target):
            placement = _urdf_origin(model, target)
        delta = _origin_delta(placement, xyz_rpy, offset=target in offsets, rpy=rpy)
        for sub_idx, value in enumerate(delta):
            if abs(value) > 1e-12:
                key = f"{_pal_joint_name(target)}_{_AXIS_SUFFIX[sub_idx]}"
                geometric_calibration[key] = float(value)

    return {"robot_state_publisher": {"geometric_calibration": geometric_calibration}}


def export_geometric_calibration_yaml(
    calibrator,
    output_path: str,
    *,
    min_sigma: Optional[float] = None,
    header_comment: Optional[str] = None,
    nominal_urdf: Optional[str] = None,
) -> str:
    """:func:`build_geometric_calibration` + write as YAML, PAL deploy-ready.

    PAL's ``master_calibration.yaml`` layout (``robot_state_publisher:
    geometric_calibration:`` with ``<joint>_<axis>`` keys), so the output
    drops in at ``/etc/calibration/master_calibration.yaml`` on a PAL robot.

    Args:
        calibrator: A solved ``BaseCalibration`` instance.
        output_path: Destination YAML file path.
        min_sigma: See :func:`build_geometric_calibration`.
        nominal_urdf: See :func:`build_geometric_calibration`; pass it.
        header_comment: Optional single-line comment written above the
            YAML document (e.g. source data file, sample count, RMSE), for
            provenance.

    Returns:
        ``output_path``, unchanged, for chaining.
    """
    data = build_geometric_calibration(
        calibrator, min_sigma=min_sigma, nominal_urdf=nominal_urdf
    )
    with open(output_path, "w") as f:
        if header_comment:
            f.write(f"# {header_comment}\n")
        yaml.dump(data, f, sort_keys=True, default_flow_style=False)
    logger.info("Geometric calibration written to %s", output_path)
    return output_path
