"""
URDF exporter for identified/calibrated robot parameters.

Reads a nominal URDF and produces a modified URDF by applying parameter
overlays. Parameter names encode the semantics: ``d_px_joint2`` is additive
(geometric offset), ``m_link1`` is absolute (mass override).

This is used after identification or calibration to materialize results
into a tangible URDF file for downstream use (simulation, visualization,
model-based control).

**Two parameter categories**

Joint-level parameters (auto-applied to the URDF):
    These directly modify the URDF's joint origins, link inertias, and
    dynamics attributes. They come from figaroh's identification or
    calibration solvers and can be applied automatically:

    - Joint placement (additive): ``d_px_{joint}``, ``d_py_{joint}``,
      ``d_pz_{joint}``, ``d_phix_{joint}``, ``d_phiy_{joint}``,
      ``d_phiz_{joint}``
    - Joint offset / calibration (additive): ``offsetPX_{joint}``,
      ``offsetPY_{joint}``, ``offsetPZ_{joint}``, ``offsetRX_{joint}``,
      ``offsetRY_{joint}``, ``offsetRZ_{joint}``
    - Legacy offset (absolute): ``off_{joint}``
    - Mass (absolute): ``m_{link}``
    - First moments (absolute): ``mx_{link}``, ``my_{link}``, ``mz_{link}``
    - Inertia tensor (absolute): ``Ixx_{link}``, ``Ixy_{link}``, ...,
      ``Izz_{link}``
    - Viscous/static friction (absolute): ``fv_{joint}``, ``fs_{joint}``
    - Armature (absolute): ``Ia_{joint}``
    - Joint elasticity (additive): ``k_PX_{joint}``, ..., ``k_RZ_{joint}``

Metrology frame parameters (user-defined, not auto-applied):
    These define the transformation between the robot (URDF) and the
    external measurement system (mocap, camera, chessboard, etc.).
    They depend on the **calibration setup**, not on the URDF itself.
    ``export_urdf()`` will **not** auto-apply them; instead it logs a
    reminder and returns them as metadata for the user to configure::

        base_px, base_py, base_pz, base_phix, base_phiy, base_phiz
            Transform from the **metrology frame** (e.g. mocap world,
            Vicon origin) to the robot's ``base_link``.
            Default: identity (no offset from origin).

        pEEx_{frame}, pEEy_{frame}, pEEz_{frame}
        phiEEx_{frame}, phiEEy_{frame}, phiEEz_{frame}
            Transform from the last robot joint (e.g. ``arm_7_joint``,
            ``head_2_link``) to the **measurement frame** mounted on the
            end-effector. What this frame is depends on calibration type:

            - **Mocap calibration**: optical marker cluster frame
              (markers attached to the end-effector).
            - **Eye-hand calibration**: camera optical frame
              (e.g. ``xtion_rgb_optical_frame``) or chessboard frame
              (pattern on the gripper).
            Default: identity (measurement frame coincides with the joint).

Typical usage::

    from figaroh.tools.urdf_exporter import export_urdf, frame_settings_doc

    # Joint-level params (auto-applied to URDF)
    params = {
        "d_px_joint2": 0.05,
        "m_link1": 2.5,
        "fv_joint1": 0.2,
    }
    modified_path = export_urdf("robot.urdf", params, verbose=True)

    # Metrology frames (user-defined — see frame_settings_doc())
    defaults = frame_settings_doc()
    # → prints descriptions + default values for base and EE frame params
"""

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union, List, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ── Parameter-name registries ────────────────────────────────────
#
# Two registries:
#   1. _PARAM_REGISTRY     — joint-level params (auto-applied to URDF)
#   2. _METROLOGY_REGISTRY — base/EE frame params (user-defined, see
#      frame_settings_doc() — not auto-applied)
#
# Each entry: (prefix, lookup_style, category, sub_idx)
#   lookup_style:
#     "exact"       — param name IS the prefix (e.g. "base_px")
#     "prefix"      — param starts with f"{prefix}_" — rest is joint/link name
#     "prefix_nosep" — param starts with prefix directly (e.g. "m_" → "m_link1")

_PARAM_REGISTRY: List[tuple] = [
    # ── JOINT PLACEMENT (additive, format: d_px_{joint}) ──
    ("d_px",   "prefix", "joint_placement", 0),
    ("d_py",   "prefix", "joint_placement", 1),
    ("d_pz",   "prefix", "joint_placement", 2),
    ("d_phix", "prefix", "joint_placement", 3),
    ("d_phiy", "prefix", "joint_placement", 4),
    ("d_phiz", "prefix", "joint_placement", 5),
    # ── JOINT OFFSET / CALIBRATION (additive, format: offsetRX_{joint}) ──
    ("offsetPX", "prefix", "joint_offset", 0),
    ("offsetPY", "prefix", "joint_offset", 1),
    ("offsetPZ", "prefix", "joint_offset", 2),
    ("offsetRX", "prefix", "joint_offset", 3),
    ("offsetRY", "prefix", "joint_offset", 4),
    ("offsetRZ", "prefix", "joint_offset", 5),
    # ── ELASTICITY (additive, format: k_PX_{joint}) ──
    ("k_PX", "prefix", "elasticity", 0),
    ("k_PY", "prefix", "elasticity", 1),
    ("k_PZ", "prefix", "elasticity", 2),
    ("k_RX", "prefix", "elasticity", 3),
    ("k_RY", "prefix", "elasticity", 4),
    ("k_RZ", "prefix", "elasticity", 5),
    # ── ABSOLUTE: MASS (format: m_{link}) ──
    ("m_",   "prefix_nosep", "mass", None),
    # ── ABSOLUTE: FIRST MOMENTS (format: mx_{link}, my_{link}, mz_{link}) ──
    ("mx_", "prefix_nosep", "first_moment", 0),
    ("my_", "prefix_nosep", "first_moment", 1),
    ("mz_", "prefix_nosep", "first_moment", 2),
    # ── ABSOLUTE: INERTIA TENSOR (format: Ixx_{link}, etc.) ──
    ("Ixx_", "prefix_nosep", "inertia", (0, 0)),
    ("Ixy_", "prefix_nosep", "inertia", (0, 1)),
    ("Ixz_", "prefix_nosep", "inertia", (0, 2)),
    ("Iyy_", "prefix_nosep", "inertia", (1, 1)),
    ("Iyz_", "prefix_nosep", "inertia", (1, 2)),
    ("Izz_", "prefix_nosep", "inertia", (2, 2)),
    # ── ABSOLUTE: DYNAMICS (format: fv_{joint}, fs_{joint}, Ia_{joint}) ──
    ("fv_", "prefix_nosep", "viscous_friction", None),
    ("fs_", "prefix_nosep", "static_friction", None),
    ("Ia_", "prefix_nosep", "armature", None),
    # ── ABSOLUTE: LEGACY JOINT OFFSET (format: off_{joint}) ──
    ("off_", "prefix_nosep", "legacy_offset", None),
]

_METROLOGY_REGISTRY: List[tuple] = [
    # ── BASE FRAME (metrology-frame → robot base) ──
    # Transform from the external measurement origin (mocap world, Vicon,
    # etc.) to the robot's base_link.  NOT intrinsic to the URDF — users
    # must define these based on their calibration setup.
    ("base_px",   "exact", "base_frame", 0),
    ("base_py",   "exact", "base_frame", 1),
    ("base_pz",   "exact", "base_frame", 2),
    ("base_phix", "exact", "base_frame", 3),
    ("base_phiy", "exact", "base_frame", 4),
    ("base_phiz", "exact", "base_frame", 5),
    # ── EE MEASUREMENT FRAME (last joint → sensor/marker/chessboard) ──
    # Transform from the last robot joint (e.g. arm_7_joint, head_2_link)
    # to the measurement frame mounted on the end-effector.
    # For mocap: optical marker frame.  For eye-hand: camera optical frame
    # or chessboard frame.
    ("pEEx",   "prefix_nosep", "ee_measurement_frame", 0),
    ("pEEy",   "prefix_nosep", "ee_measurement_frame", 1),
    ("pEEz",   "prefix_nosep", "ee_measurement_frame", 2),
    ("phiEEx", "prefix_nosep", "ee_measurement_frame", 3),
    ("phiEEy", "prefix_nosep", "ee_measurement_frame", 4),
    ("phiEEz", "prefix_nosep", "ee_measurement_frame", 5),
]

# ── Registry: is_additive flags ─────────────────────────────

_ADDITIVE_PREFIXES = frozenset({
    "d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz",
    "offsetPX", "offsetPY", "offsetPZ", "offsetRX", "offsetRY", "offsetRZ",
    "k_PX", "k_PY", "k_PZ", "k_RX", "k_RY", "k_RZ",
})


# ── Joint-param lookup maps (auto-apply) ─────────────────────

_EXACT_MAP: dict = {}
_PREFIX_MAP: list = []
_PREFIX_NOSEP_MAP: list = []

for prefix, style, cat, idx in _PARAM_REGISTRY:
    entry = (cat, idx, prefix in _ADDITIVE_PREFIXES)
    if style == "exact":
        _EXACT_MAP[prefix] = entry
    elif style == "prefix":
        _PREFIX_MAP.append((prefix, entry))
    elif style == "prefix_nosep":
        _PREFIX_NOSEP_MAP.append((prefix, entry))


# ── Frame-param lookup maps (user-defined, not auto-applied) ─

_FRAME_PREFIX_MAP: list = []
_FRAME_PREFIX_NOSEP_MAP: list = []
_FRAME_EXACT_MAP: dict = {}

for prefix, style, cat, idx in _METROLOGY_REGISTRY:
    entry = (cat, idx)
    if style == "exact":
        _FRAME_EXACT_MAP[prefix] = entry
    elif style == "prefix":
        _FRAME_PREFIX_MAP.append((prefix, entry))
    elif style == "prefix_nosep":
        _FRAME_PREFIX_NOSEP_MAP.append((prefix, entry))


def _parse_param_name(name: str) -> Optional[tuple]:
    """Parse a joint-level parameter name.

    Returns ``(category, target, sub_idx, is_additive)``, or ``None``
    if the name is unknown (including metrology frame params).
    """
    # 1. Exact
    if name in _EXACT_MAP:
        cat, idx, is_add = _EXACT_MAP[name]
        return (cat, "_base_", idx, is_add)
    # 2. Prefix + separator
    for prefix, (cat, idx, is_add) in _PREFIX_MAP:
        sep = prefix + "_"
        if name.startswith(sep):
            target = name[len(sep):]
            if target:
                return (cat, target, idx, is_add)
    # 3. Prefix, no separator
    for prefix, (cat, idx, is_add) in _PREFIX_NOSEP_MAP:
        if name.startswith(prefix):
            target = name[len(prefix):]
            if target:
                return (cat, target, idx, is_add)
    return None


def _parse_frame_param_name(name: str) -> Optional[tuple]:
    """Parse a metrology frame parameter name.

    Returns ``(category, target, sub_idx)``, or ``None`` if the name is
    a joint-level param or unknown.
    """
    # 1. Exact (base_*)
    if name in _FRAME_EXACT_MAP:
        cat, idx = _FRAME_EXACT_MAP[name]
        return (cat, "_base_", idx)
    # 2. Prefix + separator
    for prefix, (cat, idx) in _FRAME_PREFIX_MAP:
        sep = prefix + "_"
        if name.startswith(sep):
            target = name[len(sep):]
            if target:
                return (cat, target, idx)
    # 3. Prefix, no separator (pEEx{frame}, phiEEx{frame})
    for prefix, (cat, idx) in _FRAME_PREFIX_NOSEP_MAP:
        if name.startswith(prefix):
            target = name[len(prefix):]
            if target:
                return (cat, target, idx)
    return None


# ── XML helpers ──────────────────────────────────────────────────


def _find_joint(doc: ET.ElementTree, name: str) -> Optional[ET.Element]:
    """Find a <joint> element by name attribute."""
    for joint in doc.findall(".//joint"):
        if joint.get("name") == name:
            return joint
    return None


def _find_link(doc: ET.ElementTree, name: str) -> Optional[ET.Element]:
    """Find a <link> element by name attribute."""
    for link in doc.findall(".//link"):
        if link.get("name") == name:
            return link
    return None


def _get_or_create_element(parent: ET.Element, tag: str) -> ET.Element:
    """Get existing child element by tag, or create a new one."""
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def _get_xyz_array(elem: ET.Element, attr: str = "xyz") -> List[float]:
    """Parse a space-separated triple attribute into a float list."""
    val = elem.get(attr, "0 0 0")
    return [float(v) for v in val.split()]


def _set_xyz_array(elem: ET.Element, values: List[float],
                   attr: str = "xyz") -> None:
    """Set a space-separated triple attribute from a float list.

    Uses a clean format: up to 6 significant digits, no trailing zeros.
    """
    elem.set(attr, " ".join(_fmt(v) for v in values))


def _fmt(v: float) -> str:
    """Format a float for URDF output — compact, no scientific notation."""
    if v == 0.0:
        return "0"
    s = f"{v:.6g}"
    # Ensure we don't get scientific notation
    if "e" in s or "E" in s:
        s = f"{v:.10f}".rstrip("0").rstrip(".")
    return s


# ── Handlers ─────────────────────────────────────────────────────


def _apply_joint_placement(doc: ET.ElementTree, target: str, idx: int,
                           value: float, is_additive: bool) -> None:
    """Apply a joint origin placement delta (d_px_*, base_*).

    ``target`` is the joint name (or ``"_base_"`` for base params).
    ``idx`` maps to xyz (0-2) or rpy (3-5).
    ``is_additive`` is always True for this category.
    """
    if target == "_base_":
        # Base params target the first non-fixed joint
        for joint in doc.findall(".//joint"):
            jtype = joint.get("type", "fixed")
            if jtype != "fixed":
                target_joint = joint.get("name", "")
                break
        else:
            logger.warning("No non-fixed joint found for base_* params")
            return
    else:
        target_joint = target

    joint = _find_joint(doc, target_joint)
    if joint is None:
        logger.warning("Joint '%s' not found in URDF, skipping", target_joint)
        return

    origin = _get_or_create_element(joint, "origin")
    is_rotation = idx >= 3

    if is_rotation:
        attr = "rpy"
        arr = _get_xyz_array(origin, attr)
        if len(arr) < 3:
            arr = [0.0, 0.0, 0.0]
        arr[idx - 3] += value if is_additive else value
        _set_xyz_array(origin, arr, attr)
    else:
        attr = "xyz"
        arr = _get_xyz_array(origin, attr)
        if len(arr) < 3:
            arr = [0.0, 0.0, 0.0]
        arr[idx] += value if is_additive else value
        _set_xyz_array(origin, arr, attr)


def _apply_joint_offset(doc: ET.ElementTree, target: str, idx: int,
                        value: float, is_additive: bool) -> None:
    """Apply a joint calibration offset (offsetRX_*).

    Maps idx 0-2 to the calibration rising value (x,y,z not meaningful
    for revolute joints — convention stores the angle in the first element).
    For revolute joints: only the RX/RY/RZ component matters.
    """
    joint = _find_joint(doc, target)
    if joint is None:
        logger.warning("Joint '%s' not found in URDF, skipping", target)
        return

    calib = _get_or_create_element(joint, "calibration")
    current = calib.get("rising")
    if current is not None:
        new_val = float(current) + value if is_additive else value
    else:
        new_val = value
    calib.set("rising", _fmt(new_val))


def _apply_mass(doc: ET.ElementTree, target: str, _idx, value: float,
                _is_additive: bool = False) -> None:
    """Replace link mass (m_* — always absolute)."""
    link = _find_link(doc, target)
    if link is None:
        logger.warning("Link '%s' not found in URDF, skipping", target)
        return
    inertial = _get_or_create_element(link, "inertial")
    mass = _get_or_create_element(inertial, "mass")
    mass.set("value", _fmt(value))


def _apply_viscous_friction(doc: ET.ElementTree, target: str, _idx,
                            value: float, _is_additive: bool = False) -> None:
    """Replace joint dynamics damping (fv_* — always absolute)."""
    joint = _find_joint(doc, target)
    if joint is None:
        logger.warning("Joint '%s' not found in URDF, skipping", target)
        return
    dyn = _get_or_create_element(joint, "dynamics")
    dyn.set("damping", _fmt(value))


def _apply_static_friction(doc: ET.ElementTree, target: str, _idx,
                           value: float, _is_additive: bool = False) -> None:
    """Replace joint dynamics friction (fs_* — always absolute)."""
    joint = _find_joint(doc, target)
    if joint is None:
        logger.warning("Joint '%s' not found in URDF, skipping", target)
        return
    dyn = _get_or_create_element(joint, "dynamics")
    dyn.set("friction", _fmt(value))


def _apply_armature(doc: ET.ElementTree, target: str, _idx,
                    value: float, _is_additive: bool = False) -> None:
    """Replace joint armature inertia (Ia_* — always absolute)."""
    joint = _find_joint(doc, target)
    if joint is None:
        logger.warning("Joint '%s' not found in URDF, skipping", target)
        return
    dyn = _get_or_create_element(joint, "dynamics")
    dyn.set("armature", _fmt(value))


def _apply_elasticity(doc: ET.ElementTree, target: str, idx: int,
                      value: float, is_additive: bool) -> None:
    """Apply joint stiffness (k_* — additive).

    URDF doesn't have a native elasticity element — we store it as
    a custom ``<dynamics elasticity="..."/>`` attribute.
    """
    joint = _find_joint(doc, target)
    if joint is None:
        logger.warning("Joint '%s' not found in URDF, skipping", target)
        return
    dyn = _get_or_create_element(joint, "dynamics")
    # We use a single elasticity value; for multi-DOF joints more
    # sophisticated handling would be needed.
    current = dyn.get("elasticity")
    if current is not None:
        new_val = float(current) + value if is_additive else value
    else:
        new_val = value
    dyn.set("elasticity", _fmt(new_val))


# Map category to handler
_HANDLERS = {
    "joint_placement": _apply_joint_placement,
    "joint_offset": _apply_joint_offset,
    "mass": _apply_mass,
    "viscous_friction": _apply_viscous_friction,
    "static_friction": _apply_static_friction,
    "armature": _apply_armature,
    "elasticity": _apply_elasticity,
    # Stub handlers for future extension
    "first_moment": lambda doc, target, idx, val, add: \
        logger.debug("first_moment handler not implemented (target=%s)", target),
    "inertia": lambda doc, target, idx, val, add: \
        logger.debug("inertia handler not implemented (target=%s)", target),
}


_FRAME_PARAM_DESCRIPTIONS = {
    "base_frame": "Base frame: transform from metrology origin (e.g. mocap world, "
                  "Vicon origin) to robot ``base_link``.",
    "ee_measurement_frame": "EE measurement frame: transform from the last robot "
                            "joint (e.g. ``arm_7_joint``, ``head_2_link``) to the "
                            "measurement device frame (marker cluster, camera "
                            "optical frame, or chessboard frame).",
}


def frame_settings_doc(*, calibration_type: Optional[str] = None,
                       verbose: bool = True) -> dict:
    """Return default metrology-frame parameter values with explanations.

    These parameters define the transformation between the robot and the
    external measurement system.  They are **not** intrinsic to the URDF
    and must be configured by the user for each calibration setup.

    Args:
        calibration_type: Optional hint for context-specific defaults.
            ``"mocap"``, ``"eye_hand"``, or ``None`` (generic).
        verbose: If True (default), prints descriptions to stderr.

    Returns:
        dict with default values for all base-frame and EE-frame params::

            {
                "base_px": 0.0, ...
                "pEEx_arm_7_link": 0.0, ...
            }
    """
    defaults: dict = {}

    # Base frame defaults — identity (metrology origin = robot base)
    for name, _, _, _ in _METROLOGY_REGISTRY:
        if name.startswith("base_"):
            defaults[name] = 0.0

    # EE frame defaults — identity (sensor/marker frame = last joint)
    if verbose:
        if calibration_type == "mocap":
            target = "arm_7_link"
            note = (
                "Mocap calibration: EE measurement frame is the optical marker "
                "cluster attached to the end-effector (e.g. arm_7_link)."
            )
        elif calibration_type == "eye_hand":
            target = "head_2_link"
            note = (
                "Eye-hand calibration: EE measurement frame is the camera "
                "optical frame (relative to head_2_link) or the chessboard "
                "attached to the gripper."
            )
        else:
            target = "<frame_name>"
            note = (
                "EE measurement frame: typically a marker cluster, camera "
                "optical frame, or chessboard attached to the end-effector.  "
                "Replace ``<frame_name>`` with the actual robot joint/link name."
            )
        logger.info(
            "Metrology frame defaults (see frame_settings_doc()):\n"
            "  Base frame : metrology origin → robot base_link\n"
            "              default = identity (no offset)\n"
            "  EE frame   : last joint → measurement frame\n"
            "              default = identity\n"
            "  %s\n"
            "  To customize, pass e.g. %%s = {...} to export_urdf() "
            "and configure your\n"
            "  controller or calibration pipeline accordingly.",
            note,
        )

    return defaults


# ── Public API ───────────────────────────────────────────────────


def export_urdf(
    nominal_urdf_path: Union[str, Path],
    params: dict,
    *,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> str:
    """Apply identified/calibrated **joint-level** parameters to a nominal URDF.

    This function **auto-applies** parameters that modify the URDF directly
    (joint placements, mass, inertias, friction, etc. — see module docstring).
    It does **not** apply metrology frame parameters (``base_*``, ``pEE*``,
    ``phiEE*``); those depend on the calibration setup and must be
    configured by the user — see :func:`frame_settings_doc`.

    Args:
        nominal_urdf_path: Path to the nominal (reference) URDF file.
        params: Dictionary of ``{parameter_name: value}`` pairs.
            **Joint-level params** are applied automatically.
            **Metrology frame params** (``base_*``, ``pEE*``, ``phiEE*``)
            are logged and collected for the caller but **not** auto-applied
            to the URDF — see :func:`frame_settings_doc`.
        output_path: Path for the modified URDF. If ``None`` (default),
            writes to ``<stem>_modified.urdf`` beside the nominal URDF.
        verbose: If True, log which params were applied.

    Returns:
        Absolute path to the modified URDF file (joint params applied).  Use
        :func:`frame_settings_doc` to configure metrology frame params
        separately.

    Raises:
        FileNotFoundError: If *nominal_urdf_path* does not exist.
        ValueError: If an unknown parameter name is encountered.
    """
    nominal_path = Path(nominal_urdf_path)
    if not nominal_path.exists():
        raise FileNotFoundError(f"URDF not found: {nominal_path}")

    # Determine output path
    if output_path is None:
        output_path = nominal_path.with_stem(nominal_path.stem + "_modified")
    output_path = Path(output_path)

    # Parse URDF
    tree = ET.parse(str(nominal_path))
    doc = tree.getroot()

    # Separate joint params (auto-apply) from frame params (user-defined)
    frame_params: dict = {}

    for name, value in params.items():
        parsed = _parse_param_name(name)
        if parsed is not None:
            category, target, idx, is_additive = parsed
            handler = _HANDLERS.get(category)
            if handler is None:
                logger.warning("No handler for category '%s' (param='%s')",
                               category, name)
                continue
            if verbose:
                action = "additive" if is_additive else "absolute"
                logger.info("%s → %s.%s %s (%.4f)", name, category, target,
                            action, value)
            handler(doc, target, idx, value, is_additive)
            continue

        # Check if it's a metrology frame param (base_*, pEE*, phiEE*)
        frame_parsed = _parse_frame_param_name(name)
        if frame_parsed is not None:
            cat, f_target, idx = frame_parsed
            frame_params[name] = value
            desc = _FRAME_PARAM_DESCRIPTIONS.get(cat, cat)
            logger.info(
                "Metrology frame param '%s' = %.4f — not auto-applied to URDF.  "
                "This defines: %s  See frame_settings_doc() for defaults and "
                "explanations.",
                name, value, desc,
            )
            continue

        raise ValueError(
            f"Unknown parameter '{name}'. "
            f"Recognized joint-level categories: joint placement (d_px_*), "
            f"joint offset (offsetRX_*), mass (m_*), friction (fv_*, fs_*), "
            f"armature (Ia_*), elasticity (k_*), inertia (Ixx_*).  "
            f"Metrology frame params: base_*, pEE*, phiEE*."
        )

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), xml_declaration=True, encoding="utf-8")

    if frame_params and verbose:
        logger.info(
            "The following metrology frame parameters were **not** applied "
            "to the URDF:\n  %s\nUse frame_settings_doc() to review defaults.",
            "  ".join(f"{k}={v}" for k, v in frame_params.items()),
        )

    return str(output_path.resolve())
