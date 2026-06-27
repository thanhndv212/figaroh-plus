"""
URDF exporter for identified/calibrated robot parameters.

Reads a nominal URDF and produces a modified URDF by applying parameter
overlays. Parameter names encode the semantics: ``d_px_joint2`` is additive
(geometric offset), ``m_link1`` is absolute (mass override).

This is used after identification or calibration to materialize results
into a tangible URDF file for downstream use (simulation, visualization,
model-based control).

Typical usage::

    from figaroh.tools.urdf_exporter import export_urdf

    params = {
        "d_px_joint2": 0.05,   # additive joint placement offset
        "m_link1": 2.5,         # absolute inertial override
        "fv_joint1": 0.2,       # absolute dynamics override
    }
    modified_path = export_urdf("robot.urdf", params)
"""

import logging
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Union, List, Tuple

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ── Parameter-name registry ─────────────────────────────────────
# (prefix, lookup_style) → (category, sub_idx, is_additive)
#
# lookup_style:
#   "exact"   — the param name IS the prefix (e.g. "base_px") — no joint/link suffix
#   "prefix"  — the param name starts with f"{prefix}_" — rest is the joint/link name
#   "prefix_nosep" — the param name starts with prefix (no trailing _ separator).
#                   e.g. "m_link1" starts with "m_", link = "link1"

_PARAM_REGISTRY: List[tuple] = [
    # ── BASE PLACEMENT (additive, global, no joint suffix) ──
    ("base_px",   "exact",       "joint_placement", 0),
    ("base_py",   "exact",       "joint_placement", 1),
    ("base_pz",   "exact",       "joint_placement", 2),
    ("base_phix", "exact",       "joint_placement", 3),
    ("base_phiy", "exact",       "joint_placement", 4),
    ("base_phiz", "exact",       "joint_placement", 5),
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
    # ── EE MARKER (currently not fully implemented — recognized to avoid ValueError) ──
    ("pEEx",   "prefix_nosep", "ee_marker", 0),
    ("pEEy",   "prefix_nosep", "ee_marker", 1),
    ("pEEz",   "prefix_nosep", "ee_marker", 2),
    ("phiEEx", "prefix_nosep", "ee_marker", 3),
    ("phiEEy", "prefix_nosep", "ee_marker", 4),
    ("phiEEz", "prefix_nosep", "ee_marker", 5),
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

# Build lookup maps for speed
_EXACT_MAP: dict = {}
_PREFIX_MAP: list = []
_PREFIX_NOSEP_MAP: list = []

for prefix, style, cat, idx in _PARAM_REGISTRY:
    entry = (cat, idx, prefix in ("d_px", "d_py", "d_pz", "d_phix", "d_phiy",
                                   "d_phiz", "offsetPX", "offsetPY", "offsetPZ",
                                   "offsetRX", "offsetRY", "offsetRZ",
                                   "base_px", "base_py", "base_pz",
                                   "base_phix", "base_phiy", "base_phiz",
                                   "k_PX", "k_PY", "k_PZ", "k_RX", "k_RY", "k_RZ",
                                   "pEEx", "pEEy", "pEEz",
                                   "phiEEx", "phiEEy", "phiEEz"))
    if style == "exact":
        _EXACT_MAP[prefix] = entry
    elif style == "prefix":
        _PREFIX_MAP.append((prefix, entry))
    elif style == "prefix_nosep":
        _PREFIX_NOSEP_MAP.append((prefix, entry))


def _parse_param_name(name: str) -> Optional[tuple]:
    """Parse a parameter name into (category, target_name, sub_idx, is_additive).

    Args:
        name: Parameter name, e.g. ``"d_px_joint2"``, ``"m_link1"``.

    Returns:
        Tuple ``(category, target, sub_idx, is_additive)``, or None if unknown.
    """
    # 1. Check exact matches first (base_*, etc.)
    if name in _EXACT_MAP:
        cat, idx, is_add = _EXACT_MAP[name]
        # Base params target the root joint — we use a sentinel
        return (cat, "_base_", idx, is_add)

    # 2. Check prefix + "_" separator (d_px_{joint}, offsetRX_{joint})
    for prefix, (cat, idx, is_add) in _PREFIX_MAP:
        sep = prefix + "_"
        if name.startswith(sep):
            target = name[len(sep):]
            if target:  # must have a target name
                return (cat, target, idx, is_add)

    # 3. Check prefix without separator (m_{link}, fv_{joint})
    for prefix, (cat, idx, is_add) in _PREFIX_NOSEP_MAP:
        if name.startswith(prefix):
            target = name[len(prefix):]
            if target:  # must have a target name
                return (cat, target, idx, is_add)

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
    "ee_marker": lambda doc, target, idx, val, add: \
        logger.debug("ee_marker handler not implemented (target=%s)", target),
}


# ── Public API ───────────────────────────────────────────────────


def export_urdf(
    nominal_urdf_path: Union[str, Path],
    params: dict,
    *,
    output_path: Optional[Union[str, Path]] = None,
    verbose: bool = False,
) -> str:
    """Apply identified/calibrated parameters to a nominal URDF and write the result.

    Args:
        nominal_urdf_path: Path to the nominal (reference) URDF file.
        params: Dictionary of ``{parameter_name: value}`` pairs as produced by
            figaroh's identification or calibration routines.
        output_path: Path for the modified URDF. If ``None`` (default), writes to
            ``<stem>_modified.urdf`` beside the nominal URDF.
        verbose: If True, log which params were applied additively vs. absolutely.

    Returns:
        Absolute path to the modified URDF file.

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

    # Apply each parameter
    for name, value in params.items():
        parsed = _parse_param_name(name)
        if parsed is None:
            raise ValueError(
                f"Unknown parameter '{name}'. "
                f"Recognized categories: joint_placement (d_px_*), "
                f"joint_offset (offsetRX_*), base_placement (base_*), "
                f"mass (m_*), friction (fv_*, fs_*), armature (Ia_*), "
                f"elasticity (k_*), ee_marker (pEE*, phiEE*), "
                f"inertia (Ixx_*, Iyy_*, Izz_*)."
            )
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

    # Write output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(output_path), xml_declaration=True, encoding="utf-8")

    return str(output_path.resolve())
