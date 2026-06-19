"""CAD-informed convex constraints for inertial parameter identification.

This module provides helpers to add physically motivated *a priori* constraints
to the identification and reconstruction optimisation problems:

- **Mass bounds**: box constraint ``m_min <= m_j <= m_max`` per link.
- **CoM bounds**: box constraints on the first-moment vector ``h = m*c``.
- **Symmetry**: equality constraints between parameters of symmetric links.

All constraints are convex (linear or equality) and integrate cleanly with
the per-link LMI projection and full-robot SDP reconstruction.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_INERTIAL_KEYS: List[str] = [
    "m",
    "mx",
    "my",
    "mz",
    "Ixx",
    "Ixy",
    "Iyy",
    "Ixz",
    "Iyz",
    "Izz",
]
_AXIS_TO_KEY: Dict[str, str] = {"x": "mx", "y": "my", "z": "mz"}
_H_KEY_TO_P10_IDX: Dict[str, int] = {"mx": 1, "my": 2, "mz": 3}


@dataclass
class CADConstraints:
    """Container for convex CAD-informed constraints on robot inertial parameters.

    All constraints are convex (linear or equality) and safe to combine with
    the LMI physical-consistency constraint ``P_j >> 0``.

    Attributes:
        mass_bounds: Per-joint mass bounds ``{joint_name: (m_min, m_max)}``.
        com_bounds: Per-joint first-moment bounds
            ``{joint_name: {axis: (h_min, h_max)}}``.
        symmetry_pairs: List of ``(joint_a, joint_b, keys)`` tuples.
    """

    mass_bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    com_bounds: Dict[str, Dict[str, Tuple[float, float]]] = field(default_factory=dict)
    symmetry_pairs: List[Tuple[str, str, List[str]]] = field(default_factory=list)

    def is_empty(self) -> bool:
        """Return ``True`` if no constraints have been added."""
        return not self.mass_bounds and not self.com_bounds and not self.symmetry_pairs


def add_mass_bounds(
    cad: CADConstraints,
    joint_name: str,
    *,
    m_min: float,
    m_max: float,
) -> CADConstraints:
    """Add mass bounds for *joint_name* and return *cad*.

    Raises:
        ValueError: If ``m_min > m_max``.
    """
    if m_min > m_max:
        raise ValueError(
            f"mass_bounds for {joint_name!r}: m_min={m_min} > m_max={m_max}"
        )
    cad.mass_bounds[joint_name] = (float(m_min), float(m_max))
    return cad


def add_com_bounds(
    cad: CADConstraints,
    joint_name: str,
    *,
    axis: str,
    h_min: float,
    h_max: float,
) -> CADConstraints:
    """Add first-moment (CoM) bounds for *joint_name* / *axis* to *cad*.

    The constraint acts on ``h_k = m * c_k`` (first moment, not CoM position),
    which keeps the constraint linear in the inertial parameters.

    Raises:
        ValueError: If *axis* is invalid or *h_min* > *h_max*.
    """
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got {axis!r}")
    if h_min > h_max:
        raise ValueError(
            f"com_bounds for {joint_name!r}/{axis}: h_min={h_min} > h_max={h_max}"
        )
    cad.com_bounds.setdefault(joint_name, {})[axis] = (float(h_min), float(h_max))
    return cad


def add_symmetry_constraints(
    cad: CADConstraints,
    joint_a: str,
    joint_b: str,
    *,
    keys: Optional[List[str]] = None,
) -> CADConstraints:
    """Add parameter equality constraints between *joint_a* and *joint_b*.

    Imposes ``theta[key_joint_a] == theta[key_joint_b]`` for each key in
    *keys* within the full-robot SDP reconstruction.

    Args:
        keys: Inertial keys to equalize. Defaults to all 10 standard keys.
    """
    if keys is None:
        keys = list(_INERTIAL_KEYS)
    cad.symmetry_pairs.append((str(joint_a), str(joint_b), list(keys)))
    return cad


def bounds_from_urdf(
    model: Any,
    *,
    mass_scale_min: float = 0.01,
    mass_scale_max: float = 100.0,
    com_margin_abs: float = 1.0,
    com_margin_rel: float = 10.0,
) -> CADConstraints:
    """Derive mass and CoM bounds from a Pinocchio model's URDF inertials.

    **CAD data source strategy**: nominal values are read from
    ``model.inertias`` and bounds are built by applying scale factors and
    absolute margins.

    * **Mass**: ``m_min = max(m_nom * mass_scale_min, 1e-6)``,
      ``m_max = m_nom * mass_scale_max``.
    * **CoM first moment** per axis k:
      ``margin_k = max(|h_nom_k| * com_margin_rel, com_margin_abs)``
      then ``[h_nom_k - margin_k, h_nom_k + margin_k]``.

    Raises:
        ImportError: If pinocchio is not installed.
    """
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "bounds_from_urdf requires pinocchio. "
            "Install it with: conda install -c conda-forge pinocchio"
        ) from exc

    cad = CADConstraints()

    for i in range(1, model.njoints):  # index 0 = universe
        jname = str(model.names[i])
        inertia = model.inertias[i]
        m_nom = float(inertia.mass)

        m_lo = max(float(mass_scale_min) * m_nom, 1e-6)
        m_hi = float(mass_scale_max) * m_nom
        if m_hi <= m_lo:
            m_hi = m_lo + 1.0
        add_mass_bounds(cad, jname, m_min=m_lo, m_max=m_hi)

        h_nom = np.asarray(inertia.lever, dtype=float) * m_nom
        for k_idx, axis_name in enumerate(["x", "y", "z"]):
            h_k = float(h_nom[k_idx])
            margin = max(
                abs(h_k) * float(com_margin_rel),
                float(com_margin_abs),
            )
            add_com_bounds(
                cad,
                jname,
                axis=axis_name,
                h_min=h_k - margin,
                h_max=h_k + margin,
            )

    return cad


def build_cad_constraints_from_config(
    cfg: Mapping[str, Any],
    *,
    model: Optional[Any] = None,
) -> Optional[CADConstraints]:
    """Build a :class:`CADConstraints` from a YAML-derived config dict.

    Supports ``source: "urdf"`` (derives bounds from Pinocchio model) and
    ``source: "manual"`` (default, reads explicit ``mass_bounds``,
    ``com_bounds``, ``symmetry_pairs`` keys).  Both modes can be combined.

    Config YAML schema example::

        cad_constraints:
          source: urdf          # or "manual" (default)
          mass_scale_min: 0.01
          mass_scale_max: 100.0
          com_margin_abs: 1.0
          com_margin_rel: 10.0
          mass_bounds:
            joint1: [0.5, 5.0]
          com_bounds:
            joint1:
              x: [-0.2, 0.2]
              y: [-0.1, 0.1]
          symmetry_pairs:
            - [joint_left, joint_right, [m, Ixx, Iyy, Izz]]

    Returns:
        A populated :class:`CADConstraints`, or ``None`` if *cfg* is empty or
        produces no constraints.
    """
    if not cfg:
        return None

    source = str(cfg.get("source", "manual")).strip().lower()

    if source == "urdf":
        if model is None:
            raise ValueError(
                "cad_constraints source='urdf' requires a Pinocchio model; "
                "pass model= to build_cad_constraints_from_config"
            )
        cad = bounds_from_urdf(
            model,
            mass_scale_min=float(cfg.get("mass_scale_min", 0.01)),
            mass_scale_max=float(cfg.get("mass_scale_max", 100.0)),
            com_margin_abs=float(cfg.get("com_margin_abs", 1.0)),
            com_margin_rel=float(cfg.get("com_margin_rel", 10.0)),
        )
    else:
        cad = CADConstraints()

    for joint, val in cfg.get("mass_bounds", {}).items():
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            add_mass_bounds(cad, str(joint), m_min=float(val[0]), m_max=float(val[1]))
        elif isinstance(val, dict):
            add_mass_bounds(
                cad,
                str(joint),
                m_min=float(val.get("min", 0.0)),
                m_max=float(val.get("max", 1e9)),
            )

    for joint, axes in cfg.get("com_bounds", {}).items():
        if not isinstance(axes, dict):
            continue
        for axis, val in axes.items():
            if isinstance(val, (list, tuple)) and len(val) >= 2:
                add_com_bounds(
                    cad,
                    str(joint),
                    axis=str(axis),
                    h_min=float(val[0]),
                    h_max=float(val[1]),
                )
            elif isinstance(val, dict):
                add_com_bounds(
                    cad,
                    str(joint),
                    axis=str(axis),
                    h_min=float(val.get("min", -1e9)),
                    h_max=float(val.get("max", 1e9)),
                )

    for pair in cfg.get("symmetry_pairs", []):
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            j1 = str(pair[0])
            j2 = str(pair[1])
            pair_keys: Optional[List[str]] = None
            if len(pair) >= 3 and isinstance(pair[2], (list, tuple)):
                pair_keys = [str(k) for k in pair[2]]
            add_symmetry_constraints(cad, j1, j2, keys=pair_keys)

    return cad if not cad.is_empty() else None


def apply_cad_constraints_to_problem(
    problem: Any,
    theta: Any,
    params_r: Sequence[str],
    cad: CADConstraints,
) -> int:
    """Add CAD-informed linear/equality constraints to a picos Problem.

    Mutates *problem* in-place by appending constraints derived from *cad*
    onto the picos variable *theta* (shape ``(n, 1)``).

    Args:
        problem: A ``picos.Problem`` instance.
        theta: A ``picos.RealVariable`` of shape ``(n, 1)`` aligned to *params_r*.
        params_r: Ordered list of parameter names.
        cad: :class:`CADConstraints` to apply.

    Returns:
        Number of constraints added to *problem*.
    """
    name_list = list(params_r)

    def _idx(key: str, joint: str) -> Optional[int]:
        try:
            return name_list.index(f"{key}_{joint}")
        except ValueError:
            return None

    n_added = 0

    for joint, (m_lo, m_hi) in cad.mass_bounds.items():
        i = _idx("m", joint)
        if i is not None:
            problem.add_constraint(theta[i] >= m_lo)
            problem.add_constraint(theta[i] <= m_hi)
            n_added += 2

    for joint, axes in cad.com_bounds.items():
        for axis, (h_lo, h_hi) in axes.items():
            key = _AXIS_TO_KEY.get(axis)
            if key is None:
                continue
            i = _idx(key, joint)
            if i is not None:
                problem.add_constraint(theta[i] >= h_lo)
                problem.add_constraint(theta[i] <= h_hi)
                n_added += 2

    for joint_a, joint_b, keys in cad.symmetry_pairs:
        for key in keys:
            i_a = _idx(key, joint_a)
            i_b = _idx(key, joint_b)
            if i_a is not None and i_b is not None:
                problem.add_constraint(theta[i_a] == theta[i_b])
                n_added += 1

    return n_added
