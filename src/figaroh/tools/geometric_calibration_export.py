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
TALOS, ...) read at ``/etc/calibration/master_calibration.yaml`` — the
format hand-curated in ``figaroh_tiagoPro/data/master_calibration_*.yaml``
to date, generalized here so any ``BaseCalibration``-based robot can
produce one directly from a solved calibrator.

This is a different deploy target from
:mod:`figaroh.tools.urdf_exporter`: that module bakes corrections into a
*modified URDF file*; this one produces a small *runtime correction
overlay* PAL's ``robot_state_publisher`` applies on top of the original,
unmodified URDF at startup. Both read the same ``d_px_{joint}``-style
parameter names — this module reuses
:func:`figaroh.tools.urdf_exporter._parse_param_name` rather than
re-deriving that parsing.

Built on :meth:`~figaroh.calibration.base_calibration.BaseCalibration.redistribute_parameters`
rather than the base-only fit, so joints that would otherwise be silently
left at nominal (0) in the base-only deploy — because they were
structurally redundant with another parameter, not because they have no
real offset — get their share of the identified correction too. See
``TIAGO_CALIBRATION_ANALYSIS.md`` §8 for why that redistribution exists
and its limits.
"""

import logging
from typing import Dict, Optional

import yaml

from figaroh.tools.urdf_exporter import _parse_param_name

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Order matches _parse_param_name's sub_idx (0..5), which in turn matches
# FULL_PARAMTPL = ["d_px", "d_py", "d_pz", "d_phix", "d_phiy", "d_phiz"].
_AXIS_SUFFIX = ["dx", "dy", "dz", "droll", "dpitch", "dyaw"]

_JOINT_SUFFIX = "_joint"


def _pal_joint_name(target: str) -> str:
    """Strip a trailing ``_joint`` (URDF/Pinocchio convention) — PAL's
    config keys each entry by the bare joint name, e.g. ``arm_right_2``,
    not ``arm_right_2_joint``."""
    if target.endswith(_JOINT_SUFFIX):
        return target[: -len(_JOINT_SUFFIX)]
    return target


def build_geometric_calibration(
    calibrator, *, min_sigma: Optional[float] = None
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Build a PAL ``robot_state_publisher.geometric_calibration`` dict
    from a solved ``BaseCalibration`` instance's redistributed parameters.

    Only genuine per-joint placement corrections (``d_px_{joint}`` etc.)
    are included. Excluded, matching every hand-curated
    ``figaroh_tiagoPro/data/master_calibration_*.yaml`` to date:

    - Any parameter merged with the co-estimated base transform — when
      ``calib_config["known_baseframe"]`` is ``False``, that's whichever
      joint occupies the first 6 slots of ``base_mapping_row_names``, a
      mixed mocap/robot quantity, not a pure per-joint correction. Found
      structurally (via ``base_mapping_row_names``, which is never
      renamed), not by name pattern — robust regardless of whatever
      display renaming ``add_base_name``/a subclass override may have
      applied to ``calib_config["param_name"]``.
    - Marker/tip parameters (``pEE*``/``phiEE*``) and anything from a
      non-``full_params`` candidate set (e.g. joint-offset/elasticity
      parameters) — neither is a ``joint_placement`` entry in
      :func:`~figaroh.tools.urdf_exporter._parse_param_name`'s registry,
      so both are dropped by the category check below without needing
      special-casing.

    Args:
        calibrator: A solved ``BaseCalibration`` instance (``solve()``
            already called).
        min_sigma: If given, only include parameters with
            ``|value| / std_dev >= min_sigma`` — the "conservative"
            variant (see ``master_calibration_20260805_conservative.yaml``
            and ``TIAGO_CALIBRATION_ANALYSIS.md`` §7.5/§7.6, which found
            this generalizes measurably better than deploying every
            identified value regardless of statistical significance).
            ``None`` (default) includes every joint-placement parameter.

    Returns:
        ``{"robot_state_publisher": {"geometric_calibration": {key: value}}}``

    Raises:
        CalibrationError: Propagated from ``redistribute_parameters()`` if
            ``solve()``/``create_param_list()`` haven't run.
    """
    redistributed = calibrator.redistribute_parameters()

    exclude = set()
    if not calibrator.calib_config.get("known_baseframe", True):
        row_names = calibrator.calib_config.get("base_mapping_row_names", [])
        exclude.update(row_names[:6])

    geometric_calibration: Dict[str, float] = {}
    for name, info in redistributed.items():
        if name in exclude:
            continue
        parsed = _parse_param_name(name)
        if parsed is None or parsed[0] != "joint_placement":
            continue
        _, target, sub_idx, _ = parsed

        value, std_dev = info["value"], info["std_dev"]
        if min_sigma is not None:
            sigma = abs(value) / std_dev if std_dev > 0 else float("inf")
            if sigma < min_sigma:
                continue

        key = f"{_pal_joint_name(target)}_{_AXIS_SUFFIX[sub_idx]}"
        geometric_calibration[key] = value

    return {"robot_state_publisher": {"geometric_calibration": geometric_calibration}}


def export_geometric_calibration_yaml(
    calibrator,
    output_path: str,
    *,
    min_sigma: Optional[float] = None,
    header_comment: Optional[str] = None,
) -> str:
    """:func:`build_geometric_calibration` + write as YAML, PAL deploy-ready.

    Matches ``figaroh_tiagoPro/data/master_calibration_*.yaml``'s format
    exactly (same nesting, same key style), so the output drops in at
    ``/etc/calibration/master_calibration.yaml`` on a PAL robot unchanged.

    Args:
        calibrator: A solved ``BaseCalibration`` instance.
        output_path: Destination YAML file path.
        min_sigma: See :func:`build_geometric_calibration`.
        header_comment: Optional single-line comment written above the
            YAML document (e.g. source data file, sample count, RMSE) —
            matches the one-line provenance header every hand-curated
            ``master_calibration_*.yaml`` carries today.

    Returns:
        ``output_path``, unchanged, for chaining.
    """
    data = build_geometric_calibration(calibrator, min_sigma=min_sigma)
    with open(output_path, "w") as f:
        if header_comment:
            f.write(f"# {header_comment}\n")
        yaml.dump(data, f, sort_keys=True, default_flow_style=False)
    logger.info("Geometric calibration written to %s", output_path)
    return output_path
