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

"""
Legacy -> unified config migration.

Converts a legacy flat ``calibration:``/``identification:`` config into the
unified ``extends:``/``tasks:`` format, using the exact field mapping
documented in ``docs/source/concepts/config_parameters.md``. This is the
reverse of :func:`figaroh.calibration.config.unified_to_legacy_config` and
:func:`figaroh.identification.config.unified_to_legacy_identif_config`.

Fields with no unified-format consumer today (``is_static_regressor``,
``is_inertia_regressor``, ``embedded_forces``, ``sync_joint_motion``, and the
parsed-but-unused joint limit / ``ratio_essential`` fields — see
config_parameters.md) are preserved under the ``custom:`` section rather than
silently dropped.

CLI usage::

    python -m figaroh.utils.config_migration \\
        --input legacy_config.yaml --output unified_config.yaml \\
        [--task auto|calibration|identification] \\
        [--template base|manipulator|humanoid] \\
        [--urdf robot.urdf]   # enables the round-trip self-check
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

TEMPLATE_PATHS = {
    "base": "../../templates/base_robot_config.yaml",
    "manipulator": "../../templates/manipulator_robot.yaml",
    "humanoid": "../../templates/humanoid_robot.yaml",
}

# problem_params / tls_params fields with no unified-format consumer today.
_UNCONSUMED_PROBLEM_PARAMS = (
    "is_static_regressor",
    "is_inertia_regressor",
    "embedded_forces",
)
_UNCONSUMED_TLS_PARAMS = ("sync_joint_motion",)


def legacy_calibration_to_unified(calib_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy ``calibration:`` section to unified ``tasks.calibration``.

    Args:
        calib_data: The dict under the ``calibration:`` key of a legacy
            config file (e.g. ``yaml.safe_load(f)["calibration"]``).

    Returns:
        A ``{"tasks": {"calibration": {...}}}`` fragment, ready to merge
        with the identification fragment and an ``extends:`` root.
    """
    markers = calib_data.get("markers") or []
    unified_markers = [
        {
            "reference_joint": m.get("ref_joint"),
            "measurable_dof": m.get("measure", [True, True, True, False, False, False]),
        }
        for m in markers
    ]

    calibration_task: Dict[str, Any] = {
        "enabled": True,
        "type": "kinematic_calibration",
        "parameters": {
            "calibration_level": calib_data.get("calib_level", "full_params"),
            "include_non_geometric": bool(calib_data.get("non_geom", False)),
            "regularization_coefficient": calib_data.get("coeff_regularize", 0.01),
            "outlier_threshold": calib_data.get("outlier_eps", 0.05),
        },
        "kinematics": {
            "base_frame": calib_data.get("base_frame", "universe"),
            "tool_frame": calib_data.get("tool_frame"),
            "free_flying_base": bool(calib_data.get("free_flyer", False)),
        },
        "measurements": {
            "markers": unified_markers,
            "poses": {
                "base_pose": calib_data.get("base_pose"),
                "tool_pose": calib_data.get("tip_pose"),
            },
        },
        "data": {
            "source_file": calib_data.get("data_file"),
            "sample_configurations_file": calib_data.get("sample_configs_file"),
            "number_of_samples": calib_data.get("nb_sample", 500),
        },
    }

    base_to_ref_frame = calib_data.get("base_to_ref_frame")
    ref_frame = calib_data.get("ref_frame")
    if base_to_ref_frame is not None or ref_frame is not None:
        calibration_task["eye_hand"] = {
            "enabled": True,
            "camera_frame": base_to_ref_frame,
            "reference_frame": ref_frame,
        }

    return {"tasks": {"calibration": calibration_task}}


def legacy_identification_to_unified(identif_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a legacy ``identification:`` section to unified format.

    Args:
        identif_data: The dict under the ``identification:`` key of a
            legacy config file.

    Returns:
        A ``{"robot": {"properties": {...}}, "tasks": {"identification":
        {...}}, "custom": {...}}`` fragment. ``robot.properties`` holds
        fields shared across tasks (mechanics, joints); ``custom`` holds
        fields with no current unified-format consumer.
    """
    robot_params = (identif_data.get("robot_params") or [{}])[0]
    problem_params = (identif_data.get("problem_params") or [{}])[0]
    process_params = (identif_data.get("processing_params") or [{}])[0]
    tls_params = (identif_data.get("tls_params") or [{}])[0]

    robot_properties: Dict[str, Any] = {
        "joints": {
            "joint_limits": {
                "position": robot_params.get("q_lim_def"),
                "velocity": robot_params.get("dq_lim_def"),
                "acceleration": robot_params.get("ddq_lim_def"),
                "torque": robot_params.get("tau_lim_def"),
            }
        },
        "mechanics": {
            "friction_coefficients": {
                "viscous": robot_params.get("fv"),
                "static": robot_params.get("fs"),
            },
            "actuator_inertias": robot_params.get("Ia"),
            "joint_offsets": robot_params.get("offset"),
            "reduction_ratios": robot_params.get("reduction_ratio"),
            "ratio_essential": robot_params.get("ratio_essential"),
        },
        "coupling": {
            "has_coupled_wrist": bool(problem_params.get("has_coupled_wrist", False)),
            "Iam6": robot_params.get("Iam6", 0),
            "fvm6": robot_params.get("fvm6", 0),
            "fsm6": robot_params.get("fsm6", 0),
        },
    }
    active_joints = problem_params.get("active_joints")
    if active_joints:
        robot_properties["joints"]["active_joints"] = active_joints

    ts = process_params.get("ts")
    sampling_frequency = (1.0 / ts) if ts else 5000.0

    force_torque = problem_params.get("force_torque")
    force_torque_sensors: List[Any] = (
        list(force_torque)
        if isinstance(force_torque, list)
        else ([force_torque] if force_torque else [])
    )

    identification_task: Dict[str, Any] = {
        "enabled": True,
        "type": "dynamic_identification",
        "problem": {
            "include_external_forces": bool(
                problem_params.get("is_external_wrench", False)
            ),
            "use_joint_torques": bool(problem_params.get("is_joint_torques", True)),
            "force_torque_sensors": force_torque_sensors,
            "external_wrench_offsets": bool(
                problem_params.get("external_wrench_offsets", False)
            ),
            "model_components": {
                "friction": bool(problem_params.get("has_friction", True)),
                "actuator_inertia": bool(
                    problem_params.get("has_actuator_inertia", True)
                ),
                "joint_offset": bool(problem_params.get("has_joint_offset", True)),
            },
        },
        "signal_processing": {
            "sampling_frequency": sampling_frequency,
            "cutoff_frequency": process_params.get(
                "cut_off_frequency_butterworth", 100.0
            ),
        },
        "load_configuration": {
            "additional_mass": tls_params.get("mass_load", 0.0),
            "loaded_body_index": tls_params.get("which_body_loaded", 0),
        },
    }

    custom: Dict[str, Any] = {}
    unconsumed_problem = {
        k: problem_params[k] for k in _UNCONSUMED_PROBLEM_PARAMS if k in problem_params
    }
    unconsumed_tls = {
        k: tls_params[k] for k in _UNCONSUMED_TLS_PARAMS if k in tls_params
    }
    if unconsumed_problem:
        custom["legacy_problem_params"] = unconsumed_problem
    if unconsumed_tls:
        custom["legacy_tls_params"] = unconsumed_tls

    result: Dict[str, Any] = {
        "robot": {"properties": robot_properties},
        "tasks": {"identification": identification_task},
    }
    if custom:
        result["custom"] = custom
    return result


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
    """Recursively merge ``src`` into ``dst`` in place (dicts only)."""
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value


def merge_unified_sections(*docs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge fragments produced by ``legacy_*_to_unified`` into one document.

    A legacy file commonly has both ``calibration:`` and ``identification:``
    sections; each converts independently and the results are merged here.
    """
    merged: Dict[str, Any] = {"tasks": {}}
    for doc in docs:
        for key, value in doc.items():
            if key == "tasks":
                merged["tasks"].update(value)
            elif key == "robot":
                merged.setdefault("robot", {"properties": {}})
                _deep_merge(
                    merged["robot"].setdefault("properties", {}),
                    value.get("properties", {}),
                )
            elif key == "custom":
                merged.setdefault("custom", {}).update(value)
            else:
                merged[key] = value
    return merged


def convert_legacy_to_unified(
    legacy_config: Dict[str, Any], template: str = "base"
) -> Dict[str, Any]:
    """Convert a full legacy config dict (both sections, if present).

    Args:
        legacy_config: The parsed legacy YAML (top-level dict with
            ``calibration``/``identification`` keys).
        template: Which template to ``extends:`` — ``"base"`` (default,
            safest — no inherited defaults to reason about),
            ``"manipulator"``, or ``"humanoid"``.

    Returns:
        A complete unified-format config dict, ready to ``yaml.dump``.

    Raises:
        ValueError: If neither ``calibration`` nor ``identification`` is
            present, or ``template`` is not a known key.
    """
    if template not in TEMPLATE_PATHS:
        raise ValueError(
            f"Unknown template {template!r}; choose from {sorted(TEMPLATE_PATHS)}"
        )

    docs = []
    if "calibration" in legacy_config:
        docs.append(legacy_calibration_to_unified(legacy_config["calibration"]))
    if "identification" in legacy_config:
        docs.append(legacy_identification_to_unified(legacy_config["identification"]))
    if not docs:
        raise ValueError(
            "No 'calibration' or 'identification' section found in legacy config"
        )

    merged = merge_unified_sections(*docs)
    for task_name in ("calibration", "identification"):
        merged["tasks"].setdefault(task_name, {"enabled": False})

    result = {"extends": TEMPLATE_PATHS[template]}
    result.update(merged)
    return result


def self_check(
    robot, legacy_config: Dict[str, Any], unified_doc: Dict[str, Any]
) -> Dict[str, Any]:
    """Round-trip ``unified_doc`` back through the real unified parser and
    diff the recovered ``calib_config``/``identif_config`` against the
    original legacy sections.

    This reuses the exact same code path production uses
    (``create_task_config`` + ``unified_to_legacy_*config``) rather than
    reimplementing the flattening logic, so it can't silently diverge from
    what the pipeline actually does.

    Args:
        robot: A robot instance with ``.model``/``.data`` (needed for frame
            validation inside ``unified_to_legacy_config``).
        legacy_config: The original parsed legacy YAML.
        unified_doc: The output of :func:`convert_legacy_to_unified`
            (with ``extends:`` already resolved away, i.e. pass the merged
            dict as if template inheritance had already happened — for a
            from-scratch check, resolve inheritance first via
            ``UnifiedConfigParser`` if the template adds required fields).

    Returns:
        ``{"ok": bool, "mismatches": {"<section>.<field>": {"expected":
        ..., "got": ...}}}``
    """
    from figaroh.utils.config_parser import create_task_config
    from figaroh.calibration.config import unified_to_legacy_config
    from figaroh.identification.config import unified_to_legacy_identif_config

    mismatches: Dict[str, Dict[str, Any]] = {}

    if "calibration" in legacy_config:
        original = legacy_config["calibration"]
        task_config = create_task_config(robot, unified_doc, "calibration")
        recovered = unified_to_legacy_config(robot, task_config)
        checks = {
            "calib_level": ("calib_model", "full_params"),
            "non_geom": ("non_geom", False),
            "base_frame": ("start_frame", "universe"),
            "tool_frame": ("end_frame", None),
            "free_flyer": ("free_flyer", False),
            "base_to_ref_frame": ("base_to_ref_frame", None),
            "ref_frame": ("ref_frame", None),
            "coeff_regularize": ("coeff_regularize", 0.01),
            "outlier_eps": ("outlier_eps", 0.05),
            "data_file": ("data_file", None),
            "nb_sample": ("NbSample", 500),
        }
        for legacy_key, (recovered_key, default) in checks.items():
            expected = original.get(legacy_key, default)
            got = recovered.get(recovered_key, default)
            if expected != got:
                mismatches[f"calibration.{legacy_key}"] = {
                    "expected": expected,
                    "got": got,
                }

    if "identification" in legacy_config:
        original = legacy_config["identification"]
        task_config = create_task_config(robot, unified_doc, "identification")
        recovered = unified_to_legacy_identif_config(robot, task_config)
        robot_params = (original.get("robot_params") or [{}])[0]
        problem_params = (original.get("problem_params") or [{}])[0]
        checks = {
            "fv": (robot_params.get("fv"), recovered.get("fv")),
            "fs": (robot_params.get("fs"), recovered.get("fs")),
            "Ia": (robot_params.get("Ia"), recovered.get("Ia")),
            "has_friction": (
                problem_params.get("has_friction", True),
                recovered.get("has_friction"),
            ),
            "has_coupled_wrist": (
                problem_params.get("has_coupled_wrist", True),
                recovered.get("has_coupled_wrist"),
            ),
        }
        for field, (expected, got) in checks.items():
            if expected != got:
                mismatches[f"identification.{field}"] = {
                    "expected": expected,
                    "got": got,
                }

    return {"ok": not mismatches, "mismatches": mismatches}


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a legacy figaroh config to the unified format."
    )
    parser.add_argument("--input", required=True, help="Path to the legacy config YAML")
    parser.add_argument(
        "--output", required=True, help="Path to write the unified config YAML"
    )
    parser.add_argument(
        "--template",
        choices=sorted(TEMPLATE_PATHS),
        default="base",
        help="Template to extend (default: base — safest, no inherited defaults)",
    )
    parser.add_argument(
        "--urdf",
        default=None,
        help="Path to the robot URDF. If given, runs the round-trip self-check.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    with open(args.input) as f:
        legacy_config = yaml.safe_load(f)

    unified_doc = convert_legacy_to_unified(legacy_config, template=args.template)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        yaml.dump(unified_doc, f, sort_keys=False, default_flow_style=False)
    print(f"Wrote unified config to {args.output}")

    if args.urdf:
        import pinocchio as pin

        class _Robot:
            def __init__(self, model):
                self.model = model
                self.data = model.createData()
                self.q0 = pin.neutral(model)

        robot = _Robot(pin.buildModelFromUrdf(args.urdf))
        # self_check needs template inheritance already resolved (kinematics/
        # measurements/etc. must be present on unified_doc directly) --
        # convert_legacy_to_unified already produces a fully-specified
        # tasks.* block without relying on template defaults, so the merged
        # doc itself is a valid input.
        report = self_check(robot, legacy_config, copy.deepcopy(unified_doc))
        if report["ok"]:
            print("Self-check: OK — round-trip matches the original legacy config.")
        else:
            print("Self-check: MISMATCHES found:")
            for field, diff in report["mismatches"].items():
                print(f"  {field}: expected={diff['expected']!r} got={diff['got']!r}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
