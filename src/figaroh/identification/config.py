# Copyright [2021-2026] Thanh Nguyen
# Copyright [2022-2024] [CNRS, Toward SAS]

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
Configuration parsing and parameter management for robot identification.

This module handles all configuration-related functionality including:
- YAML configuration file parsing
- Unified to legacy config format conversion
- Parameter extraction and validation
- Signal processing and mechanical parameter management
"""

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Export public API
__all__ = [
    "unified_to_legacy_identif_config",
]


def unified_to_legacy_identif_config(robot, unified_identif_config) -> dict:
    """Convert unified identification format to legacy identif_config format.

    Maps the unified identification configuration structure to the
    ``identif_config`` dict the identification classes consume.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model and data
        unified_identif_config (dict): Configuration from create_task_config

    Returns:
        dict: Identification configuration (``identif_config``)

    Example:
        >>> unified_config = create_task_config(robot, parsed_config,
        ...                                    "identification")
        >>> legacy_config = unified_to_legacy_identif_config(robot,
        ...                                                  unified_config)
        >>> # legacy_config has the identif_config keys
    """
    # Initialize output configuration
    identif_config = {}

    # Extract unified config sections
    mechanics = unified_identif_config.get("mechanics", {})
    joints = unified_identif_config.get("joints", {})
    problem = unified_identif_config.get("problem", {})
    coupling = unified_identif_config.get("coupling", {})
    signal_processing = unified_identif_config.get("signal_processing", {})

    # 1. Extract basic robot information
    identif_config["robot_name"] = robot.model.name
    identif_config["instance"] = unified_identif_config.get("instance", {})

    # 2. Extract signal processing parameters
    _extract_signal_processing_params(identif_config, signal_processing)

    # 3. Extract joint limits
    _extract_joint_limits(identif_config, joints)

    # 4. Extract problem configuration
    _extract_problem_config(identif_config, problem)

    # 5. Extract mechanical parameters
    _extract_mechanical_params(identif_config, mechanics)

    # 6. Extract coupling parameters
    _extract_coupling_params(identif_config, coupling)

    # 7. Extract load parameters (defaults)
    _extract_load_params(identif_config)

    # 8. Optional physical consistency configuration (v0.4.1, default-off)
    physical_consistency = unified_identif_config.get("physical_consistency", {})
    if isinstance(physical_consistency, dict) and physical_consistency:
        identif_config["physical_consistency"] = physical_consistency

    # 9. Optional reconstruction configuration (v0.4.2, default-off)
    reconstruction = unified_identif_config.get("reconstruction", {})
    if isinstance(reconstruction, dict) and reconstruction:
        identif_config["reconstruction"] = reconstruction

    _extract_selection_config(identif_config, unified_identif_config)

    # 10. Optional held-out validation dataset (separate file/directory,
    # never a split of the training data). Mirrors calibration's
    # tasks.calibration.data.validation_data_file.
    data_section = unified_identif_config.get("data", {})
    if not isinstance(data_section, dict):
        data_section = {}
    identif_config["validation_data_file"] = data_section.get(
        "validation_data_file", ""
    )

    return identif_config


SELECT_STAGES = ("fit", "reconstruction", "physical_fit")

PHYSICAL_FIT_DEFAULTS = {
    "enabled": False,
    "solver": "cvxopt",
    "second_solver": None,
    "mass_min": 1e-6,
    "prior_weight": 1e-6,
    "max_seconds": None,
}


def _extract_selection_config(identif_config, source):
    """Parse the optional ``select_stage`` and ``physical_fit`` keys.

    Both are additive: when absent nothing is written, so legacy configs
    keep their exact ``identif_config``. ``select_stage`` must name a
    selectable stage; ``projected`` is deliberately not selectable (the
    physical-consistency projection is of the nominal model, not the fit).

    Raises:
        ValueError: unknown ``select_stage`` or ``physical_fit`` key/type.
    """
    if "select_stage" in source and source["select_stage"] is not None:
        stage = str(source["select_stage"]).strip().lower()
        if stage not in SELECT_STAGES:
            raise ValueError(
                f"select_stage={source['select_stage']!r} is not selectable; "
                f"choose one of {SELECT_STAGES}"
                + (
                    " ('projected' is a projection of the nominal model, "
                    "not of the fit)"
                    if stage == "projected"
                    else ""
                )
            )
        identif_config["select_stage"] = stage

    pf = source.get("physical_fit")
    if isinstance(pf, list):
        pf = pf[0] if pf else None
    if pf is not None:
        if not isinstance(pf, dict):
            raise ValueError("physical_fit must be a mapping")
        unknown = set(pf) - set(PHYSICAL_FIT_DEFAULTS)
        if unknown:
            raise ValueError(
                f"unknown physical_fit keys {sorted(unknown)}; "
                f"allowed: {sorted(PHYSICAL_FIT_DEFAULTS)}"
            )
        merged = dict(PHYSICAL_FIT_DEFAULTS)
        merged.update(pf)
        identif_config["physical_fit"] = merged


def _extract_signal_processing_params(identif_config, signal_processing):
    """Extract signal processing parameters.

    Args:
        identif_config (dict): Configuration dictionary to update
        signal_processing (dict): Signal processing section from unified config
    """
    sampling_freq = signal_processing.get("sampling_frequency", 5000.0)
    ts = 1.0 / sampling_freq
    cutoff_freq = signal_processing.get("cutoff_frequency", 100.0)

    identif_config["nb_samples"] = int(1 / ts)
    identif_config["ts"] = ts
    identif_config["cut_off_frequency_butterworth"] = cutoff_freq

    # Build filter configuration dictionary
    filter_config_ = {}
    filter_config_["cutoff_frequency"] = cutoff_freq
    filter_config_["sampling_frequency"] = sampling_freq
    filter_config_["filter_type"] = signal_processing.get("filter_type", "butterworth")
    filter_config_["differentiation_method"] = signal_processing.get(
        "differentiation_method", "gradient"
    )
    # _apply_filters takes f_sample/f_butter/nbutter; default them from the
    # section so the configured rates reach the Butterworth filter.
    # Explicit filter_params entries still win.
    filter_params = dict(signal_processing.get("filter_params") or {})
    filter_params.setdefault("f_sample", sampling_freq)
    filter_params.setdefault("f_butter", cutoff_freq)
    filter_params.setdefault("nbutter", signal_processing.get("filter_order", 4))
    if not 0 < filter_params["f_butter"] < filter_params["f_sample"] / 2:
        raise ValueError(
            f"signal_processing cutoff ({filter_params['f_butter']} Hz) must be "
            f"between 0 and the Nyquist frequency "
            f"({filter_params['f_sample'] / 2} Hz)"
        )
    filter_config_["filter_params"] = filter_params

    identif_config["filter_config"] = filter_config_


def _extract_joint_limits(identif_config, joints):
    """Extract joint limit parameters.

    Args:
        identif_config (dict): Configuration dictionary to update
        joints (dict): Joints section from unified config
    """
    identif_config["active_joints"] = joints.get("active_joints", [])

    joint_limits = joints.get("joint_limits", {})

    identif_config["q_lim_def"] = joint_limits.get("position", [])
    identif_config["dq_lim_def"] = joint_limits.get("velocity", [])
    identif_config["ddq_lim_def"] = joint_limits.get("acceleration", [])
    identif_config["torque_lim_def"] = joint_limits.get("torque", [])


def _extract_problem_config(identif_config, problem):
    """Extract problem configuration parameters.

    Args:
        identif_config (dict): Configuration dictionary to update
        problem (dict): Problem section from unified config
    """
    model_components = problem.get("model_components", {})

    # External forces and torques
    identif_config["is_external_wrench"] = problem.get("include_external_forces", False)
    identif_config["is_joint_torques"] = problem.get("use_joint_torques", True)
    identif_config["external_wrench_offsets"] = problem.get(
        "external_wrench_offsets", False
    )

    # Force/torque sensor
    ft_sensors = problem.get("force_torque_sensors", [])
    identif_config["force_torque"] = ft_sensors[0] if ft_sensors else None

    # Model components
    identif_config["has_friction"] = model_components.get("friction", True)
    identif_config["has_actuator_inertia"] = model_components.get(
        "actuator_inertia", True
    )
    identif_config["has_joint_offset"] = model_components.get("joint_offset", True)

    # Solver: weighted least squares refinement (see BaseIdentification.solve)
    identif_config["wls"] = problem.get("wls", False)

    # QR rank threshold relative to the largest pivot (None: absolute only)
    identif_config["qr_relative_tolerance"] = problem.get("qr_relative_tolerance")

    # Active joints whose effort enters the fit (None: all of them); the
    # others keep their kinematics in the regressor
    fit_joints = problem.get("torque_fit_joints")
    if fit_joints is not None and not isinstance(fit_joints, list):
        raise ValueError("problem.torque_fit_joints must be a list of joint names")
    identif_config["torque_fit_joints"] = fit_joints


def _extract_mechanical_params(identif_config, mechanics):
    """Extract mechanical parameters (friction, inertia, ratios).

    Args:
        identif_config (dict): Configuration dictionary to update
        mechanics (dict): Mechanics section from unified config
    """
    # Friction coefficients
    friction_coeffs = mechanics.get("friction_coefficients", {})
    identif_config["fv"] = friction_coeffs.get("viscous", [])
    identif_config["fs"] = friction_coeffs.get("static", [])

    # Actuator inertias and joint offsets
    identif_config["Ia"] = mechanics.get("actuator_inertias", [])
    identif_config["off"] = mechanics.get("joint_offsets", [])

    # Reduction ratios
    identif_config["reduction_ratio"] = mechanics.get("reduction_ratios", [])
    # threshold for essential parameters (C. Pham et al. 1995)
    identif_config["ratio_essential"] = mechanics.get("ratio_essential", 30.0)


def _extract_coupling_params(identif_config, coupling):
    """Extract coupling parameters for coupled joints.

    Args:
        identif_config (dict): Configuration dictionary to update
        coupling (dict): Coupling section from unified config
    """
    identif_config["has_coupled_wrist"] = coupling.get("has_coupled_wrist", True)
    identif_config["Iam6"] = coupling.get("Iam6", 0)
    identif_config["fvm6"] = coupling.get("fvm6", 0)
    identif_config["fsm6"] = coupling.get("fsm6", 0)


def _extract_load_params(identif_config):
    """Extract load parameters with default values.

    Args:
        identif_config (dict): Configuration dictionary to update
    """
    identif_config["mass_load"] = 0.0
    identif_config["which_body_loaded"] = 0.0
