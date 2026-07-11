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
Configuration parsing and parameter management for robot identification.

This module handles all configuration-related functionality including:
- YAML configuration file parsing
- Unified to legacy config format conversion
- Parameter extraction and validation
- Signal processing and mechanical parameter management
"""

# Export public API
__all__ = [
    "get_param_from_yaml",
    "unified_to_legacy_identif_config",
    "get_param_from_yaml_legacy",
]


def get_param_from_yaml(robot, identif_data):
    """Parse identification parameters from YAML configuration file.

    Extracts robot parameters, problem settings, signal processing options and
    total least squares parameters from a YAML config file.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model
        identif_data (dict): YAML configuration containing:
            - robot_params: Joint limits, friction, inertia settings
            - problem_params: External wrench, friction, actuator settings
            - processing_params: Sample rate, filter settings
            - tls_params: Load mass and location

    Returns:
        dict: Parameter dictionary with unified settings

    Example:
        >>> config = yaml.safe_load(config_file)
        >>> params = get_param_from_yaml(robot, config)
        >>> print(params["nb_samples"])
    """

    # robot_name: anchor as a reference point for executing
    robot_name = robot.model.name

    robots_params = identif_data["robot_params"][0]
    problem_params = identif_data["problem_params"][0]
    process_params = identif_data["processing_params"][0]
    tls_params = identif_data["tls_params"][0]

    identif_config = {
        "robot_name": robot_name,
        "nb_samples": int(1 / (process_params["ts"])),
        "q_lim_def": robots_params["q_lim_def"],
        "dq_lim_def": robots_params["dq_lim_def"],
        "is_external_wrench": problem_params["is_external_wrench"],
        "is_joint_torques": problem_params["is_joint_torques"],
        "force_torque": problem_params["force_torque"],
        "external_wrench_offsets": problem_params["external_wrench_offsets"],
        "has_friction": problem_params["has_friction"],
        "fv": robots_params["fv"],
        "fs": robots_params["fs"],
        "has_actuator_inertia": problem_params["has_actuator_inertia"],
        "Ia": robots_params["Ia"],
        "has_joint_offset": problem_params["has_joint_offset"],
        "off": robots_params["offset"],
        "has_coupled_wrist": problem_params["has_coupled_wrist"],
        "Iam6": robots_params["Iam6"],
        "fvm6": robots_params["fvm6"],
        "fsm6": robots_params["fsm6"],
        "reduction_ratio": robots_params["reduction_ratio"],
        "ratio_essential": robots_params["ratio_essential"],
        "cut_off_frequency_butterworth": process_params[
            "cut_off_frequency_butterworth"
        ],
        "ts": process_params["ts"],
        "mass_load": tls_params["mass_load"],
        "which_body_loaded": tls_params["which_body_loaded"],
    }

    # Optional physical consistency configuration (v0.4.1, default-off)
    physical_consistency = identif_data.get("physical_consistency")
    if isinstance(physical_consistency, list):
        physical_consistency = physical_consistency[0] if physical_consistency else None
    if isinstance(physical_consistency, dict):
        identif_config["physical_consistency"] = physical_consistency

    # Optional reconstruction configuration (v0.4.2, default-off)
    reconstruction = identif_data.get("reconstruction")
    if isinstance(reconstruction, list):
        reconstruction = reconstruction[0] if reconstruction else None
    if isinstance(reconstruction, dict):
        identif_config["reconstruction"] = reconstruction

    # Optional held-out validation dataset (separate from training data,
    # never a split of it). Mirrors BaseCalibration's validation_data_file.
    identif_config["validation_data_file"] = identif_data.get(
        "validation_data_file", ""
    )

    return identif_config


def unified_to_legacy_identif_config(robot, unified_identif_config) -> dict:
    """Convert unified identification format to legacy identif_config format.

    Maps the new unified identification configuration structure to produce
    the exact same output as get_param_from_yaml. This ensures backward
    compatibility while using the new unified parser.

    Args:
        robot (pin.RobotWrapper): Robot instance containing model and data
        unified_identif_config (dict): Configuration from create_task_config

    Returns:
        dict: Identification configuration matching get_param_from_yaml output

    Example:
        >>> unified_config = create_task_config(robot, parsed_config,
        ...                                    "identification")
        >>> legacy_config = unified_to_legacy_identif_config(robot,
        ...                                                  unified_config)
        >>> # legacy_config has same keys as get_param_from_yaml output
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
    filter_config_["filter_params"] = signal_processing.get("filter_params", {})

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


# Backward compatibility wrapper for get_param_from_yaml
def get_param_from_yaml_legacy(robot, identif_data) -> dict:
    """Legacy identification parameter parser for backward compatibility.

    This is the original implementation. New code should use the unified
    config parser from figaroh.utils.config_parser.

    Args:
        robot: Robot instance
        identif_data: Identification data dictionary

    Returns:
        Identification configuration dictionary
    """
    # Keep the original implementation here for compatibility
    return get_param_from_yaml(robot, identif_data)


# Import the new unified parser as the default
try:
    from ..utils.config_parser import (
        get_param_from_yaml as unified_get_param_from_yaml,
    )

    # Replace the function with unified version while maintaining signature
    def get_param_from_yaml_unified(robot, identif_data) -> dict:
        """Enhanced parameter parser using unified configuration system.

        This function provides backward compatibility while using the new
        unified configuration parser when possible.

        Args:
            robot: Robot instance
            identif_data: Configuration data (dict or file path)

        Returns:
            Identification configuration dictionary
        """
        try:
            return unified_get_param_from_yaml(robot, identif_data, "identification")
        except Exception as e:
            # Fall back to legacy parser if unified parser fails
            import warnings

            warnings.warn(
                f"Unified parser failed ({e}), falling back to legacy parser. "
                "Consider updating your configuration format.",
                UserWarning,
            )
            return get_param_from_yaml_legacy(robot, identif_data)

    # Keep the old function available but with warning
    def get_param_from_yaml_with_warning(robot, identif_data) -> dict:
        """Original function with deprecation notice."""
        import warnings

        warnings.warn(
            "Direct use of get_param_from_yaml is deprecated. "
            "Consider using the unified config parser from "
            "figaroh.utils.config_parser",
            DeprecationWarning,
            stacklevel=2,
        )
        return get_param_from_yaml_unified(robot, identif_data)

except ImportError:
    # If unified parser is not available, keep using original function
    pass
