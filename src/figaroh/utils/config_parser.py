"""
Unified Configuration Parser for FIGAROH

This module provides a unified configuration parsing system that supports:
- Template inheritance and extension
- Variant configurations
- Variable expansion
- Backward compatibility with legacy formats
- Schema validation
"""

from typing import Dict, Any, Optional, Union
import yaml
import os
import re
from pathlib import Path
import logging
from copy import deepcopy
from dataclasses import dataclass
from .error_handling import ConfigurationError

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


@dataclass
class ConfigMetadata:
    """Configuration metadata container."""

    schema_version: str = "2.0"
    config_type: str = "robot_configuration"
    source_file: Optional[str] = None
    variant: Optional[str] = None
    resolved_at: Optional[str] = None


class UnifiedConfigParser:
    """Unified configuration parser supporting inheritance, templates, and validation."""

    def __init__(self, config_path: Union[str, Path], variant: Optional[str] = None):
        """Initialize parser with configuration file path.

        Args:
            config_path: Path to configuration file
            variant: Optional variant name to use from configuration
        """
        self.config_path = Path(config_path).resolve()
        self.variant = variant
        self.logger = logging.getLogger(self.__class__.__name__)
        self._template_cache = {}
        self._variable_cache = {}

        # Validate config file exists
        if not self.config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {self.config_path}"
            )

    def parse(self) -> Dict[str, Any]:
        """Parse configuration with inheritance and validation.

        Returns:
            Complete configuration dictionary with metadata

        Raises:
            ConfigurationError: If parsing or validation fails
        """
        try:
            self.logger.info(f"Parsing configuration: {self.config_path}")

            # Load base configuration
            config = self._load_config_file(self.config_path)

            # Handle inheritance/extensions
            if "extends" in config:
                config = self._resolve_inheritance(config)

            # Apply variant if specified
            if self.variant:
                config = self._apply_variant(config, self.variant)

            # Resolve task inheritance
            config = self._resolve_task_inheritance(config)

            # Expand variables
            config = self._expand_variables(config)

            # Validate configuration
            self._validate_configuration(config)

            # Add metadata
            config["_metadata"] = ConfigMetadata(
                source_file=str(self.config_path),
                variant=self.variant,
                resolved_at=str(Path.cwd()),
            ).__dict__

            self.logger.info("Configuration parsing completed successfully")
            return config

        except Exception as e:
            self.logger.error(f"Failed to parse configuration: {e}")
            raise ConfigurationError(
                f"Failed to parse configuration {self.config_path}: {e}"
            ) from e

    def _load_config_file(self, file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file with error handling."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)

            if not isinstance(content, dict):
                raise ConfigurationError(
                    f"Configuration file must contain a YAML dictionary"
                )

            return content

        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in file {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file {file_path}: {e}")

    def _resolve_inheritance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve extends/inheritance relationships recursively."""
        extends = config.pop("extends", None)
        if not extends:
            return config

        # Resolve template path
        template_path = self._resolve_template_path(extends)

        # Load parent configuration (with caching)
        if template_path not in self._template_cache:
            self.logger.debug(f"Loading template: {template_path}")
            parent_config = self._load_config_file(template_path)

            # Recursively resolve parent inheritance
            if "extends" in parent_config:
                parent_config = self._resolve_inheritance(parent_config)

            self._template_cache[template_path] = parent_config
        else:
            parent_config = self._template_cache[template_path]

        # Deep merge configurations (child overrides parent)
        merged_config = self._deep_merge(deepcopy(parent_config), config)

        self.logger.debug(f"Merged configuration with template: {extends}")
        return merged_config

    def _resolve_template_path(self, template_reference: str) -> Path:
        """Resolve template path from reference.

        Supports multiple template path formats:
        1. templates/filename.yaml - looks for templates dir relative to config
        2. ../templates/filename.yaml - relative path from config directory
        3. filename.yaml - relative to config directory
        4. /absolute/path/filename.yaml - absolute path
        """
        template_path = None

        # Handle absolute paths
        if Path(template_reference).is_absolute():
            template_path = Path(template_reference)

        # Handle templates/ prefix - search in templates directories
        elif template_reference.startswith("templates/"):
            template_filename = template_reference[10:]  # Remove "templates/"

            # Search locations for templates directory
            search_locations = [
                self.config_path.parent / "templates",  # Same directory as config
                self.config_path.parent.parent / "templates",  # Parent directory
                self.config_path.parent.parent.parent / "templates",  # Grandparent
                Path.cwd() / "templates",  # Current working directory
            ]

            for templates_dir in search_locations:
                candidate_path = templates_dir / template_filename
                if candidate_path.exists():
                    template_path = candidate_path
                    break

            if template_path is None:
                searched_paths = [str(loc) for loc in search_locations]
                raise ConfigurationError(
                    f"Template '{template_reference}' not found. "
                    f"Searched in: {', '.join(searched_paths)}"
                )

        # Handle relative paths
        else:
            template_path = self.config_path.parent / template_reference

        # Validate final path exists
        if not template_path.exists():
            raise ConfigurationError(
                f"Template file not found: {template_path}. "
                f"Config directory: {self.config_path.parent}"
            )

        return template_path.resolve()

    def _resolve_task_inheritance(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Resolve inherits_from relationships within tasks."""
        if "tasks" not in config:
            return config

        tasks = config["tasks"]
        resolved_tasks = {}

        # Resolve dependencies using topological sort
        def resolve_task(task_name: str, visited: set, temp_visited: set):
            if task_name in temp_visited:
                raise ConfigurationError(
                    f"Circular task inheritance detected involving: {task_name}"
                )
            if task_name in visited:
                return

            temp_visited.add(task_name)

            task_config = tasks[task_name]
            if "inherits_from" in task_config:
                parent_task = task_config["inherits_from"]
                if parent_task not in tasks:
                    raise ConfigurationError(
                        f"Parent task '{parent_task}' not found for '{task_name}'"
                    )

                # Resolve parent first
                resolve_task(parent_task, visited, temp_visited)

                # Merge with parent
                parent_config = deepcopy(
                    resolved_tasks.get(parent_task, tasks[parent_task])
                )
                task_config_copy = deepcopy(task_config)
                task_config_copy.pop("inherits_from")

                resolved_tasks[task_name] = self._deep_merge(
                    parent_config, task_config_copy
                )
            else:
                resolved_tasks[task_name] = deepcopy(task_config)

            temp_visited.remove(task_name)
            visited.add(task_name)

        visited = set()
        for task_name in tasks:
            resolve_task(task_name, visited, set())

        config["tasks"] = resolved_tasks
        return config

    def _apply_variant(
        self, config: Dict[str, Any], variant_name: str
    ) -> Dict[str, Any]:
        """Apply a specific variant configuration."""
        if "variants" not in config:
            raise ConfigurationError(
                f"No variants defined, cannot apply variant: {variant_name}"
            )

        variants = config["variants"]
        if variant_name not in variants:
            available = list(variants.keys())
            raise ConfigurationError(
                f"Variant '{variant_name}' not found. Available variants: {available}"
            )

        variant_config = deepcopy(variants[variant_name])

        # Handle variant inheritance from existing tasks/sections
        if "extends" in variant_config:
            extends_path = variant_config.pop("extends")
            base_config = self._get_nested_value(config, extends_path)
            if base_config is None:
                raise ConfigurationError(
                    f"Variant extension path not found: {extends_path}"
                )
            variant_config = self._deep_merge(deepcopy(base_config), variant_config)

        # Apply variant as override to the entire configuration
        result_config = deepcopy(config)
        result_config = self._deep_merge(result_config, variant_config)

        # Remove variants section from final config
        result_config.pop("variants", None)

        self.logger.info(f"Applied variant: {variant_name}")
        return result_config

    def _deep_merge(
        self, base: Dict[str, Any], override: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Deep merge two dictionaries with special handling for lists."""
        result = deepcopy(base)

        for key, value in override.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    # Recursively merge dictionaries
                    result[key] = self._deep_merge(result[key], value)
                elif isinstance(result[key], list) and isinstance(value, list):
                    # For lists, extend rather than replace (configurable behavior)
                    if key.endswith("_extend"):
                        # Special suffix to indicate list extension
                        real_key = key[:-8]  # Remove "_extend"
                        if real_key in result:
                            result[real_key].extend(value)
                        else:
                            result[real_key] = value
                    else:
                        # Default: replace list
                        result[key] = deepcopy(value)
                else:
                    # Replace value
                    result[key] = deepcopy(value)
            else:
                # New key
                result[key] = deepcopy(value)

        return result

    def _get_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        """Get nested value using dot notation (e.g., 'tasks.calibration.parameters')."""
        keys = path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None

        return current

    def _expand_variables(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Expand ${VARIABLE} references in configuration."""

        def expand_value(value, context_path: str = ""):
            if isinstance(value, str):
                return self._expand_string_variables(value, config, context_path)
            elif isinstance(value, dict):
                return {
                    k: expand_value(v, f"{context_path}.{k}" if context_path else k)
                    for k, v in value.items()
                }
            elif isinstance(value, list):
                return [
                    expand_value(v, f"{context_path}[{i}]") for i, v in enumerate(value)
                ]
            else:
                return value

        return expand_value(config)

    def _expand_string_variables(
        self, text: str, config: Dict[str, Any], context_path: str = ""
    ) -> str:
        """Expand variables in a string value."""
        if "${" not in text:
            return text

        pattern = r"\$\{([^}]+)\}"

        def replacer(match):
            var_name = match.group(1).strip()

            # Try different variable sources in order
            value = (
                os.environ.get(var_name)  # Environment variables
                or self._get_nested_value(config, var_name)  # Config references
                or self._variable_cache.get(var_name)  # Cached variables
                or match.group(0)  # Keep original if not found
            )

            if value == match.group(0):
                self.logger.warning(
                    f"Unresolved variable '{var_name}' in {context_path}"
                )

            return str(value) if value is not None else ""

        try:
            result = re.sub(pattern, replacer, text)
            return result
        except Exception as e:
            self.logger.warning(f"Variable expansion failed for '{text}': {e}")
            return text

    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate configuration structure and required fields."""
        self.logger.debug("Validating configuration structure")

        # Basic validation
        if not isinstance(config, dict):
            raise ConfigurationError("Configuration must be a dictionary")

        # Validate meta section if present
        if "meta" in config:
            meta = config["meta"]
            schema_version = meta.get("schema_version")
            if schema_version and not self._is_compatible_version(schema_version):
                self.logger.warning(
                    f"Configuration schema version {schema_version} may not be compatible"
                )

        # Validate robot section
        if "robot" in config:
            self._validate_robot_section(config["robot"])

        # Validate tasks if present
        if "tasks" in config:
            self._validate_tasks_section(config["tasks"])

    def _is_compatible_version(self, version: str) -> bool:
        """Check if schema version is compatible."""
        try:
            major, minor = map(int, version.split(".")[:2])
            return major <= 2  # Accept versions 1.x and 2.x
        except:
            return False

    def _validate_robot_section(self, robot_config: Dict[str, Any]) -> None:
        """Validate robot configuration section."""
        if "name" not in robot_config:
            raise ConfigurationError("Robot name is required")

        if "properties" in robot_config:
            properties = robot_config["properties"]

            # Validate joint configuration if present
            if "joints" in properties:
                joints = properties["joints"]
                if "active_joints" in joints and not isinstance(
                    joints["active_joints"], list
                ):
                    raise ConfigurationError("active_joints must be a list")

    def _validate_tasks_section(self, tasks_config: Dict[str, Any]) -> None:
        """Validate tasks configuration section."""
        for task_name, task_config in tasks_config.items():
            if not isinstance(task_config, dict):
                raise ConfigurationError(f"Task '{task_name}' must be a dictionary")

            # Validate task type if specified
            task_type = task_config.get("type")
            if task_type:
                self._validate_task_type(task_name, task_type, task_config)

    def _validate_task_type(
        self, task_name: str, task_type: str, task_config: Dict[str, Any]
    ) -> None:
        """Validate specific task type configuration."""
        if task_type == "kinematic_calibration":
            required_sections = ["kinematics", "measurements"]
            for section in required_sections:
                if section not in task_config:
                    raise ConfigurationError(
                        f"Task '{task_name}' of type '{task_type}' missing required section: {section}"
                    )

        elif task_type == "dynamic_identification":
            required_sections = ["problem", "signal_processing"]
            for section in required_sections:
                if section not in task_config:
                    raise ConfigurationError(
                        f"Task '{task_name}' of type '{task_type}' missing required section: {section}"
                    )


def create_task_config(
    robot, unified_config: Dict[str, Any], task_name: str
) -> Dict[str, Any]:
    """Create task-specific configuration from unified config.

    Args:
        robot: Robot instance
        unified_config: Parsed unified configuration
        task_name: Name of task to extract configuration for

    Returns:
        Task-specific configuration dictionary

    Raises:
        ConfigurationError: If task not found or invalid
    """
    if "tasks" not in unified_config or task_name not in unified_config["tasks"]:
        available_tasks = list(unified_config.get("tasks", {}).keys())
        raise ConfigurationError(
            f"Task '{task_name}' not found in configuration. Available tasks: {available_tasks}"
        )

    task_config = unified_config["tasks"][task_name]
    robot_config = unified_config.get("robot", {})

    # Check if task is enabled
    if not task_config.get("enabled", True):
        raise ConfigurationError(f"Task '{task_name}' is disabled in configuration")

    # Combine robot properties with task configuration
    combined_config = {
        "robot_name": robot_config.get("name", robot.model.name),
        "task_type": task_config.get("type", task_name),
    }

    # Add robot properties
    if "properties" in robot_config:
        combined_config.update(robot_config["properties"])

    # Physical-asset identity, distinct from the model type above (e.g.
    # "ur10" is shared by every UR10; robot.instance.asset_id identifies
    # one physical unit). Optional — omitted configs simply carry no
    # instance block, and provenance capture renders that as
    # "unspecified unit" rather than guessing.
    if "instance" in robot_config:
        combined_config["instance"] = robot_config["instance"]

    # Add task configuration (excluding metadata)
    task_data = {
        k: v
        for k, v in task_config.items()
        if not k.startswith("_") and k not in ["enabled", "type"]
    }
    combined_config.update(task_data)

    # Add custom parameters
    if "custom" in unified_config:
        combined_config["custom"] = unified_config["custom"]

    # Add environment settings that might be relevant
    if "environment" in unified_config:
        env = unified_config["environment"]
        combined_config.update(
            {
                "working_directory": env.get("working_directory", "."),
                "data_directory": env.get("data_directory", "data"),
                "results_directory": env.get("results_directory", "results"),
            }
        )

    return combined_config


def parse_configuration(
    config_path: Union[str, Path], variant: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to parse a configuration file.

    Args:
        config_path: Path to configuration file
        variant: Optional variant name to apply

    Returns:
        Parsed configuration dictionary
    """
    parser = UnifiedConfigParser(config_path, variant)
    return parser.parse()


def get_param_from_yaml(
    robot,
    config_data: Union[Dict[str, Any], str, Path],
    task_type: str = "auto",
    variant: Optional[str] = None,
) -> Dict[str, Any]:
    """Unified parameter parser with backward compatibility.

    Args:
        robot: Robot instance
        config_data: Configuration data (dict), or path to config file
        task_type: Type of task configuration to extract ("calibration", "identification", "auto")
        variant: Optional variant to apply

    Returns:
        Task-specific parameter dictionary

    Raises:
        ConfigurationError: If parsing fails or task not found
    """
    # Handle file path input
    if isinstance(config_data, (str, Path)):
        unified_config = parse_configuration(config_data, variant)

        # Auto-detect task type if needed
        if task_type == "auto":
            tasks = unified_config.get("tasks", {})
            enabled_tasks = [
                name for name, config in tasks.items() if config.get("enabled", True)
            ]
            if len(enabled_tasks) == 1:
                task_type = enabled_tasks[0]
            else:
                raise ConfigurationError(
                    f"Cannot auto-detect task type. Available tasks: {enabled_tasks}"
                )

        return create_task_config(robot, unified_config, task_type)

    # Handle dictionary input (could be legacy or unified)
    if not isinstance(config_data, dict):
        raise ConfigurationError("config_data must be a dictionary or file path")

    # Check if it's already a unified configuration
    if "tasks" in config_data:
        # Unified format
        if task_type == "auto":
            tasks = config_data.get("tasks", {})
            enabled_tasks = [
                name for name, config in tasks.items() if config.get("enabled", True)
            ]
            if len(enabled_tasks) == 1:
                task_type = enabled_tasks[0]
            else:
                raise ConfigurationError(
                    f"Cannot auto-detect task type. Available tasks: {enabled_tasks}"
                )

        return create_task_config(robot, config_data, task_type)

    # Legacy format - use existing parsers
    return _parse_legacy_format(robot, config_data, task_type)


def _parse_legacy_format(
    robot, config_data: Dict[str, Any], task_type: str
) -> Dict[str, Any]:
    """Parse legacy configuration format using existing parsers."""
    # Import existing parsers for backward compatibility
    calibration_indicators = ["markers", "calib_level", "base_frame", "tool_frame"]
    identification_indicators = ["robot_params", "problem_params", "processing_params"]

    has_calibration = any(key in config_data for key in calibration_indicators)
    has_identification = any(key in config_data for key in identification_indicators)

    # Auto-detect if needed
    if task_type == "auto":
        if has_calibration and not has_identification:
            task_type = "calibration"
        elif has_identification and not has_calibration:
            task_type = "identification"
        elif has_calibration and has_identification:
            # Both sections present, need explicit task type
            raise ConfigurationError(
                "Both calibration and identification sections found. "
                "Please specify task_type explicitly."
            )
        else:
            raise ConfigurationError(
                "Cannot determine task type from legacy configuration"
            )

    if task_type == "calibration" and has_calibration:
        from ..calibration.calibration_tools import (
            get_param_from_yaml as legacy_cal_parser,
        )

        return legacy_cal_parser(robot, config_data)
    elif task_type == "identification" and has_identification:
        from ..identification.identification_tools import (
            get_param_from_yaml as legacy_id_parser,
        )

        return legacy_id_parser(robot, config_data)
    else:
        raise ConfigurationError(
            f"Cannot parse legacy format for task type: {task_type}. "
            f"Required sections not found in configuration."
        )


def is_unified_config(config_file: Union[str, Path]) -> bool:
    """Check if a configuration file is using the unified format.

    Args:
        config_file: Path to configuration file

    Returns:
        True if unified format, False if legacy format
    """
    try:
        config_path = Path(config_file)
        if not config_path.exists():
            return False

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        if not isinstance(config, dict):
            return False

        # Check for unified format indicators
        unified_indicators = [
            "_metadata",
            "schema_version",
            "extends",
            "variants",
            "tasks",
            "common",
            "variables",
        ]

        # If any unified indicators are present, consider it unified
        for indicator in unified_indicators:
            if indicator in config:
                logger.debug(f"Detected unified config indicator: {indicator}")
                return True

        # Check if the file has task inheritance patterns
        for key, value in config.items():
            if isinstance(value, dict) and "inherits_from" in value:
                logger.debug(f"Detected task inheritance in {key}")
                return True

        # Default to legacy format
        logger.debug("No unified format indicators found, treating as legacy")
        return False

    except Exception as e:
        logger.warning(f"Error checking config format for {config_file}: {e}")
        return False


# Legacy compatibility functions with deprecation warnings
def get_calibration_param_from_yaml(
    robot, calib_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy function for calibration parameter parsing - DEPRECATED."""
    import warnings

    warnings.warn(
        "get_calibration_param_from_yaml is deprecated. "
        "Use get_param_from_yaml with task_type='calibration'",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_param_from_yaml(robot, calib_data, "calibration")


def get_identification_param_from_yaml(
    robot, identif_data: Dict[str, Any]
) -> Dict[str, Any]:
    """Legacy function for identification parameter parsing - DEPRECATED."""
    import warnings

    warnings.warn(
        "get_identification_param_from_yaml is deprecated. "
        "Use get_param_from_yaml with task_type='identification'",
        DeprecationWarning,
        stacklevel=2,
    )
    return get_param_from_yaml(robot, identif_data, "identification")
