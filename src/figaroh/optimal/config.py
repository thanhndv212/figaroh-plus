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

import logging
import yaml
from typing import Dict, List, Tuple, Any
from yaml.loader import SafeLoader

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from figaroh.identification.identification_tools import (
    get_param_from_yaml as get_identification_param_from_yaml,
    unified_to_legacy_identif_config,
)
from figaroh.utils.config_parser import (
    UnifiedConfigParser,
    create_task_config,
    is_unified_config,
)


class ConfigurationManager:
    """Manages configuration loading and validation."""


@staticmethod
def load_param(robot, config_file: str) -> Tuple[Dict[str, Any], Any]:
    """Load trajectory parameters from YAML file."""

    try:
        logger.info(f"Loading config from {config_file}")

        # Check if this is a unified configuration format
        if is_unified_config(config_file):
            logger.info("Detected unified configuration format")
            # Use unified parser
            parser = UnifiedConfigParser(config_file)
            unified_config = parser.parse()
            unified_identif_config = create_task_config(
                robot, unified_config, "identification"
            )
            # Convert unified format to identif_config format
            identif_config = unified_to_legacy_identif_config(
                robot, unified_identif_config
            )
            unified_traj_config = create_task_config(
                robot, unified_config, "optimal_trajectory"
            )
            trajectory_config = create_config(unified_traj_config)

        else:
            logger.info("Detected legacy configuration format")
            # Use legacy format parsing
            with open(config_file, "r") as f:
                config = yaml.load(f, Loader=yaml.SafeLoader)
            identif_config = get_identification_param_from_yaml(
                robot, config["identification"]
            )
            traj_params = config["identification"].get("trajectory_params", [{}])[0]

            # Set default values if not present in config
            trajectory_config = {
                "n_wps": traj_params.get("n_wps", 5),
                "freq": traj_params.get("freq", 100),
                "t_s": traj_params.get("t_s", 2.0),
                "soft_lim": traj_params.get("soft_lim", 0.05),
                "max_attempts": traj_params.get("max_attempts", 1000),
            }
        return trajectory_config, identif_config

    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file {config_file} not found")
    except KeyError as e:
        raise ValueError(f"Missing required configuration key: {e}")
    except Exception as e:
        raise ValueError(f"Error loading configuration: {e}")


def create_config(unified_traj_config) -> dict:

    problem_params = unified_traj_config.get("problem", {})
    traj_params = unified_traj_config.get("trajectory", {})
    constraint_params = unified_traj_config.get("constraints", {})
    output_params = unified_traj_config.get("output", {})
    trajectory_config = {
        "n_wps": traj_params.get("waypoints", 5),
        "freq": traj_params.get("frequency", 100),
        "t_s": traj_params.get("segment_duration", 2.0),
        "soft_lim": problem_params.get("soft_lim", 0.05),
        "max_attempts": problem_params.get("max_attempts", 1000),
    }
    return trajectory_config
