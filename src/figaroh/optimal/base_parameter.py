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
import numpy as np
from typing import Dict, List, Tuple, Any

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from figaroh.identification.identification_tools import get_standard_parameters
from figaroh.utils.cubic_spline import (
    CubicSpline,
    WaypointsGeneration,
)
from figaroh.tools.qrdecomposition import get_baseIndex
from figaroh.tools.regressor import (
    build_regressor_basic,
    build_regressor_reduced,
    get_index_eliminate,
)
from figaroh.identification.parameter import (
    add_standard_additional_parameters,
    add_custom_parameters,
)


class BaseParameterComputer:
    """Handles base parameter computation and indexing."""

    def __init__(self, robot, identif_config, active_joints, soft_lim_pool):
        self.robot = robot
        self.model = self.robot.model
        self.standard_parameter = None
        self.identif_config = identif_config
        self.active_joints = active_joints
        self.soft_lim_pool = soft_lim_pool

    def compute_base_indices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute base parameter indices from random trajectory."""
        logging.info("Computing base parameter indices from random trajectory...")

        try:
            # Generate random trajectory for base parameter computation
            n_wps_r = 100
            freq_r = 100
            CB_r = CubicSpline(self.robot, n_wps_r, self.active_joints)
            WP_r = WaypointsGeneration(self.robot, n_wps_r, self.active_joints)
            WP_r.gen_rand_pool(self.soft_lim_pool)

            # Generate waypoints and trajectory
            wps_r, vel_wps_r, acc_wps_r = WP_r.gen_rand_wp()
            tps_r = np.matrix([0.5 * i for i in range(n_wps_r)]).transpose()
            t_r, p_r, v_r, a_r = CB_r.get_full_config(
                freq_r, tps_r, wps_r, vel_wps_r, acc_wps_r
            )

            # Compute base indices
            idx_e, idx_b = self._get_idx_from_random(p_r, v_r, a_r)
            logging.info(f"Computed {len(idx_b)} base parameters")

            return idx_e, idx_b

        except Exception as e:
            logging.error(f"Error computing base indices: {e}")
            raise

    def _get_idx_from_random(self, q, v, a) -> Tuple[np.ndarray, np.ndarray]:
        """Get indices of eliminate and base parameters."""
        W = build_regressor_basic(self.robot, q, v, a, self.identif_config)
        self.standard_parameter = get_standard_parameters(
            self.robot.model, self.identif_config
        )
        # additional parameters can be added in robot-specific subclass
        if (
            self.identif_config.get("has_friction", False)
            or self.identif_config.get("has_actuator_inertia", False)
            or self.identif_config.get("has_joint_offset", False)
        ):
            self.additional_parameters = add_standard_additional_parameters(
                self.model, self.identif_config
            )
            self.standard_parameter.update(self.additional_parameters)

        # Add custom parameters specific to the robot
        if self.identif_config.get("has_custom_parameters", False):
            self.custom_parameters = add_custom_parameters(
                self.model, self.identif_config.get("custom_parameters", {})
            )
            self.standard_parameter.update(self.custom_parameters)
        idx_e_, par_r_ = get_index_eliminate(W, self.standard_parameter, tol_e=0.001)
        # Convert to numpy arrays
        idx_e_ = np.array(idx_e_, dtype=int)
        W_e_ = build_regressor_reduced(W, idx_e_)
        idx_base_ = get_baseIndex(W_e_, par_r_)
        idx_base_ = np.array(idx_base_, dtype=int)

        return idx_e_, idx_base_
