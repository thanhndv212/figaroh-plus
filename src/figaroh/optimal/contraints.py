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

from figaroh.tools.robotcollisions import CollisionWrapper
from figaroh.utils.cubic_spline import calc_torque

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TrajectoryConstraintManager:
    """Manages trajectory constraints and bounds."""

    def __init__(self, robot, CB, trajectory_config, identif_config):
        self.robot = robot
        self.CB = CB
        self.n_wps = trajectory_config["n_wps"]
        self.freq = trajectory_config["freq"]
        self.identif_config = identif_config
        self.collision_wrapper = CollisionWrapper(robot=robot, viz=None)

    def get_variable_bounds(self) -> Tuple[List[float], List[float]]:
        """Get variable bounds for optimization."""
        lb, ub = [], []
        for i in range(1, self.n_wps):
            lb.extend(self.CB.lower_q)
            ub.extend(self.CB.upper_q)
        return lb, ub

    def get_constraint_bounds(self, Ns: int) -> Tuple[List, List]:
        """Get constraint bounds for optimization."""
        cl, cu = [], []

        # Position constraint bounds
        for i in range(1, self.n_wps):
            cl.extend(self.CB.lower_q)
            cu.extend(self.CB.upper_q)

        # Velocity constraint bounds
        for j in range(Ns):
            cl.extend(self.CB.lower_dq)
            cu.extend(self.CB.upper_dq)

        # Torque constraint bounds
        for j in range(Ns):
            cl.extend(self.CB.lower_effort)
            cu.extend(self.CB.upper_effort)

        # Collision constraint bounds
        n_cols = len(self.robot.geom_model.collisionPairs)
        cl.extend([0.01] * n_cols * (self.n_wps - 1))  # 1 cm margin
        cu.extend([2 * 1e19] * n_cols * (self.n_wps - 1))  # no upper limit

        return cl, cu

    def evaluate_constraints(
        self,
        Ns: int,
        X: np.ndarray,
        opt_cb: Dict,
        tps,
        vel_wps,
        acc_wps,
        wp_init,
    ) -> np.ndarray:
        """Evaluate all constraints for optimization."""
        try:
            # Reshape and arrange waypoints
            X = np.array(X)
            wps_X = np.reshape(X, (self.n_wps - 1, len(self.CB.act_idxq)))
            wps = np.vstack((wp_init, wps_X))
            wps = wps.transpose()

            # Generate full trajectory configuration
            t_f, p_f, v_f, a_f = self.CB.get_full_config(
                self.freq, tps, wps, vel_wps, acc_wps
            )

            # Compute joint torques
            tau = calc_torque(p_f.shape[0], self.robot, p_f, v_f, a_f)

            # Evaluate individual constraint types
            q_constraints = self._evaluate_position_constraints(p_f, tps, t_f)
            v_constraints = self._evaluate_velocity_constraints(v_f)
            tau_constraints = self._evaluate_torque_constraints(tau, Ns)
            collision_constraints = self._evaluate_collision_constraints(p_f, tps, t_f)

            # Concatenate all constraints
            return np.concatenate(
                (
                    q_constraints,
                    v_constraints,
                    tau_constraints,
                    collision_constraints,
                ),
                axis=None,
            )

        except Exception as e:
            logging.error(f"Error evaluating constraints: {e}")
            raise

    def _evaluate_position_constraints(self, p_f, tps, t_f) -> np.ndarray:
        """Evaluate position constraints at waypoints."""
        idx_waypoints = self._get_waypoint_indices(tps, t_f)
        q_constraints = p_f[idx_waypoints, :]
        return q_constraints[:, self.CB.act_idxq]

    def _evaluate_velocity_constraints(self, v_f) -> np.ndarray:
        """Evaluate velocity constraints at all samples."""
        return v_f[:, self.CB.act_idxv]

    def _evaluate_torque_constraints(self, tau, Ns) -> np.ndarray:
        """Evaluate torque constraints at all samples."""
        tau_constraints = np.zeros((Ns, len(self.CB.act_idxv)))
        for k in range(len(self.CB.act_idxv)):
            tau_constraints[:, k] = tau[
                range(self.CB.act_idxv[k] * Ns, (self.CB.act_idxv[k] + 1) * Ns)
            ]
        return tau_constraints

    def _evaluate_collision_constraints(self, p_f, tps, t_f) -> np.ndarray:
        """Evaluate collision constraints."""
        idx_waypoints = self._get_waypoint_indices(tps, t_f)
        dist_all = []
        for j in idx_waypoints:
            self.collision_wrapper.computeCollisions(p_f[j, :])
            dist_all = np.append(dist_all, self.collision_wrapper.getDistances())
        return np.asarray(dist_all)

    def _get_waypoint_indices(self, tps, t_f) -> List[int]:
        """Get indices corresponding to waypoint times."""
        idx_waypoints = []
        time_points = tps[range(1, self.n_wps), :]
        time_points_flat = np.array(time_points).flatten()

        for i in range(t_f.shape[0]):
            t_val = float(t_f[i, 0]) if hasattr(t_f[i, 0], "item") else t_f[i, 0]
            if t_val in time_points_flat:
                idx_waypoints.append(i)

        return idx_waypoints
