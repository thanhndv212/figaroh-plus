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
import pinocchio as pin
from typing import Dict, List, Tuple

from figaroh.optimal.config import COLLISION_DEFAULTS
from figaroh.tools.robotcollisions import CollisionWrapper, add_non_adjacent_pairs
from figaroh.utils.cubic_spline import calc_torque

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


class TrajectoryConstraintManager:
    """Manages trajectory constraints and bounds.

    Collision (#143): when the robot's geometry model has no collision pairs
    (they usually come from an SRDF), every two geometries on non-adjacent
    bodies are paired, less the pairs of ``trajectory_config["srdf"]`` if
    given. Per pair and waypoint interval, the constraint is the smallest
    clearance over the interval's check points (``collision_checks_per_interval``
    evenly spaced samples ending at the waypoint; default the waypoint only),
    at least ``collision_margin``. With ``collision_screen`` set, exact
    distances are computed only for pairs a collision query flags within it
    and the others count as the screen. :meth:`trajectory_clear` checks a
    whole trajectory; the optimiser requires it of every segment and initial
    guess.
    """

    def __init__(self, robot, CB, trajectory_config, identif_config):
        self.robot = robot
        self.CB = CB
        self.n_wps = trajectory_config["n_wps"]
        self.freq = trajectory_config["freq"]
        self.identif_config = identif_config
        settings = {
            k: trajectory_config.get(k, v) for k, v in COLLISION_DEFAULTS.items()
        }
        self.margin = float(settings["collision_margin"])
        screen = settings["collision_screen"]
        self.screen = None if screen is None else max(float(screen), self.margin)
        self.checks_per_interval = max(
            1, int(settings["collision_checks_per_interval"])
        )
        self.check_frequency = float(settings["collision_check_frequency"])

        geom = getattr(robot, "geom_model", None)
        if geom is not None and len(geom.collisionPairs) == 0:
            added = add_non_adjacent_pairs(robot.model, geom)
            if settings["srdf"]:
                pin.removeCollisionPairs(robot.model, geom, settings["srdf"])
            logger.info(
                "Trajectory collision: %d pairs (%d non-adjacent, SRDF %s)",
                len(geom.collisionPairs),
                added,
                settings["srdf"] or "none",
            )
        if geom is None or len(geom.collisionPairs) == 0:
            logger.warning(
                "Trajectory collision is not enforced: the robot has no "
                "collision geometry pairs"
            )
        self.collision_wrapper = CollisionWrapper(robot=robot, viz=None)
        self.n_pairs = len(geom.collisionPairs) if geom is not None else 0
        if self.n_pairs:
            # screening data (margin = screen) and checking data (margin)
            if self.screen is not None:
                for request in self.collision_wrapper.geom_data.collisionRequests:
                    request.security_margin = self.screen
            self._check_data = geom.createData()
            for request in self._check_data.collisionRequests:
                request.security_margin = self.margin

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

        # Collision constraint bounds: one row per pair and waypoint interval
        cl.extend([self.margin] * self.n_pairs * (self.n_wps - 1))
        cu.extend([2 * 1e19] * self.n_pairs * (self.n_wps - 1))  # no upper limit

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

    def clipped_distances(self, q: np.ndarray) -> np.ndarray:
        """Distance per pair at ``q``, capped at the screen distance (all
        exact when no screen is set)."""
        wrapper = self.collision_wrapper
        geom, data = wrapper.geom_model, wrapper.geom_data
        if self.screen is None:
            pin.computeDistances(wrapper.model, wrapper.data, geom, data, q)
            return np.array([r.min_distance for r in data.distanceResults])
        pin.computeCollisions(wrapper.model, wrapper.data, geom, data, q, False)
        d = np.full(self.n_pairs, self.screen)
        for k, result in enumerate(data.collisionResults):
            if result.isCollision():
                distance = pin.computeDistance(geom, data, k).min_distance
                d[k] = min(self.screen, distance)
        return d

    def _evaluate_collision_constraints(self, p_f, tps, t_f) -> np.ndarray:
        """Smallest clearance per pair over each waypoint interval's checks."""
        if not self.n_pairs:
            return np.zeros(0)
        t = np.asarray(t_f, dtype=float).ravel()
        times = np.asarray(tps, dtype=float).ravel()
        rows = []
        for k in range(1, self.n_wps):
            idx = np.where((t > times[k - 1] + 1e-9) & (t <= times[k] + 1e-9))[0]
            if len(idx) == 0:
                idx = np.array([int(np.argmin(np.abs(t - times[k])))])
            pick = np.unique(
                np.round(np.linspace(0, len(idx) - 1, self.checks_per_interval)).astype(
                    int
                )
            )
            pick = np.unique(np.append(pick, len(idx) - 1))  # always the waypoint
            rows.append(
                np.min([self.clipped_distances(p_f[idx[i]]) for i in pick], axis=0)
            )
        return np.concatenate(rows)

    def trajectory_clear(self, tps, wps, vel_wps=None, acc_wps=None) -> bool:
        """True when the whole spline, sampled at ``collision_check_frequency``,
        keeps every pair at least ``collision_margin`` apart."""
        if not self.n_pairs:
            return True
        _, p_f, _, _ = self.CB.get_full_config(
            self.check_frequency, tps, wps, vel_wps, acc_wps
        )
        wrapper = self.collision_wrapper
        return not any(
            pin.computeCollisions(
                wrapper.model,
                wrapper.data,
                wrapper.geom_model,
                self._check_data,
                q,
                True,
            )
            for q in p_f
        )

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
