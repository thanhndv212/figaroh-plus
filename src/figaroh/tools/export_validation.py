"""
URDF model comparison and validation utilities.

Provides tools to compare two URDF models (nominal vs. modified after parameter
application) through forward-kinematics analysis and viser-based visualization.

Typical usage::

    from figaroh.tools.export_validation import URDFComparison

    comp = URDFComparison("robot.urdf", "robot_modified.urdf")
    print(comp.trajectory_errors())
    # → TrajectoryErrors(rmse_position=0.051, rmse_orientation=0.098, ...)

    # Interactive viser visualization:
    comp.show_trajectory_animation()
    comp.show_static_grid()
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union
import logging
import os
import time

import numpy as np

try:
    import pinocchio as pin
except ImportError:
    pin = None

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# ── Public data classes ─────────────────────────────────────────


@dataclass
class PoseDelta:
    """FK pose difference between two models at a single configuration.

    The delta is expressed as ``M_nominal⁻¹ · M_modified``, i.e. the
    SE3 transform from the nominal end-effector frame to the modified one.
    """

    translation: np.ndarray  # (3,) position delta in nominal EE frame
    rotation: np.ndarray     # (3, 3) rotation matrix of the delta
    twist: np.ndarray        # (6,) se3 twist vector [v, ω]
    q: np.ndarray            # (nq,) joint configuration that produced this delta

    def position_error(self) -> float:
        """Euclidean distance in meters."""
        return float(np.linalg.norm(self.translation))

    def orientation_error(self) -> float:
        """Angle-axis magnitude in radians."""
        return float(np.linalg.norm(self.twist[3:]))


@dataclass
class TrajectoryErrors:
    """Aggregated FK errors across a trajectory of joint configurations."""

    rmse_position: float         # meters
    rmse_orientation: float      # radians
    max_position: float          # meters
    max_orientation: float       # radians
    per_sample: List[PoseDelta]  # one entry per configuration

    def __repr__(self):
        return (
            f"TrajectoryErrors("
            f"rmse_pos={self.rmse_position:.4f} m, "
            f"rmse_orient={self.rmse_orientation:.4f} rad, "
            f"max_pos={self.max_position:.4f} m, "
            f"max_orient={self.max_orientation:.4f} rad, "
            f"samples={len(self.per_sample)})"
        )


@dataclass
class PoseError:
    """FK error at a single static configuration with user label."""

    q: np.ndarray
    position_error_mm: float
    orientation_error_deg: float
    pose_delta: PoseDelta
    label: str = ""


# ── Core comparison class ────────────────────────────────────────


class URDFComparison:
    """Compare two URDF models via FK analysis and viser visualization.

    Loads both URDFs into Pinocchio for FK computation and into yourdfpy
    (when needed) for viser rendering.

    Args:
        nominal_urdf: Path to the original/reference URDF.
        modified_urdf: Path to the URDF after parameter application.
    """

    def __init__(
        self,
        nominal_urdf: Union[str, Path],
        modified_urdf: Union[str, Path],
    ):
        if pin is None:
            raise ImportError("pinocchio is required for URDFComparison")

        self.nominal_path = Path(nominal_urdf)
        self.modified_path = Path(modified_urdf)

        for p in (self.nominal_path, self.modified_path):
            if not p.exists():
                raise FileNotFoundError(f"URDF not found: {p}")

        # Pinocchio models for FK computation
        self.model_a = pin.buildModelFromUrdf(str(self.nominal_path))
        self.model_b = pin.buildModelFromUrdf(str(self.modified_path))
        self.data_a = self.model_a.createData()
        self.data_b = self.model_b.createData()

        # Identify end-effector: deepest frame in the model tree.
        # We skip frames attached directly to the universe (joint parent = 0)
        # to avoid picking fixed-base sensor/camera links.
        self.ee_frame_a, self.ee_id_a = self._detect_ee(self.model_a)
        self.ee_frame_b, self.ee_id_b = self._detect_ee(self.model_b)

    # ── Private helpers ──

    @staticmethod
    def _detect_ee(model):
        """Return (frame_name, frame_id) of the deepest frame in the model."""
        parent_joint = model.frames[0].parentJoint  # type: ignore
        if isinstance(parent_joint, np.ndarray):
            # Pinocchio 3.x: parentJoint is an array
            pj = parent_joint
        else:
            # Older: fall back on frame attribute
            pj = np.array([f.parentJoint for f in model.frames])
        # Compute tree depth from universe
        depth = np.zeros(model.nframes, dtype=int)
        for jid in range(1, model.njoints):
            pid = model.parents[jid]
            if pid >= 0:
                depth[jid] = depth[pid] + 1
            else:
                depth[jid] = 1
        # Depth of each frame = depth of its parent joint
        frame_depth = np.zeros(model.nframes, dtype=int)
        for fid in range(1, model.nframes):
            jp = int(pj[fid])
            frame_depth[fid] = depth[jp] if jp < len(depth) else 0
        # Pick deepest frame that is not attached to universe (jp > 0)
        candidates = [
            (frame_depth[fid], fid) for fid in range(1, model.nframes)
            if int(pj[fid]) > 0
        ]
        if not candidates:
            # Fallback: pick last frame
            fid = model.nframes - 1
            return model.frames[fid].name, fid
        fid = max(candidates, key=lambda x: x[0])[1]
        return model.frames[fid].name, fid

    # ── Public numerical API ──

    def trajectory_errors(
        self,
        n_samples: int = 100,
        seed: int = 42,
    ) -> TrajectoryErrors:
        """Compute end-effector FK errors across random joint configurations.

        Args:
            n_samples: Number of random configurations to sample.
            seed: Random seed for reproducibility.

        Returns:
            TrajectoryErrors with aggregated metrics and per-sample data.
        """
        configs = self._sample_configs(n_samples, seed)
        deltas = [self._compute_delta(q) for q in configs]
        return self._aggregate(deltas)

    def static_poses(
        self,
        poses: Optional[List[Union[List[float], np.ndarray]]] = None,
    ) -> List[PoseError]:
        """Compute FK errors at specific joint configurations.

        Args:
            poses: List of joint configuration vectors. If None, a default
                set of 20 meaningful poses is used (covering home, limits,
                mixed-angle combinations).

        Returns:
            List of PoseError, one per configuration.
        """
        if poses is None:
            poses = self._default_poses()
        results = []
        for i, q in enumerate(poses):
            q_arr = np.asarray(q, dtype=float)
            delta = self._compute_delta(q_arr)
            results.append(PoseError(
                q=q_arr,
                position_error_mm=delta.position_error() * 1000,
                orientation_error_deg=delta.orientation_error() * 180 / np.pi,
                pose_delta=delta,
                label=f"pose_{i}",
            ))
        return results

    # ── Public visualization API ──

    def show_overlay(
        self,
        server=None,
        port: int = 8080,
        orig_color: Tuple[int, int, int] = (51, 102, 229),
        mod_color: Tuple[int, int, int] = (229, 51, 51),
        duration: float = 5.0,
    ):
        """Display both models overlaid in viser.

        Args:
            server: Existing viser server, or None to create one.
            port: Port for new server (ignored if *server* is given).
            orig_color: RGB color for the original model (0-255).
            mod_color: RGB color for the modified model (0-255).
            duration: Seconds to keep the display open.
        """
        viz = self._get_viser_server(server, port)
        urdf_orig, urdf_mod = self._load_yourdfpy()

        from viser.extras import ViserUrdf

        try:
            viz.scene.remove("/robot")
        except Exception:
            pass

        ViserUrdf(
            viz, urdf_or_path=urdf_orig,
            root_node_name="/robot/original",
            load_meshes=True,
        )
        ViserUrdf(
            viz, urdf_or_path=urdf_mod,
            root_node_name="/robot/modified",
            load_meshes=True,
        )
        print(f"[vis] Both models displayed at http://localhost:{viz.port}")
        print(f"[vis] Original (RGB={orig_color}) / Modified (RGB={mod_color})")
        time.sleep(duration)

    def show_trajectory_animation(
        self,
        n_configs: int = 50,
        seed: int = 42,
        server=None,
        port: int = 8080,
        duration: float = 10.0,
    ):
        """Animate through random joint configs, tracing end-effector paths.

        Args:
            n_configs: Number of random configurations to animate.
            seed: Random seed.
            server: Existing viser server, or None to create one.
            port: Port for new server.
            duration: Seconds to keep the display open after animation.
        """
        import trimesh
        viz = self._get_viser_server(server, port)
        urdf_orig, urdf_mod = self._load_yourdfpy()
        from viser.extras import ViserUrdf

        vis_orig = ViserUrdf(
            viz, urdf_or_path=urdf_orig,
            root_node_name="/robot/orig_anim",
            load_meshes=True,
        )
        vis_mod = ViserUrdf(
            viz, urdf_or_path=urdf_mod,
            root_node_name="/robot/mod_anim",
            load_meshes=True,
        )

        configs = self._sample_configs(n_configs, seed)
        orig_trace: List[np.ndarray] = []
        mod_trace: List[np.ndarray] = []

        for q in configs:
            vis_orig.update_cfg(q.tolist())
            vis_mod.update_cfg(q.tolist())
            pin.forwardKinematics(self.model_a, self.data_a, q)
            pin.updateFramePlacements(self.model_a, self.data_a)
            pin.forwardKinematics(self.model_b, self.data_b, q)
            pin.updateFramePlacements(self.model_b, self.data_b)
            orig_trace.append(self.data_a.oMf[self.ee_id_a].translation.copy())
            mod_trace.append(self.data_b.oMf[self.ee_id_b].translation.copy())
            time.sleep(0.05)

        # Draw path traces
        r = 0.01
        sphere_mesh = trimesh.primitives.Sphere(radius=r)
        for pts, color, label in [
            (orig_trace, (51, 102, 229), "orig"),
            (mod_trace, (229, 51, 51), "mod"),
        ]:
            for i, pt in enumerate(pts):
                viz.scene.add_mesh_simple(
                    f"/traces/{label}_{i}",
                    sphere_mesh.vertices,
                    sphere_mesh.faces,
                    color=color,
                    position=tuple(pt),
                )
        print(f"[vis] Traced {n_configs} configs at http://localhost:{viz.port}")
        time.sleep(duration)

    def show_static_grid(
        self,
        poses: Optional[List[Union[List[float], np.ndarray]]] = None,
        server=None,
        port: int = 8080,
        duration: float = 15.0,
        spacing: float = 2.0,
    ):
        """Display a grid of static configurations with error labels.

        Each cell shows both models at the same configuration, with the
        position (mm) and orientation (deg) error overlaid.

        Args:
            poses: Joint configurations. If None, uses 20 default poses.
            server: Existing viser server, or None to create one.
            port: Port for new server.
            duration: Seconds to keep the display open.
            spacing: Distance between grid cells in meters.
        """
        from viser.extras import ViserUrdf

        viz = self._get_viser_server(server, port)
        urdf_orig, urdf_mod = self._load_yourdfpy()

        if poses is None:
            poses = self._default_poses()

        for i, q in enumerate(poses):
            row, col = divmod(i, 4)
            x_off = col * spacing
            z_off = row * spacing
            q_arr = np.asarray(q, dtype=float)

            ViserUrdf(
                viz, urdf_or_path=urdf_orig,
                root_node_name=f"/grid/{i}/orig",
                load_meshes=True,
            )
            ViserUrdf(
                viz, urdf_or_path=urdf_mod,
                root_node_name=f"/grid/{i}/mod",
                load_meshes=True,
            )

            delta = self._compute_delta(q_arr)
            pos_err = delta.position_error() * 1000  # mm
            orient_err = delta.orientation_error() * 180 / np.pi  # deg

            viz.scene.add_label(
                f"/grid/{i}/label",
                f"{pos_err:.1f}mm / {orient_err:.1f}°",
                position=(x_off, z_off + 1.2, 0),
            )
        print(f"[vis] {len(poses)} configs in grid at http://localhost:{viz.port}")
        time.sleep(duration)

    # ── Internal helpers ──

    def _sample_configs(self, n: int, seed: int) -> np.ndarray:
        """Sample random joint configurations within model bounds."""
        rng = np.random.default_rng(seed)
        lb = self.model_a.lowerPositionLimit
        ub = self.model_a.upperPositionLimit
        valid = np.isfinite(lb) & np.isfinite(ub)
        lb = np.where(valid, lb, -np.pi)
        ub = np.where(valid, ub, np.pi)
        return rng.uniform(lb, ub, size=(n, self.model_a.nq))

    def _compute_delta(self, q: np.ndarray) -> PoseDelta:
        """Compute FK pose delta between the two models at config q."""
        pin.forwardKinematics(self.model_a, self.data_a, q)
        pin.updateFramePlacements(self.model_a, self.data_a)
        pin.forwardKinematics(self.model_b, self.data_b, q)
        pin.updateFramePlacements(self.model_b, self.data_b)

        M_a = self.data_a.oMf[self.ee_id_a]
        M_b = self.data_b.oMf[self.ee_id_b]
        delta_se3 = M_a.inverse() * M_b
        motion = pin.log(delta_se3)  # pin.Motion in Pinocchio 3.x
        twist = np.concatenate([motion.linear, motion.angular])

        return PoseDelta(
            translation=delta_se3.translation.copy(),
            rotation=delta_se3.rotation.copy(),
            twist=twist,
            q=q.copy(),
        )

    def _aggregate(self, deltas: List[PoseDelta]) -> TrajectoryErrors:
        """Aggregate per-sample deltas into summary metrics."""
        positions = np.array([d.translation for d in deltas])
        orientations = np.array([d.twist[3:] for d in deltas])
        return TrajectoryErrors(
            rmse_position=float(np.sqrt(np.mean(np.sum(positions ** 2, axis=1)))),
            rmse_orientation=float(np.sqrt(np.mean(np.sum(orientations ** 2, axis=1)))),
            max_position=float(np.max(np.linalg.norm(positions, axis=1))),
            max_orientation=float(np.max(np.linalg.norm(orientations, axis=1))),
            per_sample=deltas,
        )

    def _default_poses(self) -> List[np.ndarray]:
        """Default set of 20 meaningful poses for static comparison."""
        return [
            [0.0, 0.0],
            [np.pi / 2, 0.0],
            [0.0, np.pi / 2],
            [-np.pi / 2, np.pi / 4],
            [np.pi, 0.0],
            [0.0, np.pi],
            [np.pi / 3, -np.pi / 3],
            [-np.pi / 4, np.pi / 6],
            [np.pi / 6, -np.pi / 4],
            [np.pi / 2, np.pi / 2],
            [-np.pi / 2, -np.pi / 2],
            [np.pi / 4, -np.pi / 8],
            [0.0, -1.57],
            [1.57, -1.57],
            [0.78, 0.78],
            [-0.78, -0.78],
            [2.0, 1.0],
            [-1.5, 0.5],
            [0.5, -2.0],
            [3.0, -3.0],
        ]

    def _load_yourdfpy(self):
        """Lazy-load yourdfpy URDF objects for viser rendering."""
        import yourdfpy
        return (
            yourdfpy.URDF.load(str(self.nominal_path)),
            yourdfpy.URDF.load(str(self.modified_path)),
        )

    @staticmethod
    def _get_viser_server(server=None, port: int = 8080):
        """Get or create a viser server."""
        if server is not None:
            return server
        import viser
        srv = viser.ViserServer(port=port, verbose=True)
        # Give the server a moment to start
        time.sleep(0.5)
        return srv
