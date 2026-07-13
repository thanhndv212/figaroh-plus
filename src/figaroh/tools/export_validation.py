"""
URDF model comparison and validation utilities.

Provides tools to compare two URDF models (nominal vs. modified after parameter
application) through forward-kinematics analysis and viser-based visualization.

Typical usage::

    from figaroh.tools.export_validation import URDFComparison

    comp = URDFComparison("robot.urdf", "robot_modified.urdf")
    print(comp.fk_consistency_check())
    # → FkConsistencyResult(rmse_position=0.051, rmse_orientation=0.098, ...)

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
class FkConsistencyResult:
    """Aggregated FK consistency check — compares nominal vs. exported URDF FK."""

    rmse_position: float         # meters
    rmse_orientation: float      # radians
    max_position: float          # meters
    max_orientation: float       # radians
    per_sample: List[PoseDelta]  # one entry per configuration

    def __repr__(self):
        return (
            f"FkConsistencyResult("
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

    def fk_consistency_check(
        self,
        n_samples: int = 100,
        seed: int = 42,
    ) -> FkConsistencyResult:
        """Check the exported URDF reproduces the calibrated FK numerically.

        This is a file-integrity check — it verifies the exported URDF file
        is self-consistent with the calibration results. It does NOT test
        against ground-truth measurements (use calibration validation for
        that).

        Args:
            n_samples: Number of random configurations to sample.
            seed: Random seed for reproducibility.

        Returns:
            FkConsistencyResult with aggregated FK consistency metrics.
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
        print(f"[vis] Both models displayed at http://localhost:{viz.get_port()}")
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
            cfg_dict = self._pinocchio_config_to_dict(q)
            vis_orig.update_cfg(cfg_dict)
            vis_mod.update_cfg(cfg_dict)
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
        print(f"[vis] Traced {n_configs} configs at http://localhost:{viz.get_port()}")
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
            poses: Joint configurations. If None, uses 24 default poses.
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

            vis_orig = ViserUrdf(
                viz, urdf_or_path=urdf_orig,
                root_node_name=f"/grid/{i}/orig",
                load_meshes=True,
            )
            vis_mod = ViserUrdf(
                viz, urdf_or_path=urdf_mod,
                root_node_name=f"/grid/{i}/mod",
                load_meshes=True,
            )
            cfg_dict = self._pinocchio_config_to_dict(q_arr)
            vis_orig.update_cfg(cfg_dict)
            vis_mod.update_cfg(cfg_dict)

            delta = self._compute_delta(q_arr)
            pos_err = delta.position_error() * 1000  # mm
            orient_err = delta.orientation_error() * 180 / np.pi  # deg

            viz.scene.add_label(
                f"/grid/{i}/label",
                f"{pos_err:.1f}mm / {orient_err:.1f}°",
                position=(x_off, z_off + 1.2, 0),
            )
        print(f"[vis] {len(poses)} configs in grid at http://localhost:{viz.get_port()}")
        time.sleep(duration)

    def show_interactive_validation(
        self,
        n_trajectory: int = 50,
        seed: int = 42,
        port: int = 8080,
    ):
        """Interactive combined visualization with trajectory, static comparison,
        error plots, replay, and opacity controls.

        Opens a single viser server containing:

        * **Trajectory animation** — 50 random configurations with path traces
          for both nominal and modified models. Replay button allows re-running.
        * **Static pose comparison** — overlays both models at the same
          configuration with opacity slider for the modified model.
        * **Error plots** — uplot charts for trajectory and static pose errors
          (position in mm, orientation in degrees).
        * **Pose selector** — slider to cycle through static configurations.

        Args:
            n_trajectory: Number of random configurations to animate.
            seed: Random seed for reproducibility.
            port: Port for the viser server.
        """
        import threading

        import trimesh
        import viser
        from viser import uplot
        from viser.extras import ViserUrdf

        server = self._get_viser_server(None, port)
        urdf_orig, urdf_mod = self._load_yourdfpy()
        traj_configs = self._sample_configs(n_trajectory, seed)
        static_configs = self._default_poses()

        # --- 3D Scene: trajectory robots ---
        # Parent frames to offset trajectory models from static comparison
        traj_parent = server.scene.add_frame(
            "/trajectory", show_axes=False
        )
        traj_vis_orig = ViserUrdf(
            server,
            urdf_or_path=urdf_orig,
            root_node_name="/trajectory/orig",
            load_meshes=True,
            mesh_color_override=(51, 102, 229),
        )
        traj_vis_mod = ViserUrdf(
            server,
            urdf_or_path=urdf_mod,
            root_node_name="/trajectory/mod",
            load_meshes=True,
            mesh_color_override=(229, 51, 51),
        )

        # --- 3D Scene: static comparison robots (offset to the right) ---
        static_parent = server.scene.add_frame(
            "/static",
            show_axes=False,
            position=(2.5, 0, 0),
        )
        static_vis_nom = ViserUrdf(
            server,
            urdf_or_path=urdf_orig,
            root_node_name="/static/nominal",
            load_meshes=True,
            mesh_color_override=(51, 102, 229),
        )
        static_vis_mod = ViserUrdf(
            server,
            urdf_or_path=urdf_mod,
            root_node_name="/static/modified",
            load_meshes=True,
            mesh_color_override=(229, 51, 51, 0.5),
        )
        # Labels for static section
        server.scene.add_label(
            "/static/label_nom",
            "Nominal",
            position=(0, 0, 1.8),
        )
        server.scene.add_label(
            "/static/label_mod",
            "Modified",
            position=(0, 0, 2.0),
        )

        # --- Track static error state ---
        static_state = {
            "idx": 0,
            "opacity": 0.5,
            "configs": static_configs,
            "errors": [self._compute_delta(q) for q in static_configs],
        }

        # --- Compute trajectory errors upfront ---
        traj_errors = [self._compute_delta(q) for q in traj_configs]
        traj_pos_mm = np.array([e.position_error() * 1000 for e in traj_errors])
        traj_orient_deg = np.array([e.orientation_error() * 180 / np.pi for e in traj_errors])
        static_pos_mm = np.array(
            [e.position_error() * 1000 for e in static_state["errors"]]
        )
        static_orient_deg = np.array(
            [e.orientation_error() * 180 / np.pi for e in static_state["errors"]]
        )

        # --- Animation state ---
        anim_state: dict = {
            "traj_configs": traj_configs,
            "running": False,
            "speed": 0.05,
            "orig_trace": [],
            "mod_trace": [],
        }

        def _clear_traces():
            try:
                server.scene.remove_by_name("/traces/orig_line")
            except Exception:
                pass
            try:
                server.scene.remove_by_name("/traces/mod_line")
            except Exception:
                pass

        def _draw_traces():
            """Draw EE path traces as line segments."""
            orig_pts = np.array(anim_state["orig_trace"])
            mod_pts = np.array(anim_state["mod_trace"])
            if len(orig_pts) < 2:
                return

            # Reshape to (N-1, 2, 3) for line segments
            def _to_segments(pts: np.ndarray) -> np.ndarray:
                return np.stack([pts[:-1], pts[1:]], axis=1)

            # Nominal trace (blue)
            server.scene.add_line_segments(
                "/traces/orig_line",
                _to_segments(orig_pts),
                colors=(51, 102, 229),
                line_width=2,
            )
            # Modified trace (red)
            server.scene.add_line_segments(
                "/traces/mod_line",
                _to_segments(mod_pts),
                colors=(229, 51, 51),
                line_width=2,
            )

        def _run_animation():
            if anim_state["running"]:
                return
            anim_state["running"] = True
            _clear_traces()
            anim_state["orig_trace"].clear()
            anim_state["mod_trace"].clear()

            for i, q in enumerate(anim_state["traj_configs"]):
                cfg_dict = self._pinocchio_config_to_dict(q)
                traj_vis_orig.update_cfg(cfg_dict)
                traj_vis_mod.update_cfg(cfg_dict)

                pin.forwardKinematics(self.model_a, self.data_a, q)
                pin.updateFramePlacements(self.model_a, self.data_a)
                pin.forwardKinematics(self.model_b, self.data_b, q)
                pin.updateFramePlacements(self.model_b, self.data_b)

                anim_state["orig_trace"].append(
                    self.data_a.oMf[self.ee_id_a].translation.copy()
                )
                anim_state["mod_trace"].append(
                    self.data_b.oMf[self.ee_id_b].translation.copy()
                )
                time.sleep(anim_state["speed"])

            _draw_traces()
            anim_state["running"] = False
            replay_btn.disabled = False

        def _update_static_pose():
            """Apply the current static pose to both overlaid models."""
            idx = static_state["idx"]
            q = static_state["configs"][idx]
            cfg_dict = self._pinocchio_config_to_dict(q)
            static_vis_nom.update_cfg(cfg_dict)
            static_vis_mod.update_cfg(cfg_dict)
            error = static_state["errors"][idx]
            pos_mm = error.position_error() * 1000
            orient_deg = error.orientation_error() * 180 / np.pi
            pose_label.content = f"Pose {idx}: {pos_mm:.1f}mm / {orient_deg:.1f}°"

        # --- GUI Controls ---
        with server.gui.add_folder("Controls"):
            # Trajectory section
            server.gui.add_markdown("**Trajectory**")
            replay_btn = server.gui.add_button(
                "Replay Trajectory", icon="player-play-filled"
            )
            speed_slider = server.gui.add_slider(
                "Animation Speed",
                0.01, 0.2, 0.01, 0.05,
                hint="Delay between configurations (seconds)",
            )

            # Static comparison section
            server.gui.add_markdown("**Static Comparison**")
            pose_slider = server.gui.add_slider(
                "Pose Index",
                0, len(static_configs) - 1, 1, 0,
                hint="Select which static configuration to display",
            )
            opacity_slider = server.gui.add_slider(
                "Modified Model Opacity",
                0.0, 1.0, 0.01, 0.5,
                hint="Transparency of the modified model overlay",
            )
            pose_label = server.gui.add_markdown(
                "Pose 0: ...",
            )

        # --- Error Plots ---
        traj_x = np.arange(len(traj_pos_mm))
        static_x = np.arange(len(static_pos_mm))

        with server.gui.add_folder("Error Plots"):
            server.gui.add_markdown("**Trajectory FK Errors**")
            server.gui.add_uplot(
                data=(traj_x, traj_pos_mm, traj_orient_deg),
                series=(
                    uplot.Series(),
                    uplot.Series(
                        label="Position Error (mm)",
                        color=(51, 102, 229),
                    ),
                    uplot.Series(
                        label="Orientation Error (deg)",
                        color=(229, 51, 51),
                    ),
                ),
                title="Trajectory",
                aspect=2.0,
            )

            server.gui.add_markdown("**Static Pose FK Errors**")
            server.gui.add_uplot(
                data=(static_x, static_pos_mm, static_orient_deg),
                series=(
                    uplot.Series(),
                    uplot.Series(
                        label="Position Error (mm)",
                        color=(51, 102, 229),
                    ),
                    uplot.Series(
                        label="Orientation Error (deg)",
                        color=(229, 51, 51),
                    ),
                ),
                title="Static Poses",
                aspect=2.0,
            )

        # --- Callbacks ---
        @replay_btn.on_click
        def _on_replay(_event):
            replay_btn.disabled = True
            thread = threading.Thread(target=_run_animation, daemon=True)
            thread.start()

        @speed_slider.on_update
        def _on_speed_change(_event):
            anim_state["speed"] = _event.target.value

        @pose_slider.on_update
        def _on_pose_change(_event):
            static_state["idx"] = int(_event.target.value)
            _update_static_pose()

        @opacity_slider.on_update
        def _on_opacity_change(_event):
            static_state["opacity"] = _event.target.value
            for mesh in static_vis_mod._meshes:
                mesh.opacity = _event.target.value

        # --- Initial state ---
        _update_static_pose()

        print("\n" + "=" * 60)
        print(f"Interactive validation at http://localhost:{server.get_port()}")
        print("  Trajectory:  blue (nominal), red (modified)")
        print("  Static:      overlaid models (offset right)")
        print("  Controls:    Replay | Speed | Pose | Opacity")
        print("  Error plots: scroll GUI panel")
        print("=" * 60)

        # --- Run initial animation ---
        replay_btn.disabled = True
        _run_animation()

        # Block until user stops the server
        try:
            server.sleep_forever()
        except KeyboardInterrupt:
            pass

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

    def _pinocchio_config_to_dict(self, q: np.ndarray) -> dict:
        """Convert pinocchio config vector to joint name→value dict for ViserUrdf.

        Only 1-DOF joints are mapped (skipping floating-base and composite joints
        that don't exist in the URDF joint space).
        """
        cfg = {}
        for jid in range(1, self.model_a.njoints):  # skip universe
            name = self.model_a.names[jid]
            nq = int(self.model_a.joints[jid].nq)
            if nq == 1:
                idx = int(self.model_a.idx_qs[jid])
                cfg[name] = float(q[idx])
        return cfg

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

    def _aggregate(self, deltas: List[PoseDelta]) -> FkConsistencyResult:
        """Aggregate per-sample deltas into summary metrics."""
        positions = np.array([d.translation for d in deltas])
        orientations = np.array([d.twist[3:] for d in deltas])
        return FkConsistencyResult(
            rmse_position=float(np.sqrt(np.mean(np.sum(positions ** 2, axis=1)))),
            rmse_orientation=float(np.sqrt(np.mean(np.sum(orientations ** 2, axis=1)))),
            max_position=float(np.max(np.linalg.norm(positions, axis=1))),
            max_orientation=float(np.max(np.linalg.norm(orientations, axis=1))),
            per_sample=deltas,
        )

    def _default_poses(self) -> List[np.ndarray]:
        """Default set of meaningful poses for static comparison.

        Generates 24 random configurations within the model's joint bounds.
        """
        configs = self._sample_configs(24, seed=42)
        return [configs[i] for i in range(configs.shape[0])]

    @staticmethod
    def _find_models_dir(from_path: Path) -> Optional[Path]:
        """Walk up from *from_path* to find a ``models/`` directory."""
        current = from_path.resolve().parent
        for _ in range(20):
            candidate = current / "models"
            if candidate.is_dir():
                return candidate.resolve()
            parent = current.parent
            if parent == current:  # reached filesystem root
                break
            current = parent
        return None

    @staticmethod
    def _create_package_handler(urdf_path: Path):
        """Create a filename handler resolving ``package://`` URIs.

        Search order:
        1. ``ROS_PACKAGE_PATH`` environment variable (ROS convention)
        2. ``models/`` directory auto-discovered by walking up from *urdf_path*

        Falls back to the original filename when unresolved.
        """
        # Build search directories
        search_dirs: list[Path] = []

        # 1. ROS_PACKAGE_PATH (ROS standard mechanism)
        rpp = os.environ.get("ROS_PACKAGE_PATH", "")
        if rpp:
            search_dirs.extend(
                Path(p).resolve() for p in rpp.split(":") if p
            )

        # 2. Auto-discovered models/ directory
        models_dir = URDFComparison._find_models_dir(urdf_path)
        if models_dir is not None:
            search_dirs.append(models_dir)

        def handler(fname: str) -> str:
            if "://" not in fname:
                return fname
            # package://pkg_name/path/to/mesh.stl → pkg_name/path/to/mesh.stl
            rest = ":".join(fname.split(":")[1:])[2:]
            pkg_name, _, subpath = rest.partition("/")
            for base in search_dirs:
                candidate = base / pkg_name / subpath
                if candidate.exists():
                    return str(candidate.resolve())
            return fname  # unresolved — leave for fallback handlers

        return handler

    def _load_yourdfpy(self):
        """Lazy-load yourdfpy URDF objects for viser rendering."""
        import yourdfpy
        handler = self._create_package_handler(self.nominal_path)
        return (
            yourdfpy.URDF.load(str(self.nominal_path), filename_handler=handler),
            yourdfpy.URDF.load(str(self.modified_path), filename_handler=handler),
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
