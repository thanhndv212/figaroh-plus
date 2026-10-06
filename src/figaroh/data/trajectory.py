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

"""Dynamic identification input: a trajectory with its conventions attached.

See docs/decisions/data-result-contract.md. Joint order, clock, signal
origin, effort kind and unit, the valid-sample mask and the source travel
with the arrays instead of being implied by an adapter.

The effort keeps the units it was recorded in (a motor current, a motor-side
torque, a load fraction, a joint torque, or a linear actuator's force).
:meth:`TrajectoryData.check_effort` decides whether a solver may use it.
"""

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np

from figaroh.data.source import DataSource

JOINT_TORQUE = "joint_torque"  # N·m, revolute joints
JOINT_FORCE = "joint_force"  # N, prismatic joints / linear actuators
MOTOR_CURRENT = "motor_current"
MOTOR_TORQUE = "motor_torque"  # before the reduction
LOAD_FRACTION = "load_fraction"
# usable only with an identified drive gain for the joint
DRIVE_KINDS = frozenset({MOTOR_CURRENT, MOTOR_TORQUE, LOAD_FRACTION})
_UNITS = {JOINT_TORQUE: {"N·m", "N.m", "Nm", "N m"}, JOINT_FORCE: {"N"}}

CLOCKS = ("recorded", "assumed")


def _per_joint(value: Union[str, Sequence[str]], n: int, name: str) -> Tuple[str, ...]:
    if isinstance(value, str):
        return (value,) * n
    value = tuple(str(v) for v in value)
    if len(value) != n:
        raise ValueError(f"{name}: {len(value)} entries for {n} joints")
    return value


@dataclass(frozen=True, eq=False)
class TrajectoryData:
    """A recorded trajectory, in the order of ``joint_names``.

    Arrays are (n_samples, n_joints); ``t`` is (n_samples,) in seconds.
    ``dq``/``ddq`` may be ``None`` (to be derived). ``mask`` marks valid
    samples; nothing is deleted, and filtering runs on the full signal
    before the mask selects rows (:meth:`stacked_mask`).
    """

    t: np.ndarray
    joint_names: Tuple[str, ...]
    q: np.ndarray
    effort: np.ndarray
    effort_kind: Tuple[str, ...]
    effort_unit: Tuple[str, ...]
    clock: str = "recorded"
    dq: Optional[np.ndarray] = None
    ddq: Optional[np.ndarray] = None
    # per signal: "measured", "derived:<method>" or "absent"
    origin: Dict[str, str] = field(default_factory=dict)
    effort_conversion: str = ""  # steps applied to reach `effort`
    effort_raw: Optional[np.ndarray] = None  # recorded signal, if converted
    effort_raw_kind: Tuple[str, ...] = ()
    effort_raw_unit: Tuple[str, ...] = ()
    mask: Optional[np.ndarray] = None
    # source row of each sample in the files (#131); default 0..n-1
    sample_index: Optional[np.ndarray] = None
    source: DataSource = field(default_factory=DataSource)

    def __post_init__(self):
        set_ = object.__setattr__
        t = np.asarray(self.t, dtype=float).reshape(-1)
        set_(self, "t", t)
        names = tuple(str(j) for j in self.joint_names)
        set_(self, "joint_names", names)
        n, nj = len(t), len(names)
        if nj == 0:
            raise ValueError("a trajectory needs at least one joint")
        if len(set(names)) != nj:
            raise ValueError(f"duplicate joint names: {names}")
        if self.clock not in CLOCKS:
            raise ValueError(f"clock must be one of {CLOCKS}, not {self.clock!r}")
        if n < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) > 0):
            raise ValueError("t must hold at least 2 finite, strictly increasing times")
        for key in ("q", "dq", "ddq", "effort", "effort_raw"):
            value = getattr(self, key)
            if value is None:
                continue
            value = np.asarray(value, dtype=float)
            if value.ndim == 1 and nj == 1:
                value = value.reshape(-1, 1)
            if value.shape != (n, nj):
                raise ValueError(f"{key} has shape {value.shape}, expected {(n, nj)}")
            if not np.all(np.isfinite(value)):
                raise ValueError(f"{key} contains non-finite values")
            set_(self, key, value)
        set_(self, "effort_kind", _per_joint(self.effort_kind, nj, "effort_kind"))
        set_(self, "effort_unit", _per_joint(self.effort_unit, nj, "effort_unit"))
        if self.effort_raw is not None:
            set_(
                self,
                "effort_raw_kind",
                _per_joint(self.effort_raw_kind, nj, "effort_raw_kind"),
            )
            set_(
                self,
                "effort_raw_unit",
                _per_joint(self.effort_raw_unit, nj, "effort_raw_unit"),
            )
        mask = np.ones(n, bool) if self.mask is None else np.asarray(self.mask)
        if mask.shape != (n,) or mask.dtype != bool:
            raise ValueError(f"mask must be a bool array of shape {(n,)}")
        set_(self, "mask", mask)
        index = (
            np.arange(n) if self.sample_index is None else np.asarray(self.sample_index)
        )
        if (
            index.shape != (n,)
            or not np.issubdtype(index.dtype, np.integer)
            or not np.all(np.diff(index) > 0)
        ):
            raise ValueError(f"sample_index must be {n} strictly increasing integers")
        set_(self, "sample_index", index)
        origin = {
            "q": "measured",
            "dq": "measured" if self.dq is not None else "absent",
            "ddq": "measured" if self.ddq is not None else "absent",
        }
        origin.update(self.origin)
        set_(self, "origin", origin)

    @property
    def n_samples(self) -> int:
        return len(self.t)

    # ── legacy dictionaries (BaseIdentification.load_trajectory_data) ──

    @classmethod
    def from_legacy(
        cls,
        raw: dict,
        joint_names: Sequence[str],
        *,
        effort_kind: Union[str, Sequence[str]],
        effort_unit: Union[str, Sequence[str]],
        clock: str = "recorded",
        source: Optional[DataSource] = None,
        origin: Optional[Dict[str, str]] = None,
        sample_index: Optional[np.ndarray] = None,
    ) -> "TrajectoryData":
        """From the dict ``load_trajectory_data`` returns today.

        Keys ``timestamps`` ((n, 1) or (n,)), ``positions``, ``velocities``,
        ``accelerations``, ``torques``; velocities/accelerations may be
        ``None``. The legacy dict says nothing about the effort, so its kind
        and unit are required here.
        """
        return cls(
            t=raw["timestamps"],
            joint_names=tuple(joint_names),
            q=raw["positions"],
            dq=raw.get("velocities"),
            ddq=raw.get("accelerations"),
            effort=raw["torques"],
            effort_kind=effort_kind,
            effort_unit=effort_unit,
            clock=clock,
            origin=dict(origin or {}),
            sample_index=sample_index,
            source=source or DataSource(),
        )

    def to_legacy(self) -> dict:
        """The legacy dict, every sample (the mask applies to regressor rows,
        after filtering; see :meth:`stacked_mask`)."""
        return {
            "timestamps": self.t.reshape(-1, 1),
            "positions": self.q,
            "velocities": self.dq,
            "accelerations": self.ddq,
            "torques": self.effort,
        }

    # ── solver-facing views ──

    def stacked_effort(self, effort: Optional[np.ndarray] = None) -> np.ndarray:
        """Joint-major stacking the regressor uses: joint 1's samples, then
        joint 2's, ... (``effort.T.flatten()``)."""
        effort = self.effort if effort is None else np.asarray(effort)
        return effort.T.flatten()

    def stacked_mask(self) -> np.ndarray:
        """Row mask matching :meth:`stacked_effort`."""
        return np.tile(self.mask, len(self.joint_names))

    def with_mask(self, mask: np.ndarray) -> "TrajectoryData":
        return replace(self, mask=np.asarray(mask, dtype=bool))

    def converted(
        self,
        scale: Union[float, Sequence[float]],
        offset: Union[float, Sequence[float]] = 0.0,
        *,
        effort_kind: Union[str, Sequence[str]],
        effort_unit: Union[str, Sequence[str]],
        description: str,
    ) -> "TrajectoryData":
        """``effort * scale + offset`` per joint, keeping the recorded signal.

        For a conversion with known constants (reduction ratio, torque
        constant, gravity offset); ``description`` records the steps.
        """
        scale = np.broadcast_to(
            np.asarray(scale, dtype=float), (len(self.joint_names),)
        )
        offset = np.broadcast_to(
            np.asarray(offset, dtype=float), (len(self.joint_names),)
        )
        raw = self.effort if self.effort_raw is None else self.effort_raw
        raw_kind = self.effort_kind if self.effort_raw is None else self.effort_raw_kind
        raw_unit = self.effort_unit if self.effort_raw is None else self.effort_raw_unit
        return replace(
            self,
            effort=self.effort * scale + offset,
            effort_kind=effort_kind,
            effort_unit=effort_unit,
            effort_conversion=(
                f"{self.effort_conversion}; {description}"
                if self.effort_conversion
                else description
            ),
            effort_raw=raw,
            effort_raw_kind=raw_kind,
            effort_raw_unit=raw_unit,
        )

    def check_effort(self, model, drive_gain_joints: Sequence[str] = ()) -> None:
        """Refuse an effort a dynamic solver must not use as joint effort.

        Accepted per joint: ``joint_torque`` (N·m) on a revolute joint,
        ``joint_force`` (N) on a prismatic joint, or a drive-side kind
        (motor current/torque, load fraction) on a joint listed in
        ``drive_gain_joints``, i.e. whose model identifies an unknown drive
        gain. Anything else would be solved as if it were N·m.

        Raises:
            ValueError: naming every joint, its kind and unit.
        """
        gains = set(drive_gain_joints)
        problems = []
        for name, kind, unit in zip(
            self.joint_names, self.effort_kind, self.effort_unit
        ):
            if not model.existJointName(name):
                problems.append(f"{name}: not in the model")
                continue
            short = model.joints[model.getJointId(name)].shortname()
            revolute = short.startswith("JointModelR")
            prismatic = short.startswith("JointModelP")
            if kind in DRIVE_KINDS and name in gains:
                continue
            if kind == JOINT_TORQUE and revolute and unit in _UNITS[JOINT_TORQUE]:
                continue
            if kind == JOINT_FORCE and prismatic and unit in _UNITS[JOINT_FORCE]:
                continue
            joint_type = "revolute" if revolute else "prismatic" if prismatic else short
            hint = (
                " (declare a drive gain for this joint, or convert it)"
                if kind in DRIVE_KINDS
                else ""
            )
            problems.append(f"{name} ({joint_type}): {kind} in {unit}{hint}")
        if problems:
            raise ValueError("Effort is not joint effort for: " + "; ".join(problems))
