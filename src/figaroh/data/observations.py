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

"""Geometric calibration input: postures and what was observed at them.

See docs/decisions/data-result-contract.md. Kept separate from
:class:`~figaroh.data.trajectory.TrajectoryData`: postures are independent
samples with points or poses in a named frame, not a time series of
efforts.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from figaroh.data.source import DataSource

DOF = ("x", "y", "z", "phix", "phiy", "phiz")


@dataclass(frozen=True, eq=False)
class PoseObservations:
    """Postures ``q`` (n, n_joints) and observations ``values``.

    ``values`` is (n, n_points, 6): x, y, z in m and phix, phiy, phiz in rad
    (convention ``orientation``) of each observed point, NaN where a DOF is
    not observed (``measurability``, (n_points, 6)). ``frame`` names the
    frame the values are expressed in and ``registered_to`` the robot frame
    it is registered to. A constraint-only dataset (e.g. a contact gap that
    must be zero) has ``values=None`` and names its ``constraint``.
    ``mask`` marks valid postures; nothing is deleted.
    """

    joint_names: Tuple[str, ...]
    q: np.ndarray
    values: Optional[np.ndarray] = None
    point_names: Tuple[str, ...] = ()
    measurability: Optional[np.ndarray] = None
    frame: str = ""
    registered_to: str = ""
    orientation: str = "unspecified"
    constraint: Optional[str] = None
    mask: Optional[np.ndarray] = None
    session: Optional[np.ndarray] = None  # (n,) session ids
    # source row of each posture in the files (#131); default 0..n-1
    sample_index: Optional[np.ndarray] = None
    source: DataSource = field(default_factory=DataSource)

    def __post_init__(self):
        set_ = object.__setattr__
        names = tuple(str(j) for j in self.joint_names)
        set_(self, "joint_names", names)
        q = np.asarray(self.q, dtype=float)
        if q.ndim != 2 or q.shape[1] != len(names):
            raise ValueError(f"q has shape {q.shape}, expected (n, {len(names)})")
        if not np.all(np.isfinite(q)):
            raise ValueError("q contains non-finite values")
        set_(self, "q", q)
        n = q.shape[0]
        if self.values is None:
            if self.constraint is None:
                raise ValueError("observations need values or a constraint kind")
        else:
            values = np.asarray(self.values, dtype=float)
            n_points = len(self.point_names)
            if values.shape != (n, n_points, 6):
                raise ValueError(
                    f"values has shape {values.shape}, expected {(n, n_points, 6)}"
                )
            meas = np.asarray(self.measurability, dtype=bool)
            if meas.shape != (n_points, 6):
                raise ValueError(f"measurability must be {(n_points, 6)}")
            if not np.all(np.isfinite(values[:, meas])):
                raise ValueError("observed values contain non-finite entries")
            values = values.copy()
            values[:, ~meas] = np.nan
            set_(self, "values", values)
            set_(self, "measurability", meas)
            set_(self, "point_names", tuple(str(p) for p in self.point_names))
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
            or len(np.unique(index)) != n
        ):
            raise ValueError(f"sample_index must be {n} unique integers")
        set_(self, "sample_index", index)
        if self.session is None:
            sid = self.source.session.id if self.source.session else ""
            session = np.full(n, sid, dtype=object)
        else:
            session = np.asarray(self.session, dtype=object)
            if session.shape != (n,):
                raise ValueError(f"session must have shape {(n,)}")
        set_(self, "session", session)

    @property
    def n_samples(self) -> int:
        return self.q.shape[0]

    def with_mask(self, mask: np.ndarray) -> "PoseObservations":
        return replace(self, mask=np.asarray(mask, dtype=bool))

    def select_points(self, names: Sequence[str]) -> "PoseObservations":
        """The same postures with only the named points (#131).

        E.g. a file with four tracked points, of which a calibration fits
        one: the adapter keeps all four and the calibration selects.
        """
        if self.values is None:
            raise ValueError("constraint-only observations have no points")
        missing = [p for p in names if p not in self.point_names]
        if missing:
            raise KeyError(f"points {missing} not in {list(self.point_names)}")
        idx = [self.point_names.index(p) for p in names]
        return replace(
            self,
            values=self.values[:, idx],
            point_names=tuple(names),
            measurability=self.measurability[idx],
        )

    # ── legacy CSV / (PEE_measured, q_measured) ──

    @classmethod
    def from_csv(
        cls,
        path: Union[str, Path],
        model,
        calib_config: dict,
        *,
        del_list: Sequence[int] = (),
        frame: str = "",
        registered_to: str = "",
        orientation: str = "unspecified",
        source: Optional[DataSource] = None,
    ) -> "PoseObservations":
        """Read the CSV layout ``calibration.data_loader.load_data`` reads.

        Columns ``x1, y1, ... phiz1`` per marker (the measured DOF only) and
        one column per active joint. ``del_list`` rows are masked, not
        deleted. A ``session_id`` column, if present, gives the sessions.
        Does not modify ``calib_config``.

        Raises:
            KeyError: listing every missing column.
        """
        df = pd.read_csv(path)
        n_points = int(calib_config["NbMarkers"])
        meas = np.array(calib_config["measurability"], dtype=bool)
        joints = [model.names[i] for i in calib_config["actJoint_idx"]]
        columns = [
            f"{DOF[d]}{i + 1}" for i in range(n_points) for d in range(6) if meas[d]
        ]
        missing = [c for c in columns + joints if c not in df.columns]
        if missing:
            raise KeyError(f"{path}: missing columns {missing}")
        n = len(df)
        values = np.full((n, n_points, 6), np.nan)
        for i in range(n_points):
            for d in range(6):
                if meas[d]:
                    values[:, i, d] = df[f"{DOF[d]}{i + 1}"].to_numpy(dtype=float)
        mask = np.ones(n, bool)
        mask[list(del_list)] = False
        session = (
            df["session_id"].astype(str).to_numpy(dtype=object)
            if ("session_id" in df.columns)
            else None
        )
        if source is None:
            source = DataSource.from_files([path], adapter="PoseObservations.from_csv")
        return cls(
            joint_names=tuple(joints),
            q=df[joints].to_numpy(dtype=float),
            values=values,
            point_names=tuple(str(i + 1) for i in range(n_points)),
            measurability=np.tile(meas, (n_points, 1)),
            frame=frame,
            registered_to=registered_to,
            orientation=orientation,
            mask=mask,
            session=session,
            sample_index=df.index.to_numpy(),
            source=source,
        )

    def to_legacy(self, model, calib_config: dict) -> Tuple[np.ndarray, np.ndarray]:
        """``(PEE_measured, q_measured)`` exactly as ``load_data`` returns them.

        Valid (unmasked) postures only. ``PEE_measured`` is component-major
        (``x1`` over samples, then ``y1``, ...); ``q_measured`` is the full
        configuration, ``calib_config["q0"]`` with the active joints set.
        Does not modify ``calib_config`` or ``q0``; the sample count is
        ``len(q_measured)``.
        """
        if self.values is None:
            raise ValueError(
                f"constraint-only observations ({self.constraint}) have no "
                "measured values"
            )
        joints = [model.names[i] for i in calib_config["actJoint_idx"]]
        if list(self.joint_names) != joints:
            raise ValueError(
                f"joint order {list(self.joint_names)} differs from the "
                f"calibration's active joints {joints}"
            )
        keep = self.mask
        columns = [
            self.values[keep, i, d]
            for i in range(len(self.point_names))
            for d in range(6)
            if self.measurability[i, d]
        ]
        pee = np.vstack(columns).flatten("C") if columns else np.zeros(0)
        q = np.tile(np.array(calib_config["q0"], dtype=float), (int(keep.sum()), 1))
        q[:, calib_config["config_idx"]] = self.q[keep]
        return pee, q
