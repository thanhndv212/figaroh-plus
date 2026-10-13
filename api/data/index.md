# Data contract

Typed inputs that carry their conventions: joint order, clock, signal origin, effort kind and unit, valid-sample mask, session and source files. See the decision record [Data and result contract](https://github.com/thanhndv212/figaroh-plus/blob/devel/docs/decisions/data-result-contract.md). Each type converts to and from the legacy dictionaries and CSV loader.

Dynamic identification input: a trajectory with its conventions attached.

See docs/decisions/data-result-contract.md. Joint order, clock, signal origin, effort kind and unit, the valid-sample mask and the source travel with the arrays instead of being implied by an adapter.

The effort keeps the units it was recorded in (a motor current, a motor-side torque, a load fraction, a joint torque, or a linear actuator's force). :meth:`TrajectoryData.check_effort` decides whether a solver may use it.

## `TrajectoryData(t, joint_names, q, effort, effort_kind, effort_unit, clock='recorded', dq=None, ddq=None, origin=dict(), effort_conversion='', effort_raw=None, effort_raw_kind=(), effort_raw_unit=(), mask=None, sample_index=None, source=DataSource())`

A recorded trajectory, in the order of `joint_names`.

Arrays are (n_samples, n_joints); `t` is (n_samples,) in seconds. `dq`/`ddq` may be `None` (to be derived). `mask` marks valid samples; nothing is deleted, and filtering runs on the full signal before the mask selects rows (:meth:`stacked_mask`).

### `from_legacy(raw, joint_names, *, effort_kind, effort_unit, clock='recorded', source=None, origin=None, sample_index=None)`

From the dict `load_trajectory_data` returns today.

Keys `timestamps` ((n, 1) or (n,)), `positions`, `velocities`, `accelerations`, `torques`; velocities/accelerations may be `None`. The legacy dict says nothing about the effort, so its kind and unit are required here.

Source code in `src/figaroh/data/trajectory.py`

```
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
```

### `to_legacy()`

The legacy dict, every sample (the mask applies to regressor rows, after filtering; see :meth:`stacked_mask`).

Source code in `src/figaroh/data/trajectory.py`

```
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
```

### `stacked_effort(effort=None)`

Joint-major stacking the regressor uses: joint 1's samples, then joint 2's, ... (`effort.T.flatten()`).

Source code in `src/figaroh/data/trajectory.py`

```
def stacked_effort(self, effort: Optional[np.ndarray] = None) -> np.ndarray:
    """Joint-major stacking the regressor uses: joint 1's samples, then
    joint 2's, ... (``effort.T.flatten()``)."""
    effort = self.effort if effort is None else np.asarray(effort)
    return effort.T.flatten()
```

### `stacked_mask()`

Row mask matching :meth:`stacked_effort`.

Source code in `src/figaroh/data/trajectory.py`

```
def stacked_mask(self) -> np.ndarray:
    """Row mask matching :meth:`stacked_effort`."""
    return np.tile(self.mask, len(self.joint_names))
```

### `converted(scale, offset=0.0, *, effort_kind, effort_unit, description)`

`effort * scale + offset` per joint, keeping the recorded signal.

For a conversion with known constants (reduction ratio, torque constant, gravity offset); `description` records the steps.

Source code in `src/figaroh/data/trajectory.py`

```
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
```

### `check_effort(model, drive_gain_joints=())`

Refuse an effort a dynamic solver must not use as joint effort.

Accepted per joint: `joint_torque` (N·m) on a revolute joint, `joint_force` (N) on a prismatic joint, or a drive-side kind (motor current/torque, load fraction) on a joint listed in `drive_gain_joints`, i.e. whose model identifies an unknown drive gain. Anything else would be solved as if it were N·m.

Raises:

| Type         | Description                            |
| ------------ | -------------------------------------- |
| `ValueError` | naming every joint, its kind and unit. |

Source code in `src/figaroh/data/trajectory.py`

```
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
```

Geometric calibration input: postures and what was observed at them.

See docs/decisions/data-result-contract.md. Kept separate from :class:`~figaroh.data.trajectory.TrajectoryData`: postures are independent samples with points or poses in a named frame, not a time series of efforts.

## `PoseObservations(joint_names, q, values=None, point_names=(), measurability=None, frame='', registered_to='', orientation='unspecified', constraint=None, mask=None, session=None, sample_index=None, source=DataSource())`

Postures `q` (n, n_joints) and observations `values`.

`values` is (n, n_points, 6): x, y, z in m and phix, phiy, phiz in rad (convention `orientation`) of each observed point, NaN where a DOF is not observed (`measurability`, (n_points, 6)). `frame` names the frame the values are expressed in and `registered_to` the robot frame it is registered to. A constraint-only dataset (e.g. a contact gap that must be zero) has `values=None` and names its `constraint`. `mask` marks valid postures; nothing is deleted.

### `select_points(names)`

The same postures with only the named points (#131).

E.g. a file with four tracked points, of which a calibration fits one: the adapter keeps all four and the calibration selects.

Source code in `src/figaroh/data/observations.py`

```
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
```

### `from_csv(path, model, calib_config, *, del_list=(), frame='', registered_to='', orientation='unspecified', source=None)`

Read the CSV layout `calibration.data_loader.load_data` reads.

Columns `x1, y1, ... phiz1` per marker (the measured DOF only) and one column per active joint. `del_list` rows are masked, not deleted. A `session_id` column, if present, gives the sessions. Does not modify `calib_config`.

Raises:

| Type       | Description                   |
| ---------- | ----------------------------- |
| `KeyError` | listing every missing column. |

Source code in `src/figaroh/data/observations.py`

```
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
```

### `to_legacy(model, calib_config)`

`(PEE_measured, q_measured)` exactly as `load_data` returns them.

Valid (unmasked) postures only. `PEE_measured` is component-major (`x1` over samples, then `y1`, ...); `q_measured` is the full configuration, `calib_config["q0"]` with the active joints set. Does not modify `calib_config` or `q0`; the sample count is `len(q_measured)`.

Source code in `src/figaroh/data/observations.py`

```
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
```

Where a dataset came from (docs/decisions/data-result-contract.md).

## `Session(id, date=None)`

A recording's identity: what it is, not how a study uses it.

Roles (training, validation, ...) belong to a :class:`~figaroh.data.protocol.Protocol`, not here.

## `DataSource(files=dict(), adapter='', session=None, notes='')`

Files (with sha256), the adapter that read them, and the session.

### `from_files(paths, adapter='', session=None, notes='')`

Hash the given files now.

Source code in `src/figaroh/data/source.py`

```
@classmethod
def from_files(
    cls,
    paths: Iterable[Union[str, Path]],
    adapter: str = "",
    session: Optional[Session] = None,
    notes: str = "",
) -> "DataSource":
    """Hash the given files now."""
    files = {str(Path(p)): file_sha256(p) for p in paths}
    return cls(files=files, adapter=adapter, session=session, notes=notes)
```

## `file_sha256(path)`

sha256 of a file's bytes.

Source code in `src/figaroh/data/source.py`

```
def file_sha256(path: Union[str, Path]) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
```

Which session plays which role in a study (data-result-contract.md).

A session's identity is part of the data (:class:`~figaroh.data.source. Session`); its role is not: the same recording can validate one study and train another. A `Protocol` assigns roles and names each session's files by sha256, so a frozen protocol cannot silently change under it.

YAML layout::

```
name: tiago-mocap-heldout
version: 1
sessions:
  - id: 2021-11-30-1544
    role: training
    files:
      data/qualisys_2021-11-30_static_postures.csv: b6c0051e...
```

## `Protocol(name, version, sessions)`

### `verify(root=None)`

Check every session file against its recorded sha256.

Paths are relative to `root` (default: the working directory).

Raises:

| Type         | Description                            |
| ------------ | -------------------------------------- |
| `ValueError` | listing every missing or changed file. |

Source code in `src/figaroh/data/protocol.py`

```
def verify(self, root: Optional[Union[str, Path]] = None) -> None:
    """Check every session file against its recorded sha256.

    Paths are relative to ``root`` (default: the working directory).

    Raises:
        ValueError: listing every missing or changed file.
    """
    root = Path(root) if root is not None else Path.cwd()
    problems = []
    for s in self.sessions:
        for rel, expected in s.files.items():
            path = root / rel
            if not path.exists():
                problems.append(f"{s.id}: {rel} is missing")
            elif file_sha256(path) != expected:
                problems.append(f"{s.id}: {rel} does not match its sha256")
    if problems:
        raise ValueError(
            f"Protocol {self.name} v{self.version}: " + "; ".join(problems)
        )
```
