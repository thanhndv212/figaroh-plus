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

"""Run provenance capture — the single source of truth for "what produced
this fit": the nominal reference model, the exact config used, software
versions, input data files, and (if configured) the physical asset
identity. Consumed identically by the terminal report, the HTML report,
the JSON verification verdict, and the run archive, so the four can never
disagree about what produced a given result.

Reuses :func:`figaroh.tools._report_common._git_commit_hash` and
:func:`figaroh.tools._report_common._config_file_sha256` rather than
duplicating hashing logic.
"""

import logging
import platform
import socket
import subprocess
from datetime import datetime, timezone
from os.path import exists, getmtime
from typing import Any, Dict, Optional

from figaroh.tools._report_common import _config_file_sha256, _git_commit_hash

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Curated, task-specific config keys worth surfacing in a report — not the
# entire resolved config dict, which is large and mostly derived/internal
# (index arrays, cached joint objects, etc.).
_IDENTIFICATION_CONFIG_KEYS = [
    "active_joints",
    "has_friction",
    "has_joint_offset",
    "has_actuator_inertia",
    "has_coupled_wrist",
    "wls",
    "is_external_wrench",
    "cut_off_frequency_butterworth",
    "ts",
    "nb_samples",
]
_CALIBRATION_CONFIG_KEYS = [
    "calib_model",
    "non_geom",
    "NbSample",
    "NbMarkers",
    "start_frame",
    "end_frame",
    "outlier_eps",
    "coeff_regularize",
    "random_seed",
    "free_flyer",
]

# Config keys that hold paths to input data files — scanned opportunistically
# (best-effort; a robot/task that doesn't set a given key simply contributes
# nothing for it, never an error).
_DATA_PATH_KEYS = [
    "pos_data",
    "vel_data",
    "torque_data",
    "data_file",
    "validation_data_file",
    "sample_configs_file",
]


def _git_dirty() -> Optional[bool]:
    """Best-effort "does the working tree have uncommitted changes" —
    never raises. ``None`` if it can't be determined (e.g. not a git repo)."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return bool(result.stdout.strip())
    except Exception:
        pass
    return None


def _package_revision() -> Dict[str, Any]:
    """Git revision of the figaroh package itself, if it runs from a checkout.

    ``software.git_commit`` is the working directory's commit: for an
    example run, the examples repository. The core revision is recorded
    here so a run names both halves of the pair (#63). ``unknown`` for an
    installed (non-git) package. Never raises.
    """
    import os

    try:
        import figaroh

        pkg_dir = os.path.dirname(os.path.abspath(figaroh.__file__))
    except Exception:
        return {"commit": "unknown", "dirty": None}
    out: Dict[str, Any] = {"commit": "unknown", "dirty": None}
    try:
        head = subprocess.run(
            ["git", "-C", pkg_dir, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if head.returncode == 0:
            out["commit"] = head.stdout.strip()
            status = subprocess.run(
                ["git", "-C", pkg_dir, "status", "--porcelain", "--", "."],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if status.returncode == 0:
                out["dirty"] = bool(status.stdout.strip())
    except Exception:
        pass
    return out


def collect_splits(obj: Any) -> Dict[str, Any]:
    """Which data trained and which validated this run (#63).

    From the data-contract objects when the run used them
    (``trajectory`` / ``observations`` and their validation counterparts:
    files with sha256, session, adapter); ``"legacy input"`` otherwise,
    whose files are listed under provenance ``data``. ``validation_source``
    comes from the run's ``validation`` stage: ``held_out``, or
    ``training_fallback`` when no held-out data was available.
    """

    def describe(data):
        if data is None:
            return None
        source = data.source
        return {
            "files": dict(source.files),
            "session": source.session.id if source.session else None,
            "adapter": source.adapter,
            "samples": int(data.n_samples),
            "masked_samples": int((~data.mask).sum()),
        }

    training = getattr(obj, "trajectory", None) or getattr(obj, "observations", None)
    validation = getattr(obj, "_val_trajectory", None) or getattr(
        obj, "_val_observations", None
    )
    status = {s.stage: s.status for s in getattr(obj, "stages", None) or []}
    source = {"ok": "held_out", "fallback": "training_fallback"}.get(
        status.get("validation"), "none"
    )
    return {
        "training": describe(training) or "legacy input (see provenance data)",
        "validation": describe(validation),
        "validation_source": source,
    }


def _software_versions() -> Dict[str, str]:
    versions = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import figaroh

        versions["figaroh"] = getattr(figaroh, "__version__", "unknown")
    except Exception:
        versions["figaroh"] = "unknown"
    try:
        import pinocchio

        versions["pinocchio"] = getattr(pinocchio, "__version__", "unknown")
    except Exception:
        versions["pinocchio"] = "unknown"
    return versions


def _asset_identity(config: Dict[str, Any]) -> Dict[str, Any]:
    """The physical unit this run was performed on, if configured.

    Reads ``config["instance"]`` (mapped from the optional YAML
    ``robot.instance`` block — see
    :func:`figaroh.utils.config_parser.create_task_config`). Example
    scripts may overlay CLI ``--asset-id``/``--operator`` values into this
    same dict before calling ``solve()``, so CLI > config is achieved by
    the caller, not here.

    Never fails: an unconfigured instance renders as an explicit
    "unspecified" placeholder rather than silently reusing the model name
    as if it uniquely identified a physical robot.
    """
    instance = config.get("instance") or {}
    if not isinstance(instance, dict):
        instance = {}
    model_name = config.get("robot_name", "robot")
    asset_id = instance.get("asset_id") or None
    return {
        "asset_id": asset_id or f"{model_name}-unspecified",
        "is_specified": bool(asset_id),
        "serial_number": instance.get("serial_number", ""),
        "label": instance.get("label", ""),
        "site": instance.get("site", ""),
        "operator": instance.get("operator", ""),
    }


def _model_identity(obj: Any) -> Dict[str, Any]:
    """The nominal reference model (URDF + its content hash) a fit was
    computed against."""
    robot = getattr(obj, "robot", None)
    model = getattr(obj, "model", None)
    urdf_path = getattr(robot, "robot_urdf", None)
    # self.robot_name (if explicitly set) takes precedence over
    # model.name — the same precedence _store_results()/
    # _store_optimization_results() already use when naming the
    # ResultsManager, so a subclass (or test double) that sets it
    # directly is honored consistently everywhere.
    robot_name = getattr(obj, "robot_name", None) or getattr(model, "name", None)
    return {
        "robot_name": robot_name or "unknown",
        "urdf_path": urdf_path or "unavailable",
        "urdf_sha256": _config_file_sha256(urdf_path),
        "nq": int(model.nq) if model is not None else None,
        "nv": int(model.nv) if model is not None else None,
        "njoints": int(model.njoints) if model is not None else None,
    }


def _config_values(config: Dict[str, Any], keys: list) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    for key in keys:
        if key not in config:
            continue
        value = config[key]
        if isinstance(value, (list, tuple)):
            value = list(value)
        values[key] = value
    return values


def _data_files_provenance(config: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort hash/mtime of input data files referenced by the
    config. A missing/unset key is simply omitted — never an error."""
    files: Dict[str, Any] = {}
    for key in _DATA_PATH_KEYS:
        path = config.get(key)
        if not path or not isinstance(path, str):
            continue
        if not exists(path):
            files[key] = {"path": path, "status": "not_found"}
            continue
        try:
            mtime = datetime.fromtimestamp(getmtime(path), tz=timezone.utc).isoformat()
        except OSError:
            mtime = "unavailable"
        files[key] = {
            "path": path,
            "sha256": _config_file_sha256(path),
            "mtime": mtime,
        }
    return files


def collect_run_provenance(obj: Any, task: str) -> Dict[str, Any]:
    """Build the full provenance record for one V&V run.

    Args:
        obj: A ``BaseIdentification`` or ``BaseCalibration`` instance
            (or duck-typed equivalent) after its config has been loaded
            — i.e. safe to call any time after ``load_param()``, though
            it is normally captured once in ``_store_results``/
            ``_store_optimization_results`` right after ``solve()``
            finishes computing, so ``run_finished`` is accurate.
        task: ``"identification"`` or ``"calibration"`` — selects which
            curated config-key allowlist to surface under
            ``config.values``.

    Returns:
        A JSON-serializable nested dict (see module docstring for the
        four consumers). Never raises: every sub-lookup is best-effort.
    """
    config = (
        getattr(obj, "identif_config", None) or getattr(obj, "calib_config", None) or {}
    )
    config_keys = (
        _IDENTIFICATION_CONFIG_KEYS
        if task == "identification"
        else _CALIBRATION_CONFIG_KEYS
    )

    asset = _asset_identity(config)
    now = datetime.now(timezone.utc).isoformat()
    git_commit = _git_commit_hash()
    run_started = getattr(obj, "_run_started_at", None) or now
    run_finished = getattr(obj, "_run_finished_at", None) or now

    run_id = "_".join(
        [
            asset["asset_id"].replace(" ", "-"),
            task,
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            git_commit[:8] if git_commit != "unknown" else "nogit",
        ]
    )

    config_path = getattr(obj, "_config_file_path", None)

    return {
        "run_id": run_id,
        "task": task,
        "asset": asset,
        "model": _model_identity(obj),
        "config": {
            "path": config_path or "unavailable",
            "sha256": _config_file_sha256(config_path),
            "values": _config_values(config, config_keys),
        },
        "software": {
            **_software_versions(),
            # working directory's repository (an example run: examples)
            "git_commit": git_commit,
            "git_dirty": _git_dirty(),
            # the figaroh package's own checkout, the other half of the pair
            "figaroh_revision": _package_revision(),
            "hostname": socket.gethostname(),
        },
        "data": _data_files_provenance(config),
        "timestamps": {
            "run_started": run_started,
            "run_finished": run_finished,
            "report_generated": now,
        },
    }
