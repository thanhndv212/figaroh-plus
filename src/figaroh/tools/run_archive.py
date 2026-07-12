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

"""Longitudinal run archive — a fleet's V&V history.

``export_html_report()``/``export_verification_report()`` write to fixed
filenames that get overwritten by the next run, so a robot calibrated
monthly only ever has its most recent report on disk. :func:`archive_run`
instead writes each run to its own immutable, timestamped directory under
``results/runs/{asset_id}/{task}/{timestamp}_{git_short}/`` and appends a
one-line summary to ``results/runs/index.jsonl`` — a zero-dependency,
git-diffable, ``jq``-able index of every run ever performed, per asset.

Call after :meth:`~figaroh.identification.base_identification.BaseIdentification.solve`
(and, typically, after ``export_html_report()``/``export_verification_report()``
so their output gets copied in alongside the raw provenance/parameters).
"""

import csv
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _safe_path_component(value: Any) -> str:
    """Collapse anything that isn't alnum/dash/dot/underscore into '-',
    so a user-supplied asset_id can never escape the archive root or
    collide with filesystem-reserved characters."""
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(value)).strip("-")
    return cleaned or "unnamed"


def _yaml_safe(value: Any) -> Any:
    """Best-effort conversion to pure Python primitives for YAML/JSON
    dumping. Config dicts carry numpy arrays and, for some robots,
    live pinocchio joint objects (``act_J``) — this never raises;
    anything it doesn't recognize becomes its ``str()``."""
    if isinstance(value, dict):
        return {str(k): _yaml_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_yaml_safe(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.integer, np.floating)):
            return value.item()
        if isinstance(value, np.bool_):
            return bool(value)
    except ImportError:
        pass
    return str(value)


def _extract_parameters(obj: Any) -> Tuple[List[str], List[Any]]:
    """(names, values) of the fitted result — identification and
    calibration objects expose this under different attribute names."""
    result = getattr(obj, "result", None)
    if isinstance(result, dict) and "base parameters names" in result:
        return (
            list(result.get("base parameters names", [])),
            list(result.get("base parameters values", [])),
        )
    calib_config = getattr(obj, "calib_config", None)
    results_data = getattr(obj, "results_data", None)
    if isinstance(calib_config, dict) and isinstance(results_data, dict):
        return (
            list(calib_config.get("param_name", [])),
            list(results_data.get("calibrated parameters values", [])),
        )
    return [], []


def _append_index(
    root: Path,
    provenance: Dict[str, Any],
    run_dir: Path,
) -> None:
    passed = None
    metrics: Dict[str, Any] = {}
    verdict_path = run_dir / "verdict.json"
    if verdict_path.exists():
        try:
            with open(verdict_path) as f:
                verdict_data = json.load(f)
            passed = verdict_data.get("passed")
            metrics = verdict_data.get("metrics", {})
        except (OSError, ValueError) as e:
            logger.warning(f"Could not read verdict for index entry: {e}")

    entry = {
        "run_id": provenance.get("run_id"),
        "asset_id": provenance.get("asset", {}).get("asset_id"),
        "is_specified": provenance.get("asset", {}).get("is_specified"),
        "task": provenance.get("task"),
        "robot_name": provenance.get("model", {}).get("robot_name"),
        "run_finished": provenance.get("timestamps", {}).get("run_finished"),
        "passed": passed,
        "metrics": metrics,
        "path": str(run_dir),
    }
    index_path = root / "index.jsonl"
    with open(index_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def compute_run_dir(obj: Any, root: str = "results/runs") -> Path:
    """Compute this run's dedicated directory path from provenance.

    Deterministic for a given ``obj._run_provenance`` — safe to call more
    than once (e.g. once for the HTML report's output_path, again for the
    verdict's) and get back the identical path, since the timestamp is
    read from the already-fixed provenance record, never re-derived from
    wall clock.

    Args:
        obj: A ``BaseIdentification``/``BaseCalibration`` instance after
            :meth:`solve` has been called (i.e. ``_run_provenance`` is
            populated).
        root: Archive root directory (created if missing).

    Returns:
        Path: The directory for this run, created if missing.

    Raises:
        AttributeError: If called before :meth:`solve`.
    """
    provenance = getattr(obj, "_run_provenance", None)
    if provenance is None:
        raise AttributeError(
            "No run provenance available. Run solve() first."
        )

    task = provenance.get("task", "unknown")
    asset_id = provenance.get("asset", {}).get("asset_id", "unknown")

    run_finished = provenance.get("timestamps", {}).get("run_finished")
    try:
        dt = datetime.fromisoformat(run_finished)
    except (TypeError, ValueError):
        dt = datetime.now(timezone.utc)
    ts_compact = dt.strftime("%Y%m%dT%H%M%SZ")
    git_short = provenance.get("software", {}).get("git_commit", "nogit")[:8]

    root_path = Path(root)
    run_dir = (
        root_path
        / _safe_path_component(asset_id)
        / _safe_path_component(task)
        / f"{ts_compact}_{_safe_path_component(git_short)}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def archive_run(obj: Any, run_dir: Path) -> str:
    """Finalize archiving into an already-computed run_dir.

    Writes provenance.json / config.snapshot.yaml / parameters.csv, and
    appends the index.jsonl entry (reading run_dir/verdict.json for
    pass/metrics if the caller already wrote one there).

    Args:
        obj: A ``BaseIdentification``/``BaseCalibration`` instance after
            :meth:`solve` has been called (i.e. ``_run_provenance`` is
            populated).
        run_dir: The directory for this run (typically obtained via
            :func:`compute_run_dir`). Expected to already exist; the
            caller should have written ``report.html`` and/or
            ``verdict.json`` there before calling this function.

    Returns:
        str: The path to this run's archive directory (echoed back for
            convenience).

    Raises:
        AttributeError: If called before :meth:`solve`.
    """
    provenance = getattr(obj, "_run_provenance", None)
    if provenance is None:
        raise AttributeError(
            "No run provenance available. Run solve() first."
        )

    with open(run_dir / "provenance.json", "w") as f:
        json.dump(provenance, f, indent=2, default=str)

    config = getattr(obj, "identif_config", None) or getattr(
        obj, "calib_config", None
    )
    if config:
        with open(run_dir / "config.snapshot.yaml", "w") as f:
            yaml.dump(
                _yaml_safe(config), f, default_flow_style=False, sort_keys=True
            )

    param_names, param_values = _extract_parameters(obj)
    if param_names:
        with open(run_dir / "parameters.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["parameter", "value"])
            writer.writerows(zip(param_names, param_values))

    # Determine the root for index (run_dir is .../root/asset/task/timestamp/)
    root_path = run_dir.parent.parent.parent
    _append_index(root_path, provenance, run_dir)
    logger.info(f"Run archived to {run_dir}")
    return str(run_dir)
