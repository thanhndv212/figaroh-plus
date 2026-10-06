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

"""Which step produced a result, and whether it ran, failed or fell back.

See docs/decisions/data-result-contract.md (#55). Identification and
calibration record one :class:`StageResult` per step they run on
``obj.stages``; results dictionaries carry them under ``"stages"`` and the
run archive writes them to ``stages.json``. A stage that did not run has no
record, or an explicit ``not_run`` when that is worth stating.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

STAGES = ("data", "fit", "validation", "physical", "export")
STATUSES = ("ok", "failed", "fallback", "not_run")
# version of the stages.json / verdict.json layout
SCHEMA_VERSION = 1


@dataclass
class StageResult:
    stage: str
    status: str
    reason: str = ""
    # name -> {"value": float, "unit": str}
    metrics: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)

    def __post_init__(self):
        if self.stage not in STAGES:
            raise ValueError(f"stage must be one of {STAGES}, not {self.stage!r}")
        if self.status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, not {self.status!r}")


def record_stage(
    obj,
    stage: str,
    status: str,
    reason: str = "",
    metrics: Optional[Dict[str, Any]] = None,
    artifacts: Optional[List[str]] = None,
) -> StageResult:
    """Record ``stage`` on ``obj.stages``, replacing an earlier record of it.

    ``metrics`` maps a name to ``(value, unit)`` or to a bare value (unit
    ``""``).
    """
    clean = {}
    for name, value in (metrics or {}).items():
        if isinstance(value, tuple):
            value, unit = value
        else:
            unit = ""
        clean[name] = {"value": value, "unit": unit}
    result = StageResult(stage, status, reason, clean, list(artifacts or []))
    stages = [s for s in getattr(obj, "stages", None) or [] if s.stage != stage]
    stages.append(result)
    stages.sort(key=lambda s: STAGES.index(s.stage))
    obj.stages = stages
    return result


def stages_as_dicts(obj) -> List[Dict[str, Any]]:
    """``obj.stages`` as plain dicts (for results dictionaries and JSON)."""
    return [asdict(s) for s in getattr(obj, "stages", None) or []]


def with_schema(obj, verdict: Dict[str, Any]) -> Dict[str, Any]:
    """A verdict dict with ``schema_version`` and the run's ``stages``."""
    return {"schema_version": SCHEMA_VERSION, **verdict, "stages": stages_as_dicts(obj)}
