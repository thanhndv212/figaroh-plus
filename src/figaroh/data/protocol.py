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

"""Which session plays which role in a study (data-result-contract.md).

A session's identity is part of the data (:class:`~figaroh.data.source.
Session`); its role is not: the same recording can validate one study and
train another. A ``Protocol`` assigns roles and names each session's files
by sha256, so a frozen protocol cannot silently change under it.

YAML layout::

    name: tiago-mocap-heldout
    version: 1
    sessions:
      - id: 2021-11-30-1544
        role: training
        files:
          data/qualisys_2021-11-30_static_postures.csv: b6c0051e...
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import yaml

from figaroh.data.source import file_sha256


@dataclass(frozen=True)
class ProtocolSession:
    id: str
    role: str
    files: Dict[str, str] = field(default_factory=dict)  # path -> sha256


@dataclass(frozen=True)
class Protocol:
    name: str
    version: int
    sessions: List[ProtocolSession]

    def __post_init__(self):
        ids = [s.id for s in self.sessions]
        if len(set(ids)) != len(ids):
            raise ValueError(f"Protocol {self.name}: duplicate session ids {ids}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Protocol":
        with open(path) as f:
            raw = yaml.safe_load(f)
        sessions = [
            ProtocolSession(
                id=str(s["id"]),
                role=str(s["role"]),
                files={str(k): str(v) for k, v in (s.get("files") or {}).items()},
            )
            for s in raw.get("sessions", [])
        ]
        return cls(
            name=str(raw["name"]), version=int(raw["version"]), sessions=sessions
        )

    def session(self, session_id: str) -> ProtocolSession:
        for s in self.sessions:
            if s.id == session_id:
                return s
        raise KeyError(f"Protocol {self.name}: no session {session_id!r}")

    def with_role(self, role: str) -> List[ProtocolSession]:
        return [s for s in self.sessions if s.role == role]

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
