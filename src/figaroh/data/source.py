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

"""Where a dataset came from (docs/decisions/data-result-contract.md)."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Optional, Union


def file_sha256(path: Union[str, Path]) -> str:
    """sha256 of a file's bytes."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class Session:
    """A recording's identity: what it is, not how a study uses it.

    Roles (training, validation, ...) belong to a
    :class:`~figaroh.data.protocol.Protocol`, not here.
    """

    id: str
    date: Optional[str] = None  # ISO 8601 date or timestamp


@dataclass(frozen=True)
class DataSource:
    """Files (with sha256), the adapter that read them, and the session."""

    files: Dict[str, str] = field(default_factory=dict)  # path -> sha256
    adapter: str = ""
    session: Optional[Session] = None
    notes: str = ""

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
