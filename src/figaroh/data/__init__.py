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

"""Data contract types (docs/decisions/data-result-contract.md, #55).

Additive: the legacy dictionaries and CSV loaders keep working, and each
type converts to and from them.
"""

from figaroh.data.observations import PoseObservations
from figaroh.data.protocol import Protocol, ProtocolSession
from figaroh.data.source import DataSource, Session, file_sha256
from figaroh.data.trajectory import (
    DRIVE_KINDS,
    JOINT_FORCE,
    JOINT_TORQUE,
    LOAD_FRACTION,
    MOTOR_CURRENT,
    MOTOR_TORQUE,
    TrajectoryData,
)

__all__ = [
    "DataSource",
    "Session",
    "file_sha256",
    "Protocol",
    "ProtocolSession",
    "TrajectoryData",
    "PoseObservations",
    "JOINT_TORQUE",
    "JOINT_FORCE",
    "MOTOR_CURRENT",
    "MOTOR_TORQUE",
    "LOAD_FRACTION",
    "DRIVE_KINDS",
]
