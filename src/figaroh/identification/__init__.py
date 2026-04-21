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

from .base_identification import BaseIdentification

from .reconstruction import (  # noqa: F401
    BaseResult,
    ReconstructionResult,
    prior_vector_from_dict,
    reconstruct_from_base,
    reconstruct_full_parameters,
    reconstruct_theta_r,
    run_reconstruction,
    run_option_a_reconstruction,
)

# Optional (default-off) physical consistency helpers
from .physical_consistency import (  # noqa: F401
    ProjectionReport,
    RobotProjectionReport,
    check_p10_feasibility,
    is_feasible_link,
    p10_by_joint_from_param_dict,
    param_dict_with_p10_by_joint,
    project_link,
    project_p10_lmi,
    project_robot_p10_lmi,
    pseudo_inertia_matrix_from_p10,
)

__all__ = [
    "BaseIdentification",
    "BaseResult",
    "ReconstructionResult",
    "prior_vector_from_dict",
    "reconstruct_from_base",
    "reconstruct_full_parameters",
    "reconstruct_theta_r",
    "run_reconstruction",
    "run_option_a_reconstruction",
    "ProjectionReport",
    "RobotProjectionReport",
    "check_p10_feasibility",
    "is_feasible_link",
    "p10_by_joint_from_param_dict",
    "param_dict_with_p10_by_joint",
    "project_link",
    "project_p10_lmi",
    "project_robot_p10_lmi",
    "pseudo_inertia_matrix_from_p10",
]
