# Copyright [2021-2026] Thanh Nguyen
# Copyright [2022-2024] [CNRS, Toward SAS]

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Random-number sources for FIGAROH's generators (figaroh-plus#266).

Every generator that draws random numbers takes ``rng``:

- ``None`` (default): draw from NumPy's global legacy state, as before, so
  a caller that seeds with ``np.random.seed(s)`` gets the same numbers;
- an ``int`` or ``np.random.SeedSequence``: a fresh ``np.random.Generator``
  seeded with it, independent of the global state;
- an ``np.random.Generator``: used as is, so several calls can share one
  stream.
"""

from typing import Any, Union

import numpy as np

__all__ = ["RandomSource", "resolve_rng"]

RandomSource = Union[None, int, np.random.SeedSequence, np.random.Generator]


def resolve_rng(rng: RandomSource = None) -> Any:
    """Return the object to draw from for ``rng``.

    The result has the ``uniform``/``normal``/``choice`` methods both the
    ``np.random`` module and ``np.random.Generator`` provide.

    Args:
        rng: ``None``, a seed, a ``SeedSequence`` or a ``Generator``.

    Returns:
        The ``np.random`` module for ``None``; otherwise a ``Generator``.
    """
    # the module itself comes back in when a resolved ``None`` is passed on
    # (BaseOptimalTrajectory -> WaypointsGeneration)
    if rng is None or rng is np.random:
        return np.random
    return np.random.default_rng(rng)
