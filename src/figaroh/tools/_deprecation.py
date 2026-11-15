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

"""Warnings for defaults that change in a later major release."""

import warnings
from typing import Dict, Optional

__all__ = ["resolve_changing_defaults"]


def resolve_changing_defaults(
    owner: str,
    values: Dict[str, Optional[bool]],
    future: str = "2.0",
    stacklevel: int = 2,
) -> Dict[str, bool]:
    """Resolve flags whose default goes from ``True`` to ``False``.

    A flag passed as ``None`` (left unset) keeps today's default, ``True``.
    If any flag is unset, one ``FutureWarning`` names them all. Flags
    passed explicitly are returned as given and never warn.

    ``FutureWarning`` rather than ``DeprecationWarning``: it is the change
    of what a plain call does, aimed at script authors, and Python shows it
    by default wherever the call is (figaroh-plus#267).

    Args:
        owner: Qualified name of the call, for the message.
        values: Flag name -> value passed (``None`` when unset).
        future: Release in which the default becomes ``False``.
        stacklevel: Frames above the caller of this helper to attribute the
            warning to; the default points at whoever called ``owner``.

    Returns:
        Flag name -> resolved ``bool``.
    """
    unset = [name for name, value in values.items() if value is None]
    if unset:
        names = ", ".join(f"{name}=" for name in unset)
        warnings.warn(
            f"{owner}() called without {names}: today it defaults to True; "
            f"from {future} the default is False. Pass "
            f"{'it' if len(unset) == 1 else 'them'} explicitly to keep the "
            "current output or to turn it off.",
            FutureWarning,
            stacklevel=stacklevel + 1,
        )
    return {
        name: True if value is None else bool(value) for name, value in values.items()
    }
