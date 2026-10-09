"""Backward-compatible alias of :mod:`figaroh.identification.physical_fit`.

The direct LMI effort fit and its companions were promoted to a public
module. Importing this name returns that very module object, so every name
(including the private helpers used by the figaroh-examples D4 comparison)
resolves identically and monkeypatching through either name is equivalent.
"""

import sys

from figaroh.identification import physical_fit as _pf

sys.modules[__name__] = _pf
