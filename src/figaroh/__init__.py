# Copyright [2021-2026] Thanh Nguyen
"""
figaroh: Robot and Human Identification Tools
"""

from . import tools
from . import calibration
from . import identification
from . import measurements
from . import utils
from . import visualisation
from . import optimal
from . import data
from . import backends
from . import integration

# Subpackages are imported eagerly so ``import figaroh`` exposes them as
# attributes (``figaroh.tools.robot``, ...); __all__ declares that surface.
# The stability of each one is stated in docs/source/api/stability.md.
__all__ = [
    "tools",
    "calibration",
    "identification",
    "measurements",
    "utils",
    "visualisation",
    "optimal",
    "data",
    "backends",
    "integration",
]

__version__ = "1.0.1"
