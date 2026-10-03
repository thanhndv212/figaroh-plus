# Copyright [2021-2025] Thanh Nguyen
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

# Subpackages are imported eagerly so ``import figaroh`` exposes them as
# attributes (``figaroh.tools.robot``, ...); __all__ declares that surface.
__all__ = [
    "tools",
    "calibration",
    "identification",
    "measurements",
    "utils",
    "visualisation",
    "optimal",
]

__version__ = "0.5.0"
