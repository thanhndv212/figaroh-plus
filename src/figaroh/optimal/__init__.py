"""
Optimal calibration and trajectory optimization module.

This module provides base classes for optimal calibration and trajectory
optimization for robotic systems.
"""

from .base_optimal_calibration import BaseOptimalCalibration
from .base_optimal_trajectory import BaseOptimalTrajectory

__all__ = ["BaseOptimalCalibration", "BaseOptimalTrajectory"]
