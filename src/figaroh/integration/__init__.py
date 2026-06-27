"""
FIGAROH Integration API

High-level API for robot identification and calibration workflows.
Provides simple, intuitive interfaces that wrap the backend abstraction
and workflow classes.

Quick Start:
-----------
>>> from figaroh.integration import RobotIdentificationSystem
>>>
>>> # One-line identification
>>> system = RobotIdentificationSystem.from_urdf("robot.urdf", backend="pinocchio")
>>> results = system.identify_parameters(config="config.yaml", data_dir="trajectory/")
"""

from .api import RobotIdentificationSystem, IdentificationResult

__all__ = ["RobotIdentificationSystem", "IdentificationResult"]
