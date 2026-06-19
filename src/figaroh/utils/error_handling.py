"""
Custom exceptions and error handling for FIGAROH examples.

This module provides standardized error handling and validation
utilities for robot identification and calibration workflows.
"""

import functools
import logging
from typing import Any, Callable, Dict, Optional, TypeVar, Union
import numpy as np

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Type variable for decorators
F = TypeVar("F", bound=Callable[..., Any])


# Custom Exception Classes
class FigarohExampleError(Exception):
    """Base exception for FIGAROH examples."""

    pass


class RobotInitializationError(FigarohExampleError):
    """Exception raised when robot initialization fails."""

    pass


class ConfigurationError(FigarohExampleError):
    """Exception raised for configuration-related issues."""

    pass


class DataProcessingError(FigarohExampleError):
    """Exception raised during data processing operations."""

    pass


class CalibrationError(FigarohExampleError):
    """Exception raised during calibration procedures."""

    pass


class IdentificationError(FigarohExampleError):
    """Exception raised during identification procedures."""

    pass


class ValidationError(FigarohExampleError):
    """Exception raised when validation fails."""

    pass


# Validation Functions
def validate_robot_config(config: Dict[str, Any]) -> None:
    """
    Validate robot configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Raises:
        ValidationError: If configuration is invalid
    """
    if not isinstance(config, dict):
        raise ValidationError("Configuration must be a dictionary")

    # Check for required fields
    required_fields = ["robot_name"]
    missing_fields = [field for field in required_fields if field not in config]

    if missing_fields:
        raise ValidationError(f"Missing required fields: {missing_fields}")


def validate_trajectory_data(
    q: np.ndarray,
    qd: Optional[np.ndarray] = None,
    qdd: Optional[np.ndarray] = None,
    tau: Optional[np.ndarray] = None,
) -> None:
    """
    Validate trajectory data arrays.

    Args:
        q: Joint positions
        qd: Joint velocities (optional)
        qdd: Joint accelerations (optional)
        tau: Joint torques (optional)

    Raises:
        ValidationError: If data is invalid
    """
    if not isinstance(q, np.ndarray):
        raise ValidationError("Joint positions must be numpy array")

    if q.ndim != 2:
        raise ValidationError("Joint positions must be 2D array (n_samples, n_joints)")

    n_samples, n_joints = q.shape

    # Validate other arrays if provided
    arrays_to_check = [("velocities", qd), ("accelerations", qdd), ("torques", tau)]

    for name, array in arrays_to_check:
        if array is not None:
            if not isinstance(array, np.ndarray):
                raise ValidationError(f"{name} must be numpy array")

            if array.shape != (n_samples, n_joints):
                raise ValidationError(
                    f"{name} shape {array.shape} doesn't match "
                    f"positions shape {(n_samples, n_joints)}"
                )

            if np.any(np.isnan(array)):
                raise ValidationError(f"{name} contains NaN values")

            if np.any(np.isinf(array)):
                raise ValidationError(f"{name} contains infinite values")


def validate_numeric_range(
    value: Union[float, int, np.ndarray],
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "value",
) -> None:
    """
    Validate that numeric value(s) are within specified range.

    Args:
        value: Value or array to validate
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        name: Name of the value for error messages

    Raises:
        ValidationError: If value is out of range
    """
    if isinstance(value, np.ndarray):
        if min_val is not None and np.any(value < min_val):
            raise ValidationError(f"{name} contains values below {min_val}")

        if max_val is not None and np.any(value > max_val):
            raise ValidationError(f"{name} contains values above {max_val}")
    else:
        if min_val is not None and value < min_val:
            raise ValidationError(f"{name} {value} is below minimum {min_val}")

        if max_val is not None and value > max_val:
            raise ValidationError(f"{name} {value} is above maximum {max_val}")


# Decorator Functions
def validate_robot_initialization(func: F) -> F:
    """
    Decorator to validate robot initialization.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with robot validation
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            raise RobotInitializationError(f"Robot initialization failed: {e}") from e

    return wrapper


def validate_input_data(func: F) -> F:
    """
    Decorator to validate input data for processing functions.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with input validation
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Basic validation - can be extended based on function signature
            for arg in args:
                if isinstance(arg, np.ndarray):
                    if np.any(np.isnan(arg)):
                        raise ValidationError("Input data contains NaN values")
                    if np.any(np.isinf(arg)):
                        raise ValidationError("Input data contains infinite values")

            return func(*args, **kwargs)
        except ValidationError:
            raise
        except Exception as e:
            raise DataProcessingError(f"Data processing failed: {e}") from e

    return wrapper


def handle_calibration_errors(func: F) -> F:
    """
    Decorator to handle calibration-specific errors.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with calibration error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, np.linalg.LinAlgError) as e:
            raise CalibrationError(f"Calibration failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in calibration: {e}")
            raise CalibrationError(f"Unexpected calibration error: {e}") from e

    return wrapper


def handle_identification_errors(func: F) -> F:
    """
    Decorator to handle identification-specific errors.

    Args:
        func: Function to decorate

    Returns:
        Decorated function with identification error handling
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except (ValueError, np.linalg.LinAlgError) as e:
            raise IdentificationError(f"Identification failed: {e}") from e
        except Exception as e:
            logger.error(f"Unexpected error in identification: {e}")
            raise IdentificationError(f"Unexpected identification error: {e}") from e

    return wrapper


def safe_execute(func: F, *args, **kwargs) -> tuple:
    """
    Safely execute a function and return (success, result_or_error).

    Args:
        func: Function to execute
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function

    Returns:
        Tuple of (success_flag, result_or_exception)
    """
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        logger.error(f"Function {func.__name__} failed: {e}")
        return False, e


# Logging Utilities
def setup_example_logging(
    log_level: str = "INFO", log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup logging for FIGAROH examples.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("figaroh_examples")
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Context Managers
class ErrorContext:
    """Context manager for structured error handling."""

    def __init__(self, operation_name: str, raise_on_error: bool = True):
        self.operation_name = operation_name
        self.raise_on_error = raise_on_error
        self.error = None

    def __enter__(self):
        logger.info(f"Starting {self.operation_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            error_msg = f"{self.operation_name} failed: {exc_val}"
            logger.error(error_msg)
            self.error = exc_val

            if self.raise_on_error:
                return False  # Re-raise the exception
            else:
                return True  # Suppress the exception
        else:
            logger.info(f"{self.operation_name} completed successfully")
            return True
