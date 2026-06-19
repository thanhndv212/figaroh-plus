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

"""Robot loading utilities with support for multiple backends."""

import os
from typing import Optional, Union, Any
import pinocchio as pin
from pinocchio.robot_wrapper import RobotWrapper

# Import Robot class from the same package
from .robot import Robot


def load_robot(
    robot_urdf: str,
    package_dirs: Optional[str] = None,
    isFext: bool = False,
    load_by_urdf: bool = True,
    robot_pkg: Optional[str] = None,
    loader: str = "figaroh",
    **kwargs,
) -> Union[Robot, RobotWrapper, Any]:
    """Load robot model from various sources with multiple loader options.

    Args:
        robot_urdf: Path to URDF file or robot name for robot_description
        package_dirs: Package directories for mesh files
        isFext: Whether to add floating base joint
        load_by_urdf: Whether to load from URDF file (vs ROS param server)
        robot_pkg: Name of robot package for path resolution
        loader: Loader type - "figaroh", "robot_description", "yourdfpy"
        **kwargs: Additional arguments passed to the specific loader

    Returns:
        Robot object based on loader type:
        - "figaroh": Robot class instance (default, backward compatible)
        - "robot_description": RobotWrapper from pinocchio
        - "yourdfpy": URDF object from yourdfpy (suitable for viser)

    Raises:
        FileNotFoundError: If URDF file not found
        ImportError: If required packages not available
        ValueError: If the specified loader is not supported

    Note: For loading by URDF, robot_urdf and package_dirs can be different.
          1/ If package_dirs is not provided directly, robot_pkg is used to
          look up the package directory.
          2/ If no mesh files, package_dirs and robot_pkg are not used.
          3/ If load_by_urdf is False, the robot is loaded from the ROS
          parameter server.
          4/ For robot_description loader, robot_urdf should be the robot name.
          5/ For yourdfpy loader, returns URDF object suitable for viser visualization.
    """
    # Handle different loaders
    if loader == "robot_description":
        return _load_robot_description(robot_urdf, isFext, **kwargs)
    elif loader == "yourdfpy":
        return _load_yourdfpy(robot_urdf, package_dirs, robot_pkg, **kwargs)
    elif loader == "figaroh":
        # Original figaroh implementation (backward compatible)
        return _load_figaroh_original(
            robot_urdf, package_dirs, isFext, load_by_urdf, robot_pkg
        )
    else:
        raise ValueError(
            f"Unsupported loader: {loader}. Supported loaders: figaroh, robot_description, yourdfpy"
        )


def _check_package_available(package_name: str) -> bool:
    """Check if a package is available for import."""
    try:
        __import__(package_name)
        return True
    except ImportError:
        return False


def _load_robot_description(
    robot_name: str, isFext: bool = False, **kwargs
) -> RobotWrapper:
    """Load robot from robot_descriptions package."""
    if not _check_package_available("robot_descriptions"):
        raise ImportError("robot_descriptions package is not available")

    try:
        from robot_descriptions.loaders.pinocchio import load_robot_description

        # Prepare kwargs for load_robot_description
        loader_kwargs = kwargs.copy()

        # Handle free-flyer joint if requested
        if isFext:
            loader_kwargs["root_joint"] = pin.JointModelFreeFlyer()

        # Try to load robot description, with fallback to "_description" suffix
        try:
            robot = load_robot_description(robot_name, **loader_kwargs)
        except ModuleNotFoundError:
            try:
                robot = load_robot_description(
                    f"{robot_name}_description", **loader_kwargs
                )
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    f"Robot description '{robot_name}' not found. "
                    f"Try specifying the full name like '{robot_name}_description' "
                    f"or check available robot descriptions."
                )

        return robot

    except ImportError as e:
        raise ImportError(
            f"Required packages not available for robot_description loader: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load robot description '{robot_name}': {e}")


def _load_yourdfpy(
    robot_urdf: str, package_dirs: Optional[str], robot_pkg: Optional[str], **kwargs
) -> Any:
    """Load robot using yourdfpy (suitable for viser)."""
    if not _check_package_available("yourdfpy"):
        raise ImportError("yourdfpy package is not available")

    try:
        import yourdfpy

        # Prepare package_dirs
        package_dirs = _prepare_package_dirs(robot_urdf, package_dirs, robot_pkg)

        # Load with yourdfpy
        robot = yourdfpy.URDF.load(
            robot_urdf,
            mesh_dir=package_dirs,
            build_collision_scene_graph=kwargs.get("build_collision_scene_graph", True),
            load_meshes=kwargs.get("load_meshes", True),
            build_scene_graph=kwargs.get("build_scene_graph", True),
            load_collision_meshes=kwargs.get("load_collision_meshes", False),
            force_collision_mesh=kwargs.get("force_collision_mesh", False),
            force_mesh=kwargs.get("force_mesh", False),
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "build_collision_scene_graph",
                    "load_meshes",
                    "build_scene_graph",
                    "load_collision_meshes",
                    "force_collision_mesh",
                    "force_mesh",
                ]
            },
        )

        return robot

    except ImportError as e:
        raise ImportError(f"yourdfpy package not available: {e}")


def _load_figaroh_original(
    robot_urdf: str,
    package_dirs: Optional[str],
    isFext: bool,
    load_by_urdf: bool,
    robot_pkg: Optional[str],
) -> Union[Robot, RobotWrapper]:
    """Original figaroh implementation for backward compatibility."""
    if load_by_urdf:
        package_dirs = _prepare_package_dirs(robot_urdf, package_dirs, robot_pkg)
        _validate_urdf_exists(robot_urdf)
        return Robot(robot_urdf, package_dirs=package_dirs, isFext=isFext)
    else:
        return _load_from_ros_param(isFext)


def _prepare_package_dirs(
    robot_urdf: str, package_dirs: Optional[str], robot_pkg: Optional[str]
) -> str:
    """Prepare package directories for robot loading."""
    if package_dirs is None:
        if robot_pkg is not None:
            """Resolve package directory from installed ROS robot package."""
            try:
                import rospkg

                package_dirs = rospkg.RosPack().get_path(robot_pkg)
            except (ImportError, Exception):
                # Resolve relative path to models directory
                package_dirs = _get_models_directory()
        else:
            package_dirs = os.path.dirname(os.path.abspath(robot_urdf))
    elif package_dirs == "models":
        package_dirs = _get_models_directory()

    return package_dirs


def _get_models_directory() -> str:
    """Get models directory from figaroh-examples package.

    This function tries to locate the models directory from the
    figaroh-examples package, which should be installed separately
    from the figaroh package.

    Returns:
        str: Path to the models directory

    Raises:
        ImportError: If figaroh-examples package is not found or
                    models directory doesn't exist
    """
    try:
        # Method 1: Try to import figaroh_examples as a package
        import figaroh_examples

        examples_root = os.path.dirname(figaroh_examples.__file__)
        models_dir = os.path.join(examples_root, "models")

        if os.path.exists(models_dir):
            return models_dir
        else:
            raise FileNotFoundError(f"Models directory not found at: {models_dir}")

    except ImportError:
        try:
            # Method 2: Try to find figaroh-examples using importlib
            import importlib.util
            import sys

            # Look for figaroh-examples in installed packages
            for path in sys.path:
                if os.path.isdir(path):
                    # Check for figaroh-examples or figaroh_examples
                    for pkg_name in ["figaroh-examples", "figaroh_examples"]:
                        pkg_path = os.path.join(path, pkg_name)
                        if os.path.isdir(pkg_path):
                            models_dir = os.path.join(pkg_path, "models")
                            if os.path.exists(models_dir):
                                return models_dir

                        # Also check for .egg-info directories
                        egg_pattern = f"{pkg_name.replace('-', '_')}*.egg-info"
                        import glob

                        egg_dirs = glob.glob(os.path.join(path, egg_pattern))
                        if egg_dirs:
                            # Package is installed, try to find actual location
                            try:
                                spec = importlib.util.find_spec("figaroh_examples")
                                if spec and spec.origin:
                                    pkg_dir = os.path.dirname(spec.origin)
                                    models_dir = os.path.join(pkg_dir, "models")
                                    if os.path.exists(models_dir):
                                        return models_dir
                            except (ImportError, AttributeError):
                                continue

            # Method 3: Try importlib.metadata (Python 3.8+) or pkg_resources
            try:
                # Try modern importlib.metadata first (Python 3.8+)
                try:
                    from importlib import metadata

                    dist = metadata.distribution("figaroh-examples")
                    # Get files in the distribution
                    if dist.files:
                        for file in dist.files:
                            if "models" in str(file):
                                models_dir = os.path.join(
                                    dist.locate_file("."), "models"
                                )
                                if os.path.exists(models_dir):
                                    return models_dir
                except (ImportError, metadata.PackageNotFoundError):
                    pass

                # Fallback to pkg_resources for older Python versions
                try:
                    import pkg_resources

                    try:
                        dist = pkg_resources.get_distribution("figaroh-examples")
                        examples_root = dist.location
                        # Handle different installation layouts
                        possible_paths = [
                            os.path.join(examples_root, "models"),
                            os.path.join(examples_root, "figaroh_examples", "models"),
                            os.path.join(examples_root, "figaroh-examples", "models"),
                        ]
                        for models_dir in possible_paths:
                            if os.path.exists(models_dir):
                                return models_dir
                    except pkg_resources.DistributionNotFound:
                        pass
                except ImportError:
                    pass

            except Exception:
                pass

        except ImportError:
            pass

        # Method 4: Fallback to development/local locations
        possible_locations = [
            # Check if we're in a development environment nearby
            # From figaroh/src/figaroh/tools -> figaroh-examples
            os.path.join(
                os.path.dirname(
                    os.path.dirname(
                        os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    )
                ),
                "figaroh-examples",
            ),
            # Check user's home directory
            os.path.expanduser("~/figaroh-examples"),
            # Check current working directory
            os.path.join(os.getcwd(), "figaroh-examples"),
        ]

        for location in possible_locations:
            models_dir = os.path.join(location, "models")
            if os.path.exists(models_dir):
                return models_dir

        # If none found, raise an informative error
        raise ImportError(
            "figaroh-examples package not found. Please install "
            "figaroh-examples or ensure the models directory is available. "
            "You can install it with: pip install figaroh-examples"
        )


def _validate_urdf_exists(robot_urdf: str) -> None:
    """Validate that URDF file exists."""
    if not os.path.exists(robot_urdf):
        raise FileNotFoundError(f"URDF file not found: {robot_urdf}")


def _load_from_ros_param(isFext: bool) -> RobotWrapper:
    """Load robot from ROS parameter server."""
    if not _check_package_available("rospy"):
        raise ImportError(
            "rospy package is not available for ROS parameter server loading"
        )

    try:
        import rospy
        from pinocchio.robot_wrapper import RobotWrapper
    except ImportError as e:
        raise ImportError(f"ROS packages not available: {e}")

    robot_xml = rospy.get_param("robot_description")
    root_joint = pin.JointModelFreeFlyer() if isFext else None
    model = pin.buildModelFromXML(robot_xml, root_joint=root_joint)

    return RobotWrapper(model)


def get_available_loaders() -> dict:
    """Get information about available robot loaders.

    Returns:
        dict: Information about each loader and its availability
    """
    loaders = {
        "figaroh": {
            "description": "Original figaroh Robot class",
            "available": True,
            "returns": "Robot instance",
            "features": ["URDF loading", "ROS param server", "Free-flyer support"],
        },
        "robot_description": {
            "description": "Load from robot_descriptions package",
            "available": _check_package_available("robot_descriptions"),
            "returns": "RobotWrapper instance",
            "features": ["Pre-defined robot models", "Easy robot switching"],
        },
        "yourdfpy": {
            "description": "Load with yourdfpy (for visualization)",
            "available": _check_package_available("yourdfpy"),
            "returns": "URDF object",
            "features": ["Visualization support", "Mesh loading", "Scene graphs"],
        },
    }

    return loaders


def list_available_robots(loader: str = "robot_description") -> list:
    """List available robot descriptions for a given loader.

    Args:
        loader: Loader type to check for available robots

    Returns:
        list: Available robot names
    """
    if loader == "robot_description":
        if not _check_package_available("robot_descriptions"):
            return []

        try:
            import robot_descriptions

            # Get all available robot descriptions with URDF format
            available_robots = []

            # Check if DESCRIPTIONS attribute exists
            if hasattr(robot_descriptions, "DESCRIPTIONS"):
                descriptions = robot_descriptions.DESCRIPTIONS
                for robot_name, robot_info in descriptions.items():
                    # Check if the robot has URDF format available
                    if hasattr(robot_info, "has_urdf") and robot_info.has_urdf:
                        available_robots.append(robot_name)
                    # Fallback: check if robot_info has URDF_PATH attribute
                    elif hasattr(robot_info, "URDF_PATH"):
                        available_robots.append(robot_name)
            else:
                # Fallback to original method if DESCRIPTIONS not available
                for attr_name in dir(robot_descriptions):
                    if attr_name.startswith("_"):
                        continue
                    attr_obj = getattr(robot_descriptions, attr_name)
                    if hasattr(attr_obj, "URDF_PATH"):
                        available_robots.append(attr_name)

            return sorted(available_robots)
        except Exception:
            return []

    return []
