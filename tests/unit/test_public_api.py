"""Package public surface: eager re-exports declared in __all__ (#82).

The package ``__init__`` files import their submodules eagerly so that
``import figaroh`` exposes ``figaroh.tools.robot`` and friends, which the
examples rely on. These tests fail if a re-export is dropped.
"""

import subprocess
import sys
import types

import pytest

import figaroh
import figaroh.tools
import figaroh.visualisation

EXPECTED = {
    figaroh: {
        "tools",
        "calibration",
        "identification",
        "measurements",
        "utils",
        "visualisation",
        "optimal",
    },
    figaroh.tools: {
        "robot",
        "randomdata",
        "regressor",
        "qrdecomposition",
        "robotvisualization",
        "robotcollisions",
        "robotipopt",
        "urdf_exporter",
        "geometric_calibration_export",
        "export_validation",
        "report",
        "identification_report",
        "compare_report",
    },
    figaroh.visualisation: {"colors", "MeshcatVisualizer"},
}


@pytest.mark.parametrize("package", list(EXPECTED), ids=lambda m: m.__name__)
def test_all_declares_the_public_surface(package):
    assert set(package.__all__) == EXPECTED[package]
    for name in package.__all__:
        assert hasattr(package, name), f"{package.__name__}.{name} missing"


def test_subpackages_are_modules():
    for name in EXPECTED[figaroh]:
        assert isinstance(getattr(figaroh, name), types.ModuleType)


def test_bare_import_exposes_nested_modules():
    """Attribute access works in a fresh interpreter after only ``import figaroh``."""
    code = (
        "import figaroh; "
        "figaroh.tools.robot.Robot; "
        "figaroh.tools.regressor.build_regressor_basic; "
        "figaroh.visualisation.MeshcatVisualizer"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
