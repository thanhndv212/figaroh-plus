"""Package public surface: eager re-exports declared in __all__ (#82, #184).

The last tests tie the ``__all__`` lists to docs/source/api/stability.md.

The package ``__init__`` files import their submodules eagerly so that
``import figaroh`` exposes ``figaroh.tools.robot`` and friends, which the
examples rely on. These tests fail if a re-export is dropped.
"""

import importlib
import re
import subprocess
import sys
import types
from pathlib import Path

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
        "data",
        "backends",
        "integration",
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
        "run_archive",
        "stages",
        "solver",
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


PAGE = Path(__file__).resolve().parents[2] / "docs/source/api/stability.md"
ENTRY = re.compile(r"^- `(figaroh[\w.]*)`: (.+)$")


def _section(title):
    text = PAGE.read_text()
    match = re.search(rf"^## {re.escape(title)}\n(.*?)(?=^## |\Z)", text, re.S | re.M)
    assert match, f"section {title!r} missing from {PAGE.name}"
    return match.group(1)


def _entries(title):
    out = {}
    for line in _section(title).splitlines():
        m = ENTRY.match(line)
        if m:
            out[m.group(1)] = re.findall(r"`([^`]+)`", m.group(2))
    return out


def _all_packages_and_modules():
    """Every package in figaroh.__all__ plus the modules they list."""
    for name in figaroh.__all__:
        pkg = getattr(figaroh, name)
        yield f"figaroh.{name}", pkg
        for sub in getattr(pkg, "__all__", []):
            obj = getattr(pkg, sub)
            if isinstance(obj, types.ModuleType):
                yield f"figaroh.{name}.{sub}", obj


@pytest.mark.parametrize(
    "dotted,module",
    list(_all_packages_and_modules()),
    ids=lambda v: v if isinstance(v, str) else "",
)
def test_every_all_name_imports(dotted, module):
    """Every name in an ``__all__`` resolves, for packages and listed modules."""
    names = getattr(module, "__all__", None)
    if dotted == "figaroh.measurements" or names is not None:
        assert names, f"{dotted} has an empty __all__"
    assert len(set(names)) == len(names), f"{dotted}.__all__ has duplicates"
    for name in names:
        assert hasattr(module, name), f"{dotted}.{name} listed but missing"


def test_page_top_level_table_matches_top_level_all():
    rows = re.findall(r"^\| `figaroh\.(\w+)` \|", _section("Top-level packages"), re.M)
    assert set(rows) == set(figaroh.__all__)
    assert len(rows) == len(set(rows))


@pytest.mark.parametrize("title", ["Supported entry points", "Experimental areas"])
def test_page_lists_match_module_all(title):
    """Each listed package or module names exactly its ``__all__``."""
    entries = _entries(title)
    assert entries, f"no entries parsed from {title!r}"
    for dotted, names in entries.items():
        module = importlib.import_module(dotted)
        assert sorted(names) == sorted(module.__all__), dotted


def test_page_covers_every_listed_package_module_and_nothing_internal():
    listed = set(_entries("Supported entry points")) | set(
        _entries("Experimental areas")
    )
    for dotted, module in _all_packages_and_modules():
        assert dotted in listed, f"{dotted} is in an __all__ but not on the page"
    for dotted in listed:
        assert not any(part.startswith("_") for part in dotted.split("."))


def test_private_modules_are_not_exported():
    for dotted, module in _all_packages_and_modules():
        for name in module.__all__:
            assert not name.startswith("_"), f"{dotted}.__all__ exports {name}"
