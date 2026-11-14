"""Packaging metadata stays single-sourced and declares what the code names."""

import re
from pathlib import Path

import figaroh

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = (ROOT / "pyproject.toml").read_text()


def _project_table():
    return PYPROJECT.split("[project]", 1)[1].split("\n[", 1)[0]


def test_version_has_one_source():
    project = _project_table()
    assert not re.search(r"^version\s*=", project, flags=re.M)
    assert re.search(r"^dynamic\s*=\s*\[\"version\"\]", project, flags=re.M)
    hatch = PYPROJECT.split("[tool.hatch.version]", 1)[1].split("\n[", 1)[0]
    assert 'path = "src/figaroh/__init__.py"' in hatch
    assert re.fullmatch(r"\d+\.\d+\.\d+([.-]?\w+)?", figaroh.__version__)


def test_default_sdp_solver_is_declared():
    # "cvxopt" is the default solver name across identification/ and optimal/.
    assert '"cvxopt"' in _project_table()


def test_manifest_in_is_gone():
    # hatchling does not read it; a copy would only mislead.
    assert not (ROOT / "MANIFEST.in").exists()
