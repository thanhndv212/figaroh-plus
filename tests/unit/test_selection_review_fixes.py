"""Regression tests for the #61 code-review findings."""

import os
import sys

import numpy as np
import pytest

try:
    import pinocchio as pin
except ImportError:
    pytest.skip("Pinocchio not available", allow_module_level=True)

sys.path.insert(0, os.path.dirname(__file__))

from test_data_contract_wiring import _Ident, _trajectory  # noqa: E402


@pytest.fixture(scope="module")
def model():
    return pin.buildSampleModelManipulator()


@pytest.fixture(scope="module")
def traj(model):
    return _trajectory(model)


def _run(model, traj, **cfg):
    ident = _Ident(model)
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.identif_config.update(cfg)
    ident.solve(decimate=False, plotting=False)
    return ident


def test_max_seconds_is_mapped_to_picos_timelimit(model, traj):
    pytest.importorskip("picos")
    ident = _run(
        model,
        traj,
        select_stage="physical_fit",
        physical_fit={"max_seconds": 60},
    )
    assert ident.selected.accepted, ident.selected.reason
