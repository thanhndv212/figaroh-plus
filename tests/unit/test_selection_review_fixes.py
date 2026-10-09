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


# -- export_urdf: only identified joints are written --

from pathlib import Path  # noqa: E402

from figaroh.identification.selection import SelectedEstimate  # noqa: E402

PENDULUM = str(Path(__file__).resolve().parent.parent / "fixtures" / "pendulum.urdf")


def _nominal_selection(ident):
    values = np.array(list(ident.standard_parameter.values()), dtype=float)
    ident.selected = SelectedEstimate(
        stage="physical_fit",
        requested="physical_fit",
        status="accepted",
        reason="test",
        space="standard",
        names=list(ident.standard_parameter.keys()),
        values=values,
    )


def test_export_urdf_with_a_floating_base_model(tmp_path):
    m = pin.buildModelFromUrdf(PENDULUM, pin.JointModelFreeFlyer())
    ident = _Ident(m)
    ident.initialize_standard_parameters()
    ident._idx_eliminated = []
    _nominal_selection(ident)
    out = ident.export_urdf(PENDULUM, output_path=str(tmp_path / "o.urdf"))
    assert pin.buildModelFromUrdf(out).nv == 2


MERGED = """<?xml version="1.0"?>
<robot name="merged">
  <link name="world"/>
  <joint name="joint1" type="revolute">
    <parent link="world"/><child link="link1"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="10" velocity="10"/>
  </joint>
  <link name="link1">
    <inertial><mass value="1"/><origin xyz="0 0 0.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="joint2" type="revolute">
    <origin xyz="0 0 1"/>
    <parent link="link1"/><child link="link2"/><axis xyz="0 0 1"/>
    <limit lower="-3" upper="3" effort="10" velocity="10"/>
  </joint>
  <link name="link2">
    <inertial><mass value="1"/><origin xyz="0 0 0.5"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.01"/></inertial>
  </link>
  <joint name="fix" type="fixed">
    <origin xyz="0 0 0.5"/><parent link="link2"/><child link="tool"/>
  </joint>
  <link name="tool">
    <inertial><mass value="0.5"/><origin xyz="0 0 0.1"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/></inertial>
  </link>
</robot>
"""


def test_non_identified_merged_body_joint_does_not_block_export(tmp_path):
    urdf = tmp_path / "merged.urdf"
    urdf.write_text(MERGED)
    m = pin.buildModelFromUrdf(str(urdf))
    ident = _Ident(m)
    ident.initialize_standard_parameters()
    names = list(ident.standard_parameter.keys())
    ident._idx_eliminated = [i for i, n in enumerate(names) if n.endswith("_joint2")]
    _nominal_selection(ident)
    out = ident.export_urdf(str(urdf), output_path=str(tmp_path / "o.urdf"))
    assert pin.buildModelFromUrdf(out).nv == 2


# -- solve_with_custom_solver refreshes the rows the selection uses --


def test_custom_solver_after_solve_uses_the_new_rows(model, traj):
    pytest.importorskip("picos")
    ident = _Ident(model)
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.identif_config.update(select_stage="physical_fit")
    ident.solve(decimate=True, decimation_factor=4, plotting=False)
    decimated_rows = ident._solve_W.shape[0]
    ident.solve_with_custom_solver(decimate=False)
    assert ident._solve_W.shape[0] == len(ident.tau_noised) != decimated_rows
    assert ident._solve_tau.shape[0] == len(ident.tau_noised)
    assert ident.selected.accepted, ident.selected.reason
    assert ident._wls_row_weight is None and ident._recon_result is None


def test_custom_solver_without_a_prior_solve_does_not_fail(model, traj):
    pytest.importorskip("picos")
    ident = _Ident(model)
    ident.trajectory_to_return = traj
    ident.initialize()
    ident.identif_config.update(select_stage="physical_fit")
    ident.solve_with_custom_solver(decimate=False)
    assert ident.selected.accepted, ident.selected.reason
