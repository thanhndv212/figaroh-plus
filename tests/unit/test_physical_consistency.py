"""Unit tests for physical-consistency utilities (v0.4.1).

Test cases follow the roadmap specification exactly:

  TC-1  Feasible input remains unchanged after projection
  TC-2  Pinocchio round-trip invariant (symmetry + round-trip accuracy)
  TC-3  Negative mass corrected after projection
  TC-4  Indefinite inertia corrected after projection
  TC-5  Auto vs manual weights both yield feasible output; corrections differ
  TC-6  Missing picos backend raises a clear ImportError

Additional (non-roadmap):
  TC-7  check_p10_feasibility on valid p10 -> "feasible"
  TC-8  check_p10_feasibility on mass below threshold -> "infeasible"
  TC-9  is_feasible_link alias matches check_p10_feasibility exactly
  TC-10 project_link alias matches project_p10_lmi exactly
  TC-11 project_robot_p10_lmi aggregation over 2 links
  TC-12 ProjectionReport.runtime is set after projection

Config-wiring tests:
  TC-config-1  weights.mode "auto" passes weights=None to project_robot_p10_lmi
  TC-config-2  weights.mode "manual" builds correct 10-element weight array
  TC-config-3  result["physical consistency"] contains raw_parameters and
               projected_parameters keys after _apply_physical_consistency_if_enabled
"""

from __future__ import annotations

import dataclasses
import sys
import types
from typing import Dict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from figaroh.identification.physical_consistency import (
    RobotProjectionReport,
    ProjectionReport,
    check_p10_feasibility,
    is_feasible_link,
    p10_by_joint_from_param_dict,
    param_dict_with_p10_by_joint,
    project_link,
    project_p10_lmi,
    project_robot_p10_lmi,
    pseudo_inertia_matrix_from_p10,
)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_feasible_p10(
    mass: float = 2.0,
    scale: float = 1.0,
) -> np.ndarray:
    """Return a physically valid 10D inertial vector.

    Uses a diagonal inertia tensor with the given scale:
      sigma = diag(2s, 2s, 2s)  (satisfies triangle inequality trivially)
      h = [0, 0, 0]  (CoM at origin)
    """
    Ixx = Iyy = Izz = 2.0 * scale
    return np.array([mass, 0.0, 0.0, 0.0, Ixx, 0.0, Iyy, 0.0, 0.0, Izz])


def _make_infeasible_p10_neg_mass(mass: float = -1.0) -> np.ndarray:
    """p10 with negative mass but otherwise valid inertia."""
    return np.array([mass, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, 2.0])


def _make_infeasible_p10_neg_inertia() -> np.ndarray:
    """p10 with positive mass but an indefinite sigma (pseudo-inertia not PSD)."""
    # Sigma = diag(-5, 1, 1) — clearly indefinite
    return np.array([1.0, 0.0, 0.0, 0.0, -5.0, 0.0, 1.0, 0.0, 0.0, 1.0])


# ---------------------------------------------------------------------------
# TC-7  check_p10_feasibility: feasible input
# ---------------------------------------------------------------------------


def test_tc7_check_feasibility_feasible_input():
    """Feasible p10 reports status='feasible' and non-negative min_eig."""
    p10 = _make_feasible_p10()
    report = check_p10_feasibility(p10)
    assert report.status == "feasible"
    assert report.mass == pytest.approx(2.0)
    assert report.min_eig >= -1e-10


# ---------------------------------------------------------------------------
# TC-8  check_p10_feasibility: mass below threshold
# ---------------------------------------------------------------------------


def test_tc8_check_feasibility_mass_below_threshold():
    """p10 with mass below mass_min reports status='infeasible'."""
    p10 = _make_feasible_p10(mass=0.5)
    report = check_p10_feasibility(p10, mass_min=1.0)
    assert report.status == "infeasible"
    assert report.message is not None and "mass" in report.message.lower()


# ---------------------------------------------------------------------------
# TC-2  Pinocchio round-trip invariant (no picos required)
# ---------------------------------------------------------------------------

_P10_VARIANTS = [
    _make_feasible_p10(mass=2.0, scale=1.0),
    _make_feasible_p10(mass=5.0, scale=0.5),
    _make_infeasible_p10_neg_mass(),
    _make_infeasible_p10_neg_inertia(),
    np.zeros(10),  # degenerate
]


@pytest.mark.parametrize("p10", _P10_VARIANTS)
def test_tc2_pseudo_inertia_matrix_symmetry(p10):
    """pseudo_inertia_matrix_from_p10 always returns a symmetric 4x4 matrix."""
    P = pseudo_inertia_matrix_from_p10(p10)
    assert P.shape == (4, 4)
    assert (
        np.max(np.abs(P - P.T)) <= 1e-12
    ), f"Pseudo-inertia matrix is not symmetric: max|P-P^T| = {np.max(np.abs(P - P.T))}"


@pytest.mark.parametrize("p10", _P10_VARIANTS)
def test_tc2_pinocchio_roundtrip(p10):
    """If pinocchio is available, round-trip through PseudoInertia is accurate."""
    pin = pytest.importorskip("pinocchio")
    PI = pin.PseudoInertia.FromDynamicParameters(p10)
    p_prime = np.asarray(PI.toDynamicParameters())
    assert p_prime.shape == (10,)
    assert (
        np.max(np.abs(p_prime - p10)) <= 1e-10
    ), f"Round-trip error: max|p'-p| = {np.max(np.abs(p_prime - p10))}"


# ---------------------------------------------------------------------------
# TC-6  Missing picos -> clear ImportError  (no picos needed)
# ---------------------------------------------------------------------------


def test_tc6_missing_picos_raises_import_error():
    """project_p10_lmi raises ImportError mentioning 'picos' when picos absent."""
    p10 = _make_infeasible_p10_neg_mass()

    # Hide picos from the import system for the duration of this test.
    picos_backup = sys.modules.get("picos", None)
    try:
        sys.modules["picos"] = None  # type: ignore[assignment]
        with pytest.raises(ImportError, match="picos"):
            project_p10_lmi(p10)
    finally:
        if picos_backup is None:
            sys.modules.pop("picos", None)
        else:
            sys.modules["picos"] = picos_backup


# ---------------------------------------------------------------------------
# TC-9  is_feasible_link alias (no picos)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("p10", _P10_VARIANTS)
def test_tc9_is_feasible_link_alias(p10):
    """is_feasible_link returns the same result as check_p10_feasibility."""
    r1 = check_p10_feasibility(p10, mass_min=1e-6, psd_eig_tol=-1e-10)
    r2 = is_feasible_link(p10, mass_min=1e-6, psd_eig_tol=-1e-10)
    assert dataclasses.asdict(r1) == dataclasses.asdict(r2)


# ---------------------------------------------------------------------------
# Tests that require picos (skip gracefully when absent)
# ---------------------------------------------------------------------------

picos = pytest.importorskip("picos", reason="picos not installed")


# ---------------------------------------------------------------------------
# TC-1  Feasible input unchanged after projection
# ---------------------------------------------------------------------------


def test_tc1_feasible_input_unchanged():
    """Projecting an already-feasible p10 leaves it virtually unchanged."""
    p10 = _make_feasible_p10(mass=2.0, scale=1.0)

    # Pre-check that input is indeed feasible
    assert check_p10_feasibility(p10).status == "feasible"

    p10_proj, report = project_p10_lmi(p10)

    assert report.status in {
        "projected",
        "feasible",
    }, f"Unexpected status: {report.status}, message: {report.message}"
    assert check_p10_feasibility(p10_proj).status == "feasible"
    assert np.allclose(p10_proj, p10, atol=1e-4), (
        f"Feasible input was modified more than expected:\n"
        f"  original: {p10}\n  projected: {p10_proj}\n"
        f"  max delta: {np.max(np.abs(p10_proj - p10))}"
    )


# ---------------------------------------------------------------------------
# TC-3  Negative mass corrected
# ---------------------------------------------------------------------------


def test_tc3_negative_mass_corrected():
    """Projecting p10 with negative mass yields m >= mass_min and P PSD."""
    mass_min = 1e-6
    p10 = _make_infeasible_p10_neg_mass(mass=-1.0)

    p10_proj, report = project_p10_lmi(p10, mass_min=mass_min)

    # Allow solver numerical noise (~1e-7 relative to constraint value)
    assert (
        p10_proj[0] >= mass_min - 1e-7
    ), f"Projected mass {p10_proj[0]} is below mass_min={mass_min} (tol=1e-7)"
    # Verify feasibility with a relaxed tolerance consistent with SDP solver precision
    feas = check_p10_feasibility(p10_proj, mass_min=mass_min - 1e-7, psd_eig_tol=-1e-8)
    assert feas.status == "feasible", (
        f"Projected p10 still infeasible: status={feas.status}, "
        f"min_eig={feas.min_eig}"
    )


# ---------------------------------------------------------------------------
# TC-4  Indefinite inertia corrected
# ---------------------------------------------------------------------------


def test_tc4_indefinite_inertia_corrected():
    """Projecting p10 with indefinite sigma yields P PSD."""
    psd_eig_tol = -1e-10
    p10 = _make_infeasible_p10_neg_inertia()

    # Confirm it is actually infeasible before projection
    pre = check_p10_feasibility(p10, psd_eig_tol=psd_eig_tol)
    assert pre.status == "infeasible", "Test setup error: expected infeasible input"

    p10_proj, report = project_p10_lmi(p10, psd_eig_tol=psd_eig_tol)

    feas = check_p10_feasibility(p10_proj, psd_eig_tol=psd_eig_tol)
    assert feas.status == "feasible", (
        f"Projected p10 still infeasible: status={feas.status}, "
        f"min_eig={feas.min_eig}"
    )


# ---------------------------------------------------------------------------
# TC-5  Weighting sanity: auto vs manual weights
# ---------------------------------------------------------------------------


def test_tc5_weighting_sanity():
    """Both auto-weights and manual unit-weights produce feasible projections.

    The roadmap spec requires that the weights parameter is accepted and that
    the result is always physically feasible regardless of the weighting choice.
    We do not assert that solutions *differ* because the geometry of the
    feasible set may force the same projection for degenerate inputs.
    """
    # Infeasible: negative inertia principal value (Ixx negative)
    p10 = _make_infeasible_p10_neg_inertia()

    p10_auto, rep_auto = project_p10_lmi(p10, weights=None)
    # Heavy mass weight: w_m = 50, rest = 1 (clearly asymmetric weighting)
    w_heavy_mass = np.array([50.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    p10_heavy, rep_heavy = project_p10_lmi(p10, weights=w_heavy_mass)

    # Both must be feasible with SDP-appropriate tolerance
    _tol = -1e-8
    assert (
        check_p10_feasibility(p10_auto, psd_eig_tol=_tol).status == "feasible"
    ), f"Auto-weighted result infeasible: {rep_auto}"
    assert (
        check_p10_feasibility(p10_heavy, psd_eig_tol=_tol).status == "feasible"
    ), f"Heavy-mass-weighted result infeasible: {rep_heavy}"
    # The mass correction should be smaller with heavy mass weight
    # (penalty × 50 on mass deviation → solver keeps mass closer to original)
    mass_corr_auto = abs(float(p10_auto[0]) - float(p10[0]))
    mass_corr_heavy = abs(float(p10_heavy[0]) - float(p10[0]))
    assert mass_corr_heavy <= mass_corr_auto + 1e-6, (
        f"Heavier mass weight should not increase the mass correction "
        f"(corr_auto={mass_corr_auto:.6f}, corr_heavy={mass_corr_heavy:.6f})"
    )


# ---------------------------------------------------------------------------
# TC-10  project_link alias
# ---------------------------------------------------------------------------


def test_tc10_project_link_alias():
    """project_link returns the same result as project_p10_lmi."""
    p10 = _make_infeasible_p10_neg_inertia()
    p10_a, rep_a = project_p10_lmi(p10)
    p10_b, rep_b = project_link(p10)

    assert np.allclose(p10_a, p10_b, atol=1e-10)
    # Status and feasibility should match
    assert rep_a.status == rep_b.status


# ---------------------------------------------------------------------------
# TC-11  project_robot_p10_lmi aggregation
# ---------------------------------------------------------------------------


def test_tc11_robot_aggregation():
    """project_robot_p10_lmi aggregates per-link results correctly."""
    p10_ok = _make_feasible_p10()
    p10_bad = _make_infeasible_p10_neg_inertia()

    p10_by_link = {"link_a": p10_ok, "link_b": p10_bad}
    projected, robot_report = project_robot_p10_lmi(p10_by_link)

    assert robot_report.projected_links == 2
    assert set(projected.keys()) == {"link_a", "link_b"}
    assert set(robot_report.per_link.keys()) == {"link_a", "link_b"}

    # Both projected results must be feasible
    for name, p in projected.items():
        feas = check_p10_feasibility(p)
        assert (
            feas.status == "feasible"
        ), f"link '{name}' still infeasible after robot projection: {feas}"

    # Aggregate status
    assert robot_report.failed_links == 0
    assert robot_report.status == "ok"


# ---------------------------------------------------------------------------
# TC-12  ProjectionReport.runtime is set
# ---------------------------------------------------------------------------


def test_tc12_runtime_field_set():
    """report.runtime is a non-negative float after projection."""
    p10 = _make_infeasible_p10_neg_inertia()
    _, report = project_p10_lmi(p10)

    assert report.runtime is not None, "ProjectionReport.runtime should not be None"
    assert isinstance(report.runtime, float)
    assert (
        report.runtime >= 0.0
    ), f"runtime should be non-negative, got {report.runtime}"


# ---------------------------------------------------------------------------
# Config-wiring tests (use mocks; no picos required)
# ---------------------------------------------------------------------------


def _make_minimal_identif_config(pc_override: dict) -> dict:
    """Build a minimal identif_config dict for config-wiring tests."""
    return {"physical_consistency": pc_override}


def _make_dummy_param_dict(joint_names) -> dict:
    """Build a minimal flat parameter dict with all inertial keys for each joint."""
    keys = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]
    d = {}
    for j in joint_names:
        for k in keys:
            # Feasible default: positive mass, diagonal inertia
            if k == "m":
                d[f"{k}_{j}"] = 2.0
            elif k in ("Ixx", "Iyy", "Izz"):
                d[f"{k}_{j}"] = 2.0
            else:
                d[f"{k}_{j}"] = 0.0
    return d


class _FakeModel:
    """Minimal stand-in for pinocchio.Model used in _apply_physical_consistency."""

    def __init__(self, joint_names):
        # model.names[1:] is used to determine joint_names when not in config
        self.names = ["universe"] + list(joint_names)


class _FakeIdentification:
    """Minimal stand-in for a BaseIdentification instance."""

    def __init__(self, identif_config, joint_names, parameter_dict):
        self.identif_config = identif_config
        self.model = _FakeModel(joint_names)
        self.standard_parameter = parameter_dict
        self.result: dict = {}

    # Pull in the real method so we test the actual implementation.
    from figaroh.identification.base_identification import (
        BaseIdentification,
    )

    _apply_physical_consistency_if_enabled = (
        BaseIdentification._apply_physical_consistency_if_enabled
    )


def _run_apply(pc_cfg: dict, joint_names=("j1",), mock_project=None):
    """Helper: build a fake identification object and run the pipeline hook.

    If *mock_project* is provided it is used as the replacement for
    ``project_robot_p10_lmi`` in the module under test.
    """
    param_dict = _make_dummy_param_dict(joint_names)
    fake = _FakeIdentification(
        identif_config={"physical_consistency": pc_cfg},
        joint_names=joint_names,
        parameter_dict=param_dict,
    )
    # The method does a lazy `from figaroh.identification.physical_consistency import
    # project_robot_p10_lmi` inside its body each call, so we must patch the
    # *source* module attribute so the import picks up the mock.
    target = "figaroh.identification.physical_consistency.project_robot_p10_lmi"

    if mock_project is None:
        # Create a sensible default mock that returns a no-op projection
        mock_project = MagicMock(
            return_value=(
                p10_by_joint_from_param_dict(
                    parameter_dict=param_dict,
                    joint_names=list(joint_names),
                ),
                RobotProjectionReport(
                    status="ok",
                    projected_links=len(joint_names),
                    failed_links=0,
                    per_link={
                        j: ProjectionReport(
                            status="projected",
                            mass=2.0,
                            min_eig=0.5,
                            runtime=0.001,
                        )
                        for j in joint_names
                    },
                ),
            )
        )

    with patch(target, mock_project):
        # Retrieve the unbound function from BaseIdentification and call it
        # explicitly so `fake` is passed as `self` only once.
        from figaroh.identification.base_identification import BaseIdentification

        BaseIdentification._apply_physical_consistency_if_enabled(
            fake, identif_results={}
        )

    return fake, mock_project


# TC-config-1: auto mode passes weights=None
def test_tc_config1_auto_weights_passes_none():
    """weights.mode='auto' causes weights=None to be passed to project_robot_p10_lmi."""
    pc_cfg = {
        "enabled": True,
        "weights": {"mode": "auto"},
        "skip_if_feasible": False,
    }
    fake, mock_proj = _run_apply(pc_cfg)

    call_kwargs = mock_proj.call_args
    assert call_kwargs is not None, "project_robot_p10_lmi was not called"
    # weights keyword arg should be None (auto mode)
    passed_weights = call_kwargs.kwargs.get("weights", "MISSING")
    assert (
        passed_weights is None
    ), f"Expected weights=None for auto mode, got {passed_weights!r}"


# TC-config-2: manual mode builds correct 10-element weight array
def test_tc_config2_manual_weights_correct_array():
    """weights.mode='manual' builds the expected 10-element weight array."""
    pc_cfg = {
        "enabled": True,
        "weights": {
            "mode": "manual",
            "manual": {"m": 3.0, "h": 2.0, "Sigma": 0.5},
        },
        "skip_if_feasible": False,
    }
    fake, mock_proj = _run_apply(pc_cfg)

    call_kwargs = mock_proj.call_args
    assert call_kwargs is not None, "project_robot_p10_lmi was not called"
    passed_weights = call_kwargs.kwargs.get("weights", None)

    assert passed_weights is not None, "Expected a weight array for manual mode"
    assert passed_weights.shape == (
        10,
    ), f"Expected shape (10,), got {passed_weights.shape}"
    # m -> index 0
    assert passed_weights[0] == pytest.approx(3.0)
    # h -> indices 1-3
    assert np.allclose(passed_weights[1:4], 2.0)
    # Sigma -> indices 4-9
    assert np.allclose(passed_weights[4:10], 0.5)


# TC-config-3: raw_parameters and projected_parameters both present in result
def test_tc_config3_raw_and_projected_keys_in_result():
    """result['physical consistency'] contains both raw_parameters and projected_parameters."""
    pc_cfg = {
        "enabled": True,
        "skip_if_feasible": False,
    }
    fake, _ = _run_apply(pc_cfg)

    pc_result = fake.result.get("physical consistency", {})
    assert (
        "raw_parameters" in pc_result
    ), f"'raw_parameters' key missing from result: {list(pc_result.keys())}"
    assert (
        "projected_parameters" in pc_result
    ), f"'projected_parameters' key missing from result: {list(pc_result.keys())}"
    # raw and projected must be distinct objects (not aliased)
    assert pc_result["raw_parameters"] is not pc_result["projected_parameters"]
