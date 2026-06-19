"""Unit tests for figaroh.identification.cad_constraints (v0.4.3)."""

import pytest
import numpy as np

from figaroh.identification.cad_constraints import (
    CADConstraints,
    _INERTIAL_KEYS,
    add_mass_bounds,
    add_com_bounds,
    add_symmetry_constraints,
    build_cad_constraints_from_config,
    apply_cad_constraints_to_problem,
)

# ---------------------------------------------------------------------------
# CADConstraints data container
# ---------------------------------------------------------------------------


def test_cad_constraints_is_empty_default():
    cad = CADConstraints()
    assert cad.is_empty()


def test_add_mass_bounds_basic():
    cad = CADConstraints()
    result = add_mass_bounds(cad, "joint1", m_min=0.5, m_max=5.0)
    assert result is cad  # returns same object
    assert cad.mass_bounds["joint1"] == (0.5, 5.0)
    assert not cad.is_empty()


def test_add_mass_bounds_invalid_range():
    cad = CADConstraints()
    with pytest.raises(ValueError, match="m_min"):
        add_mass_bounds(cad, "joint1", m_min=5.0, m_max=0.5)


def test_add_com_bounds_basic():
    cad = CADConstraints()
    result = add_com_bounds(cad, "joint1", axis="x", h_min=-0.2, h_max=0.2)
    assert result is cad
    assert cad.com_bounds["joint1"]["x"] == (-0.2, 0.2)
    assert not cad.is_empty()


def test_add_com_bounds_invalid_axis():
    cad = CADConstraints()
    with pytest.raises(ValueError, match="axis"):
        add_com_bounds(cad, "joint1", axis="w", h_min=-0.1, h_max=0.1)


def test_add_com_bounds_invalid_range():
    cad = CADConstraints()
    with pytest.raises(ValueError, match="h_min"):
        add_com_bounds(cad, "joint1", axis="y", h_min=1.0, h_max=-1.0)


def test_add_symmetry_constraints_default_keys():
    cad = CADConstraints()
    add_symmetry_constraints(cad, "jL", "jR")
    assert len(cad.symmetry_pairs) == 1
    j_a, j_b, keys = cad.symmetry_pairs[0]
    assert j_a == "jL" and j_b == "jR"
    assert keys == list(_INERTIAL_KEYS)
    assert len(keys) == 10


def test_add_symmetry_constraints_custom_keys():
    cad = CADConstraints()
    add_symmetry_constraints(cad, "jL", "jR", keys=["m", "Ixx"])
    _, _, keys = cad.symmetry_pairs[0]
    assert keys == ["m", "Ixx"]


# ---------------------------------------------------------------------------
# build_cad_constraints_from_config
# ---------------------------------------------------------------------------


def test_build_from_config_empty():
    assert build_cad_constraints_from_config({}) is None
    assert build_cad_constraints_from_config(None) is None


def test_build_from_config_mass_bounds_list():
    cfg = {"mass_bounds": {"joint1": [0.5, 5.0]}}
    cad = build_cad_constraints_from_config(cfg)
    assert cad is not None
    assert cad.mass_bounds["joint1"] == (0.5, 5.0)


def test_build_from_config_symmetry():
    cfg = {"symmetry_pairs": [["jL", "jR", ["m", "Ixx"]]]}
    cad = build_cad_constraints_from_config(cfg)
    assert cad is not None
    assert len(cad.symmetry_pairs) == 1
    j_a, j_b, keys = cad.symmetry_pairs[0]
    assert j_a == "jL" and j_b == "jR"
    assert keys == ["m", "Ixx"]


def test_build_from_config_com_bounds():
    cfg = {"com_bounds": {"j1": {"x": [-0.3, 0.3], "z": [-0.1, 0.1]}}}
    cad = build_cad_constraints_from_config(cfg)
    assert cad is not None
    assert cad.com_bounds["j1"]["x"] == (-0.3, 0.3)
    assert cad.com_bounds["j1"]["z"] == (-0.1, 0.1)


# ---------------------------------------------------------------------------
# project_p10_lmi with mass_bounds (requires picos)
# ---------------------------------------------------------------------------


def test_project_p10_lmi_mass_bounds_enforced():
    """Tight upper mass bound must clamp the projected mass."""
    pc = pytest.importorskip("picos")  # skip if picos unavailable

    from figaroh.identification.physical_consistency import project_p10_lmi

    # Construct a physically consistent p10 with mass = 5.0 kg
    m = 5.0
    p10_hat = np.array([m, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0])

    # Apply tight upper bound of 2.0 kg -- projected mass must be <= 2.0
    mass_bounds = (1e-6, 2.0)
    p10_proj, report = project_p10_lmi(
        p10_hat, mass_bounds=mass_bounds, solver="cvxopt"
    )
    assert report.status in {"projected", "feasible"}, report
    assert p10_proj[0] <= 2.0 + 1e-5, f"mass {p10_proj[0]} > upper bound 2.0"


def test_cad_constraints_tighten_feasible_set():
    """ROADMAP gate: tight but valid mass bound tightens solution without infeasibility."""
    pc = pytest.importorskip("picos")  # skip if picos unavailable

    from figaroh.identification.physical_consistency import project_p10_lmi

    # Physically consistent p10, mass = 3.0 kg
    m0 = 3.0
    p10_hat = np.array([m0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.5])

    # Unconstrained projection (should keep mass ~ 3.0)
    p10_free, rep_free = project_p10_lmi(p10_hat, solver="cvxopt")
    assert rep_free.status in {"projected", "feasible"}, rep_free

    # Constrained with mass_bounds = (1e-6, 1.5)
    p10_cst, rep_cst = project_p10_lmi(
        p10_hat, mass_bounds=(1e-6, 1.5), solver="cvxopt"
    )
    assert rep_cst.status in {"projected", "feasible"}, rep_cst
    # Constrained solution must respect the bound
    assert p10_cst[0] <= 1.5 + 1e-5, f"mass {p10_cst[0]} > 1.5"
    # Constrained solution must be more different from p10_hat than unconstrained
    assert np.linalg.norm(p10_cst - p10_hat) > np.linalg.norm(p10_free - p10_hat) - 1e-8
