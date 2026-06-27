"""Tests for reconstruction utilities."""

import numpy as np
import pytest

from figaroh.identification.reconstruction import (
    BaseResult,
    ReconstructionResult,
    prior_vector_from_dict,
    reconstruct_from_base,
    reconstruct_full_parameters,
    reconstruct_theta_r,
    run_reconstruction,
    _load_prior_from_yaml,
    _p10_indices_for_joints,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_underdetermined(n: int = 5, r: int = 3):
    """Return (M, theta_true, phi) for an underdetermined system."""
    rng = np.random.default_rng(42)
    M = rng.standard_normal((r, n))
    theta_true = rng.standard_normal(n)
    phi = M @ theta_true
    return M, theta_true, phi


# ---------------------------------------------------------------------------
# Existing tests (unchanged)
# ---------------------------------------------------------------------------


def test_prior_vector_from_dict_defaults():
    params_r = ["a", "b", "c"]
    theta0 = prior_vector_from_dict(params_r, None, default=1.5)
    assert theta0.shape == (3,)
    assert np.allclose(theta0, 1.5)


def test_reconstruct_theta_r_exact_square():
    # Full-rank square: unique solution.
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    theta_true = np.array([0.5, -1.0])
    phi = M @ theta_true

    theta_hat, residual = reconstruct_theta_r(M, phi)
    assert theta_hat.shape == (2,)
    assert np.linalg.norm(residual) < 1e-10
    assert np.allclose(theta_hat, theta_true)


def test_reconstruct_theta_r_projection_to_prior():
    # Under-determined: pick solution closest to prior.
    # Constraint: theta1 + theta2 = 1.
    M = np.array([[1.0, 1.0]])
    phi = np.array([1.0])

    theta0 = np.array([10.0, 10.0])
    theta_hat, residual = reconstruct_theta_r(M, phi, theta0=theta0)

    assert np.linalg.norm(residual) < 1e-10
    # Closest point to (10,10) on line x+y=1 is (0.5, 0.5).
    assert np.allclose(theta_hat, np.array([0.5, 0.5]))


def test_reconstruct_from_base_labels_and_dict():
    M = np.array([[1.0, 0.0, 1.0]])
    params_r = ["p1", "p2", "p3"]
    phi = np.array([2.0])

    prior = {"p1": 0.0, "p2": 5.0, "p3": 0.0}
    res = reconstruct_from_base(M, phi, params_r, params_std_prior=prior)

    assert res.theta_r.shape == (3,)
    assert res.residual.shape == (1,)
    assert np.linalg.norm(res.residual) < 1e-10

    as_dict = res.as_dict()
    assert set(as_dict.keys()) == set(params_r)
    assert np.isclose(as_dict["p2"], 5.0)


# ---------------------------------------------------------------------------
# New v0.4.2 tests
# ---------------------------------------------------------------------------

# --- ReconstructionResult new fields ---


def test_reconstruction_result_new_fields_defaults():
    """ReconstructionResult defaults: status='ok', base_residual_norm=None, objective=None."""
    theta = np.zeros(3)
    res = ReconstructionResult(
        theta_r=theta, params_r=["a", "b", "c"], residual=np.zeros(2)
    )
    assert res.status == "ok"
    assert res.base_residual_norm is None
    assert res.objective is None


def test_reconstruction_result_fields_explicit():
    """ReconstructionResult accepts all new fields explicitly."""
    theta = np.array([1.0, 2.0])
    residual = np.array([0.01])
    res = ReconstructionResult(
        theta_r=theta,
        params_r=["x", "y"],
        residual=residual,
        status="ok",
        base_residual_norm=0.01,
        objective=3.14,
    )
    assert res.status == "ok"
    assert np.isclose(res.base_residual_norm, 0.01)
    assert np.isclose(res.objective, 3.14)


# --- BaseResult ---


def test_base_result_construction():
    """BaseResult stores M, phi_base, params_r correctly."""
    M = np.eye(3)
    phi = np.array([1.0, 2.0, 3.0])
    params = ["a", "b", "c"]
    br = BaseResult(M=M, phi_base=phi, params_r=params)
    assert br.M.shape == (3, 3)
    assert br.phi_base.shape == (3,)
    assert br.params_r == params
    assert br.phi_base_dict is None


# --- reconstruct_from_base sets base_residual_norm ---


def test_reconstruct_from_base_sets_residual_norm():
    """reconstruct_from_base must set base_residual_norm."""
    M = np.array([[1.0, 1.0]])
    phi = np.array([1.0])
    res = reconstruct_from_base(M, phi, ["p", "q"])
    assert res.base_residual_norm is not None
    assert np.isclose(res.base_residual_norm, 0.0, atol=1e-10)


# --- _load_prior_from_yaml ---


def test_load_prior_from_yaml(tmp_path):
    """_load_prior_from_yaml reads correct values and defaults missing ones."""
    import yaml

    yaml_content = {"m_j1": 2.5, "mx_j1": 0.1, "Ixx_j1": 0.05}
    yaml_file = tmp_path / "prior.yaml"
    yaml_file.write_text(yaml.safe_dump(yaml_content))

    params_r = ["m_j1", "mx_j1", "Ixx_j1", "unk"]
    result = _load_prior_from_yaml(str(yaml_file), params_r, default=-1.0)
    assert np.isclose(result["m_j1"], 2.5)
    assert np.isclose(result["mx_j1"], 0.1)
    assert np.isclose(result["Ixx_j1"], 0.05)
    assert np.isclose(result["unk"], -1.0)


def test_load_prior_from_yaml_bad_format(tmp_path):
    """_load_prior_from_yaml raises ValueError for non-dict YAML."""
    yaml_file = tmp_path / "bad.yaml"
    yaml_file.write_text("- a\n- b\n")
    with pytest.raises(ValueError, match="top-level mapping"):
        _load_prior_from_yaml(str(yaml_file), ["a"])


# --- _p10_indices_for_joints ---


def test_p10_indices_for_joints_complete():
    """_p10_indices_for_joints returns all 10 indices when params_r is complete."""
    keys = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]
    params_r = [f"{k}_j1" for k in keys]
    result = _p10_indices_for_joints(params_r, ["j1"])
    assert "j1" in result
    assert len(result["j1"]) == 10
    assert result["j1"]["m"] == 0


def test_p10_indices_for_joints_incomplete_skipped():
    """Joints missing any of the 10 keys are silently skipped."""
    params_r = ["m_j1", "mx_j1"]  # incomplete
    result = _p10_indices_for_joints(params_r, ["j1"])
    assert "j1" not in result


# --- reconstruct_full_parameters (nullspace) ---


def test_reconstruct_full_parameters_nullspace_base_result():
    """reconstruct_full_parameters with a BaseResult returns constraint-satisfying theta."""
    M, theta_true, phi = _make_underdetermined()
    br = BaseResult(M=M, phi_base=phi, params_r=[f"p{i}" for i in range(M.shape[1])])
    result = reconstruct_full_parameters(br, method="nullspace")
    assert result.status == "ok"
    assert result.base_residual_norm is not None
    assert result.base_residual_norm < 1e-8
    assert result.objective is None  # no SDP


def test_reconstruct_full_parameters_nullspace_tuple_input():
    """reconstruct_full_parameters accepts (M, phi_base, params_r) tuple."""
    M, _, phi = _make_underdetermined(n=4, r=2)
    params_r = [f"q{i}" for i in range(4)]
    result = reconstruct_full_parameters((M, phi, params_r), method="nullspace")
    assert result.status == "ok"
    assert result.base_residual_norm < 1e-8


def test_reconstruct_full_parameters_auto_falls_back_to_nullspace():
    """method='auto' must succeed even if picos is missing (falls back to nullspace)."""
    M, _, phi = _make_underdetermined(n=3, r=2)
    params_r = [f"r{i}" for i in range(3)]
    # Patch picos import to simulate it being absent
    import sys
    import unittest.mock as mock

    orig = sys.modules.get("picos")
    try:
        sys.modules["picos"] = None  # type: ignore[assignment]
        result = reconstruct_full_parameters(
            (M, phi, params_r),
            method="auto",
            joint_names=["j1"],  # joint_names not needed for nullspace
        )
        assert result.status == "ok"
        assert result.base_residual_norm < 1e-8
    finally:
        if orig is None:
            sys.modules.pop("picos", None)
        else:
            sys.modules["picos"] = orig


def test_reconstruct_full_parameters_prior_from_yaml(tmp_path):
    """prior_source='yaml' loads theta0 from file."""
    import yaml

    M = np.array([[1.0, 1.0]])
    phi = np.array([3.0])
    params_r = ["p", "q"]
    prior_val = {"p": 1.0, "q": 2.0}
    f = tmp_path / "prior.yaml"
    f.write_text(yaml.safe_dump(prior_val))

    result = reconstruct_full_parameters(
        (M, phi, params_r),
        method="nullspace",
        prior_source="yaml",
        prior_yaml_path=str(f),
    )
    assert result.status == "ok"
    assert result.base_residual_norm < 1e-8
    # Both solutions satisfy p+q=3; closest to (1,2) is (1,2)
    assert np.isclose(result.theta_r[0], 1.0)
    assert np.isclose(result.theta_r[1], 2.0)


def test_reconstruct_full_parameters_unsupported_method():
    """Unsupported method name raises ValueError."""
    M, _, phi = _make_underdetermined(n=3, r=2)
    with pytest.raises(ValueError, match="Unsupported method"):
        reconstruct_full_parameters((M, phi, ["a", "b", "c"]), method="garbage")
