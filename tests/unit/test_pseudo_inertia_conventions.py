"""Independent convention regressions for physical projection/reconstruction."""

import sys

import numpy as np
import pytest

from figaroh.identification.physical_consistency import (
    project_p10_lmi,
    pseudo_inertia_matrix_from_p10,
)
from figaroh.identification.reconstruction import (
    _load_prior_from_urdf,
    reconstruct_full_parameters,
)


def _physical_body():
    """Dynamic parameters of a translated, anisotropic body."""
    mass = 2.0
    com = np.array([0.2, -0.1, 0.3])
    inertia_com = np.array([[3.0, 0.1, -0.2], [0.1, 4.0, 0.3], [-0.2, 0.3, 5.0]])
    inertia_origin = inertia_com + mass * (
        np.dot(com, com) * np.eye(3) - np.outer(com, com)
    )
    p10 = np.array(
        [
            mass,
            *(mass * com),
            inertia_origin[0, 0],
            inertia_origin[0, 1],
            inertia_origin[1, 1],
            inertia_origin[0, 2],
            inertia_origin[1, 2],
            inertia_origin[2, 2],
        ]
    )
    second_moment = (
        0.5 * np.trace(inertia_com) * np.eye(3)
        - inertia_com
        + mass * np.outer(com, com)
    )
    expected = np.block(
        [
            [second_moment, (mass * com)[:, None]],
            [(mass * com)[None, :], np.array([[mass]])],
        ]
    )
    return p10, expected


@pytest.mark.parametrize("translated", [False, True])
def test_fallback_matches_independent_second_moment_and_pinocchio(
    monkeypatch, translated
):
    pin = pytest.importorskip("pinocchio")
    if translated:
        p10, expected = _physical_body()
    else:
        p10 = np.array([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 4.0, 0.0, 0.0, 5.0])
        expected = np.diag([3.0, 2.0, 1.0, 2.0])
    canonical = np.asarray(pin.PseudoInertia.FromDynamicParameters(p10).toMatrix())
    monkeypatch.setitem(sys.modules, "pinocchio", None)
    actual = pseudo_inertia_matrix_from_p10(p10)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(actual, canonical, rtol=1e-10, atol=1e-12)


def test_projection_keeps_translated_physical_body_and_dynamic_objective():
    pytest.importorskip("picos")
    p10, expected = _physical_body()
    weights = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    projected, report = project_p10_lmi(p10, weights=weights)
    assert report.status == "projected", report
    np.testing.assert_allclose(projected, p10, rtol=0, atol=1e-4)
    np.testing.assert_allclose(
        pseudo_inertia_matrix_from_p10(projected), expected, rtol=0, atol=1e-4
    )
    assert report.objective == pytest.approx(
        np.sum((weights * (projected - p10)) ** 2), abs=1e-8
    )


def test_projection_repairs_triangle_inequality_with_dynamic_parameter_weights():
    pytest.importorskip("picos")
    pin = pytest.importorskip("pinocchio")
    # Positive rotational inertia is insufficient: 3 > 1 + 1.
    p10 = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0])
    weights = np.ones(10)
    weights[9] = 2.0
    # Weighted projection onto Izz <= Ixx + Iyy: changes (4/9, 4/9, -1/9).
    expected = p10.copy()
    expected[[4, 6, 9]] = [13.0 / 9.0, 13.0 / 9.0, 26.0 / 9.0]
    projected, report = project_p10_lmi(p10, weights=weights)
    assert report.status == "projected", report
    np.testing.assert_allclose(projected, expected, rtol=0, atol=1e-4)
    canonical = np.asarray(
        pin.PseudoInertia.FromDynamicParameters(projected).toMatrix()
    )
    assert np.linalg.eigvalsh(canonical).min() >= -1e-8
    assert report.objective == pytest.approx(
        np.sum((weights * (projected - p10)) ** 2), abs=1e-7
    )


def test_reconstruction_enforces_triangle_inequality_and_base_equalities():
    pytest.importorskip("picos")
    pin = pytest.importorskip("pinocchio")
    keys = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]
    params = [f"{key}_body" for key in keys]
    prior = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 3.0])
    # Preserve the first nine base combinations; only Izz is free to change.
    matrix = np.eye(10)[:9]
    result = reconstruct_full_parameters(
        (matrix, matrix @ prior, params),
        method="sdp",
        theta0=prior,
        joint_names=["body"],
    )
    assert result.status == "ok"
    np.testing.assert_allclose(
        matrix @ result.theta_r, matrix @ prior, rtol=0, atol=1e-8
    )
    assert result.theta_r[9] == pytest.approx(2.0, abs=1e-5)
    canonical = np.asarray(
        pin.PseudoInertia.FromDynamicParameters(result.theta_r).toMatrix()
    )
    assert np.linalg.eigvalsh(canonical).min() >= -1e-8
    assert result.objective == pytest.approx(1.0, abs=1e-5)


def test_projection_preserves_first_moment_bound_semantics():
    pytest.importorskip("picos")
    pin = pytest.importorskip("pinocchio")
    p10, _ = _physical_body()
    projected, report = project_p10_lmi(
        p10, mass_bounds=(1.0, 1.5), com_bounds={"x": (-0.1, 0.1)}
    )
    assert report.status == "projected", report
    assert 1.0 - 1e-7 <= projected[0] <= 1.5 + 1e-7
    # Existing com_bounds options bound h=m*c, not c itself.
    assert -0.1 - 1e-7 <= projected[1] <= 0.1 + 1e-7
    canonical = np.asarray(
        pin.PseudoInertia.FromDynamicParameters(projected).toMatrix()
    )
    assert np.linalg.eigvalsh(canonical).min() >= -1e-8


def test_urdf_reconstruction_prior_uses_link_origin_inertia():
    pin = pytest.importorskip("pinocchio")
    p10, _ = _physical_body()
    model = pin.Model()
    joint = model.addJoint(0, pin.JointModelRZ(), pin.SE3.Identity(), "body")
    model.appendBodyToJoint(
        joint, pin.Inertia.FromDynamicParameters(p10), pin.SE3.Identity()
    )
    keys = ["m", "mx", "my", "mz", "Ixx", "Ixy", "Iyy", "Ixz", "Iyz", "Izz"]
    params = [f"{key}_body" for key in keys]
    # Request a reordered subset as well as unknown/non-inertial defaults.
    requested = list(reversed(params)) + ["fv_body", "m_unknown"]
    prior = _load_prior_from_urdf(model, requested, default=-1.0)
    np.testing.assert_allclose(
        [prior[name] for name in params], p10, rtol=1e-10, atol=1e-12
    )
    assert prior["fv_body"] == -1.0
    assert prior["m_unknown"] == -1.0
