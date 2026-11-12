"""Numerical tests for the log-Cholesky inertial mappings (#23)."""

import subprocess
import sys

import numpy as np
import pytest

from figaroh.identification import inertial_parameterization as ip

pin = pytest.importorskip("pinocchio")

RTOL, ATOL = 1e-10, 1e-12
# (mass scale kg, length scale m): a 1 g fingertip to a 1 t base.
SCALES = [(1e-3, 1e-2), (1.0, 1.0), (50.0, 0.3), (1e3, 2.0)]


def _rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def _body(rng, mass_scale=1.0, length_scale=1.0):
    """Random solid ellipsoid: mass, CoM, inertia about the CoM."""
    m = mass_scale * rng.uniform(0.5, 2.0)
    axes = length_scale * rng.uniform(0.2, 1.0, size=3)
    a2 = axes**2
    principal = m / 5.0 * np.array([a2[1] + a2[2], a2[0] + a2[2], a2[0] + a2[1]])
    R = _rotation(rng)
    c = length_scale * rng.uniform(-0.5, 0.5, size=3)
    return m, c, R @ np.diag(principal) @ R.T


def _p10_from_body(m, c, I_C):
    """Parallel-axis oracle in Pinocchio order."""
    I_O = I_C + m * (c @ c * np.eye(3) - np.outer(c, c))
    return np.array(
        [m, *(m * c), I_O[0, 0], I_O[0, 1], I_O[1, 1], I_O[0, 2], I_O[1, 2], I_O[2, 2]]
    )


def _pseudo_from_body(m, c, I_C):
    """Block-formula oracle built from the CoM second moment."""
    sigma_C = 0.5 * np.trace(I_C) * np.eye(3) - I_C
    P = np.empty((4, 4))
    P[:3, :3] = sigma_C + m * np.outer(c, c)
    P[:3, 3] = P[3, :3] = m * c
    P[3, 3] = m
    return P


def _bodies(seed=2301, per_scale=25):
    rng = np.random.default_rng(seed)
    for M, L in SCALES:
        for _ in range(per_scale):
            yield (M, L), _body(rng, M, L)


def _decoded_checks(p10, tol=1e-9):
    m, h = p10[0], p10[1:4]
    assert m > 0
    c = h / m
    I_O = np.array(
        [[p10[4], p10[5], p10[7]], [p10[5], p10[6], p10[8]], [p10[7], p10[8], p10[9]]]
    )
    I_C = I_O - m * (c @ c * np.eye(3) - np.outer(c, c))
    lam = np.linalg.eigvalsh(I_C)
    scale = lam.max()
    assert lam.min() > -tol * scale
    # Principal-moment triangle inequalities.
    assert lam[0] + lam[1] >= lam[2] - tol * scale


@pytest.mark.parametrize("scale_index", range(len(SCALES)))
def test_round_trip_matches_pinocchio_and_block_oracle(scale_index):
    for (M, L), (m, c, I_C) in _bodies():
        if (M, L) != SCALES[scale_index]:
            continue
        p10 = _p10_from_body(m, c, I_C)
        P_oracle = _pseudo_from_body(m, c, I_C)

        P = ip.pseudo_inertia_from_p10(p10)
        np.testing.assert_allclose(P, P_oracle, rtol=RTOL, atol=ATOL * M * L**2)
        np.testing.assert_allclose(
            P, pin.PseudoInertia.FromDynamicParameters(p10).toMatrix(), rtol=RTOL
        )

        z = ip.log_cholesky_from_p10(p10)
        lc = pin.LogCholeskyParameters(z)
        np.testing.assert_allclose(z[0], 0.5 * np.log(m), rtol=RTOL)
        np.testing.assert_allclose(z[7:], c, rtol=RTOL, atol=ATOL * L)
        np.testing.assert_allclose(
            lc.toDynamicParameters(), p10, rtol=RTOL, atol=ATOL * M * L**2
        )

        back = ip.p10_from_log_cholesky(z)
        np.testing.assert_allclose(back, p10, rtol=RTOL, atol=ATOL * M * L**2)
        np.testing.assert_allclose(
            ip.pseudo_inertia_from_log_cholesky(z),
            P_oracle,
            rtol=RTOL,
            atol=ATOL * M * L**2,
        )
        np.testing.assert_allclose(
            ip.p10_from_pseudo_inertia(P_oracle), p10, rtol=RTOL, atol=ATOL * M * L**2
        )
        # Re-encoding the decoded body recovers the same coordinates.
        np.testing.assert_allclose(ip.log_cholesky_from_p10(back), z, atol=1e-9)
        _decoded_checks(back)


def test_log_determinant_is_linear():
    for _, body in _bodies(per_scale=3):
        z = ip.log_cholesky_from_p10(_p10_from_body(*body))
        _, logdet = np.linalg.slogdet(ip.pseudo_inertia_from_log_cholesky(z))
        assert logdet == pytest.approx(8 * z[0] + 2 * z[1:4].sum(), rel=1e-10)


def test_any_finite_coordinates_decode_to_a_physical_body():
    rng = np.random.default_rng(2302)
    for _ in range(200):
        z = np.concatenate([rng.uniform(-3, 3, 4), rng.uniform(-1, 1, 6)])
        p10 = ip.p10_from_log_cholesky(z)
        np.testing.assert_allclose(
            p10,
            pin.LogCholeskyParameters(z).toDynamicParameters(),
            rtol=RTOL,
            atol=1e-11,
        )
        assert np.linalg.eigvalsh(ip.pseudo_inertia_from_p10(p10)).min() > 0
        _decoded_checks(p10)


@pytest.mark.parametrize("scale_index", range(len(SCALES)))
def test_jacobian_matches_central_differences_and_pinocchio(scale_index):
    M, L = SCALES[scale_index]
    rng = np.random.default_rng(2303 + scale_index)
    for _ in range(10):
        z = ip.log_cholesky_from_p10(_p10_from_body(*_body(rng, M, L)))
        J = ip.p10_jacobian_from_log_cholesky(z)
        np.testing.assert_allclose(
            J, pin.LogCholeskyParameters(z).calculateJacobian(), rtol=1e-10, atol=1e-12
        )
        # Step relative to each coordinate's natural size.
        steps = 1e-6 * np.maximum(1.0, np.abs(z))
        steps[4:] = 1e-6 * max(L, np.abs(z[4:]).max())
        J_fd = np.empty((10, 10))
        for k in range(10):
            e = np.zeros(10)
            e[k] = steps[k]
            J_fd[:, k] = (
                ip.p10_from_log_cholesky(z + e) - ip.p10_from_log_cholesky(z - e)
            ) / (2 * steps[k])
        col_scale = np.abs(J).max(axis=0)
        np.testing.assert_allclose(J, J_fd, rtol=1e-5, atol=1e-7 * col_scale.max())


@pytest.mark.parametrize(
    "bad, match",
    [
        (np.zeros(9), r"shape \(10,\)"),
        (np.ones((2, 10)), r"shape \(10,\)"),
        (np.r_[1.0, np.nan, np.zeros(8)], "nonfinite entries at indices \\[1\\]"),
        (np.r_[np.inf, np.zeros(9)], "nonfinite"),
    ],
)
def test_rejects_malformed_p10(bad, match):
    with pytest.raises(ValueError, match=match):
        ip.log_cholesky_from_p10(bad)
    with pytest.raises(ValueError, match=match):
        ip.repair_p10(bad)


def test_rejects_malformed_coordinates_and_pseudo_inertia():
    with pytest.raises(ValueError, match="shape"):
        ip.p10_from_log_cholesky(np.zeros(11))
    with pytest.raises(ValueError, match="nonfinite"):
        ip.p10_jacobian_from_log_cholesky(np.r_[np.nan, np.zeros(9)])
    with pytest.raises(ValueError, match=r"shape \(4, 4\)"):
        ip.log_cholesky_from_pseudo_inertia(np.eye(3))
    P = np.eye(4)
    P[0, 1] = 0.1
    with pytest.raises(ValueError, match="not symmetric"):
        ip.log_cholesky_from_pseudo_inertia(P)
    with pytest.raises(ValueError, match="not symmetric"):
        ip.p10_from_pseudo_inertia(P)


def test_rejects_massless_singular_and_inconsistent_links():
    with pytest.raises(ValueError, match="massless or fixed link"):
        ip.log_cholesky_from_p10(np.zeros(10))
    with pytest.raises(ValueError, match="massless or fixed link"):
        ip.log_cholesky_from_p10(np.r_[-1.0, np.zeros(3), np.ones(6)])
    # Point mass at the origin: Sigma = 0, so P is singular.
    with pytest.raises(ValueError, match="not positive definite.*repair_p10"):
        ip.log_cholesky_from_p10(np.r_[1.0, np.zeros(9)])
    # Violates the triangle inequality (Izz > Ixx + Iyy).
    with pytest.raises(ValueError, match="not positive definite"):
        ip.log_cholesky_from_p10([1.0, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0.5])


def test_exponent_guards():
    z = np.zeros(10)
    for idx, value in [(0, 31.0), (2, -31.0), (3, 400.0)]:
        bad = z.copy()
        bad[idx] = value
        name = ip.LOG_CHOLESKY_ORDER[idx]
        for fn in (
            ip.p10_from_log_cholesky,
            ip.pseudo_inertia_from_log_cholesky,
            ip.p10_jacobian_from_log_cholesky,
        ):
            with pytest.raises(ValueError, match=rf"\|{name}\|.*max_log_scale"):
                fn(bad)
    # The bound is explicit and adjustable.
    bad = z.copy()
    bad[0] = 31.0
    assert ip.p10_from_log_cholesky(bad, max_log_scale=40.0)[0] == pytest.approx(
        np.exp(62.0)
    )
    # Off-diagonal overflow is caught after decoding.
    huge = z.copy()
    huge[0], huge[4] = 29.0, 1e300
    with pytest.raises(ValueError, match="overflowed"):
        ip.p10_from_log_cholesky(huge)
    # A well-conditioned body whose encoding lies beyond the bound (1e-27 kg:
    # a unit error) is rejected on encode too.
    tiny = ip.p10_from_log_cholesky(np.r_[-31.0, np.zeros(9)], max_log_scale=40.0)
    with pytest.raises(ValueError, match=r"encoding: \|alpha\|.*Check units"):
        ip.log_cholesky_from_p10(tiny)


def test_repair_is_opt_in_reported_and_preserves_input():
    point = np.r_[2.0, np.zeros(9)]  # point mass at the origin: singular P
    original = point.copy()
    report = ip.repair_p10(point)
    np.testing.assert_array_equal(point, original)
    np.testing.assert_array_equal(report.p10_original, original)
    assert report.p10_original is not point
    assert report.repaired
    assert report.min_eig_before <= 1e-12
    assert report.min_eig_after == pytest.approx(
        1e-8 * np.linalg.eigvalsh(ip.pseudo_inertia_from_p10(original)).max(), rel=1e-6
    )
    assert 0 < report.p10_change_norm < 1e-6
    assert report.p10_relative_change == pytest.approx(
        report.p10_change_norm / np.linalg.norm(original)
    )
    z = ip.log_cholesky_from_p10(report.p10)
    np.testing.assert_allclose(ip.p10_from_log_cholesky(z), report.p10, atol=1e-12)

    # An inconsistent body (offset point mass, Sigma < h h^T / m) moves by
    # an amount the report makes visible.
    report = ip.repair_p10(np.r_[2.0, 0.2, 0.0, 0.0, np.zeros(6)])
    assert report.min_eig_before < -1e-3 and report.p10_relative_change > 1e-3
    ip.log_cholesky_from_p10(report.p10)

    # Explicit floor, and a physically consistent body is left alone.
    report = ip.repair_p10([1.0, 0, 0, 0, 0.1, 0, 0.1, 0, 0, 0.5], min_eig=1e-3)
    assert report.repaired and report.min_eig_after == pytest.approx(1e-3)
    good = _p10_from_body(*_body(np.random.default_rng(1)))
    report = ip.repair_p10(good)
    assert not report.repaired and report.p10_change_norm == 0.0
    np.testing.assert_array_equal(report.p10, good)

    with pytest.raises(ValueError, match="does not invent mass"):
        ip.repair_p10(np.zeros(10))
    with pytest.raises(ValueError, match="positive and finite"):
        ip.repair_p10(good, min_eig=-1.0)


def _param_dict(rng):
    d = {"fv_j1": 0.3, "off_j2": -0.1}
    for joint in ("j1", "j2", "j3"):
        p10 = _p10_from_body(*_body(rng))
        d.update({f"{k}_{joint}": float(v) for k, v in zip(ip.P10_ORDER, p10)})
    d.update({f"{k}_fixed": 0.0 for k in ip.P10_ORDER})
    d["Ia_j3"] = 0.05
    return d


def test_per_joint_mappings_preserve_keys_and_do_not_mutate():
    params = _param_dict(np.random.default_rng(2304))
    snapshot = dict(params)
    joints = ["j1", "j2", "j3", "fixed"]

    with pytest.raises(ValueError, match="joint 'fixed': mass"):
        ip.log_cholesky_by_joint_from_param_dict(
            parameter_dict=params, joint_names=joints
        )
    with pytest.raises(ValueError, match="'raise' or 'skip'"):
        ip.log_cholesky_by_joint_from_param_dict(
            parameter_dict=params, joint_names=joints, massless="drop"
        )
    with pytest.raises(KeyError, match="m_missing"):
        ip.log_cholesky_by_joint_from_param_dict(
            parameter_dict=params, joint_names=["missing"]
        )

    z_by_joint = ip.log_cholesky_by_joint_from_param_dict(
        parameter_dict=params, joint_names=joints, massless="skip"
    )
    assert list(z_by_joint) == ["j1", "j2", "j3"]
    assert params == snapshot

    out = ip.param_dict_with_log_cholesky_by_joint(
        parameter_dict=params, log_cholesky_by_joint=z_by_joint
    )
    assert params == snapshot
    assert out is not params
    assert list(out) == list(params)
    for key, value in params.items():
        assert out[key] == pytest.approx(value, rel=1e-10, abs=1e-12), key

    moved = dict(z_by_joint)
    moved["j2"] = z_by_joint["j2"] + 0.1
    out = ip.param_dict_with_log_cholesky_by_joint(
        parameter_dict=params, log_cholesky_by_joint=moved
    )
    expected = ip.p10_from_log_cholesky(moved["j2"])
    assert [out[f"{k}_j2"] for k in ip.P10_ORDER] == pytest.approx(expected)
    for key in params:
        if not key.endswith("_j2"):
            assert out[key] == pytest.approx(params[key], rel=1e-10, abs=1e-12), key
    bad = {"j1": np.r_[99.0, np.zeros(9)]}
    with pytest.raises(ValueError, match="joint 'j1'.*max_log_scale"):
        ip.param_dict_with_log_cholesky_by_joint(
            parameter_dict=params, log_cholesky_by_joint=bad
        )


def test_mappings_do_not_import_sdp_solvers():
    code = (
        "import sys\n"
        "import figaroh.identification.inertial_parameterization as ip\n"
        "ip.p10_from_log_cholesky(ip.log_cholesky_from_p10("
        "[1, 0, 0, 0, 1, 0, 1, 0, 0, 1]))\n"
        "ip.repair_p10([1, 0, 0, 0, 0, 0, 0, 0, 0, 0])\n"
        "print(sorted(m for m in ('picos', 'cvxopt') if m in sys.modules))\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert out.stdout.strip().splitlines()[-1] == "[]"
