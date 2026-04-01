"""Tests for reconstruction utilities."""

import numpy as np

from figaroh.identification.reconstruction import (
    prior_vector_from_dict,
    reconstruct_from_base,
    reconstruct_theta_r,
)


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
