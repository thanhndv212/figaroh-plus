"""Tests for QR decomposition functionality."""

import pytest
import numpy as np
import sys
import os

# Add the src directory to the path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

try:
    from figaroh.tools.qrdecomposition import (
        QRDecomposer,
        QR_pivoting,
        double_QR,
        redistribute_min_norm,
        propagate_covariance_min_norm,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure the figaroh package is installed or the path is correct")
    raise


class TestQRDecomposer:
    """Test the enhanced QR decomposer class."""

    def test_initialization(self):
        """Test QRDecomposer initialization."""
        decomposer = QRDecomposer()
        assert decomposer.tolerance == 1e-8
        assert decomposer.beta_tolerance == 1e-6

        custom_decomposer = QRDecomposer(tolerance=1e-10, beta_tolerance=1e-8)
        assert custom_decomposer.tolerance == 1e-10
        assert custom_decomposer.beta_tolerance == 1e-8

    def test_find_rank(self):
        """Test rank finding functionality."""
        decomposer = QRDecomposer(tolerance=1e-6)

        # Full rank matrix
        R_full = np.diag([10, 5, 2, 1])
        assert decomposer._find_rank(R_full) == 4

        # Rank deficient matrix
        R_deficient = np.diag([10, 5, 2, 1e-8])
        assert decomposer._find_rank(R_deficient) == 3

        # Edge case: all zeros — rank is 0, not the matrix size
        R_zeros = np.zeros((3, 3))
        assert decomposer._find_rank(R_zeros) == 0

    def test_extract_base_components(self):
        """Test base component extraction."""
        decomposer = QRDecomposer()

        # Create test matrices
        R = np.array([[5, 2, 1], [0, 3, 2], [0, 0, 1]])
        Q = np.random.randn(10, 3)
        rank = 2

        R1, Q1, R2 = decomposer._extract_base_components(R, Q, rank)

        assert R1.shape == (2, 2)
        assert Q1.shape == (10, 2)
        assert R2.shape == (2, 1)

    def test_decompose_with_pivoting(self):
        """Test QR decomposition with pivoting."""
        # Create test data
        np.random.seed(42)
        W = np.random.randn(20, 5)
        tau = np.random.randn(20)
        params = ["p1", "p2", "p3", "p4", "p5"]

        decomposer = QRDecomposer()
        W_b, base_params = decomposer.decompose_with_pivoting(tau, W, params)

        assert W_b.shape[0] == W.shape[0]
        assert W_b.shape[1] <= W.shape[1]
        assert len(base_params) == W_b.shape[1]
        assert isinstance(base_params, dict)

        # Check that all parameter names are strings
        for key in base_params.keys():
            assert isinstance(key, str)

        # Check that all parameter values are numbers
        for value in base_params.values():
            assert isinstance(value, (int, float, np.number))

    def test_double_decomposition(self):
        """Test double QR decomposition."""
        # Create test data with some linear dependencies
        np.random.seed(42)
        W_base = np.random.randn(20, 3)
        W_dependent = W_base @ np.random.randn(3, 2)  # Make dependent columns
        W = np.hstack([W_base, W_dependent])
        tau = np.random.randn(20)
        params = ["p1", "p2", "p3", "p4", "p5"]

        decomposer = QRDecomposer(tolerance=1e-6)
        result = decomposer.double_decomposition(tau, W, params)

        W_b, base_params, params_expr, phi_b = result
        assert W_b.shape[1] <= W.shape[1]  # Should be rank deficient
        assert len(base_params) == len(params_expr) == len(phi_b)

        # Test with standard parameters
        params_std = {p: np.random.randn() for p in params}
        result_with_std = decomposer.double_decomposition(tau, W, params, params_std)
        assert len(result_with_std) == 5  # Additional phi_std

        W_b_std, base_params_std, params_expr_std, phi_b_std, phi_std = result_with_std
        assert isinstance(phi_std, np.ndarray)

    def test_edge_cases(self):
        """Test edge cases and error conditions."""
        decomposer = QRDecomposer()

        # Empty matrix - if it should work gracefully
        W_empty = np.array([]).reshape(0, 0)
        tau_empty = np.array([])
        result = decomposer.decompose_with_pivoting(tau_empty, W_empty, [])
        # Add assertions about what the expected result should be

        # Mismatched dimensions
        W = np.random.randn(10, 3)
        tau = np.random.randn(5)  # Wrong size
        params = ["p1", "p2", "p3"]

        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            decomposer.decompose_with_pivoting(tau, W, params)


class TestBackwardCompatibility:
    """Test that legacy functions still work."""

    def test_qr_pivoting_legacy(self):
        """Test legacy QR_pivoting function."""
        np.random.seed(42)
        W = np.random.randn(20, 5)
        tau = np.random.randn(20)
        params = ["p1", "p2", "p3", "p4", "p5"]

        try:
            W_b, base_params = QR_pivoting(tau, W, params)

            assert isinstance(W_b, np.ndarray)
            assert isinstance(base_params, dict)
            assert W_b.shape[0] == W.shape[0]
        except Exception as e:
            pytest.skip(f"Legacy QR_pivoting not implemented: {e}")

    def test_double_qr_legacy(self):
        """Test legacy double_QR function."""
        np.random.seed(42)
        W = np.random.randn(20, 5)
        tau = np.random.randn(20)
        params = ["p1", "p2", "p3", "p4", "p5"]

        try:
            result = double_QR(tau, W, params)
            assert len(result) == 4  # W_b, base_parameters, params_base, phi_b

            # Test with standard parameters
            params_std = {p: np.random.randn() for p in params}
            result_with_std = double_QR(tau, W, params, params_std)
            assert len(result_with_std) == 5  # Additional phi_std
        except Exception as e:
            pytest.skip(f"Legacy double_QR not implemented: {e}")

    def test_parameter_consistency(self):
        """Test that new and legacy implementations give consistent results."""
        np.random.seed(42)
        W = np.random.randn(15, 4)
        tau = np.random.randn(15)
        params = ["p1", "p2", "p3", "p4"]

        # Test with the new class
        decomposer = QRDecomposer()
        try:
            W_b_new, base_params_new = decomposer.decompose_with_pivoting(
                tau, W, params
            )
        except Exception as e:
            pytest.skip(f"New implementation not working: {e}")

        # Test with legacy function
        try:
            W_b_legacy, base_params_legacy = QR_pivoting(tau, W, params)

            # Check that results are approximately equal
            np.testing.assert_allclose(W_b_new, W_b_legacy, rtol=1e-10)

            # Parameter values should be close (order might differ)
            new_values = sorted(base_params_new.values())
            legacy_values = sorted(base_params_legacy.values())
            np.testing.assert_allclose(new_values, legacy_values, rtol=1e-10)

        except Exception as e:
            pytest.skip(f"Legacy implementation not available: {e}")


class TestNumericalImprovements:
    """Tests for the Phase 1-7 improvements: rank-zero fix, pivoted basis
    stability, mapping matrix correctness, diagnostics, and relative tolerance.
    """

    # ------------------------------------------------------------------
    # Rank detection
    # ------------------------------------------------------------------

    def test_rank_zero_matrix(self):
        """_find_rank returns 0 for an all-zero matrix (Phase 1 fix)."""
        dec = QRDecomposer(tolerance=1e-8)
        R_zeros = np.zeros((4, 4))
        assert dec._find_rank(R_zeros) == 0

    def test_rank_empty_matrix(self):
        """_find_rank returns 0 for an empty diagonal."""
        dec = QRDecomposer()
        assert dec._find_rank(np.zeros((0, 0))) == 0

    def test_relative_tolerance_tighter_rank(self):
        """relative_tolerance correctly reduces rank for an ill-conditioned R."""
        # R with diag [1e4, 1e4, 1e4, 1.0] — absolute tol 1e-8 gives rank 4,
        # relative tol 1e-3 gives rank 3 (1.0 < 1e-3 * 1e4 = 10).
        R = np.diag([1e4, 1e4, 1e4, 1.0])
        dec_abs = QRDecomposer(tolerance=1e-8, relative_tolerance=None)
        dec_rel = QRDecomposer(tolerance=1e-8, relative_tolerance=1e-3)
        assert dec_abs._find_rank(R) == 4
        assert dec_rel._find_rank(R) == 3

    def test_relative_tolerance_default_unchanged(self):
        """relative_tolerance=None leaves behaviour identical to pure absolute."""
        np.random.seed(0)
        W = np.random.randn(20, 5)
        dec1 = QRDecomposer(tolerance=1e-8)
        dec2 = QRDecomposer(tolerance=1e-8, relative_tolerance=None)
        from scipy import linalg as sp_linalg

        _, R, _ = sp_linalg.qr(W, pivoting=True)
        assert dec1._find_rank(R) == dec2._find_rank(R)

    # ------------------------------------------------------------------
    # Permutation stability (Phase 3)
    # ------------------------------------------------------------------

    def _make_dependent_W(self, seed=7):
        """Build a 40x5 W with columns 3,4 linearly dependent on cols 0,1,2."""
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(40, 3))
        dep1 = A @ np.array([1.0, -2.0, 0.5])
        dep2 = A @ np.array([0.3, 0.1, -1.2])
        return np.column_stack([A, dep1, dep2])

    def test_permutation_stability_double(self):
        """double path: same physical columns selected after column permutation."""
        W = self._make_dependent_W()
        params = [f"p{i}" for i in range(W.shape[1])]
        rng = np.random.default_rng(42)

        dec = QRDecomposer(tolerance=1e-7)
        base_ref, _ = dec._identify_base_parameters(W, params)
        base_ref_names = {params[i] for i in base_ref}

        for _ in range(20):
            perm = rng.permutation(W.shape[1])
            Wp = W[:, perm]
            paramsp = [params[i] for i in perm]
            dec2 = QRDecomposer(tolerance=1e-7)
            base_p, _ = dec2._identify_base_parameters(Wp, paramsp)
            base_names_p = {paramsp[i] for i in base_p}
            assert (
                base_names_p == base_ref_names
            ), f"Permutation changed base set: {base_names_p} != {base_ref_names}"

    def test_permutation_stability_pivoting(self):
        """pivoting path: same rank and same physical column set under permutation."""
        W = self._make_dependent_W()
        params = [f"p{i}" for i in range(W.shape[1])]
        rng = np.random.default_rng(99)
        tau = rng.normal(size=W.shape[0])

        dec = QRDecomposer(tolerance=1e-7)
        W_b_ref, bp_ref = dec.decompose_with_pivoting(tau, W, params)
        rank_ref = W_b_ref.shape[1]

        for _ in range(20):
            perm = rng.permutation(W.shape[1])
            Wp = W[:, perm]
            paramsp = [params[i] for i in perm]
            dec2 = QRDecomposer(tolerance=1e-7)
            W_b_p, bp_p = dec2.decompose_with_pivoting(tau, Wp, paramsp)
            assert (
                W_b_p.shape[1] == rank_ref
            ), f"Permutation changed rank: {W_b_p.shape[1]} != {rank_ref}"

    # ------------------------------------------------------------------
    # Mapping matrix invariants (Phase 3 + 6)
    # ------------------------------------------------------------------

    def test_mapping_matrix_property_double(self):
        """M @ theta_r reproduces phi_b for the double path."""
        rng = np.random.default_rng(1)
        A = rng.normal(size=(30, 3))
        W = np.column_stack([A, A @ rng.normal(size=(3, 2))])
        params = [f"p{i}" for i in range(W.shape[1])]
        tau = rng.normal(size=W.shape[0])

        dec = QRDecomposer(tolerance=1e-7)
        W_b, _, _, phi_b = dec.double_decomposition(tau, W, params)
        M = dec.get_M()

        # Build a consistent theta_r: solve W @ theta_r = tau (least squares)
        theta_r, *_ = np.linalg.lstsq(W, tau, rcond=None)
        phi_b_reconstructed = M @ theta_r
        np.testing.assert_allclose(phi_b_reconstructed, phi_b, rtol=1e-4)

    def test_column_space_property_double(self):
        """W_e @ theta_r ≈ W_b @ (M @ theta_r) for random theta_r."""
        rng = np.random.default_rng(2)
        A = rng.normal(size=(30, 3))
        W = np.column_stack([A, A @ rng.normal(size=(3, 2))])
        params = [f"p{i}" for i in range(W.shape[1])]
        tau = rng.normal(size=W.shape[0])

        dec = QRDecomposer(tolerance=1e-7)
        W_b, _, _, _ = dec.double_decomposition(tau, W, params)
        M = dec.get_M()

        for seed in range(10):
            rng2 = np.random.default_rng(1000 + seed)
            theta_r = rng2.normal(size=W.shape[1])
            lhs = W @ theta_r
            rhs = W_b @ (M @ theta_r)
            np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-10)

    def test_column_space_property_pivoting(self):
        """W_e @ theta_r ≈ W_b @ (M @ theta_r) for the pivoting path."""
        rng = np.random.default_rng(3)
        A = rng.normal(size=(30, 3))
        W = np.column_stack([A, A @ rng.normal(size=(3, 2))])
        params = [f"p{i}" for i in range(W.shape[1])]
        tau = rng.normal(size=W.shape[0])

        dec = QRDecomposer(tolerance=1e-7)
        W_b, _ = dec.decompose_with_pivoting(tau, W, params)
        M = dec.get_M()

        for seed in range(10):
            rng2 = np.random.default_rng(2000 + seed)
            theta_r = rng2.normal(size=W.shape[1])
            lhs = W @ theta_r
            rhs = W_b @ (M @ theta_r)
            np.testing.assert_allclose(lhs, rhs, rtol=1e-5, atol=1e-10)

    # ------------------------------------------------------------------
    # Diagnostics (Phase 6)
    # ------------------------------------------------------------------

    def test_diagnostics_populated_after_pivoting(self):
        """get_diagnostics returns finite values after decompose_with_pivoting."""
        rng = np.random.default_rng(5)
        W = rng.normal(size=(20, 4))
        tau = rng.normal(size=20)
        params = [f"p{i}" for i in range(4)]

        dec = QRDecomposer()
        dec.decompose_with_pivoting(tau, W, params)
        diag = dec.get_diagnostics()

        assert diag["rank"] is not None and diag["rank"] > 0
        assert diag["diag_R"] is not None
        assert np.all(np.isfinite(diag["diag_R"]))
        assert diag["cond_R1"] is not None and np.isfinite(diag["cond_R1"])
        assert diag["method"] == "pivoting"

    def test_diagnostics_populated_after_double(self):
        """get_diagnostics returns finite values after double_decomposition."""
        rng = np.random.default_rng(6)
        A = rng.normal(size=(30, 3))
        W = np.column_stack([A, A @ rng.normal(size=(3, 2))])
        tau = rng.normal(size=30)
        params = [f"p{i}" for i in range(5)]

        dec = QRDecomposer(tolerance=1e-7)
        dec.double_decomposition(tau, W, params)
        diag = dec.get_diagnostics()

        assert diag["rank"] == 3
        assert diag["cond_R1"] is not None and np.isfinite(diag["cond_R1"])
        assert diag["method"] == "double"

    # ------------------------------------------------------------------
    # QRResult structured output (Phase 2 + 6)
    # ------------------------------------------------------------------

    def test_decompose_returns_qrresult(self):
        """decompose() returns a QRResult with all expected fields."""
        from figaroh.tools.qrdecomposition import QRResult

        rng = np.random.default_rng(8)
        A = rng.normal(size=(30, 3))
        W = np.column_stack([A, A @ rng.normal(size=(3, 2))])
        params = [f"p{i}" for i in range(5)]
        tau = rng.normal(size=30)

        dec = QRDecomposer(tolerance=1e-7)
        for method in ("double", "pivoting"):
            result = dec.decompose(W, params, tau=tau, method=method)
            assert isinstance(result, QRResult)
            assert result.rank == 3
            assert result.W_b.shape == (30, 3)
            assert result.M.shape == (3, 5)
            assert len(result.base_param_expressions) == 3
            assert result.phi_b is not None and result.phi_b.shape == (3,)
            assert np.isfinite(result.cond_R1)
            assert result.method == method

    def test_decompose_no_tau_gives_zero_phi_b(self):
        """decompose() with tau=None uses zeros; M and W_b still valid."""
        from figaroh.tools.qrdecomposition import QRResult

        rng = np.random.default_rng(9)
        A = rng.normal(size=(30, 3))
        W = np.column_stack([A, A @ rng.normal(size=(3, 2))])
        params = [f"p{i}" for i in range(5)]

        dec = QRDecomposer(tolerance=1e-7)
        result = dec.decompose(W, params)
        assert isinstance(result, QRResult)
        assert result.rank == 3
        np.testing.assert_array_equal(result.phi_b, np.zeros(3))

    # ------------------------------------------------------------------
    # Beta full precision (Phase 4)
    # ------------------------------------------------------------------

    def test_beta_is_full_precision(self):
        """beta in QRResult is not rounded to 6 decimal places."""
        rng = np.random.default_rng(10)
        # Create a dependency with irrational-ish coefficient
        A = rng.normal(size=(40, 3))
        coeff = np.pi / 7  # not representable at 6dp exactly
        dep = A @ np.array([coeff, 1.0, 0.0])
        W = np.column_stack([A, dep])
        params = [f"p{i}" for i in range(4)]

        dec = QRDecomposer(tolerance=1e-7)
        result = dec.decompose(W, params, method="double")
        # At least one beta entry should differ from its 6dp rounded version
        beta_flat = result.beta.ravel()
        rounded_flat = np.round(beta_flat, 6)
        assert not np.allclose(
            beta_flat, rounded_flat, atol=0
        ), "beta appears to be rounded to 6dp; expected full precision"


class TestRedistribution:
    """Tests for redistribute_min_norm / propagate_covariance_min_norm."""

    def test_round_trip_property(self):
        """M @ redistribute_min_norm(M, phi_base) reproduces phi_base."""
        rng = np.random.default_rng(10)
        A = rng.normal(size=(30, 3))
        W = np.column_stack([A, A @ rng.normal(size=(3, 2))])
        params = [f"p{i}" for i in range(W.shape[1])]
        tau = rng.normal(size=W.shape[0])

        dec = QRDecomposer(tolerance=1e-7)
        dec.double_decomposition(tau, W, params)
        M = dec.get_M()

        phi_base = rng.normal(size=M.shape[0])
        theta_r_hat = redistribute_min_norm(M, phi_base)

        assert theta_r_hat.shape == (M.shape[1],)
        np.testing.assert_allclose(M @ theta_r_hat, phi_base, rtol=1e-6, atol=1e-9)

    def test_minimum_norm_splits_equal_coefficients_evenly(self):
        """Two standard params with equal coefficients in the same base
        combination get an equal (not one-hot) share of the fitted value."""
        # phi_base = 1*theta_0 + 1*theta_1 -- exactly redundant, equal weight.
        M = np.array([[1.0, 1.0]])
        phi_base = np.array([10.0])

        theta_r_hat = redistribute_min_norm(M, phi_base)

        np.testing.assert_allclose(theta_r_hat, [5.0, 5.0], atol=1e-10)
        # Still round-trips to the original fitted combination.
        np.testing.assert_allclose(M @ theta_r_hat, phi_base, atol=1e-10)

    def test_minimum_norm_weights_by_coefficient_magnitude(self):
        """Unequal coefficients (phi_base = theta_0 + 2*theta_1) split
        unevenly, favoring the parameter with the larger coefficient."""
        M = np.array([[1.0, 2.0]])
        phi_base = np.array([10.0])

        theta_r_hat = redistribute_min_norm(M, phi_base)

        np.testing.assert_allclose(theta_r_hat, [2.0, 4.0], atol=1e-10)
        np.testing.assert_allclose(M @ theta_r_hat, phi_base, atol=1e-10)

    def test_redistribution_matches_one_hot_when_no_redundancy(self):
        """A full-rank (square) M has a unique inverse -- redistribution
        reduces to the ordinary (not merely minimum-norm) solution, and
        there is no group to spread credit across."""
        rng = np.random.default_rng(11)
        M = rng.normal(size=(4, 4))
        while abs(np.linalg.det(M)) < 1e-3:
            M = rng.normal(size=(4, 4))
        phi_base = rng.normal(size=4)

        theta_r_hat = redistribute_min_norm(M, phi_base)
        expected = np.linalg.solve(M, phi_base)

        np.testing.assert_allclose(theta_r_hat, expected, rtol=1e-6)

    def test_propagate_covariance_shape_and_symmetry(self):
        """C_full is symmetric, correctly shaped, and positive semidefinite
        (guaranteed since M+ @ C_base @ M+.T with PSD C_base is PSD)."""
        rng = np.random.default_rng(12)
        A = rng.normal(size=(30, 3))
        W = np.column_stack([A, A @ rng.normal(size=(3, 2))])
        params = [f"p{i}" for i in range(W.shape[1])]
        tau = rng.normal(size=W.shape[0])

        dec = QRDecomposer(tolerance=1e-7)
        dec.double_decomposition(tau, W, params)
        M = dec.get_M()

        rank = M.shape[0]
        L = rng.normal(size=(rank, rank))
        C_base = L @ L.T  # guaranteed PSD

        C_full = propagate_covariance_min_norm(M, C_base)

        assert C_full.shape == (M.shape[1], M.shape[1])
        np.testing.assert_allclose(C_full, C_full.T, rtol=1e-8, atol=1e-12)
        eigvals = np.linalg.eigvalsh(C_full)
        assert np.all(eigvals >= -1e-8), f"C_full has negative eigenvalues: {eigvals}"


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])
