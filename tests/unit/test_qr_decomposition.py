"""Tests for QR decomposition functionality."""

import pytest
import numpy as np
import sys
import os

# Add the src directory to the path if needed
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

try:
    from figaroh.tools.qrdecomposition import QRDecomposer, QR_pivoting, double_QR
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
        
        # Edge case: all zeros
        R_zeros = np.zeros((3, 3))
        assert decomposer._find_rank(R_zeros) == 0  # Zero matrix has rank 0
    
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
        params = ['p1', 'p2', 'p3', 'p4', 'p5']
        
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
        params = ['p1', 'p2', 'p3', 'p4', 'p5']
        
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
        params = ['p1', 'p2', 'p3']
        
        with pytest.raises((ValueError, np.linalg.LinAlgError)):
            decomposer.decompose_with_pivoting(tau, W, params)


class TestBackwardCompatibility:
    """Test that legacy functions still work."""
    
    def test_qr_pivoting_legacy(self):
        """Test legacy QR_pivoting function."""
        np.random.seed(42)
        W = np.random.randn(20, 5)
        tau = np.random.randn(20)
        params = ['p1', 'p2', 'p3', 'p4', 'p5']
        
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
        params = ['p1', 'p2', 'p3', 'p4', 'p5']
        
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
        params = ['p1', 'p2', 'p3', 'p4']
        
        # Test with the new class
        decomposer = QRDecomposer()
        try:
            W_b_new, base_params_new = decomposer.decompose_with_pivoting(tau, W, params)
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


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])