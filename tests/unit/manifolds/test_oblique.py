"""Unit tests for the Oblique manifold."""

import numpy as np
import pytest
import riemannopt


class TestOblique:
    """Test cases for the Oblique manifold."""
    
    def test_creation(self):
        """Test Oblique manifold creation."""
        # Valid creation
        manifold = riemannopt.Oblique(5, 3)
        assert manifold.n == 5
        assert manifold.p == 3
        assert manifold.dim == 3 * (5 - 1)  # p*(n-1)
        
        # Invalid creation
        with pytest.raises(ValueError):
            riemannopt.Oblique(0, 3)
        with pytest.raises(ValueError):
            riemannopt.Oblique(5, 0)
    
    def test_projection(self):
        """Test projection onto the manifold."""
        manifold = riemannopt.Oblique(4, 3)
        
        # Random matrix
        X = np.random.randn(4, 3)
        X_proj = manifold.project(X)
        
        # Check that columns have unit norm
        col_norms = np.linalg.norm(X_proj, axis=0)
        np.testing.assert_allclose(col_norms, np.ones(3), rtol=1e-10)
        
        # Test with zero column
        X_zero = X.copy()
        X_zero[:, 1] = 0
        X_proj_zero = manifold.project(X_zero)
        # Should handle gracefully
        assert X_proj_zero.shape == (4, 3)
    
    def test_tangent_projection(self):
        """Test tangent space projection."""
        manifold = riemannopt.Oblique(4, 3)
        
        # Generate a point on the manifold
        X = manifold.random_point()
        
        # Random vector
        V = np.random.randn(4, 3)
        V_tan = manifold.tangent_projection(X, V)
        
        # Check orthogonality: each column of V_tan should be orthogonal to corresponding column of X
        for j in range(3):
            inner_prod = np.dot(X[:, j], V_tan[:, j])
            assert abs(inner_prod) < 1e-10
    
    def test_retraction(self):
        """Test retraction operation."""
        manifold = riemannopt.Oblique(5, 2)
        
        X = manifold.random_point()
        V = manifold.random_tangent(X)
        
        # Small step retraction
        Y = manifold.retract(X, 0.1 * V)
        
        # Check that result is on manifold
        col_norms = np.linalg.norm(Y, axis=0)
        np.testing.assert_allclose(col_norms, np.ones(2), rtol=1e-10)
        
        # Zero tangent vector should give same point
        Y_zero = manifold.retract(X, np.zeros_like(V))
        np.testing.assert_allclose(Y_zero, X, rtol=1e-10)
    
    def test_inner_product(self):
        """Test inner product computation."""
        manifold = riemannopt.Oblique(4, 3)
        
        X = manifold.random_point()
        U = manifold.random_tangent(X)
        V = manifold.random_tangent(X)
        
        # Inner product should be symmetric
        ip_uv = manifold.inner_product(X, U, V)
        ip_vu = manifold.inner_product(X, V, U)
        assert abs(ip_uv - ip_vu) < 1e-10
        
        # Inner product with self should be positive
        ip_uu = manifold.inner_product(X, U, U)
        assert ip_uu > 0
        
        # Standard Euclidean inner product
        expected_ip = np.sum(U * V)
        assert abs(ip_uv - expected_ip) < 1e-10
    
    def test_distance(self):
        """Test distance computation."""
        manifold = riemannopt.Oblique(4, 2)
        
        X = manifold.random_point()
        Y = manifold.random_point()
        
        # Distance properties
        dist_xy = manifold.distance(X, Y)
        dist_yx = manifold.distance(Y, X)
        
        # Symmetry
        assert abs(dist_xy - dist_yx) < 1e-10
        
        # Non-negativity
        assert dist_xy >= 0
        
        # Self-distance
        dist_xx = manifold.distance(X, X)
        assert abs(dist_xx) < 1e-6
        
        # Triangle inequality (with third point)
        Z = manifold.random_point()
        dist_xz = manifold.distance(X, Z)
        dist_zy = manifold.distance(Z, Y)
        assert dist_xy <= dist_xz + dist_zy + 1e-10
    
    def test_random_generation(self):
        """Test random point and tangent generation."""
        manifold = riemannopt.Oblique(5, 3)
        
        # Random point
        X = manifold.random_point()
        assert X.shape == (5, 3)
        col_norms = np.linalg.norm(X, axis=0)
        np.testing.assert_allclose(col_norms, np.ones(3), rtol=1e-10)
        
        # Random tangent
        V = manifold.random_tangent(X)
        assert V.shape == (5, 3)
        
        # Check tangent space constraint
        for j in range(3):
            inner_prod = np.dot(X[:, j], V[:, j])
            assert abs(inner_prod) < 1e-10
    
    def test_check_point(self):
        """Test point validation."""
        manifold = riemannopt.Oblique(4, 2)
        
        # Valid point
        X = manifold.random_point()
        assert manifold.check_point(X)
        
        # Invalid point (not unit columns)
        X_invalid = np.random.randn(4, 2)
        assert not manifold.check_point(X_invalid)
        
        # Wrong shape
        X_wrong = np.random.randn(3, 2)
        assert not manifold.check_point(X_wrong)
    
    def test_special_cases(self):
        """Test special cases and edge conditions."""
        # Single column (equivalent to sphere)
        manifold1 = riemannopt.Oblique(5, 1)
        X1 = manifold1.random_point()
        assert X1.shape == (5, 1)
        assert abs(np.linalg.norm(X1) - 1.0) < 1e-10
        
        # Single row per column
        manifold2 = riemannopt.Oblique(1, 5)
        X2 = manifold2.random_point()
        assert X2.shape == (1, 5)
        # Each element should be ±1
        assert np.all(np.abs(np.abs(X2) - 1.0) < 1e-10)
    
    def test_optimization_compatibility(self):
        """Test that Oblique manifold works with optimizers."""
        manifold = riemannopt.Oblique(5, 3)
        
        # Define a simple cost function: minimize Frobenius norm
        def cost(X):
            return 0.5 * np.sum(X**2)
        
        def grad(X):
            return X
        
        # Initial point
        X0 = manifold.random_point()
        
        # Create optimizer
        optimizer = riemannopt.SGD(
            step_size=0.1,
            momentum=0.0
        )
        
        # Run a few iterations
        X = X0.copy()
        for _ in range(10):
            g = grad(X)
            X = optimizer.step(manifold, X, g)
            
            # Check that X stays on manifold
            col_norms = np.linalg.norm(X, axis=0)
            np.testing.assert_allclose(col_norms, np.ones(3), rtol=1e-6)
        
        # Cost should not increase (for this simple problem)
        assert cost(X) <= cost(X0) + 1e-6