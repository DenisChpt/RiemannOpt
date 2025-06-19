"""Unit tests for the PSD cone manifold."""

import numpy as np
import pytest
import riemannopt


class TestPSDCone:
    """Test cases for the PSD cone manifold."""
    
    def test_creation(self):
        """Test PSD cone manifold creation."""
        # Valid creation
        manifold = riemannopt.PSDCone(4)
        assert manifold.n == 4
        assert manifold.dim == 4 * (4 + 1) // 2  # n*(n+1)/2
        
        # Invalid creation
        with pytest.raises(ValueError):
            riemannopt.PSDCone(0)
    
    def test_projection(self):
        """Test projection onto the PSD cone."""
        manifold = riemannopt.PSDCone(3)
        
        # Create a non-PSD matrix (negative eigenvalue)
        X = np.array([[1, 0.5, 0.3],
                      [0.5, -1, 0.2],
                      [0.3, 0.2, 0.5]])
        
        # Check that original is not PSD
        eigenvalues = np.linalg.eigvalsh(X)
        assert np.min(eigenvalues) < 0
        
        # Make sure X is float64
        X = X.astype(np.float64)
        
        # Project
        X_proj = manifold.project(X)
        
        # Check that projection is PSD
        eigenvalues_proj = np.linalg.eigvalsh(X_proj)
        assert np.all(eigenvalues_proj >= -1e-10)
        
        # Check symmetry
        np.testing.assert_allclose(X_proj, X_proj.T, rtol=1e-10)
    
    def test_tangent_projection(self):
        """Test tangent space projection."""
        manifold = riemannopt.PSDCone(3)
        
        # Generate a point on the manifold
        X = manifold.random_point()
        
        # Non-symmetric matrix
        V = np.array([[1, 2, 3],
                      [0, 1, 2],
                      [0, 0, 1]], dtype=np.float64)
        
        V_tan = manifold.tangent_projection(X, V)
        
        # Check symmetry of projection
        np.testing.assert_allclose(V_tan, V_tan.T, rtol=1e-10)
        
        # Check that projection is idempotent
        V_tan2 = manifold.tangent_projection(X, V_tan)
        np.testing.assert_allclose(V_tan, V_tan2, rtol=1e-10)
    
    def test_retraction(self):
        """Test retraction operation."""
        manifold = riemannopt.PSDCone(4)
        
        X = manifold.random_point()
        V = manifold.random_tangent(X)
        
        # Small step retraction
        Y = manifold.retract(X, 0.1 * V)
        
        # Check that result is PSD
        eigenvalues = np.linalg.eigvalsh(Y)
        assert np.all(eigenvalues >= -1e-10)
        
        # Check symmetry
        np.testing.assert_allclose(Y, Y.T, rtol=1e-10)
        
        # Zero tangent vector should give same point
        Y_zero = manifold.retract(X, np.zeros_like(V))
        np.testing.assert_allclose(Y_zero, X, rtol=1e-10)
    
    def test_inner_product(self):
        """Test inner product computation."""
        manifold = riemannopt.PSDCone(3)
        
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
        
        # Frobenius inner product
        expected_ip = np.trace(U.T @ V)
        assert abs(ip_uv - expected_ip) < 1e-10
    
    def test_distance(self):
        """Test distance computation."""
        manifold = riemannopt.PSDCone(3)
        
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
        assert abs(dist_xx) < 1e-10
        
        # Frobenius distance
        expected_dist = np.linalg.norm(X - Y, 'fro')
        assert abs(dist_xy - expected_dist) < 1e-10
    
    def test_random_generation(self):
        """Test random point and tangent generation."""
        manifold = riemannopt.PSDCone(4)
        
        # Random point
        X = manifold.random_point()
        assert X.shape == (4, 4)
        
        # Check PSD
        eigenvalues = np.linalg.eigvalsh(X)
        assert np.all(eigenvalues >= -1e-10)
        
        # Check symmetry
        np.testing.assert_allclose(X, X.T, rtol=1e-10)
        
        # Random tangent
        V = manifold.random_tangent(X)
        assert V.shape == (4, 4)
        
        # Check symmetry
        np.testing.assert_allclose(V, V.T, rtol=1e-10)
    
    def test_check_point(self):
        """Test point validation."""
        manifold = riemannopt.PSDCone(3)
        
        # Valid point (PSD)
        X = manifold.random_point()
        assert manifold.check_point(X)
        
        # Invalid point (negative eigenvalue)
        X_invalid = np.array([[1, 0, 0],
                              [0, -1, 0],
                              [0, 0, 1]], dtype=np.float64)
        assert not manifold.check_point(X_invalid)
        
        # Non-symmetric
        X_nonsym = np.array([[1, 2, 3],
                             [0, 1, 2],
                             [0, 0, 1]], dtype=np.float64)
        assert not manifold.check_point(X_nonsym)
        
        # Wrong shape
        X_wrong = np.random.randn(2, 3)
        with pytest.raises(ValueError):
            manifold.check_point(X_wrong)
    
    def test_special_cases(self):
        """Test special cases and edge conditions."""
        # Small dimension
        manifold1 = riemannopt.PSDCone(1)
        X1 = manifold1.random_point()
        assert X1.shape == (1, 1)
        assert X1[0, 0] >= 0
        
        # Rank-deficient case
        manifold2 = riemannopt.PSDCone(3)
        # Create rank-1 PSD matrix
        v = np.array([[1], [2], [3]], dtype=np.float64)
        X2 = v @ v.T
        assert manifold2.check_point(X2)
        
        # Zero matrix
        X_zero = np.zeros((3, 3))
        assert manifold2.check_point(X_zero)
    
    def test_optimization_compatibility(self):
        """Test that PSD cone works with optimizers."""
        manifold = riemannopt.PSDCone(3)
        
        # Define a simple cost function: minimize trace
        def cost(X):
            return np.trace(X)
        
        def grad(X):
            return np.eye(3)
        
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
            
            # Check that X stays PSD
            eigenvalues = np.linalg.eigvalsh(X)
            assert np.all(eigenvalues >= -1e-10)
        
        # Cost should not increase (for this simple problem)
        assert cost(X) <= cost(X0) + 1e-6
    
    def test_boundary_behavior(self):
        """Test behavior at the boundary of the cone."""
        manifold = riemannopt.PSDCone(3)
        
        # Create a matrix at the boundary (rank-deficient)
        v1 = np.array([[1], [0], [0]], dtype=np.float64)
        v2 = np.array([[0], [1], [0]], dtype=np.float64)
        X = v1 @ v1.T + v2 @ v2.T  # Rank 2
        
        assert manifold.check_point(X)
        
        # Tangent vector that moves into the interior
        V = np.eye(3) * 0.1
        Y = manifold.retract(X, V)
        
        # Y should have full rank
        rank = np.linalg.matrix_rank(Y, tol=1e-10)
        assert rank == 3