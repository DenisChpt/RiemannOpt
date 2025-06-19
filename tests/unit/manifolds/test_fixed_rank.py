"""Unit tests for the Fixed-rank manifold."""

import numpy as np
import pytest
import riemannopt


class TestFixedRank:
    """Test cases for the Fixed-rank manifold."""
    
    def test_creation(self):
        """Test Fixed-rank manifold creation."""
        # Valid creation
        manifold = riemannopt.FixedRank(10, 8, 3)
        assert manifold.m == 10
        assert manifold.n == 8
        assert manifold.k == 3
        assert manifold.dim == 3 * (10 + 8 - 3)  # k*(m+n-k)
        
        # Invalid creation
        with pytest.raises(ValueError):
            riemannopt.FixedRank(0, 8, 3)
        with pytest.raises(ValueError):
            riemannopt.FixedRank(10, 0, 3)
        with pytest.raises(ValueError):
            riemannopt.FixedRank(10, 8, 0)
        with pytest.raises(ValueError):
            riemannopt.FixedRank(5, 4, 6)  # k > min(m,n)
    
    def test_projection(self):
        """Test projection onto the manifold."""
        manifold = riemannopt.FixedRank(6, 4, 2)
        
        # Random full-rank matrix
        X = np.random.randn(6, 4)
        X_proj = manifold.project(X)
        
        # Check that projection has correct shape
        assert X_proj.shape == (6, 4)
        
        # Check rank
        rank = np.linalg.matrix_rank(X_proj, tol=1e-10)
        assert rank == 2
    
    def test_tangent_projection(self):
        """Test tangent space projection."""
        manifold = riemannopt.FixedRank(5, 4, 2)
        
        # Generate a point on the manifold
        X = manifold.random_point()
        
        # Random vector
        V = np.random.randn(5, 4)
        V_tan = manifold.tangent_projection(X, V)
        
        # Check shape
        assert V_tan.shape == (5, 4)
        
        # Check that projection is idempotent
        V_tan2 = manifold.tangent_projection(X, V_tan)
        np.testing.assert_allclose(V_tan, V_tan2, rtol=1e-10)
    
    def test_retraction(self):
        """Test retraction operation."""
        manifold = riemannopt.FixedRank(6, 5, 3)
        
        X = manifold.random_point()
        V = manifold.random_tangent(X)
        
        # Small step retraction
        Y = manifold.retract(X, 0.1 * V)
        
        # Check that result has correct rank
        rank = np.linalg.matrix_rank(Y, tol=1e-10)
        assert rank == 3
        
        # Zero tangent vector should give same point
        Y_zero = manifold.retract(X, np.zeros_like(V))
        np.testing.assert_allclose(Y_zero, X, rtol=1e-10)
    
    def test_inner_product(self):
        """Test inner product computation."""
        manifold = riemannopt.FixedRank(4, 3, 2)
        
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
    
    def test_random_generation(self):
        """Test random point and tangent generation."""
        manifold = riemannopt.FixedRank(7, 5, 3)
        
        # Random point
        X = manifold.random_point()
        assert X.shape == (7, 5)
        rank = np.linalg.matrix_rank(X, tol=1e-10)
        assert rank == 3
        
        # Random tangent
        V = manifold.random_tangent(X)
        assert V.shape == (7, 5)
    
    def test_check_point(self):
        """Test point validation."""
        manifold = riemannopt.FixedRank(5, 4, 2)
        
        # Valid point (rank 2)
        U = np.random.randn(5, 2)
        V = np.random.randn(4, 2)
        X = U @ V.T
        assert manifold.check_point(X)
        
        # Invalid point (wrong rank)
        X_full = np.random.randn(5, 4)
        assert not manifold.check_point(X_full)
        
        # Wrong shape
        X_wrong = np.random.randn(3, 4)
        with pytest.raises(ValueError):
            manifold.check_point(X_wrong)
    
    def test_special_cases(self):
        """Test special cases and edge conditions."""
        # Square matrices
        manifold1 = riemannopt.FixedRank(5, 5, 2)
        X1 = manifold1.random_point()
        assert X1.shape == (5, 5)
        assert np.linalg.matrix_rank(X1, tol=1e-10) == 2
        
        # Rank 1 matrices
        manifold2 = riemannopt.FixedRank(6, 4, 1)
        X2 = manifold2.random_point()
        assert X2.shape == (6, 4)
        assert np.linalg.matrix_rank(X2, tol=1e-10) == 1
        
        # Maximum rank
        manifold3 = riemannopt.FixedRank(5, 3, 3)
        X3 = manifold3.random_point()
        assert X3.shape == (5, 3)
        assert np.linalg.matrix_rank(X3, tol=1e-10) == 3
    
    def test_optimization_compatibility(self):
        """Test that Fixed-rank manifold works with optimizers."""
        manifold = riemannopt.FixedRank(6, 4, 2)
        
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
            
            # Check that X maintains its rank
            rank = np.linalg.matrix_rank(X, tol=1e-10)
            assert rank == 2
        
        # Cost should not increase (for this simple problem)
        assert cost(X) <= cost(X0) + 1e-6
    
    def test_low_rank_approximation(self):
        """Test that projection gives best low-rank approximation."""
        manifold = riemannopt.FixedRank(8, 6, 3)
        
        # Create a full-rank matrix
        A = np.random.randn(8, 6)
        
        # Project to get rank-3 approximation
        A_proj = manifold.project(A)
        
        # Compute SVD for verification
        U, s, Vt = np.linalg.svd(A, full_matrices=False)
        A_best = U[:, :3] @ np.diag(s[:3]) @ Vt[:3, :]
        
        # The projection should be close to the best rank-3 approximation
        np.testing.assert_allclose(A_proj, A_best, rtol=1e-10)