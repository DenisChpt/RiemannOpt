"""
Unit tests for the Grassmann manifold.

This module tests all operations and properties of the Grassmann manifold Gr(n,p),
including projection to horizontal space, retractions, and geometric properties.
"""

import pytest
import numpy as np
from typing import List
from conftest import riemannopt, TOLERANCES, MATRIX_DIMENSION_CONFIGS


class TestGrassmannCreation:
    """Test Grassmann manifold creation and basic properties."""
    
    @pytest.mark.parametrize("n,p", [(10, 3), (20, 5), (100, 10)])
    def test_create_valid_grassmann(self, grassmann_factory, n, p):
        """Test creation of Grassmann manifold with valid dimensions."""
        grassmann = grassmann_factory(n, p)
        
        assert grassmann.n == n
        assert grassmann.p == p
        assert grassmann.ambient_dim == n
        assert grassmann.subspace_dim == p
        
        # Check manifold dimension
        expected_dim = p * (n - p)
        assert grassmann.dim == expected_dim
        assert grassmann.manifold_dim == expected_dim
    
    def test_create_invalid_grassmann(self):
        """Test that invalid dimensions raise appropriate errors."""
        # p > n
        with pytest.raises(ValueError):
            riemannopt.Grassmann(5, 10)
        
        # Zero dimensions
        with pytest.raises(ValueError):
            riemannopt.Grassmann(0, 5)
        with pytest.raises(ValueError):
            riemannopt.Grassmann(5, 0)
    
    def test_grassmann_representation(self, grassmann_factory):
        """Test string representation of Grassmann manifold."""
        grassmann = grassmann_factory(10, 3)
        assert str(grassmann) == "Grassmann(n=10, p=3)"
        assert repr(grassmann) == "Grassmann(n=10, p=3)"
    
    def test_grassmann_vs_stiefel_dimension(self, grassmann_factory, stiefel_factory):
        """Test that Grassmann has correct dimension relative to Stiefel."""
        n, p = 10, 3
        grassmann = grassmann_factory(n, p)
        stiefel = stiefel_factory(n, p)
        
        # Grassmann is quotient of Stiefel by O(p)
        # dim(Gr) = dim(St) - dim(O(p))
        o_p_dim = p * (p - 1) // 2  # Dimension of O(p)
        assert grassmann.dim == stiefel.dim - o_p_dim


class TestGrassmannProjection:
    """Test projection operations for Grassmann manifold."""
    
    @pytest.mark.parametrize("n,p", MATRIX_DIMENSION_CONFIGS['small'])
    def test_projection_creates_orthonormal_basis(self, grassmann_factory, assert_helpers, n, p):
        """Test that projection creates orthonormal basis for subspace."""
        grassmann = grassmann_factory(n, p)
        
        # Test various matrices
        test_matrices = [
            np.random.randn(n, p),
            np.ones((n, p)),
            np.eye(n, p),
            np.random.randn(n, p) * 100,
        ]
        
        for M in test_matrices:
            X = grassmann.project(M)
            assert X.shape == (n, p)
            assert_helpers.assert_is_orthogonal(X)
    
    def test_projection_preserves_subspace(self, grassmann_factory):
        """Test that projection preserves the subspace spanned by columns."""
        grassmann = grassmann_factory(10, 3)
        
        # Create two matrices spanning the same subspace
        M1 = np.random.randn(10, 3)
        # M2 spans same subspace but with different basis
        R = np.random.randn(3, 3)
        while np.linalg.matrix_rank(R) < 3:
            R = np.random.randn(3, 3)
        M2 = M1 @ R
        
        X1 = grassmann.project(M1)
        X2 = grassmann.project(M2)
        
        # X1 and X2 should span the same subspace
        # Check if X1 = X2 @ Q for some orthogonal Q
        Q, residual, rank, _ = np.linalg.lstsq(X2, X1, rcond=None)
        assert rank == 3
        assert np.allclose(X1, X2 @ Q, atol=TOLERANCES['relaxed'])
        
        # Q should be orthogonal
        assert np.allclose(Q @ Q.T, np.eye(3), atol=TOLERANCES['relaxed'])
    
    def test_projection_idempotent(self, grassmann_factory, assert_helpers):
        """Test that projecting twice gives an equivalent representative."""
        grassmann = grassmann_factory(15, 4)
        M = np.random.randn(15, 4)
        
        X1 = grassmann.project(M)
        X2 = grassmann.project(X1)
        
        # X1 and X2 might differ by orthogonal transformation
        # but should span same subspace
        Q, _, rank, _ = np.linalg.lstsq(X1, X2, rcond=None)
        assert rank == 4
        assert np.allclose(X2, X1 @ Q, atol=TOLERANCES['strict'])
        assert_helpers.assert_is_orthogonal(X2)


class TestGrassmannTangentSpace:
    """Test tangent space (horizontal space) operations on Grassmann."""
    
    @pytest.mark.parametrize("n,p", MATRIX_DIMENSION_CONFIGS['small'])
    def test_tangent_projection_horizontal(self, grassmann_factory, assert_helpers, n, p):
        """Test that tangent projection gives horizontal vectors (X^T V = 0)."""
        grassmann = grassmann_factory(n, p)
        X = grassmann.random_point()
        
        for _ in range(10):
            H = np.random.randn(n, p)
            V = grassmann.tangent_projection(X, H)
            assert_helpers.assert_in_tangent_space_grassmann(X, V)
    
    def test_tangent_projection_removes_vertical(self, grassmann_factory):
        """Test that tangent projection removes vertical component."""
        grassmann = grassmann_factory(10, 3)
        X = grassmann.random_point()
        
        # Create a vector with known vertical component
        A = np.random.randn(3, 3)
        A = (A - A.T) / 2  # Skew-symmetric
        V_vertical = X @ A  # Vertical vector
        
        H_horizontal = np.random.randn(10, 3)
        H_horizontal = H_horizontal - X @ (X.T @ H_horizontal)  # Make horizontal
        
        # Combined vector
        H = V_vertical + H_horizontal
        
        # Project to tangent space
        V = grassmann.tangent_projection(X, H)
        
        # Should equal horizontal part
        assert np.allclose(V, H_horizontal, atol=TOLERANCES['default'])
    
    def test_tangent_projection_linearity(self, grassmann_factory):
        """Test linearity of tangent projection."""
        grassmann = grassmann_factory(12, 4)
        X = grassmann.random_point()
        
        H1 = np.random.randn(12, 4)
        H2 = np.random.randn(12, 4)
        alpha, beta = 2.5, -1.3
        
        V1 = grassmann.tangent_projection(X, H1)
        V2 = grassmann.tangent_projection(X, H2)
        
        H_combo = alpha * H1 + beta * H2
        V_combo = grassmann.tangent_projection(X, H_combo)
        
        expected = alpha * V1 + beta * V2
        assert np.allclose(V_combo, expected, atol=TOLERANCES['default'])
    
    def test_tangent_space_dimension(self, grassmann_factory):
        """Test that horizontal space has correct dimension."""
        n, p = 10, 3
        grassmann = grassmann_factory(n, p)
        X = grassmann.random_point()
        
        # Generate random horizontal vectors
        horizontal_vectors = []
        expected_dim = p * (n - p)
        
        # Generate basis for horizontal space
        for i in range(n):
            for j in range(p):
                if i < n:  # All indices valid
                    E_ij = np.zeros((n, p))
                    E_ij[i, j] = 1.0
                    V = grassmann.tangent_projection(X, E_ij)
                    if np.linalg.norm(V, 'fro') > TOLERANCES['default']:
                        horizontal_vectors.append(V.flatten())
        
        # Check dimension of span
        if len(horizontal_vectors) >= expected_dim:
            horizontal_matrix = np.array(horizontal_vectors[:expected_dim + 5]).T
            rank = np.linalg.matrix_rank(horizontal_matrix, tol=TOLERANCES['numerical'])
            assert rank == expected_dim


class TestGrassmannRetraction:
    """Test retraction operations on Grassmann manifold."""
    
    def test_retraction_at_zero(self, grassmann_factory, assert_helpers):
        """Test that retracting zero returns equivalent point."""
        grassmann = grassmann_factory(10, 3)
        X = grassmann.random_point()
        zero = np.zeros((10, 3))
        
        Y = grassmann.retract(X, zero)
        
        # Y should represent same subspace as X
        Q, _, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
        assert rank == 3
        assert np.allclose(Y, X @ Q, atol=TOLERANCES['strict'])
        assert_helpers.assert_is_orthogonal(Y)
    
    @pytest.mark.parametrize("n,p", MATRIX_DIMENSION_CONFIGS['small'])
    def test_retraction_preserves_manifold(self, grassmann_factory, assert_helpers, n, p):
        """Test that retraction returns valid points on Grassmann."""
        grassmann = grassmann_factory(n, p)
        X = grassmann.random_point()
        
        for scale in [0.001, 0.01, 0.1, 1.0, 10.0]:
            H = np.random.randn(n, p) * scale
            V = grassmann.tangent_projection(X, H)
            Y = grassmann.retract(X, V)
            assert_helpers.assert_is_orthogonal(Y)


class TestGrassmannRandomGeneration:
    """Test random point generation on Grassmann manifold."""
    
    @pytest.mark.parametrize("n,p", MATRIX_DIMENSION_CONFIGS['small'])
    def test_random_points_orthonormal(self, grassmann_factory, assert_helpers, n, p):
        """Test that random points are valid orthonormal bases."""
        grassmann = grassmann_factory(n, p)
        
        for _ in range(20):
            X = grassmann.random_point()
            assert X.shape == (n, p)
            assert_helpers.assert_is_orthogonal(X)
    
    def test_random_points_diverse_subspaces(self, grassmann_factory):
        """Test that random points represent diverse subspaces."""
        grassmann = grassmann_factory(10, 2)
        
        # Generate random points
        n_samples = 50
        points = [grassmann.random_point() for _ in range(n_samples)]
        
        # Check principal angles between subspaces
        min_principal_angles = []
        for i in range(min(20, n_samples)):
            angles = []
            for j in range(n_samples):
                if i != j:
                    # Compute principal angles via SVD of X_i^T X_j
                    M = points[i].T @ points[j]
                    singular_values = np.linalg.svd(M, compute_uv=False)
                    # Clamp to [-1, 1] for arccos
                    singular_values = np.clip(singular_values, -1, 1)
                    principal_angles = np.arccos(singular_values)
                    angles.append(np.min(principal_angles))
            if angles:
                min_principal_angles.append(min(angles))
        
        # Check that subspaces are diverse (not all aligned)
        assert np.mean(min_principal_angles) > 0.1  # Average minimum angle > 0.1 rad
        assert max(min_principal_angles) < np.pi/2 - 0.1  # Not all orthogonal


class TestGrassmannGeometry:
    """Test geometric operations on Grassmann manifold."""
    
    def test_grassmann_distance_properties(self, grassmann_factory):
        """Test properties of geodesic distance on Grassmann."""
        grassmann = grassmann_factory(10, 3)
        
        # Generate points
        X = grassmann.random_point()
        Y = grassmann.random_point()
        Z = grassmann.random_point()
        
        # Distance to self
        d_XX = grassmann.distance(X, X)
        assert abs(d_XX) < TOLERANCES['strict']
        
        # Symmetry
        d_XY = grassmann.distance(X, Y)
        d_YX = grassmann.distance(Y, X)
        assert abs(d_XY - d_YX) < TOLERANCES['default']
        
        # Triangle inequality
        d_XZ = grassmann.distance(X, Z)
        d_YZ = grassmann.distance(Y, Z)
        assert d_XZ <= d_XY + d_YZ + TOLERANCES['default']
        
        # Distance bounded by π√p
        assert d_XY <= np.pi * np.sqrt(3) + TOLERANCES['default']
    
    def test_principal_angles(self, grassmann_factory):
        """Test computation of principal angles between subspaces."""
        grassmann = grassmann_factory(10, 3)
        
        # Create two subspaces with known principal angles
        # First subspace: span of first 3 standard basis vectors
        X = np.zeros((10, 3))
        X[:3, :] = np.eye(3)
        
        # Second subspace: rotated version
        theta = np.pi / 4  # 45 degrees
        Y = np.zeros((10, 3))
        Y[0, 0] = np.cos(theta)
        Y[3, 0] = np.sin(theta)
        Y[1, 1] = 1.0
        Y[2, 2] = 1.0
        
        # Principal angles should be [π/4, 0, 0]
        M = X.T @ Y
        singular_values = np.linalg.svd(M, compute_uv=False)
        principal_angles = np.arccos(np.clip(singular_values, -1, 1))
        
        expected_angles = np.array([theta, 0.0, 0.0])
        assert np.allclose(np.sort(principal_angles), np.sort(expected_angles), 
                          atol=TOLERANCES['default'])


class TestGrassmannOptimizationProperties:
    """Test properties relevant for optimization on Grassmann."""
    
    def test_gradient_conversion(self, grassmann_factory, assert_helpers):
        """Test conversion from Euclidean to Riemannian gradient."""
        grassmann = grassmann_factory(10, 3)
        X = grassmann.random_point()
        
        # Euclidean gradient
        G_eucl = np.random.randn(10, 3)
        
        # Convert to Riemannian gradient (should be horizontal)
        G_riem = grassmann.euclidean_to_riemannian_gradient(X, G_eucl)
        
        assert_helpers.assert_in_tangent_space_grassmann(X, G_riem)
        
        # Should be same as tangent projection
        G_proj = grassmann.tangent_projection(X, G_eucl)
        assert np.allclose(G_riem, G_proj, atol=TOLERANCES['default'])
    
    def test_cost_function_invariance(self, grassmann_factory):
        """Test that Grassmann cost functions are invariant to basis change."""
        grassmann = grassmann_factory(10, 3)
        X = grassmann.random_point()
        
        # Define a cost function that depends only on subspace
        C = np.random.randn(10, 10)
        C = C + C.T  # Symmetric
        
        def cost_fn(Y):
            # Trace of projection of C onto subspace
            return np.trace(Y @ Y.T @ C)
        
        # Cost at X
        cost_X = cost_fn(X)
        
        # Transform basis by orthogonal matrix
        Q = np.linalg.qr(np.random.randn(3, 3))[0]
        X_transformed = X @ Q
        
        # Cost should be same
        cost_X_transformed = cost_fn(X_transformed)
        assert abs(cost_X - cost_X_transformed) < TOLERANCES['default']


class TestGrassmannRelationshipToStiefel:
    """Test relationship between Grassmann and Stiefel manifolds."""
    
    def test_grassmann_as_quotient_of_stiefel(self, grassmann_factory, stiefel_factory):
        """Test that Grassmann is quotient of Stiefel by O(p)."""
        n, p = 10, 3
        grassmann = grassmann_factory(n, p)
        stiefel = stiefel_factory(n, p)
        
        # Point on Stiefel
        X_stiefel = stiefel.random_point()
        
        # Different representatives of same Grassmann point
        representatives = []
        for _ in range(5):
            Q = np.linalg.qr(np.random.randn(p, p))[0]
            representatives.append(X_stiefel @ Q)
        
        # All should project to equivalent Grassmann points
        grassmann_points = [grassmann.project(rep) for rep in representatives]
        
        # Check all represent same subspace
        X0 = grassmann_points[0]
        for Xi in grassmann_points[1:]:
            # Xi = X0 @ Qi for some orthogonal Qi
            Qi, _, rank, _ = np.linalg.lstsq(X0, Xi, rcond=None)
            assert rank == p
            assert np.allclose(Xi, X0 @ Qi, atol=TOLERANCES['default'])
            assert np.allclose(Qi @ Qi.T, np.eye(p), atol=TOLERANCES['default'])
    
    def test_horizontal_lift_from_grassmann_to_stiefel(self, grassmann_factory, 
                                                      stiefel_factory, assert_helpers):
        """Test horizontal lift of tangent vectors from Grassmann to Stiefel."""
        n, p = 10, 3
        grassmann = grassmann_factory(n, p)
        stiefel = stiefel_factory(n, p)
        
        # Point on both manifolds (same representative)
        X = grassmann.random_point()
        
        # Horizontal vector on Grassmann
        V_grass = grassmann.tangent_projection(X, np.random.randn(n, p))
        
        # This is also tangent to Stiefel
        assert_helpers.assert_in_tangent_space_stiefel(X, V_grass)
        
        # And it's horizontal (X^T V = 0)
        assert np.allclose(X.T @ V_grass, np.zeros((p, p)), atol=TOLERANCES['default'])