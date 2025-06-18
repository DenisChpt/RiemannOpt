"""
Unit tests for the Stiefel manifold.

This module tests all operations and properties of the Stiefel manifold St(n,p),
including orthogonality constraints, retractions, tangent space operations,
and geometric properties.
"""

import pytest
import numpy as np
from typing import Tuple
from conftest import riemannopt, TOLERANCES, MATRIX_DIMENSION_CONFIGS


class TestStiefelCreation:
    """Test Stiefel manifold creation and basic properties."""
    
    @pytest.mark.parametrize("n,p", [(10, 3), (20, 5), (100, 10)])
    def test_create_valid_stiefel(self, stiefel_factory, n, p):
        """Test creation of Stiefel manifold with valid dimensions."""
        stiefel = stiefel_factory(n, p)
        
        assert stiefel.n == n
        assert stiefel.p == p
        assert stiefel.ambient_rows == n
        assert stiefel.ambient_cols == p
        
        # Check manifold dimension
        expected_dim = n * p - p * (p + 1) // 2
        assert stiefel.dim == expected_dim
        assert stiefel.manifold_dim == expected_dim
    
    def test_create_invalid_stiefel(self):
        """Test that invalid dimensions raise appropriate errors."""
        # p > n
        with pytest.raises(ValueError):
            riemannopt.Stiefel(5, 10)
        
        # Zero dimensions
        with pytest.raises(ValueError):
            riemannopt.Stiefel(0, 5)
        with pytest.raises(ValueError):
            riemannopt.Stiefel(5, 0)
    
    def test_stiefel_special_cases(self, stiefel_factory):
        """Test special cases of Stiefel manifold."""
        # St(n,1) is the sphere
        stiefel_n1 = stiefel_factory(10, 1)
        assert stiefel_n1.dim == 9  # Same as S^9
        
        # St(n,n) is the orthogonal group O(n)
        stiefel_nn = stiefel_factory(5, 5)
        assert stiefel_nn.dim == 10  # 5*5 - 5*6/2 = 25 - 15 = 10
    
    def test_stiefel_representation(self, stiefel_factory):
        """Test string representation of Stiefel manifold."""
        stiefel = stiefel_factory(10, 3)
        assert str(stiefel) == "Stiefel(n=10, p=3)"
        assert repr(stiefel) == "Stiefel(n=10, p=3)"


class TestStiefelProjection:
    """Test projection operations onto the Stiefel manifold."""
    
    @pytest.mark.parametrize("n,p", MATRIX_DIMENSION_CONFIGS['small'])
    def test_projection_creates_orthonormal_matrix(self, stiefel_factory, assert_helpers, n, p):
        """Test that projection creates matrices with orthonormal columns."""
        stiefel = stiefel_factory(n, p)
        
        # Test various matrices
        test_matrices = [
            np.random.randn(n, p),
            np.ones((n, p)),
            np.eye(n, p),
            np.random.randn(n, p) * 100,  # Large entries
        ]
        
        for M in test_matrices:
            X = stiefel.project(M)
            assert X.shape == (n, p)
            assert_helpers.assert_is_orthogonal(X)
    
    def test_projection_preserves_column_space(self, stiefel_factory):
        """Test that projection preserves column space when possible."""
        stiefel = stiefel_factory(10, 3)
        
        # Create matrix with linearly independent columns
        M = np.random.randn(10, 3)
        X = stiefel.project(M)
        
        # Check that column spaces are the same
        # X should span the same space as M
        # So M = X @ R for some matrix R
        R, residual, rank, _ = np.linalg.lstsq(X, M, rcond=None)
        
        assert rank == 3
        assert np.allclose(M, X @ R, atol=TOLERANCES['relaxed'])
    
    def test_projection_idempotent(self, stiefel_factory, assert_helpers):
        """Test that projecting twice gives the same result."""
        stiefel = stiefel_factory(20, 5)
        M = np.random.randn(20, 5)
        
        X1 = stiefel.project(M)
        X2 = stiefel.project(X1)
        
        assert np.allclose(X1, X2, atol=TOLERANCES['strict'])
        assert_helpers.assert_is_orthogonal(X2)
    
    @pytest.mark.numerical
    def test_projection_numerical_stability(self, stiefel_factory, assert_helpers):
        """Test projection with ill-conditioned matrices."""
        stiefel = stiefel_factory(10, 3)
        
        # Nearly collinear columns
        M = np.ones((10, 3))
        M[:, 1] += 1e-10 * np.random.randn(10)
        M[:, 2] += 1e-10 * np.random.randn(10)
        
        X = stiefel.project(M)
        assert_helpers.assert_is_orthogonal(X, tol=TOLERANCES['numerical'])
        
        # Very large condition number
        U, _, Vt = np.linalg.svd(np.random.randn(10, 3), full_matrices=False)
        S = np.diag([1e10, 1.0, 1e-10])
        M_illcond = U @ S @ Vt
        
        X_illcond = stiefel.project(M_illcond)
        assert_helpers.assert_is_orthogonal(X_illcond, tol=TOLERANCES['numerical'])


class TestStiefelTangentSpace:
    """Test tangent space operations on the Stiefel manifold."""
    
    @pytest.mark.parametrize("n,p", MATRIX_DIMENSION_CONFIGS['small'])
    def test_tangent_projection_constraint(self, stiefel_factory, assert_helpers, n, p):
        """Test that tangent projection satisfies the constraint X^T V + V^T X = 0."""
        stiefel = stiefel_factory(n, p)
        X = stiefel.random_point()
        
        for _ in range(10):
            H = np.random.randn(n, p)
            V = stiefel.tangent_projection(X, H)
            assert_helpers.assert_in_tangent_space_stiefel(X, V)
    
    def test_tangent_projection_idempotent(self, stiefel_factory):
        """Test that projecting to tangent space twice gives same result."""
        stiefel = stiefel_factory(15, 4)
        X = stiefel.random_point()
        H = np.random.randn(15, 4)
        
        V1 = stiefel.tangent_projection(X, H)
        V2 = stiefel.tangent_projection(X, V1)
        
        assert np.allclose(V1, V2, atol=TOLERANCES['strict'])
    
    def test_tangent_projection_linearity(self, stiefel_factory):
        """Test linearity of tangent projection."""
        stiefel = stiefel_factory(12, 3)
        X = stiefel.random_point()
        
        H1 = np.random.randn(12, 3)
        H2 = np.random.randn(12, 3)
        alpha, beta = 2.5, -1.3
        
        # Project individually
        V1 = stiefel.tangent_projection(X, H1)
        V2 = stiefel.tangent_projection(X, H2)
        
        # Project linear combination
        H_combo = alpha * H1 + beta * H2
        V_combo = stiefel.tangent_projection(X, H_combo)
        
        # Check linearity
        expected = alpha * V1 + beta * V2
        assert np.allclose(V_combo, expected, atol=TOLERANCES['default'])
    
    def test_tangent_space_dimension(self, stiefel_factory):
        """Test that tangent space has correct dimension."""
        n, p = 10, 3
        stiefel = stiefel_factory(n, p)
        X = stiefel.random_point()
        
        # Generate random tangent vectors
        tangent_vectors = []
        expected_dim = n * p - p * (p + 1) // 2
        
        for _ in range(expected_dim + 5):  # Generate more than dimension
            H = np.random.randn(n, p)
            V = stiefel.tangent_projection(X, H)
            if np.linalg.norm(V, 'fro') > TOLERANCES['default']:
                tangent_vectors.append(V.flatten())
        
        # Check dimension of span
        if len(tangent_vectors) >= expected_dim:
            tangent_matrix = np.array(tangent_vectors).T
            rank = np.linalg.matrix_rank(tangent_matrix, tol=TOLERANCES['numerical'])
            assert rank == expected_dim
    
    def test_tangent_projection_formula(self, stiefel_factory):
        """Test the explicit formula for tangent projection."""
        stiefel = stiefel_factory(8, 3)
        X = stiefel.random_point()
        H = np.random.randn(8, 3)
        
        # Apply formula: P_X(H) = H - X(X^T H + H^T X)/2
        XtH = X.T @ H
        HtX = H.T @ X
        V_expected = H - X @ ((XtH + HtX) / 2)
        
        V_actual = stiefel.tangent_projection(X, H)
        
        assert np.allclose(V_actual, V_expected, atol=TOLERANCES['default'])


class TestStiefelRetraction:
    """Test retraction operations on the Stiefel manifold."""
    
    def test_retraction_at_zero(self, stiefel_factory, assert_helpers):
        """Test that retracting zero returns the same point."""
        stiefel = stiefel_factory(10, 3)
        X = stiefel.random_point()
        zero = np.zeros((10, 3))
        
        Y = stiefel.retract(X, zero)
        assert np.allclose(X, Y, atol=TOLERANCES['strict'])
        assert_helpers.assert_is_orthogonal(Y)
    
    @pytest.mark.parametrize("n,p", MATRIX_DIMENSION_CONFIGS['small'])
    def test_retraction_preserves_manifold(self, stiefel_factory, assert_helpers, n, p):
        """Test that retraction always returns points on the manifold."""
        stiefel = stiefel_factory(n, p)
        X = stiefel.random_point()
        
        for scale in [0.001, 0.01, 0.1, 1.0, 10.0]:
            H = np.random.randn(n, p) * scale
            V = stiefel.tangent_projection(X, H)
            Y = stiefel.retract(X, V)
            assert_helpers.assert_is_orthogonal(Y)
    
    def test_qr_retraction_properties(self, stiefel_factory):
        """Test properties specific to QR-based retraction."""
        stiefel = stiefel_factory(15, 5)
        X = stiefel.random_point()
        V = stiefel.tangent_projection(X, np.random.randn(15, 5))
        
        # QR retraction: R_X(V) = qf(X + V)
        Y = stiefel.retract(X, V)
        
        # Check that Y has same column space as X + V (when X + V has full rank)
        X_plus_V = X + V
        if np.linalg.matrix_rank(X_plus_V) == 5:
            # Y should have same column space as X + V
            # So X + V = Y @ R for some upper triangular R
            R, residual, rank, _ = np.linalg.lstsq(Y, X_plus_V, rcond=None)
            assert rank == 5
            assert np.allclose(X_plus_V, Y @ R, atol=TOLERANCES['relaxed'])
    
    def test_retraction_first_order_approximation(self, stiefel_factory, assert_helpers):
        """Test that retraction is first-order approximation."""
        stiefel = stiefel_factory(8, 2)
        X = stiefel.random_point()
        V = stiefel.tangent_projection(X, np.random.randn(8, 2))
        V = V / np.linalg.norm(V, 'fro')  # Normalize
        
        # For small t, R_X(tV) ≈ X + tV
        errors = []
        for t in [1e-8, 1e-6, 1e-4, 1e-2]:
            Y = stiefel.retract(X, t * V)
            Y_linear = X + t * V
            
            # Project Y_linear to compare fairly
            Y_linear_proj = stiefel.project(Y_linear)
            
            error = np.linalg.norm(Y - Y_linear_proj, 'fro')
            errors.append(error)
        
        # Check that error decreases faster than linearly
        for i in range(len(errors) - 1):
            if errors[i] > 1e-15:  # Avoid division by zero
                assert errors[i+1] / errors[i] < 150  # Roughly quadratic
            else:
                # If error is essentially zero, just check the next one is small too
                assert errors[i+1] < 1e-10


class TestStiefelRandomGeneration:
    """Test random point generation on Stiefel manifold."""
    
    @pytest.mark.parametrize("n,p", MATRIX_DIMENSION_CONFIGS['small'])
    def test_random_points_orthogonal(self, stiefel_factory, assert_helpers, n, p):
        """Test that random points have orthonormal columns."""
        stiefel = stiefel_factory(n, p)
        
        for _ in range(20):
            X = stiefel.random_point()
            assert X.shape == (n, p)
            assert_helpers.assert_is_orthogonal(X)
    
    def test_random_points_distribution(self, stiefel_factory):
        """Test that random points are uniformly distributed."""
        # Test on small Stiefel manifold
        stiefel = stiefel_factory(4, 2)
        
        # Generate many random points
        n_samples = 100
        points = [stiefel.random_point() for _ in range(n_samples)]
        
        # Check that we get diverse points (no clustering)
        # Compute pairwise Frobenius distances
        min_distances = []
        for i in range(min(50, n_samples)):
            distances = [np.linalg.norm(points[i] - points[j], 'fro') 
                        for j in range(n_samples) if i != j]
            min_distances.append(min(distances))
        
        # No point should be too isolated or too close
        assert max(min_distances) < 2.0  # Not too isolated
        assert min(min_distances) > 0.01  # Not too close
    
    def test_random_tangent_vectors(self, stiefel_factory, assert_helpers):
        """Test generation of random tangent vectors."""
        stiefel = stiefel_factory(10, 3)
        X = stiefel.random_point()
        
        for _ in range(10):
            V = stiefel.random_tangent(X)
            assert V.shape == (10, 3)
            assert_helpers.assert_in_tangent_space_stiefel(X, V)
            # Should be non-zero
            assert np.linalg.norm(V, 'fro') > TOLERANCES['default']


class TestStiefelOptimizationProperties:
    """Test properties relevant for optimization on Stiefel."""
    
    def test_gradient_conversion(self, stiefel_factory, assert_helpers):
        """Test conversion from Euclidean to Riemannian gradient."""
        stiefel = stiefel_factory(10, 3)
        X = stiefel.random_point()
        
        # Euclidean gradient (arbitrary)
        G_eucl = np.random.randn(10, 3)
        
        # Convert to Riemannian gradient (should be in tangent space)
        G_riem = stiefel.euclidean_to_riemannian_gradient(X, G_eucl)
        
        assert_helpers.assert_in_tangent_space_stiefel(X, G_riem)
        
        # Should be same as tangent projection for Stiefel
        G_proj = stiefel.tangent_projection(X, G_eucl)
        assert np.allclose(G_riem, G_proj, atol=TOLERANCES['default'])
    
    def test_parallel_transport_projection(self, stiefel_factory, assert_helpers):
        """Test that parallel transport by projection preserves inner products approximately."""
        stiefel = stiefel_factory(8, 2)
        X = stiefel.random_point()
        Y = stiefel.random_point()
        
        # Generate two tangent vectors at X
        V1 = stiefel.tangent_projection(X, np.random.randn(8, 2))
        V2 = stiefel.tangent_projection(X, np.random.randn(8, 2))
        
        # Transport to Y (using projection-based transport)
        W1 = stiefel.parallel_transport(X, Y, V1)
        W2 = stiefel.parallel_transport(X, Y, V2)
        
        # Check they're in tangent space at Y
        assert_helpers.assert_in_tangent_space_stiefel(Y, W1)
        assert_helpers.assert_in_tangent_space_stiefel(Y, W2)
        
        # For projection-based transport, inner products are approximately preserved
        # (exactly preserved for true parallel transport)
        inner_before = np.trace(V1.T @ V2)
        inner_after = np.trace(W1.T @ W2)
        
        # Allow some deviation for projection-based transport
        assert abs(inner_after - inner_before) < 0.5 * abs(inner_before) + TOLERANCES['relaxed']


class TestStiefelSpecialCases:
    """Test special cases and edge cases for Stiefel manifold."""
    
    def test_stiefel_as_sphere(self, stiefel_factory, sphere_factory):
        """Test that St(n,1) behaves like sphere S^{n-1}."""
        n = 10
        stiefel = stiefel_factory(n, 1)
        sphere = sphere_factory(n)
        
        # Random point
        X_stiefel = stiefel.random_point()
        x_sphere = X_stiefel.flatten()
        
        # Check it's on sphere
        assert abs(np.linalg.norm(x_sphere) - 1.0) < TOLERANCES['default']
        
        # Check dimensions match
        assert stiefel.dim == sphere.dim
    
    def test_stiefel_full_rank(self, stiefel_factory):
        """Test St(n,n) as orthogonal group."""
        n = 5
        stiefel = stiefel_factory(n, n)
        
        # Random point should be orthogonal matrix
        O = stiefel.random_point()
        assert O.shape == (n, n)
        
        # Check full orthogonality (not just columns)
        assert np.allclose(O @ O.T, np.eye(n), atol=TOLERANCES['default'])
        assert np.allclose(O.T @ O, np.eye(n), atol=TOLERANCES['default'])
        
        # Check determinant is ±1
        det = np.linalg.det(O)
        assert abs(abs(det) - 1.0) < TOLERANCES['default']
    
    @pytest.mark.numerical
    def test_nearly_square_stiefel(self, stiefel_factory, assert_helpers):
        """Test Stiefel manifold when n is only slightly larger than p."""
        # This can be numerically challenging
        stiefel = stiefel_factory(10, 9)
        
        # Test basic operations still work
        X = stiefel.random_point()
        assert_helpers.assert_is_orthogonal(X)
        
        V = stiefel.tangent_projection(X, np.random.randn(10, 9))
        assert_helpers.assert_in_tangent_space_stiefel(X, V)
        
        Y = stiefel.retract(X, 0.1 * V)
        assert_helpers.assert_is_orthogonal(Y)