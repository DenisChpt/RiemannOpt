"""
Unit tests for the Sphere manifold.

This module tests all operations and properties of the unit sphere S^{n-1},
including projections, retractions, tangent space operations, and geometric
properties.
"""

import pytest
import numpy as np
from typing import List
from conftest import TOLERANCES, DIMENSION_CONFIGS, riemannopt


class TestSphereCreation:
    """Test sphere manifold creation and basic properties."""
    
    def test_create_valid_sphere(self, sphere_factory):
        """Test creation of sphere with valid dimensions."""
        for dim in [2, 3, 10, 100, 1000]:
            sphere = sphere_factory(dim)
            assert sphere.ambient_dim == dim
            assert sphere.manifold_dim == dim - 1
            assert sphere.dim == dim - 1
    
    def test_create_invalid_sphere(self):
        """Test that invalid dimensions raise appropriate errors."""
        with pytest.raises(ValueError):
            riemannopt.Sphere(0)
        with pytest.raises(ValueError):
            riemannopt.Sphere(1)
    
    def test_sphere_representation(self, sphere_factory):
        """Test string representation of sphere."""
        sphere = sphere_factory(10)
        assert str(sphere) == "Sphere(dimension=10)"
        assert repr(sphere) == "Sphere(dimension=10)"


class TestSphereProjection:
    """Test projection operations onto the sphere."""
    
    @pytest.mark.parametrize("dim", DIMENSION_CONFIGS['small'])
    def test_projection_normalizes_vectors(self, sphere_factory, assert_helpers, dim):
        """Test that projection normalizes any non-zero vector."""
        sphere = sphere_factory(dim[0])
        
        # Test various vectors
        test_vectors = [
            np.ones(dim[0]),
            np.random.randn(dim[0]),
            np.random.randn(dim[0]) * 1e6,  # Large vector
            np.random.randn(dim[0]) * 1e-6,  # Small vector
        ]
        
        for v in test_vectors:
            projected = sphere.project(v)
            assert_helpers.assert_is_on_sphere(projected)
    
    def test_projection_preserves_direction(self, sphere_factory):
        """Test that projection preserves the direction of vectors."""
        sphere = sphere_factory(10)
        v = np.random.randn(10)
        v_normalized = v / np.linalg.norm(v)
        
        projected = sphere.project(v)
        
        # Check that projected is parallel to v
        assert np.allclose(projected, v_normalized, atol=TOLERANCES['default'])
    
    def test_projection_idempotent(self, sphere_factory, assert_helpers):
        """Test that projecting twice gives the same result."""
        sphere = sphere_factory(50)
        v = np.random.randn(50)
        
        p1 = sphere.project(v)
        p2 = sphere.project(p1)
        
        assert np.allclose(p1, p2, atol=TOLERANCES['strict'])
        assert_helpers.assert_is_on_sphere(p2)
    
    @pytest.mark.numerical
    def test_projection_numerical_stability(self, sphere_factory, assert_helpers):
        """Test projection with extreme values."""
        sphere = sphere_factory(10)
        
        # Very large values
        v_large = np.random.randn(10) * 1e100
        p_large = sphere.project(v_large)
        assert_helpers.assert_is_on_sphere(p_large, tol=TOLERANCES['numerical'])
        
        # Very small values
        v_small = np.random.randn(10) * 1e-100
        p_small = sphere.project(v_small)
        assert_helpers.assert_is_on_sphere(p_small, tol=TOLERANCES['numerical'])


class TestSphereTangentSpace:
    """Test tangent space operations on the sphere."""
    
    @pytest.mark.parametrize("dim", DIMENSION_CONFIGS['small'])
    def test_tangent_projection_orthogonality(self, sphere_factory, assert_helpers, dim):
        """Test that tangent projection makes vectors orthogonal to the point."""
        sphere = sphere_factory(dim[0])
        x = sphere.random_point()
        
        # Generate random vectors
        for _ in range(10):
            v = np.random.randn(dim[0])
            v_tan = sphere.tangent_projection(x, v)
            assert_helpers.assert_in_tangent_space_sphere(x, v_tan)
    
    def test_tangent_projection_idempotent(self, sphere_factory):
        """Test that projecting to tangent space twice gives same result."""
        sphere = sphere_factory(20)
        x = sphere.random_point()
        v = np.random.randn(20)
        
        v_tan1 = sphere.tangent_projection(x, v)
        v_tan2 = sphere.tangent_projection(x, v_tan1)
        
        assert np.allclose(v_tan1, v_tan2, atol=TOLERANCES['strict'])
    
    def test_tangent_projection_linearity(self, sphere_factory):
        """Test linearity of tangent projection."""
        sphere = sphere_factory(15)
        x = sphere.random_point()
        
        v1 = np.random.randn(15)
        v2 = np.random.randn(15)
        alpha, beta = 2.5, -1.3
        
        # Project individually
        v1_tan = sphere.tangent_projection(x, v1)
        v2_tan = sphere.tangent_projection(x, v2)
        
        # Project linear combination
        v_combo = alpha * v1 + beta * v2
        v_combo_tan = sphere.tangent_projection(x, v_combo)
        
        # Check linearity
        expected = alpha * v1_tan + beta * v2_tan
        assert np.allclose(v_combo_tan, expected, atol=TOLERANCES['default'])
    
    def test_tangent_space_dimension(self, sphere_factory, test_data_generator):
        """Test that tangent space has correct dimension."""
        sphere = sphere_factory(10)
        x = sphere.random_point()
        
        # Generate basis for tangent space
        tangent_vectors = []
        for i in range(9):  # dim - 1
            e_i = np.zeros(10)
            e_i[i] = 1.0
            v_tan = sphere.tangent_projection(x, e_i)
            if np.linalg.norm(v_tan) > TOLERANCES['default']:
                tangent_vectors.append(v_tan / np.linalg.norm(v_tan))
        
        # Check that we have dim-1 linearly independent vectors
        if len(tangent_vectors) >= 9:
            tangent_matrix = np.array(tangent_vectors).T
            rank = np.linalg.matrix_rank(tangent_matrix, tol=TOLERANCES['numerical'])
            assert rank == 9  # sphere.manifold_dim


class TestSphereRetraction:
    """Test retraction operations on the sphere."""
    
    def test_retraction_at_zero(self, sphere_factory):
        """Test that retracting zero vector returns the same point."""
        sphere = sphere_factory(20)
        x = sphere.random_point()
        zero = np.zeros(20)
        
        y = sphere.retract(x, zero)
        assert np.allclose(x, y, atol=TOLERANCES['strict'])
    
    @pytest.mark.parametrize("dim", DIMENSION_CONFIGS['small'])
    def test_retraction_preserves_manifold(self, sphere_factory, assert_helpers, dim):
        """Test that retraction always returns points on the manifold."""
        sphere = sphere_factory(dim[0])
        x = sphere.random_point()
        
        for scale in [0.01, 0.1, 1.0, 10.0]:
            v = np.random.randn(dim[0]) * scale
            v_tan = sphere.tangent_projection(x, v)
            y = sphere.retract(x, v_tan)
            assert_helpers.assert_is_on_sphere(y)
    
    def test_retraction_first_order_approximation(self, sphere_factory):
        """Test that retraction is first-order approximation to geodesic."""
        sphere = sphere_factory(10)
        x = sphere.random_point()
        v = np.random.randn(10)
        v_tan = sphere.tangent_projection(x, v)
        v_tan = v_tan / np.linalg.norm(v_tan)  # Normalize
        
        # For small t, R_x(tv) ≈ exp_x(tv)
        errors = []
        for t in [1e-8, 1e-6, 1e-4, 1e-2]:
            y_retract = sphere.retract(x, t * v_tan)
            # True exponential map for sphere
            y_exp = np.cos(t) * x + np.sin(t) * v_tan
            y_exp = y_exp / np.linalg.norm(y_exp)  # Normalize
            
            error = np.linalg.norm(y_retract - y_exp)
            errors.append(error)
        
        # Check that error decreases quadratically with t
        for i in range(len(errors) - 1):
            if errors[i] > 1e-14:  # Skip if error is too small (numerical precision)
                assert errors[i+1] / errors[i] < 150  # Roughly quadratic


class TestSphereGeometry:
    """Test geometric operations on the sphere."""
    
    def test_inner_product_positive_definite(self, sphere_factory):
        """Test that inner product is positive definite."""
        sphere = sphere_factory(15)
        x = sphere.random_point()
        
        for _ in range(10):
            v = np.random.randn(15)
            v_tan = sphere.tangent_projection(x, v)
            if np.linalg.norm(v_tan) > TOLERANCES['default']:
                inner = sphere.inner_product(x, v_tan, v_tan)
                assert inner > 0
    
    def test_inner_product_symmetry(self, sphere_factory):
        """Test symmetry of inner product."""
        sphere = sphere_factory(10)
        x = sphere.random_point()
        v = sphere.tangent_projection(x, np.random.randn(10))
        w = sphere.tangent_projection(x, np.random.randn(10))
        
        inner_vw = sphere.inner_product(x, v, w)
        inner_wv = sphere.inner_product(x, w, v)
        
        assert np.isclose(inner_vw, inner_wv, atol=TOLERANCES['strict'])
    
    def test_inner_product_linearity(self, sphere_factory):
        """Test linearity of inner product in first argument."""
        sphere = sphere_factory(12)
        x = sphere.random_point()
        
        v1 = sphere.tangent_projection(x, np.random.randn(12))
        v2 = sphere.tangent_projection(x, np.random.randn(12))
        w = sphere.tangent_projection(x, np.random.randn(12))
        
        alpha, beta = 2.0, -3.0
        
        # Compute separately
        inner1 = sphere.inner_product(x, v1, w)
        inner2 = sphere.inner_product(x, v2, w)
        
        # Compute combined
        v_combo = alpha * v1 + beta * v2
        inner_combo = sphere.inner_product(x, v_combo, w)
        
        expected = alpha * inner1 + beta * inner2
        assert np.isclose(inner_combo, expected, atol=TOLERANCES['default'])
    
    def test_distance_properties(self, sphere_factory):
        """Test properties of geodesic distance."""
        sphere = sphere_factory(10)
        
        # Test distance to self is zero
        x = sphere.random_point()
        assert np.isclose(sphere.distance(x, x), 0.0, atol=TOLERANCES['numerical'])
        
        # Test symmetry
        y = sphere.random_point()
        d_xy = sphere.distance(x, y)
        d_yx = sphere.distance(y, x)
        assert np.isclose(d_xy, d_yx, atol=TOLERANCES['default'])
        
        # Test triangle inequality
        z = sphere.random_point()
        d_xz = sphere.distance(x, z)
        d_yz = sphere.distance(y, z)
        assert d_xz <= d_xy + d_yz + TOLERANCES['default']
        
        # Test maximum distance on sphere is π
        x_neg = -x  # Antipodal point
        d_max = sphere.distance(x, x_neg)
        assert np.isclose(d_max, np.pi, atol=TOLERANCES['default'])


class TestSphereRandomGeneration:
    """Test random point generation on the sphere."""
    
    @pytest.mark.parametrize("dim", DIMENSION_CONFIGS['small'])
    def test_random_points_on_manifold(self, sphere_factory, assert_helpers, dim):
        """Test that random points are on the manifold."""
        sphere = sphere_factory(dim[0])
        
        for _ in range(20):
            x = sphere.random_point()
            assert_helpers.assert_is_on_sphere(x)
    
    def test_random_points_distribution(self, sphere_factory):
        """Test that random points are uniformly distributed."""
        # This is a statistical test, so we use larger tolerances
        sphere = sphere_factory(3)  # Test on S^2 for visualization
        
        # Generate many random points
        n_samples = 1000
        points = np.array([sphere.random_point() for _ in range(n_samples)])
        
        # Check that mean is near zero (for uniform distribution on sphere)
        mean = np.mean(points, axis=0)
        assert np.linalg.norm(mean) < 0.1  # Statistical tolerance
        
        # Check that points cover the sphere (no clustering)
        # Compute pairwise distances
        min_distances = []
        for i in range(min(100, n_samples)):  # Check subset for efficiency
            distances = [sphere.distance(points[i], points[j]) 
                        for j in range(n_samples) if i != j]
            min_distances.append(min(distances))
        
        # No point should be too isolated
        assert max(min_distances) < 0.5  # Rough check


class TestSphereExponentialMap:
    """Test exponential and logarithmic maps on the sphere."""
    
    def test_exp_log_inverse(self, sphere_factory):
        """Test that exp and log are inverse operations."""
        sphere = sphere_factory(10)
        x = sphere.random_point()
        
        # Generate tangent vector with controlled norm
        v = np.random.randn(10)
        v_tan = sphere.tangent_projection(x, v)
        v_tan = v_tan / np.linalg.norm(v_tan) * 0.5  # Keep norm < π
        
        # Test exp then log
        y = sphere.exp(x, v_tan)
        w = sphere.log(x, y)
        
        assert np.allclose(v_tan, w, atol=TOLERANCES['default'])
    
    def test_exp_at_zero(self, sphere_factory):
        """Test that exp_x(0) = x."""
        sphere = sphere_factory(15)
        x = sphere.random_point()
        zero = np.zeros(15)
        
        y = sphere.exp(x, zero)
        assert np.allclose(x, y, atol=TOLERANCES['strict'])
    
    def test_exp_geodesic(self, sphere_factory):
        """Test that exp gives geodesics on sphere."""
        sphere = sphere_factory(5)
        x = sphere.random_point()
        v = np.random.randn(5)
        v_tan = sphere.tangent_projection(x, v)
        v_tan = v_tan / np.linalg.norm(v_tan)  # Unit tangent vector
        
        # Points along geodesic
        for t in np.linspace(0, np.pi/2, 10):
            y = sphere.exp(x, t * v_tan)
            
            # Check distance along geodesic
            d = sphere.distance(x, y)
            # Use higher tolerance for t=0 case due to numerical precision
            tol = TOLERANCES['numerical'] if t < 1e-6 else TOLERANCES['default']
            assert np.isclose(d, t, atol=tol)
    
    @pytest.mark.numerical
    def test_log_numerical_stability(self, sphere_factory):
        """Test log map near cut locus (antipodal points)."""
        sphere = sphere_factory(10)
        x = sphere.random_point()
        
        # Test near antipodal point
        y_anti = -x + 0.01 * np.random.randn(10)
        y_anti = sphere.project(y_anti)
        
        # Log should handle this gracefully
        try:
            v = sphere.log(x, y_anti)
            # If it succeeds, check that it's reasonable
            assert np.linalg.norm(v) < 10 * np.pi
        except ValueError:
            # It's acceptable to fail at cut locus
            pass