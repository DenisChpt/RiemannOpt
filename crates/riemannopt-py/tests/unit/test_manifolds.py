"""
Comprehensive unit tests for all manifold wrappers.

This module tests the core functionality of each manifold implementation,
ensuring correctness, performance, and adherence to Riemannian geometry principles.
"""

import pytest
import numpy as np
import numpy.testing as npt
from typing import List, Tuple, Any

try:
    import riemannopt as ro
except ImportError:
    pytest.skip("riemannopt not available", allow_module_level=True)


class TestBaseManifoldsInterface:
    """Test the common interface of all manifolds."""
    
    @pytest.fixture(params=[
        ("Sphere", (10,)),
        ("Stiefel", (10, 3)),
        ("Grassmann", (10, 3)),
        ("SPD", (5,)),
        ("Hyperbolic", (5,)),
    ])
    def manifold_params(self, request):
        """Parametrized fixture for different manifolds."""
        manifold_name, args = request.param
        return manifold_name, args
    
    def test_manifold_creation(self, manifold_params):
        """Test that manifolds can be created with valid parameters."""
        manifold_name, args = manifold_params
        manifold_class = getattr(ro.manifolds, manifold_name)
        manifold = manifold_class(*args)
        
        # Check basic properties
        assert hasattr(manifold, 'dim'), f"{manifold_name} missing dim property"
        assert hasattr(manifold, 'ambient_dim'), f"{manifold_name} missing ambient_dim property"
        assert isinstance(manifold.dim, int), f"{manifold_name}.dim not int"
        assert isinstance(manifold.ambient_dim, int), f"{manifold_name}.ambient_dim not int"
        assert manifold.dim > 0, f"{manifold_name}.dim must be positive"
        assert manifold.ambient_dim > 0, f"{manifold_name}.ambient_dim must be positive"
    
    def test_manifold_invalid_parameters(self, manifold_params):
        """Test that manifolds reject invalid parameters."""
        manifold_name, args = manifold_params
        manifold_class = getattr(ro.manifolds, manifold_name)
        
        # Test with zero dimensions
        if len(args) == 1:
            with pytest.raises(ValueError):
                manifold_class(0)
        elif len(args) == 2:
            with pytest.raises(ValueError):
                manifold_class(0, args[1])
            with pytest.raises(ValueError):
                manifold_class(args[0], 0)
            # For Stiefel and Grassmann, p > n should fail
            if manifold_name in ["Stiefel", "Grassmann"]:
                with pytest.raises(ValueError):
                    manifold_class(3, 5)  # p > n
    
    def test_manifold_string_representation(self, manifold_params):
        """Test string representation is informative."""
        manifold_name, args = manifold_params
        manifold_class = getattr(ro.manifolds, manifold_name)
        manifold = manifold_class(*args)
        
        repr_str = repr(manifold)
        assert manifold_name in repr_str, f"Manifold name not in repr: {repr_str}"
        assert "ambient_dim" in repr_str, f"ambient_dim not in repr: {repr_str}"
        assert "intrinsic_dim" in repr_str, f"intrinsic_dim not in repr: {repr_str}"


class TestSphereManifold:
    """Comprehensive tests for the Sphere manifold."""
    
    @pytest.fixture
    def sphere(self):
        """Create a sphere manifold for testing."""
        return ro.manifolds.Sphere(10)
    
    @pytest.fixture
    def point_on_sphere(self, sphere):
        """Generate a valid point on the sphere."""
        return sphere.random_point()
    
    def test_sphere_properties(self, sphere):
        """Test sphere manifold properties."""
        assert sphere.dim == 9  # S^{n-1} has dimension n-1
        assert sphere.ambient_dim == 10
        assert sphere.n == 10
    
    def test_sphere_random_point(self, sphere):
        """Test random point generation."""
        point = sphere.random_point()
        
        # Check shape
        assert point.shape == (10,), f"Expected shape (10,), got {point.shape}"
        
        # Check unit norm
        norm = np.linalg.norm(point)
        npt.assert_allclose(norm, 1.0, atol=1e-10, err_msg="Random point not unit norm")
        
        # Check on manifold
        assert sphere.contains(point), "Random point not on sphere"
    
    def test_sphere_projection(self, sphere):
        """Test projection onto sphere."""
        # Random vector
        x = np.random.randn(10)
        projected = sphere.project(x)
        
        # Check unit norm
        norm = np.linalg.norm(projected)
        npt.assert_allclose(norm, 1.0, atol=1e-10, err_msg="Projected point not unit norm")
        
        # Check on manifold
        assert sphere.contains(projected), "Projected point not on sphere"
        
        # Check projection is in same direction (for non-zero input)
        if np.linalg.norm(x) > 1e-10:
            cosine = np.dot(x, projected) / np.linalg.norm(x)
            assert cosine > 0, "Projection changed direction"
    
    def test_sphere_tangent_space(self, sphere, point_on_sphere):
        """Test tangent space operations."""
        # Random tangent vector
        tangent = sphere.random_tangent(point_on_sphere)
        
        # Check shape
        assert tangent.shape == point_on_sphere.shape, "Tangent vector wrong shape"
        
        # Check orthogonality to point
        dot_product = np.dot(point_on_sphere, tangent)
        npt.assert_allclose(dot_product, 0.0, atol=1e-10, 
                           err_msg="Tangent vector not orthogonal to point")
        
        # Check tangent space membership
        assert sphere.is_tangent(point_on_sphere, tangent), "Vector not in tangent space"
        
        # Test tangent projection
        ambient_vector = np.random.randn(10)
        projected_tangent = sphere.project_tangent(point_on_sphere, ambient_vector)
        
        # Should be orthogonal to point
        dot_product = np.dot(point_on_sphere, projected_tangent)
        npt.assert_allclose(dot_product, 0.0, atol=1e-10,
                           err_msg="Projected tangent not orthogonal")
    
    def test_sphere_exponential_map(self, sphere, point_on_sphere):
        """Test exponential map."""
        tangent = sphere.random_tangent(point_on_sphere, scale=0.1)
        result = sphere.exp(point_on_sphere, tangent)
        
        # Check result is on manifold
        assert sphere.contains(result), "Exponential map result not on sphere"
        
        # Test zero tangent vector
        zero_tangent = np.zeros_like(point_on_sphere)
        result_zero = sphere.exp(point_on_sphere, zero_tangent)
        npt.assert_allclose(result_zero, point_on_sphere, atol=1e-10,
                           err_msg="exp(x, 0) ≠ x")
    
    def test_sphere_logarithmic_map(self, sphere, point_on_sphere):
        """Test logarithmic map."""
        other_point = sphere.random_point()
        tangent = sphere.log(point_on_sphere, other_point)
        
        # Check result is in tangent space
        assert sphere.is_tangent(point_on_sphere, tangent), "Log result not in tangent space"
        
        # Test log(x, x) = 0 (for points not antipodal)
        if np.dot(point_on_sphere, point_on_sphere) > -0.99:  # Not antipodal
            zero_tangent = sphere.log(point_on_sphere, point_on_sphere)
            npt.assert_allclose(zero_tangent, 0, atol=1e-10,
                               err_msg="log(x, x) ≠ 0")
    
    def test_sphere_retraction(self, sphere, point_on_sphere):
        """Test retraction mapping."""
        tangent = sphere.random_tangent(point_on_sphere, scale=0.1)
        result = sphere.retract(point_on_sphere, tangent)
        
        # Check result is on manifold
        assert sphere.contains(result), "Retraction result not on sphere"
        
        # Test zero tangent vector
        zero_tangent = np.zeros_like(point_on_sphere)
        result_zero = sphere.retract(point_on_sphere, zero_tangent)
        npt.assert_allclose(result_zero, point_on_sphere, atol=1e-10,
                           err_msg="retract(x, 0) ≠ x")
    
    def test_sphere_inner_product(self, sphere, point_on_sphere):
        """Test Riemannian inner product."""
        tangent1 = sphere.random_tangent(point_on_sphere)
        tangent2 = sphere.random_tangent(point_on_sphere)
        
        # Test symmetry
        inner1 = sphere.inner(point_on_sphere, tangent1, tangent2)
        inner2 = sphere.inner(point_on_sphere, tangent2, tangent1)
        npt.assert_allclose(inner1, inner2, atol=1e-10, err_msg="Inner product not symmetric")
        
        # Test positive definiteness
        inner_self = sphere.inner(point_on_sphere, tangent1, tangent1)
        assert inner_self >= 0, "Inner product not positive semidefinite"
        
        if np.linalg.norm(tangent1) > 1e-10:
            assert inner_self > 0, "Inner product not positive definite for non-zero vector"
        
        # Test linearity in first argument
        alpha = 2.5
        scaled_inner = sphere.inner(point_on_sphere, alpha * tangent1, tangent2)
        expected = alpha * inner1
        npt.assert_allclose(scaled_inner, expected, atol=1e-10,
                           err_msg="Inner product not linear in first argument")
    
    def test_sphere_norm_and_distance(self, sphere, point_on_sphere):
        """Test norm and distance computations."""
        tangent = sphere.random_tangent(point_on_sphere)
        
        # Test norm consistency with inner product
        norm = sphere.norm(point_on_sphere, tangent)
        inner_sqrt = np.sqrt(sphere.inner(point_on_sphere, tangent, tangent))
        npt.assert_allclose(norm, inner_sqrt, atol=1e-10,
                           err_msg="Norm inconsistent with inner product")
        
        # Test distance symmetry
        other_point = sphere.random_point()
        dist1 = sphere.distance(point_on_sphere, other_point)
        dist2 = sphere.distance(other_point, point_on_sphere)
        npt.assert_allclose(dist1, dist2, atol=1e-10, err_msg="Distance not symmetric")
        
        # Test distance positivity
        assert dist1 >= 0, "Distance not non-negative"
        
        # Test distance(x, x) = 0
        dist_self = sphere.distance(point_on_sphere, point_on_sphere)
        npt.assert_allclose(dist_self, 0, atol=1e-10, err_msg="distance(x, x) ≠ 0")


class TestStiefelManifold:
    """Comprehensive tests for the Stiefel manifold."""
    
    @pytest.fixture
    def stiefel(self):
        """Create a Stiefel manifold for testing."""
        return ro.manifolds.Stiefel(10, 3)
    
    @pytest.fixture
    def point_on_stiefel(self, stiefel):
        """Generate a valid point on the Stiefel manifold."""
        return stiefel.random_point()
    
    def test_stiefel_properties(self, stiefel):
        """Test Stiefel manifold properties."""
        assert stiefel.dim == 3 * (10 - 3) + 3 * (3 - 1) // 2  # np - p(p-1)/2
        assert stiefel.ambient_dim == 30  # n * p
        assert stiefel.n == 10
        assert stiefel.p == 3
    
    def test_stiefel_random_point(self, stiefel):
        """Test random point generation."""
        point = stiefel.random_point()
        
        # Check shape
        assert point.shape == (10, 3), f"Expected shape (10, 3), got {point.shape}"
        
        # Check orthonormality: X^T X = I
        XtX = point.T @ point
        I = np.eye(3)
        npt.assert_allclose(XtX, I, atol=1e-10, err_msg="Random point not orthonormal")
        
        # Check on manifold
        assert stiefel.contains(point), "Random point not on Stiefel manifold"
    
    def test_stiefel_projection(self, stiefel):
        """Test projection onto Stiefel manifold."""
        # Random matrix
        X = np.random.randn(10, 3)
        projected = stiefel.project(X)
        
        # Check orthonormality
        XtX = projected.T @ projected
        I = np.eye(3)
        npt.assert_allclose(XtX, I, atol=1e-10, err_msg="Projected point not orthonormal")
        
        # Check on manifold
        assert stiefel.contains(projected), "Projected point not on Stiefel manifold"
    
    def test_stiefel_tangent_space(self, stiefel, point_on_stiefel):
        """Test tangent space operations."""
        # Random tangent vector
        tangent = stiefel.random_tangent(point_on_stiefel)
        
        # Check shape
        assert tangent.shape == point_on_stiefel.shape, "Tangent vector wrong shape"
        
        # Check tangent space constraint: X^T V + V^T X = 0
        constraint = point_on_stiefel.T @ tangent + tangent.T @ point_on_stiefel
        npt.assert_allclose(constraint, 0, atol=1e-10,
                           err_msg="Tangent vector violates constraint")
        
        # Check tangent space membership
        assert stiefel.is_tangent(point_on_stiefel, tangent), "Vector not in tangent space"


class TestGrassmannManifold:
    """Comprehensive tests for the Grassmann manifold."""
    
    @pytest.fixture
    def grassmann(self):
        """Create a Grassmann manifold for testing."""
        return ro.manifolds.Grassmann(10, 3)
    
    @pytest.fixture
    def point_on_grassmann(self, grassmann):
        """Generate a valid point on the Grassmann manifold."""
        return grassmann.random_point()
    
    def test_grassmann_properties(self, grassmann):
        """Test Grassmann manifold properties."""
        assert grassmann.dim == 3 * (10 - 3)  # p * (n - p)
        assert grassmann.ambient_dim == 30  # n * p
        assert grassmann.n == 10
        assert grassmann.p == 3
    
    def test_grassmann_random_point(self, grassmann):
        """Test random point generation."""
        point = grassmann.random_point()
        
        # Check shape
        assert point.shape == (10, 3), f"Expected shape (10, 3), got {point.shape}"
        
        # Check orthonormality: X^T X = I
        XtX = point.T @ point
        I = np.eye(3)
        npt.assert_allclose(XtX, I, atol=1e-10, err_msg="Random point not orthonormal")
        
        # Check on manifold
        assert grassmann.contains(point), "Random point not on Grassmann manifold"


class TestSPDManifold:
    """Comprehensive tests for the SPD manifold."""
    
    @pytest.fixture
    def spd(self):
        """Create an SPD manifold for testing."""
        return ro.manifolds.SPD(5)
    
    @pytest.fixture
    def point_on_spd(self, spd):
        """Generate a valid point on the SPD manifold."""
        return spd.random_point()
    
    def test_spd_properties(self, spd):
        """Test SPD manifold properties."""
        assert spd.dim == 5 * (5 + 1) // 2  # n(n+1)/2
        assert spd.ambient_dim == 25  # n * n
        assert spd.n == 5
    
    def test_spd_random_point(self, spd):
        """Test random point generation."""
        point = spd.random_point()
        
        # Check shape
        assert point.shape == (5, 5), f"Expected shape (5, 5), got {point.shape}"
        
        # Check symmetry
        npt.assert_allclose(point, point.T, atol=1e-10, err_msg="Random point not symmetric")
        
        # Check positive definiteness
        eigenvals = np.linalg.eigvals(point)
        assert np.all(eigenvals > 0), "Random point not positive definite"
        
        # Check on manifold
        assert spd.contains(point), "Random point not on SPD manifold"
    
    def test_spd_tangent_space(self, spd, point_on_spd):
        """Test tangent space operations."""
        # Random tangent vector
        tangent = spd.random_tangent(point_on_spd)
        
        # Check shape
        assert tangent.shape == point_on_spd.shape, "Tangent vector wrong shape"
        
        # Check symmetry (tangent space of SPD is symmetric matrices)
        npt.assert_allclose(tangent, tangent.T, atol=1e-10,
                           err_msg="Tangent vector not symmetric")
        
        # Check tangent space membership
        assert spd.is_tangent(point_on_spd, tangent), "Vector not in tangent space"


class TestHyperbolicManifold:
    """Comprehensive tests for the Hyperbolic manifold."""
    
    @pytest.fixture
    def hyperbolic(self):
        """Create a Hyperbolic manifold for testing."""
        return ro.manifolds.Hyperbolic(5)
    
    @pytest.fixture
    def point_on_hyperbolic(self, hyperbolic):
        """Generate a valid point on the Hyperbolic manifold."""
        return hyperbolic.random_point()
    
    def test_hyperbolic_properties(self, hyperbolic):
        """Test Hyperbolic manifold properties."""
        assert hyperbolic.dim == 5  # Intrinsic dimension
        assert hyperbolic.ambient_dim == 6  # n + 1
        assert hyperbolic.n == 5
        assert hyperbolic.curvature == -1.0
    
    def test_hyperbolic_random_point(self, hyperbolic):
        """Test random point generation."""
        point = hyperbolic.random_point()
        
        # Check shape
        assert point.shape == (6,), f"Expected shape (6,), got {point.shape}"
        
        # Check Minkowski constraint: -x[0]^2 + sum(x[1:]^2) = -1
        minkowski_norm = -point[0]**2 + np.sum(point[1:]**2)
        npt.assert_allclose(minkowski_norm, -1.0, atol=1e-10,
                           err_msg="Random point violates Minkowski constraint")
        
        # Check on manifold
        assert hyperbolic.contains(point), "Random point not on Hyperbolic manifold"
    
    def test_hyperbolic_tangent_space(self, hyperbolic, point_on_hyperbolic):
        """Test tangent space operations."""
        # Random tangent vector
        tangent = hyperbolic.random_tangent(point_on_hyperbolic)
        
        # Check shape
        assert tangent.shape == point_on_hyperbolic.shape, "Tangent vector wrong shape"
        
        # Check tangent constraint: <x, v>_L = 0 (Minkowski inner product)
        minkowski_inner = -point_on_hyperbolic[0] * tangent[0] + \
                         np.sum(point_on_hyperbolic[1:] * tangent[1:])
        npt.assert_allclose(minkowski_inner, 0.0, atol=1e-10,
                           err_msg="Tangent vector violates orthogonality constraint")
        
        # Check tangent space membership
        assert hyperbolic.is_tangent(point_on_hyperbolic, tangent), "Vector not in tangent space"
    
    def test_hyperbolic_minkowski_inner(self, hyperbolic):
        """Test Minkowski inner product."""
        x = hyperbolic.random_point()
        y = hyperbolic.random_point()
        
        # Test constraint satisfaction
        inner_x = hyperbolic.minkowski_inner(x, x)
        npt.assert_allclose(inner_x, 1.0, atol=1e-10,  # Note: divided by -curvature
                           err_msg="Minkowski self-inner product incorrect")


class TestProductManifold:
    """Comprehensive tests for the Product manifold."""
    
    @pytest.fixture
    def product_manifold(self):
        """Create a Product manifold for testing."""
        sphere = ro.manifolds.Sphere(5)
        stiefel = ro.manifolds.Stiefel(4, 2)
        return ro.manifolds.ProductManifold([sphere, stiefel])
    
    def test_product_properties(self, product_manifold):
        """Test Product manifold properties."""
        # Dimensions should be sum of component dimensions
        expected_dim = 4 + 6  # Sphere(5).dim + Stiefel(4,2).dim
        expected_ambient_dim = 5 + 8  # Sphere(5).ambient_dim + Stiefel(4,2).ambient_dim
        
        assert product_manifold.dim == expected_dim
        assert product_manifold.ambient_dim == expected_ambient_dim
        assert product_manifold.n_manifolds == 2
    
    def test_product_random_point(self, product_manifold):
        """Test random point generation."""
        point = product_manifold.random_point()
        
        # Should be a tuple
        assert isinstance(point, tuple), "Product manifold point should be tuple"
        assert len(point) == 2, "Product manifold point should have 2 components"
        
        # Check individual components
        sphere_point, stiefel_point = point
        assert sphere_point.shape == (5,), "Sphere component wrong shape"
        assert stiefel_point.shape == (4, 2), "Stiefel component wrong shape"
        
        # Check components are on their respective manifolds
        assert product_manifold.contains(point), "Product point not on manifold"


# Integration tests
class TestManifoldIntegration:
    """Integration tests combining multiple manifold operations."""
    
    def test_exp_log_inverse(self):
        """Test that exp and log are inverse operations (where defined)."""
        sphere = ro.manifolds.Sphere(5)
        
        x = sphere.random_point()
        v = sphere.random_tangent(x, scale=0.1)  # Small tangent vector
        
        # exp followed by log should give back original vector (approximately)
        y = sphere.exp(x, v)
        v_recovered = sphere.log(x, y)
        
        npt.assert_allclose(v, v_recovered, atol=1e-8,
                           err_msg="exp and log not inverse")
    
    def test_retract_consistency(self):
        """Test retraction consistency across manifolds."""
        manifolds = [
            ro.manifolds.Sphere(5),
            ro.manifolds.Stiefel(6, 3),
            ro.manifolds.SPD(3),
        ]
        
        for manifold in manifolds:
            x = manifold.random_point()
            v = manifold.random_tangent(x, scale=0.01)
            
            # Retraction should give point on manifold
            y = manifold.retract(x, v)
            assert manifold.contains(y), f"Retraction failed for {type(manifold).__name__}"
    
    @pytest.mark.slow
    def test_geodesic_properties(self):
        """Test geodesic properties (slow test)."""
        sphere = ro.manifolds.Sphere(10)
        
        x = sphere.random_point()
        y = sphere.random_point()
        
        # Distance should be symmetric
        d1 = sphere.distance(x, y)
        d2 = sphere.distance(y, x)
        npt.assert_allclose(d1, d2, atol=1e-10, err_msg="Distance not symmetric")
        
        # Triangle inequality (approximate for geodesics)
        z = sphere.random_point()
        d_xz = sphere.distance(x, z)
        d_zy = sphere.distance(z, y)
        d_xy = sphere.distance(x, y)
        
        # Allow small tolerance for numerical errors
        assert d_xy <= d_xz + d_zy + 1e-10, "Triangle inequality violated"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])