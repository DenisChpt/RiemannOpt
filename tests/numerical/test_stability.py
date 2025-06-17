"""
Numerical stability tests for RiemannOpt.

This module tests numerical stability, precision, and edge cases
to ensure robust behavior in extreme conditions.
"""

import pytest
import numpy as np
from typing import Tuple, List
from conftest import riemannopt, TOLERANCES


class TestManifoldNumericalStability:
    """Test numerical stability of manifold operations."""
    
    def test_sphere_projection_near_zero(self, sphere_factory):
        """Test sphere projection with near-zero vectors."""
        sphere = sphere_factory(10)
        
        # Very small vectors
        small_vectors = [
            np.ones(10) * 1e-15,
            np.random.randn(10) * 1e-14,
            np.array([1e-16] + [0] * 9),
        ]
        
        for v in small_vectors:
            projected = sphere.project(v)
            # Should normalize to unit sphere
            assert abs(np.linalg.norm(projected) - 1.0) < TOLERANCES['strict']
            # Should not contain NaN or Inf
            assert np.all(np.isfinite(projected))
    
    def test_sphere_projection_large_vectors(self, sphere_factory):
        """Test sphere projection with very large vectors."""
        sphere = sphere_factory(20)
        
        # Very large vectors
        large_vectors = [
            np.ones(20) * 1e15,
            np.random.randn(20) * 1e20,
            np.full(20, 1e308),  # Near float64 max
        ]
        
        for v in large_vectors:
            projected = sphere.project(v)
            assert abs(np.linalg.norm(projected) - 1.0) < TOLERANCES['strict']
            assert np.all(np.isfinite(projected))
    
    def test_stiefel_orthogonalization_numerical(self, stiefel_factory, assert_helpers):
        """Test Stiefel orthogonalization with ill-conditioned matrices."""
        stiefel = stiefel_factory(50, 10)
        
        # Create nearly rank-deficient matrix
        U = np.random.randn(50, 10)
        U[:, 5] = U[:, 4] + 1e-10 * np.random.randn(50)  # Nearly dependent columns
        
        # Project should still produce orthogonal matrix
        X = stiefel.project(U)
        assert_helpers.assert_is_orthogonal(X, tol=TOLERANCES['numerical'])
    
    def test_grassmann_projection_stability(self, grassmann_factory):
        """Test Grassmann projection numerical stability."""
        grassmann = grassmann_factory(30, 5)
        
        # Matrix with very different column scales
        U = np.random.randn(30, 5)
        U[:, 0] *= 1e10
        U[:, 1] *= 1e-10
        
        X = grassmann.project(U)
        
        # Should be orthonormal
        gram = X.T @ X
        assert np.allclose(gram, np.eye(5), atol=TOLERANCES['numerical'])
    
    def test_manifold_distance_precision(self, sphere_factory):
        """Test distance computation precision."""
        sphere = sphere_factory(100)
        
        # Very close points
        x = sphere.random_point()
        epsilon = 1e-12
        v = sphere.random_tangent(x)
        v = v / np.linalg.norm(v) * epsilon
        y = sphere.retraction(x, v)
        
        # Distance should be approximately epsilon
        d = sphere.distance(x, y)
        assert abs(d - epsilon) < epsilon * 0.1
        
        # Same point should have zero distance
        assert sphere.distance(x, x) < TOLERANCES['strict']


class TestOptimizationNumericalStability:
    """Test numerical stability in optimization."""
    
    def test_sgd_gradient_overflow(self, sgd_factory, sphere_factory):
        """Test SGD with gradient overflow."""
        sphere = sphere_factory(20)
        sgd = sgd_factory(step_size=1e-10)  # Very small step size
        
        x = sphere.random_point()
        
        # Huge gradient
        huge_grad = np.random.randn(20) * 1e100
        
        # Should handle gracefully
        x_new = sgd.step(sphere, x, huge_grad)
        
        # Should remain on manifold and finite
        assert abs(np.linalg.norm(x_new) - 1.0) < TOLERANCES['numerical']
        assert np.all(np.isfinite(x_new))
    
    def test_adam_numerical_stability(self, adam_factory, stiefel_factory, assert_helpers):
        """Test Adam optimizer numerical stability."""
        stiefel = stiefel_factory(20, 5)
        adam = adam_factory(learning_rate=0.001, epsilon=1e-8)
        
        X = stiefel.random_point()
        
        # Alternating large and small gradients
        for i in range(20):
            if i % 2 == 0:
                grad = np.random.randn(20, 5) * 1e10
            else:
                grad = np.random.randn(20, 5) * 1e-10
            
            X = adam.step(stiefel, X, grad)
            
            # Should maintain orthogonality
            assert_helpers.assert_is_orthogonal(X, tol=TOLERANCES['numerical'])
    
    def test_optimization_accumulation_errors(self, sgd_factory, grassmann_factory):
        """Test error accumulation over many iterations."""
        grassmann = grassmann_factory(30, 5)
        sgd = sgd_factory(step_size=0.01)
        
        # Simple gradient
        A = np.random.randn(30, 30)
        
        X = grassmann.random_point()
        
        # Many iterations
        constraint_violations = []
        for i in range(1000):
            grad = A @ X
            X = sgd.step(grassmann, X, grad)
            
            # Check constraint periodically
            if i % 100 == 0:
                violation = np.linalg.norm(X.T @ X - np.eye(5), 'fro')
                constraint_violations.append(violation)
        
        # Errors should not accumulate significantly
        assert all(v < TOLERANCES['numerical'] for v in constraint_violations)
        # Should not be increasing
        assert constraint_violations[-1] < constraint_violations[0] * 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_sphere_dimension_one(self):
        """Test sphere with dimension 1 (just {-1, 1})."""
        sphere = riemannopt.Sphere(1)
        
        # Projection
        assert abs(sphere.project(np.array([5.0]))[0]) == pytest.approx(1.0)
        assert abs(sphere.project(np.array([-5.0]))[0]) == pytest.approx(1.0)
        
        # Distance
        x = np.array([1.0])
        y = np.array([-1.0])
        assert sphere.distance(x, y) == pytest.approx(np.pi)
    
    def test_stiefel_square_case(self, stiefel_factory, assert_helpers):
        """Test Stiefel when n=p (orthogonal group)."""
        stiefel = stiefel_factory(5, 5)
        
        # Random point should be orthogonal matrix
        O = stiefel.random_point()
        assert_helpers.assert_is_orthogonal(O)
        assert abs(np.linalg.det(O)) == pytest.approx(1.0, abs=TOLERANCES['default'])
        
        # Retraction should preserve orthogonality
        V = stiefel.random_tangent(O)
        O_new = stiefel.retraction(O, V)
        assert_helpers.assert_is_orthogonal(O_new)
    
    def test_grassmann_rank_one(self, grassmann_factory):
        """Test Grassmann manifold with rank 1 (projective space)."""
        grassmann = grassmann_factory(10, 1)
        
        # Points are just unit vectors (up to sign)
        x = grassmann.random_point()
        assert x.shape == (10, 1)
        assert abs(np.linalg.norm(x) - 1.0) < TOLERANCES['strict']
        
        # Opposite vectors represent same point
        y = -x
        d = grassmann.distance(x, y)
        assert d < TOLERANCES['strict']
    
    def test_extreme_aspect_ratios(self, stiefel_factory, assert_helpers):
        """Test Stiefel with extreme aspect ratios."""
        # Very tall and thin
        stiefel_tall = stiefel_factory(1000, 2)
        X_tall = stiefel_tall.random_point()
        assert_helpers.assert_is_orthogonal(X_tall)
        
        # Very short and wide
        stiefel_wide = stiefel_factory(5, 4)
        X_wide = stiefel_wide.random_point()
        assert_helpers.assert_is_orthogonal(X_wide)


class TestGradientChecking:
    """Test gradient computations numerically."""
    
    def test_finite_difference_gradient_check(self, sphere_factory):
        """Check gradients using finite differences."""
        sphere = sphere_factory(10)
        
        # Test function: f(x) = x^T A x
        A = np.random.randn(10, 10)
        A = A + A.T
        
        def cost_fn(x):
            return float(x.T @ A @ x)
        
        def grad_fn(x):
            return 2 * A @ x
        
        x = sphere.random_point()
        
        # Analytic gradient (projected)
        grad_analytic = sphere.tangent_projection(x, grad_fn(x))
        
        # Finite difference gradient
        eps = 1e-7
        grad_fd = np.zeros(10)
        
        for i in range(10):
            # Tangent direction
            e_i = np.zeros(10)
            e_i[i] = 1.0
            v = sphere.tangent_projection(x, e_i)
            
            if np.linalg.norm(v) > TOLERANCES['strict']:
                # Forward difference along geodesic
                x_plus = sphere.retraction(x, eps * v)
                f_plus = cost_fn(x_plus)
                f_0 = cost_fn(x)
                
                directional_deriv = (f_plus - f_0) / eps
                grad_fd += directional_deriv * v
        
        # Compare
        relative_error = np.linalg.norm(grad_analytic - grad_fd) / np.linalg.norm(grad_analytic)
        assert relative_error < 1e-5
    
    def test_hessian_vector_product_check(self, stiefel_factory):
        """Check Hessian-vector products numerically."""
        stiefel = stiefel_factory(10, 3)
        
        # Test function
        C = np.random.randn(10, 10)
        C = C + C.T
        
        def cost_fn(X):
            return np.trace(X.T @ C @ X)
        
        def grad_fn(X):
            return 2 * C @ X
        
        X = stiefel.random_point()
        V = stiefel.random_tangent(X)
        
        # Hessian-vector product by finite differences
        eps = 1e-8
        grad_0 = grad_fn(X)
        X_plus = stiefel.retraction(X, eps * V)
        grad_plus = grad_fn(X_plus)
        
        # Project gradients to tangent space
        grad_0_proj = stiefel.tangent_projection(X, grad_0)
        grad_plus_proj = stiefel.tangent_projection(X_plus, grad_plus)
        
        # Transport grad_plus back to T_X M
        # For Stiefel, we can use parallel transport approximation
        grad_plus_transported = stiefel.tangent_projection(X, grad_plus_proj)
        
        hess_v_fd = (grad_plus_transported - grad_0_proj) / eps
        
        # For this quadratic function, Hessian is simple
        hess_v_exact = stiefel.tangent_projection(X, 2 * C @ V)
        
        relative_error = np.linalg.norm(hess_v_fd - hess_v_exact, 'fro') / np.linalg.norm(hess_v_exact, 'fro')
        assert relative_error < 1e-3


class TestNumericalPrecision:
    """Test precision of numerical computations."""
    
    def test_inner_product_symmetry(self, sphere_factory):
        """Test that inner product is symmetric."""
        sphere = sphere_factory(50)
        
        x = sphere.random_point()
        u = sphere.random_tangent(x)
        v = sphere.random_tangent(x)
        
        ip_uv = sphere.inner_product(x, u, v)
        ip_vu = sphere.inner_product(x, v, u)
        
        assert abs(ip_uv - ip_vu) < TOLERANCES['strict']
    
    def test_retraction_second_order(self, stiefel_factory):
        """Test that retraction is second-order accurate."""
        stiefel = stiefel_factory(20, 5)
        
        X = stiefel.random_point()
        V = stiefel.random_tangent(X)
        
        # Check R(X, tV) = X + tV + O(t^2)
        errors = []
        for t in [1e-2, 1e-3, 1e-4, 1e-5]:
            Y = stiefel.retraction(X, t * V)
            linear_approx = X + t * V
            error = np.linalg.norm(Y - linear_approx, 'fro')
            errors.append(error)
        
        # Error should scale as t^2
        ratios = [errors[i] / errors[i+1] for i in range(len(errors)-1)]
        # Each ratio should be approximately 100 (since t changes by factor of 10)
        assert all(50 < ratio < 150 for ratio in ratios)
    
    def test_distance_metric_properties(self, grassmann_factory):
        """Test that distance satisfies metric properties."""
        grassmann = grassmann_factory(10, 3)
        
        # Generate points
        X = grassmann.random_point()
        Y = grassmann.random_point()
        Z = grassmann.random_point()
        
        # Identity
        assert grassmann.distance(X, X) < TOLERANCES['strict']
        
        # Symmetry
        d_xy = grassmann.distance(X, Y)
        d_yx = grassmann.distance(Y, X)
        assert abs(d_xy - d_yx) < TOLERANCES['strict']
        
        # Triangle inequality
        d_xz = grassmann.distance(X, Z)
        d_yz = grassmann.distance(Y, Z)
        assert d_xz <= d_xy + d_yz + TOLERANCES['relaxed']


class TestCatastrophicCancellation:
    """Test for catastrophic cancellation issues."""
    
    def test_sphere_tangent_projection_cancellation(self, sphere_factory):
        """Test tangent projection when vector is nearly radial."""
        sphere = sphere_factory(10)
        
        x = sphere.random_point()
        
        # Vector nearly parallel to x
        v = x + 1e-10 * np.random.randn(10)
        
        # Project to tangent space
        v_tan = sphere.tangent_projection(x, v)
        
        # Should be orthogonal to x
        assert abs(np.dot(x, v_tan)) < TOLERANCES['strict']
        
        # Should not have precision loss
        assert np.linalg.norm(v_tan) > 0
    
    def test_stiefel_gram_schmidt_cancellation(self, stiefel_factory):
        """Test Gram-Schmidt with nearly dependent vectors."""
        stiefel = stiefel_factory(50, 5)
        
        # Create matrix with nearly dependent columns
        Q, _ = np.linalg.qr(np.random.randn(50, 5))
        A = Q.copy()
        # Make columns 3 and 4 nearly parallel
        A[:, 4] = A[:, 3] + 1e-12 * np.random.randn(50)
        
        # Project should handle this
        X = stiefel.project(A)
        
        # Check orthogonality is preserved
        gram = X.T @ X
        off_diagonal_max = np.max(np.abs(gram - np.diag(np.diag(gram))))
        assert off_diagonal_max < TOLERANCES['numerical']


class TestSpecialFunctions:
    """Test special mathematical functions for numerical issues."""
    
    @pytest.mark.skip(reason="Exponential map not yet implemented")
    def test_exponential_map_accuracy(self, sphere_factory):
        """Test exponential map numerical accuracy."""
        sphere = sphere_factory(10)
        
        x = sphere.random_point()
        
        # Small tangent vector
        v = sphere.random_tangent(x) * 1e-10
        
        # Exp map should be accurate for small vectors
        y = sphere.exp(x, v)
        
        # Should be close to x + v (first order)
        linear_approx = sphere.project(x + v)
        assert sphere.distance(y, linear_approx) < 1e-18
    
    @pytest.mark.skip(reason="Logarithm map not yet implemented")
    def test_logarithm_map_inverse(self, stiefel_factory):
        """Test that log is inverse of exp."""
        stiefel = stiefel_factory(10, 3)
        
        X = stiefel.random_point()
        V = stiefel.random_tangent(X) * 0.1  # Small to ensure we're in normal neighborhood
        
        # exp then log should recover V
        Y = stiefel.exp(X, V)
        V_recovered = stiefel.log(X, Y)
        
        assert np.linalg.norm(V - V_recovered, 'fro') < TOLERANCES['strict']


class TestNumericalOverflow:
    """Test handling of numerical overflow conditions."""
    
    def test_cost_function_overflow(self, sgd_factory, sphere_factory):
        """Test optimization when cost function overflows."""
        sphere = sphere_factory(10)
        sgd = sgd_factory(step_size=1e-20)  # Tiny step size
        
        # Cost function that can overflow
        def cost_fn(x):
            return np.exp(1000 * x[0])  # Overflows for x[0] > ~0.7
        
        def grad_fn(x):
            grad = np.zeros_like(x)
            grad[0] = 1000 * np.exp(1000 * x[0])
            return grad
        
        # Start from safe point
        x = sphere.random_point()
        x[0] = 0.0
        x = sphere.project(x)
        
        # Take steps - should handle overflow gracefully
        for _ in range(10):
            try:
                grad = grad_fn(x)
                if not np.all(np.isfinite(grad)):
                    grad = np.zeros_like(x)  # Fallback
                x = sgd.step(sphere, x, grad)
            except OverflowError:
                # Should handle gracefully
                pass
        
        # Should still be on manifold
        assert abs(np.linalg.norm(x) - 1.0) < TOLERANCES['numerical']