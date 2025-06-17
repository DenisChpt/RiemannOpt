"""
Unit tests for the Riemannian SGD optimizer.

This module tests the SGD optimizer with various configurations including
momentum, step size schedules, and convergence properties.
"""

import pytest
import numpy as np
from typing import Callable, Tuple
from conftest import riemannopt, TOLERANCES


class TestSGDCreation:
    """Test SGD optimizer creation and configuration."""
    
    def test_create_basic_sgd(self, sgd_factory):
        """Test creation of basic SGD optimizer."""
        sgd = sgd_factory(step_size=0.1)
        assert sgd is not None
        assert hasattr(sgd, 'step')
    
    def test_create_sgd_with_momentum(self, sgd_factory):
        """Test creation of SGD with momentum."""
        sgd = sgd_factory(step_size=0.01, momentum=0.9)
        assert sgd is not None
    
    def test_sgd_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # Negative step size
        with pytest.raises(ValueError):
            riemannopt.SGD(step_size=-0.1)
        
        # Invalid momentum
        with pytest.raises(ValueError):
            riemannopt.SGD(step_size=0.1, momentum=1.5)
    
    def test_sgd_configuration(self, sgd_factory):
        """Test SGD configuration properties."""
        sgd = sgd_factory(
            step_size=0.05,
            momentum=0.95,
            max_iterations=500,
            tolerance=1e-5
        )
        
        # Check configuration is stored correctly
        config = sgd.config
        assert abs(config['learning_rate'] - 0.05) < TOLERANCES['strict']
        assert abs(config['momentum'] - 0.95) < TOLERANCES['strict']


class TestSGDStepFunction:
    """Test the step function of SGD optimizer."""
    
    def test_sgd_step_sphere(self, sgd_factory, sphere_factory, cost_functions):
        """Test SGD step on sphere manifold."""
        sphere = sphere_factory(10)
        sgd = sgd_factory(step_size=0.1)
        
        # Quadratic cost
        A = np.random.randn(10, 10)
        A = A + A.T  # Symmetric
        cost_fn, grad_fn = cost_functions.quadratic(A)
        
        # Initial point
        x = sphere.random_point()
        initial_cost = cost_fn(x)
        
        # Take a step
        grad = grad_fn(x)
        x_new = sgd.step(sphere, x, grad)
        
        # Check new point is on manifold
        assert abs(np.linalg.norm(x_new) - 1.0) < TOLERANCES['default']
        
        # For gradient descent, cost should decrease (if step size appropriate)
        new_cost = cost_fn(x_new)
        # Allow small increase due to manifold constraint
        assert new_cost <= initial_cost + TOLERANCES['relaxed']
    
    def test_sgd_step_stiefel(self, sgd_factory, stiefel_factory, cost_functions, assert_helpers):
        """Test SGD step on Stiefel manifold."""
        stiefel = stiefel_factory(10, 3)
        sgd = sgd_factory(step_size=0.01)
        
        # Trace objective
        C = np.random.randn(10, 10)
        C = C + C.T
        cost_fn, grad_fn = cost_functions.trace_objective(C)
        
        # Initial point
        X = stiefel.random_point()
        
        # Take a step
        grad = grad_fn(X)
        X_new = sgd.step(stiefel, X, grad)
        
        # Check new point is on manifold
        assert_helpers.assert_is_orthogonal(X_new)
    
    def test_sgd_zero_gradient(self, sgd_factory, sphere_factory):
        """Test SGD with zero gradient (should not move)."""
        sphere = sphere_factory(20)
        sgd = sgd_factory(step_size=0.1)
        
        x = sphere.random_point()
        zero_grad = np.zeros(20)
        
        x_new = sgd.step(sphere, x, zero_grad)
        
        assert np.allclose(x, x_new, atol=TOLERANCES['strict'])


class TestSGDMomentum:
    """Test SGD with momentum."""
    
    def test_momentum_accumulation(self, sgd_factory, sphere_factory):
        """Test that momentum accumulates over iterations."""
        sphere = sphere_factory(10)
        sgd = sgd_factory(step_size=0.01, momentum=0.9)
        
        # Constant gradient direction
        x = sphere.random_point()
        grad_direction = np.random.randn(10)
        grad_direction = grad_direction / np.linalg.norm(grad_direction)
        
        # Track movement
        positions = [x.copy()]
        
        # Multiple steps with same gradient direction
        for _ in range(5):
            # Project gradient to maintain direction
            grad = sphere.tangent_projection(x, grad_direction)
            x = sgd.step(sphere, x, grad)
            positions.append(x.copy())
        
        # With momentum, later steps should be larger
        distances = []
        for i in range(len(positions) - 1):
            d = sphere.distance(positions[i], positions[i+1])
            distances.append(d)
        
        # Later distances should generally be larger (momentum effect)
        # Check that average of last half > average of first half
        mid = len(distances) // 2
        assert np.mean(distances[mid:]) > np.mean(distances[:mid]) * 0.9
    
    def test_momentum_direction_change(self, sgd_factory, sphere_factory):
        """Test momentum behavior when gradient direction changes."""
        sphere = sphere_factory(10)
        sgd = sgd_factory(step_size=0.1, momentum=0.9)
        
        x = sphere.random_point()
        
        # First direction
        grad1 = sphere.tangent_projection(x, np.random.randn(10))
        x1 = sgd.step(sphere, x, grad1)
        
        # Opposite direction
        grad2 = -grad1
        x2 = sgd.step(sphere, x1, grad2)
        
        # With high momentum, should not immediately reverse
        # (momentum resists direction change)
        d_forward = sphere.distance(x, x1)
        d_backward = sphere.distance(x1, x2)
        
        # Movement backward should be less than forward due to momentum
        assert d_backward < d_forward


class TestSGDConvergence:
    """Test convergence properties of SGD."""
    
    @pytest.mark.slow
    def test_sgd_convergence_sphere_eigenvalue(self, sgd_factory, sphere_factory):
        """Test SGD convergence to smallest eigenvalue on sphere."""
        n = 20
        sphere = sphere_factory(n)
        
        # Create matrix with known eigenvalues
        eigenvalues = np.linspace(-5, 3, n)
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        A = Q @ np.diag(eigenvalues) @ Q.T
        
        # Cost function: f(x) = x^T A x
        def cost_fn(x):
            return float(x.T @ A @ x)
        
        def grad_fn(x):
            return 2 * A @ x
        
        # Initial point
        x = sphere.random_point()
        
        # Run SGD
        sgd = sgd_factory(step_size=0.01, momentum=0.5)
        costs = [cost_fn(x)]
        
        for _ in range(500):
            grad = grad_fn(x)
            x = sgd.step(sphere, x, grad)
            costs.append(cost_fn(x))
        
        # Should converge close to minimum eigenvalue
        final_cost = costs[-1]
        min_eigenvalue = eigenvalues[0]
        
        assert abs(final_cost - min_eigenvalue) < 0.1
        
        # Check decreasing trend
        # Average of last 10% should be less than average of first 10%
        n_check = len(costs) // 10
        assert np.mean(costs[-n_check:]) < np.mean(costs[:n_check])
    
    @pytest.mark.slow
    def test_sgd_convergence_stiefel_pca(self, sgd_factory, stiefel_factory):
        """Test SGD convergence for PCA on Stiefel manifold."""
        n, p = 20, 3
        stiefel = stiefel_factory(n, p)
        
        # Create covariance matrix with clear principal components
        U = np.linalg.qr(np.random.randn(n, n))[0]
        eigenvalues = np.concatenate([np.array([10, 8, 6]), np.ones(n-3)])
        C = U @ np.diag(eigenvalues) @ U.T
        
        # Cost function: maximize tr(X^T C X)
        def cost_fn(X):
            return -np.trace(X.T @ C @ X)
        
        def grad_fn(X):
            return -2 * C @ X
        
        # Run SGD
        sgd = sgd_factory(step_size=0.001, momentum=0.9)
        X = stiefel.random_point()
        
        costs = [cost_fn(X)]
        for _ in range(1000):
            grad = grad_fn(X)
            X = sgd.step(stiefel, X, grad)
            costs.append(cost_fn(X))
        
        # Should converge to negative sum of top eigenvalues
        expected_cost = -(10 + 8 + 6)
        final_cost = costs[-1]
        
        assert abs(final_cost - expected_cost) < 0.5
        
        # Check that X spans top eigenvectors (approximately)
        top_eigvecs = U[:, :p]
        # X and top_eigvecs should span similar subspaces
        _, s, _ = np.linalg.svd(top_eigvecs.T @ X)
        # All singular values should be close to 1 if subspaces align
        assert np.all(s > 0.9)


class TestSGDStepSizeSchedules:
    """Test different step size schedules for SGD."""
    
    def test_constant_step_size(self, sgd_factory, sphere_factory):
        """Test SGD with constant step size."""
        sphere = sphere_factory(10)
        sgd = sgd_factory(step_size=0.1)
        
        x = sphere.random_point()
        grad = np.random.randn(10)
        
        # Multiple steps should use same step size
        step_sizes = []
        for _ in range(5):
            x_old = x.copy()
            x = sgd.step(sphere, x, grad)
            # Approximate step size from movement
            movement = sphere.distance(x_old, x)
            step_sizes.append(movement)
        
        # Step sizes should be similar (modulo curvature effects)
        assert np.std(step_sizes) < 0.05 * np.mean(step_sizes)
    
    @pytest.mark.skip(reason="Step size schedules not yet implemented")
    def test_decaying_step_size(self, sgd_factory, sphere_factory):
        """Test SGD with decaying step size."""
        # This would test 1/t decay, exponential decay, etc.
        pass


class TestSGDRobustness:
    """Test robustness of SGD optimizer."""
    
    def test_sgd_large_gradient(self, sgd_factory, sphere_factory):
        """Test SGD with very large gradients."""
        sphere = sphere_factory(10)
        sgd = sgd_factory(step_size=0.01)
        
        x = sphere.random_point()
        large_grad = np.random.randn(10) * 1e6
        
        x_new = sgd.step(sphere, x, large_grad)
        
        # Should still be on manifold
        assert abs(np.linalg.norm(x_new) - 1.0) < TOLERANCES['numerical']
        
        # Should not have moved too far
        d = sphere.distance(x, x_new)
        assert d < np.pi  # Less than maximum distance on sphere
    
    def test_sgd_numerical_gradient_noise(self, sgd_factory, sphere_factory):
        """Test SGD with noisy gradients (simulating numerical errors)."""
        sphere = sphere_factory(20)
        sgd = sgd_factory(step_size=0.01, momentum=0.9)
        
        # True gradient direction
        true_direction = np.random.randn(20)
        true_direction = true_direction / np.linalg.norm(true_direction)
        
        x = sphere.random_point()
        
        # Take steps with noisy gradients
        for _ in range(10):
            noise = np.random.randn(20) * 0.1
            noisy_grad = sphere.tangent_projection(x, true_direction + noise)
            x = sgd.step(sphere, x, noisy_grad)
        
        # Should still be on manifold despite noise
        assert abs(np.linalg.norm(x) - 1.0) < TOLERANCES['numerical']


class TestSGDComparison:
    """Compare SGD with theoretical expectations."""
    
    def test_sgd_vs_gradient_flow(self, sgd_factory, sphere_factory):
        """Compare discrete SGD steps with continuous gradient flow."""
        sphere = sphere_factory(10)
        
        # Simple quadratic
        A = np.eye(10)
        A[0, 0] = -1  # Make first direction attractive
        
        def grad_fn(x):
            return 2 * A @ x
        
        # Small step size for better approximation
        sgd = sgd_factory(step_size=0.001)
        
        # Initial point
        x0 = sphere.random_point()
        
        # Discrete steps
        x_discrete = x0.copy()
        for _ in range(100):
            grad = grad_fn(x_discrete)
            x_discrete = sgd.step(sphere, x_discrete, grad)
        
        # Approximate continuous flow (more steps, smaller step size)
        sgd_fine = sgd_factory(step_size=0.0001)
        x_continuous = x0.copy()
        for _ in range(1000):
            grad = grad_fn(x_continuous)
            x_continuous = sgd_fine.step(sphere, x_continuous, grad)
        
        # Results should be similar
        d = sphere.distance(x_discrete, x_continuous)
        assert d < 0.1  # Reasonable tolerance for approximation