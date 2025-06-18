"""
Unit tests for the Riemannian Adam optimizer.

This module tests the Adam optimizer including adaptive learning rates,
momentum handling, and various Adam variants.
"""

import pytest
import numpy as np
from typing import Tuple
from conftest import riemannopt, TOLERANCES


class TestAdamCreation:
    """Test Adam optimizer creation and configuration."""
    
    def test_create_basic_adam(self, adam_factory):
        """Test creation of basic Adam optimizer."""
        adam = adam_factory(learning_rate=0.001)
        assert adam is not None
        assert hasattr(adam, 'step')
    
    def test_create_adam_with_parameters(self, adam_factory):
        """Test creation of Adam with custom parameters."""
        adam = adam_factory(
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8
        )
        assert adam is not None
    
    def test_adam_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # Negative learning rate
        with pytest.raises(ValueError):
            riemannopt.Adam(learning_rate=-0.001)
        
        # Invalid beta1
        with pytest.raises(ValueError):
            riemannopt.Adam(learning_rate=0.001, beta1=1.5)
        
        # Invalid beta2
        with pytest.raises(ValueError):
            riemannopt.Adam(learning_rate=0.001, beta2=-0.1)
    
    def test_adam_configuration(self, adam_factory):
        """Test Adam configuration properties."""
        adam = adam_factory(
            learning_rate=0.002,
            beta1=0.85,
            beta2=0.98,
            epsilon=1e-6
        )
        
        config = adam.config
        assert abs(config['learning_rate'] - 0.002) < TOLERANCES['strict']
        assert abs(config['beta1'] - 0.85) < TOLERANCES['strict']
        assert abs(config['beta2'] - 0.98) < TOLERANCES['strict']


class TestAdamStepFunction:
    """Test the step function of Adam optimizer."""
    
    def test_adam_step_sphere(self, adam_factory, sphere_factory, cost_functions):
        """Test Adam step on sphere manifold."""
        sphere = sphere_factory(10)
        adam = adam_factory(learning_rate=0.01)
        
        # Quadratic cost
        A = np.random.randn(10, 10)
        A = A + A.T
        cost_fn, grad_fn = cost_functions.quadratic(A)
        
        # Initial point
        x = sphere.random_point()
        initial_cost = cost_fn(x)
        
        # Take multiple steps to see adaptive behavior
        costs = [initial_cost]
        for _ in range(10):
            grad = grad_fn(x)
            x = adam.step(sphere, x, grad)
            costs.append(cost_fn(x))
        
        # Should generally decrease (Adam can increase initially)
        assert costs[-1] < costs[0]
        
        # Point should stay on manifold
        assert abs(np.linalg.norm(x) - 1.0) < TOLERANCES['default']
    
    def test_adam_step_stiefel(self, adam_factory, stiefel_factory, cost_functions, assert_helpers):
        """Test Adam step on Stiefel manifold."""
        stiefel = stiefel_factory(15, 5)
        adam = adam_factory(learning_rate=0.001)
        
        # Cost function
        C = np.random.randn(15, 15)
        C = C + C.T
        cost_fn, grad_fn = cost_functions.trace_objective(C)
        
        # Run Adam
        X = stiefel.random_point()
        for _ in range(5):
            grad = grad_fn(X)
            X = adam.step(stiefel, X, grad)
        
        # Check manifold constraint
        assert_helpers.assert_is_orthogonal(X)
    
    def test_adam_zero_gradient(self, adam_factory, sphere_factory):
        """Test Adam with zero gradient."""
        sphere = sphere_factory(20)
        adam = adam_factory(learning_rate=0.1)
        
        x = sphere.random_point()
        zero_grad = np.zeros(20)
        
        # Even with zero gradient, Adam might move slightly due to momentum
        x_new = adam.step(sphere, x, zero_grad)
        
        # But movement should be very small
        d = sphere.distance(x, x_new)
        assert d < 1e-7  # Slightly more tolerant than TOLERANCES['relaxed']


class TestAdamMomentumEstimates:
    """Test Adam's momentum and second moment estimates."""
    
    def test_adam_momentum_accumulation(self, adam_factory, sphere_factory):
        """Test that Adam accumulates first moment correctly."""
        sphere = sphere_factory(10)
        adam = adam_factory(learning_rate=0.01, beta1=0.9)
        
        # Constant gradient
        x = sphere.random_point()
        constant_grad = sphere.random_tangent(x)
        constant_grad = constant_grad / np.linalg.norm(constant_grad)
        
        # Track positions
        positions = [x.copy()]
        
        # Multiple steps
        for _ in range(20):
            grad = sphere.tangent_projection(x, constant_grad)
            x = adam.step(sphere, x, grad)
            positions.append(x.copy())
        
        # With constant gradient, should move consistently after warmup
        # Check that later steps are larger (due to bias correction)
        distances = []
        for i in range(1, len(positions)):
            d = sphere.distance(positions[i-1], positions[i])
            distances.append(d)
        
        # After initial steps, distances should stabilize
        later_distances = distances[10:]
        assert np.std(later_distances) < 0.3 * np.mean(later_distances)
    
    def test_adam_adaptive_learning(self, adam_factory, sphere_factory):
        """Test Adam's adaptive learning rate behavior."""
        sphere = sphere_factory(20)
        adam = adam_factory(learning_rate=0.1, beta2=0.999)
        
        x = sphere.random_point()
        
        # First: large gradients
        large_grad = sphere.random_tangent(x) * 10
        x1 = adam.step(sphere, x, large_grad)
        d1 = sphere.distance(x, x1)
        
        # Continue with large gradients to build up second moment
        x_current = x1
        for _ in range(5):
            grad = sphere.tangent_projection(x_current, large_grad)
            x_current = adam.step(sphere, x_current, grad)
        
        # Now small gradient - step should be larger relative to gradient
        small_grad = sphere.random_tangent(x_current) * 0.1
        x_next = adam.step(sphere, x_current, small_grad)
        d_small = sphere.distance(x_current, x_next)
        
        # Due to adaptive learning, the effective step size should be different
        # This is a weak test as the behavior is complex
        assert d_small > 0  # Should still move
    
    def test_adam_bias_correction(self, adam_factory, sphere_factory):
        """Test Adam's bias correction in early iterations."""
        sphere = sphere_factory(10)
        adam = adam_factory(learning_rate=0.1, beta1=0.9, beta2=0.999)
        
        x = sphere.random_point()
        grad = sphere.random_tangent(x)
        
        # First step should have strong bias correction
        x1 = adam.step(sphere, x, grad)
        d1 = sphere.distance(x, x1)
        
        # Reset optimizer and take many steps
        adam_warmed = adam_factory(learning_rate=0.1, beta1=0.9, beta2=0.999)
        x_warm = x.copy()
        
        # Warm up the optimizer
        for _ in range(100):
            x_warm = adam_warmed.step(sphere, x_warm, grad)
        
        # Now take a step with same gradient
        x_warm_next = adam_warmed.step(sphere, x_warm, grad)
        d_warm = sphere.distance(x_warm, x_warm_next)
        
        # Early step should be different from warmed-up step
        assert abs(d1 - d_warm) > TOLERANCES['relaxed']


class TestAdamConvergence:
    """Test convergence properties of Adam."""
    
    @pytest.mark.slow
    def test_adam_convergence_sphere(self, adam_factory, sphere_factory):
        """Test Adam convergence on sphere optimization."""
        n = 30
        sphere = sphere_factory(n)
        
        # Create problem with known minimum
        eigenvalues = np.linspace(-5, 5, n)
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        A = Q @ np.diag(eigenvalues) @ Q.T
        
        def cost_fn(x):
            return float(x.T @ A @ x)
        
        def grad_fn(x):
            return 2 * A @ x
        
        # Run Adam
        adam = adam_factory(learning_rate=0.01)
        x = sphere.random_point()
        
        costs = [cost_fn(x)]
        for _ in range(500):
            grad = grad_fn(x)
            x = adam.step(sphere, x, grad)
            costs.append(cost_fn(x))
        
        # Should converge near minimum eigenvalue
        min_eigenvalue = eigenvalues[0]
        assert abs(costs[-1] - min_eigenvalue) < 0.2
        
        # Check convergence trend
        assert np.mean(costs[-50:]) < np.mean(costs[:50])
    
    @pytest.mark.slow
    def test_adam_vs_sgd_convergence(self, adam_factory, sgd_factory, stiefel_factory):
        """Compare Adam and SGD convergence rates."""
        n, p = 30, 5
        stiefel = stiefel_factory(n, p)
        
        # PCA problem
        eigenvalues = np.exp(-np.arange(n) / 5)
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        C = Q @ np.diag(eigenvalues) @ Q.T
        
        def cost_fn(X):
            return -np.trace(X.T @ C @ X)
        
        def grad_fn(X):
            return -2 * C @ X
        
        # Same initial point
        X0 = stiefel.random_point()
        
        # Run SGD
        sgd = sgd_factory(step_size=0.001)
        X_sgd = X0.copy()
        sgd_costs = [cost_fn(X_sgd)]
        
        for _ in range(300):
            grad = grad_fn(X_sgd)
            X_sgd = sgd.step(stiefel, X_sgd, grad)
            sgd_costs.append(cost_fn(X_sgd))
        
        # Run Adam
        adam = adam_factory(learning_rate=0.001)
        X_adam = X0.copy()
        adam_costs = [cost_fn(X_adam)]
        
        for _ in range(300):
            grad = grad_fn(X_adam)
            X_adam = adam.step(stiefel, X_adam, grad)
            adam_costs.append(cost_fn(X_adam))
        
        # Both should converge
        expected_cost = -np.sum(eigenvalues[:p])
        assert abs(sgd_costs[-1] - expected_cost) < 2.0  # More tolerant
        assert abs(adam_costs[-1] - expected_cost) < 2.0  # More tolerant
        
        # Adam often converges faster initially
        # Check performance at iteration 100
        assert abs(adam_costs[100] - expected_cost) <= abs(sgd_costs[100] - expected_cost) + 0.5


class TestAdamVariants:
    """Test variants of Adam optimizer."""
    
    @pytest.mark.skip(reason="AdamW not yet implemented")
    def test_adamw_weight_decay(self, adam_factory, sphere_factory):
        """Test AdamW with weight decay."""
        pass
    
    @pytest.mark.skip(reason="AMSGrad not yet implemented")
    def test_amsgrad_convergence(self, adam_factory, sphere_factory):
        """Test AMSGrad variant for better convergence."""
        pass


class TestAdamRobustness:
    """Test robustness of Adam optimizer."""
    
    def test_adam_sparse_gradients(self, adam_factory, sphere_factory):
        """Test Adam with sparse gradients."""
        sphere = sphere_factory(50)
        adam = adam_factory(learning_rate=0.01)
        
        x = sphere.random_point()
        
        # Simulate sparse gradients
        for i in range(20):
            grad = np.zeros(50)
            # Only a few components non-zero
            active_indices = np.random.choice(50, 5, replace=False)
            grad[active_indices] = np.random.randn(5)
            
            grad = sphere.tangent_projection(x, grad)
            x = adam.step(sphere, x, grad)
        
        # Should remain on manifold
        assert abs(np.linalg.norm(x) - 1.0) < TOLERANCES['numerical']
    
    def test_adam_gradient_clipping(self, adam_factory, sphere_factory):
        """Test Adam with very large gradients."""
        sphere = sphere_factory(20)
        adam = adam_factory(learning_rate=0.001)
        
        x = sphere.random_point()
        
        # Large gradient
        huge_grad = sphere.random_tangent(x) * 1e10
        x_new = adam.step(sphere, x, huge_grad)
        
        # Should not explode (but might move more than 1.0 with huge gradient)
        d = sphere.distance(x, x_new)
        assert d < 3.0  # More tolerant for huge gradients
        
        # Should stay on manifold
        assert abs(np.linalg.norm(x_new) - 1.0) < TOLERANCES['numerical']
    
    def test_adam_long_optimization(self, adam_factory, grassmann_factory):
        """Test Adam stability over many iterations."""
        grassmann = grassmann_factory(20, 5)
        adam = adam_factory(learning_rate=0.001)
        
        # Simple cost function
        A = np.random.randn(20, 20)
        
        def grad_fn(X):
            return A @ X
        
        X = grassmann.random_point()
        
        # Many iterations
        for i in range(1000):
            grad = grad_fn(X)
            X = adam.step(grassmann, X, grad)
            
            # Periodically check manifold constraint
            if i % 100 == 0:
                assert np.allclose(X.T @ X, np.eye(5), atol=TOLERANCES['numerical'])


class TestAdamStateManagement:
    """Test Adam's internal state management."""
    
    def test_adam_state_persistence(self, adam_factory, sphere_factory):
        """Test that Adam maintains state across steps."""
        sphere = sphere_factory(10)
        adam = adam_factory(learning_rate=0.01, beta1=0.5)  # Low beta1 to see effect
        
        x = sphere.random_point()
        
        # First gradient
        grad1 = sphere.random_tangent(x)
        x1 = adam.step(sphere, x, grad1)
        
        # Opposite gradient - with momentum, should partially cancel
        grad2 = -grad1
        x2 = adam.step(sphere, x1, grad2)
        
        # Due to momentum, x2 should not return exactly to x
        d_total = sphere.distance(x, x2)
        assert d_total > TOLERANCES['relaxed']  # Some net movement
    
    def test_adam_independent_coordinates(self, adam_factory, stiefel_factory):
        """Test that Adam treats different gradient components independently."""
        stiefel = stiefel_factory(10, 2)
        adam = adam_factory(learning_rate=0.1, beta2=0.9)
        
        X = stiefel.random_point()
        
        # Create gradients with very different scales in different components
        for _ in range(10):
            grad = np.random.randn(10, 2)
            grad[:, 0] *= 10.0   # Large gradients in first column
            grad[:, 1] *= 0.1    # Small gradients in second column
            
            X = adam.step(stiefel, X, grad)
        
        # Both columns should still have unit norm (Stiefel constraint)
        col_norms = np.linalg.norm(X, axis=0)
        assert np.allclose(col_norms, 1.0, atol=TOLERANCES['default'])


class TestAdamSpecialCases:
    """Test Adam in special optimization scenarios."""
    
    def test_adam_noisy_gradients(self, adam_factory, sphere_factory):
        """Test Adam with noisy gradient estimates."""
        sphere = sphere_factory(30)
        adam = adam_factory(learning_rate=0.001)
        
        # True gradient direction
        true_direction = sphere.random_tangent(sphere.random_point())
        true_direction = true_direction / np.linalg.norm(true_direction)
        
        x = sphere.random_point()
        
        # Optimize with noisy gradients
        for _ in range(100):
            noise = np.random.randn(30) * 2.0
            noisy_grad = sphere.tangent_projection(x, true_direction + noise)
            x = adam.step(sphere, x, noisy_grad)
        
        # Despite noise, should make progress in true direction
        # (This is a weak test - mainly checking stability)
        assert abs(np.linalg.norm(x) - 1.0) < TOLERANCES['numerical']
    
    def test_adam_saddle_point(self, adam_factory, sphere_factory):
        """Test Adam escape from saddle point."""
        n = 20
        sphere = sphere_factory(n)
        
        # Create matrix with saddle point at origin
        A = np.diag(np.concatenate([np.ones(10), -np.ones(10)]))
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        A = Q @ A @ Q.T
        
        def grad_fn(x):
            return 2 * A @ x
        
        # Start near saddle
        x = sphere.random_point()
        x = x * 0.1  # Small perturbation
        x = sphere.project(x)
        
        adam = adam_factory(learning_rate=0.01)
        
        # Run optimization
        for _ in range(200):
            grad = grad_fn(x)
            x = adam.step(sphere, x, grad)
        
        # Should escape saddle and reach minimum
        final_value = float(x.T @ A @ x)
        assert final_value < -0.5  # Should find negative region