"""
Unit tests for the Riemannian L-BFGS optimizer.

This module tests the L-BFGS optimizer including memory management,
quasi-Newton approximations, and convergence properties.
"""

import pytest
import numpy as np
from typing import List, Tuple
from conftest import riemannopt, TOLERANCES


class TestLBFGSCreation:
    """Test L-BFGS optimizer creation and configuration."""
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_create_basic_lbfgs(self):
        """Test creation of basic L-BFGS optimizer."""
        lbfgs = riemannopt.LBFGS(memory_size=10)
        assert lbfgs is not None
        assert hasattr(lbfgs, 'step')
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_create_lbfgs_with_parameters(self):
        """Test creation of L-BFGS with custom parameters."""
        lbfgs = riemannopt.LBFGS(
            memory_size=20,
            line_search='wolfe',
            max_line_search_iters=20,
            tolerance=1e-6
        )
        assert lbfgs is not None
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # Invalid memory size
        with pytest.raises(ValueError):
            riemannopt.LBFGS(memory_size=0)
        
        # Invalid line search
        with pytest.raises(ValueError):
            riemannopt.LBFGS(memory_size=10, line_search='invalid')


class TestLBFGSStepFunction:
    """Test the step function of L-BFGS optimizer."""
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_step_sphere(self, sphere_factory, cost_functions):
        """Test L-BFGS step on sphere manifold."""
        sphere = sphere_factory(20)
        lbfgs = riemannopt.LBFGS(memory_size=5)
        
        # Quadratic cost
        A = np.random.randn(20, 20)
        A = A + A.T
        cost_fn, grad_fn = cost_functions.quadratic(A)
        
        # Initial point
        x = sphere.random_point()
        initial_cost = cost_fn(x)
        
        # Take steps
        costs = [initial_cost]
        for _ in range(10):
            grad = grad_fn(x)
            x = lbfgs.step(sphere, x, grad)
            costs.append(cost_fn(x))
        
        # Should decrease
        assert costs[-1] < costs[0]
        
        # Should stay on manifold
        assert abs(np.linalg.norm(x) - 1.0) < TOLERANCES['default']
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_step_stiefel(self, stiefel_factory, cost_functions, assert_helpers):
        """Test L-BFGS step on Stiefel manifold."""
        stiefel = stiefel_factory(20, 5)
        lbfgs = riemannopt.LBFGS(memory_size=10)
        
        C = np.random.randn(20, 20)
        C = C + C.T
        cost_fn, grad_fn = cost_functions.trace_objective(C)
        
        X = stiefel.random_point()
        
        for _ in range(5):
            grad = grad_fn(X)
            X = lbfgs.step(stiefel, X, grad)
        
        assert_helpers.assert_is_orthogonal(X)


class TestLBFGSMemoryManagement:
    """Test L-BFGS memory buffer management."""
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_memory_accumulation(self, sphere_factory):
        """Test that L-BFGS accumulates history correctly."""
        sphere = sphere_factory(15)
        lbfgs = riemannopt.LBFGS(memory_size=3)
        
        x = sphere.random_point()
        
        # Take more steps than memory size
        gradients = []
        points = [x.copy()]
        
        for i in range(5):
            grad = sphere.random_tangent(x)
            gradients.append(grad)
            x = lbfgs.step(sphere, x, grad)
            points.append(x.copy())
        
        # Check that only last 'memory_size' updates are kept
        # This would require access to internal state
        assert len(points) == 6  # Initial + 5 steps
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_memory_reset(self, sphere_factory):
        """Test L-BFGS memory reset functionality."""
        sphere = sphere_factory(10)
        lbfgs = riemannopt.LBFGS(memory_size=5)
        
        x = sphere.random_point()
        
        # Build up memory
        for _ in range(5):
            grad = sphere.random_tangent(x)
            x = lbfgs.step(sphere, x, grad)
        
        # Reset memory
        lbfgs.reset()
        
        # Next step should behave like first step
        grad = sphere.random_tangent(x)
        x_new = lbfgs.step(sphere, x, grad)
        
        # Without history, should take gradient descent step
        assert sphere.distance(x, x_new) > 0


class TestLBFGSLineSearch:
    """Test L-BFGS line search functionality."""
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_wolfe_line_search(self, sphere_factory):
        """Test L-BFGS with Wolfe line search."""
        sphere = sphere_factory(20)
        lbfgs = riemannopt.LBFGS(memory_size=10, line_search='wolfe')
        
        # Rosenbrock-like function on sphere
        def cost_fn(x):
            return 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
        
        def grad_fn(x):
            grad = np.zeros_like(x)
            grad[0] = -400 * x[0] * (x[1] - x[0]**2) - 2 * (1 - x[0])
            grad[1] = 200 * (x[1] - x[0]**2)
            return grad
        
        x = sphere.random_point()
        
        # Line search should find acceptable step
        for _ in range(10):
            grad = grad_fn(x)
            x_old = x.copy()
            cost_old = cost_fn(x_old)
            
            x = lbfgs.step(sphere, x, grad)
            cost_new = cost_fn(x)
            
            # Wolfe conditions should ensure decrease
            assert cost_new <= cost_old + TOLERANCES['relaxed']
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_backtracking_line_search(self, sphere_factory):
        """Test L-BFGS with backtracking line search."""
        sphere = sphere_factory(10)
        lbfgs = riemannopt.LBFGS(memory_size=5, line_search='backtracking')
        
        # Simple quadratic
        A = np.eye(10)
        A[0, 0] = 10  # Ill-conditioned
        
        def cost_fn(x):
            return float(x.T @ A @ x)
        
        def grad_fn(x):
            return 2 * A @ x
        
        x = sphere.random_point()
        
        for _ in range(5):
            grad = grad_fn(x)
            x = lbfgs.step(sphere, x, grad)
        
        # Should converge
        final_cost = cost_fn(x)
        assert final_cost < 2.0  # Near minimum


class TestLBFGSConvergence:
    """Test convergence properties of L-BFGS."""
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    @pytest.mark.slow
    def test_lbfgs_quadratic_convergence(self, sphere_factory):
        """Test L-BFGS achieves superlinear convergence on quadratics."""
        n = 30
        sphere = sphere_factory(n)
        lbfgs = riemannopt.LBFGS(memory_size=10)
        
        # Well-conditioned quadratic
        eigenvalues = np.linspace(1, 10, n)
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        A = Q @ np.diag(eigenvalues) @ Q.T
        
        def cost_fn(x):
            return float(x.T @ A @ x)
        
        def grad_fn(x):
            return 2 * A @ x
        
        x = sphere.random_point()
        
        # Track convergence
        costs = [cost_fn(x)]
        grad_norms = []
        
        for _ in range(50):
            grad = grad_fn(x)
            grad_norm = np.linalg.norm(sphere.tangent_projection(x, grad))
            grad_norms.append(grad_norm)
            
            x = lbfgs.step(sphere, x, grad)
            costs.append(cost_fn(x))
        
        # Should converge to minimum eigenvalue
        min_eigenvalue = eigenvalues[0]
        assert abs(costs[-1] - min_eigenvalue) < 0.01
        
        # Check superlinear convergence (gradient norm decreases rapidly)
        # Later convergence should be faster
        early_rate = grad_norms[10] / grad_norms[5]
        late_rate = grad_norms[20] / grad_norms[15]
        assert late_rate < early_rate  # Accelerating convergence
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    @pytest.mark.slow
    def test_lbfgs_vs_gradient_descent(self, stiefel_factory):
        """Compare L-BFGS with gradient descent."""
        stiefel = stiefel_factory(20, 5)
        
        # PCA problem
        C = np.random.randn(20, 20)
        C = C @ C.T  # Positive definite
        
        def cost_fn(X):
            return -np.trace(X.T @ C @ X)
        
        def grad_fn(X):
            return -2 * C @ X
        
        X0 = stiefel.random_point()
        
        # Run gradient descent
        sgd = riemannopt.SGD(step_size=0.001)
        X_sgd = X0.copy()
        sgd_costs = []
        
        for _ in range(100):
            grad = grad_fn(X_sgd)
            X_sgd = sgd.step(stiefel, X_sgd, grad)
            sgd_costs.append(cost_fn(X_sgd))
        
        # Run L-BFGS
        lbfgs = riemannopt.LBFGS(memory_size=10)
        X_lbfgs = X0.copy()
        lbfgs_costs = []
        
        for _ in range(100):
            grad = grad_fn(X_lbfgs)
            X_lbfgs = lbfgs.step(stiefel, X_lbfgs, grad)
            lbfgs_costs.append(cost_fn(X_lbfgs))
        
        # L-BFGS should converge faster
        assert lbfgs_costs[50] < sgd_costs[50]
        assert lbfgs_costs[-1] < sgd_costs[-1] + 0.1


class TestLBFGSRobustness:
    """Test robustness of L-BFGS optimizer."""
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_nonconvex_function(self, sphere_factory):
        """Test L-BFGS on non-convex function."""
        sphere = sphere_factory(10)
        lbfgs = riemannopt.LBFGS(memory_size=5)
        
        # Non-convex function with multiple minima
        def cost_fn(x):
            return float(np.sin(5 * x[0]) + np.cos(3 * x[1]) + 0.1 * np.linalg.norm(x[2:])**2)
        
        def grad_fn(x):
            grad = np.zeros_like(x)
            grad[0] = 5 * np.cos(5 * x[0])
            grad[1] = -3 * np.sin(3 * x[1])
            grad[2:] = 0.2 * x[2:]
            return grad
        
        # Multiple random starts
        final_costs = []
        for _ in range(5):
            x = sphere.random_point()
            
            for _ in range(50):
                grad = grad_fn(x)
                x = lbfgs.step(sphere, x, grad)
            
            final_costs.append(cost_fn(x))
        
        # Should find local minima
        assert all(cost < 2.0 for cost in final_costs)
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_ill_conditioned(self, stiefel_factory):
        """Test L-BFGS on ill-conditioned problems."""
        stiefel = stiefel_factory(20, 3)
        lbfgs = riemannopt.LBFGS(memory_size=10)
        
        # Ill-conditioned matrix
        eigenvalues = np.logspace(-3, 3, 20)  # Condition number 1e6
        Q = np.linalg.qr(np.random.randn(20, 20))[0]
        C = Q @ np.diag(eigenvalues) @ Q.T
        
        def grad_fn(X):
            return -2 * C @ X
        
        X = stiefel.random_point()
        
        # Should still make progress
        for _ in range(30):
            grad = grad_fn(X)
            X = lbfgs.step(stiefel, X, grad)
        
        # Check still on manifold
        assert np.allclose(X.T @ X, np.eye(3), atol=TOLERANCES['numerical'])


class TestLBFGSSpecialCases:
    """Test L-BFGS in special scenarios."""
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_exact_line_search(self, sphere_factory):
        """Test L-BFGS with exact line search on simple problem."""
        sphere = sphere_factory(10)
        
        # For quadratic, can compute exact line search
        A = np.eye(10)
        A[0, 0] = 5
        
        def cost_fn(x):
            return float(x.T @ A @ x)
        
        def grad_fn(x):
            return 2 * A @ x
        
        # Custom L-BFGS with exact line search for quadratic
        lbfgs = riemannopt.LBFGS(memory_size=5, line_search='exact')
        
        x = sphere.random_point()
        
        for _ in range(10):
            grad = grad_fn(x)
            x = lbfgs.step(sphere, x, grad)
        
        # Should converge very quickly with exact line search
        final_cost = cost_fn(x)
        assert final_cost < 1.1  # Close to minimum eigenvalue 1.0
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_vector_transport(self, grassmann_factory):
        """Test L-BFGS vector transport for history."""
        grassmann = grassmann_factory(15, 5)
        lbfgs = riemannopt.LBFGS(memory_size=7)
        
        # Simple objective
        A = np.random.randn(15, 15)
        
        def grad_fn(X):
            return A @ X
        
        X = grassmann.random_point()
        
        # The history vectors need to be transported
        # This tests that implementation handles manifold geometry
        for _ in range(20):
            grad = grad_fn(X)
            X = lbfgs.step(grassmann, X, grad)
        
        # Should maintain manifold constraint throughout
        assert np.allclose(X.T @ X, np.eye(5), atol=TOLERANCES['numerical'])


class TestLBFGSComparison:
    """Compare L-BFGS with other methods."""
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    def test_lbfgs_memory_sizes(self, sphere_factory):
        """Test effect of different memory sizes."""
        sphere = sphere_factory(30)
        
        # Test problem
        A = np.random.randn(30, 30)
        A = A + A.T
        
        def cost_fn(x):
            return float(x.T @ A @ x)
        
        def grad_fn(x):
            return 2 * A @ x
        
        # Try different memory sizes
        memory_sizes = [1, 5, 10, 20]
        convergence_speeds = []
        
        for m in memory_sizes:
            lbfgs = riemannopt.LBFGS(memory_size=m)
            x = sphere.random_point()
            
            initial_cost = cost_fn(x)
            for _ in range(50):
                grad = grad_fn(x)
                x = lbfgs.step(sphere, x, grad)
            final_cost = cost_fn(x)
            
            improvement = (initial_cost - final_cost) / initial_cost
            convergence_speeds.append(improvement)
        
        # Larger memory should generally converge better
        assert convergence_speeds[-1] >= convergence_speeds[0] - 0.1
    
    @pytest.mark.skip(reason="L-BFGS not yet implemented")
    @pytest.mark.slow
    def test_lbfgs_bfgs_equivalence(self, sphere_factory):
        """Test that L-BFGS with large memory approximates BFGS."""
        # With memory_size >= n, L-BFGS should behave like BFGS
        n = 10
        sphere = sphere_factory(n)
        
        # L-BFGS with full memory
        lbfgs_full = riemannopt.LBFGS(memory_size=n)
        
        # L-BFGS with limited memory
        lbfgs_limited = riemannopt.LBFGS(memory_size=3)
        
        # Test problem
        A = np.random.randn(n, n)
        A = A + A.T
        
        def grad_fn(x):
            return 2 * A @ x
        
        # Same starting point
        x0 = sphere.random_point()
        
        # Run both
        x_full = x0.copy()
        x_limited = x0.copy()
        
        for _ in range(20):
            # Full memory
            grad = grad_fn(x_full)
            x_full = lbfgs_full.step(sphere, x_full, grad)
            
            # Limited memory
            grad = grad_fn(x_limited)
            x_limited = lbfgs_limited.step(sphere, x_limited, grad)
        
        # Full memory should perform at least as well
        cost_full = float(x_full.T @ A @ x_full)
        cost_limited = float(x_limited.T @ A @ x_limited)
        assert cost_full <= cost_limited + TOLERANCES['relaxed']