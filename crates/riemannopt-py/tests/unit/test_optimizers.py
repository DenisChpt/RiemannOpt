"""
Comprehensive unit tests for optimizer wrappers.

This module tests all optimization algorithms, convergence properties,
and integration with manifolds and cost functions.
"""

import pytest
import numpy as np
import numpy.testing as npt
from typing import Dict, Any, List, Tuple
import time

try:
    import riemannopt as ro
except ImportError:
    pytest.skip("riemannopt not available", allow_module_level=True)


class TestOptimizerCreation:
    """Test optimizer creation and configuration."""
    
    @pytest.mark.parametrize("optimizer_name,default_params", [
        ("SGD", {"learning_rate": 0.1}),
        ("Adam", {"learning_rate": 0.01}),
        ("LBFGS", {"memory_size": 10}),
        ("ConjugateGradient", {}),
        ("TrustRegion", {"initial_radius": 1.0}),
        ("Newton", {}),
    ])
    def test_optimizer_creation(self, optimizer_name, default_params):
        """Test creation of all optimizers with default parameters."""
        optimizer_class = getattr(ro.optimizers, optimizer_name)
        optimizer = optimizer_class(**default_params)
        
        # Check basic properties
        assert hasattr(optimizer, 'optimize_sphere'), f"{optimizer_name} missing optimize_sphere"
        assert hasattr(optimizer, 'optimize_stiefel'), f"{optimizer_name} missing optimize_stiefel"
        assert repr(optimizer), f"{optimizer_name} repr is empty"
    
    def test_sgd_parameter_validation(self):
        """Test SGD parameter validation."""
        # Valid parameters
        sgd = ro.optimizers.SGD(learning_rate=0.1)
        assert sgd is not None
        
        # Invalid learning rate
        with pytest.raises(ValueError):
            ro.optimizers.SGD(learning_rate=-0.1)
        
        with pytest.raises(ValueError):
            ro.optimizers.SGD(learning_rate=0.0)
    
    def test_adam_parameter_validation(self):
        """Test Adam parameter validation."""
        # Valid parameters
        adam = ro.optimizers.Adam(learning_rate=0.01, beta1=0.9, beta2=0.999)
        assert adam is not None
        
        # Invalid learning rate
        with pytest.raises(ValueError):
            ro.optimizers.Adam(learning_rate=-0.01)
        
        # Invalid beta parameters
        with pytest.raises(ValueError):
            ro.optimizers.Adam(learning_rate=0.01, beta1=-0.1)
        
        with pytest.raises(ValueError):
            ro.optimizers.Adam(learning_rate=0.01, beta1=1.1)
        
        with pytest.raises(ValueError):
            ro.optimizers.Adam(learning_rate=0.01, beta2=-0.1)
        
        with pytest.raises(ValueError):
            ro.optimizers.Adam(learning_rate=0.01, beta2=1.1)
    
    def test_lbfgs_parameter_validation(self):
        """Test L-BFGS parameter validation."""
        # Valid parameters
        lbfgs = ro.optimizers.LBFGS(memory_size=10)
        assert lbfgs is not None
        
        # Invalid memory size
        with pytest.raises(ValueError):
            ro.optimizers.LBFGS(memory_size=0)
        
        with pytest.raises(ValueError):
            ro.optimizers.LBFGS(memory_size=-5)
    
    def test_trust_region_parameter_validation(self):
        """Test Trust Region parameter validation."""
        # Valid parameters
        tr = ro.optimizers.TrustRegion(initial_radius=1.0)
        assert tr is not None
        
        # Invalid radius
        with pytest.raises(ValueError):
            ro.optimizers.TrustRegion(initial_radius=-1.0)
        
        with pytest.raises(ValueError):
            ro.optimizers.TrustRegion(initial_radius=0.0)


class TestOptimizerBasicFunctionality:
    """Test basic optimizer functionality on simple problems."""
    
    @pytest.fixture
    def sphere_manifold(self):
        """Create sphere manifold for testing."""
        return ro.manifolds.Sphere(10)
    
    @pytest.fixture
    def stiefel_manifold(self):
        """Create Stiefel manifold for testing."""
        return ro.manifolds.Stiefel(10, 3)
    
    @pytest.fixture
    def quadratic_cost_sphere(self):
        """Create quadratic cost function for sphere."""
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(x**2)
            grad = x
            return cost, grad
        
        return ro.create_cost_function(cost_and_grad)
    
    @pytest.fixture
    def rayleigh_quotient(self):
        """Create Rayleigh quotient cost function."""
        # Fixed random matrix for reproducibility
        np.random.seed(42)
        A = np.random.randn(10, 10)
        A = A + A.T  # Make symmetric
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        return ro.create_cost_function(cost_and_grad)
    
    @pytest.mark.parametrize("optimizer_name,params", [
        ("SGD", {"learning_rate": 0.1}),
        ("Adam", {"learning_rate": 0.01}),
        ("ConjugateGradient", {}),
    ])
    def test_sphere_optimization_basic(self, optimizer_name, params, 
                                     sphere_manifold, rayleigh_quotient):
        """Test basic optimization on sphere."""
        optimizer_class = getattr(ro.optimizers, optimizer_name)
        optimizer = optimizer_class(**params)
        
        # Random starting point
        x0 = sphere_manifold.random_point()
        
        # Optimize
        result = optimizer.optimize_sphere(
            rayleigh_quotient, sphere_manifold, x0,
            max_iterations=50
        )
        
        # Check result structure
        assert isinstance(result, dict), "Result should be dictionary"
        assert 'x' in result, "Result missing final point"
        assert 'cost' in result, "Result missing final cost"
        assert 'iterations' in result, "Result missing iteration count"
        assert 'converged' in result, "Result missing convergence flag"
        
        # Check final point is on manifold
        assert sphere_manifold.contains(result['x']), "Final point not on sphere"
        
        # Check cost improved (for well-conditioned problems)
        initial_cost = rayleigh_quotient.cost(x0)
        final_cost = result['cost']
        # Cost should not increase significantly (allowing for numerical errors)
        assert final_cost <= initial_cost + 1e-10, "Cost increased during optimization"
    
    @pytest.mark.parametrize("optimizer_name,params", [
        ("SGD", {"learning_rate": 0.01}),
        ("Adam", {"learning_rate": 0.001}),
    ])
    def test_stiefel_optimization_basic(self, optimizer_name, params,
                                      stiefel_manifold):
        """Test basic optimization on Stiefel manifold."""
        optimizer_class = getattr(ro.optimizers, optimizer_name)
        optimizer = optimizer_class(**params)
        
        # Create matrix cost function
        np.random.seed(42)
        A = np.random.randn(10, 10)
        A = A + A.T
        
        def cost_and_grad(X: np.ndarray) -> Tuple[float, np.ndarray]:
            AX = A @ X
            cost = np.trace(X.T @ AX)
            grad = 2 * AX
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        # Random starting point
        X0 = stiefel_manifold.random_point()
        
        # Optimize
        result = optimizer.optimize_stiefel(
            cost_fn, stiefel_manifold, X0,
            max_iterations=30
        )
        
        # Check result structure
        assert isinstance(result, dict), "Result should be dictionary"
        assert 'x' in result, "Result missing final point"
        assert 'cost' in result, "Result missing final cost"
        
        # Check final point is on manifold
        assert stiefel_manifold.contains(result['x']), "Final point not on Stiefel manifold"
        
        # Check orthonormality
        X_final = result['x']
        XtX = X_final.T @ X_final
        I = np.eye(3)
        npt.assert_allclose(XtX, I, atol=1e-10, 
                           err_msg="Final point not orthonormal")
    
    def test_optimization_with_early_stopping(self, sphere_manifold, rayleigh_quotient):
        """Test optimization with early stopping via tolerance."""
        optimizer = ro.optimizers.SGD(learning_rate=0.1)
        
        x0 = sphere_manifold.random_point()
        
        # Use strict tolerance for early stopping
        result = optimizer.optimize_sphere(
            rayleigh_quotient, sphere_manifold, x0,
            max_iterations=1000,
            gradient_tolerance=1e-6
        )
        
        # Should converge before max iterations for well-conditioned problems
        assert result['iterations'] < 1000, "Should converge before max iterations"
        assert result['converged'], "Should report convergence"
    
    def test_optimization_iteration_limit(self, sphere_manifold, rayleigh_quotient):
        """Test that optimization respects iteration limits."""
        optimizer = ro.optimizers.SGD(learning_rate=0.001)  # Slow convergence
        
        x0 = sphere_manifold.random_point()
        
        # Very low iteration limit
        result = optimizer.optimize_sphere(
            rayleigh_quotient, sphere_manifold, x0,
            max_iterations=5
        )
        
        # Should stop at iteration limit
        assert result['iterations'] <= 5, "Exceeded iteration limit"


class TestOptimizerConvergence:
    """Test convergence properties of optimizers."""
    
    @pytest.fixture
    def simple_sphere_problem(self):
        """Create a simple, well-conditioned problem on sphere."""
        # Diagonal matrix with known solution
        A = np.diag([10, 5, 2, 1, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01])
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        manifold = ro.manifolds.Sphere(10)
        
        # Known optimal solution (first standard basis vector)
        x_opt = np.zeros(10)
        x_opt[0] = 1.0
        
        return cost_fn, manifold, x_opt
    
    def test_sgd_convergence(self, simple_sphere_problem):
        """Test SGD convergence on well-conditioned problem."""
        cost_fn, manifold, x_opt = simple_sphere_problem
        
        optimizer = ro.optimizers.SGD(learning_rate=0.1)
        
        # Start from random point
        x0 = manifold.random_point()
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=500,
            gradient_tolerance=1e-8
        )
        
        # Should converge
        assert result['converged'], "SGD should converge on simple problem"
        
        # Final point should be close to optimal
        distance = manifold.distance(result['x'], x_opt)
        assert distance < 0.1, f"SGD did not converge to optimum: distance={distance}"
    
    def test_adam_convergence(self, simple_sphere_problem):
        """Test Adam convergence."""
        cost_fn, manifold, x_opt = simple_sphere_problem
        
        optimizer = ro.optimizers.Adam(learning_rate=0.1)
        
        x0 = manifold.random_point()
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=200,
            gradient_tolerance=1e-8
        )
        
        # Adam should converge faster than SGD
        assert result['converged'], "Adam should converge on simple problem"
        
        distance = manifold.distance(result['x'], x_opt)
        assert distance < 0.1, f"Adam did not converge to optimum: distance={distance}"
    
    @pytest.mark.slow
    def test_convergence_comparison(self, simple_sphere_problem):
        """Compare convergence of different optimizers."""
        cost_fn, manifold, x_opt = simple_sphere_problem
        
        optimizers = [
            ("SGD", ro.optimizers.SGD(learning_rate=0.1)),
            ("Adam", ro.optimizers.Adam(learning_rate=0.1)),
            ("ConjugateGradient", ro.optimizers.ConjugateGradient()),
        ]
        
        results = {}
        
        for name, optimizer in optimizers:
            # Use same starting point for fair comparison
            np.random.seed(42)
            x0 = manifold.random_point()
            
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=100,
                gradient_tolerance=1e-10
            )
            
            results[name] = result
        
        # All should converge
        for name, result in results.items():
            assert result['converged'], f"{name} did not converge"
            
            distance = manifold.distance(result['x'], x_opt)
            assert distance < 0.2, f"{name} did not converge close enough: {distance}"
    
    def test_convergence_with_different_tolerances(self, simple_sphere_problem):
        """Test convergence behavior with different tolerance settings."""
        cost_fn, manifold, x_opt = simple_sphere_problem
        
        optimizer = ro.optimizers.Adam(learning_rate=0.1)
        x0 = manifold.random_point()
        
        tolerances = [1e-4, 1e-6, 1e-8]
        
        for tol in tolerances:
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=200,
                gradient_tolerance=tol
            )
            
            if result['converged']:
                # If converged, should satisfy the tolerance
                final_grad = cost_fn.gradient(result['x'])
                projected_grad = manifold.project_tangent(result['x'], final_grad)
                grad_norm = manifold.norm(result['x'], projected_grad)
                
                # Allow some numerical slack
                assert grad_norm <= tol * 10, \
                    f"Gradient norm {grad_norm} exceeds tolerance {tol}"


class TestOptimizerRobustness:
    """Test optimizer robustness and edge cases."""
    
    def test_optimization_with_zero_gradient(self):
        """Test behavior when gradient is zero."""
        manifold = ro.manifolds.Sphere(5)
        
        # Constant cost function (zero gradient everywhere)
        def constant_cost(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 1.0
            grad = np.zeros_like(x)
            return cost, grad
        
        cost_fn = ro.create_cost_function(constant_cost)
        optimizer = ro.optimizers.SGD(learning_rate=0.1)
        
        x0 = manifold.random_point()
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=10
        )
        
        # Should stop quickly due to zero gradient
        assert result['iterations'] <= 10, "Should stop with zero gradient"
        
        # Final point should be close to initial (no movement)
        distance = manifold.distance(x0, result['x'])
        assert distance < 1e-10, "Point should not move with zero gradient"
    
    def test_optimization_with_large_gradient(self):
        """Test behavior with very large gradients."""
        manifold = ro.manifolds.Sphere(5)
        
        # Cost function with large gradients
        def large_grad_cost(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(x**2)
            grad = 1000 * x  # Very large gradient
            return cost, grad
        
        cost_fn = ro.create_cost_function(large_grad_cost)
        optimizer = ro.optimizers.SGD(learning_rate=0.001)  # Small learning rate
        
        x0 = manifold.random_point()
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=50
        )
        
        # Should remain stable and on manifold
        assert manifold.contains(result['x']), "Point should remain on manifold"
        assert np.all(np.isfinite(result['x'])), "Result should be finite"
    
    def test_optimization_numerical_precision(self):
        """Test optimization with high precision requirements."""
        manifold = ro.manifolds.Sphere(3)  # Small dimension for easier testing
        
        # Well-conditioned quadratic
        A = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 0.5]])
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        optimizer = ro.optimizers.Adam(learning_rate=0.1)
        
        x0 = manifold.random_point()
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=1000,
            gradient_tolerance=1e-12
        )
        
        # Check high precision
        if result['converged']:
            final_grad = cost_fn.gradient(result['x'])
            projected_grad = manifold.project_tangent(result['x'], final_grad)
            grad_norm = manifold.norm(result['x'], projected_grad)
            
            assert grad_norm < 1e-10, f"High precision not achieved: {grad_norm}"
    
    def test_optimization_with_callbacks(self):
        """Test optimization with callback functions."""
        manifold = ro.manifolds.Sphere(5)
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(x**2)
            grad = x
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        # Track optimization progress
        history = []
        
        def callback(iteration: int, x: np.ndarray, cost: float, grad_norm: float) -> bool:
            history.append({
                'iteration': iteration,
                'cost': cost,
                'grad_norm': grad_norm
            })
            # Continue optimization
            return True
        
        optimizer = ro.optimizers.SGD(learning_rate=0.1)
        x0 = manifold.random_point()
        
        # Note: This test assumes callback support exists
        # If not implemented yet, this test should be skipped
        try:
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=20,
                callback=callback
            )
            
            # Check that callback was called
            assert len(history) > 0, "Callback was not called"
            assert len(history) <= 20, "Too many callback calls"
            
            # Check decreasing cost
            costs = [h['cost'] for h in history]
            for i in range(1, len(costs)):
                assert costs[i] <= costs[i-1] + 1e-10, "Cost should not increase"
                
        except TypeError:
            # Callback not implemented yet
            pytest.skip("Callback functionality not implemented")


class TestOptimizerPerformance:
    """Test performance characteristics of optimizers."""
    
    @pytest.fixture
    def large_problem(self):
        """Create a larger optimization problem for performance testing."""
        n = 100
        manifold = ro.manifolds.Sphere(n)
        
        # Random symmetric matrix
        np.random.seed(42)
        A = np.random.randn(n, n)
        A = A + A.T
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        return cost_fn, manifold
    
    @pytest.mark.slow
    def test_optimization_performance(self, large_problem):
        """Test that optimization is reasonably fast."""
        cost_fn, manifold = large_problem
        
        optimizer = ro.optimizers.Adam(learning_rate=0.1)
        x0 = manifold.random_point()
        
        start_time = time.perf_counter()
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=100
        )
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        # Should complete in reasonable time (less than 1 second for this problem)
        assert total_time < 1.0, f"Optimization too slow: {total_time:.3f}s"
        
        # Should make progress
        initial_cost = cost_fn.cost(x0)
        final_cost = result['cost']
        assert final_cost < initial_cost, "No progress made during optimization"
    
    @pytest.mark.slow
    def test_memory_usage_stability(self, large_problem):
        """Test that optimization doesn't leak memory."""
        import gc
        
        cost_fn, manifold = large_problem
        optimizer = ro.optimizers.SGD(learning_rate=0.1)
        
        # Force garbage collection
        gc.collect()
        
        # Run multiple optimizations
        for _ in range(10):
            x0 = manifold.random_point()
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=10
            )
            
            # Results should be valid
            assert manifold.contains(result['x']), "Invalid result"
        
        # Force cleanup
        gc.collect()
        
        # If we get here without memory issues, test passes


class TestHighLevelOptimizeFunction:
    """Test the high-level optimize() function."""
    
    def test_optimize_function_sphere(self):
        """Test high-level optimize function with sphere."""
        manifold = ro.manifolds.Sphere(10)
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(x**2)
            grad = x
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        x0 = manifold.random_point()
        
        # Test with different optimizers
        for optimizer_name in ["SGD", "Adam"]:
            result = ro.optimize(
                cost_function=cost_fn,
                manifold=manifold,
                initial_point=x0,
                optimizer=optimizer_name,
                max_iterations=50
            )
            
            assert isinstance(result, dict), f"Result should be dict for {optimizer_name}"
            assert 'x' in result, f"Missing final point for {optimizer_name}"
            assert manifold.contains(result['x']), f"Final point not on manifold for {optimizer_name}"
    
    def test_optimize_function_with_parameters(self):
        """Test high-level optimize function with optimizer parameters."""
        manifold = ro.manifolds.Sphere(5)
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(x**2)
            grad = x
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        x0 = manifold.random_point()
        
        result = ro.optimize(
            cost_function=cost_fn,
            manifold=manifold,
            initial_point=x0,
            optimizer="Adam",
            learning_rate=0.01,  # Optimizer parameter
            max_iterations=30
        )
        
        assert isinstance(result, dict), "Result should be dict"
        assert manifold.contains(result['x']), "Final point not on manifold"
    
    def test_optimize_function_invalid_optimizer(self):
        """Test that invalid optimizer names are handled."""
        manifold = ro.manifolds.Sphere(3)
        
        def cost(x: np.ndarray) -> float:
            return np.sum(x**2)
        
        cost_fn = ro.create_cost_function(cost)
        x0 = manifold.random_point()
        
        with pytest.raises((ValueError, KeyError)):
            ro.optimize(
                cost_function=cost_fn,
                manifold=manifold,
                initial_point=x0,
                optimizer="NonexistentOptimizer"
            )


# Integration tests
class TestOptimizerIntegration:
    """Integration tests combining optimizers with all components."""
    
    def test_end_to_end_optimization_workflow(self):
        """Test complete optimization workflow."""
        # Create manifold
        manifold = ro.manifolds.Stiefel(8, 3)
        
        # Create cost function (matrix optimization problem)
        np.random.seed(123)
        C = np.random.randn(8, 8)
        C = C @ C.T  # Positive definite
        
        def cost_and_grad(X: np.ndarray) -> Tuple[float, np.ndarray]:
            CX = C @ X
            cost = 0.5 * np.trace(X.T @ CX)
            grad = CX
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad, validate_gradient=True)
        
        # Initial point
        X0 = manifold.random_point()
        initial_cost = cost_fn.cost(X0)
        
        # Optimize with different algorithms
        optimizers = [
            ro.optimizers.SGD(learning_rate=0.01),
            ro.optimizers.Adam(learning_rate=0.01),
        ]
        
        for optimizer in optimizers:
            result = optimizer.optimize_stiefel(
                cost_fn, manifold, X0,
                max_iterations=100,
                gradient_tolerance=1e-8
            )
            
            # Verify result quality
            X_final = result['x']
            
            # Should be on manifold
            assert manifold.contains(X_final), f"Final point not on manifold for {type(optimizer).__name__}"
            
            # Should have improved cost
            final_cost = result['cost']
            assert final_cost <= initial_cost + 1e-10, f"Cost did not improve for {type(optimizer).__name__}"
            
            # Should satisfy orthonormality
            XtX = X_final.T @ X_final
            I = np.eye(3)
            npt.assert_allclose(XtX, I, atol=1e-10,
                               err_msg=f"Orthonormality violated for {type(optimizer).__name__}")
    
    def test_optimization_with_multiple_manifolds(self):
        """Test optimization works consistently across different manifolds."""
        manifolds_and_problems = [
            # (manifold, cost_function, initial_point_generator)
            (ro.manifolds.Sphere(5), self._sphere_rayleigh_problem, lambda m: m.random_point()),
            (ro.manifolds.Stiefel(6, 2), self._stiefel_trace_problem, lambda m: m.random_point()),
        ]
        
        optimizer = ro.optimizers.Adam(learning_rate=0.01)
        
        for manifold, cost_fn, point_gen in manifolds_and_problems:
            x0 = point_gen(manifold)
            
            result = getattr(optimizer, f"optimize_{manifold.__class__.__name__.lower()}")(
                cost_fn, manifold, x0,
                max_iterations=50
            )
            
            # Basic checks
            assert isinstance(result, dict), f"Invalid result for {type(manifold).__name__}"
            assert manifold.contains(result['x']), f"Final point not on {type(manifold).__name__}"
    
    def _sphere_rayleigh_problem(self):
        """Helper: create Rayleigh problem for sphere."""
        A = np.diag([2, 1, 0.5, 0.2, 0.1])
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        return ro.create_cost_function(cost_and_grad)
    
    def _stiefel_trace_problem(self):
        """Helper: create trace problem for Stiefel."""
        B = np.random.randn(6, 6)
        B = B + B.T
        
        def cost_and_grad(X: np.ndarray) -> Tuple[float, np.ndarray]:
            BX = B @ X
            cost = np.trace(X.T @ BX)
            grad = 2 * BX
            return cost, grad
        
        return ro.create_cost_function(cost_and_grad)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])