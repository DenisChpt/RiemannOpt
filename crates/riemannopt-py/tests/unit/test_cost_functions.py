"""
Comprehensive unit tests for cost function wrappers.

This module tests the cost function interface, gradient validation,
and performance characteristics of the Python-Rust bridge.
"""

import pytest
import numpy as np
import numpy.testing as npt
from typing import Callable, Tuple, Any
import time

try:
    import riemannopt as ro
except ImportError:
    pytest.skip("riemannopt not available", allow_module_level=True)


class TestCostFunctionCreation:
    """Test cost function creation and validation."""
    
    def test_cost_only_function(self):
        """Test creating cost function with only cost."""
        def cost(x: np.ndarray) -> float:
            return 0.5 * np.sum(x**2)
        
        cost_fn = ro.create_cost_function(cost)
        
        # Test evaluation
        x = np.random.randn(5)
        result = cost_fn.cost(x)
        expected = 0.5 * np.sum(x**2)
        
        npt.assert_allclose(result, expected, rtol=1e-12,
                           err_msg="Cost function evaluation incorrect")
    
    def test_cost_with_gradient_function(self):
        """Test creating cost function with gradient."""
        def cost(x: np.ndarray) -> float:
            return 0.5 * np.sum(x**2)
        
        def grad(x: np.ndarray) -> np.ndarray:
            return x
        
        cost_fn = ro.create_cost_function(cost, grad)
        
        # Test cost evaluation
        x = np.random.randn(5)
        result = cost_fn.cost(x)
        expected = 0.5 * np.sum(x**2)
        npt.assert_allclose(result, expected, rtol=1e-12)
        
        # Test gradient evaluation
        grad_result = cost_fn.gradient(x)
        npt.assert_allclose(grad_result, x, rtol=1e-12,
                           err_msg="Gradient evaluation incorrect")
    
    def test_combined_cost_and_gradient_function(self):
        """Test creating cost function that returns both cost and gradient."""
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(x**2)
            grad = x
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        # Test cost evaluation
        x = np.random.randn(5)
        result = cost_fn.cost(x)
        expected = 0.5 * np.sum(x**2)
        npt.assert_allclose(result, expected, rtol=1e-12)
        
        # Test gradient evaluation
        grad_result = cost_fn.gradient(x)
        npt.assert_allclose(grad_result, x, rtol=1e-12)
        
        # Test combined evaluation
        cost_result, grad_result = cost_fn.cost_and_gradient(x)
        npt.assert_allclose(cost_result, expected, rtol=1e-12)
        npt.assert_allclose(grad_result, x, rtol=1e-12)
    
    def test_invalid_function_signatures(self):
        """Test that invalid function signatures are rejected."""
        # Function with wrong return type
        def bad_cost(x: np.ndarray) -> str:
            return "not a float"
        
        with pytest.raises((TypeError, ValueError)):
            ro.create_cost_function(bad_cost)
        
        # Gradient function with wrong return type
        def good_cost(x: np.ndarray) -> float:
            return 0.5 * np.sum(x**2)
        
        def bad_grad(x: np.ndarray) -> str:
            return "not an array"
        
        with pytest.raises((TypeError, ValueError)):
            ro.create_cost_function(good_cost, bad_grad)
    
    def test_cost_function_with_validation(self):
        """Test cost function creation with gradient validation."""
        def cost(x: np.ndarray) -> float:
            return 0.5 * np.sum(x**2)
        
        def grad(x: np.ndarray) -> np.ndarray:
            return x
        
        # Should pass validation
        cost_fn = ro.create_cost_function(cost, grad, validate_gradient=True)
        
        # Test that it still works
        x = np.random.randn(5)
        result = cost_fn.cost(x)
        npt.assert_allclose(result, 0.5 * np.sum(x**2), rtol=1e-12)
    
    def test_cost_function_validation_failure(self):
        """Test that incorrect gradients are caught by validation."""
        def cost(x: np.ndarray) -> float:
            return 0.5 * np.sum(x**2)
        
        def wrong_grad(x: np.ndarray) -> np.ndarray:
            return 2 * x  # Wrong by factor of 2
        
        # Should fail validation
        with pytest.raises((ValueError, AssertionError)):
            ro.create_cost_function(cost, wrong_grad, validate_gradient=True)


class TestCostFunctionEvaluation:
    """Test cost function evaluation in various scenarios."""
    
    @pytest.fixture
    def quadratic_cost_fn(self):
        """Create a quadratic cost function."""
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(x**2)
            grad = x
            return cost, grad
        
        return ro.create_cost_function(cost_and_grad)
    
    @pytest.fixture
    def matrix_cost_fn(self):
        """Create a cost function for matrices."""
        def cost_and_grad(X: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.5 * np.sum(X**2)
            grad = X
            return cost, grad
        
        return ro.create_cost_function(cost_and_grad)
    
    def test_vector_input_evaluation(self, quadratic_cost_fn):
        """Test evaluation with vector inputs."""
        x = np.random.randn(10)
        
        # Test cost
        cost = quadratic_cost_fn.cost(x)
        expected_cost = 0.5 * np.sum(x**2)
        npt.assert_allclose(cost, expected_cost, rtol=1e-12)
        
        # Test gradient
        grad = quadratic_cost_fn.gradient(x)
        npt.assert_allclose(grad, x, rtol=1e-12)
        
        # Test combined
        cost_combined, grad_combined = quadratic_cost_fn.cost_and_gradient(x)
        npt.assert_allclose(cost_combined, expected_cost, rtol=1e-12)
        npt.assert_allclose(grad_combined, x, rtol=1e-12)
    
    def test_matrix_input_evaluation(self, matrix_cost_fn):
        """Test evaluation with matrix inputs."""
        X = np.random.randn(5, 3)
        
        # Test cost
        cost = matrix_cost_fn.cost(X)
        expected_cost = 0.5 * np.sum(X**2)
        npt.assert_allclose(cost, expected_cost, rtol=1e-12)
        
        # Test gradient
        grad = matrix_cost_fn.gradient(X)
        npt.assert_allclose(grad, X, rtol=1e-12)
        
        # Test combined
        cost_combined, grad_combined = matrix_cost_fn.cost_and_gradient(X)
        npt.assert_allclose(cost_combined, expected_cost, rtol=1e-12)
        npt.assert_allclose(grad_combined, X, rtol=1e-12)
    
    def test_evaluation_consistency(self, quadratic_cost_fn):
        """Test that different evaluation methods give consistent results."""
        x = np.random.randn(7)
        
        # Get results from different methods
        cost_only = quadratic_cost_fn.cost(x)
        grad_only = quadratic_cost_fn.gradient(x)
        cost_combined, grad_combined = quadratic_cost_fn.cost_and_gradient(x)
        
        # Check consistency
        npt.assert_allclose(cost_only, cost_combined, rtol=1e-15,
                           err_msg="Cost values inconsistent")
        npt.assert_allclose(grad_only, grad_combined, rtol=1e-15,
                           err_msg="Gradient values inconsistent")
    
    def test_multiple_evaluations(self, quadratic_cost_fn):
        """Test that multiple evaluations give consistent results."""
        x = np.random.randn(5)
        
        # Multiple evaluations should give same result
        results = [quadratic_cost_fn.cost(x) for _ in range(10)]
        
        for result in results[1:]:
            npt.assert_allclose(result, results[0], rtol=1e-15,
                               err_msg="Multiple evaluations inconsistent")
    
    def test_different_input_sizes(self):
        """Test cost function with different input sizes."""
        def cost(x: np.ndarray) -> float:
            return np.sum(x**2)
        
        cost_fn = ro.create_cost_function(cost)
        
        # Test different sizes
        for size in [1, 5, 10, 100]:
            x = np.random.randn(size)
            result = cost_fn.cost(x)
            expected = np.sum(x**2)
            npt.assert_allclose(result, expected, rtol=1e-12,
                               err_msg=f"Cost evaluation failed for size {size}")


class TestGradientValidation:
    """Test the gradient validation functionality."""
    
    def test_finite_difference_validation(self):
        """Test finite difference gradient validation."""
        # Rosenbrock function with known gradient
        def rosenbrock(x: np.ndarray) -> float:
            return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
        
        def rosenbrock_grad(x: np.ndarray) -> np.ndarray:
            grad = np.zeros_like(x)
            grad[:-1] = -400 * x[:-1] * (x[1:] - x[:-1]**2) - 2 * (1 - x[:-1])
            grad[1:] += 200 * (x[1:] - x[:-1]**2)
            return grad
        
        # Should pass validation
        cost_fn = ro.create_cost_function(rosenbrock, rosenbrock_grad, 
                                         validate_gradient=True)
        
        # Test evaluation
        x = np.array([0.5, 0.5, 0.5])
        cost = cost_fn.cost(x)
        grad = cost_fn.gradient(x)
        
        # Check that gradient is approximately correct
        expected_grad = rosenbrock_grad(x)
        npt.assert_allclose(grad, expected_grad, rtol=1e-10)
    
    def test_validation_with_wrong_gradient(self):
        """Test that validation catches incorrect gradients."""
        def cost(x: np.ndarray) -> float:
            return np.sum(x**3)  # Cubic function
        
        def wrong_grad(x: np.ndarray) -> np.ndarray:
            return 2 * x  # Gradient of quadratic, not cubic
        
        # Should fail validation
        with pytest.raises((ValueError, AssertionError)):
            ro.create_cost_function(cost, wrong_grad, validate_gradient=True)
    
    def test_validation_tolerance(self):
        """Test gradient validation with different tolerances."""
        def cost(x: np.ndarray) -> float:
            return 0.5 * np.sum(x**2)
        
        def approx_grad(x: np.ndarray) -> np.ndarray:
            return x * 1.001  # Slightly wrong gradient
        
        # Should fail with tight tolerance
        with pytest.raises((ValueError, AssertionError)):
            ro.create_cost_function(cost, approx_grad, validate_gradient=True)
        
        # But might pass with very loose tolerance (if we add that option)
        # This is for future enhancement
    
    def test_matrix_gradient_validation(self):
        """Test gradient validation for matrix inputs."""
        def matrix_cost(X: np.ndarray) -> float:
            return 0.5 * np.trace(X @ X.T)
        
        def matrix_grad(X: np.ndarray) -> np.ndarray:
            return X
        
        # Should pass validation
        cost_fn = ro.create_cost_function(matrix_cost, matrix_grad,
                                         validate_gradient=True)
        
        # Test with random matrix
        X = np.random.randn(3, 4)
        cost = cost_fn.cost(X)
        grad = cost_fn.gradient(X)
        npt.assert_allclose(grad, X, rtol=1e-10)


class TestCostFunctionPerformance:
    """Test performance characteristics of cost functions."""
    
    @pytest.fixture
    def large_cost_fn(self):
        """Create a cost function for performance testing."""
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            # Simple but realistic computation
            cost = 0.5 * np.sum(x**2) + 0.1 * np.sum(np.sin(x))
            grad = x + 0.1 * np.cos(x)
            return cost, grad
        
        return ro.create_cost_function(cost_and_grad)
    
    @pytest.mark.slow
    def test_evaluation_performance(self, large_cost_fn):
        """Test that cost function evaluation is reasonably fast."""
        x = np.random.randn(1000)
        
        # Warmup
        for _ in range(10):
            large_cost_fn.cost(x)
        
        # Time evaluations
        start = time.perf_counter()
        n_evals = 1000
        for _ in range(n_evals):
            large_cost_fn.cost(x)
        end = time.perf_counter()
        
        time_per_eval = (end - start) / n_evals
        
        # Should be fast (less than 1ms per evaluation for this simple function)
        assert time_per_eval < 1e-3, f"Cost evaluation too slow: {time_per_eval:.6f}s"
    
    @pytest.mark.slow
    def test_gradient_performance(self, large_cost_fn):
        """Test gradient evaluation performance."""
        x = np.random.randn(1000)
        
        # Time combined evaluation (should be faster than separate calls)
        start = time.perf_counter()
        n_evals = 500
        for _ in range(n_evals):
            large_cost_fn.cost_and_gradient(x)
        end = time.perf_counter()
        
        time_per_eval = (end - start) / n_evals
        
        # Should be reasonable
        assert time_per_eval < 2e-3, f"Combined evaluation too slow: {time_per_eval:.6f}s"
    
    def test_memory_usage(self, large_cost_fn):
        """Test that cost function doesn't leak memory."""
        import gc
        
        x = np.random.randn(100)
        
        # Force garbage collection
        gc.collect()
        
        # Multiple evaluations shouldn't significantly increase memory
        # This is a basic test - proper memory profiling would be better
        for _ in range(100):
            result = large_cost_fn.cost(x)
            grad = large_cost_fn.gradient(x)
        
        # Force cleanup
        gc.collect()
        
        # If we get here without crashing, memory usage is probably OK


class TestCostFunctionEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_input(self):
        """Test behavior with empty arrays."""
        def cost(x: np.ndarray) -> float:
            return np.sum(x**2) if len(x) > 0 else 0.0
        
        cost_fn = ro.create_cost_function(cost)
        
        # Test with empty array
        x = np.array([])
        result = cost_fn.cost(x)
        assert result == 0.0, "Cost of empty array should be 0"
    
    def test_single_element_input(self):
        """Test behavior with single-element arrays."""
        def cost(x: np.ndarray) -> float:
            return 0.5 * np.sum(x**2)
        
        cost_fn = ro.create_cost_function(cost)
        
        x = np.array([2.0])
        result = cost_fn.cost(x)
        expected = 0.5 * 4.0
        npt.assert_allclose(result, expected, rtol=1e-12)
    
    def test_very_large_input(self):
        """Test behavior with very large arrays."""
        def cost(x: np.ndarray) -> float:
            return np.mean(x**2)  # Use mean to avoid overflow
        
        cost_fn = ro.create_cost_function(cost)
        
        # Large array
        x = np.random.randn(10000)
        result = cost_fn.cost(x)
        
        # Should be approximately 1 for standard normal
        assert 0.5 < result < 1.5, f"Unexpected result for large array: {result}"
    
    def test_special_values(self):
        """Test behavior with special floating point values."""
        def cost(x: np.ndarray) -> float:
            return np.sum(x**2)
        
        cost_fn = ro.create_cost_function(cost)
        
        # Test with zeros
        x = np.zeros(5)
        result = cost_fn.cost(x)
        assert result == 0.0, "Cost of zero array should be 0"
        
        # Test with very small values
        x = np.full(5, 1e-10)
        result = cost_fn.cost(x)
        expected = 5 * 1e-20
        npt.assert_allclose(result, expected, rtol=1e-12)
    
    def test_function_exceptions(self):
        """Test that Python exceptions in cost functions are handled."""
        def bad_cost(x: np.ndarray) -> float:
            if len(x) > 5:
                raise ValueError("Array too large")
            return np.sum(x**2)
        
        cost_fn = ro.create_cost_function(bad_cost)
        
        # Should work for small arrays
        x_small = np.random.randn(3)
        result = cost_fn.cost(x_small)
        assert isinstance(result, float)
        
        # Should propagate exception for large arrays
        x_large = np.random.randn(10)
        with pytest.raises(ValueError, match="Array too large"):
            cost_fn.cost(x_large)


# Integration tests
class TestCostFunctionIntegration:
    """Integration tests combining cost functions with other components."""
    
    def test_cost_function_with_manifold_operations(self):
        """Test cost function evaluation on manifold points."""
        sphere = ro.manifolds.Sphere(5)
        
        # Rayleigh quotient on sphere
        A = np.random.randn(5, 5)
        A = A + A.T  # Make symmetric
        
        def rayleigh(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(rayleigh, validate_gradient=True)
        
        # Test on manifold points
        for _ in range(10):
            x = sphere.random_point()
            
            # Point should be on sphere
            assert sphere.contains(x), "Random point not on sphere"
            
            # Evaluate cost function
            cost = cost_fn.cost(x)
            grad = cost_fn.gradient(x)
            
            # Results should be finite
            assert np.isfinite(cost), "Cost not finite"
            assert np.all(np.isfinite(grad)), "Gradient not finite"
            
            # Gradient should have same shape as point
            assert grad.shape == x.shape, "Gradient shape mismatch"
    
    def test_cost_function_evaluation_count(self):
        """Test that evaluation counts work correctly."""
        call_count = 0
        
        def counting_cost(x: np.ndarray) -> float:
            nonlocal call_count
            call_count += 1
            return np.sum(x**2)
        
        cost_fn = ro.create_cost_function(counting_cost)
        
        x = np.random.randn(5)
        
        # Reset counter
        call_count = 0
        
        # Multiple evaluations
        for i in range(5):
            cost_fn.cost(x)
            assert call_count == i + 1, f"Call count incorrect: expected {i+1}, got {call_count}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])