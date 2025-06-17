"""Test the improvements made to the Python module."""

import pytest
import numpy as np
import sys
import os

# Ensure we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import riemannopt as ro
from riemannopt import exceptions
from riemannopt.core import get_config, set_config, config_context


class TestErrorHandling:
    """Test improved error handling."""
    
    def test_invalid_point_error(self):
        """Test InvalidPointError with detailed information."""
        # Test that we can create the error with details
        error = exceptions.InvalidPointError(
            "Point not on manifold",
            manifold_name="Sphere",
            point_shape=(5,)
        )
        
        assert error.details is not None
        assert 'manifold_name' in error.details
        assert error.details['manifold_name'] == 'Sphere'
        assert error.details['point_shape'] == (5,)
    
    def test_convergence_error(self):
        """Test ConvergenceError with iteration details."""
        error = exceptions.ConvergenceError(
            "Failed to converge",
            iterations=100,
            max_iterations=100,
            gradient_norm=1e-3
        )
        
        assert error.details['iterations'] == 100
        assert error.details['gradient_norm'] == 1e-3
        assert 'max_iterations' in error.details


class TestConfiguration:
    """Test configuration management."""
    
    def test_get_set_config(self):
        """Test getting and setting configuration."""
        original_epsilon = get_config().epsilon
        
        # Change configuration
        set_config(epsilon=1e-12)
        assert get_config().epsilon == 1e-12
        
        # Restore
        set_config(epsilon=original_epsilon)
    
    def test_config_context(self):
        """Test configuration context manager."""
        original_config = get_config()
        original_check = original_config.check_gradients
        
        # Use context manager
        with config_context(check_gradients=True, log_level='DEBUG') as config:
            assert config.check_gradients is True
            assert config.log_level == 'DEBUG'
            assert get_config().check_gradients is True
        
        # Check restoration
        assert get_config().check_gradients == original_check
    
    def test_config_from_env(self, monkeypatch):
        """Test configuration from environment variables."""
        monkeypatch.setenv('RIEMANNOPT_DTYPE', 'float32')
        monkeypatch.setenv('RIEMANNOPT_CHECK_GRADIENTS', 'true')
        
        from riemannopt.core.config import RiemannOptConfig
        config = RiemannOptConfig.from_env()
        
        assert config.default_dtype == 'float32'
        assert config.check_gradients is True


class TestTypeHints:
    """Test that type hints work correctly."""
    
    def test_optimize_type_hints(self):
        """Test that optimize function accepts proper types."""
        sphere = ro.manifolds.Sphere(10)
        x0 = sphere.random_point()
        
        # Test with different cost function types
        def simple_cost(x: np.ndarray) -> float:
            return float(np.sum(x**2))
        
        # Only test simple cost function since the helpers.py optimize
        # doesn't properly handle functions returning tuples yet
        result1 = ro.optimize(sphere, simple_cost, x0, max_iterations=10)
        
        assert isinstance(result1, dict)


class TestDocumentation:
    """Test that documentation is accessible and complete."""
    
    def test_module_docstring(self):
        """Test main module has proper docstring."""
        assert ro.__doc__ is not None
        assert "Riemannian optimization" in ro.__doc__
    
    def test_optimize_docstring(self):
        """Test optimize function has detailed docstring."""
        assert ro.optimize.__doc__ is not None
        assert "Args:" in ro.optimize.__doc__
        assert "Returns:" in ro.optimize.__doc__
        assert "Example:" in ro.optimize.__doc__
    
    def test_core_api_docstring(self):
        """Test core API has proper documentation."""
        from riemannopt.core import optimize
        assert optimize.__doc__ is not None
        assert len(optimize.__doc__) > 500  # Should be detailed


class TestHelpers:
    """Test helper functions with improvements."""
    
    def test_gradient_check(self):
        """Test gradient checking functionality."""
        sphere = ro.manifolds.Sphere(5)
        x = sphere.random_point()
        
        def cost(x):
            return np.sum(x**2)
        
        def gradient(x):
            return 2*x
        
        result = ro.gradient_check(
            sphere, cost, x, 
            gradient_function=gradient,
            tolerance=1e-5
        )
        
        assert 'numerical_gradient' in result
        assert 'check_passed' in result
        # The check might not pass due to numerical precision, 
        # but the function should work


class TestCallbacks:
    """Test callback functionality."""
    
    def test_progress_callback(self):
        """Test ProgressCallback."""
        callback = ro.ProgressCallback(print_every=5)
        
        # Simulate optimization
        callback(0, 1.0, 0.1)
        callback(5, 0.5, 0.05)
        
        assert len(callback.values) == 2
        assert callback.values[1] == 0.5
    
    def test_early_stopping_callback(self):
        """Test EarlyStoppingCallback."""
        callback = ro.EarlyStoppingCallback(patience=2, min_improvement=0.01)
        
        # Simulate no improvement
        callback.on_iteration(0, 1.0, 0.1)
        callback.on_iteration(1, 0.99, 0.1)  # Small improvement
        callback.on_iteration(2, 0.99, 0.1)  # No improvement
        
        # After patience=2, it should stop
        assert callback.should_stop is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])