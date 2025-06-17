"""pytest configuration for RiemannOpt tests."""

import pytest
import numpy as np
from typing import Callable
import sys
import os

# Add parent directory to path for development testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# Fixtures for manifolds
@pytest.fixture
def sphere_manifold():
    """Create a sphere manifold for testing."""
    import riemannopt as ro
    return ro.manifolds.Sphere(10)


@pytest.fixture
def stiefel_manifold():
    """Create a Stiefel manifold for testing."""
    import riemannopt as ro
    return ro.manifolds.Stiefel(10, 3)


@pytest.fixture
def grassmann_manifold():
    """Create a Grassmann manifold for testing."""
    import riemannopt as ro
    return ro.manifolds.Grassmann(10, 3)


# Fixtures for cost functions
@pytest.fixture
def quadratic_cost() -> Callable:
    """Simple quadratic cost function."""
    def cost(x: np.ndarray) -> float:
        return 0.5 * np.sum(x**2)
    return cost


@pytest.fixture
def quadratic_cost_with_grad() -> Callable:
    """Quadratic cost function that returns gradient."""
    def cost_and_grad(x: np.ndarray):
        cost = 0.5 * np.sum(x**2)
        grad = x
        return cost, grad
    return cost_and_grad


@pytest.fixture
def rosenbrock_function() -> Callable:
    """Rosenbrock function for testing."""
    def cost(x: np.ndarray) -> float:
        return np.sum(100 * (x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2)
    return cost


@pytest.fixture
def rayleigh_quotient(sphere_manifold):
    """Rayleigh quotient cost function on sphere."""
    # Random symmetric matrix
    np.random.seed(42)
    n = sphere_manifold.dim + 1
    A = np.random.randn(n, n)
    A = A + A.T  # Make symmetric
    
    def cost_and_grad(x: np.ndarray):
        Ax = A @ x
        cost = x.T @ Ax
        grad = 2 * Ax
        return cost, grad
    
    return cost_and_grad


# Fixtures for optimizers
@pytest.fixture
def sgd_optimizer(sphere_manifold):
    """Create SGD optimizer."""
    import riemannopt as ro
    return ro.optimizers.SGD(sphere_manifold, learning_rate=0.1)


@pytest.fixture
def adam_optimizer(sphere_manifold):
    """Create Adam optimizer."""
    import riemannopt as ro
    return ro.optimizers.Adam(sphere_manifold, learning_rate=0.01)


# Test data fixtures
@pytest.fixture
def random_point_on_sphere():
    """Generate random point on unit sphere."""
    x = np.random.randn(10)
    return x / np.linalg.norm(x)


@pytest.fixture
def random_stiefel_point():
    """Generate random point on Stiefel manifold."""
    X = np.random.randn(10, 3)
    Q, _ = np.linalg.qr(X)
    return Q


# Configuration fixtures
@pytest.fixture
def clean_config():
    """Reset configuration to defaults after test."""
    import riemannopt as ro
    
    # Store original config
    original = ro.get_config().__dict__.copy()
    
    yield
    
    # Restore original config
    ro.set_config(**original)


# Markers
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "benchmark: performance benchmarks")
    config.addinivalue_line("markers", "gpu: requires GPU")
    config.addinivalue_line("markers", "visualization: requires matplotlib")
    config.addinivalue_line("markers", "torch: requires PyTorch")
    config.addinivalue_line("markers", "jax: requires JAX")


# Test utilities
@pytest.fixture
def assert_allclose():
    """Assert that arrays are close."""
    def _assert_allclose(actual, expected, rtol=1e-7, atol=1e-9, err_msg=''):
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, err_msg=err_msg)
    return _assert_allclose


@pytest.fixture
def assert_on_manifold():
    """Assert that point is on manifold."""
    def _assert_on_manifold(manifold, point, tol=1e-10):
        assert manifold.check_point(point), f"Point not on {manifold.__class__.__name__}"
        
        # Additional checks for specific manifolds
        if hasattr(manifold, 'n') and hasattr(manifold, 'p'):  # Stiefel
            XtX = point.T @ point
            I = np.eye(manifold.p)
            np.testing.assert_allclose(XtX, I, atol=tol)
        elif hasattr(manifold, 'dim') and point.ndim == 1:  # Sphere
            norm = np.linalg.norm(point)
            np.testing.assert_allclose(norm, 1.0, atol=tol)
    
    return _assert_on_manifold


@pytest.fixture
def assert_in_tangent_space():
    """Assert that vector is in tangent space."""
    def _assert_in_tangent_space(manifold, point, vector, tol=1e-10):
        assert manifold.check_vector(point, vector), \
            f"Vector not in tangent space of {manifold.__class__.__name__}"
    
    return _assert_in_tangent_space


# Performance testing utilities
@pytest.fixture
def benchmark_manifold_operation():
    """Benchmark a manifold operation."""
    import time
    
    def _benchmark(operation, *args, n_runs=100, warmup=10):
        # Warmup
        for _ in range(warmup):
            operation(*args)
        
        # Time runs
        times = []
        for _ in range(n_runs):
            start = time.perf_counter()
            operation(*args)
            times.append(time.perf_counter() - start)
        
        times = np.array(times)
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times),
        }
    
    return _benchmark