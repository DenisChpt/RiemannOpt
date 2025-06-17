"""
pytest configuration and shared fixtures for RiemannOpt tests.

This module provides common fixtures, utilities, and configuration
for all test modules in the RiemannOpt test suite.
"""

import pytest
import numpy as np
from typing import List, Tuple, Dict, Any
import sys
import os

# Import riemannopt
from test_imports import riemannopt, HAS_RIEMANNOPT

if not HAS_RIEMANNOPT:
    pytest.skip("riemannopt module not installed", allow_module_level=True)


# ============================================================================
# Test Configuration
# ============================================================================

# Tolerances for different types of tests
TOLERANCES = {
    'strict': 1e-12,
    'default': 1e-10,
    'relaxed': 1e-8,
    'numerical': 1e-6,
    'optimization': 1e-4,
}

# Dimension configurations for parametrized tests
DIMENSION_CONFIGS = {
    'tiny': [(2,), (3,), (5,)],
    'small': [(10,), (20,), (50,)],
    'medium': [(100,), (200,), (500,)],
    'large': [(1000,), (2000,), (5000,)],
    'extreme': [(10000,), (50000,), (100000,)],
}

# Stiefel/Grassmann dimension configurations (n, p)
MATRIX_DIMENSION_CONFIGS = {
    'tiny': [(3, 1), (4, 2), (5, 2)],
    'small': [(10, 3), (20, 5), (50, 10)],
    'medium': [(100, 10), (200, 20), (500, 50)],
    'large': [(1000, 50), (2000, 100), (5000, 100)],
}


# ============================================================================
# Fixtures for Manifolds
# ============================================================================

@pytest.fixture
def sphere_factory():
    """Factory fixture for creating Sphere manifolds."""
    def _create_sphere(dim: int):
        return riemannopt.Sphere(dim)
    return _create_sphere


@pytest.fixture
def stiefel_factory():
    """Factory fixture for creating Stiefel manifolds."""
    def _create_stiefel(n: int, p: int):
        return riemannopt.Stiefel(n, p)
    return _create_stiefel


@pytest.fixture
def grassmann_factory():
    """Factory fixture for creating Grassmann manifolds."""
    def _create_grassmann(n: int, p: int):
        return riemannopt.Grassmann(n, p)
    return _create_grassmann


# ============================================================================
# Fixtures for Optimizers
# ============================================================================

@pytest.fixture
def sgd_factory():
    """Factory fixture for creating SGD optimizers."""
    def _create_sgd(step_size: float = 0.01, momentum: float = 0.0, **kwargs):
        return riemannopt.SGD(
            step_size=step_size,
            momentum=momentum,
            max_iterations=kwargs.get('max_iterations', 1000),
            tolerance=kwargs.get('tolerance', 1e-6)
        )
    return _create_sgd


@pytest.fixture
def adam_factory():
    """Factory fixture for creating Adam optimizers."""
    def _create_adam(learning_rate: float = 0.001, **kwargs):
        return riemannopt.Adam(
            learning_rate=learning_rate,
            beta1=kwargs.get('beta1', 0.9),
            beta2=kwargs.get('beta2', 0.999),
            epsilon=kwargs.get('epsilon', 1e-8)
        )
    return _create_adam


# ============================================================================
# Utility Fixtures
# ============================================================================

@pytest.fixture
def random_symmetric_matrix():
    """Generate random symmetric matrices for testing."""
    def _create_matrix(n: int, eigenvalue_range: Tuple[float, float] = (-1.0, 1.0)):
        """
        Create a random symmetric matrix with specified eigenvalue range.
        
        Args:
            n: Matrix dimension
            eigenvalue_range: (min, max) eigenvalues
            
        Returns:
            Symmetric matrix with known eigenvalue range
        """
        # Generate random orthogonal matrix
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        
        # Generate eigenvalues in specified range
        min_eig, max_eig = eigenvalue_range
        eigenvalues = np.random.uniform(min_eig, max_eig, n)
        eigenvalues.sort()
        
        # Construct symmetric matrix
        A = Q @ np.diag(eigenvalues) @ Q.T
        
        # Store eigenvalues as attribute for testing
        A = np.array(A)
        setattr(A, 'eigenvalues', eigenvalues)
        setattr(A, 'eigenvectors', Q)
        
        return A
    
    return _create_matrix


@pytest.fixture
def cost_functions():
    """Common cost functions for optimization tests."""
    class CostFunctions:
        @staticmethod
        def quadratic(A: np.ndarray):
            """Quadratic cost function f(x) = x^T A x."""
            def cost_fn(x: np.ndarray) -> float:
                return float(x.T @ A @ x)
            
            def gradient_fn(x: np.ndarray) -> np.ndarray:
                return 2 * A @ x
            
            return cost_fn, gradient_fn
        
        @staticmethod
        def rayleigh_quotient(A: np.ndarray, B: np.ndarray = None):
            """Rayleigh quotient f(x) = (x^T A x) / (x^T B x)."""
            if B is None:
                B = np.eye(A.shape[0])
                
            def cost_fn(x: np.ndarray) -> float:
                return float((x.T @ A @ x) / (x.T @ B @ x))
            
            def gradient_fn(x: np.ndarray) -> np.ndarray:
                xTBx = x.T @ B @ x
                xTAx = x.T @ A @ x
                return 2 * (A @ x * xTBx - B @ x * xTAx) / (xTBx ** 2)
            
            return cost_fn, gradient_fn
        
        @staticmethod
        def trace_objective(C: np.ndarray):
            """Trace objective for Stiefel: f(X) = -tr(X^T C X)."""
            def cost_fn(X: np.ndarray) -> float:
                return -np.trace(X.T @ C @ X)
            
            def gradient_fn(X: np.ndarray) -> np.ndarray:
                return -2 * C @ X
            
            return cost_fn, gradient_fn
    
    return CostFunctions()


@pytest.fixture
def assert_helpers():
    """Helper functions for common assertions."""
    class AssertHelpers:
        @staticmethod
        def assert_is_on_sphere(x: np.ndarray, tol: float = TOLERANCES['default']):
            """Assert that x is on the unit sphere."""
            norm = np.linalg.norm(x)
            assert abs(norm - 1.0) < tol, f"Point not on unit sphere: ||x|| = {norm}"
        
        @staticmethod
        def assert_is_orthogonal(X: np.ndarray, tol: float = TOLERANCES['default']):
            """Assert that X has orthonormal columns."""
            n, p = X.shape
            gram = X.T @ X
            eye = np.eye(p)
            deviation = np.linalg.norm(gram - eye, 'fro')
            assert deviation < tol, f"Matrix not orthogonal: ||X^T X - I|| = {deviation}"
        
        @staticmethod
        def assert_in_tangent_space_sphere(x: np.ndarray, v: np.ndarray, 
                                         tol: float = TOLERANCES['default']):
            """Assert that v is in tangent space of sphere at x."""
            inner = np.dot(x, v)
            assert abs(inner) < tol, f"Vector not in tangent space: <x,v> = {inner}"
        
        @staticmethod
        def assert_in_tangent_space_stiefel(X: np.ndarray, V: np.ndarray,
                                           tol: float = TOLERANCES['default']):
            """Assert that V is in tangent space of Stiefel at X."""
            XtV = X.T @ V
            VtX = V.T @ X
            skew_norm = np.linalg.norm(XtV + VtX, 'fro')
            assert skew_norm < tol, f"Vector not in tangent space: ||X^T V + V^T X|| = {skew_norm}"
        
        @staticmethod
        def assert_in_tangent_space_grassmann(X: np.ndarray, V: np.ndarray,
                                            tol: float = TOLERANCES['default']):
            """Assert that V is in horizontal space of Grassmann at X."""
            XtV = X.T @ V
            norm = np.linalg.norm(XtV, 'fro')
            assert norm < tol, f"Vector not in horizontal space: ||X^T V|| = {norm}"
    
    return AssertHelpers()


@pytest.fixture
def performance_timer():
    """Context manager for timing operations."""
    import time
    from contextlib import contextmanager
    
    class TimingResults:
        def __init__(self):
            self.times = []
            
        def add(self, elapsed: float):
            self.times.append(elapsed)
            
        @property
        def mean(self) -> float:
            return np.mean(self.times) if self.times else 0.0
            
        @property
        def std(self) -> float:
            return np.std(self.times) if len(self.times) > 1 else 0.0
            
        @property
        def min(self) -> float:
            return np.min(self.times) if self.times else 0.0
            
        @property
        def max(self) -> float:
            return np.max(self.times) if self.times else 0.0
    
    @contextmanager
    def timer():
        results = TimingResults()
        
        def time_it(iterations: int = 1):
            def decorator(func):
                def wrapper(*args, **kwargs):
                    for _ in range(iterations):
                        start = time.perf_counter()
                        result = func(*args, **kwargs)
                        elapsed = time.perf_counter() - start
                        results.add(elapsed)
                    return result
                return wrapper
            return decorator
        
        yield time_it, results
    
    return timer


# ============================================================================
# Test Data Generators
# ============================================================================

@pytest.fixture
def test_data_generator():
    """Generate various test data patterns."""
    class TestDataGenerator:
        @staticmethod
        def random_points_on_sphere(n_dim: int, n_points: int) -> np.ndarray:
            """Generate random points on the unit sphere."""
            points = np.random.randn(n_points, n_dim)
            norms = np.linalg.norm(points, axis=1, keepdims=True)
            return points / norms
        
        @staticmethod
        def random_stiefel_matrices(n: int, p: int, n_matrices: int) -> List[np.ndarray]:
            """Generate random orthonormal matrices."""
            matrices = []
            for _ in range(n_matrices):
                M = np.random.randn(n, p)
                Q, _ = np.linalg.qr(M)
                matrices.append(Q[:, :p])
            return matrices
        
        @staticmethod
        def tangent_vectors_sphere(x: np.ndarray, n_vectors: int) -> List[np.ndarray]:
            """Generate random tangent vectors at point x on sphere."""
            vectors = []
            for _ in range(n_vectors):
                v = np.random.randn(len(x))
                v = v - np.dot(x, v) * x  # Project to tangent space
                vectors.append(v)
            return vectors
        
        @staticmethod
        def tangent_vectors_stiefel(X: np.ndarray, n_vectors: int) -> List[np.ndarray]:
            """Generate random tangent vectors at point X on Stiefel."""
            n, p = X.shape
            vectors = []
            for _ in range(n_vectors):
                V = np.random.randn(n, p)
                # Project to tangent space: V - X(X^T V + V^T X)/2
                XtV = X.T @ V
                V = V - X @ ((XtV + XtV.T) / 2)
                vectors.append(V)
            return vectors
    
    return TestDataGenerator()


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "numerical: marks numerical accuracy tests")
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")
    config.addinivalue_line("markers", "integration: marks integration tests")
    config.addinivalue_line("markers", "manifold: marks manifold-specific tests")
    config.addinivalue_line("markers", "optimizer: marks optimizer-specific tests")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names."""
    for item in items:
        # Add markers based on test module names
        if "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.benchmark)
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        if "numerical" in item.nodeid:
            item.add_marker(pytest.mark.numerical)