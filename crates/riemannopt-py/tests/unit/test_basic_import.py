"""Basic import tests for RiemannOpt."""

import pytest
import sys
import os

# Ensure we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))


def test_import_main_module():
    """Test that we can import the main module."""
    import riemannopt
    assert riemannopt is not None


def test_import_submodules():
    """Test that we can import submodules."""
    import riemannopt
    
    # Test core imports
    assert hasattr(riemannopt, 'manifolds')
    assert hasattr(riemannopt, 'optimizers')
    assert hasattr(riemannopt, 'optimize')
    assert hasattr(riemannopt, 'create_cost_function')


def test_version():
    """Test that version is accessible."""
    import riemannopt
    assert hasattr(riemannopt, '__version__')
    assert isinstance(riemannopt.__version__, str)
    
    # Version should be in format X.Y.Z
    parts = riemannopt.__version__.split('.')
    assert len(parts) >= 2


def test_manifolds_available():
    """Test that manifold classes are available."""
    import riemannopt as ro
    
    # Check manifold module exists
    assert hasattr(ro, 'manifolds')
    
    # Check specific manifolds
    expected_manifolds = ['Sphere', 'Stiefel', 'Grassmann']
    for manifold_name in expected_manifolds:
        assert hasattr(ro.manifolds, manifold_name), f"Missing manifold: {manifold_name}"


def test_optimizers_available():
    """Test that optimizer classes are available."""
    import riemannopt as ro
    
    # Check optimizer module exists
    assert hasattr(ro, 'optimizers')
    
    # Check specific optimizers
    expected_optimizers = ['SGD', 'Adam']
    for optimizer_name in expected_optimizers:
        assert hasattr(ro.optimizers, optimizer_name), f"Missing optimizer: {optimizer_name}"


def test_sphere_creation():
    """Test creating a sphere manifold."""
    import riemannopt as ro
    
    # Create sphere
    sphere = ro.manifolds.Sphere(10)
    assert sphere is not None
    assert sphere.dim == 9  # Sphere S^{n-1} has dimension n-1


def test_stiefel_creation():
    """Test creating a Stiefel manifold."""
    import riemannopt as ro
    
    # Create Stiefel manifold
    stiefel = ro.manifolds.Stiefel(10, 3)
    assert stiefel is not None
    assert stiefel.n == 10
    assert stiefel.p == 3


def test_simple_optimization():
    """Test a simple optimization on sphere."""
    import riemannopt as ro
    import numpy as np
    
    # Create sphere
    sphere = ro.manifolds.Sphere(10)
    
    # Create initial point
    x0 = sphere.random_point()
    assert x0.shape == (10,)
    np.testing.assert_allclose(np.linalg.norm(x0), 1.0, rtol=1e-10)
    
    # Define simple cost function
    def cost(x):
        return -x[0]  # Minimize negative of first component
    
    # Run optimization
    result = ro.optimize(
        sphere, cost, x0, 
        optimizer='sgd', 
        learning_rate=0.1,
        max_iterations=50
    )
    
    # Check result structure
    assert isinstance(result, dict)
    assert 'point' in result
    assert 'value' in result  # The API returns 'value' not 'cost'
    assert 'iterations' in result
    assert 'converged' in result  # The API returns 'converged' not 'success'
    
    # Final point should still be on sphere
    final_point = result['point']
    np.testing.assert_allclose(np.linalg.norm(final_point), 1.0, rtol=1e-10)
    
    # First component should be larger (more negative cost)
    assert final_point[0] > x0[0]


@pytest.mark.parametrize("optimizer_name", ["sgd", "adam"])
def test_optimizer_types(optimizer_name):
    """Test different optimizer types."""
    import riemannopt as ro
    import numpy as np
    
    sphere = ro.manifolds.Sphere(5)
    x0 = sphere.random_point()
    
    def cost(x):
        return np.sum(x**2)
    
    # Should not raise
    result = ro.optimize(
        sphere, cost, x0,
        optimizer=optimizer_name,
        max_iterations=10
    )
    
    assert result is not None
    # Handle both dict and OptimizationResult object
    if hasattr(result, 'x'):
        # PyOptimizationResult uses 'x' as property name
        assert result.x is not None
    elif hasattr(result, 'point'):
        assert result.point is not None
    else:
        assert 'point' in result