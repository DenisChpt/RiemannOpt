#!/usr/bin/env python3
"""
Test script for RiemannOpt Python bindings.
"""

import numpy as np
import riemannopt

def test_sphere():
    """Test Sphere manifold operations."""
    print("Testing Sphere manifold...")
    
    # Create sphere
    dim = 10
    sphere = riemannopt.Sphere(dim)
    print(f"Created Sphere(dim={dim})")
    print(f"  Manifold dimension: {sphere.dim}")
    print(f"  Ambient dimension: {sphere.ambient_dim}")
    
    # Random point
    x = sphere.random_point()
    print(f"  Random point norm: {np.linalg.norm(x):.6f}")
    
    # Projection
    y = np.random.randn(dim)
    y_proj = sphere.project(y)
    print(f"  Projection test: {np.linalg.norm(y_proj):.6f}")
    
    # Tangent vector
    v = sphere.random_tangent(x)
    print(f"  Tangent vector inner product with point: {np.dot(x, v):.6e}")
    
    # Retraction
    x_new = sphere.retraction(x, 0.1 * v)
    print(f"  Retraction preserves norm: {np.linalg.norm(x_new):.6f}")
    
    # Distance
    y_on_sphere = sphere.random_point()
    dist = sphere.distance(x, y_on_sphere)
    print(f"  Distance between two points: {dist:.6f}")
    
    print("✓ Sphere tests passed\n")

def test_stiefel():
    """Test Stiefel manifold operations."""
    print("Testing Stiefel manifold...")
    
    # Create Stiefel manifold
    n, p = 10, 5
    stiefel = riemannopt.Stiefel(n, p)
    print(f"Created Stiefel(n={n}, p={p})")
    print(f"  Manifold dimension: {stiefel.dim}")
    
    # Random point
    X = stiefel.random_point()
    X_mat = X.reshape(n, p)
    print(f"  Random point shape: {X_mat.shape}")
    print(f"  Orthogonality check: ||X'X - I|| = {np.linalg.norm(X_mat.T @ X_mat - np.eye(p)):.6e}")
    
    # Projection
    Y = np.random.randn(n * p)
    Y_proj = stiefel.project(Y)
    Y_proj_mat = Y_proj.reshape(n, p)
    print(f"  Projection orthogonality: ||Y'Y - I|| = {np.linalg.norm(Y_proj_mat.T @ Y_proj_mat - np.eye(p)):.6e}")
    
    print("✓ Stiefel tests passed\n")

def test_optimizers():
    """Test optimization functionality."""
    print("Testing optimizers...")
    
    # Create a simple cost function on the sphere
    dim = 3
    sphere = riemannopt.Sphere(dim)
    
    # Target point
    target = sphere.random_point()
    
    # Cost function: distance to target
    def cost(x):
        return 0.5 * np.linalg.norm(x - target)**2
    
    def gradient(x):
        return x - target
    
    # Create cost function object
    cost_fn = riemannopt.CostFunction(cost, gradient)
    
    # Initial point
    x0 = sphere.random_point()
    
    # Test SGD
    print(f"  Initial cost: {cost(x0):.6f}")
    sgd = riemannopt.SGD(sphere, cost_fn, learning_rate=0.1, max_iter=100)
    result = sgd.optimize(x0)
    x_opt, final_cost, info = result
    print(f"  Final cost (SGD): {final_cost:.6f}")
    print(f"  Distance to target: {sphere.distance(x_opt, target):.6f}")
    
    # Test Adam
    adam = riemannopt.Adam(sphere, cost_fn, learning_rate=0.1, max_iter=100)
    result = adam.optimize(x0)
    x_opt, final_cost, info = result
    print(f"  Final cost (Adam): {final_cost:.6f}")
    
    print("✓ Optimizer tests passed\n")

def test_exceptions():
    """Test error handling."""
    print("Testing error handling...")
    
    sphere = riemannopt.Sphere(5)
    
    # Test dimension mismatch
    try:
        x = np.random.randn(10)  # Wrong dimension
        sphere.project(x)
        print("  ERROR: Should have raised exception for dimension mismatch")
    except Exception as e:
        print(f"  ✓ Caught expected error: {type(e).__name__}")
    
    print("✓ Exception tests passed\n")

def main():
    """Run all tests."""
    print("="*60)
    print("RiemannOpt Python Bindings Test Suite")
    print("="*60)
    print(f"Version: {riemannopt.__version__}\n")
    
    # List available classes
    manifolds = [name for name in dir(riemannopt) if name[0].isupper() and not name.endswith('Error')]
    print(f"Available manifolds: {', '.join(manifolds[:5])}")
    
    optimizers = ['SGD', 'Adam', 'LBFGS', 'ConjugateGradient', 'TrustRegions']
    available_optimizers = [opt for opt in optimizers if hasattr(riemannopt, opt)]
    print(f"Available optimizers: {', '.join(available_optimizers)}\n")
    
    # Run tests
    test_sphere()
    test_stiefel()
    test_optimizers()
    test_exceptions()
    
    print("="*60)
    print("All tests completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()