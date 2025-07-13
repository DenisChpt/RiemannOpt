#!/usr/bin/env python3
"""Test script for basic SGD implementation."""

import sys
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, '../../../target/debug')

try:
    import riemannopt as ro
    print("Successfully imported riemannopt")
    
    # Test 1: Create a sphere manifold
    sphere = ro.manifolds.Sphere(10)
    print(f"Created sphere: {sphere}")
    
    # Test 2: Generate a random point
    x0 = sphere.random_point()
    print(f"Random point shape: {x0.shape}")
    print(f"Random point norm: {np.linalg.norm(x0)}")
    
    # Test 3: Create a simple cost function
    A = np.random.randn(10, 10)
    A = A.T @ A  # Make it symmetric positive definite
    
    def cost_fn(x):
        return -x.T @ A @ x  # Negative to maximize
    
    def grad_fn(x):
        return -2 * A @ x
    
    # Test 4: Create cost function wrapper
    cost = ro.create_cost_function(
        cost=cost_fn,
        gradient=grad_fn,
        dimension=10
    )
    print(f"Created cost function: {cost}")
    
    # Test 5: Create SGD optimizer
    optimizer = ro.optimizers.SGD(learning_rate=0.01, momentum=0.0)
    print(f"Created optimizer: {optimizer}")
    
    # Test 6: Run optimization (simple version)
    result = optimizer.optimize_sphere(cost, sphere, x0, max_iterations=100)
    print(f"Optimization result: {result}")
    
except ImportError as e:
    print(f"Failed to import riemannopt: {e}")
    print("Make sure to build the library first with: cd .. && maturin develop")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()