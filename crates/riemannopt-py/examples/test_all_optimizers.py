#!/usr/bin/env python3
"""Test script for all optimizers."""

import sys
import numpy as np
import time

# Add the build directory to Python path
sys.path.insert(0, '../../../target/debug')

def test_optimizer(optimizer_name, optimizer, sphere, cost_fn, x0, max_iter=50):
    """Test a single optimizer."""
    print(f"\n=== Testing {optimizer_name} ===")
    try:
        start_time = time.time()
        
        if hasattr(optimizer, 'optimize_sphere'):
            result = optimizer.optimize_sphere(cost_fn, sphere, x0, max_iter)
        else:
            print(f"  Skipping {optimizer_name}: optimize_sphere method not found")
            return
            
        end_time = time.time()
        
        print(f"  ✓ Completed in {end_time - start_time:.3f}s")
        if isinstance(result, dict):
            print(f"  Final value: {result.get('value', 'N/A'):.6f}")
            print(f"  Iterations: {result.get('iterations', 'N/A')}")
            print(f"  Converged: {result.get('converged', 'N/A')}")
        else:
            print(f"  Result type: {type(result)}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    try:
        import riemannopt as ro
        print("Successfully imported riemannopt")
        
        # Create test problem
        np.random.seed(42)
        dim = 10
        sphere = ro.manifolds.Sphere(dim)
        
        # Create a quadratic cost function
        A = np.random.randn(dim, dim)
        A = A.T @ A  # Make symmetric positive definite
        
        def cost_fn(x):
            return x.T @ A @ x
            
        def grad_fn(x):
            return 2 * A @ x
        
        # Create cost function wrapper
        cost = ro.create_cost_function(
            cost=cost_fn,
            gradient=grad_fn,
            dimension=dim
        )
        print(f"Created cost function: {cost}")
        
        # Generate initial point
        x0 = sphere.random_point()
        print(f"Initial point shape: {x0.shape}, norm: {np.linalg.norm(x0):.6f}")
        
        # Test all optimizers
        optimizers = [
            ("SGD", lambda: ro.optimizers.SGD(learning_rate=0.01, momentum=0.0)),
            ("SGD + Momentum", lambda: ro.optimizers.SGD(learning_rate=0.01, momentum=0.9)),
            ("Adam", lambda: ro.optimizers.Adam(learning_rate=0.001)),
            # ("LBFGS", lambda: ro.optimizers.LBFGS(memory_size=5)),
            # ("ConjugateGradient", lambda: ro.optimizers.ConjugateGradient()),
            # ("TrustRegion", lambda: ro.optimizers.TrustRegion()),
            # ("Newton", lambda: ro.optimizers.Newton()),
        ]
        
        print(f"\nTesting {len(optimizers)} optimizers...")
        
        for name, optimizer_factory in optimizers:
            try:
                optimizer = optimizer_factory()
                test_optimizer(name, optimizer, sphere, cost, x0.copy())
            except Exception as e:
                print(f"\n=== {name} ===")
                print(f"  ✗ Failed to create: {e}")
        
        # Test high-level API
        print(f"\n=== Testing High-Level API ===")
        try:
            result = ro.optimizers.optimize(
                cost_function=cost,
                manifold=sphere,
                initial_point=x0,
                optimizer="Adam",
                max_iterations=100
            )
            print(f"  ✓ High-level API works!")
            print(f"  Result type: {type(result)}")
        except Exception as e:
            print(f"  ✗ High-level API failed: {e}")
        
        print(f"\n=== Summary ===")
        print("Basic functionality tests completed.")
        print("For full functionality, all optimizers need to be compiled successfully.")
        
    except ImportError as e:
        print(f"Failed to import riemannopt: {e}")
        print("Make sure to build the library first with: maturin develop")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()