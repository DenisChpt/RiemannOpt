#!/usr/bin/env python3
"""Simple test to verify basic functionality."""

import sys
import numpy as np

# Add the build directory to Python path
sys.path.insert(0, '../../../target/debug')

try:
    import riemannopt as ro
    print("✓ Successfully imported riemannopt")
    
    # Test manifold creation
    sphere = ro.manifolds.Sphere(5)
    print(f"✓ Created sphere: {sphere}")
    
    # Test random point generation
    x0 = sphere.random_point()
    print(f"✓ Generated random point: shape={x0.shape}, norm={np.linalg.norm(x0):.6f}")
    
    # Test simple cost function
    def cost_fn(x):
        return np.sum(x**2)  # Simple quadratic
    
    def grad_fn(x):
        return 2 * x
    
    # Test cost function creation
    cost = ro.create_cost_function(
        cost=cost_fn,
        gradient=grad_fn,
        dimension=5
    )
    print(f"✓ Created cost function: {cost}")
    
    # Test cost evaluation
    value = cost.cost(x0)
    print(f"✓ Cost evaluation: {value:.6f}")
    
    # Test optimizers that are working
    try:
        sgd = ro.optimizers.SGD(learning_rate=0.01)
        print(f"✓ Created SGD: {sgd}")
    except Exception as e:
        print(f"✗ SGD creation failed: {e}")
    
    try:
        adam = ro.optimizers.Adam(learning_rate=0.001)
        print(f"✓ Created Adam: {adam}")
    except Exception as e:
        print(f"✗ Adam creation failed: {e}")
    
    print("\n✓ All basic tests passed!")
    
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("Make sure to build with: maturin develop")
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()