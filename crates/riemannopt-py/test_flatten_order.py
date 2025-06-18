#!/usr/bin/env python3
"""
Test to understand the flattening order issue in the optimizer.
"""

import numpy as np
import sys
import os

# Add the build directory to Python path
sys.path.insert(0, os.path.abspath('.'))

try:
    import riemannopt
except ImportError:
    print("riemannopt module not found. Make sure to build with 'maturin develop'")
    sys.exit(1)


def test_flattening_issue():
    """Test if flattening order is causing the issue."""
    print("Testing flattening order issue...")
    print("-" * 60)
    
    n, p = 4, 2
    
    # Create a matrix with clear pattern
    X = np.array([
        [1.0, 5.0],
        [2.0, 6.0],
        [3.0, 7.0],
        [4.0, 8.0]
    ], dtype=np.float64)
    
    print("Original matrix X:")
    print(X)
    print(f"\nRow-major flatten (C-order): {X.flatten('C')}")
    print(f"Column-major flatten (F-order): {X.flatten('F')}")
    
    # In the Rust code, it does:
    # let point_vec = DVector::from_column_slice(point_array.as_slice()?);
    # This means it's taking the row-major NumPy array and treating it as column-major
    
    # Let's simulate what might be happening
    flat_c = X.flatten('C')  # NumPy default: [1, 5, 2, 6, 3, 7, 4, 8]
    
    # If Rust interprets this as column-major for a 4x2 matrix, it would read:
    # Column 1: [1, 5, 2, 6]
    # Column 2: [3, 7, 4, 8]
    # Resulting in matrix:
    rust_interpretation = np.array([
        [1.0, 3.0],
        [5.0, 7.0],
        [2.0, 4.0],
        [6.0, 8.0]
    ])
    
    print("\nWhat Rust might see (if misinterpreting order):")
    print(rust_interpretation)
    
    # The correct way would be to ensure consistent ordering
    print("\nTo fix this, we need to ensure consistent matrix layout.")


def test_stiefel_internals():
    """Test Stiefel manifold internal representation."""
    print("\n\nTesting Stiefel internals...")
    print("-" * 60)
    
    n, p = 3, 2
    stiefel = riemannopt.Stiefel(n, p)
    
    # Create a specific orthogonal matrix
    X = np.array([
        [1/np.sqrt(2), 0],
        [0, 1],
        [1/np.sqrt(2), 0]
    ], dtype=np.float64)
    
    print("Test matrix X:")
    print(X)
    print(f"Is orthogonal: {np.allclose(X.T @ X, np.eye(p))}")
    
    # Create a gradient that should not change X much
    G = np.zeros((n, p))
    G[0, 1] = 0.1  # Small gradient in one component
    
    print("\nGradient G:")
    print(G)
    
    # Test with SGD first (which works)
    sgd = riemannopt.SGD(step_size=0.01)
    X_sgd = sgd.step(stiefel, X, G)
    print(f"\nSGD orthogonality: {np.linalg.norm(X_sgd.T @ X_sgd - np.eye(p), 'fro'):.2e}")
    
    # Test with Adam
    adam = riemannopt.Adam(learning_rate=0.01)
    X_adam = adam.step(stiefel, X, G)
    print(f"Adam orthogonality: {np.linalg.norm(X_adam.T @ X_adam - np.eye(p), 'fro'):.2e}")
    
    print("\nDifference between SGD and Adam results:")
    print(np.linalg.norm(X_sgd - X_adam, 'fro'))


if __name__ == "__main__":
    test_flattening_issue()
    test_stiefel_internals()