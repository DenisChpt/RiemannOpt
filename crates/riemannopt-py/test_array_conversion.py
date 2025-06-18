#!/usr/bin/env python3
"""
Test array conversion between NumPy and Rust to debug the ordering issue.
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


def test_array_ordering():
    """Test how arrays are converted between NumPy and Rust."""
    print("Testing array conversion ordering...")
    print("-" * 60)
    
    n, p = 4, 2
    stiefel = riemannopt.Stiefel(n, p)
    
    # Create a specific matrix pattern to track ordering
    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0]
    ])
    
    print("Original matrix X:")
    print(X)
    print(f"Shape: {X.shape}")
    print(f"Flat (C-order): {X.flatten()}")
    print(f"Flat (F-order): {X.flatten('F')}")
    
    # Project to manifold
    X_proj = stiefel.project(X)
    print("\nProjected matrix:")
    print(X_proj)
    print(f"Orthogonality check: {np.linalg.norm(X_proj.T @ X_proj - np.eye(p), 'fro'):.2e}")
    
    # Create a simple gradient
    G = np.ones((n, p))
    print("\nGradient G:")
    print(G)
    
    # Test Adam step
    adam = riemannopt.Adam(learning_rate=0.001)
    X_orth = stiefel.random_point()
    print(f"\nOrthogonal starting point check: {np.linalg.norm(X_orth.T @ X_orth - np.eye(p), 'fro'):.2e}")
    
    X_new = adam.step(stiefel, X_orth, G)
    print(f"After Adam step: {np.linalg.norm(X_new.T @ X_new - np.eye(p), 'fro'):.2e}")
    
    # Test direct retraction
    V = stiefel.tangent_projection(X_orth, G)
    X_retract = stiefel.retract(X_orth, -0.001 * V)
    print(f"Direct retraction: {np.linalg.norm(X_retract.T @ X_retract - np.eye(p), 'fro'):.2e}")


def test_gradient_conversion_issue():
    """Test if the issue is in gradient conversion."""
    print("\n\nTesting gradient conversion...")
    print("-" * 60)
    
    n, p = 6, 3
    stiefel = riemannopt.Stiefel(n, p)
    
    # Create orthogonal matrix
    X = stiefel.random_point()
    
    # Create a specific gradient pattern
    G = np.zeros((n, p))
    G[0, 0] = 1.0  # Only one non-zero element
    
    print("Gradient with single non-zero element at (0,0):")
    print(G)
    
    # Project to tangent space
    V = stiefel.tangent_projection(X, G)
    print("\nProjected tangent vector:")
    print(V)
    print(f"Tangent space check: {np.linalg.norm(X.T @ V + V.T @ X, 'fro'):.2e}")
    
    # Take Adam step
    adam = riemannopt.Adam(learning_rate=0.1)  # Larger step to see effect
    X_new = adam.step(stiefel, X, G)
    
    print(f"\nOrthogonality after Adam: {np.linalg.norm(X_new.T @ X_new - np.eye(p), 'fro'):.2e}")
    
    # Check the difference
    diff = X_new - X
    print("\nDifference X_new - X:")
    print(diff)


if __name__ == "__main__":
    test_array_ordering()
    test_gradient_conversion_issue()