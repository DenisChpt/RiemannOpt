#!/usr/bin/env python3
"""Basic test script for RiemannOpt Python bindings."""

import numpy as np
import sys

# This would normally import the compiled module
# For now, we'll just test that the structure makes sense

def test_sphere_optimization():
    """Test optimization on sphere manifold."""
    print("Testing sphere optimization...")
    
    # Create a 10-dimensional sphere
    n = 10
    
    # Random initial point (will be projected to sphere)
    x0 = np.random.randn(n)
    x0 = x0 / np.linalg.norm(x0)  # Project to sphere
    
    # Define cost function: f(x) = -x[0] (maximize first component)
    def cost(x):
        return -x[0]
    
    def gradient(x):
        g = np.zeros_like(x)
        g[0] = -1
        return g
    
    print(f"Initial point norm: {np.linalg.norm(x0):.6f}")
    print(f"Initial cost: {cost(x0):.6f}")
    
    # The optimal point should be [1, 0, 0, ..., 0]
    x_opt = np.zeros(n)
    x_opt[0] = 1.0
    print(f"Optimal cost: {cost(x_opt):.6f}")

def test_stiefel_optimization():
    """Test optimization on Stiefel manifold."""
    print("\nTesting Stiefel optimization...")
    
    # Create a Stiefel manifold St(10, 3)
    n, p = 10, 3
    
    # Random initial point
    X0 = np.random.randn(n, p)
    X0, _ = np.linalg.qr(X0)  # Project to Stiefel
    
    # Define cost function: trace of X^T A X for some matrix A
    A = np.random.randn(n, n)
    A = A.T @ A  # Make it symmetric positive definite
    
    def cost(X):
        return np.trace(X.T @ A @ X)
    
    def gradient(X):
        return 2 * A @ X
    
    print(f"Initial orthogonality check: {np.linalg.norm(X0.T @ X0 - np.eye(p)):.6e}")
    print(f"Initial cost: {cost(X0):.6f}")

def main():
    """Run basic tests."""
    print("=== RiemannOpt Python Bindings Test ===\n")
    
    test_sphere_optimization()
    test_stiefel_optimization()
    
    print("\nNote: This is a placeholder test. The actual module needs to be compiled with maturin.")
    print("To build: cd crates/riemannopt-py && maturin develop")

if __name__ == "__main__":
    main()