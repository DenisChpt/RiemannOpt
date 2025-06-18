#!/usr/bin/env python3
"""
Test script to verify Stiefel manifold retraction preserves orthogonality.

This script tests whether the retraction operation correctly maintains
the orthogonality constraint X^T X = I.
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


def test_retraction_preserves_orthogonality():
    """Test that retraction preserves orthogonality."""
    print("Testing Stiefel manifold retraction...")
    print("-" * 60)
    
    n, p = 10, 3
    stiefel = riemannopt.Stiefel(n, p)
    
    # Generate random orthogonal matrix
    X = stiefel.random_point()
    print(f"Initial point orthogonality error: {np.linalg.norm(X.T @ X - np.eye(p), 'fro'):.2e}")
    
    # Generate random tangent vector
    V = stiefel.random_tangent(X)
    
    # Check if V is in tangent space
    XtV = X.T @ V
    VtX = V.T @ X
    tangent_error = np.linalg.norm(XtV + VtX, 'fro')
    print(f"Tangent vector error (should be ~0): {tangent_error:.2e}")
    
    # Test retraction with different step sizes
    step_sizes = [0.001, 0.01, 0.1, 1.0]
    
    for step_size in step_sizes:
        Y = stiefel.retract(X, step_size * V)
        orth_error = np.linalg.norm(Y.T @ Y - np.eye(p), 'fro')
        print(f"Step size {step_size}: orthogonality error = {orth_error:.2e}")


def test_gradient_projection():
    """Test gradient projection to tangent space."""
    print("\n\nTesting gradient projection...")
    print("-" * 60)
    
    n, p = 10, 3
    stiefel = riemannopt.Stiefel(n, p)
    
    X = stiefel.random_point()
    
    # Create arbitrary gradient
    G = np.random.randn(n, p)
    
    # Project to tangent space
    V = stiefel.tangent_projection(X, G)
    
    # Check if V is in tangent space
    XtV = X.T @ V
    VtX = V.T @ X
    tangent_error = np.linalg.norm(XtV + VtX, 'fro')
    print(f"Projected gradient tangent space error: {tangent_error:.2e}")
    
    # Compare with manual projection
    XtG = X.T @ G
    sym_part = (XtG + XtG.T) / 2
    V_manual = G - X @ sym_part
    
    print(f"Difference between manual and library projection: {np.linalg.norm(V - V_manual, 'fro'):.2e}")


def test_adam_single_step():
    """Test a single Adam step in detail."""
    print("\n\nTesting single Adam step...")
    print("-" * 60)
    
    n, p = 10, 3
    stiefel = riemannopt.Stiefel(n, p)
    adam = riemannopt.Adam(learning_rate=0.001)
    
    # Initial point
    X = stiefel.random_point()
    print(f"Initial orthogonality: {np.linalg.norm(X.T @ X - np.eye(p), 'fro'):.2e}")
    
    # Create gradient
    G = np.random.randn(n, p) * 10  # Large gradient to see the effect
    print(f"Gradient norm: {np.linalg.norm(G, 'fro'):.3f}")
    
    # Project gradient manually
    XtG = X.T @ G
    sym_part = (XtG + XtG.T) / 2
    V_tangent = G - X @ sym_part
    print(f"Tangent gradient norm: {np.linalg.norm(V_tangent, 'fro'):.3f}")
    
    # Check tangent space condition
    XtV = X.T @ V_tangent
    VtX = V_tangent.T @ X
    print(f"Manual tangent space check: {np.linalg.norm(XtV + VtX, 'fro'):.2e}")
    
    # Take Adam step
    X_new = adam.step(stiefel, X, G)
    
    # Check orthogonality
    orth_error = np.linalg.norm(X_new.T @ X_new - np.eye(p), 'fro')
    print(f"\nOrthogonality after Adam step: {orth_error:.2e}")
    
    # Distance moved
    print(f"Distance moved: {np.linalg.norm(X_new - X, 'fro'):.6f}")
    
    # Try manual retraction with projected gradient
    X_manual = stiefel.retract(X, -0.001 * V_tangent)
    manual_orth_error = np.linalg.norm(X_manual.T @ X_manual - np.eye(p), 'fro')
    print(f"\nManual retraction orthogonality: {manual_orth_error:.2e}")
    print(f"Difference between Adam and manual: {np.linalg.norm(X_new - X_manual, 'fro'):.2e}")


if __name__ == "__main__":
    test_retraction_preserves_orthogonality()
    test_gradient_projection()
    test_adam_single_step()