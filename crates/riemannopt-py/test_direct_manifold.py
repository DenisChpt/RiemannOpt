#!/usr/bin/env python3
"""
Test Stiefel manifold methods directly to isolate the issue.
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


def test_manifold_methods():
    """Test individual manifold methods."""
    print("Testing Stiefel manifold methods...")
    print("-" * 60)
    
    n, p = 4, 2
    stiefel = riemannopt.Stiefel(n, p)
    
    # Create a simple orthogonal matrix
    X = np.zeros((n, p))
    X[0, 0] = 1.0
    X[1, 1] = 1.0
    
    print("Initial matrix X:")
    print(X)
    print(f"Is orthogonal: {np.allclose(X.T @ X, np.eye(p))}")
    
    # Test 1: Project a non-orthogonal matrix
    Y = X + 0.1 * np.random.randn(n, p)
    print("\nNon-orthogonal matrix Y:")
    print(f"Orthogonality error: {np.linalg.norm(Y.T @ Y - np.eye(p), 'fro'):.3f}")
    
    Y_proj = stiefel.project(Y)
    print(f"After projection: {np.linalg.norm(Y_proj.T @ Y_proj - np.eye(p), 'fro'):.2e}")
    
    # Test 2: Tangent projection
    V = np.random.randn(n, p)
    V_tan = stiefel.tangent_projection(X, V)
    
    # Check tangent space condition
    XtV = X.T @ V_tan
    VtX = V_tan.T @ X
    print(f"\nTangent projection error: {np.linalg.norm(XtV + VtX, 'fro'):.2e}")
    
    # Test 3: Retraction with small step
    step_size = 0.01
    X_new = stiefel.retract(X, step_size * V_tan)
    print(f"\nRetraction orthogonality: {np.linalg.norm(X_new.T @ X_new - np.eye(p), 'fro'):.2e}")
    
    # Test 4: Check if the issue is in gradient conversion
    # Let's manually do what the optimizer should do
    G = np.random.randn(n, p)  # Euclidean gradient
    
    # The optimizer should:
    # 1. Convert to Riemannian gradient (project to tangent space)
    G_riem = stiefel.tangent_projection(X, G)
    
    # 2. Take a step in the negative gradient direction
    learning_rate = 0.001
    update = -learning_rate * G_riem
    
    # 3. Retract back to manifold
    X_manual = stiefel.retract(X, update)
    
    print(f"\nManual optimization step orthogonality: {np.linalg.norm(X_manual.T @ X_manual - np.eye(p), 'fro'):.2e}")


def test_optimizer_gradient_path():
    """Test the exact path the gradient takes in the optimizer."""
    print("\n\nTracing optimizer gradient path...")
    print("-" * 60)
    
    n, p = 3, 2
    stiefel = riemannopt.Stiefel(n, p)
    
    # Simple orthogonal matrix
    X = np.array([[1.0, 0.0],
                  [0.0, 1.0],
                  [0.0, 0.0]], dtype=np.float64)
    
    # Simple gradient
    G = np.array([[0.0, 1.0],
                  [1.0, 0.0],
                  [0.0, 0.0]], dtype=np.float64)
    
    print("X:")
    print(X)
    print("\nG:")
    print(G)
    
    # What should happen:
    # 1. Project G to tangent space
    XtG = X.T @ G
    print("\nX^T @ G:")
    print(XtG)
    
    sym_part = (XtG + XtG.T) / 2
    print("\nSymmetric part:")
    print(sym_part)
    
    G_tan = G - X @ sym_part
    print("\nTangent gradient:")
    print(G_tan)
    
    # Verify it's in tangent space
    print(f"\nTangent check: {np.linalg.norm(X.T @ G_tan + G_tan.T @ X, 'fro'):.2e}")
    
    # 2. Update and retract
    lr = 0.001
    update = -lr * G_tan
    X_expected = stiefel.retract(X, update)
    print(f"\nExpected orthogonality: {np.linalg.norm(X_expected.T @ X_expected - np.eye(p), 'fro'):.2e}")
    
    # 3. What the optimizer actually does
    sgd = riemannopt.SGD(step_size=lr)
    X_sgd = sgd.step(stiefel, X, G)
    print(f"SGD orthogonality: {np.linalg.norm(X_sgd.T @ X_sgd - np.eye(p), 'fro'):.2e}")
    
    print(f"\nDifference between expected and SGD: {np.linalg.norm(X_expected - X_sgd, 'fro'):.2e}")


if __name__ == "__main__":
    test_manifold_methods()
    test_optimizer_gradient_path()