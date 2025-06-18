#!/usr/bin/env python3
"""
Debug script for Adam optimizer on Stiefel manifold.

This script reproduces the orthogonality constraint issue where
||X^T X - I|| becomes 0.7 after 5 Adam steps.
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


def test_adam_stiefel_orthogonality():
    """Test Adam optimizer orthogonality preservation on Stiefel manifold."""
    print("Testing Adam optimizer on Stiefel manifold...")
    print("-" * 60)
    
    # Create Stiefel manifold St(15, 5)
    n, p = 15, 5
    stiefel = riemannopt.Stiefel(n, p)
    print(f"Created Stiefel manifold: St({n}, {p})")
    
    # Create Adam optimizer with small learning rate
    learning_rate = 0.001
    adam = riemannopt.Adam(learning_rate=learning_rate)
    print(f"Created Adam optimizer with learning_rate={learning_rate}")
    
    # Create cost function: f(X) = -tr(X^T C X)
    C = np.random.randn(n, n)
    C = C + C.T  # Make symmetric
    print(f"Created symmetric matrix C with shape {C.shape}")
    
    def cost_fn(X):
        return -np.trace(X.T @ C @ X)
    
    def grad_fn(X):
        return -2 * C @ X
    
    # Generate random initial point on Stiefel manifold
    X = stiefel.random_point()
    print(f"\nInitial point X shape: {X.shape}")
    
    # Check initial orthogonality
    XtX = X.T @ X
    I = np.eye(p)
    initial_error = np.linalg.norm(XtX - I, 'fro')
    print(f"Initial orthogonality error ||X^T X - I||_F = {initial_error:.2e}")
    print(f"Initial cost: {cost_fn(X):.6f}")
    
    # Perform 5 Adam steps
    print("\nPerforming Adam optimization steps...")
    print("-" * 60)
    
    for i in range(5):
        # Compute gradient
        grad = grad_fn(X)
        print(f"\nStep {i+1}:")
        print(f"  Gradient norm (Euclidean): {np.linalg.norm(grad, 'fro'):.6f}")
        
        # Check if gradient is in tangent space (for Stiefel: X^T V + V^T X = 0)
        XtV = X.T @ grad
        VtX = grad.T @ X
        tangent_error = np.linalg.norm(XtV + VtX, 'fro')
        print(f"  Gradient tangent space error: {tangent_error:.2e}")
        
        # Take Adam step
        X_new = adam.step(stiefel, X, grad)
        
        # Check orthogonality after step
        XtX_new = X_new.T @ X_new
        orth_error = np.linalg.norm(XtX_new - I, 'fro')
        print(f"  Orthogonality error after step: {orth_error:.2e}")
        
        # Check distance moved
        distance = np.linalg.norm(X_new - X, 'fro')
        print(f"  Distance moved: {distance:.6f}")
        
        # Cost change
        new_cost = cost_fn(X_new)
        print(f"  Cost: {new_cost:.6f} (change: {new_cost - cost_fn(X):.6f})")
        
        # Update X for next iteration
        X = X_new
        
        # Stop if orthogonality is severely violated
        if orth_error > 0.1:
            print(f"\n*** WARNING: Orthogonality severely violated! ***")
            print(f"*** ||X^T X - I||_F = {orth_error:.3f} ***")
            
            # Analyze the violation
            print("\nAnalyzing orthogonality violation:")
            eigenvals = np.linalg.eigvals(XtX_new)
            print(f"  Eigenvalues of X^T X: {eigenvals}")
            print(f"  Min eigenvalue: {np.min(eigenvals):.6f}")
            print(f"  Max eigenvalue: {np.max(eigenvals):.6f}")
            
            # Check column norms
            col_norms = np.linalg.norm(X_new, axis=0)
            print(f"  Column norms: {col_norms}")
            
            break
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Final orthogonality error: {orth_error:.3f}")
    print(f"Expected: < 1e-10, Got: {orth_error:.3f}")
    
    if orth_error > 1e-6:
        print("\nDIAGNOSIS: The Adam optimizer is not properly maintaining")
        print("the orthogonality constraint. This is likely because:")
        print("1. The current implementation is just gradient descent")
        print("2. It doesn't implement proper Adam momentum updates")
        print("3. The gradient may not be properly projected to tangent space")
        print("4. The retraction may not be working correctly")
    
    return orth_error


def debug_gradient_projection():
    """Debug the gradient projection to tangent space."""
    print("\n\nDebugging gradient projection...")
    print("-" * 60)
    
    n, p = 10, 3
    stiefel = riemannopt.Stiefel(n, p)
    
    # Create orthogonal matrix
    X = stiefel.random_point()
    
    # Create arbitrary gradient
    G = np.random.randn(n, p)
    
    # Project to tangent space manually
    # For Stiefel, tangent space at X is {V : X^T V + V^T X = 0}
    # Projection: V = G - X * sym(X^T G)
    XtG = X.T @ G
    sym_part = (XtG + XtG.T) / 2
    V_manual = G - X @ sym_part
    
    # Check if manual projection is in tangent space
    XtV = X.T @ V_manual
    VtX = V_manual.T @ X
    tangent_check = np.linalg.norm(XtV + VtX, 'fro')
    print(f"Manual projection tangent space error: {tangent_check:.2e}")
    
    # Compare with what the optimizer does
    adam = riemannopt.Adam(learning_rate=0.001)
    X_new = adam.step(stiefel, X, G)
    
    # Check if result is still orthogonal
    orth_error = np.linalg.norm(X_new.T @ X_new - np.eye(p), 'fro')
    print(f"Orthogonality error after Adam step: {orth_error:.2e}")


if __name__ == "__main__":
    # Run the main test
    error = test_adam_stiefel_orthogonality()
    
    # Run additional debugging if there's an issue
    if error > 1e-6:
        debug_gradient_projection()