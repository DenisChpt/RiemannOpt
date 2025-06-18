"""
Test RiemannOpt against known solutions and test problems.

This module contains test problems with known analytical solutions
or well-established numerical results from the literature.
"""

import pytest
import numpy as np
from conftest import riemannopt, TOLERANCES


class TestKnownSolutions:
    """Test problems with known analytical or numerical solutions."""
    
    def test_sphere_eigenvalue_problem(self):
        """
        Rayleigh quotient minimization on sphere.
        
        Known solution: The minimum is the smallest eigenvalue,
        achieved at the corresponding eigenvector.
        """
        n = 10
        sphere = riemannopt.Sphere(n)
        
        # Create a symmetric matrix with known eigenvalues
        eigenvalues = np.array([-5, -3, -1, 0, 1, 2, 3, 4, 5, 6])
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        A = Q @ np.diag(eigenvalues) @ Q.T
        
        # The minimum of x^T A x on the sphere is the smallest eigenvalue
        expected_min = eigenvalues[0]
        
        # Run optimization
        sgd = riemannopt.SGD(step_size=0.01, momentum=0.9)
        x = sphere.random_point()
        
        # Track convergence
        history = []
        for i in range(1000):
            value = float(x.T @ A @ x)
            history.append(value)
            
            grad = 2 * A @ x
            x = sgd.step(sphere, x, grad)
        
        final_value = float(x.T @ A @ x)
        
        print(f"Eigenvalue problem convergence:")
        print(f"  Initial: {history[0]:.6f}")
        print(f"  After 100 iter: {history[100]:.6f}")
        print(f"  After 500 iter: {history[500]:.6f}")
        print(f"  Final: {final_value:.6f}")
        print(f"  Expected: {expected_min:.6f}")
        print(f"  Error: {abs(final_value - expected_min):.6e}")
        
        assert abs(final_value - expected_min) < 0.05, \
            f"Should converge to {expected_min}, got {final_value}"
    
    def test_stiefel_trace_minimization(self):
        """
        Trace minimization on Stiefel manifold.
        
        Problem: min_X tr(X^T A X) subject to X^T X = I
        Known solution: X = eigenvectors corresponding to smallest eigenvalues
        """
        n, p = 20, 5
        stiefel = riemannopt.Stiefel(n, p)
        
        # Create symmetric matrix
        eigenvalues = np.sort(np.random.uniform(-2, 5, n))
        Q, _ = np.linalg.qr(np.random.randn(n, n))
        A = Q @ np.diag(eigenvalues) @ Q.T
        
        # Expected minimum: sum of p smallest eigenvalues
        expected_min = sum(eigenvalues[:p])
        
        # Run optimization
        sgd = riemannopt.SGD(step_size=0.01, momentum=0.9)
        X = stiefel.random_point()
        
        history = []
        for i in range(800):
            value = np.trace(X.T @ A @ X)
            history.append(value)
            
            grad = 2 * A @ X
            X = sgd.step(stiefel, X, grad)
        
        final_value = np.trace(X.T @ A @ X)
        
        print(f"\nStiefel trace minimization:")
        print(f"  Initial: {history[0]:.6f}")
        print(f"  After 200 iter: {history[200]:.6f}")
        print(f"  Final: {final_value:.6f}")
        print(f"  Expected: {expected_min:.6f}")
        print(f"  Error: {abs(final_value - expected_min):.6e}")
        
        assert abs(final_value - expected_min) < 0.1, \
            f"Should converge to {expected_min}, got {final_value}"
    
    def test_sphere_geodesic_properties(self):
        """Test geodesic properties on sphere."""
        sphere = riemannopt.Sphere(5)
        
        # Test 1: Geodesic between two points has length = distance
        x = sphere.random_point()
        y = sphere.random_point()
        
        dist = sphere.distance(x, y)
        
        # Verify distance formula
        cos_angle = np.clip(np.dot(x, y), -1, 1)
        expected_dist = np.arccos(cos_angle)
        
        assert abs(dist - expected_dist) < TOLERANCES['numerical'], \
            f"Distance formula incorrect: got {dist}, expected {expected_dist}"
        
        # Test 2: exp and log are inverses
        v = np.random.randn(5)
        v = sphere.tangent_projection(x, v)
        v_norm = np.linalg.norm(v)
        
        if v_norm < np.pi - 0.1:  # Avoid cut locus
            y = sphere.exp(x, v)
            v_recovered = sphere.log(x, y)
            
            assert np.allclose(v, v_recovered, atol=TOLERANCES['numerical']), \
                "exp and log should be inverses"
    
    def test_optimization_with_known_minimizer(self):
        """Test optimization where we know the exact minimizer."""
        n = 8
        sphere = riemannopt.Sphere(n)
        
        # Quadratic function with known minimizer
        x_star = np.zeros(n)
        x_star[0] = 1.0  # Minimizer is first basis vector
        
        # f(x) = ||x - 2*x_star||^2 = ||x||^2 - 4*<x, x_star> + 4
        # On sphere: f(x) = 1 - 4*x[0] + 4 = 5 - 4*x[0]
        # Minimum at x = x_star with value = 1
        
        sgd = riemannopt.SGD(step_size=0.1, momentum=0.9)
        x = sphere.random_point()
        
        for _ in range(500):
            # Gradient of f(x) = 5 - 4*x[0]
            grad = np.zeros(n)
            grad[0] = -4
            x = sgd.step(sphere, x, grad)
        
        # Check convergence to known minimizer
        assert abs(x[0] - 1.0) < 0.01, f"Should converge to x[0]=1, got {x[0]}"
        
        # Check other components are small
        assert np.linalg.norm(x[1:]) < 0.01, "Other components should be near zero"


class TestLiteratureProblems:
    """Test problems from optimization literature."""
    
    def test_brockett_cost_function(self):
        """
        Brockett cost function on Stiefel manifold.
        
        Problem from: Edelman et al. "The Geometry of Algorithms with 
        Orthogonality Constraints" (1998)
        
        min_X tr(X^T A X N) where N = diag(1, 2, ..., p)
        """
        n, p = 12, 4
        stiefel = riemannopt.Stiefel(n, p)
        
        # Problem data
        A = np.random.randn(n, n)
        A = A + A.T  # Symmetric
        N = np.diag(np.arange(1, p+1, dtype=float))
        
        # This problem tends to sort eigenvalues
        sgd = riemannopt.SGD(step_size=0.01, momentum=0.9)
        X = stiefel.random_point()
        
        for _ in range(1000):
            # Gradient of tr(X^T A X N)
            grad = 2 * A @ X @ N
            X = sgd.step(stiefel, X, grad)
        
        # Check orthogonality is preserved
        gram = X.T @ X
        assert np.allclose(gram, np.eye(p), atol=TOLERANCES['numerical']), \
            "Solution should remain on Stiefel manifold"
        
        # The solution tends to align with eigenvectors of A
        # weighted by the diagonal of N
        final_value = np.trace(X.T @ A @ X @ N)
        print(f"\nBrockett problem final value: {final_value:.6f}")


if __name__ == "__main__":
    print("Testing known solutions...\n")
    
    test = TestKnownSolutions()
    test.test_sphere_eigenvalue_problem()
    test.test_stiefel_trace_minimization()
    test.test_sphere_geodesic_properties()
    test.test_optimization_with_known_minimizer()
    
    print("\nTesting literature problems...")
    test_lit = TestLiteratureProblems()
    test_lit.test_brockett_cost_function()
    
    print("\nAll tests with known solutions passed!")