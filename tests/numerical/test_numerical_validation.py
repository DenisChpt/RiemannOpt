"""
Numerical validation tests comparing RiemannOpt with known reference values.

This module tests that RiemannOpt produces numerically correct results
by comparing with:
1. Analytical solutions where available
2. Reference implementations from established libraries
3. Known test cases from literature
"""

import pytest
import numpy as np
from conftest import riemannopt, TOLERANCES

class TestAnalyticalSolutions:
    """Test against known analytical solutions."""
    
    def test_sphere_geodesic_distance(self):
        """Test geodesic distance formula on sphere."""
        sphere = riemannopt.Sphere(3)
        
        # Test cases with known distances
        test_cases = [
            # Same point -> distance 0
            ([1, 0, 0], [1, 0, 0], 0.0),
            # Orthogonal points -> distance π/2
            ([1, 0, 0], [0, 1, 0], np.pi/2),
            # Antipodal points -> distance π
            ([1, 0, 0], [-1, 0, 0], np.pi),
            # 45 degree angle -> distance π/4
            ([1/np.sqrt(2), 1/np.sqrt(2), 0], [1, 0, 0], np.pi/4),
        ]
        
        for x_data, y_data, expected_dist in test_cases:
            x = np.array(x_data, dtype=np.float64)
            y = np.array(y_data, dtype=np.float64)
            
            # Normalize to ensure on sphere
            x = x / np.linalg.norm(x)
            y = y / np.linalg.norm(y)
            
            dist = sphere.distance(x, y)
            assert np.abs(dist - expected_dist) < TOLERANCES['numerical'], \
                f"Distance between {x_data} and {y_data} should be {expected_dist}, got {dist}"
    
    def test_stiefel_dimension_formula(self):
        """Test that Stiefel manifold has correct dimension."""
        # St(n,p) has dimension np - p(p+1)/2
        test_cases = [
            (10, 3, 10*3 - 3*4//2),  # 30 - 6 = 24
            (20, 5, 20*5 - 5*6//2),  # 100 - 15 = 85
            (8, 2, 8*2 - 2*3//2),    # 16 - 3 = 13
        ]
        
        for n, p, expected_dim in test_cases:
            stiefel = riemannopt.Stiefel(n, p)
            assert stiefel.manifold_dim == expected_dim, \
                f"St({n},{p}) should have dimension {expected_dim}, got {stiefel.manifold_dim}"
    
    def test_sphere_exponential_map(self):
        """Test exponential map gives correct geodesics."""
        sphere = riemannopt.Sphere(3)
        
        # North pole
        x = np.array([0, 0, 1], dtype=np.float64)
        
        # Tangent vector pointing towards equator
        v = np.array([1, 0, 0], dtype=np.float64)
        
        # Points along geodesic
        for t in [0, np.pi/6, np.pi/4, np.pi/3, np.pi/2]:
            y = sphere.exp(x, t * v)
            
            # Check we're on the sphere
            assert np.abs(np.linalg.norm(y) - 1) < TOLERANCES['strict']
            
            # Check geodesic formula: geodesic should lie in the xz-plane
            assert np.abs(y[1]) < TOLERANCES['strict']  # y-component should be 0
            
            # Check position along geodesic
            expected_x = np.sin(t)
            expected_z = np.cos(t)
            assert np.abs(y[0] - expected_x) < TOLERANCES['numerical']
            assert np.abs(y[2] - expected_z) < TOLERANCES['numerical']


class TestOptimizationProblems:
    """Test optimization problems with known solutions."""
    
    def test_rayleigh_quotient_minimization(self):
        """Test that we find the minimum eigenvalue."""
        n = 10
        sphere = riemannopt.Sphere(n)
        
        # Create symmetric matrix with known eigenvalues
        eigenvalues = np.sort(np.random.uniform(-5, 5, n))
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        A = Q @ np.diag(eigenvalues) @ Q.T
        
        min_eigenvalue = eigenvalues[0]
        
        # Optimize Rayleigh quotient
        sgd = riemannopt.SGD(step_size=0.01, momentum=0.9)
        x = sphere.random_point()
        
        for i in range(500):
            # Gradient of x^T A x on sphere
            grad = 2 * A @ x
            x = sgd.step(sphere, x, grad)
            
            if i % 100 == 0:
                current_value = float(x.T @ A @ x)
                print(f"Iteration {i}: {current_value:.6f} (target: {min_eigenvalue:.6f})")
        
        final_value = float(x.T @ A @ x)
        assert np.abs(final_value - min_eigenvalue) < 0.1, \
            f"Should converge to minimum eigenvalue {min_eigenvalue}, got {final_value}"
    
    def test_orthogonal_procrustes(self):
        """Test orthogonal Procrustes problem: min ||AX - B||_F over orthogonal X."""
        n, p = 10, 5
        stiefel = riemannopt.Stiefel(n, p)
        
        # Generate true orthogonal matrix
        X_true, _ = np.linalg.qr(np.random.randn(n, p))
        
        # Generate data
        A = np.random.randn(20, n)
        B = A @ X_true + 0.01 * np.random.randn(20, p)  # Small noise
        
        # Cost function: ||AX - B||_F^2
        def cost_and_gradient(X):
            residual = A @ X - B
            cost = 0.5 * np.linalg.norm(residual, 'fro')**2
            grad = A.T @ residual
            return cost, grad
        
        # Optimize
        sgd = riemannopt.SGD(step_size=0.01, momentum=0.9)
        X = stiefel.random_point()
        
        for _ in range(300):
            _, grad = cost_and_gradient(X)
            X = sgd.step(stiefel, X, grad)
        
        # Check solution quality
        # The solution should satisfy A^T A X = A^T B (normal equations)
        # up to the orthogonality constraint
        residual = A @ X - B
        final_cost = 0.5 * np.linalg.norm(residual, 'fro')**2
        
        # Compute optimal cost using SVD solution
        U, S, Vt = np.linalg.svd(A.T @ B, full_matrices=False)
        X_svd = U @ Vt
        optimal_cost = 0.5 * np.linalg.norm(A @ X_svd - B, 'fro')**2
        
        assert final_cost < optimal_cost + 0.1, \
            f"Should be close to optimal cost {optimal_cost}, got {final_cost}"


class TestGradientChecking:
    """Test gradient computations against finite differences."""
    
    def test_sphere_gradient_projection(self):
        """Test that gradient projection is correct."""
        sphere = riemannopt.Sphere(5)
        
        # Test point
        x = sphere.random_point()
        
        # Euclidean gradient
        euclidean_grad = np.random.randn(5)
        
        # Project to tangent space
        riemannian_grad = sphere.tangent_projection(x, euclidean_grad)
        
        # Check orthogonality
        assert np.abs(np.dot(x, riemannian_grad)) < TOLERANCES['strict'], \
            "Riemannian gradient should be orthogonal to point"
        
        # Check projection formula: grad_R = grad_E - <grad_E, x>x
        expected = euclidean_grad - np.dot(euclidean_grad, x) * x
        assert np.allclose(riemannian_grad, expected, atol=TOLERANCES['numerical']), \
            "Gradient projection formula incorrect"


class TestMetricProperties:
    """Test Riemannian metric properties."""
    
    def test_sphere_metric_is_restriction(self):
        """Test that sphere metric is restriction of Euclidean metric."""
        sphere = riemannopt.Sphere(5)
        x = sphere.random_point()
        
        # Generate tangent vectors
        v1 = np.random.randn(5)
        v1 = sphere.tangent_projection(x, v1)
        v2 = np.random.randn(5)
        v2 = sphere.tangent_projection(x, v2)
        
        # Riemannian inner product should equal Euclidean inner product
        riem_inner = sphere.inner_product(x, v1, v2)
        eucl_inner = np.dot(v1, v2)
        
        assert np.abs(riem_inner - eucl_inner) < TOLERANCES['strict'], \
            "Sphere metric should be restriction of Euclidean metric"


if __name__ == "__main__":
    # Run tests manually for debugging
    test = TestAnalyticalSolutions()
    test.test_sphere_geodesic_distance()
    test.test_sphere_exponential_map()
    
    print("All analytical solution tests passed!")
    
    test_opt = TestOptimizationProblems()
    test_opt.test_rayleigh_quotient_minimization()
    print("Optimization tests passed!")