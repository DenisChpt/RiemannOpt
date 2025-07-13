"""
End-to-end integration tests for RiemannOpt.

This module tests complete optimization workflows, realistic problems,
and performance characteristics of the entire system.
"""

import pytest
import numpy as np
import numpy.testing as npt
from typing import Dict, Any, List, Tuple, Optional
import time

try:
    import riemannopt as ro
except ImportError:
    pytest.skip("riemannopt not available", allow_module_level=True)


class TestClassicOptimizationProblems:
    """Test on classic Riemannian optimization problems."""
    
    def test_rayleigh_quotient_optimization(self):
        """Test Rayleigh quotient optimization on sphere.
        
        This is a classic eigenvalue problem: min x^T A x subject to ||x|| = 1.
        The solution should be the eigenvector corresponding to the smallest eigenvalue.
        """
        n = 20
        manifold = ro.manifolds.Sphere(n)
        
        # Create symmetric matrix with known eigenvalues
        np.random.seed(42)
        eigenvals = np.linspace(0.1, 10, n)
        Q, _ = np.linalg.qr(np.random.randn(n, n))  # Random orthogonal matrix
        A = Q @ np.diag(eigenvals) @ Q.T
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax  # Euclidean gradient
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad, validate_gradient=True)
        
        # Test multiple optimizers
        optimizers = [
            ("SGD", ro.optimizers.SGD(learning_rate=0.1)),
            ("Adam", ro.optimizers.Adam(learning_rate=0.1)),
            ("ConjugateGradient", ro.optimizers.ConjugateGradient()),
        ]
        
        for name, optimizer in optimizers:
            # Start from random point
            x0 = manifold.random_point()
            
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=500,
                gradient_tolerance=1e-8
            )
            
            x_final = result['x']
            final_cost = result['cost']
            
            # Verify solution quality
            assert manifold.contains(x_final), f"{name}: Final point not on sphere"
            
            # Should be close to smallest eigenvalue
            min_eigenval = np.min(eigenvals)
            assert abs(final_cost - min_eigenval) < 0.1, \
                f"{name}: Final cost {final_cost} not close to min eigenvalue {min_eigenval}"
            
            # Check that gradient is small
            if result['converged']:
                grad = cost_fn.gradient(x_final)
                proj_grad = manifold.project_tangent(x_final, grad)
                grad_norm = manifold.norm(x_final, proj_grad)
                assert grad_norm < 1e-6, f"{name}: Large gradient norm {grad_norm}"
    
    def test_procrustes_problem(self):
        """Test Procrustes problem on Stiefel manifold.
        
        Find Q ∈ St(n,p) that minimizes ||A - QQ^T B||_F^2.
        This is equivalent to maximizing tr(Q^T A B^T Q).
        """
        n, p = 15, 5
        manifold = ro.manifolds.Stiefel(n, p)
        
        # Create problem data
        np.random.seed(123)
        A = np.random.randn(n, n)
        B = np.random.randn(n, n)
        C = A @ B.T  # Target matrix for optimization
        
        def cost_and_grad(Q: np.ndarray) -> Tuple[float, np.ndarray]:
            CQ = C @ Q
            cost = -np.trace(Q.T @ CQ)  # Negative for minimization
            grad = -C.T @ Q  # Euclidean gradient
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad, validate_gradient=True)
        
        # Optimize
        optimizer = ro.optimizers.Adam(learning_rate=0.01)
        Q0 = manifold.random_point()
        
        result = optimizer.optimize_stiefel(
            cost_fn, manifold, Q0,
            max_iterations=200,
            gradient_tolerance=1e-8
        )
        
        Q_final = result['x']
        
        # Verify solution
        assert manifold.contains(Q_final), "Final point not on Stiefel manifold"
        
        # Check orthonormality
        QtQ = Q_final.T @ Q_final
        I = np.eye(p)
        npt.assert_allclose(QtQ, I, atol=1e-10, 
                           err_msg="Final point not orthonormal")
        
        # Cost should have improved significantly
        initial_cost = cost_fn.cost(Q0)
        final_cost = result['cost']
        assert final_cost < initial_cost - 0.1, "Insufficient improvement in cost"
    
    def test_burer_monteiro_sdp_relaxation(self):
        """Test Burer-Monteiro approach using Grassmann manifold.
        
        Solve a semidefinite program using low-rank factorization:
        min tr(C X) subject to X ≽ 0, tr(A_i X) = b_i, rank(X) ≤ r
        Parameterize X = Y Y^T where Y ∈ Gr(n, r).
        """
        n, r = 10, 3
        manifold = ro.manifolds.Grassmann(n, r)
        
        # Create SDP problem data
        np.random.seed(456)
        C = np.random.randn(n, n)
        C = C + C.T  # Symmetric
        
        def cost_and_grad(Y: np.ndarray) -> Tuple[float, np.ndarray]:
            # Cost: tr(C Y Y^T) = tr(Y^T C Y)
            CY = C @ Y
            cost = np.trace(Y.T @ CY)
            # Gradient: 2 C Y (in Euclidean space)
            grad = 2 * CY
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad, validate_gradient=True)
        
        # Optimize
        optimizer = ro.optimizers.Adam(learning_rate=0.01)
        Y0 = manifold.random_point()
        
        result = optimizer.optimize_stiefel(  # Grassmann uses Stiefel representation
            cost_fn, manifold, Y0,
            max_iterations=150,
            gradient_tolerance=1e-8
        )
        
        Y_final = result['x']
        
        # Verify solution
        assert manifold.contains(Y_final), "Final point not on Grassmann manifold"
        
        # Check that we found a critical point
        if result['converged']:
            grad = cost_fn.gradient(Y_final)
            proj_grad = manifold.project_tangent(Y_final, grad)
            grad_norm = manifold.norm(Y_final, proj_grad)
            assert grad_norm < 1e-6, f"Large gradient norm: {grad_norm}"
    
    def test_karcher_mean_on_spd(self):
        """Test Karcher mean computation on SPD manifold.
        
        Find the Riemannian center of mass of SPD matrices:
        min sum_i d²(X, X_i) where d is the Riemannian distance.
        """
        n = 4
        manifold = ro.manifolds.SPD(n)
        
        # Generate random SPD matrices
        np.random.seed(789)
        num_matrices = 5
        target_matrices = []
        
        for _ in range(num_matrices):
            A = np.random.randn(n, n)
            X = A @ A.T + 0.1 * np.eye(n)  # Ensure positive definiteness
            target_matrices.append(X)
        
        def cost_and_grad(X: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = 0.0
            grad = np.zeros_like(X)
            
            for Xi in target_matrices:
                # Distance squared and its gradient
                # This is a simplified version - real implementation would use
                # proper Riemannian distance and gradient
                diff = X - Xi
                cost += 0.5 * np.trace(diff @ diff.T)
                grad += diff
            
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad, validate_gradient=True)
        
        # Optimize
        optimizer = ro.optimizers.Adam(learning_rate=0.001)
        X0 = manifold.random_point()
        
        result = optimizer.optimize_stiefel(  # SPD uses matrix representation
            cost_fn, manifold, X0,
            max_iterations=100,
            gradient_tolerance=1e-8
        )
        
        X_final = result['x']
        
        # Verify solution
        assert manifold.contains(X_final), "Final point not on SPD manifold"
        
        # Check SPD properties
        eigenvals = np.linalg.eigvals(X_final)
        assert np.all(eigenvals > 0), "Final matrix not positive definite"
        
        # Check symmetry
        npt.assert_allclose(X_final, X_final.T, atol=1e-10,
                           err_msg="Final matrix not symmetric")


class TestRobustnessAndEdgeCases:
    """Test robustness and handling of edge cases."""
    
    def test_optimization_near_manifold_boundary(self):
        """Test optimization when starting near manifold boundaries."""
        manifold = ro.manifolds.Sphere(10)
        
        # Create cost function with minimum near boundary
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            # Cost has minimum at (1, 0, 0, ..., 0)
            target = np.zeros_like(x)
            target[0] = 1.0
            diff = x - target
            cost = 0.5 * np.sum(diff**2)
            grad = diff
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        # Start very close to the target
        x0 = np.zeros(10)
        x0[0] = 0.99999
        x0[1] = np.sqrt(1 - x0[0]**2)  # Ensure unit norm
        
        optimizer = ro.optimizers.Adam(learning_rate=0.1)
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=100,
            gradient_tolerance=1e-10
        )
        
        x_final = result['x']
        
        # Should converge to target
        target = np.zeros(10)
        target[0] = 1.0
        
        distance = manifold.distance(x_final, target)
        assert distance < 0.01, f"Did not converge to target: distance={distance}"
        
        # Should remain on manifold
        assert manifold.contains(x_final), "Final point not on sphere"
    
    def test_optimization_with_rank_deficient_hessian(self):
        """Test optimization with rank-deficient Hessian."""
        manifold = ro.manifolds.Sphere(5)
        
        # Cost function: only depends on first coordinate
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            cost = x[0]**2
            grad = np.zeros_like(x)
            grad[0] = 2 * x[0]
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        # Start away from minimum
        x0 = np.array([0.5, 0.5, 0.5, 0.5, 0.0])
        x0 = x0 / np.linalg.norm(x0)  # Normalize
        
        optimizer = ro.optimizers.Adam(learning_rate=0.1)
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=100
        )
        
        x_final = result['x']
        
        # Should minimize first coordinate (make it close to 0)
        assert abs(x_final[0]) < 0.1, f"Did not minimize first coordinate: {x_final[0]}"
        
        # Should remain on manifold
        assert manifold.contains(x_final), "Final point not on sphere"
    
    def test_optimization_with_multiple_minima(self):
        """Test optimization with multiple local minima."""
        manifold = ro.manifolds.Sphere(3)
        
        # Cost function with multiple minima
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            # Two minima at (±1, 0, 0)
            cost1 = (x[0] - 1)**2 + x[1]**2 + x[2]**2
            cost2 = (x[0] + 1)**2 + x[1]**2 + x[2]**2
            cost = min(cost1, cost2)
            
            # Gradient toward nearest minimum
            if cost1 < cost2:
                grad = 2 * np.array([x[0] - 1, x[1], x[2]])
            else:
                grad = 2 * np.array([x[0] + 1, x[1], x[2]])
            
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        # Test convergence from different starting points
        starting_points = [
            np.array([0.9, 0.1, 0.1]),    # Near (+1, 0, 0)
            np.array([-0.9, 0.1, 0.1]),   # Near (-1, 0, 0)
            np.array([0.0, 1.0, 0.0]),    # Equidistant
        ]
        
        optimizer = ro.optimizers.SGD(learning_rate=0.1)
        
        for i, x0 in enumerate(starting_points):
            x0 = x0 / np.linalg.norm(x0)  # Normalize
            
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=100
            )
            
            x_final = result['x']
            
            # Should converge to one of the minima
            dist_to_min1 = np.linalg.norm(x_final - np.array([1, 0, 0]))
            dist_to_min2 = np.linalg.norm(x_final - np.array([-1, 0, 0]))
            min_dist = min(dist_to_min1, dist_to_min2)
            
            assert min_dist < 0.2, f"Starting point {i}: Did not converge to minimum, min_dist={min_dist}"
            assert manifold.contains(x_final), f"Starting point {i}: Final point not on sphere"
    
    def test_optimization_with_noisy_gradients(self):
        """Test optimization robustness to gradient noise."""
        manifold = ro.manifolds.Sphere(8)
        
        # Clean problem
        A = np.diag([5, 4, 3, 2, 1, 0.5, 0.2, 0.1])
        
        # Add noise to gradients
        np.random.seed(42)
        noise_level = 0.01
        
        def noisy_cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            clean_grad = 2 * Ax
            
            # Add noise to gradient
            noise = np.random.normal(0, noise_level, x.shape)
            noisy_grad = clean_grad + noise
            
            return cost, noisy_grad
        
        cost_fn = ro.create_cost_function(noisy_cost_and_grad)
        
        # Use robust optimizer
        optimizer = ro.optimizers.Adam(learning_rate=0.01)  # Smaller learning rate for stability
        x0 = manifold.random_point()
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=500,
            gradient_tolerance=1e-6  # More relaxed tolerance
        )
        
        x_final = result['x']
        
        # Should still make progress despite noise
        clean_cost_fn = ro.create_cost_function(lambda x: (x @ A @ x, 2 * A @ x))
        initial_cost = clean_cost_fn.cost(x0)
        final_cost = clean_cost_fn.cost(x_final)
        
        assert final_cost < initial_cost - 0.01, "No progress with noisy gradients"
        assert manifold.contains(x_final), "Final point not on manifold"


class TestPerformanceAndScalability:
    """Test performance and scalability characteristics."""
    
    @pytest.mark.slow
    def test_large_scale_sphere_optimization(self):
        """Test optimization on large sphere."""
        n = 1000  # Large dimension
        manifold = ro.manifolds.Sphere(n)
        
        # Sparse eigenvalue problem
        np.random.seed(123)
        eigenvals = np.concatenate([
            [0.1],  # Smallest eigenvalue
            np.linspace(1, 100, n-1)  # Other eigenvalues
        ])
        
        # Don't form full matrix - use matrix-vector products
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            # Diagonal matrix multiplication
            Ax = eigenvals * x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        # Use efficient optimizer
        optimizer = ro.optimizers.Adam(learning_rate=0.1)
        x0 = manifold.random_point()
        
        start_time = time.perf_counter()
        
        result = optimizer.optimize_sphere(
            cost_fn, manifold, x0,
            max_iterations=100
        )
        
        end_time = time.perf_counter()
        total_time = end_time - start_time
        
        x_final = result['x']
        final_cost = result['cost']
        
        # Performance check
        assert total_time < 10.0, f"Large scale optimization too slow: {total_time:.2f}s"
        
        # Quality check
        assert abs(final_cost - 0.1) < 0.5, f"Did not find small eigenvalue: {final_cost}"
        assert manifold.contains(x_final), "Final point not on sphere"
        
        print(f"Large scale optimization (n={n}): {total_time:.3f}s, final cost: {final_cost:.6f}")
    
    @pytest.mark.slow
    def test_optimization_memory_usage(self):
        """Test that optimization has reasonable memory usage."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create moderately large problem
        n = 500
        manifold = ro.manifolds.Sphere(n)
        
        A = np.random.randn(n, n)
        A = A + A.T
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        optimizer = ro.optimizers.Adam(learning_rate=0.01)
        
        # Run multiple optimizations
        for _ in range(10):
            x0 = manifold.random_point()
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=50
            )
            
            assert manifold.contains(result['x']), "Invalid result"
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB for this problem)
        assert memory_increase < 100, f"Excessive memory usage: {memory_increase:.1f}MB increase"
        
        print(f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)")
    
    def test_convergence_rate_comparison(self):
        """Compare convergence rates of different optimizers."""
        manifold = ro.manifolds.Sphere(20)
        
        # Well-conditioned problem
        eigenvals = np.logspace(-2, 0, 20)  # Condition number = 100
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = eigenvals * x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        
        optimizers = [
            ("SGD", ro.optimizers.SGD(learning_rate=0.1)),
            ("Adam", ro.optimizers.Adam(learning_rate=0.1)),
            ("ConjugateGradient", ro.optimizers.ConjugateGradient()),
        ]
        
        results = {}
        x0 = manifold.random_point()  # Same starting point
        target_cost = eigenvals[0]  # Minimum possible cost
        
        for name, optimizer in optimizers:
            start_time = time.perf_counter()
            
            result = optimizer.optimize_sphere(
                cost_fn, manifold, x0,
                max_iterations=200,
                gradient_tolerance=1e-12
            )
            
            end_time = time.perf_counter()
            
            results[name] = {
                'final_cost': result['cost'],
                'iterations': result['iterations'],
                'time': end_time - start_time,
                'converged': result['converged'],
                'cost_error': abs(result['cost'] - target_cost)
            }
        
        # Print comparison
        print("\nOptimizer Performance Comparison:")
        print("Optimizer           | Final Cost  | Error     | Iterations | Time   | Converged")
        print("-" * 80)
        for name, res in results.items():
            print(f"{name:<18} | {res['final_cost']:.6f} | {res['cost_error']:.2e} | "
                  f"{res['iterations']:>8} | {res['time']:.3f}s | {res['converged']}")
        
        # All should converge for this well-conditioned problem
        for name, res in results.items():
            assert res['converged'], f"{name} did not converge"
            assert res['cost_error'] < 0.01, f"{name} did not reach target cost"


class TestHighLevelAPIIntegration:
    """Test the high-level API integration."""
    
    def test_complete_workflow_with_high_level_api(self):
        """Test complete optimization workflow using high-level API."""
        # Problem: find principal component of data matrix
        np.random.seed(42)
        data = np.random.randn(100, 20)  # 100 samples, 20 features
        
        # Covariance matrix
        C = data.T @ data / (data.shape[0] - 1)
        
        # Manifold: unit sphere (for principal component)
        manifold = ro.manifolds.Sphere(20)
        
        # Cost function: negative Rayleigh quotient
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Cx = C @ x
            cost = -(x @ Cx)  # Negative for maximization
            grad = -2 * Cx
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad, validate_gradient=True)
        
        # Initial point
        x0 = manifold.random_point()
        
        # Use high-level optimize function
        result = ro.optimize(
            cost_function=cost_fn,
            manifold=manifold,
            initial_point=x0,
            optimizer="Adam",
            learning_rate=0.01,
            max_iterations=200,
            gradient_tolerance=1e-8
        )
        
        x_final = result['x']
        
        # Verify result
        assert manifold.contains(x_final), "Final point not on sphere"
        
        # Should be close to principal eigenvector
        eigenvals, eigenvecs = np.linalg.eigh(C)
        principal_component = eigenvecs[:, -1]  # Largest eigenvalue
        
        # Check alignment (allowing for sign ambiguity)
        alignment = abs(x_final @ principal_component)
        assert alignment > 0.9, f"Poor alignment with principal component: {alignment}"
        
        # Final cost should be close to negative largest eigenvalue
        expected_cost = -eigenvals[-1]
        cost_error = abs(result['cost'] - expected_cost)
        assert cost_error < 0.1, f"Cost error too large: {cost_error}"
    
    def test_different_manifold_types_with_high_level_api(self):
        """Test high-level API with different manifold types."""
        test_cases = [
            # (manifold, problem_generator)
            (ro.manifolds.Sphere(10), self._generate_sphere_problem),
            (ro.manifolds.Stiefel(8, 3), self._generate_stiefel_problem),
        ]
        
        for manifold, problem_gen in test_cases:
            cost_fn, x0, expected_improvement = problem_gen(manifold)
            
            result = ro.optimize(
                cost_function=cost_fn,
                manifold=manifold,
                initial_point=x0,
                optimizer="Adam",
                learning_rate=0.01,
                max_iterations=100
            )
            
            # Basic checks
            assert manifold.contains(result['x']), f"Final point not on {type(manifold).__name__}"
            
            # Check improvement
            initial_cost = cost_fn.cost(x0)
            final_cost = result['cost']
            improvement = initial_cost - final_cost
            
            assert improvement > expected_improvement, \
                f"Insufficient improvement on {type(manifold).__name__}: {improvement}"
    
    def _generate_sphere_problem(self, manifold):
        """Generate test problem for sphere."""
        n = manifold.ambient_dim
        A = np.diag(np.linspace(1, 0.1, n))
        
        def cost_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
            Ax = A @ x
            cost = x @ Ax
            grad = 2 * Ax
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        x0 = manifold.random_point()
        expected_improvement = 0.05  # Should reduce cost by at least this much
        
        return cost_fn, x0, expected_improvement
    
    def _generate_stiefel_problem(self, manifold):
        """Generate test problem for Stiefel."""
        n, p = manifold.n, manifold.p
        B = np.random.randn(n, n)
        B = B + B.T
        
        def cost_and_grad(X: np.ndarray) -> Tuple[float, np.ndarray]:
            BX = B @ X
            cost = np.trace(X.T @ BX)
            grad = 2 * BX
            return cost, grad
        
        cost_fn = ro.create_cost_function(cost_and_grad)
        X0 = manifold.random_point()
        expected_improvement = 0.1
        
        return cost_fn, X0, expected_improvement


if __name__ == "__main__":
    pytest.main([__file__, "-v"])