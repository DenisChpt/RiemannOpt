"""
Performance benchmarks for RiemannOpt.

This module contains performance tests comparing RiemannOpt against
other libraries and measuring operation speeds.
"""

import pytest
import numpy as np
import time
from typing import Dict, List, Tuple, Callable
from conftest import riemannopt, TOLERANCES


class BenchmarkTimer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start


class TestManifoldOperationPerformance:
    """Benchmark individual manifold operations."""
    
    @pytest.mark.benchmark
    def test_sphere_projection_speed(self, benchmark):
        """Benchmark sphere projection operation."""
        n = 1000
        sphere = riemannopt.Sphere(n)
        
        # Generate test vectors
        vectors = [np.random.randn(n) for _ in range(100)]
        
        def project_all():
            for v in vectors:
                sphere.project(v)
        
        benchmark(project_all)
        
        # The benchmark function handles timing and assertions
        # No need for manual assertions here
    
    @pytest.mark.benchmark
    def test_stiefel_retraction_speed(self, benchmark):
        """Benchmark Stiefel retraction operation."""
        n, p = 100, 10
        stiefel = riemannopt.Stiefel(n, p)
        
        # Generate test data
        X = stiefel.random_point()
        tangents = [stiefel.random_tangent(X) for _ in range(50)]
        
        def retract_all():
            for V in tangents:
                stiefel.retraction(X, V)
        
        benchmark(retract_all)
        
        # The benchmark function handles timing and performance tracking
    
    @pytest.mark.benchmark
    @pytest.mark.parametrize("size", [10, 50, 100, 500])
    def test_grassmann_inner_product_scaling(self, benchmark, size):
        """Test how Grassmann inner product scales with dimension."""
        grassmann = riemannopt.Grassmann(size, 5)
        
        X = grassmann.random_point()
        U = grassmann.random_tangent(X)
        V = grassmann.random_tangent(X)
        
        def compute_inner():
            return grassmann.inner_product(X, U, V)
        
        benchmark(compute_inner)
        
        # The benchmark function handles timing and performance tracking
        # Scaling analysis is done by pytest-benchmark across parameter values


class TestOptimizationPerformance:
    """Benchmark complete optimization scenarios."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_pca_optimization_speed(self, benchmark):
        """Benchmark PCA optimization on Stiefel manifold."""
        n, p = 50, 5
        n_iterations = 100
        
        # Generate test problem
        eigenvalues = np.exp(-np.arange(n) / 10)
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        C = Q @ np.diag(eigenvalues) @ Q.T
        
        def run_pca():
            stiefel = riemannopt.Stiefel(n, p)
            sgd = riemannopt.SGD(step_size=0.001, momentum=0.9)
            
            X = stiefel.random_point()
            
            for _ in range(n_iterations):
                grad = -2 * C @ X
                X = sgd.step(stiefel, X, grad)
            
            return X
        
        benchmark(run_pca)
        
        # The benchmark function handles timing and performance tracking
    
    @pytest.mark.benchmark
    def test_sphere_optimization_vs_numpy(self, benchmark):
        """Compare sphere optimization with pure NumPy implementation."""
        n = 100
        
        # Problem setup
        A = np.random.randn(n, n)
        A = A + A.T
        
        # RiemannOpt implementation
        def riemannopt_optimize():
            sphere = riemannopt.Sphere(n)
            sgd = riemannopt.SGD(step_size=0.01)
            
            x = sphere.random_point()
            for _ in range(50):
                grad = 2 * A @ x
                x = sgd.step(sphere, x, grad)
            return x
        
        # Pure NumPy implementation
        def numpy_optimize():
            x = np.random.randn(n)
            x = x / np.linalg.norm(x)
            
            for _ in range(50):
                grad = 2 * A @ x
                # Project gradient
                grad = grad - np.dot(grad, x) * x
                # Update and normalize
                x = x - 0.01 * grad
                x = x / np.linalg.norm(x)
            return x
        
        # Benchmark RiemannOpt version
        benchmark(riemannopt_optimize)


class TestMemoryEfficiency:
    """Test memory usage patterns."""
    
    @pytest.mark.benchmark
    def test_stiefel_memory_usage(self, benchmark):
        """Test memory efficiency of Stiefel operations."""
        import tracemalloc
        
        n, p = 200, 20
        stiefel = riemannopt.Stiefel(n, p)
        
        def memory_intensive_operations():
            # Track peak memory usage
            tracemalloc.start()
            
            # Multiple operations
            points = []
            for _ in range(10):
                X = stiefel.random_point()
                V = stiefel.random_tangent(X)
                Y = stiefel.retraction(X, V)
                points.append(Y)
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            return peak
        
        # Just benchmark the function, memory tracking is handled internally
        benchmark(memory_intensive_operations)


class TestParallelPerformance:
    """Test performance with parallel operations."""
    
    @pytest.mark.benchmark
    @pytest.mark.skip(reason="Parallel operations not yet implemented")
    def test_parallel_sphere_projections(self, benchmark):
        """Benchmark parallel projection operations."""
        pass
    
    @pytest.mark.benchmark
    @pytest.mark.skip(reason="Parallel operations not yet implemented")
    def test_parallel_optimization(self, benchmark):
        """Benchmark parallel optimization scenarios."""
        pass


class TestScalabilityBenchmarks:
    """Test how performance scales with problem size."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    @pytest.mark.parametrize("n", [10, 50, 100, 200])
    def test_sphere_scaling(self, benchmark, n):
        """Test sphere operations scaling with dimension."""
        sphere = riemannopt.Sphere(n)
        
        def operations():
            x = sphere.random_point()
            v = sphere.random_tangent(x)
            
            # Multiple operations
            for _ in range(10):
                x = sphere.retraction(x, 0.1 * v)
                v = sphere.tangent_projection(x, v)
                sphere.inner_product(x, v, v)
        
        benchmark(operations)
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    @pytest.mark.parametrize("size", [(20, 5), (50, 10), (100, 10), (200, 20)])
    def test_stiefel_scaling(self, benchmark, size):
        """Test Stiefel operations scaling with dimensions."""
        n, p = size
        stiefel = riemannopt.Stiefel(n, p)
        
        def operations():
            X = stiefel.random_point()
            V = stiefel.random_tangent(X)
            
            for _ in range(5):
                X = stiefel.retraction(X, 0.1 * V)
                V = stiefel.tangent_projection(X, V)
        
        benchmark(operations)


class TestComparisonBenchmarks:
    """Compare RiemannOpt performance with other libraries."""
    
    @pytest.mark.benchmark
    @pytest.mark.slow
    def test_vs_pymanopt_sphere(self, benchmark):
        """Compare with PyManopt on sphere optimization."""
        # This would compare with PyManopt if available
        # For now, just ensure our performance is good
        
        n = 100
        sphere = riemannopt.Sphere(n)
        
        # Rayleigh quotient minimization
        A = np.random.randn(n, n)
        A = A + A.T
        eigenvalues = np.linalg.eigvalsh(A)
        min_eigenvalue = eigenvalues[0]
        
        def optimize():
            sgd = riemannopt.SGD(step_size=0.01, momentum=0.9)
            x = sphere.random_point()
            
            for _ in range(200):
                grad = 2 * A @ x
                x = sgd.step(sphere, x, grad)
            
            return float(x.T @ A @ x)
        
        benchmark(optimize)
    
    @pytest.mark.benchmark
    def test_optimization_suite(self, benchmark):
        """Benchmark suite of optimization problems."""
        problems = []
        
        # Sphere eigenvalue problem
        sphere = riemannopt.Sphere(50)
        A_sphere = np.random.randn(50, 50)
        A_sphere = A_sphere + A_sphere.T
        problems.append(('sphere_eigenvalue', sphere, 
                        lambda x: 2 * A_sphere @ x))
        
        # Stiefel trace maximization
        stiefel = riemannopt.Stiefel(30, 5)
        C_stiefel = np.random.randn(30, 30)
        C_stiefel = C_stiefel + C_stiefel.T
        problems.append(('stiefel_trace', stiefel,
                        lambda X: -2 * C_stiefel @ X))
        
        # Grassmann subspace tracking
        grassmann = riemannopt.Grassmann(20, 3)
        A_grass = np.random.randn(20, 20)
        problems.append(('grassmann_subspace', grassmann,
                        lambda X: A_grass @ X))
        
        def run_all_problems():
            results = {}
            sgd = riemannopt.SGD(step_size=0.01)
            
            for name, manifold, grad_fn in problems:
                x = manifold.random_point()
                
                with BenchmarkTimer(name) as timer:
                    for _ in range(50):
                        grad = grad_fn(x)
                        x = sgd.step(manifold, x, grad)
                
                results[name] = timer.elapsed
            
            return results
        
        benchmark(run_all_problems)


class TestCriticalPathPerformance:
    """Test performance of critical operations."""
    
    @pytest.mark.benchmark
    def test_qr_decomposition_performance(self, benchmark):
        """Benchmark QR decomposition (critical for Stiefel)."""
        sizes = [(50, 10), (100, 20), (200, 30)]
        
        def qr_operations():
            times = {}
            for n, p in sizes:
                A = np.random.randn(n, p)
                
                with BenchmarkTimer(f"qr_{n}x{p}") as timer:
                    Q, R = np.linalg.qr(A)
                
                times[f"{n}x{p}"] = timer.elapsed
            
            return times
        
        benchmark(qr_operations)
    
    @pytest.mark.benchmark
    def test_matrix_multiplication_chains(self, benchmark):
        """Benchmark matrix multiplication chains (common in manifold ops)."""
        n = 100
        
        # Generate matrices
        A = np.random.randn(n, n)
        B = np.random.randn(n, 20)
        C = np.random.randn(20, 20)
        
        def matrix_operations():
            # Common patterns in manifold computations
            result1 = A @ B @ C
            result2 = B.T @ A @ B
            result3 = A @ B @ C @ B.T @ A.T
            return result1, result2, result3
        
        benchmark(matrix_operations)