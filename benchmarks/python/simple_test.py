#!/usr/bin/env python3
"""
Simple test to verify benchmark setup and run basic comparisons.
"""

import sys
import time
import numpy as np
from pathlib import Path

# Add parent directory to path for riemannopt imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Try imports
libraries_available = {}

try:
    import riemannopt
    libraries_available['riemannopt'] = True
    print("✓ RiemannOpt available")
except ImportError as e:
    libraries_available['riemannopt'] = False
    print(f"✗ RiemannOpt not available: {e}")

try:
    import pymanopt
    from pymanopt.manifolds import Sphere as PymanoptSphere
    libraries_available['pymanopt'] = True
    print("✓ PyManopt available")
except ImportError as e:
    libraries_available['pymanopt'] = False
    print(f"✗ PyManopt not available: {e}")

try:
    import geomstats
    from geomstats.geometry import Hypersphere
    libraries_available['geomstats'] = True
    print("✓ Geomstats available")
except ImportError as e:
    libraries_available['geomstats'] = False
    print(f"✗ Geomstats not available: {e}")


def time_operation(func, *args, **kwargs):
    """Time a single operation."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    return end - start, result


def benchmark_sphere_projection(n=1000, iterations=100):
    """Benchmark sphere projection across libraries."""
    print(f"\n{'='*60}")
    print(f"Sphere Projection Benchmark (n={n}, iterations={iterations})")
    print(f"{'='*60}")
    
    # Generate test data
    x = np.random.randn(n)
    results = {}
    
    # RiemannOpt
    if libraries_available['riemannopt']:
        try:
            sphere = riemannopt.manifolds.Sphere(n)
            times = []
            for _ in range(iterations):
                t, _ = time_operation(sphere.project, x)
                times.append(t)
            avg_time = np.mean(times) * 1000  # Convert to ms
            results['riemannopt'] = avg_time
            print(f"RiemannOpt: {avg_time:.3f} ms")
        except Exception as e:
            print(f"RiemannOpt error: {e}")
    
    # PyManopt
    if libraries_available['pymanopt']:
        try:
            sphere = PymanoptSphere(n)
            times = []
            for _ in range(iterations):
                t, _ = time_operation(sphere.proj, x)
                times.append(t)
            avg_time = np.mean(times) * 1000
            results['pymanopt'] = avg_time
            print(f"PyManopt: {avg_time:.3f} ms")
        except Exception as e:
            print(f"PyManopt error: {e}")
    
    # Geomstats
    if libraries_available['geomstats']:
        try:
            sphere = Hypersphere(dim=n-1)
            times = []
            for _ in range(iterations):
                t, _ = time_operation(sphere.projection, x)
                times.append(t)
            avg_time = np.mean(times) * 1000
            results['geomstats'] = avg_time
            print(f"Geomstats: {avg_time:.3f} ms")
        except Exception as e:
            print(f"Geomstats error: {e}")
    
    # Calculate speedups
    if 'riemannopt' in results and len(results) > 1:
        print(f"\nSpeedup vs RiemannOpt:")
        for lib, time_ms in results.items():
            if lib != 'riemannopt':
                speedup = time_ms / results['riemannopt']
                print(f"  {lib}: {speedup:.2f}x")


def benchmark_stiefel_projection(n=500, p=50, iterations=100):
    """Benchmark Stiefel projection across libraries."""
    print(f"\n{'='*60}")
    print(f"Stiefel Projection Benchmark (n={n}, p={p}, iterations={iterations})")
    print(f"{'='*60}")
    
    # Generate test data
    X = np.random.randn(n, p)
    results = {}
    
    # RiemannOpt
    if libraries_available['riemannopt']:
        try:
            stiefel = riemannopt.manifolds.Stiefel(n, p)
            X_flat = X.flatten()  # RiemannOpt expects flattened
            times = []
            for _ in range(iterations):
                t, _ = time_operation(stiefel.project, X_flat)
                times.append(t)
            avg_time = np.mean(times) * 1000
            results['riemannopt'] = avg_time
            print(f"RiemannOpt: {avg_time:.3f} ms")
        except Exception as e:
            print(f"RiemannOpt error: {e}")
    
    # PyManopt
    if libraries_available['pymanopt']:
        try:
            from pymanopt.manifolds import Stiefel as PymanoptStiefel
            stiefel = PymanoptStiefel(n, p)
            times = []
            for _ in range(iterations):
                t, _ = time_operation(stiefel.proj, X)
                times.append(t)
            avg_time = np.mean(times) * 1000
            results['pymanopt'] = avg_time
            print(f"PyManopt: {avg_time:.3f} ms")
        except Exception as e:
            print(f"PyManopt error: {e}")
    
    # Calculate speedups
    if 'riemannopt' in results and len(results) > 1:
        print(f"\nSpeedup vs RiemannOpt:")
        for lib, time_ms in results.items():
            if lib != 'riemannopt':
                speedup = time_ms / results['riemannopt']
                print(f"  {lib}: {speedup:.2f}x")


def run_scaling_test():
    """Test scaling behavior."""
    print(f"\n{'='*60}")
    print("Scaling Test - Sphere Projection")
    print(f"{'='*60}")
    
    sizes = [100, 1000, 10000, 100000]
    
    for n in sizes:
        print(f"\nSize n={n}:")
        x = np.random.randn(n)
        
        if libraries_available['riemannopt']:
            try:
                sphere = riemannopt.manifolds.Sphere(n)
                t, _ = time_operation(sphere.project, x)
                print(f"  RiemannOpt: {t*1000:.3f} ms")
            except Exception as e:
                print(f"  RiemannOpt error: {e}")
        
        if libraries_available['pymanopt']:
            try:
                sphere = PymanoptSphere(n)
                t, _ = time_operation(sphere.proj, x)
                print(f"  PyManopt: {t*1000:.3f} ms")
            except Exception as e:
                print(f"  PyManopt error: {e}")


def main():
    """Run simple benchmarks."""
    print("RiemannOpt Simple Benchmark Test")
    print("================================\n")
    
    # Check what's available
    if not any(libraries_available.values()):
        print("\nNo libraries available for benchmarking!")
        print("Please install at least one of: riemannopt, pymanopt, geomstats")
        return
    
    # Run benchmarks
    benchmark_sphere_projection(n=1000, iterations=100)
    benchmark_sphere_projection(n=10000, iterations=20)
    
    benchmark_stiefel_projection(n=500, p=50, iterations=20)
    benchmark_stiefel_projection(n=1000, p=100, iterations=10)
    
    run_scaling_test()
    
    print("\n\nBenchmark test complete!")


if __name__ == "__main__":
    main()