#!/usr/bin/env python3
"""
Benchmark PyManopt operations with different sizes.
This will serve as a baseline for RiemannOpt comparisons.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, List, Tuple
import gc

import pymanopt
from pymanopt.manifolds import Sphere, Stiefel, Grassmann
from pymanopt.manifolds import SymmetricPositiveDefinite as SPD


class PyManoptBenchmark:
    """Benchmark PyManopt manifold operations."""
    
    def __init__(self):
        self.results = []
        self.warmup_rounds = 3
        self.measurement_rounds = 10
    
    def time_operation(self, func, *args, **kwargs):
        """Time a single operation with warmup."""
        # Warmup
        for _ in range(self.warmup_rounds):
            func(*args, **kwargs)
            gc.collect()
        
        # Measure
        times = []
        for _ in range(self.measurement_rounds):
            gc.collect()
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    def format_size(self, size):
        """Format size for display."""
        if isinstance(size, tuple):
            return f"{size[0]}x{size[1]}"
        return str(size)
    
    def benchmark_sphere(self):
        """Benchmark sphere operations."""
        print("\n" + "="*60)
        print("SPHERE MANIFOLD BENCHMARKS (PyManopt)")
        print("="*60)
        
        sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000]
        
        for n in sizes:
            print(f"\n--- Dimension: {n} ---")
            
            try:
                # Create manifold
                manifold = Sphere(n)
                
                # Test data
                x = np.random.randn(n)
                x_on_sphere = x / np.linalg.norm(x)
                v = np.random.randn(n)
                v_tangent = v - np.dot(v, x_on_sphere) * x_on_sphere
                
                # Benchmark projection
                stats = self.time_operation(manifold.proj, x)
                print(f"Projection: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'sphere',
                    'operation': 'projection',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Benchmark retraction
                stats = self.time_operation(manifold.retr, x_on_sphere, v_tangent)
                print(f"Retraction: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'sphere',
                    'operation': 'retraction',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Benchmark inner product
                stats = self.time_operation(manifold.inner, x_on_sphere, v_tangent, v_tangent)
                print(f"Inner product: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'sphere',
                    'operation': 'inner_product',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Benchmark norm
                stats = self.time_operation(manifold.norm, x_on_sphere, v_tangent)
                print(f"Norm: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'sphere',
                    'operation': 'norm',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
            except Exception as e:
                print(f"Error at size {n}: {e}")
    
    def benchmark_stiefel(self):
        """Benchmark Stiefel operations."""
        print("\n" + "="*60)
        print("STIEFEL MANIFOLD BENCHMARKS (PyManopt)")
        print("="*60)
        
        sizes = [
            (10, 5), (20, 10), (50, 20), (100, 20),
            (200, 50), (500, 100), (1000, 200),
            (2000, 500), (5000, 1000)
        ]
        
        for n, p in sizes:
            print(f"\n--- Size: {n}x{p} ---")
            
            try:
                # Create manifold
                manifold = Stiefel(n, p)
                
                # Test data
                X = np.random.randn(n, p)
                Q, _ = np.linalg.qr(X)
                X_on_stiefel = Q[:, :p]
                V = np.random.randn(n, p)
                
                # Benchmark projection
                stats = self.time_operation(manifold.proj, X)
                print(f"Projection: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'stiefel',
                    'operation': 'projection',
                    'size': (n, p),
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Benchmark tangent projection
                stats = self.time_operation(manifold.proj, X_on_stiefel, V)
                print(f"Tangent projection: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'stiefel',
                    'operation': 'tangent_projection',
                    'size': (n, p),
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Benchmark retraction
                V_tangent = manifold.proj(X_on_stiefel, V)
                stats = self.time_operation(manifold.retr, X_on_stiefel, V_tangent)
                print(f"Retraction: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'stiefel',
                    'operation': 'retraction',
                    'size': (n, p),
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
            except Exception as e:
                print(f"Error at size {n}x{p}: {e}")
    
    def benchmark_spd(self):
        """Benchmark SPD operations."""
        print("\n" + "="*60)
        print("SPD MANIFOLD BENCHMARKS (PyManopt)")
        print("="*60)
        
        sizes = [5, 10, 20, 50, 100, 200, 500]
        
        for n in sizes:
            print(f"\n--- Size: {n}x{n} ---")
            
            try:
                # Create manifold
                manifold = SPD(n)
                
                # Test data - create positive definite matrix
                A = np.random.randn(n, n)
                X = A @ A.T + np.eye(n)  # Ensure positive definite
                
                # Random symmetric matrix for tangent vector
                B = np.random.randn(n, n)
                V = (B + B.T) / 2
                
                # Benchmark projection
                stats = self.time_operation(manifold.proj, X)
                print(f"Projection: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'spd',
                    'operation': 'projection',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Benchmark tangent projection
                stats = self.time_operation(manifold.proj, X, V)
                print(f"Tangent projection: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'spd',
                    'operation': 'tangent_projection',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Benchmark retraction
                V_tangent = manifold.proj(X, V)
                stats = self.time_operation(manifold.retr, X, V_tangent)
                print(f"Retraction: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'spd',
                    'operation': 'retraction',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
            except Exception as e:
                print(f"Error at size {n}: {e}")
    
    def save_results(self):
        """Save results to files."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = results_dir / f"pymanopt_benchmark_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {json_path}")
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = results_dir / f"pymanopt_benchmark_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        return df
    
    def print_summary(self, df):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("SUMMARY: PyManopt Performance")
        print("="*60)
        
        # Group by manifold and operation
        for manifold in df['manifold'].unique():
            print(f"\n{manifold.upper()} MANIFOLD:")
            manifold_df = df[df['manifold'] == manifold]
            
            for operation in manifold_df['operation'].unique():
                print(f"\n  {operation}:")
                op_df = manifold_df[manifold_df['operation'] == operation]
                
                for _, row in op_df.iterrows():
                    size_str = self.format_size(row['size'])
                    print(f"    Size {size_str}: {row['time_ms']:.3f} ± {row['std_ms']:.3f} ms")
    
    def run(self):
        """Run all benchmarks."""
        print("PyManopt Performance Benchmark")
        print(f"Started at: {datetime.now()}")
        print(f"Warmup rounds: {self.warmup_rounds}")
        print(f"Measurement rounds: {self.measurement_rounds}")
        
        # Run benchmarks
        self.benchmark_sphere()
        self.benchmark_stiefel()
        self.benchmark_spd()
        
        # Save and summarize
        df = self.save_results()
        self.print_summary(df)
        
        print(f"\nCompleted at: {datetime.now()}")


def main():
    """Run PyManopt benchmark."""
    benchmark = PyManoptBenchmark()
    benchmark.run()


if __name__ == "__main__":
    main()