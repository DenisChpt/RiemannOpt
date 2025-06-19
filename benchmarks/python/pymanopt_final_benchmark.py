#!/usr/bin/env python3
"""
Final corrected PyManopt benchmark with proper API usage.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import gc

import pymanopt
from pymanopt.manifolds import Sphere, Stiefel, Grassmann


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
    
    def benchmark_sphere(self):
        """Benchmark sphere operations."""
        print("\n" + "="*60)
        print("SPHERE MANIFOLD BENCHMARKS (PyManopt)")
        print("="*60)
        
        sizes = [10, 50, 100, 500, 1000, 5000, 10000, 50000]
        
        for n in sizes:
            print(f"\n--- Dimension: {n} ---")
            
            try:
                # Create manifold
                manifold = Sphere(n)
                
                # Test data
                x = np.random.randn(n)
                
                # Benchmark projection to manifold
                stats = self.time_operation(manifold.projection, x, x)
                print(f"Projection to manifold: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'sphere',
                    'operation': 'projection',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Get point on manifold
                x_on_sphere = manifold.projection(x, x)
                v = np.random.randn(n)
                
                # Benchmark tangent projection
                stats = self.time_operation(manifold.to_tangent_space, x_on_sphere, v)
                print(f"Tangent projection: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'sphere',
                    'operation': 'tangent_projection',
                    'size': n,
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Get tangent vector
                v_tangent = manifold.to_tangent_space(x_on_sphere, v)
                
                # Benchmark retraction
                stats = self.time_operation(manifold.retraction, x_on_sphere, v_tangent)
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
                stats = self.time_operation(manifold.inner_product, x_on_sphere, v_tangent, v_tangent)
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
            (200, 50), (500, 100), (1000, 200)
        ]
        
        for n, p in sizes:
            print(f"\n--- Size: {n}x{p} ---")
            
            try:
                # Create manifold
                manifold = Stiefel(n, p)
                
                # Test data
                X = np.random.randn(n, p)
                
                # Benchmark projection to manifold
                stats = self.time_operation(manifold.projection, X, X)
                print(f"Projection to manifold: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'stiefel',
                    'operation': 'projection',
                    'size': (n, p),
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Get point on manifold
                X_on_stiefel = manifold.projection(X, X)
                V = np.random.randn(n, p)
                
                # Benchmark tangent projection
                stats = self.time_operation(manifold.to_tangent_space, X_on_stiefel, V)
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
                
                # Get tangent vector
                V_tangent = manifold.to_tangent_space(X_on_stiefel, V)
                
                # Benchmark retraction
                stats = self.time_operation(manifold.retraction, X_on_stiefel, V_tangent)
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
                
                # Benchmark inner product
                stats = self.time_operation(manifold.inner_product, X_on_stiefel, V_tangent, V_tangent)
                print(f"Inner product: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'stiefel',
                    'operation': 'inner_product',
                    'size': (n, p),
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
            except Exception as e:
                print(f"Error at size {n}x{p}: {e}")
    
    def benchmark_grassmann(self):
        """Benchmark Grassmann operations."""
        print("\n" + "="*60)
        print("GRASSMANN MANIFOLD BENCHMARKS (PyManopt)")
        print("="*60)
        
        sizes = [
            (10, 5), (20, 10), (50, 20), (100, 20),
            (200, 50), (500, 100)
        ]
        
        for n, p in sizes:
            print(f"\n--- Size: {n}x{p} ---")
            
            try:
                # Create manifold
                manifold = Grassmann(n, p)
                
                # Test data - Grassmann needs orthonormal matrices
                X = np.random.randn(n, p)
                X_orth, _ = np.linalg.qr(X)
                
                # Random matrix for tangent
                V = np.random.randn(n, p)
                
                # Benchmark tangent projection
                stats = self.time_operation(manifold.to_tangent_space, X_orth, V)
                print(f"Tangent projection: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'grassmann',
                    'operation': 'tangent_projection',
                    'size': (n, p),
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Get tangent vector
                V_tangent = manifold.to_tangent_space(X_orth, V)
                
                # Benchmark retraction
                stats = self.time_operation(manifold.retraction, X_orth, V_tangent)
                print(f"Retraction: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'grassmann',
                    'operation': 'retraction',
                    'size': (n, p),
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
                # Benchmark inner product
                stats = self.time_operation(manifold.inner_product, X_orth, V_tangent, V_tangent)
                print(f"Inner product: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                self.results.append({
                    'manifold': 'grassmann',
                    'operation': 'inner_product',
                    'size': (n, p),
                    'time_ms': stats['mean'] * 1000,
                    'std_ms': stats['std'] * 1000,
                    'min_ms': stats['min'] * 1000,
                    'max_ms': stats['max'] * 1000
                })
                
            except Exception as e:
                print(f"Error at size {n}x{p}: {e}")
    
    def save_results(self):
        """Save results to files."""
        results_dir = Path("results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_path = results_dir / f"pymanopt_results_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to: {json_path}")
        
        # Save as CSV
        df = pd.DataFrame(self.results)
        csv_path = results_dir / f"pymanopt_results_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        return df
    
    def print_summary(self, df):
        """Print summary statistics."""
        print("\n" + "="*60)
        print("SUMMARY: PyManopt Performance")
        print("="*60)
        
        # Check if dataframe is empty
        if df.empty:
            print("No results to summarize")
            return
        
        # Group by manifold and operation
        for manifold in df['manifold'].unique():
            print(f"\n{manifold.upper()} MANIFOLD:")
            manifold_df = df[df['manifold'] == manifold]
            
            # Create pivot table for better visualization
            pivot = manifold_df.pivot_table(
                values='time_ms',
                index='size',
                columns='operation',
                aggfunc='mean'
            )
            
            print(pivot.round(3))
            
            # Print scaling information
            print(f"\nScaling analysis for {manifold}:")
            for operation in manifold_df['operation'].unique():
                op_df = manifold_df[manifold_df['operation'] == operation].sort_values('size')
                if len(op_df) > 1:
                    # Calculate scaling factor
                    sizes = []
                    times = []
                    for _, row in op_df.iterrows():
                        size = row['size']
                        if isinstance(size, tuple):
                            sizes.append(np.prod(size))
                        else:
                            sizes.append(size)
                        times.append(row['time_ms'])
                    
                    if len(sizes) > 2:
                        # Log-log regression for complexity analysis
                        log_sizes = np.log(sizes[1:])
                        log_times = np.log(times[1:])
                        slope, _ = np.polyfit(log_sizes, log_times, 1)
                        print(f"  {operation}: O(n^{slope:.2f})")
    
    def print_performance_report(self, df):
        """Print a detailed performance report."""
        print("\n" + "="*60)
        print("PERFORMANCE REPORT")
        print("="*60)
        
        if df.empty:
            print("No data for performance report")
            return
        
        # Operations summary
        print("\nOperations timing summary (ms):")
        print("-" * 40)
        
        summary = df.groupby(['manifold', 'operation'])['time_ms'].agg(['mean', 'min', 'max'])
        print(summary.round(3))
        
        # Find slowest operations
        print("\nSlowest operations:")
        print("-" * 40)
        
        slowest = df.nlargest(5, 'time_ms')[['manifold', 'operation', 'size', 'time_ms']]
        for _, row in slowest.iterrows():
            size_str = str(row['size'])
            print(f"{row['manifold']}.{row['operation']} @ size {size_str}: {row['time_ms']:.3f} ms")
        
        # Find fastest operations
        print("\nFastest operations:")
        print("-" * 40)
        
        fastest = df.nsmallest(5, 'time_ms')[['manifold', 'operation', 'size', 'time_ms']]
        for _, row in fastest.iterrows():
            size_str = str(row['size'])
            print(f"{row['manifold']}.{row['operation']} @ size {size_str}: {row['time_ms']:.3f} ms")
        
        # Performance characteristics
        print("\n\nPerformance Characteristics:")
        print("-" * 40)
        
        # Analyze by manifold
        for manifold in df['manifold'].unique():
            mdf = df[df['manifold'] == manifold]
            avg_time = mdf['time_ms'].mean()
            print(f"\n{manifold.upper()}:")
            print(f"  Average operation time: {avg_time:.3f} ms")
            
            # Most expensive operation
            slowest_op = mdf.loc[mdf['time_ms'].idxmax()]
            print(f"  Slowest: {slowest_op['operation']} @ size {slowest_op['size']} ({slowest_op['time_ms']:.3f} ms)")
            
            # Least expensive operation
            fastest_op = mdf.loc[mdf['time_ms'].idxmin()]
            print(f"  Fastest: {fastest_op['operation']} @ size {fastest_op['size']} ({fastest_op['time_ms']:.3f} ms)")
    
    def run(self):
        """Run all benchmarks."""
        print("PyManopt Performance Benchmark")
        print(f"Started at: {datetime.now()}")
        print(f"Warmup rounds: {self.warmup_rounds}")
        print(f"Measurement rounds: {self.measurement_rounds}")
        
        # Run benchmarks
        self.benchmark_sphere()
        self.benchmark_stiefel()
        self.benchmark_grassmann()
        
        # Save and summarize
        df = self.save_results()
        self.print_summary(df)
        
        print(f"\nCompleted at: {datetime.now()}")
        
        # Print performance report
        self.print_performance_report(df)


def main():
    """Run PyManopt benchmark."""
    benchmark = PyManoptBenchmark()
    benchmark.run()


if __name__ == "__main__":
    main()