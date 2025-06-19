#!/usr/bin/env python3
"""
RiemannOpt Comprehensive Benchmark Suite

This module provides a complete benchmarking framework for comparing
RiemannOpt with other Riemannian optimization libraries.
"""

import time
import json
import csv
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from contextlib import contextmanager

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
import psutil
import gc

# Import benchmark targets
try:
    import riemannopt
    RIEMANNOPT_AVAILABLE = True
except ImportError:
    RIEMANNOPT_AVAILABLE = False
    print("Warning: riemannopt not available")

try:
    import pymanopt
    from pymanopt.manifolds import Sphere as PymanoptSphere
    from pymanopt.manifolds import Stiefel as PymanoptStiefel
    from pymanopt.manifolds import Grassmann as PymanoptGrassmann
    from pymanopt.optimizers import SteepestDescent
    PYMANOPT_AVAILABLE = True
except ImportError:
    PYMANOPT_AVAILABLE = False
    print("Warning: pymanopt not available")

try:
    import geomstats
    from geomstats.geometry import Hypersphere
    from geomstats.geometry import SpecialOrthogonal
    GEOMSTATS_AVAILABLE = True
except ImportError:
    GEOMSTATS_AVAILABLE = False
    print("Warning: geomstats not available")


@dataclass
class BenchmarkResult:
    """Stores results from a single benchmark run."""
    library: str
    manifold: str
    operation: str
    size: Any  # Can be int or tuple
    time_seconds: float
    memory_mb: float = 0.0
    iterations: int = 1
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def time_per_iter_ms(self) -> float:
        """Time per iteration in milliseconds."""
        return (self.time_seconds / self.iterations) * 1000
    
    @property
    def ops_per_second(self) -> float:
        """Operations per second."""
        return self.iterations / self.time_seconds if self.time_seconds > 0 else 0


class BenchmarkTimer:
    """Context manager for timing operations with memory tracking."""
    
    def __init__(self, warmup_rounds: int = 5, measurement_rounds: int = 20):
        self.warmup_rounds = warmup_rounds
        self.measurement_rounds = measurement_rounds
        self.times = []
        self.memory_usage = []
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    @contextmanager
    def measure(self):
        """Measure a single operation."""
        gc.collect()
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        start = time.perf_counter()
        yield
        end = time.perf_counter()
        
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        self.times.append(end - start)
        self.memory_usage.append(mem_after - mem_before)
    
    def run(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """Run benchmark with warmup and measurements."""
        # Warmup
        for _ in range(self.warmup_rounds):
            func(*args, **kwargs)
        
        # Measure
        self.times.clear()
        self.memory_usage.clear()
        
        for _ in range(self.measurement_rounds):
            with self.measure():
                func(*args, **kwargs)
        
        # Calculate statistics
        times = np.array(self.times)
        memory = np.array(self.memory_usage)
        
        return {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'min_time': np.min(times),
            'max_time': np.max(times),
            'median_time': np.median(times),
            'mean_memory': np.mean(memory),
            'total_runs': len(times)
        }


class ManifoldBenchmark:
    """Base class for manifold benchmarks."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results: List[BenchmarkResult] = []
        self.timer = BenchmarkTimer(
            warmup_rounds=config['benchmark']['warmup_rounds'],
            measurement_rounds=config['benchmark']['measurement_rounds']
        )
    
    def benchmark_operation(self, library: str, manifold_name: str, 
                          operation: str, size: Any, func: Callable,
                          *args, **kwargs) -> Optional[BenchmarkResult]:
        """Benchmark a single operation."""
        try:
            stats = self.timer.run(func, *args, **kwargs)
            
            return BenchmarkResult(
                library=library,
                manifold=manifold_name,
                operation=operation,
                size=size,
                time_seconds=stats['mean_time'],
                memory_mb=stats['mean_memory'],
                iterations=stats['total_runs'],
                metadata={
                    'std_time': stats['std_time'],
                    'min_time': stats['min_time'],
                    'max_time': stats['max_time'],
                    'median_time': stats['median_time']
                }
            )
        except Exception as e:
            return BenchmarkResult(
                library=library,
                manifold=manifold_name,
                operation=operation,
                size=size,
                time_seconds=0.0,
                error=str(e)
            )
    
    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all benchmarks for this manifold."""
        raise NotImplementedError


class SphereBenchmark(ManifoldBenchmark):
    """Benchmarks for sphere manifold."""
    
    def create_manifolds(self, n: int) -> Dict[str, Any]:
        """Create sphere manifolds for each library."""
        manifolds = {}
        
        if RIEMANNOPT_AVAILABLE:
            manifolds['riemannopt'] = riemannopt.manifolds.Sphere(n)
        
        if PYMANOPT_AVAILABLE:
            manifolds['pymanopt'] = PymanoptSphere(n)
        
        if GEOMSTATS_AVAILABLE:
            manifolds['geomstats'] = Hypersphere(dim=n-1)
        
        return manifolds
    
    def create_test_point(self, n: int) -> np.ndarray:
        """Create a random point on the sphere."""
        x = np.random.randn(n)
        return x / np.linalg.norm(x)
    
    def create_test_tangent(self, x: np.ndarray) -> np.ndarray:
        """Create a random tangent vector."""
        v = np.random.randn(len(x))
        v = v - np.dot(v, x) * x  # Project to tangent space
        return v
    
    def benchmark_projection(self, manifolds: Dict[str, Any], n: int) -> List[BenchmarkResult]:
        """Benchmark point projection."""
        results = []
        x = np.random.randn(n)  # Non-normalized point
        
        # RiemannOpt
        if 'riemannopt' in manifolds:
            result = self.benchmark_operation(
                'riemannopt', 'sphere', 'projection', n,
                lambda: manifolds['riemannopt'].project(x)
            )
            if result:
                results.append(result)
        
        # PyManopt
        if 'pymanopt' in manifolds:
            result = self.benchmark_operation(
                'pymanopt', 'sphere', 'projection', n,
                lambda: manifolds['pymanopt'].proj(x)
            )
            if result:
                results.append(result)
        
        # Geomstats
        if 'geomstats' in manifolds:
            result = self.benchmark_operation(
                'geomstats', 'sphere', 'projection', n,
                lambda: manifolds['geomstats'].projection(x)
            )
            if result:
                results.append(result)
        
        return results
    
    def benchmark_retraction(self, manifolds: Dict[str, Any], n: int) -> List[BenchmarkResult]:
        """Benchmark retraction."""
        results = []
        x = self.create_test_point(n)
        v = self.create_test_tangent(x)
        
        # RiemannOpt
        if 'riemannopt' in manifolds:
            result = self.benchmark_operation(
                'riemannopt', 'sphere', 'retraction', n,
                lambda: manifolds['riemannopt'].retract(x, v)
            )
            if result:
                results.append(result)
        
        # PyManopt
        if 'pymanopt' in manifolds:
            result = self.benchmark_operation(
                'pymanopt', 'sphere', 'retraction', n,
                lambda: manifolds['pymanopt'].retr(x, v)
            )
            if result:
                results.append(result)
        
        # Geomstats
        if 'geomstats' in manifolds:
            result = self.benchmark_operation(
                'geomstats', 'sphere', 'retraction', n,
                lambda: manifolds['geomstats'].exp(v, x)
            )
            if result:
                results.append(result)
        
        return results
    
    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all sphere benchmarks."""
        results = []
        
        for size_category, sizes in self.config['sizes'].items():
            print(f"\nRunning sphere benchmarks for {size_category} sizes...")
            
            for n in tqdm(sizes.get('sphere', []), desc=f"Sphere {size_category}"):
                manifolds = self.create_manifolds(n)
                
                # Basic operations
                if 'projection' in self.config['operations']['basic']:
                    results.extend(self.benchmark_projection(manifolds, n))
                
                if 'retraction' in self.config['operations']['basic']:
                    results.extend(self.benchmark_retraction(manifolds, n))
        
        return results


class StiefelBenchmark(ManifoldBenchmark):
    """Benchmarks for Stiefel manifold."""
    
    def create_manifolds(self, n: int, p: int) -> Dict[str, Any]:
        """Create Stiefel manifolds for each library."""
        manifolds = {}
        
        if RIEMANNOPT_AVAILABLE:
            manifolds['riemannopt'] = riemannopt.manifolds.Stiefel(n, p)
        
        if PYMANOPT_AVAILABLE:
            manifolds['pymanopt'] = PymanoptStiefel(n, p)
        
        return manifolds
    
    def create_test_point(self, n: int, p: int) -> np.ndarray:
        """Create a random point on Stiefel manifold."""
        X = np.random.randn(n, p)
        Q, _ = np.linalg.qr(X)
        return Q[:, :p]
    
    def benchmark_projection(self, manifolds: Dict[str, Any], n: int, p: int) -> List[BenchmarkResult]:
        """Benchmark point projection."""
        results = []
        X = np.random.randn(n, p)  # Non-orthonormal matrix
        
        # RiemannOpt
        if 'riemannopt' in manifolds:
            # Flatten for riemannopt (expects vectors)
            X_flat = X.flatten()
            result = self.benchmark_operation(
                'riemannopt', 'stiefel', 'projection', (n, p),
                lambda: manifolds['riemannopt'].project(X_flat)
            )
            if result:
                results.append(result)
        
        # PyManopt
        if 'pymanopt' in manifolds:
            result = self.benchmark_operation(
                'pymanopt', 'stiefel', 'projection', (n, p),
                lambda: manifolds['pymanopt'].proj(X)
            )
            if result:
                results.append(result)
        
        return results
    
    def run_benchmarks(self) -> List[BenchmarkResult]:
        """Run all Stiefel benchmarks."""
        results = []
        
        for size_category, sizes in self.config['sizes'].items():
            print(f"\nRunning Stiefel benchmarks for {size_category} sizes...")
            
            for n, p in tqdm(sizes.get('stiefel', []), desc=f"Stiefel {size_category}"):
                manifolds = self.create_manifolds(n, p)
                
                # Basic operations
                if 'projection' in self.config['operations']['basic']:
                    results.extend(self.benchmark_projection(manifolds, n, p))
        
        return results


class BenchmarkRunner:
    """Main benchmark runner."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.results: List[BenchmarkResult] = []
        self.start_time = None
        self.end_time = None
    
    def run(self) -> None:
        """Run all benchmarks."""
        self.start_time = datetime.now()
        print(f"Starting benchmark suite at {self.start_time}")
        print(f"Available libraries: {self._get_available_libraries()}")
        
        # Run manifold benchmarks
        benchmarks = [
            SphereBenchmark(self.config),
            StiefelBenchmark(self.config),
            # Add more manifold benchmarks here
        ]
        
        for benchmark in benchmarks:
            self.results.extend(benchmark.run_benchmarks())
        
        self.end_time = datetime.now()
        print(f"\nBenchmark completed at {self.end_time}")
        print(f"Total runtime: {self.end_time - self.start_time}")
        
        # Save results
        self.save_results()
        self.print_summary()
    
    def _get_available_libraries(self) -> List[str]:
        """Get list of available libraries."""
        libs = []
        if RIEMANNOPT_AVAILABLE:
            libs.append('riemannopt')
        if PYMANOPT_AVAILABLE:
            libs.append('pymanopt')
        if GEOMSTATS_AVAILABLE:
            libs.append('geomstats')
        return libs
    
    def save_results(self) -> None:
        """Save results in multiple formats."""
        output_dir = Path(self.config['output']['output_dir'])
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        if 'json' in self.config['output']['format']:
            json_path = output_dir / f"results_{timestamp}.json"
            with open(json_path, 'w') as f:
                json.dump([asdict(r) for r in self.results], f, indent=2)
            print(f"Results saved to {json_path}")
        
        # Save as CSV
        if 'csv' in self.config['output']['format']:
            csv_path = output_dir / f"results_{timestamp}.csv"
            df = pd.DataFrame([asdict(r) for r in self.results])
            df.to_csv(csv_path, index=False)
            print(f"Results saved to {csv_path}")
    
    def print_summary(self) -> None:
        """Print summary of results."""
        if not self.results:
            print("No results to summarize")
            return
        
        # Group results
        df = pd.DataFrame([asdict(r) for r in self.results])
        
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Summary by operation
        for manifold in df['manifold'].unique():
            print(f"\n{manifold.upper()} MANIFOLD")
            print("-"*40)
            
            manifold_df = df[df['manifold'] == manifold]
            
            for operation in manifold_df['operation'].unique():
                print(f"\nOperation: {operation}")
                op_df = manifold_df[manifold_df['operation'] == operation]
                
                # Create pivot table
                pivot = op_df.pivot_table(
                    values='time_per_iter_ms',
                    index='size',
                    columns='library',
                    aggfunc='mean'
                )
                
                print(pivot.round(3))
                
                # Calculate speedups
                if 'riemannopt' in pivot.columns and len(pivot.columns) > 1:
                    print("\nSpeedup vs other libraries:")
                    for lib in pivot.columns:
                        if lib != 'riemannopt':
                            speedup = pivot[lib] / pivot['riemannopt']
                            print(f"  vs {lib}: {speedup.mean():.2f}x")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RiemannOpt Benchmark Suite")
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    parser.add_argument('--sizes', choices=['tiny', 'small', 'medium', 'large', 'huge'],
                       help='Run only specific size category')
    args = parser.parse_args()
    
    # Modify config if size specified
    if args.sizes:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        # Keep only specified size
        config['sizes'] = {args.sizes: config['sizes'][args.sizes]}
        
        # Save temporary config
        temp_config = 'temp_config.yaml'
        with open(temp_config, 'w') as f:
            yaml.dump(config, f)
        
        runner = BenchmarkRunner(temp_config)
        os.remove(temp_config)
    else:
        runner = BenchmarkRunner(args.config)
    
    runner.run()


if __name__ == "__main__":
    main()