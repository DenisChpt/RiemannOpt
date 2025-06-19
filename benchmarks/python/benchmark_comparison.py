#!/usr/bin/env python3
"""
Comprehensive benchmark comparison between RiemannOpt, PyManopt, and Geomstats.
"""

import time
import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
import gc
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any
import yaml

# Load configuration
def load_config():
    """Load configuration from config.yaml."""
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

# Configuration
config = load_config()
WARMUP_ROUNDS = config.get('benchmark', {}).get('warmup_rounds', 3)
MEASUREMENT_ROUNDS = config.get('benchmark', {}).get('measurement_rounds', 10)
RESULTS_DIR = Path(config.get('output', {}).get('output_dir', 'results'))
ENABLED_LIBRARIES = config.get('libraries', ['riemannopt', 'pymanopt', 'geomstats'])


@dataclass
class BenchmarkResult:
    """Store benchmark results."""
    library: str
    manifold: str
    operation: str
    size: Any
    time_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    success: bool
    error: Optional[str] = None


class LibraryWrapper:
    """Base class for library wrappers."""
    
    def __init__(self, name: str):
        self.name = name
        self.available = False
        self.error_msg = None
    
    def is_available(self) -> bool:
        return self.available
    
    def get_error(self) -> str:
        return self.error_msg or "Library not available"


class PyManoptWrapper(LibraryWrapper):
    """Wrapper for PyManopt operations."""
    
    def __init__(self):
        super().__init__("pymanopt")
        try:
            import pymanopt
            from pymanopt.manifolds import Sphere, Stiefel, Grassmann
            self.Sphere = Sphere
            self.Stiefel = Stiefel
            self.Grassmann = Grassmann
            self.available = True
        except ImportError as e:
            self.error_msg = str(e)
    
    def sphere_projection(self, x: np.ndarray) -> np.ndarray:
        """Project point to sphere."""
        manifold = self.Sphere(len(x))
        return manifold.projection(x, x)
    
    def sphere_tangent_projection(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Project vector to tangent space of sphere."""
        manifold = self.Sphere(len(x))
        x_on = manifold.projection(x, x)
        return manifold.to_tangent_space(x_on, v)
    
    def sphere_retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Retract on sphere."""
        manifold = self.Sphere(len(x))
        x_on = manifold.projection(x, x)
        v_tan = manifold.to_tangent_space(x_on, v)
        return manifold.retraction(x_on, v_tan)
    
    def stiefel_projection(self, X: np.ndarray) -> np.ndarray:
        """Project matrix to Stiefel manifold."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        return manifold.projection(X, X)
    
    def stiefel_tangent_projection(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Project matrix to tangent space of Stiefel."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        X_on = manifold.projection(X, X)
        return manifold.to_tangent_space(X_on, V)
    
    def stiefel_retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Retract on Stiefel manifold."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        X_on = manifold.projection(X, X)
        V_tan = manifold.to_tangent_space(X_on, V)
        return manifold.retraction(X_on, V_tan)
    
    def grassmann_tangent_projection(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Project to tangent space of Grassmann."""
        n, p = X.shape
        manifold = self.Grassmann(n, p)
        Q, _ = np.linalg.qr(X)
        X_on = Q[:, :p]
        return manifold.to_tangent_space(X_on, V)
    
    def grassmann_retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Retract on Grassmann manifold."""
        n, p = X.shape
        manifold = self.Grassmann(n, p)
        Q, _ = np.linalg.qr(X)
        X_on = Q[:, :p]
        V_tan = manifold.to_tangent_space(X_on, V)
        return manifold.retraction(X_on, V_tan)


class GeomstatsWrapper(LibraryWrapper):
    """Wrapper for Geomstats operations."""
    
    def __init__(self):
        super().__init__("geomstats")
        try:
            import geomstats
            # Geomstats v2.8+ doesn't need backend setting
            from geomstats.geometry.hypersphere import Hypersphere
            from geomstats.geometry.special_orthogonal import SpecialOrthogonal
            self.Hypersphere = Hypersphere
            # Note: Geomstats doesn't have direct Stiefel/Grassmann, we'll use workarounds
            self.available = True
            self.has_stiefel = False
            self.has_grassmann = False
        except Exception as e:
            self.error_msg = str(e)
            self.available = False
    
    def sphere_projection(self, x: np.ndarray) -> np.ndarray:
        """Project point to sphere."""
        manifold = self.Hypersphere(dim=len(x)-1)
        return manifold.projection(x)
    
    def sphere_tangent_projection(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Project vector to tangent space of sphere."""
        manifold = self.Hypersphere(dim=len(x)-1)
        x_on = manifold.projection(x)
        return manifold.to_tangent(v, x_on)
    
    def sphere_retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Exponential map on sphere."""
        manifold = self.Hypersphere(dim=len(x)-1)
        x_on = manifold.projection(x)
        v_tan = manifold.to_tangent(v, x_on)
        return manifold.exp(v_tan, x_on)
    
    def stiefel_projection(self, X: np.ndarray) -> np.ndarray:
        """Project matrix to Stiefel manifold."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        return manifold.projection(X)
    
    def stiefel_tangent_projection(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Project matrix to tangent space of Stiefel."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        X_on = manifold.projection(X)
        return manifold.to_tangent(V, X_on)
    
    def stiefel_retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Exponential map on Stiefel manifold."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        X_on = manifold.projection(X)
        V_tan = manifold.to_tangent(V, X_on)
        return manifold.exp(V_tan, X_on)
    
    def grassmann_tangent_projection(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Project to tangent space of Grassmann."""
        n, p = X.shape
        manifold = self.Grassmann(n, p)
        Q, _ = np.linalg.qr(X)
        X_on = Q[:, :p]
        return manifold.to_tangent(V, X_on)
    
    def grassmann_retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Exponential map on Grassmann manifold."""
        n, p = X.shape
        manifold = self.Grassmann(n, p)
        Q, _ = np.linalg.qr(X)
        X_on = Q[:, :p]
        V_tan = manifold.to_tangent(V, X_on)
        return manifold.exp(V_tan, X_on)


class RiemannOptWrapper(LibraryWrapper):
    """Wrapper for RiemannOpt operations."""
    
    def __init__(self):
        super().__init__("riemannopt")
        try:
            # Try to add parent directory to path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            import riemannopt
            from riemannopt import Sphere, Stiefel, Grassmann
            self.Sphere = Sphere
            self.Stiefel = Stiefel
            self.Grassmann = Grassmann
            self.available = True
        except ImportError as e:
            self.error_msg = str(e)
    
    def sphere_projection(self, x: np.ndarray) -> np.ndarray:
        """Project point to sphere."""
        manifold = self.Sphere(len(x))
        return manifold.project(x)
    
    def sphere_tangent_projection(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Project vector to tangent space of sphere."""
        manifold = self.Sphere(len(x))
        x_on = manifold.project(x)
        return manifold.tangent_projection(x_on, v)
    
    def sphere_retraction(self, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Retract on sphere."""
        manifold = self.Sphere(len(x))
        x_on = manifold.project(x)
        v_tan = manifold.tangent_projection(x_on, v)
        return manifold.retraction(x_on, v_tan)
    
    def stiefel_projection(self, X: np.ndarray) -> np.ndarray:
        """Project matrix to Stiefel manifold."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        return manifold.project(X)
    
    def stiefel_tangent_projection(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Project matrix to tangent space of Stiefel."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        X_on = manifold.project(X)
        return manifold.tangent_projection(X_on, V)
    
    def stiefel_retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Retract on Stiefel manifold."""
        n, p = X.shape
        manifold = self.Stiefel(n, p)
        X_on = manifold.project(X)
        V_tan = manifold.tangent_projection(X_on, V)
        return manifold.retraction(X_on, V_tan)
    
    def grassmann_tangent_projection(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Project to tangent space of Grassmann."""
        n, p = X.shape
        manifold = self.Grassmann(n, p)
        Q, _ = np.linalg.qr(X)
        X_on = Q[:, :p]
        return manifold.tangent_projection(X_on, V)
    
    def grassmann_retraction(self, X: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Retract on Grassmann manifold."""
        n, p = X.shape
        manifold = self.Grassmann(n, p)
        Q, _ = np.linalg.qr(X)
        X_on = Q[:, :p]
        V_tan = manifold.tangent_projection(X_on, V)
        return manifold.retraction(X_on, V_tan)


class BenchmarkRunner:
    """Run benchmarks across all libraries."""
    
    def __init__(self):
        self.libraries = {}
        if 'pymanopt' in ENABLED_LIBRARIES:
            self.libraries['pymanopt'] = PyManoptWrapper()
        if 'geomstats' in ENABLED_LIBRARIES:
            self.libraries['geomstats'] = GeomstatsWrapper()
        if 'riemannopt' in ENABLED_LIBRARIES:
            self.libraries['riemannopt'] = RiemannOptWrapper()
        self.results = []
        self.warmup_rounds = WARMUP_ROUNDS
        self.measurement_rounds = MEASUREMENT_ROUNDS
    
    def time_operation(self, func, *args, **kwargs) -> Dict[str, float]:
        """Time a single operation with warmup."""
        # Warmup
        for _ in range(self.warmup_rounds):
            try:
                func(*args, **kwargs)
            except:
                pass
            gc.collect()
        
        # Measure
        times = []
        for _ in range(self.measurement_rounds):
            gc.collect()
            start = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                end = time.perf_counter()
                times.append(end - start)
            except:
                # If operation fails, return None
                return None
        
        if not times:
            return None
        
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
    
    def benchmark_sphere(self, sizes: List[int]):
        """Benchmark sphere operations."""
        print("\n" + "="*60)
        print("SPHERE MANIFOLD BENCHMARKS")
        print("="*60)
        
        operations = [
            ('projection', lambda lib, x: lib.sphere_projection(x)),
            ('tangent_projection', lambda lib, x: lib.sphere_tangent_projection(x, np.random.randn(len(x)))),
            ('retraction', lambda lib, x: lib.sphere_retraction(x, np.random.randn(len(x))))
        ]
        
        for n in sizes:
            print(f"\n--- Dimension: {n} ---")
            x = np.random.randn(n)
            
            for op_name, op_func in operations:
                print(f"\n{op_name}:")
                
                for lib_name, lib in self.libraries.items():
                    if not lib.is_available():
                        print(f"  {lib_name}: Not available ({lib.get_error()})")
                        continue
                    
                    try:
                        stats = self.time_operation(lambda: op_func(lib, x))
                        if stats:
                            print(f"  {lib_name}: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                            self.results.append(BenchmarkResult(
                                library=lib_name,
                                manifold='sphere',
                                operation=op_name,
                                size=n,
                                time_ms=stats['mean'] * 1000,
                                std_ms=stats['std'] * 1000,
                                min_ms=stats['min'] * 1000,
                                max_ms=stats['max'] * 1000,
                                success=True
                            ))
                        else:
                            print(f"  {lib_name}: Failed")
                            self.results.append(BenchmarkResult(
                                library=lib_name,
                                manifold='sphere',
                                operation=op_name,
                                size=n,
                                time_ms=0,
                                std_ms=0,
                                min_ms=0,
                                max_ms=0,
                                success=False,
                                error="Operation failed"
                            ))
                    except Exception as e:
                        print(f"  {lib_name}: Error - {str(e)}")
                        self.results.append(BenchmarkResult(
                            library=lib_name,
                            manifold='sphere',
                            operation=op_name,
                            size=n,
                            time_ms=0,
                            std_ms=0,
                            min_ms=0,
                            max_ms=0,
                            success=False,
                            error=str(e)
                        ))
    
    def benchmark_stiefel(self, sizes: List[Tuple[int, int]]):
        """Benchmark Stiefel operations."""
        print("\n" + "="*60)
        print("STIEFEL MANIFOLD BENCHMARKS")
        print("="*60)
        
        operations = [
            ('projection', lambda lib, X: lib.stiefel_projection(X)),
            ('tangent_projection', lambda lib, X: lib.stiefel_tangent_projection(X, np.random.randn(*X.shape))),
            ('retraction', lambda lib, X: lib.stiefel_retraction(X, np.random.randn(*X.shape)))
        ]
        
        for n, p in sizes:
            print(f"\n--- Size: {n}x{p} ---")
            X = np.random.randn(n, p)
            
            for op_name, op_func in operations:
                print(f"\n{op_name}:")
                
                for lib_name, lib in self.libraries.items():
                    if not lib.is_available():
                        continue
                    
                    try:
                        stats = self.time_operation(lambda: op_func(lib, X))
                        if stats:
                            print(f"  {lib_name}: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                            self.results.append(BenchmarkResult(
                                library=lib_name,
                                manifold='stiefel',
                                operation=op_name,
                                size=(n, p),
                                time_ms=stats['mean'] * 1000,
                                std_ms=stats['std'] * 1000,
                                min_ms=stats['min'] * 1000,
                                max_ms=stats['max'] * 1000,
                                success=True
                            ))
                        else:
                            print(f"  {lib_name}: Failed")
                    except Exception as e:
                        print(f"  {lib_name}: Error - {str(e)}")
    
    def benchmark_grassmann(self, sizes: List[Tuple[int, int]]):
        """Benchmark Grassmann operations."""
        print("\n" + "="*60)
        print("GRASSMANN MANIFOLD BENCHMARKS")
        print("="*60)
        
        operations = [
            ('tangent_projection', lambda lib, X: lib.grassmann_tangent_projection(X, np.random.randn(*X.shape))),
            ('retraction', lambda lib, X: lib.grassmann_retraction(X, np.random.randn(*X.shape)))
        ]
        
        for n, p in sizes:
            print(f"\n--- Size: {n}x{p} ---")
            X = np.random.randn(n, p)
            
            for op_name, op_func in operations:
                print(f"\n{op_name}:")
                
                for lib_name, lib in self.libraries.items():
                    if not lib.is_available():
                        continue
                    
                    try:
                        stats = self.time_operation(lambda: op_func(lib, X))
                        if stats:
                            print(f"  {lib_name}: {stats['mean']*1000:.3f} ± {stats['std']*1000:.3f} ms")
                            self.results.append(BenchmarkResult(
                                library=lib_name,
                                manifold='grassmann',
                                operation=op_name,
                                size=(n, p),
                                time_ms=stats['mean'] * 1000,
                                std_ms=stats['std'] * 1000,
                                min_ms=stats['min'] * 1000,
                                max_ms=stats['max'] * 1000,
                                success=True
                            ))
                        else:
                            print(f"  {lib_name}: Failed")
                    except Exception as e:
                        print(f"  {lib_name}: Error - {str(e)}")
    
    def save_results(self):
        """Save results to files."""
        RESULTS_DIR.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert results to dict
        results_dict = []
        for r in self.results:
            results_dict.append({
                'library': r.library,
                'manifold': r.manifold,
                'operation': r.operation,
                'size': r.size,
                'time_ms': r.time_ms,
                'std_ms': r.std_ms,
                'min_ms': r.min_ms,
                'max_ms': r.max_ms,
                'success': r.success,
                'error': r.error
            })
        
        # Save as JSON
        json_path = RESULTS_DIR / f"comparison_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(results_dict, f, indent=2, default=str)
        print(f"\nResults saved to: {json_path}")
        
        # Save as CSV
        df = pd.DataFrame(results_dict)
        csv_path = RESULTS_DIR / f"comparison_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"Results saved to: {csv_path}")
        
        return df
    
    def print_summary(self, df: pd.DataFrame):
        """Print summary with library comparisons."""
        print("\n" + "="*60)
        print("PERFORMANCE COMPARISON SUMMARY")
        print("="*60)
        
        # Filter successful operations only
        df_success = df[df['success'] == True]
        
        if df_success.empty:
            print("No successful operations to compare")
            return
        
        # Get available libraries
        available_libs = df_success['library'].unique()
        print(f"\nAvailable libraries: {', '.join(available_libs)}")
        
        # Comparison by manifold and operation
        for manifold in df_success['manifold'].unique():
            print(f"\n{manifold.upper()} MANIFOLD:")
            mdf = df_success[df_success['manifold'] == manifold]
            
            for operation in mdf['operation'].unique():
                print(f"\n  {operation}:")
                op_df = mdf[mdf['operation'] == operation]
                
                # Create comparison table
                for size in sorted(op_df['size'].unique()):
                    size_df = op_df[op_df['size'] == size]
                    size_str = str(size) if not isinstance(size, tuple) else f"{size[0]}x{size[1]}"
                    print(f"    Size {size_str}:")
                    
                    # Get times for each library
                    times = {}
                    for lib in available_libs:
                        lib_data = size_df[size_df['library'] == lib]
                        if not lib_data.empty:
                            times[lib] = lib_data['time_ms'].values[0]
                    
                    # Print times and calculate speedups
                    if times:
                        min_time = min(times.values())
                        fastest_lib = [k for k, v in times.items() if v == min_time][0]
                        
                        for lib, time_ms in sorted(times.items()):
                            speedup = times[fastest_lib] / time_ms if time_ms > 0 else 0
                            marker = " ⚡" if lib == fastest_lib else ""
                            print(f"      {lib}: {time_ms:.3f} ms (speedup: {speedup:.2f}x){marker}")
    
    def run_full_benchmark(self):
        """Run complete benchmark suite."""
        print("Riemannian Manifold Operations - Library Comparison")
        print(f"Started at: {datetime.now()}")
        print(f"Warmup rounds: {self.warmup_rounds}")
        print(f"Measurement rounds: {self.measurement_rounds}")
        
        # Check available libraries
        print("\nLibrary availability:")
        print(f"Enabled libraries from config: {', '.join(ENABLED_LIBRARIES)}")
        for name in ENABLED_LIBRARIES:
            if name in self.libraries:
                lib = self.libraries[name]
                status = "✓ Available" if lib.is_available() else f"✗ Not available: {lib.get_error()}"
                print(f"  {name}: {status}")
            else:
                print(f"  {name}: ✗ Not initialized (check library name in config)")
        
        # Define test sizes
        sphere_sizes = [10, 100, 1000, 10000, 50000]
        stiefel_sizes = [(10, 5), (50, 20), (100, 20), (500, 100), (1000, 200)]
        grassmann_sizes = [(10, 5), (50, 20), (100, 20), (200, 50), (500, 100)]
        
        # Run benchmarks
        self.benchmark_sphere(sphere_sizes)
        self.benchmark_stiefel(stiefel_sizes)
        self.benchmark_grassmann(grassmann_sizes)
        
        # Save and summarize
        df = self.save_results()
        self.print_summary(df)
        
        print(f"\nCompleted at: {datetime.now()}")
        
        return df


def main():
    """Main entry point."""
    runner = BenchmarkRunner()
    df = runner.run_full_benchmark()
    
    # Print final statistics
    print("\n" + "="*60)
    print("FINAL STATISTICS")
    print("="*60)
    
    successful_ops = len(df[df['success'] == True])
    failed_ops = len(df[df['success'] == False])
    
    print(f"\nTotal operations benchmarked: {len(df)}")
    print(f"Successful: {successful_ops}")
    print(f"Failed: {failed_ops}")
    
    # Show which library is fastest overall
    df_success = df[df['success'] == True]
    if not df_success.empty:
        avg_times = df_success.groupby('library')['time_ms'].mean()
        fastest = avg_times.idxmin()
        print(f"\nFastest library (on average): {fastest} ({avg_times[fastest]:.3f} ms)")
        
        print("\nAverage operation time by library:")
        for lib in sorted(avg_times.index):
            print(f"  {lib}: {avg_times[lib]:.3f} ms")


if __name__ == "__main__":
    main()