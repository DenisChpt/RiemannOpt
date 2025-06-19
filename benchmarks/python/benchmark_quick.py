#!/usr/bin/env python3
"""
Quick benchmark comparison with smaller sizes for demonstration.
"""

import sys
sys.path.insert(0, '.')
from benchmark_comparison import BenchmarkRunner
from datetime import datetime

def main():
    """Run quick benchmark with smaller sizes."""
    runner = BenchmarkRunner()
    
    # Override with smaller test sizes
    print("Quick Benchmark - Smaller Sizes for Fast Results")
    print(f"Started at: {datetime.now()}")
    print(f"Warmup rounds: {runner.warmup_rounds}")
    print(f"Measurement rounds: {runner.measurement_rounds}")
    
    # Check available libraries
    print("\nLibrary availability:")
    for name, lib in runner.libraries.items():
        status = "✓ Available" if lib.is_available() else f"✗ Not available: {lib.get_error()}"
        print(f"  {name}: {status}")
    
    # Define smaller test sizes
    sphere_sizes = [10, 100, 1000]
    stiefel_sizes = [(10, 5), (50, 20), (100, 20)]
    grassmann_sizes = [(10, 5), (50, 20)]
    
    # Run benchmarks
    runner.benchmark_sphere(sphere_sizes)
    runner.benchmark_stiefel(stiefel_sizes)
    runner.benchmark_grassmann(grassmann_sizes)
    
    # Save and summarize
    df = runner.save_results()
    runner.print_summary(df)
    
    print(f"\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()