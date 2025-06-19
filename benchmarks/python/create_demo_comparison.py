#!/usr/bin/env python3
"""
Create a demo comparison with simulated results for multiple libraries.
This demonstrates how the benchmark comparison would look with all libraries available.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path

def create_simulated_results():
    """Create realistic benchmark results for demonstration."""
    results = []
    
    # Simulate performance characteristics for each library
    # RiemannOpt is fastest, PyManopt is baseline, Geomstats is slower
    performance_factors = {
        'riemannopt': 0.5,    # 2x faster than PyManopt
        'pymanopt': 1.0,      # Baseline
        'geomstats': 1.8      # 1.8x slower than PyManopt
    }
    
    # Base times for PyManopt (from real benchmarks)
    base_times = {
        'sphere': {
            'projection': {10: 0.106, 100: 0.117, 1000: 0.111, 10000: 0.125, 50000: 0.197},
            'tangent_projection': {10: 0.130, 100: 0.129, 1000: 0.159, 10000: 0.345, 50000: 1.438},
            'retraction': {10: 0.151, 100: 0.146, 1000: 0.170, 10000: 0.390, 50000: 1.516}
        },
        'stiefel': {
            'projection': {(10, 5): 0.074, (50, 20): 0.091, (100, 20): 0.091, (500, 100): 0.419, (1000, 200): 2.456},
            'tangent_projection': {(10, 5): 0.097, (50, 20): 0.132, (100, 20): 0.163, (500, 100): 21.597, (1000, 200): 9.661},
            'retraction': {(10, 5): 0.367, (50, 20): 0.426, (100, 20): 0.447, (500, 100): 7.940, (1000, 200): 25.155}
        },
        'grassmann': {
            'tangent_projection': {(10, 5): 0.166, (50, 20): 0.207, (100, 20): 0.238, (200, 50): 0.860, (500, 100): 5.362},
            'retraction': {(10, 5): 0.214, (50, 20): 0.351, (100, 20): 0.404, (200, 50): 2.224, (500, 100): 10.813}
        }
    }
    
    # Generate results for each library
    for library, factor in performance_factors.items():
        for manifold, operations in base_times.items():
            for operation, sizes in operations.items():
                for size, base_time in sizes.items():
                    # Add some realistic variation
                    mean_time = base_time * factor * np.random.uniform(0.9, 1.1)
                    std_time = mean_time * 0.1  # 10% standard deviation
                    
                    results.append({
                        'library': library,
                        'manifold': manifold,
                        'operation': operation,
                        'size': size,
                        'time_ms': mean_time,
                        'std_ms': std_time,
                        'min_ms': mean_time - 2 * std_time,
                        'max_ms': mean_time + 2 * std_time,
                        'success': True,
                        'error': None
                    })
    
    # Add some failed operations for realism
    for lib in ['geomstats']:
        results.append({
            'library': lib,
            'manifold': 'hyperbolic',
            'operation': 'projection',
            'size': 100,
            'time_ms': 0,
            'std_ms': 0,
            'min_ms': 0,
            'max_ms': 0,
            'success': False,
            'error': 'Not implemented'
        })
    
    return results

def main():
    """Create and save demo comparison results."""
    results = create_simulated_results()
    
    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"demo_comparison_{timestamp}.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Demo comparison results saved to: {output_path}")
    print(f"Total operations: {len(results)}")
    print(f"Libraries included: riemannopt, pymanopt, geomstats")
    print("\nTo visualize: python visualize_comparison.py " + str(output_path))

if __name__ == "__main__":
    main()