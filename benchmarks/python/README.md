# RiemannOpt Python Benchmarks

Comprehensive benchmarking suite for comparing Riemannian optimization libraries.

## Overview

This benchmarking system compares the performance of:
- **RiemannOpt**: High-performance Rust implementation with Python bindings
- **PyManopt**: Pure Python implementation
- **Geomstats**: Python implementation with geometric deep learning focus

## Features

- **Comprehensive Coverage**: Tests all major manifold operations
- **Multiple Scales**: From tiny (n=10) to huge (n=100,000) problem sizes
- **Statistical Rigor**: Warmup rounds, multiple measurements, standard deviations
- **Visual Analysis**: Automatic generation of comparison charts
- **Extensible Design**: Easy to add new manifolds, operations, or libraries

## Quick Start

### 1. Setup Environment

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Benchmarks

```bash
# Full benchmark comparison
./run_benchmarks.sh

# Or manually:
python benchmark_comparison.py
```

### 3. Visualize Results

```bash
# Visualizations are generated automatically, or manually:
python visualize_comparison.py results/comparison_*.json
```

## Benchmark Architecture

### Core Components

1. **benchmark_comparison.py**: Main comparison framework
   - Library wrappers for uniform API
   - Statistical timing measurements
   - Result aggregation and storage

2. **visualize_comparison.py**: Visualization suite
   - Performance comparison plots
   - Speedup heatmaps
   - Scaling analysis charts

3. **config.yaml**: Benchmark configuration
   - Problem sizes for each manifold
   - Measurement parameters
   - Library-specific settings

### Manifolds Tested

- **Sphere**: S^n with dimensions 10 to 50,000
- **Stiefel**: St(n,p) with sizes from 10×5 to 1000×200
- **Grassmann**: Gr(n,p) with sizes from 10×5 to 500×100
- **SPD**: Symmetric positive definite matrices (when available)
- **Hyperbolic**: Hyperbolic space (when available)

### Operations Benchmarked

- **Projection**: Project point onto manifold
- **Tangent Projection**: Project vector to tangent space
- **Retraction**: Move along geodesic
- **Inner Product**: Riemannian inner product
- **Norm**: Induced norm computation

## Results

### Performance Summary (Demo Results)

| Library | Average Time (ms) | Relative Speed |
|---------|------------------|----------------|
| RiemannOpt | 1.14 | 2.1x faster |
| PyManopt | 2.38 | 1.0x (baseline) |
| Geomstats | 4.11 | 0.6x slower |

### Scaling Analysis

All libraries show similar scaling characteristics:
- **Sphere**: O(n^0.2) - Excellent scalability
- **Stiefel**: O(n^0.56) - Good scalability
- **Grassmann**: O(n^0.53) - Good scalability

## Visualizations

The benchmark suite generates several visualizations:

1. **comparison_*.png**: Side-by-side operation comparisons
2. **speedup_comparison.png**: Relative performance bars
3. **operation_heatmap.png**: Performance matrix across all tests
4. **overall_comparison.png**: Summary performance statistics
5. **scaling_comparison.png**: Log-log scaling plots

## Extending the Benchmarks

### Adding a New Library

1. Create a wrapper class in `benchmark_comparison.py`:
```python
class NewLibraryWrapper(LibraryWrapper):
    def __init__(self):
        super().__init__("newlibrary")
        # Import and setup
    
    def sphere_projection(self, x):
        # Implement operation
```

2. Add to the libraries dictionary:
```python
self.libraries = {
    'newlibrary': NewLibraryWrapper(),
    # ...
}
```

### Adding a New Manifold

1. Add operations to wrapper classes
2. Create benchmark method in `BenchmarkRunner`
3. Update visualization code if needed

### Adding a New Operation

1. Add to operation lists in benchmark methods
2. Implement in all wrapper classes
3. Update visualization labels

## Troubleshooting

### Library Not Available

If a library shows as "not available":
1. Check installation: `pip install [library]`
2. Verify import path in wrapper class
3. Check for API changes in newer versions

### Memory Issues

For very large problems:
1. Reduce measurement rounds in config
2. Run specific size ranges
3. Use memory profiling: `python -m memory_profiler benchmark_comparison.py`

### Visualization Errors

1. Ensure all dependencies installed: `pip install matplotlib seaborn plotly`
2. Check for empty result files
3. Verify data types match expected formats

## Performance Tips

1. **Disable CPU frequency scaling** for consistent results
2. **Close other applications** to reduce variance
3. **Run multiple times** and average results
4. **Use release builds** for compiled libraries

## Contributing

When adding benchmarks:
1. Follow existing code patterns
2. Document new operations clearly
3. Include appropriate test sizes
4. Update this README

## License

Same as RiemannOpt project.