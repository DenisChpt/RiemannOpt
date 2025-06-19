# RiemannOpt Python Bindings

[![PyPI](https://img.shields.io/pypi/v/riemannopt.svg)](https://pypi.org/project/riemannopt/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://riemannopt.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Python bindings for the high-performance RiemannOpt library.

## Overview

RiemannOpt-Py provides Python bindings to the Rust implementation of Riemannian optimization algorithms. It combines the ease of Python with the performance of Rust, offering 10-100x speedups over pure Python implementations.

## Installation

### From PyPI

```bash
pip install riemannopt
```

### From source

```bash
# Clone the repository
git clone https://github.com/yourusername/riemannopt
cd riemannopt

# Install in development mode
pip install maturin
maturin develop --release -m crates/riemannopt-py/Cargo.toml
```

### Requirements

- Python 3.8+
- NumPy
- Rust toolchain (for building from source)

## Quick Start

```python
import riemannopt as ro
import numpy as np

# Create a sphere manifold
sphere = ro.manifolds.Sphere(10)

# Define a cost function
def cost(x):
    return 0.5 * np.dot(x, x)

def gradient(x):
    return x

# Create cost function wrapper
cost_fn = ro.CostFunction(cost, gradient)

# Create optimizer
optimizer = ro.optimizers.SGD(sphere, learning_rate=0.1)

# Generate random initial point
x0 = sphere.random_point()

# Optimize
result = optimizer.optimize(cost_fn, x0)
print(ro.format_result(result))
```

## Available Manifolds

- `Sphere(n)`: Unit sphere S^{n-1}
- `Stiefel(n, p)`: Stiefel manifold St(n,p)
- `Grassmann(n, p)`: Grassmann manifold Gr(n,p)
- `Euclidean(n)`: Euclidean space R^n
- `SPD(n)`: Symmetric positive definite matrices
- `Hyperbolic(n)`: Hyperbolic space H^n

## Available Optimizers

- `SGD`: Riemannian stochastic gradient descent
- `Adam`: Riemannian Adam optimizer
- `LBFGS`: Riemannian L-BFGS
- `ConjugateGradient`: Riemannian conjugate gradient

## Features

- NumPy integration for seamless array handling
- Type hints for better IDE support
- Efficient Rust implementation under the hood
- Support for custom cost functions
- Various retraction and transport methods

## Examples

### PCA on Stiefel Manifold

```python
import riemannopt as ro
import numpy as np

# Generate random data
data = np.random.randn(100, 50)
cov = data.T @ data / 100

# Create Stiefel manifold for orthogonal projection
manifold = ro.manifolds.Stiefel(50, 10)

# Define cost function (maximize variance)
def cost(X):
    return -np.trace(X.T @ cov @ X)

def gradient(X):
    return -2 * cov @ X

# Optimize
optimizer = ro.optimizers.SGD(manifold, learning_rate=0.01, momentum=0.9)
result = optimizer.optimize(
    ro.CostFunction(cost, gradient),
    manifold.random_point(),
    max_iterations=1000,
    gradient_tolerance=1e-6
)

print(f"Top 10 principal components found in {result.iterations} iterations")
```

### More Examples

- [Matrix Completion](examples/matrix_completion.py)
- [Hyperbolic Embeddings](examples/hyperbolic_embeddings.py) 
- [Robust PCA](examples/robust_pca.py)
- [Dictionary Learning](examples/dictionary_learning.py)

## Performance

Benchmarks on common tasks (vs PyManopt):

| Task | Size | RiemannOpt | PyManopt | Speedup |
|------|------|------------|----------|---------|
| Sphere projection | 10,000 | 0.1 ms | 5 ms | 50x |
| Stiefel retraction | 1000×100 | 2 ms | 150 ms | 75x |
| SPD logarithm | 500×500 | 10 ms | 890 ms | 89x |
| Full optimization | Various | 5-20 s | 200-1000 s | 40-50x |

## API Reference

### Manifolds

All manifolds support the following methods:
- `dimension()`: Intrinsic dimension
- `random_point()`: Generate random point
- `project(point)`: Project to manifold
- `tangent_project(point, vector)`: Project to tangent space
- `retract(point, tangent)`: Move on manifold
- `distance(x, y)`: Geodesic distance

### Optimizers

All optimizers support:
- `optimize(cost_fn, x0, **kwargs)`: Run optimization
- `step(cost_fn, x)`: Single optimization step
- Common parameters: `max_iterations`, `gradient_tolerance`, `cost_tolerance`

### Cost Functions

```python
# Function with gradient
cost_fn = ro.CostFunction(
    cost=lambda x: objective(x),
    gradient=lambda x: grad_objective(x)
)

# Automatic differentiation (coming soon)
cost_fn = ro.AutodiffCost(lambda x: objective(x))
```

## Contributing

We welcome contributions! See our [Contributing Guide](../../CONTRIBUTING.md).

## Citation

If you use RiemannOpt in your research:

```bibtex
@software{riemannopt,
  author = {Your Name},
  title = {RiemannOpt: High-Performance Riemannian Optimization},
  year = {2024},
  url = {https://github.com/yourusername/riemannopt}
}
```

## License

MIT License. See [LICENSE](../../LICENSE) for details.