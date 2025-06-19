# RiemannOpt

[![Crates.io](https://img.shields.io/crates/v/riemannopt.svg)](https://crates.io/crates/riemannopt)
[![Documentation](https://docs.rs/riemannopt/badge.svg)](https://docs.rs/riemannopt)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance Riemannian optimization library for Rust.

## Overview

RiemannOpt is a comprehensive library for optimization on Riemannian manifolds. It provides a unified interface for various manifolds and optimization algorithms, with a focus on performance and ease of use.

## Features

- **Multiple manifolds**: Sphere, Stiefel, Grassmann, SPD, Hyperbolic, and more
- **Various optimizers**: SGD, Adam, L-BFGS, Trust Region, Conjugate Gradient
- **High performance**: 10-100x faster than pure Python implementations
- **Parallel computing**: Automatic CPU parallelization with intelligent thresholds
- **Type safety**: Leverage Rust's type system for correctness
- **Python bindings**: Use from Python with minimal overhead

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
riemannopt = "0.1"
```

### Example: PCA on Stiefel Manifold

```rust
use riemannopt::prelude::*;
use nalgebra::{DMatrix, DVector};

// Create data matrix (n samples × p features)
let data = DMatrix::<f64>::new_random(100, 50);

// Compute covariance matrix
let cov = &data.transpose() * &data / 100.0;

// Create Stiefel manifold for p×k orthonormal matrices
let manifold = Stiefel::new(50, 10)?;

// Define cost function (negative trace of projection)
let cost_fn = QuadraticCost::new(cov.clone());

// Setup optimizer
let optimizer = SGD::new(
    SGDConfig::new()
        .with_step_size(StepSizeSchedule::Constant(0.01))
        .with_momentum(MomentumMethod::Nesterov { coefficient: 0.9 })
);

// Random initialization
let x0 = manifold.random_point();

// Optimize
let result = optimizer.optimize(
    &cost_fn, 
    &manifold, 
    &x0,
    &StoppingCriterion::new()
        .with_max_iterations(1000)
        .with_gradient_tolerance(1e-6)
)?;

println!("Converged in {} iterations", result.iterations);
println!("Final cost: {}", result.value);
```

## Available Manifolds

- **Sphere** `S^{n-1}`: Unit vectors in R^n
- **Stiefel** `St(n,p)`: Orthonormal n×p matrices
- **Grassmann** `Gr(n,p)`: p-dimensional subspaces of R^n
- **SPD** `S^+_n`: Symmetric positive definite matrices
- **Hyperbolic** `H^n`: Hyperbolic space
- **Fixed-Rank**: Low-rank matrix manifold
- **Product**: Cartesian products of manifolds
- **Oblique**: Product of spheres

## Available Optimizers

- **SGD**: Stochastic gradient descent with momentum
- **Adam**: Adaptive moment estimation
- **L-BFGS**: Limited-memory quasi-Newton method
- **Trust Region**: Robust second-order method
- **Conjugate Gradient**: Memory-efficient first-order method
- **Natural Gradient**: Information geometry-based method

## Performance

RiemannOpt is designed for high performance:

- **Parallel operations**: Automatic parallelization for large-scale problems
- **SIMD acceleration**: Vectorized operations using CPU instructions
- **Memory efficiency**: Careful memory management and reuse
- **Zero-cost abstractions**: No runtime overhead from trait abstractions

Benchmarks show 10-100x speedup compared to pure Python implementations.

## Documentation

- [API Documentation](https://docs.rs/riemannopt)
- [User Guide](https://github.com/yourusername/riemannopt/wiki)
- [Examples](https://github.com/yourusername/riemannopt/tree/main/examples)

## Contributing

We welcome contributions! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## Citation

If you use RiemannOpt in your research, please cite:

```bibtex
@software{riemannopt,
  author = {Your Name},
  title = {RiemannOpt: High-Performance Riemannian Optimization in Rust},
  year = {2024},
  url = {https://github.com/yourusername/riemannopt}
}
```

## License

MIT License. See [LICENSE](../../LICENSE) for details.