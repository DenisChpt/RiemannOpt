# RiemannOpt Core

[![Crates.io](https://img.shields.io/crates/v/riemannopt-core.svg)](https://crates.io/crates/riemannopt-core)
[![Documentation](https://docs.rs/riemannopt-core/badge.svg)](https://docs.rs/riemannopt-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Core traits and types for Riemannian optimization in Rust.

## Overview

`riemannopt-core` provides the foundational abstractions for implementing Riemannian optimization algorithms. It defines the mathematical traits needed to work with smooth manifolds, tangent spaces, and Riemannian metrics.

## Features

- **Manifold trait**: Abstract interface for Riemannian manifolds
- **Optimizer framework**: Traits for optimization algorithms
- **Cost functions**: Interface for objective functions on manifolds
- **Retractions**: Various retraction methods (exponential map, QR, polar)
- **Vector transport**: Parallel transport and vector transport methods
- **Line search**: Backtracking and strong Wolfe line search
- **Parallel computing**: Intelligent thresholds for CPU parallelization
- **SIMD support**: Vectorized operations for performance
- **GPU support** (optional): CUDA acceleration framework

## Usage

Add this to your `Cargo.toml`:

```toml
[dependencies]
riemannopt-core = "0.1"
```

### Basic Example

```rust
use riemannopt_core::prelude::*;

// Define a manifold (implementation required)
struct MyManifold;

impl<T: Scalar> Manifold<T, Dyn> for MyManifold {
    fn name(&self) -> &str { "MyManifold" }
    fn dimension(&self) -> usize { 10 }
    // ... implement required methods
}

// Use with optimization
let manifold = MyManifold;
let point = manifold.random_point();
let tangent = manifold.random_tangent(&point)?;
let retracted = manifold.retract(&point, &tangent)?;
```

## Key Traits

### Manifold

The core trait defining a Riemannian manifold:

```rust
pub trait Manifold<T: Scalar, D: Dim> {
    fn dimension(&self) -> usize;
    fn project_point(&self, point: &Point<T, D>) -> Point<T, D>;
    fn project_tangent(&self, point: &Point<T, D>, vector: &TangentVector<T, D>) 
        -> Result<TangentVector<T, D>>;
    fn inner_product(&self, point: &Point<T, D>, u: &TangentVector<T, D>, 
        v: &TangentVector<T, D>) -> Result<T>;
    fn retract(&self, point: &Point<T, D>, tangent: &TangentVector<T, D>) 
        -> Result<Point<T, D>>;
    // ... more methods
}
```

### Optimizer

Framework for optimization algorithms:

```rust
pub trait Optimizer<T: Scalar, D: Dim> {
    fn optimize<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>;
}
```

## Feature Flags

- `std` (default): Standard library support
- `serde`: Serialization support
- `parallel` (default): CPU parallel computing
- `cuda`: GPU acceleration (requires CUDA toolkit)
- `test-utils`: Testing utilities

## Performance Features

### Parallel Computing

The crate includes an intelligent threshold system that automatically determines when to use parallel execution:

```rust
use riemannopt_core::parallel_thresholds::{ParallelDecision, ParallelThresholdsBuilder};

// Configure thresholds
let config = ParallelThresholdsBuilder::new()
    .vector_threshold(5000)
    .matrix_matrix_threshold(64)
    .build();

// Check if operation should be parallelized
if ParallelDecision::dot_product::<f64>(vector_size) {
    // Use parallel implementation
}
```

### SIMD Operations

SIMD-accelerated operations for supported types:

```rust
use riemannopt_core::simd::{SimdOps, SimdVectorOps};

// Automatic SIMD acceleration when beneficial
let result = SimdOps::dot_product(&x, &y);
```

## Dependencies

- `nalgebra`: Linear algebra operations
- `num-traits`: Generic numeric traits
- `rayon`: Data parallelism
- `wide`: SIMD operations
- `thiserror`: Error handling

## License

MIT License. See [LICENSE](../../LICENSE) for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.