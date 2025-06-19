# RiemannOpt Manifolds

[![Crates.io](https://img.shields.io/crates/v/riemannopt-manifolds.svg)](https://crates.io/crates/riemannopt-manifolds)
[![Documentation](https://docs.rs/riemannopt-manifolds/badge.svg)](https://docs.rs/riemannopt-manifolds)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Comprehensive collection of Riemannian manifolds for optimization.

## Overview

This crate provides efficient implementations of commonly used Riemannian manifolds in optimization, machine learning, and scientific computing. All manifolds implement the unified `Manifold` trait from `riemannopt-core`.

## Implemented Manifolds

### Sphere Manifold `S^{n-1}`
Unit vectors in n-dimensional space.
- **Applications**: Normalized features, directional data, spherical PCA
- **Key features**: Exact exponential/logarithmic maps, SIMD acceleration

### Stiefel Manifold `St(n,p)`
Matrices with orthonormal columns (n×p with p≤n).
- **Applications**: PCA, ICA, orthogonal neural networks
- **Key features**: Multiple retractions (QR, polar, Cayley), parallel QR

### Grassmann Manifold `Gr(n,p)`
p-dimensional linear subspaces of R^n.
- **Applications**: Subspace tracking, computer vision, model reduction
- **Key features**: Quotient geometry, principal angles distance

### Symmetric Positive Definite `SPD(n)`
n×n symmetric positive definite matrices.
- **Applications**: Covariance matrices, kernel learning, medical imaging
- **Key features**: Affine-invariant metric, Log-Euclidean metric

### Hyperbolic Space `H^n`
Constant negative curvature space.
- **Applications**: Hierarchical embeddings, tree-like data, NLP
- **Key features**: Poincaré ball model, Möbius operations

### Product Manifold
Cartesian product of manifolds M₁ × M₂ × ... × Mₖ.
- **Applications**: Multi-task learning, structured optimization
- **Key features**: Component-wise operations, mixed manifolds

### Fixed-Rank Manifold
Matrices with fixed rank r.
- **Applications**: Matrix completion, recommender systems
- **Key features**: SVD-based representation, efficient storage

### Oblique Manifold
Product of unit spheres (S¹)^n.
- **Applications**: Normalized columns, probability simplices
- **Key features**: Column-wise operations, diagonal metric

## Testing

The crate has a comprehensive test suite:

### Unit Tests

Run unit tests (inline with source):
```bash
cargo test --lib
```

### Integration Tests

Run integration tests:
```bash
cargo test --test integration_tests
```

### Performance Tests

Run performance tests:
```bash
cargo test --test performance_tests
```

### Stress Tests

Run stress tests with large dimensions:
```bash
cargo test --test performance_tests -- --ignored
```

### All Tests

Run all tests:
```bash
cargo test
```

## Test Structure

```
tests/
├── integration_tests.rs    # Basic functionality and cross-manifold tests
└── performance_tests.rs    # Performance benchmarks and stress tests

src/
├── sphere.rs              # Sphere implementation with inline tests
├── stiefel.rs             # Stiefel implementation with inline tests
├── grassmann.rs           # Grassmann implementation with inline tests
├── spd.rs                 # SPD implementation with inline tests
├── hyperbolic.rs          # Hyperbolic implementation with inline tests
└── product.rs             # Product manifold with inline tests
```

## Test Independence

This crate can be tested independently without any external dependencies:
- No file I/O in tests
- No dependencies on example files
- All test data generated programmatically
- Only depends on `riemannopt-core` for trait definitions

## Usage

Add to your `Cargo.toml`:

```toml
[dependencies]
riemannopt-manifolds = "0.1"
```

### Basic Example

```rust
use riemannopt_manifolds::{Sphere, Stiefel};
use riemannopt_core::manifold::Manifold;

// Create a sphere in R^10
let sphere = Sphere::new(10)?;
let point = sphere.random_point();
let tangent = sphere.random_tangent(&point)?;

// Create a Stiefel manifold St(5,2)
let stiefel = Stiefel::new(5, 2)?;
let matrix = stiefel.random_point();
```

## Performance

All manifolds are optimized for performance:

- **Parallel operations**: Automatic parallelization for large dimensions
- **SIMD acceleration**: Vectorized operations for sphere projections
- **Efficient algorithms**: State-of-the-art numerical methods
- **Memory efficiency**: Minimal allocations in hot paths

### Benchmarks

| Operation | Size | Time |
|-----------|------|------|
| Sphere projection | n=1000 | < 10 μs |
| Stiefel retraction | 50×10 | < 5 ms |
| Grassmann distance | 20×5 | < 10 ms |
| SPD logarithm | 100×100 | < 50 ms |

## Feature Flags

- `parallel` (default): Enable CPU parallelization
- `serde`: Serialization support

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](../../LICENSE) for details.