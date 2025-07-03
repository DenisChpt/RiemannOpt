# RiemannOpt Manifolds

[![Crates.io](https://img.shields.io/crates/v/riemannopt-manifolds.svg)](https://crates.io/crates/riemannopt-manifolds)
[![Documentation](https://docs.rs/riemannopt-manifolds/badge.svg)](https://docs.rs/riemannopt-manifolds)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance implementations of Riemannian manifolds for optimization, featuring zero-copy matrix operations, SIMD acceleration, and compile-time optimizations.

## Overview

This crate provides comprehensive implementations of commonly used Riemannian manifolds in optimization, machine learning, and scientific computing. All manifolds implement the unified `Manifold` trait from `riemannopt-core`, with additional optimizations for specific use cases.

## Key Features

- **🚀 Performance**: SIMD-accelerated operations, specialized small manifold implementations
- **🔧 Zero-copy design**: Matrix manifold trait eliminates vector-matrix conversions
- **📦 Flexible composition**: Both static and dynamic product manifolds
- **🧮 Rich geometry**: Multiple metrics and retractions for each manifold
- **✅ Battle-tested**: Comprehensive test suite with integration and performance tests

## Implemented Manifolds

### Core Manifolds

#### Sphere Manifold `S^{n-1}`
Unit vectors in n-dimensional space.
- **Applications**: Normalized features, directional data, spherical PCA
- **Features**: 
  - Exact exponential/logarithmic maps
  - SIMD acceleration (AVX2/AVX-512)
  - Optimized for dimensions ≥ 4 (f64) or ≥ 8 (f32)
- **Variants**: `Sphere`, `SphereSIMD`

#### Stiefel Manifold `St(n,p)`
Matrices with orthonormal columns (n×p with p≤n).
- **Applications**: PCA, ICA, orthogonal neural networks, eigenvalue problems
- **Features**: 
  - Multiple retractions: QR (default), polar, Cayley
  - Parallel QR decomposition for large matrices
  - Specialized implementations for small manifolds (e.g., St(3,2), St(4,2))
- **Variants**: `Stiefel`, `StiefelMatrix`, `StiefelSmall`

#### Grassmann Manifold `Gr(n,p)`
p-dimensional linear subspaces of R^n.
- **Applications**: Subspace tracking, computer vision, model reduction
- **Features**: 
  - Quotient geometry of Stiefel manifold
  - Principal angles distance
  - Efficient projection via SVD
- **Variants**: `Grassmann`, `GrassmannMatrix`

#### Symmetric Positive Definite `SPD(n)`
n×n symmetric positive definite matrices.
- **Applications**: Covariance matrices, kernel learning, medical imaging, DTI
- **Features**: 
  - Affine-invariant metric (default)
  - Log-Euclidean metric option
  - Cholesky-based operations
- **Variants**: `SPD`, `SPDMatrix`

#### Positive Semi-Definite Cone `PSD(n)` *(New)*
n×n symmetric positive semi-definite matrices.
- **Applications**: Semidefinite programming, quantum state tomography, kernel methods
- **Features**:
  - Efficient projection to PSD cone
  - Support for boundary points (rank-deficient matrices)
  - Multiple metrics: Euclidean, Log-Euclidean, Bures-Wasserstein
- **Special**: Handles matrices with zero eigenvalues

#### Hyperbolic Space `H^n`
Constant negative curvature space.
- **Applications**: Hierarchical embeddings, tree-like data, NLP, social networks
- **Features**: 
  - Poincaré ball model
  - Möbius operations
  - Gyrovector space structure
- **Metrics**: Conformal metric with curvature -1

#### Product Manifold
Cartesian product of manifolds M₁ × M₂ × ... × Mₖ.
- **Applications**: Multi-task learning, structured optimization, mixed constraints
- **Features**: 
  - Component-wise operations
  - Mixed manifold types
  - Dynamic and static variants
- **Variants**: 
  - `ProductManifold`: Runtime manifold composition
  - `ProductManifoldStatic`: Compile-time dispatch (zero-cost)

#### Fixed-Rank Manifold
Matrices with fixed rank r.
- **Applications**: Matrix completion, recommender systems, low-rank approximation
- **Features**: 
  - SVD-based representation
  - Efficient storage (U, S, V factors)
  - Rank-preserving retractions

#### Oblique Manifold
Product of unit spheres (S¹)^n - matrices with unit-norm columns.
- **Applications**: Normalized features, probability simplices, topic models
- **Features**: 
  - Column-wise operations
  - Diagonal metric
  - Efficient parallel normalization
- **Variants**: `Oblique`, `ObliqueMatrix`

### Matrix Manifold Trait

A key innovation in this crate is the `MatrixManifold` trait, which provides zero-copy matrix operations:

```rust
pub trait MatrixManifold<T: Scalar>: Manifold<T, Dyn> {
    fn project_matrix(&self, matrix: &MatrixView<T>, result: &mut Matrix<T>, 
                      workspace: &mut Workspace<T>);
    
    fn retract_matrix(&self, point: &MatrixView<T>, tangent: &MatrixView<T>,
                      result: &mut Matrix<T>, workspace: &mut Workspace<T>) 
                      -> Result<()>;
    // ... other matrix-specific operations
}
```

Benefits:
- Eliminates costly vector-matrix conversions
- More intuitive API for matrix algorithms
- Better cache locality
- Automatic implementation of vector API via macro

## Usage

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
riemannopt-manifolds = "0.1"
```

### Feature Flags

- `default = ["std", "parallel"]`
- `std`: Standard library support
- `parallel`: CPU parallelization with Rayon
- `serde`: Serialization support

### Basic Example

```rust
use riemannopt_manifolds::{Sphere, Stiefel, StiefelMatrix};
use riemannopt_core::{Manifold, MatrixManifold};
use nalgebra::{DVector, DMatrix};

// Create a sphere in R^10
let sphere = Sphere::new(10)?;
let mut workspace = Workspace::with_size(10);

// Generate random point and tangent
let mut point = DVector::zeros(10);
sphere.random_point(&mut point, &mut workspace);

let mut tangent = DVector::zeros(10);
sphere.random_tangent(&point, &mut tangent, &mut workspace)?;

// Use matrix manifold for Stiefel
let stiefel = StiefelMatrix::new(5, 2)?;
let mut matrix = DMatrix::zeros(5, 2);
stiefel.random_matrix(&mut matrix, &mut workspace);
```

### SIMD Acceleration

```rust
use riemannopt_manifolds::SphereSIMD;

// Automatic SIMD acceleration for large vectors
let sphere = SphereSIMD::new(1000)?;

// SIMD operations are used automatically when beneficial
let mut point = DVector::zeros(1000);
sphere.project_point(&point, &mut result, &mut workspace);
```

### Static Product Manifold

```rust
use riemannopt_manifolds::{product, Sphere, Stiefel};

// Compile-time manifold composition for better performance
let sphere = Sphere::new(10)?;
let stiefel = Stiefel::new(5, 2)?;
let product_manifold = product(&sphere, &stiefel);

// Zero-cost abstraction - no virtual function calls
let dim = product_manifold.dimension(); // 10 + 5*2 = 20
```

### Working with Matrix Manifolds

```rust
use riemannopt_manifolds::{SPDMatrix, StiefelMatrix};
use nalgebra::DMatrix;

// SPD manifold with matrix interface
let spd = SPDMatrix::new(3)?;
let mut covariance = DMatrix::identity(3, 3);
let mut workspace = Workspace::with_size(9);

// Direct matrix operations
spd.project_matrix(&covariance.view(), &mut result, &mut workspace);

// Stiefel with efficient QR retraction
let stiefel = StiefelMatrix::new(10, 3)?;
let mut orthogonal = DMatrix::zeros(10, 3);
stiefel.random_matrix(&mut orthogonal, &mut workspace);
```

## Performance

All manifolds are optimized for high performance:

### Optimization Strategies

1. **SIMD Acceleration**: Hardware vectorization for compatible operations
2. **Parallel Operations**: Automatic parallelization for large dimensions
3. **Specialized Implementations**: Hand-optimized code for small manifolds
4. **Zero-Copy Operations**: Matrix manifold trait avoids conversions
5. **Compile-Time Dispatch**: Static product manifolds eliminate virtual calls

### Benchmarks

Performance on Intel i9-12900K:

| Operation | Manifold | Size | Time | Notes |
|-----------|----------|------|------|--------|
| Projection | Sphere | n=1000 | 0.8 μs | SIMD accelerated |
| Retraction | Stiefel | 50×10 | 4.2 ms | Parallel QR |
| Distance | Grassmann | 20×5 | 8.5 ms | Principal angles |
| Logarithm | SPD | 100×100 | 45 ms | Eigen decomposition |
| Retraction | Hyperbolic | n=100 | 1.2 μs | Möbius addition |
| Full step | Product | 2 manifolds | 2x single | No overhead |

### Performance Tips

1. **Use matrix variants**: For matrix manifolds, prefer `StiefelMatrix` over `Stiefel`
2. **Enable parallelization**: Keep the `parallel` feature enabled
3. **Static products**: Use `ProductManifoldStatic` for compile-time composition
4. **Workspace reuse**: Reuse workspace objects across operations
5. **SIMD thresholds**: SIMD benefits kick in at ~4 elements (f64) or ~8 (f32)

## Testing

Comprehensive test suite with multiple levels:

### Run Tests

```bash
# Unit tests (fast, inline with source)
cargo test --lib

# Integration tests (cross-manifold compatibility)
cargo test --test integration_tests

# Performance tests (benchmarks)
cargo test --test performance_tests

# Stress tests (large dimensions, ignored by default)
cargo test --test performance_tests -- --ignored

# All tests
cargo test
```

### Test Categories

1. **Unit Tests**: Basic operations, mathematical properties
2. **Integration Tests**: Cross-manifold compatibility, edge cases
3. **Performance Tests**: Benchmarks, scaling behavior
4. **Property Tests**: Mathematical invariants, numerical stability

All tests are self-contained without external dependencies.

## Mathematical References

Each manifold implementation includes detailed mathematical documentation:

- Geometric definitions and properties
- Optimization applications
- Implementation complexity
- Numerical considerations
- Literature references

See the [API documentation](https://docs.rs/riemannopt-manifolds) for mathematical details.

## Examples

The crate includes example programs demonstrating:

- SIMD usage patterns (`examples/simd_usage.rs`)
- Matrix manifold operations (`examples/matrix_manifold_demo.rs`)

## Contributing

Contributions are welcome! Areas of interest:

- Additional manifold implementations
- Further SIMD optimizations
- GPU acceleration support
- More specialized small manifold variants

See our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](../../LICENSE) for details.