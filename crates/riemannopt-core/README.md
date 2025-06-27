# RiemannOpt Core

[![Crates.io](https://img.shields.io/crates/v/riemannopt-core.svg)](https://crates.io/crates/riemannopt-core)
[![Documentation](https://docs.rs/riemannopt-core/badge.svg)](https://docs.rs/riemannopt-core)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance core traits and types for Riemannian optimization in Rust.

## Overview

`riemannopt-core` provides the foundational abstractions for implementing Riemannian optimization algorithms with a focus on performance and zero-allocation design. It features a sophisticated memory management system, flexible compute backends, and comprehensive support for manifold operations.

## Key Features

### 🚀 Performance
- **Zero-allocation design**: Pre-allocated workspaces eliminate allocations in hot paths
- **SIMD acceleration**: Hardware-optimized operations via the `wide` crate
- **Intelligent parallelization**: Automatic CPU parallelization with configurable thresholds
- **Memory pooling**: Efficient buffer management for dynamic allocations
- **Backend flexibility**: Runtime selection between CPU, SIMD, and GPU backends

### 🧮 Mathematical Framework
- **Manifold trait**: Complete abstraction for Riemannian manifolds
- **Optimizer framework**: Flexible trait-based optimization algorithms
- **Cost functions**: Efficient interface for objective functions with caching
- **Retractions**: Multiple retraction methods (exponential map, QR, polar)
- **Vector transport**: Parallel transport and vector transport implementations
- **Fisher information**: Various approximation strategies for natural gradient methods

### 💾 Memory Management
- **Workspace system**: Pre-allocated buffers with named access
- **Memory pools**: Dynamic allocation pools for vectors and matrices
- **Builder pattern**: Configurable workspace construction
- **Buffer categories**: Organized buffers (Gradient, Direction, Momentum, etc.)

### 🖥️ Compute Backends
- **Backend abstraction**: Unified interface for different compute targets
- **Auto-selection**: Intelligent backend selection based on problem size
- **SIMD backend**: Explicit SIMD operations for f32x8 and f64x4
- **GPU support** (optional): CUDA acceleration for large-scale problems
- **Batch operations**: Optimized batch processing capabilities

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
riemannopt-core = "0.1"
```

### Feature Flags

- `std` (default): Standard library support
- `serde`: Serialization/deserialization support
- `parallel` (default): CPU parallel computing via Rayon
- `cuda`: GPU acceleration (requires CUDA toolkit)
- `test-utils`: Testing utilities
- `profiling`: Performance profiling support

## Usage

### Basic Example

```rust
use riemannopt_core::prelude::*;
use nalgebra::DVector;

// Create a workspace for memory management
let n = 100;
let mut workspace = Workspace::with_size(n);

// Define your manifold (e.g., using a provided implementation)
let manifold = Sphere::new(n)?;

// Generate a random point on the manifold
let mut point = DVector::zeros(n);
manifold.random_point(&mut point, &mut workspace);

// Generate a random tangent vector
let mut tangent = DVector::zeros(n);
manifold.random_tangent(&point, &mut tangent, &mut workspace)?;

// Perform retraction
let mut new_point = DVector::zeros(n);
manifold.retract(&point, &tangent, &mut new_point, &mut workspace)?;
```

### Using the Workspace System

```rust
use riemannopt_core::memory::{Workspace, WorkspaceBuilder, BufferId};

// Create workspace with standard optimization buffers
let workspace = WorkspaceBuilder::new()
    .with_standard_buffers(n)      // Gradient, Direction, etc.
    .with_momentum_buffers(n)      // For momentum-based methods
    .with_hessian_buffers(n)       // For second-order methods
    .build();

// Access pre-allocated buffers
let gradient = workspace.get_or_create_vector(BufferId::Gradient, n);
let direction = workspace.get_or_create_vector(BufferId::Direction, n);
```

### Backend Selection

```rust
use riemannopt_core::compute::{BackendSelector, BackendSelection};

// Create backend selector with automatic selection
let mut selector = BackendSelector::new(BackendSelection::Auto);

// Or use fixed backend
let mut selector = BackendSelector::new(BackendSelection::Fixed(Backend::Simd));

// Or use adaptive selection based on problem size
let mut selector = BackendSelector::new(BackendSelection::Adaptive {
    cpu_threshold: 1000,
    gpu_threshold: 10000,
});

// Select backend for given dimension
let backend = selector.select_backend(dimension);

// Use backend for operations
let result = backend.dot(&a, &b)?;
```

### Complete Optimization Example

```rust
use riemannopt_core::prelude::*;
use riemannopt_core::optimization::{SGD, SGDConfig, StoppingCriterion};

// 1. Define your cost function
struct MyObjective { /* ... */ }

impl<T: Scalar, D: Dim> CostFunction<T, D> for MyObjective {
    fn cost_and_gradient(
        &self, 
        point: &Point<T, D>, 
        workspace: &mut Workspace<T>,
        gradient: &mut TangentVector<T, D>
    ) -> Result<T> {
        // Compute cost and gradient
        // ...
    }
}

// 2. Configure optimizer
let mut optimizer = SGD::new(
    SGDConfig::new()
        .with_constant_step_size(0.1)
        .with_classical_momentum(0.9)
);

// 3. Set stopping criteria
let stopping_criterion = StoppingCriterion::new()
    .with_max_iterations(1000)
    .with_gradient_tolerance(1e-6)
    .with_cost_tolerance(1e-8);

// 4. Run optimization
let manifold = Stiefel::new(n, p)?;
let cost_fn = MyObjective::new();
let initial_point = manifold.random_point();

let result = optimizer.optimize(
    &cost_fn,
    &manifold,
    &initial_point,
    &stopping_criterion
)?;

println!("Converged: {}", result.converged);
println!("Final cost: {}", result.value);
println!("Iterations: {}", result.iterations);
```

## Core Traits

### Manifold

The central trait for Riemannian manifolds:

```rust
pub trait Manifold<T: Scalar, D: Dim> {
    // Manifold properties
    fn name(&self) -> &str;
    fn dimension(&self) -> usize;
    
    // Point operations (non-allocating)
    fn project_point(&self, point: &Point<T, D>, result: &mut Point<T, D>, 
                     workspace: &mut Workspace<T>);
    
    // Tangent space operations
    fn project_tangent(&self, point: &Point<T, D>, vector: &TangentVector<T, D>,
                       result: &mut TangentVector<T, D>, workspace: &mut Workspace<T>) 
                       -> Result<()>;
    
    // Riemannian metric
    fn inner_product(&self, point: &Point<T, D>, u: &TangentVector<T, D>, 
                     v: &TangentVector<T, D>, workspace: &mut Workspace<T>) 
                     -> Result<T>;
    
    // Retraction and transport
    fn retract(&self, point: &Point<T, D>, tangent: &TangentVector<T, D>,
               result: &mut Point<T, D>, workspace: &mut Workspace<T>) 
               -> Result<()>;
               
    fn vector_transport(&self, from: &Point<T, D>, to: &Point<T, D>, 
                        vector: &TangentVector<T, D>, result: &mut TangentVector<T, D>,
                        workspace: &mut Workspace<T>) -> Result<()>;
}
```

### MatrixManifold

Specialized trait for matrix-based manifolds with efficient operations:

```rust
pub trait MatrixManifold<T: Scalar>: Manifold<T, Dyn> {
    fn ambient_dimension(&self) -> (usize, usize);
    
    // Matrix-specific operations
    fn project_matrix(&self, matrix: &MatrixView<T>, result: &mut Matrix<T>, 
                      workspace: &mut Workspace<T>);
                      
    fn retract_matrix(&self, point: &MatrixView<T>, tangent: &MatrixView<T>,
                      result: &mut Matrix<T>, workspace: &mut Workspace<T>) 
                      -> Result<()>;
}
```

### CostFunction

Efficient interface for objective functions:

```rust
pub trait CostFunction<T: Scalar, D: Dim> {
    // Non-allocating version (preferred)
    fn cost_and_gradient(&self, point: &Point<T, D>, workspace: &mut Workspace<T>,
                         gradient: &mut TangentVector<T, D>) -> Result<T>;
    
    // Cost-only evaluation
    fn cost(&self, point: &Point<T, D>, workspace: &mut Workspace<T>) -> Result<T>;
    
    // Optional: Hessian-vector products for second-order methods
    fn hessian_vector_product(&self, point: &Point<T, D>, vector: &TangentVector<T, D>,
                              workspace: &mut Workspace<T>, result: &mut TangentVector<T, D>) 
                              -> Result<()> {
        Err(OptimizerError::NotImplemented("Hessian-vector product"))
    }
}
```

### ComputeBackend

Abstraction for different compute targets:

```rust
pub trait ComputeBackend<T: Scalar>: Send + Sync {
    fn name(&self) -> &str;
    fn backend_type(&self) -> Backend;
    
    // Vector operations
    fn dot(&self, a: &[T], b: &[T]) -> Result<T>;
    fn norm(&self, x: &[T]) -> Result<T>;
    fn normalize(&self, x: &mut [T]) -> Result<()>;
    
    // Matrix operations
    fn matrix_multiply(&self, a: &MatrixView<T>, b: &MatrixView<T>, 
                       c: &mut Matrix<T>) -> Result<()>;
                       
    // Batch operations
    fn batch_dot_products(&self, vectors: &[&[T]], results: &mut [T]) -> Result<()>;
}
```

## Advanced Features

### Fisher Information Approximations

Support for natural gradient methods with various approximations:

```rust
use riemannopt_core::fisher::{FisherApproximation, FisherConfig};

let fisher_approx = FisherApproximation::KFAC {
    damping: 1e-3,
    update_frequency: 10,
};

// Use in natural gradient optimizer
let config = NaturalGradientConfig::new()
    .with_fisher_approximation(fisher_approx);
```

### SIMD Operations

Automatic SIMD acceleration for supported operations:

```rust
use riemannopt_core::simd::{SimdOps, SimdDispatcher};

// Automatic SIMD dispatch
let dispatcher = SimdDispatcher::new();
let result = dispatcher.dot_product::<f64>(&x, &y);

// Or use specific SIMD backend
use riemannopt_core::simd::wide_backend::WideBackend;
let backend = WideBackend;
let result = backend.dot_f64x4(&x, &y);
```

### Parallel Thresholds

Intelligent parallelization based on problem size:

```rust
use riemannopt_core::parallel::{ParallelThresholds, ParallelDecision};

// Configure custom thresholds
let thresholds = ParallelThresholds::builder()
    .vector_threshold(5000)
    .matrix_matrix_threshold(64)
    .eigenvalue_threshold(100)
    .build();

// Check if operation should be parallelized
if ParallelDecision::should_parallelize_dot_product(vector_size) {
    // Use parallel implementation
}
```

## Performance Tips

1. **Use workspaces**: Always pass workspaces to avoid allocations
2. **Batch operations**: Use batch methods when processing multiple items
3. **Backend selection**: Let auto-selection choose optimal backend
4. **Parallel thresholds**: Tune thresholds for your specific hardware
5. **SIMD alignment**: Ensure data is properly aligned for SIMD operations

## Error Handling

Comprehensive error types for different failure modes:

```rust
use riemannopt_core::error::{ManifoldError, OptimizerError};

match manifold.retract(&point, &tangent, &mut result, &mut workspace) {
    Ok(()) => { /* success */ }
    Err(ManifoldError::InvalidTangent) => { /* handle invalid tangent */ }
    Err(ManifoldError::NumericalInstability) => { /* handle instability */ }
    Err(e) => { /* handle other errors */ }
}
```

## Dependencies

- `nalgebra`: Linear algebra operations
- `num-traits`: Generic numeric traits
- `rayon`: Data parallelism (optional)
- `wide`: SIMD operations
- `thiserror`: Error handling
- `rand`: Random number generation

## License

MIT License. See [LICENSE](../../LICENSE) for details.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

Areas of interest:
- Additional backend implementations
- Performance optimizations
- New manifold traits
- Documentation improvements