# RiemannOpt Optim

[![Crates.io](https://img.shields.io/crates/v/riemannopt-optim.svg)](https://crates.io/crates/riemannopt-optim)
[![Documentation](https://docs.rs/riemannopt-optim/badge.svg)](https://docs.rs/riemannopt-optim)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

State-of-the-art optimization algorithms for Riemannian manifolds.

## Overview

This crate provides a comprehensive collection of optimization algorithms specifically designed for Riemannian manifolds. All optimizers handle the non-Euclidean geometry through proper use of retractions, vector transport, and the Riemannian metric.

## Implemented Optimizers

### First-Order Methods

#### Stochastic Gradient Descent (SGD)
- **Variants**: Vanilla, momentum, Nesterov acceleration
- **Features**: Step size scheduling, gradient clipping, line search
- **Best for**: Large-scale problems, online learning

#### Adaptive Moment Estimation (Adam)
- **Variants**: Standard Adam, AMSGrad, AdamW
- **Features**: Adaptive learning rates, bias correction
- **Best for**: Non-stationary objectives, deep learning

#### Conjugate Gradient (CG)
- **Variants**: Fletcher-Reeves, Polak-Ribière, Hestenes-Stiefel
- **Features**: Memory efficient, no hyperparameter tuning
- **Best for**: Quadratic-like objectives, large problems

### Second-Order Methods

#### Limited-Memory BFGS (L-BFGS)
- **Features**: Quasi-Newton approximation, configurable memory
- **Best for**: Smooth problems, fast convergence needed

#### Trust Region
- **Features**: Global convergence, robust to non-convexity
- **Best for**: Ill-conditioned problems, high accuracy needed

#### Natural Gradient
- **Features**: Information geometry, optimal preconditioning
- **Best for**: Statistical models, when Fisher information is available

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

### Benchmarks

Run benchmarks:
```bash
cargo bench
```

### All Tests

Run all tests:
```bash
cargo test
```

## Test Structure

```
tests/
├── integration_tests.rs    # Cross-optimizer and cross-manifold tests
└── performance_tests.rs    # Performance benchmarks and scaling tests

benches/
└── optimizer_comparison.rs # Criterion benchmarks for optimizers

src/
├── sgd.rs                 # SGD with inline tests
├── adam.rs                # Adam with inline tests
├── lbfgs.rs               # L-BFGS with inline tests
├── trust_region.rs        # Trust region with inline tests
├── conjugate_gradient.rs  # CG methods with inline tests
├── natural_gradient.rs    # Natural gradient with inline tests
└── parallel_sgd.rs        # Parallel utilities
```

## Test Independence

This crate can be tested independently without any external dependencies:
- No file I/O in tests
- No dependencies on example files
- All test data generated programmatically
- Only depends on `riemannopt-core` and `riemannopt-manifolds`

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
riemannopt-optim = "0.1"
```

## Quick Start

```rust
use riemannopt_optim::{SGD, SGDConfig};
use riemannopt_core::optimizer::StoppingCriterion;

// Configure optimizer
let mut optimizer = SGD::new(
    SGDConfig::new()
        .with_step_size(StepSizeSchedule::Constant(0.01))
        .with_momentum(MomentumMethod::Nesterov { coefficient: 0.9 })
);

// Set stopping criteria
let stopping = StoppingCriterion::new()
    .with_max_iterations(1000)
    .with_gradient_tolerance(1e-6)
    .with_cost_tolerance(1e-9);

// Optimize
let result = optimizer.optimize(&cost_fn, &manifold, &x0, &stopping)?;
```

## Performance

Optimizers are designed for efficiency:

- **Parallel batch processing**: Automatic parallelization for batch operations
- **Memory reuse**: Minimal allocations per iteration
- **Efficient vector transport**: Cached transport operations
- **Smart step size selection**: Adaptive algorithms reduce line search

### Benchmarks

| Optimizer | Dimension | Time/Iteration |
|-----------|-----------|----------------|
| SGD | 100 | < 50 μs |
| Adam | 100 | < 100 μs |
| L-BFGS | 100 | < 500 μs |
| Trust Region | 100 | < 2 ms |

## Usage Examples

### Basic SGD
```rust
use riemannopt_optim::{SGD, SGDConfig};
use riemannopt_core::optimizer::StoppingCriterion;

let mut sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
let stopping_criterion = StoppingCriterion::new()
    .with_max_iterations(1000)
    .with_gradient_tolerance(1e-6);

let result = sgd.optimize(&cost_fn, &manifold, &x0, &stopping_criterion)?;
```

### Adam with Custom Parameters
```rust
use riemannopt_optim::{Adam, AdamConfig};

let adam = Adam::new(
    AdamConfig::new()
        .with_learning_rate(0.001)
        .with_beta1(0.9)
        .with_beta2(0.999)
        .with_amsgrad()
);
```

### L-BFGS for Second-Order Optimization
```rust
use riemannopt_optim::{LBFGS, LBFGSConfig};

let lbfgs = LBFGS::new(
    LBFGSConfig::new()
        .with_memory_size(10)
        .with_line_search_max_iter(20)
);
```

## Optimizer Selection Guide

| Optimizer | When to Use | Pros | Cons |
|-----------|-------------|------|------|
| **SGD** | Large-scale, online learning | Simple, low memory, proven | Requires tuning, slower convergence |
| **Adam** | Non-stationary, noisy gradients | Adaptive, less tuning | Can diverge, more memory |
| **L-BFGS** | Smooth, medium-scale | Fast convergence | High memory, not for stochastic |
| **Trust Region** | Non-convex, high accuracy | Robust, global convergence | Expensive per iteration |
| **CG** | Quadratic-like, large-scale | No tuning, low memory | Sensitive to non-quadratic |
| **Natural Gradient** | Statistical models | Theoretically optimal | Very expensive |

## Feature Flags

- `parallel` (default): Enable parallel batch processing
- `serde`: Serialization support for optimizer state

## Contributing

Contributions are welcome! Please see our [Contributing Guide](../../CONTRIBUTING.md) for details.

## License

MIT License. See [LICENSE](../../LICENSE) for details.