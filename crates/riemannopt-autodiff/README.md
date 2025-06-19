# RiemannOpt Automatic Differentiation

[![Crates.io](https://img.shields.io/crates/v/riemannopt-autodiff.svg)](https://crates.io/crates/riemannopt-autodiff)
[![Documentation](https://docs.rs/riemannopt-autodiff/badge.svg)](https://docs.rs/riemannopt-autodiff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Automatic differentiation for Riemannian optimization.

## Overview

This crate provides a reverse-mode automatic differentiation engine specifically designed for Riemannian manifolds. It enables gradient computation through complex manifold operations while respecting geometric constraints.

## Features

- **Dynamic computation graphs**: Build models with runtime graph construction
- **Reverse-mode AD**: Efficient gradient computation via backpropagation  
- **Manifold-aware**: Proper handling of manifold constraints and projections
- **Broadcasting**: NumPy-style tensor broadcasting for flexible operations
- **Graph optimization**: Automatic operation fusion and dead code elimination

## Usage

### Basic Example

```rust
use riemannopt_autodiff::prelude::*;
use nalgebra::DMatrix;

// Create a computation graph
let graph = Graph::new();

// Create variables
let x = graph.variable(DMatrix::from_element(2, 2, 1.0));
let y = graph.variable(DMatrix::from_element(2, 2, 2.0));

// Build computation: z = x * y + x
let xy = graph.apply_op(Box::new(Multiply), &[x.id, y.id]);
let z = graph.apply_op(Box::new(Add), &[xy, x.id]);

// Forward pass
let result = graph.forward(z).unwrap();

// Backward pass
let gradients = backward(&graph, z, None);
```

### Manifold Operations

```rust
use riemannopt_autodiff::prelude::*;
use nalgebra::DMatrix;

// Create a manifold function
let graph = Graph::new();
let mut func = ManifoldFunction::new(graph);

// Add input on sphere manifold
let x = func.add_input(
    DMatrix::from_row_slice(3, 1, &[0.6, 0.8, 0.0]),
    "sphere",
);

// Define objective function
let squared = func.graph.apply_op(
    Box::new(Multiply),
    &[x, x],
);
let output = func.graph.apply_op(
    Box::new(Sum::all()),
    &[squared],
);
func.set_output(output);

// Compute value and Riemannian gradients
let mut inputs = HashMap::new();
inputs.insert(x, DMatrix::from_row_slice(3, 1, &[0.6, 0.8, 0.0]));
let (value, grads) = func.value_and_grad(&inputs);
```

## Architecture

The autodiff engine consists of several key components:

1. **Graph Module** (`graph.rs`): Manages the computation graph structure
2. **Operations** (`ops.rs`): Defines forward and backward operations
3. **Backward Pass** (`backward.rs`): Implements backpropagation algorithm
4. **Manifold Operations** (`manifold_ops.rs`): Manifold-specific operations
5. **Broadcasting** (`broadcast.rs`): Tensor broadcasting support
6. **Integration** (`integration.rs`): Bridges autodiff with manifold constraints

## Supported Operations

### Basic Operations
- Addition, Multiplication, Matrix multiplication
- Transpose, Negation
- Sum, Mean (with axis support)
- Power, Exponential, Logarithm
- Activation functions: ReLU, Sigmoid, Tanh

### Manifold Operations
- Tangent space projection
- Sphere projection
- Stiefel projection
- Manifold inner product
- Exponential and logarithmic maps

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
riemannopt-autodiff = "0.1"
```

## Performance

The autodiff engine is designed for efficiency:

- **Tape-based AD**: Minimal overhead for gradient computation
- **Operation fusion**: Automatic merging of compatible operations
- **Memory pooling**: Reuse of temporary buffers
- **Sparse gradients**: Support for sparse gradient accumulation

## Integration with Optimizers

The autodiff engine integrates seamlessly with RiemannOpt optimizers:

```rust
use riemannopt_autodiff::prelude::*;
use riemannopt_optim::SGD;

// Create autodiff-enabled cost function
let cost_fn = AutodiffCostFunction::new(|x| {
    // Define objective using autodiff operations
    let squared = x * x;
    sum(squared)
});

// Use with any optimizer
let optimizer = SGD::default();
let result = optimizer.optimize(&cost_fn, &manifold, &x0, &stopping)?;
```

## Limitations

- Currently only supports first-order derivatives
- No support for sparse tensors (coming soon)
- Limited to operations defined in the operation library

## Contributing

Contributions are welcome! Areas of interest:
- Higher-order derivatives
- More manifold operations
- Performance optimizations
- GPU support

## License

MIT License. See [LICENSE](../../LICENSE) for details.