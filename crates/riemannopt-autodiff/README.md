# riemannopt-autodiff

Automatic differentiation module for [RiemannOpt](https://github.com/DenisChpt/RiemannOpt).

Provides tape-based reverse-mode AD (Wengert list) for computing gradients in
Riemannian optimization, with zero graph-traversal overhead.

## Features

- **`Tape`** — flat computation tape recording operations in topological order
- **`Var`** — differentiable variable wrapping scalars and matrices
- **`backward()`** — reverse-pass gradient computation
- **`AutoDiffCostFunction`** / **`AutoDiffMatCostFunction`** — adapters to plug AD-defined objectives into RiemannOpt optimizers

## License

See the [workspace license](../../LICENSE).
