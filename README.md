# RiemannOpt

[![Crates.io](https://img.shields.io/crates/v/riemannopt)](https://crates.io/crates/riemannopt)
[![Documentation](https://docs.rs/riemannopt/badge.svg)](https://docs.rs/riemannopt)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://github.com/DenisChpt/RiemannOpt/workflows/CI/badge.svg)](https://github.com/DenisChpt/RiemannOpt/actions)

**RiemannOpt** is a Riemannian optimization library written in Rust with first-class Python bindings. It provides manifold implementations, solvers, and a reverse-mode automatic differentiation engine — all designed around a zero-allocation hot path.

## Overview

### Manifolds

| Manifold | Description |
|----------|-------------|
| **Euclidean** | Standard R^n |
| **Sphere** | Unit sphere S^{n-1} |
| **Stiefel** | Orthonormal matrices St(n, p) |
| **Grassmann** | Subspaces Gr(n, p) |
| **SPD** | Symmetric positive-definite matrices (affine-invariant, log-Euclidean metrics) |
| **Hyperbolic** | Poincaré ball model with configurable curvature |
| **Oblique** | Product of unit spheres |
| **FixedRank** | Fixed-rank matrices |
| **PSD Cone** | Positive semi-definite cone |
| **Product** | Cartesian product of any two manifolds |

Each manifold implements: projection onto the tangent space, Riemannian metric, exponential map (retraction), logarithmic map, geodesic distance, and parallel transport. All operations write into pre-allocated buffers via the `Workspace` pattern — no heap allocation during optimization.

### Solvers

| Solver | Type |
|--------|------|
| **SGD** | First-order (momentum, Nesterov) |
| **Adam** | Adaptive first-order |
| **Conjugate Gradient** | First-order |
| **L-BFGS** | Quasi-Newton |
| **Trust Region** | Second-order |
| **Newton** | Second-order (truncated CG, requires Hessian-vector products) |
| **Natural Gradient** | Fisher information-based (full, diagonal, identity, empirical) |

Solvers expose configurable stopping criteria (max iterations, wall-clock time, gradient tolerance, target cost value) and return structured results with termination reason and evaluation counts.

### Automatic Differentiation

A reverse-mode AD engine (`riemannopt-autodiff`) records a flat instruction tape during the first forward pass, then replays it for all subsequent iterations with zero allocation. Supported operations:

- **Scalars**: arithmetic, `exp`, `log`, `sqrt`, `sin`, `cos`, `abs`, `pow`
- **Vectors**: arithmetic, component-wise multiply, dot product, norms
- **Matrices**: arithmetic, matmul, matvec, transpose, trace, Frobenius inner product
- **Fused**: linear layers, quadratic forms

Forward-over-reverse composition provides Hessian-vector products via dual numbers.

### Linear Algebra Backends

Two compile-time backends, selectable via Cargo features:

- **faer** (default) — pure Rust, SIMD-optimized
- **nalgebra** — drop-in alternative

All manifold and solver code is generic over `LinAlgBackend<T>`, so switching backends requires no code changes.

### Pre-built Problems

22 ready-to-use optimization problems:

| Domain | Problems |
|--------|----------|
| Sphere | Rayleigh quotient, MaxCut relaxation, spherical k-means |
| Stiefel | Ordered Brockett, orthogonal Procrustes, orthogonal ICA |
| Grassmann | Brockett cost, robust PCA |
| SPD | Fréchet mean, Gaussian mixture covariance, metric learning |
| Hyperbolic | Poincaré embedding, hyperbolic logistic regression |
| Oblique | Dictionary learning, ICA, phase retrieval |
| Fixed Rank | Matrix completion, matrix sensing |
| PSD Cone | MaxCut SDP, nearest correlation |
| Product | Coupled factorization, pose estimation |
| Euclidean | Rosenbrock, Rastrigin, ridge regression, logistic regression |

## Architecture

```
crates/
├── riemannopt-core      # Manifolds, solvers, linalg backends, problem trait
├── riemannopt-autodiff  # Reverse-mode AD (tape, buffer pool, VJP)
├── riemannopt-py        # Python bindings (PyO3 + rust-numpy)
└── riemannopt           # Facade crate (re-exports core + optional autodiff)
```

## License

MIT — see [LICENSE](LICENSE).

## Citation

```bibtex
@software{riemannopt2025,
  author = {Chaput, Denis},
  title = {RiemannOpt: Riemannian Optimization in Rust},
  year = {2025},
  url = {https://github.com/DenisChpt/RiemannOpt}
}
```
