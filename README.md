# RiemannOpt

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

**Note: This library is currently in beta and under active development. The API may change and not all planned features are fully implemented.**

RiemannOpt is a Riemannian optimization library written in Rust with Python bindings. It provides algorithms for optimization on smooth manifolds, addressing problems where the optimization domain is a curved geometric space rather than flat Euclidean space.

## What is Riemannian Optimization?

Many computational problems involve optimization over spaces with geometric constraints. Rather than treating these constraints as obstacles to work around, Riemannian optimization exploits the underlying geometric structure of these spaces. The library implements optimization algorithms that move naturally along the curved surface of these manifolds, respecting their intrinsic geometry.

### Problem Domains

Riemannian optimization is relevant for problems involving:

- **Orthogonality constraints**: Matrices whose columns must remain orthonormal, appearing in principal component analysis, dimensionality reduction, and neural network architectures
- **Rotation and orientation**: 3D rotations in computer vision, robotics, and graphics applications
- **Positive definiteness**: Covariance matrices and metric learning where matrices must remain symmetric and positive definite
- **Low-rank structures**: Matrix factorization and completion problems with fixed-rank constraints
- **Hyperbolic geometry**: Hierarchical data representation and graph embeddings that naturally live in hyperbolic space
- **Unit norm constraints**: Directions and normalized representations in various machine learning contexts

Traditional optimization approaches handle these constraints through projections or penalty methods, which can be computationally expensive or numerically unstable. Riemannian methods work directly on the constraint manifold.

## Architecture

The library follows a modular design with separate components:

- **riemannopt-core**: Foundational traits defining manifolds, cost functions, and optimization interfaces
- **riemannopt-manifolds**: Concrete implementations of geometric spaces (sphere, Stiefel, Grassmann, SPD, hyperbolic, and others)
- **riemannopt-optim**: Optimization algorithms adapted to Riemannian geometry (gradient descent, Adam, L-BFGS, trust region, conjugate gradient, Newton methods)
- **riemannopt-autodiff**: Automatic differentiation capabilities (in development)
- **riemannopt-py**: Python bindings enabling use from Python with NumPy integration

### Technical Approach

The implementation emphasizes:

- **Type safety**: Rust's type system ensures geometric constraints are respected at compile time
- **Memory efficiency**: Workspace-based memory management minimizes allocations in optimization loops
- **Performance**: SIMD vectorization and parallel computing for batch operations where beneficial
- **Correctness**: Multiple retraction methods (exponential map, QR-based, polar decomposition) with different accuracy/performance tradeoffs
- **Numerical stability**: Built-in validation and stability checking throughout geometric computations

### Implemented Manifolds

Current manifold implementations include:

- **Sphere**: Unit sphere S^(n-1) in n-dimensional space
- **Stiefel**: Manifold of orthonormal n×p matrices
- **Grassmann**: Space of p-dimensional subspaces in n-dimensional space
- **SPD**: Symmetric positive definite matrices
- **Hyperbolic**: Hyperbolic space with configurable curvature
- **Oblique**: Product of unit spheres
- **Product**: Cartesian products of manifolds for composite constraint structures
- **Fixed-rank**: Matrices constrained to specific rank
- **PSD Cone**: Positive semidefinite matrices

### Optimization Algorithms

Available optimizers include:

- **Riemannian Gradient Descent**: First-order method with momentum variants
- **Riemannian Adam**: Adaptive learning rate method
- **L-BFGS**: Limited-memory quasi-Newton method
- **Trust Region**: Second-order method with adaptive step sizing
- **Conjugate Gradient**: Various conjugate gradient variants
- **Riemannian Newton**: Newton method with conjugate gradient solver
- **Natural Gradient**: Fisher information-based optimization

Each optimizer supports configurable line search strategies, step size scheduling, and callback mechanisms for monitoring convergence.

## License

RiemannOpt is licensed under the MIT License.

## Contact

- Author: Denis Chaput
- Email: denis.chaput@pm.me
- GitHub: [@DenisChpt](https://github.com/DenisChpt)

For bug reports and feature requests, please use [GitHub Issues](https://github.com/DenisChpt/RiemannOpt/issues).