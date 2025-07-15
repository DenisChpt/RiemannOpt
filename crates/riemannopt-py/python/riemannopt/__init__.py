"""RiemannOpt: High-performance Riemannian optimization in Python.

This package provides state-of-the-art algorithms for optimization on Riemannian manifolds,
delivering 10-100x performance improvements over pure Python implementations through
Rust-powered computations while maintaining a simple, Pythonic API.

Key Features
============
- **Multiple Manifolds**: Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, PSD Cone
- **Advanced Optimizers**: SGD, Adam, L-BFGS, Conjugate Gradient, Trust Region, Newton
- **High Performance**: Rust backend with Python bindings for optimal speed
- **Product Manifolds**: Optimize on Cartesian products of manifolds with tuple API
- **Automatic Gradients**: Finite difference approximation when gradients not provided
- **Type Safety**: Complete type hints for better IDE support and error catching
- **Comprehensive Testing**: Property-based tests ensure mathematical correctness

Quick Start
===========
>>> import riemannopt as ro
>>> import numpy as np

# Optimization on the unit sphere
>>> sphere = ro.manifolds.Sphere(10)
>>> def cost(x):
...     return 0.5 * np.sum(x**2)  # Minimize distance to origin
>>> 
>>> x0 = sphere.random_point()
>>> result = ro.optimize(sphere, cost, x0, optimizer='Adam', learning_rate=0.01)
>>> print(f"Converged: {result.converged}, Final cost: {result.cost:.6f}")

# Principal Component Analysis on Stiefel manifold
>>> n, p = 50, 5
>>> data = np.random.randn(n, 100)  # n features, 100 samples
>>> C = data @ data.T / 100  # Covariance matrix
>>> 
>>> stiefel = ro.manifolds.Stiefel(n, p)
>>> def pca_cost(X):
...     return -np.trace(X.T @ C @ X)  # Maximize trace (minimize negative)
>>> 
>>> X0 = stiefel.random_point()
>>> result = ro.optimize(stiefel, pca_cost, X0, optimizer='LBFGS')
>>> principal_components = result.point

# Geometric mean on SPD manifolds
>>> spd = ro.manifolds.SPD(5)
>>> matrices = [spd.random_point() for _ in range(10)]
>>> 
>>> def geometric_mean_cost(X):
...     total = 0.0
...     for A in matrices:
...         eigenvals = np.linalg.eigvalsh(np.linalg.inv(X) @ A)
...         total += 0.5 * np.sum(np.log(eigenvals)**2)
...     return total / len(matrices)
>>> 
>>> X0 = spd.random_point()
>>> result = ro.optimize(spd, geometric_mean_cost, X0, optimizer='Adam')
>>> geom_mean = result.point

# Product manifold optimization (Sphere × SPD)
>>> sphere = ro.manifolds.Sphere(3)
>>> spd = ro.manifolds.SPD(2)
>>> product = ro.manifolds.ProductManifold([sphere, spd])
>>> 
>>> def joint_cost(x_tuple):
...     x_sphere, x_spd = x_tuple
...     cost1 = np.sum(x_sphere**2)  # Sphere component
...     cost2 = np.trace(x_spd)      # SPD component
...     return cost1 + cost2
>>> 
>>> x0 = product.random_point()  # Returns tuple (sphere_point, spd_point)
>>> result = ro.optimize(product, joint_cost, x0)

Advanced Usage
==============
# Custom cost function with explicit gradient
>>> def cost_with_gradient(x):
...     cost = 0.5 * np.sum(x**2)
...     gradient = x
...     return cost, gradient
>>> 
>>> cost_fn = ro.create_cost_function(cost_with_gradient)
>>> optimizer = ro.optimizers.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999)
>>> result = optimizer.optimize(cost_fn, sphere, x0, max_iterations=1000)

# Progress monitoring with callbacks
>>> def progress_callback(iteration, point, cost, grad_norm):
...     if iteration % 100 == 0:
...         print(f"Iteration {iteration}: cost = {cost:.6f}, grad_norm = {grad_norm:.3e}")
...     return True  # Continue optimization
>>> 
>>> result = ro.optimize(sphere, cost, x0, callback=progress_callback)

# Gradient checking for validation
>>> is_accurate, max_error = ro.gradient_check(cost_fn, x0)
>>> print(f"Gradient accurate: {is_accurate}, Max error: {max_error:.2e}")

Available Manifolds
===================
- **Sphere(n)**: Unit sphere S^(n-1) in R^n
- **Stiefel(n, p)**: Stiefel manifold St(n,p) of orthonormal n×p matrices
- **Grassmann(n, p)**: Grassmann manifold Gr(n,p) of p-dimensional subspaces
- **SPD(n)**: Symmetric Positive Definite matrices of size n×n
- **Hyperbolic(n)**: Hyperbolic space H^n (Lorentz model)
- **Oblique(m, n)**: Oblique manifold of unit-norm columns
- **PSDCone(n)**: Positive Semidefinite cone
- **ProductManifold([M1, M2, ...])**: Cartesian product of manifolds

Available Optimizers
====================
- **SGD**: Stochastic Gradient Descent with momentum and Nesterov acceleration
- **Adam**: Adaptive moment estimation with bias correction
- **LBFGS**: Limited-memory BFGS with line search
- **ConjugateGradient**: Nonlinear conjugate gradient methods
- **TrustRegion**: Trust region methods with subproblem solvers
- **Newton**: Riemannian Newton method with CG solver

Performance Notes
=================
For best performance:
1. Provide explicit gradients when possible (avoid finite differences)
2. Use compiled optimizers (Adam, LBFGS) for larger problems
3. Enable cost function caching for expensive evaluations
4. Use product manifolds for block-structured problems

References
==========
- Absil, P.-A., Mahony, R., & Sepulchre, R. (2008). Optimization algorithms on matrix manifolds.
- Boumal, N. (2023). An introduction to optimization on smooth manifolds.
- Edelman, A., Arias, T. A., & Smith, S. T. (1998). The geometry of algorithms with orthogonality constraints.
"""

from . import _riemannopt
from ._riemannopt import *

# Re-export key classes and functions from Rust
from ._riemannopt import (
    # Manifolds
    manifolds,
    # Optimizers  
    optimizers,
    # Cost function
    CostFunction,
    create_cost_function,
)

# Import Python convenience functions
from .helpers import (
    optimize,
    gradient_check,
    OptimizationCallback,
    ProgressCallback,
    EarlyStoppingCallback,
    plot_convergence,
    benchmark_optimizers,
)

# Import exceptions
from .exceptions import (
    RiemannOptError,
    ManifoldValidationError,
    OptimizationFailedError,
    ConvergenceError,
    LineSearchError,
    DimensionMismatchError,
    ConfigurationError,
    BackendError,
)

# Import visualization (optional)
# try:
#     from .visualization import (
#         plot_sphere_optimization,
#         plot_stiefel_columns,
#         plot_grassmann_subspace,
#         plot_convergence_comparison,
#         plot_manifold_tangent_space,
#         create_optimization_animation,
#     )
#     _has_visualization = True
# except ImportError:
#     _has_visualization = False

# Import decorators
# from .decorators import (
#     validate_arrays,
#     ensure_on_manifold,
#     handle_rust_exceptions,
#     deprecated,
#     cache_result,
#     time_function,
#     require_gradient,
#     property_cached,
#     vectorize_manifold_operation,
# )

# Import integration features
# try:
#     from .integrations import (
#         TensorConverter,
#         PyTorchAdapter,
#         JAXAdapter,
#         AutogradAdapter,
#         optimize_pytorch_model,
#         optimize_jax_function,
#         ProgressReporter,
#         create_sklearn_compatible_optimizer,
#     )
#     _has_integrations = True
# except ImportError:
#     _has_integrations = False

# Import callbacks
# from .callbacks import (
#     BaseCallback,
#     CallbackManager,
#     ProgressCallback as AdvancedProgressCallback,
#     HistoryCallback,
#     EarlyStoppingCallback as AdvancedEarlyStoppingCallback,
#     CheckpointCallback,
#     MetricsCallback,
#     LoggingCallback,
#     AdaptiveCallback,
# )

# Import validation utilities
# from .validation import (
#     validate_manifold,
#     validate_gradient,
#     validate_optimizer,
#     compare_optimizers,
# )

__version__ = _riemannopt.__version__

__all__ = [
    # Core Rust bindings
    "manifolds",
    "optimizers", 
    "CostFunction",
    "create_cost_function",
    
    # Python convenience functions
    "optimize",
    "gradient_check", 
    "OptimizationCallback",
    "ProgressCallback",
    "EarlyStoppingCallback",
    "plot_convergence",
    "benchmark_optimizers",
    
    # Exceptions
    "RiemannOptError",
    "ManifoldValidationError",
    "OptimizationFailedError",
    "ConvergenceError",
    "LineSearchError",
    "DimensionMismatchError",
    "ConfigurationError",
    "BackendError",
]

# Add integration features if available  
# if _has_integrations:
#     __all__.extend([
#         "TensorConverter",
#         "PyTorchAdapter",
#         "JAXAdapter", 
#         "AutogradAdapter",
#         "optimize_pytorch_model",
#         "optimize_jax_function",
#         "ProgressReporter",
#         "create_sklearn_compatible_optimizer",
#     ])

# Add visualization functions if available
# if _has_visualization:
#     __all__.extend([
#         "plot_sphere_optimization",
#         "plot_stiefel_columns", 
#         "plot_grassmann_subspace",
#         "plot_convergence_comparison",
#         "plot_manifold_tangent_space",
#         "create_optimization_animation",
#     ])