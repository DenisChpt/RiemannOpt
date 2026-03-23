"""RiemannOpt: High-performance Riemannian optimization in Python.

Provides state-of-the-art algorithms for optimization on Riemannian manifolds,
powered by a Rust backend for 10-100x performance over pure Python.

Quick Start
-----------
>>> import riemannopt as ro
>>> import numpy as np
>>>
>>> sphere = ro.manifolds.Sphere(10)
>>> cost_fn = ro.create_cost_function(lambda x: float(x @ x))
>>> x0 = sphere.random_point()
>>> result = ro.optimize(cost_fn, sphere, x0, optimizer="Adam", max_iterations=500)
>>> print(f"Converged: {result.converged}, cost: {result.cost:.6f}")

Manifolds
---------
- ``Sphere(n)`` -- Unit sphere S^{n-1} in R^n
- ``Stiefel(n, p)`` -- Orthonormal n x p matrices (X^T X = I_p)
- ``Grassmann(n, p)`` -- p-dimensional subspaces of R^n
- ``SPD(n)`` -- n x n symmetric positive-definite matrices
- ``Hyperbolic(n)`` -- Hyperbolic space H^n (Poincare ball model)
- ``Oblique(n, p)`` -- n x p matrices with unit-norm columns
- ``PSDCone(n)`` -- Positive semidefinite cone
- ``Euclidean(n)`` -- Flat R^n (baseline)
- ``ProductManifold([M1, M2, ...])`` -- Cartesian product

Optimizers
----------
- ``SGD`` -- Riemannian stochastic gradient descent with optional momentum
- ``Adam`` -- Riemannian Adam with bias correction
- ``LBFGS`` -- Riemannian limited-memory BFGS
- ``ConjugateGradient`` -- Nonlinear CG (FR, PR, HS, DY variants)
- ``TrustRegion`` -- Trust-region with Steihaug-CG subproblem solver
- ``Newton`` -- Riemannian Newton with CG inner solver
- ``NaturalGradient`` -- Natural gradient descent

References
----------
.. [1] Absil, Mahony, Sepulchre. *Optimization Algorithms on Matrix Manifolds*, 2008.
.. [2] Boumal. *An Introduction to Optimization on Smooth Manifolds*, 2023.
"""

from __future__ import annotations

# Rust native module
from . import _riemannopt

# Submodules -- manifold and optimizer classes live here
from ._riemannopt import manifolds, optimizers

# Cost function utilities
from ._riemannopt import CostFunction, create_cost_function

# Native Rust cost functions (zero Python callback overhead)
from ._riemannopt import (
    RayleighQuotient,
    TraceMinimization,
    Brockett,
    LogDetDivergence,
    Quadratic,
)

# Callback system
from ._riemannopt import CallbackInfo, CallbackManager

# High-level convenience API (Python wrappers)
from .helpers import (
    optimize,
    gradient_check,
    create_cost_function as make_cost_function,
    OptimizationCallback,
    ProgressCallback,
    EarlyStoppingCallback,
    plot_convergence,
    benchmark_optimizers,
)

# Exceptions -- re-export from Rust module
from ._riemannopt import (
    RiemannOptError,
    ManifoldValidationError,
    OptimizationFailedError,
    ConvergenceError,
    LineSearchError,
    DimensionMismatchError,
    NumericalError,
    NotImplementedMethodError,
)

# Python-side exceptions
from .exceptions import ConfigurationError, BackendError

__version__: str = _riemannopt.__version__

__all__ = [
    # Submodules
    "manifolds",
    "optimizers",
    # Cost function
    "CostFunction",
    "create_cost_function",
    # Native cost functions
    "RayleighQuotient",
    "TraceMinimization",
    "Brockett",
    "LogDetDivergence",
    "Quadratic",
    # Callbacks
    "CallbackInfo",
    "CallbackManager",
    # High-level API
    "optimize",
    "gradient_check",
    "make_cost_function",
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
    "NumericalError",
    "NotImplementedMethodError",
    "ConfigurationError",
    "BackendError",
]
