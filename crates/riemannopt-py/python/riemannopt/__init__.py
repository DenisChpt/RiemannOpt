"""RiemannOpt: High-performance Riemannian optimization in Python.

This package provides tools for optimization on Riemannian manifolds,
including various manifold types and optimization algorithms.

Example:
    >>> import riemannopt as ro
    >>> import numpy as np
    >>> 
    >>> # Create a sphere manifold
    >>> sphere = ro.manifolds.Sphere(10)
    >>> 
    >>> # Define a cost function
    >>> def cost(x):
    ...     return 0.5 * np.dot(x, x)
    >>> 
    >>> # Simple optimization
    >>> x0 = sphere.random_point()
    >>> result = ro.optimize(sphere, cost, x0, optimizer='sgd', learning_rate=0.1)
    >>> 
    >>> # Or create explicit optimizer
    >>> cost_fn = ro.create_cost_function(cost)
    >>> optimizer = ro.optimizers.SGD(sphere, learning_rate=0.1)
    >>> result = optimizer.optimize(cost_fn, x0)
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
    quadratic_cost,
    rosenbrock_cost,
    # Utilities
    check_point_on_manifold,
    check_vector_in_tangent_space,
    format_result,
    validate_point_shape,
    default_line_search,
)

# Import Python convenience functions
from .helpers import (
    optimize,
    gradient_check,
    create_cost_function,
    OptimizationCallback,
    ProgressCallback,
    EarlyStoppingCallback,
    plot_convergence,
    benchmark_optimizers,
)

# Import exceptions
from .exceptions import (
    RiemannOptError,
    ManifoldError,
    InvalidPointError,
    InvalidTangentError,
    DimensionMismatchError,
    OptimizationError,
    ConvergenceError,
    LineSearchError,
    InvalidConfigurationError,
    NumericalError,
    GradientError,
)

# Import visualization (optional)
try:
    from .visualization import (
        plot_sphere_optimization,
        plot_stiefel_columns,
        plot_grassmann_subspace,
        plot_convergence_comparison,
        plot_manifold_tangent_space,
        create_optimization_animation,
    )
    _has_visualization = True
except ImportError:
    _has_visualization = False

# Import decorators
from .decorators import (
    validate_arrays,
    ensure_on_manifold,
    handle_rust_exceptions,
    deprecated,
    cache_result,
    time_function,
    require_gradient,
    property_cached,
    vectorize_manifold_operation,
)

# Import integration features
try:
    from .integrations import (
        TensorConverter,
        PyTorchAdapter,
        JAXAdapter,
        AutogradAdapter,
        optimize_pytorch_model,
        optimize_jax_function,
        ProgressReporter,
        create_sklearn_compatible_optimizer,
    )
    _has_integrations = True
except ImportError:
    _has_integrations = False

# Import callbacks
from .callbacks import (
    BaseCallback,
    CallbackManager,
    ProgressCallback as AdvancedProgressCallback,
    HistoryCallback,
    EarlyStoppingCallback as AdvancedEarlyStoppingCallback,
    CheckpointCallback,
    MetricsCallback,
    LoggingCallback,
    AdaptiveCallback,
)

# Import validation utilities
from .validation import (
    validate_manifold,
    validate_gradient,
    validate_optimizer,
    compare_optimizers,
)

__version__ = _riemannopt.__version__

__all__ = [
    # Core Rust bindings
    "manifolds",
    "optimizers", 
    "CostFunction",
    "quadratic_cost",
    "rosenbrock_cost",
    "check_point_on_manifold",
    "check_vector_in_tangent_space",
    "format_result",
    "validate_point_shape",
    "default_line_search",
    
    # Python convenience functions
    "optimize",
    "gradient_check", 
    "create_cost_function",
    "OptimizationCallback",
    "ProgressCallback",
    "EarlyStoppingCallback",
    "plot_convergence",
    "benchmark_optimizers",
    
    # Exceptions
    "RiemannOptError",
    "ManifoldError",
    "InvalidPointError",
    "InvalidTangentError", 
    "DimensionMismatchError",
    "OptimizationError",
    "ConvergenceError",
    "LineSearchError",
    "InvalidConfigurationError",
    "NumericalError",
    "GradientError",
    
    # Decorators
    "validate_arrays",
    "ensure_on_manifold",
    "handle_rust_exceptions",
    "deprecated",
    "cache_result",
    "time_function",
    "require_gradient",
    "property_cached",
    "vectorize_manifold_operation",
    
    # Callbacks
    "BaseCallback",
    "CallbackManager",
    "AdvancedProgressCallback",
    "HistoryCallback",
    "AdvancedEarlyStoppingCallback",
    "CheckpointCallback",
    "MetricsCallback",
    "LoggingCallback",
    "AdaptiveCallback",
    
    # Validation
    "validate_manifold",
    "validate_gradient",
    "validate_optimizer",
    "compare_optimizers",
]

# Add integration features if available
if _has_integrations:
    __all__.extend([
        "TensorConverter",
        "PyTorchAdapter",
        "JAXAdapter", 
        "AutogradAdapter",
        "optimize_pytorch_model",
        "optimize_jax_function",
        "ProgressReporter",
        "create_sklearn_compatible_optimizer",
    ])

# Add visualization functions if available
if _has_visualization:
    __all__.extend([
        "plot_sphere_optimization",
        "plot_stiefel_columns", 
        "plot_grassmann_subspace",
        "plot_convergence_comparison",
        "plot_manifold_tangent_space",
        "create_optimization_animation",
    ])