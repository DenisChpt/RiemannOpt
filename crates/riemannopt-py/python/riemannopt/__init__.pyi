"""Type stubs for RiemannOpt."""

from typing import Optional, Tuple, Dict, Any, Callable, Union, overload, List, Protocol
import numpy as np
import numpy.typing as npt

__version__: str

# Core classes
class CostFunction:
    """Cost function wrapper for optimization."""
    
    def cost(self, point: npt.NDArray[np.float64]) -> float:
        """Evaluate cost at a point."""
        ...
    
    def gradient(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Evaluate gradient at a point."""
        ...
    
    def cost_and_gradient(self, point: npt.NDArray[np.float64]) -> Tuple[float, npt.NDArray[np.float64]]:
        """Evaluate both cost and gradient."""
        ...
    
    @property
    def eval_counts(self) -> Dict[str, int]:
        """Get evaluation counts."""
        ...
    
    def reset_counts(self) -> None:
        """Reset evaluation counters."""
        ...

class OptimizationResult:
    """Result from optimization."""
    
    point: npt.NDArray[np.float64]
    value: float
    gradient_norm: Optional[float]
    converged: bool
    iterations: int
    function_evals: int
    gradient_evals: int
    time_seconds: float
    termination_reason: str
    history: Optional[Dict[str, Any]]
    
    @property
    def cost(self) -> float:
        """Alias for value."""
        ...
    
    @property
    def x(self) -> npt.NDArray[np.float64]:
        """Alias for point."""
        ...
    
    @property
    def success(self) -> bool:
        """Alias for converged."""
        ...
    
    @property
    def as_dict(self) -> Dict[str, Any]:
        """Get result as dictionary."""
        ...
    
    def summary(self) -> str:
        """Get summary string."""
        ...

# Python convenience functions
def optimize(
    manifold: Any,
    cost_function: Union[Callable[[npt.NDArray[np.float64]], float], CostFunction],
    initial_point: npt.NDArray[np.float64],
    optimizer: str = "Adam",
    max_iterations: int = 1000,
    gradient_tolerance: float = 1e-6,
    callback: Optional[Callable[[int, npt.NDArray[np.float64], float, float], bool]] = None,
    **optimizer_kwargs: Any
) -> 'OptimizationResult':
    """High-level optimization interface.
    
    Parameters
    ----------
    manifold : Manifold
        The manifold on which to optimize.
    cost_function : callable or CostFunction
        Cost function to minimize.
    initial_point : array_like
        Starting point for optimization.
    optimizer : str, default="Adam"
        Name of the optimizer to use.
    max_iterations : int, default=1000
        Maximum number of iterations.
    gradient_tolerance : float, default=1e-6
        Tolerance for gradient norm.
    callback : callable, optional
        Callback function.
    **optimizer_kwargs
        Additional optimizer parameters.
    
    Returns
    -------
    OptimizationResult
        Result object with optimization details.
    """
    ...

def gradient_check(
    cost_function: CostFunction,
    point: npt.NDArray[np.float64],
    epsilon: float = 1e-8,
    tolerance: float = 1e-5
) -> Tuple[bool, float]:
    """Check gradient accuracy using finite differences.
    
    Parameters
    ----------
    cost_function : CostFunction
        Cost function with gradient.
    point : array_like
        Point at which to check gradient.
    epsilon : float, default=1e-8
        Step size for finite differences.
    tolerance : float, default=1e-5
        Tolerance for gradient accuracy.
    
    Returns
    -------
    is_accurate : bool
        True if gradient is accurate.
    max_error : float
        Maximum relative error found.
    """
    ...

def create_cost_function(
    cost_fn: Callable[[npt.NDArray[np.float64]], Union[float, Tuple[float, npt.NDArray[np.float64]]]],
    gradient_fn: Optional[Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]] = None,
    validate_gradient: bool = False,
    dimension: Optional[Union[int, Tuple[int, int]]] = None
) -> CostFunction:
    """Create a cost function wrapper for optimization.
    
    Parameters
    ----------
    cost_fn : callable
        Function that computes the cost, or returns (cost, gradient).
    gradient_fn : callable, optional
        Function that computes the gradient.
    validate_gradient : bool, default=False
        Whether to validate gradient using finite differences.
    dimension : int or tuple, optional
        Problem dimension.
    
    Returns
    -------
    CostFunction
        Wrapped cost function.
    """
    ...

class OptimizationCallback(Protocol):
    """Protocol for optimization callbacks."""
    
    def __call__(self, iteration: int, value: float, gradient_norm: float) -> None: ...

class ProgressCallback:
    """Progress callback that prints optimization progress."""
    
    def __init__(self, print_every: int = 10, verbose: bool = True) -> None: ...
    def __call__(self, iteration: int, value: float, gradient_norm: float) -> None: ...
    def reset(self) -> None: ...

class EarlyStoppingCallback:
    """Early stopping based on improvement criteria."""
    
    def __init__(self, patience: int = 10, min_improvement: float = 1e-8) -> None: ...
    def __call__(self, iteration: int, value: float, gradient_norm: float) -> None: ...
    @property
    def should_stop(self) -> bool: ...

def plot_convergence(
    result: Dict[str, Any], 
    callback: Optional[OptimizationCallback] = None
) -> None:
    """Plot optimization convergence."""
    ...

def benchmark_optimizers(
    manifold: Any,
    cost_function: Callable[[npt.NDArray[np.float64]], float],
    initial_point: npt.NDArray[np.float64],
    optimizers_config: Optional[Dict[str, Dict[str, Any]]] = None,
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """Benchmark different optimizers on the same problem."""
    ...

# Exceptions
class RiemannOptError(Exception): ...
class ManifoldError(RiemannOptError): ...
class InvalidPointError(ManifoldError): 
    point: Optional[npt.NDArray[np.float64]]
    manifold: Optional[Any]

class InvalidTangentError(ManifoldError):
    vector: Optional[npt.NDArray[np.float64]]
    point: Optional[npt.NDArray[np.float64]]
    manifold: Optional[Any]

class DimensionMismatchError(ManifoldError):
    expected_shape: Optional[Tuple[int, ...]]
    actual_shape: Optional[Tuple[int, ...]]

class OptimizationError(RiemannOptError): ...
class ConvergenceError(OptimizationError):
    iterations: Optional[int]
    final_value: Optional[float]

class LineSearchError(OptimizationError):
    step_size: Optional[float]

class InvalidConfigurationError(OptimizationError): ...
class NumericalError(RiemannOptError): ...
class GradientError(RiemannOptError):
    relative_error: Optional[float]
    tolerance: Optional[float]

# Decorators
def validate_arrays(*array_args: str, shapes: Optional[List[Optional[Tuple[int, ...]]]] = None, dtypes: Optional[List[Optional[type]]] = None) -> Callable: ...
def ensure_on_manifold(manifold_arg: str = 'self', point_args: Optional[List[str]] = None) -> Callable: ...
def handle_rust_exceptions(func: Callable) -> Callable: ...
def deprecated(reason: str) -> Callable: ...
def cache_result(maxsize: int = 128) -> Callable: ...
def time_function(func: Callable) -> Callable: ...
def require_gradient(func: Callable) -> Callable: ...

class property_cached:
    """Cached property decorator."""
    def __init__(self, func: Callable) -> None: ...
    def __get__(self, obj: Any, type: Optional[type] = None) -> Any: ...
    def __set__(self, obj: Any, value: Any) -> None: ...
    def __delete__(self, obj: Any) -> None: ...

def vectorize_manifold_operation(signature: str = "(n),()->(n)") -> Callable: ...

# Visualization (optional - depends on matplotlib)
def plot_sphere_optimization(
    sphere: Any,
    cost_function: Callable[[npt.NDArray[np.float64]], float],
    result: Dict[str, Any],
    trajectory: Optional[List[npt.NDArray[np.float64]]] = None,
    ax: Optional[Any] = None,
    title: str = "Sphere Optimization"
) -> None: ...

def plot_stiefel_columns(
    stiefel: Any,
    point: npt.NDArray[np.float64],
    title: str = "Stiefel Manifold Point"
) -> None: ...

def plot_grassmann_subspace(
    grassmann: Any,
    point: npt.NDArray[np.float64],
    title: str = "Grassmann Manifold Subspace"
) -> None: ...

def plot_convergence_comparison(
    results: Dict[str, Dict[str, Any]],
    metric: str = 'value',
    title: str = "Optimizer Comparison"
) -> None: ...

def plot_manifold_tangent_space(
    manifold: Any,
    point: npt.NDArray[np.float64],
    tangent_vectors: Optional[List[npt.NDArray[np.float64]]] = None,
    title: str = "Manifold and Tangent Space"
) -> None: ...

def create_optimization_animation(
    manifold: Any,
    cost_function: Callable[[npt.NDArray[np.float64]], float],
    trajectory: List[npt.NDArray[np.float64]],
    filename: str = "optimization.gif",
    fps: int = 5
) -> None: ...

# Original functions from Rust
def quadratic_cost() -> CostFunction: ...
def rosenbrock_cost() -> CostFunction: ...
def check_point_on_manifold(manifold: Any, point: npt.NDArray[np.float64], tolerance: Optional[float] = None) -> bool: ...
def check_vector_in_tangent_space(manifold: Any, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64], tolerance: Optional[float] = None) -> bool: ...
def format_result(result: Dict[str, Any]) -> str: ...
def validate_point_shape(manifold: Any, point: npt.NDArray[np.float64]) -> None: ...
def default_line_search() -> Callable: ...

# Manifolds
class Sphere:
    """Unit sphere manifold S^(n-1)."""
    def __init__(self, dimension: int) -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def ambient_dim(self) -> int: ...
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def project_tangent(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def retract(self, point: npt.NDArray[np.float64], tangent: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def exp(self, point: npt.NDArray[np.float64], tangent: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def log(self, point: npt.NDArray[np.float64], other: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inner(self, point: npt.NDArray[np.float64], u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float: ...
    def norm(self, point: npt.NDArray[np.float64], tangent: npt.NDArray[np.float64]) -> float: ...
    def distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float: ...
    def random_point(self) -> npt.NDArray[np.float64]: ...
    def random_tangent(self, point: npt.NDArray[np.float64], scale: float = 1.0) -> npt.NDArray[np.float64]: ...
    def parallel_transport(self, from_point: npt.NDArray[np.float64], to_point: npt.NDArray[np.float64], tangent: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def contains(self, point: npt.NDArray[np.float64], atol: float = 1e-10) -> bool: ...
    def is_tangent(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64], atol: float = 1e-10) -> bool: ...

class Stiefel:
    """Stiefel manifold St(n,p)."""
    def __init__(self, n: int, p: int) -> None: ...
    @property
    def n(self) -> int: ...
    @property
    def p(self) -> int: ...
    @property
    def dim(self) -> int: ...
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def project_tangent(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def retract(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inner(self, point: npt.NDArray[np.float64], u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float: ...
    def norm(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> float: ...
    def random_point(self) -> npt.NDArray[np.float64]: ...
    def random_tangent(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class Grassmann:
    """Grassmann manifold Gr(n,p)."""
    def __init__(self, n: int, p: int) -> None: ...
    @property
    def n(self) -> int: ...
    @property
    def p(self) -> int: ...
    @property
    def dim(self) -> int: ...
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def project_tangent(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def retract(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inner(self, point: npt.NDArray[np.float64], u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float: ...
    def norm(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> float: ...
    def random_point(self) -> npt.NDArray[np.float64]: ...
    def random_tangent(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class SPD:
    """Symmetric Positive Definite matrices manifold."""
    def __init__(self, n: int) -> None: ...
    @property
    def n(self) -> int: ...
    @property
    def dim(self) -> int: ...
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def project_tangent(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def retract(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inner(self, point: npt.NDArray[np.float64], u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float: ...
    def norm(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> float: ...
    def random_point(self) -> npt.NDArray[np.float64]: ...
    def random_tangent(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class Hyperbolic:
    """Hyperbolic manifold H^n."""
    def __init__(self, dim: int, model: str = "lorentz") -> None: ...
    @property
    def dim(self) -> int: ...
    @property
    def model(self) -> str: ...
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def project_tangent(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def retract(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def inner(self, point: npt.NDArray[np.float64], u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float: ...
    def norm(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> float: ...
    def random_point(self) -> npt.NDArray[np.float64]: ...
    def random_tangent(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class ProductManifold:
    """Product of manifolds."""
    def __init__(self, manifolds: List[Any]) -> None: ...
    @property
    def manifolds(self) -> List[Any]: ...
    @property
    def dim(self) -> int: ...
    def project(self, point: List[npt.NDArray[np.float64]]) -> List[npt.NDArray[np.float64]]: ...
    def project_tangent(self, point: List[npt.NDArray[np.float64]], vector: List[npt.NDArray[np.float64]]) -> List[npt.NDArray[np.float64]]: ...
    def retract(self, point: List[npt.NDArray[np.float64]], vector: List[npt.NDArray[np.float64]]) -> List[npt.NDArray[np.float64]]: ...

# Optimizers
class SGD:
    """Stochastic Gradient Descent optimizer."""
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        gradient_clip: Optional[float] = None,
        line_search: bool = False
    ) -> None: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...
    
    def optimize_sphere(
        self,
        cost_function: CostFunction,
        sphere: Sphere,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult: ...
    
    def optimize_stiefel(
        self,
        cost_function: CostFunction,
        stiefel: Stiefel,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult: ...

class Adam:
    """Adam optimizer for Riemannian manifolds."""
    def __init__(
        self,
        manifold: Any,
        cost_fn: CostFunction,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> None: ...
    def optimize(self, initial_point: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], float, Dict[str, Any]]: ...

class LBFGS:
    """L-BFGS optimizer for Riemannian manifolds."""
    def __init__(
        self,
        manifold: Any,
        cost_fn: CostFunction,
        memory_size: int = 10,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> None: ...
    def optimize(self, initial_point: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], float, Dict[str, Any]]: ...

class ConjugateGradient:
    """Conjugate Gradient optimizer."""
    def __init__(
        self,
        manifold: Any,
        cost_fn: CostFunction,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> None: ...
    def optimize(self, initial_point: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], float, Dict[str, Any]]: ...

class TrustRegions:
    """Trust Region optimizer."""
    def __init__(
        self,
        manifold: Any,
        cost_fn: CostFunction,
        max_iter: int = 100,
        tolerance: float = 1e-6,
        verbose: bool = False
    ) -> None: ...
    def optimize(self, initial_point: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], float, Dict[str, Any]]: ...