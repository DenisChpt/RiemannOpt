"""Main API functions for RiemannOpt."""

import numpy as np
from typing import Union, Callable, Optional, Dict, Any, Tuple, List
import warnings

from ..types import Point, Scalar, Gradient, OptimizationResult, Manifold, CostFunction as CostFunctionProtocol
from ..exceptions import OptimizationError, InvalidConfigurationError
from .config import get_config
from .logging import get_logger, OptimizationLogger
from .._compat import HAS_TQDM

if HAS_TQDM:
    from tqdm import tqdm

logger = get_logger(__name__)


def optimize(
    manifold: Manifold,
    cost_function: Union[Callable, CostFunctionProtocol],
    initial_point: Point,
    optimizer: Union[str, Any] = "sgd",
    max_iterations: Optional[int] = None,
    gradient_tolerance: Optional[float] = None,
    cost_tolerance: Optional[float] = None,
    callbacks: Optional[List[Any]] = None,
    verbose: bool = False,
    **optimizer_kwargs
) -> OptimizationResult:
    """Optimize a cost function on a manifold.
    
    This is the main entry point for optimization in RiemannOpt.
    It provides a simple interface for common optimization tasks.
    
    The function supports various cost function formats:
    - Simple callable: `f(x) -> float`
    - Callable with gradient: `f(x) -> (float, gradient)`
    - CostFunction object with separate value and gradient methods
    
    Args:
        manifold: The manifold to optimize on. Must have methods like
            `project`, `egrad2rgrad`, `norm`, and `check_point`.
        cost_function: Cost function to minimize. Can be:
            - Callable returning just the cost value
            - Callable returning (cost, gradient) tuple
            - CostFunction object with value_and_gradient method
        initial_point: Starting point on the manifold. Will be projected
            if not already on the manifold.
        optimizer: Optimizer to use. Can be:
            - String: 'sgd', 'adam', 'lbfgs', 'cg' (case-insensitive)
            - Optimizer instance with a `step` method
        max_iterations: Maximum number of iterations. If None, uses the
            value from global configuration (default: 1000).
        gradient_tolerance: Convergence tolerance for gradient norm.
            If None, uses config value (default: 1e-6).
        cost_tolerance: Convergence tolerance for cost function change.
            If None, uses config value (default: 1e-8).
        callbacks: List of callback objects that implement:
            - `on_optimization_start(point)`
            - `on_iteration_end(iteration, point, cost, gradient) -> bool`
            - `on_optimization_end(result)`
        verbose: If True, shows progress bar (requires tqdm) and logs
            convergence information.
        **optimizer_kwargs: Additional arguments passed to the optimizer
            constructor. Common options include:
            - step_size/learning_rate: Step size for gradient methods
            - momentum: Momentum coefficient (for SGD)
            - beta1, beta2: Coefficients for Adam
            - memory_size: Memory size for L-BFGS
        
    Returns:
        Dictionary with optimization results:
            - 'point': Final point on manifold (ndarray)
            - 'cost': Final cost value (float)
            - 'gradient': Final Euclidean gradient (ndarray)
            - 'gradient_norm': Final Riemannian gradient norm (float)
            - 'iterations': Number of iterations performed (int)
            - 'success': Whether optimization converged (bool)
            - 'message': Convergence or termination message (str)
            - 'history': Dictionary with convergence history:
                - 'cost': Array of cost values at each iteration
                - 'gradient_norm': Array of gradient norms
    
    Raises:
        InvalidConfigurationError: If optimizer name is invalid or
            configuration parameters are incompatible.
        OptimizationError: If optimization fails due to numerical
            issues or other errors.
        ValueError: If cost_function is not callable or CostFunction.
    
    Example:
        >>> import riemannopt as ro
        >>> import numpy as np
        >>> 
        >>> # Simple sphere optimization
        >>> sphere = ro.manifolds.Sphere(10)
        >>> x0 = sphere.random_point()
        >>> 
        >>> # Minimize quadratic function
        >>> def cost(x):
        ...     return 0.5 * np.sum(x**2)
        >>> 
        >>> result = ro.optimize(
        ...     sphere, cost, x0, 
        ...     optimizer='sgd',
        ...     learning_rate=0.1,
        ...     max_iterations=100,
        ...     verbose=True
        ... )
        >>> 
        >>> print(f"Final cost: {result['cost']:.6f}")
        >>> print(f"Converged: {result['success']}")
    
    Note:
        The optimization is performed in the Riemannian setting, meaning
        gradients are automatically converted to Riemannian gradients
        and updates respect the manifold geometry.
    """
    config = get_config()
    
    # Set default values from config
    max_iterations = max_iterations or config.max_iterations
    gradient_tolerance = gradient_tolerance or config.gradient_tolerance
    cost_tolerance = cost_tolerance or config.cost_tolerance
    
    # Create cost function wrapper if needed
    cost_fn = create_cost_function(cost_function)
    
    # Create optimizer instance if string provided
    if isinstance(optimizer, str):
        optimizer_instance = _create_optimizer(
            optimizer, manifold, **optimizer_kwargs
        )
    else:
        optimizer_instance = optimizer
    
    # Validate initial point
    if config.validate_inputs:
        if not manifold.check_point(initial_point):
            logger.warning("Initial point not on manifold, projecting...")
            initial_point = manifold.project(initial_point)
    
    # Setup callbacks
    callback_list = callbacks or []
    if verbose and HAS_TQDM:
        # Add progress bar callback
        from ..callbacks import ProgressCallback
        callback_list.append(ProgressCallback(max_iterations))
    
    # Initialize optimization
    point = initial_point.copy()
    cost_history = []
    gradient_history = []
    
    # Optimization loop with logging
    log_name = f"{manifold.__class__.__name__}_{optimizer_instance.__class__.__name__}"
    with OptimizationLogger(log_name) as opt_logger:
        
        # Call callbacks at start
        for callback in callback_list:
            if hasattr(callback, 'on_optimization_start'):
                callback.on_optimization_start(point)
        
        # Main optimization loop
        for iteration in range(max_iterations):
            # Compute cost and gradient
            cost, gradient = cost_fn.value_and_gradient(point)
            
            # Convert to Riemannian gradient
            riemannian_gradient = manifold.egrad2rgrad(point, gradient)
            gradient_norm = manifold.norm(point, riemannian_gradient)
            
            # Store history
            cost_history.append(cost)
            gradient_history.append(gradient_norm)
            
            # Log iteration
            if verbose and iteration % 10 == 0:
                opt_logger.log_iteration(iteration, cost, gradient_norm)
            
            # Check convergence
            converged, message = _check_convergence(
                iteration, cost_history, gradient_norm, 
                gradient_tolerance, cost_tolerance
            )
            
            if converged:
                logger.info(f"Converged: {message}")
                break
            
            # Perform optimization step
            point = optimizer_instance.step(riemannian_gradient, point)
            
            # Ensure point stays on manifold
            if config.validate_inputs and iteration % 10 == 0:
                if not manifold.check_point(point):
                    point = manifold.project(point)
            
            # Call iteration callbacks
            stop = False
            for callback in callback_list:
                if hasattr(callback, 'on_iteration_end'):
                    if not callback.on_iteration_end(
                        iteration, point, cost, riemannian_gradient
                    ):
                        stop = True
                        break
            
            if stop:
                message = "Stopped by callback"
                break
        else:
            # No convergence
            converged = False
            message = f"Maximum iterations ({max_iterations}) reached"
        
        # Final evaluation
        final_cost, final_gradient = cost_fn.value_and_gradient(point)
        final_rgrad = manifold.egrad2rgrad(point, final_gradient)
        final_grad_norm = manifold.norm(point, final_rgrad)
        
        # Prepare result
        result = {
            'point': point,
            'cost': final_cost,
            'gradient': final_gradient,
            'gradient_norm': final_grad_norm,
            'iterations': iteration + 1,
            'success': converged,
            'message': message,
            'history': {
                'cost': np.array(cost_history),
                'gradient_norm': np.array(gradient_history),
            }
        }
        
        # Call end callbacks
        for callback in callback_list:
            if hasattr(callback, 'on_optimization_end'):
                callback.on_optimization_end(result)
        
        # Log result
        opt_logger.log_result(result)
        
        return result


def create_cost_function(
    func: Union[Callable, CostFunctionProtocol],
    gradient: Optional[Callable] = None,
    check_gradient: Optional[bool] = None
) -> CostFunctionProtocol:
    """Create a cost function object from a callable.
    
    This function wraps a Python callable into a CostFunction object
    that can be used with RiemannOpt optimizers.
    
    Args:
        func: Function that computes cost value or (cost, gradient)
        gradient: Optional separate gradient function
        check_gradient: Whether to validate gradient (uses config if None)
        
    Returns:
        CostFunction object
        
    Example:
        >>> # Function returning only cost
        >>> def f(x):
        ...     return np.sum(x**2)
        >>> cost_fn = create_cost_function(f)
        >>> 
        >>> # Function returning cost and gradient
        >>> def f_and_g(x):
        ...     cost = np.sum(x**2)
        ...     grad = 2 * x
        ...     return cost, grad
        >>> cost_fn = create_cost_function(f_and_g)
        >>> 
        >>> # Separate cost and gradient
        >>> def f(x):
        ...     return np.sum(x**2)
        >>> def g(x):
        ...     return 2 * x
        >>> cost_fn = create_cost_function(f, gradient=g)
    """
    from .. import _riemannopt
    
    # If already a CostFunction, return as-is
    if hasattr(func, 'value_and_gradient'):
        return func
    
    config = get_config()
    check_gradient = check_gradient if check_gradient is not None else config.check_gradients
    
    # Create wrapper based on function signature
    if gradient is not None:
        # Separate cost and gradient functions
        class CostFunctionWrapper:
            def __call__(self, x):
                return func(x)
            
            def gradient(self, x):
                return gradient(x)
            
            def value_and_gradient(self, x):
                return func(x), gradient(x)
    
    else:
        # Check if function returns gradient
        test_input = np.random.randn(10)  # Small test input
        try:
            result = func(test_input)
            returns_gradient = isinstance(result, tuple) and len(result) == 2
        except:
            returns_gradient = False
        
        if returns_gradient:
            # Function returns (cost, gradient)
            class CostFunctionWrapper:
                def __call__(self, x):
                    return func(x)[0]
                
                def gradient(self, x):
                    return func(x)[1]
                
                def value_and_gradient(self, x):
                    return func(x)
        else:
            # Function returns only cost, use finite differences
            class CostFunctionWrapper:
                def __call__(self, x):
                    return func(x)
                
                def gradient(self, x):
                    # Use finite differences
                    return _finite_difference_gradient(func, x)
                
                def value_and_gradient(self, x):
                    cost = func(x)
                    grad = self.gradient(x)
                    return cost, grad
    
    wrapper = CostFunctionWrapper()
    
    # Optionally check gradient
    if check_gradient:
        _validate_gradient(wrapper)
    
    return wrapper


def _create_optimizer(name: str, manifold: Manifold, **kwargs) -> Any:
    """Create optimizer instance from string name."""
    from .. import optimizers
    
    optimizer_map = {
        'sgd': optimizers.SGD,
        'adam': optimizers.Adam,
        'lbfgs': optimizers.LBFGS,
        'cg': optimizers.ConjugateGradient,
        'conjugategradient': optimizers.ConjugateGradient,
    }
    
    name_lower = name.lower()
    if name_lower not in optimizer_map:
        available = list(optimizer_map.keys())
        raise InvalidConfigurationError(
            f"Unknown optimizer '{name}'. Available: {available}"
        )
    
    optimizer_class = optimizer_map[name_lower]
    return optimizer_class(manifold, **kwargs)


def _check_convergence(
    iteration: int,
    cost_history: List[float],
    gradient_norm: float,
    gradient_tolerance: float,
    cost_tolerance: float
) -> Tuple[bool, str]:
    """Check convergence criteria."""
    # Gradient convergence
    if gradient_norm < gradient_tolerance:
        return True, f"Gradient norm ({gradient_norm:.2e}) below tolerance"
    
    # Cost convergence
    if len(cost_history) > 1:
        cost_change = abs(cost_history[-1] - cost_history[-2])
        if cost_change < cost_tolerance:
            return True, f"Cost change ({cost_change:.2e}) below tolerance"
    
    return False, ""


def _finite_difference_gradient(
    func: Callable[[np.ndarray], float],
    x: np.ndarray,
    epsilon: Optional[float] = None
) -> np.ndarray:
    """Compute gradient using finite differences."""
    config = get_config()
    epsilon = epsilon or config.epsilon
    
    grad = np.zeros_like(x)
    f0 = func(x)
    
    for i in range(x.shape[0]):
        x_plus = x.copy()
        x_plus[i] += epsilon
        f_plus = func(x_plus)
        grad[i] = (f_plus - f0) / epsilon
    
    return grad


def _validate_gradient(cost_function: CostFunctionProtocol) -> None:
    """Validate gradient computation using finite differences."""
    # Create random test point
    test_point = np.random.randn(10)
    
    # Compute analytical gradient
    _, grad_analytical = cost_function.value_and_gradient(test_point)
    
    # Compute finite difference gradient
    grad_fd = _finite_difference_gradient(cost_function, test_point)
    
    # Compare
    rel_error = np.linalg.norm(grad_analytical - grad_fd) / (
        np.linalg.norm(grad_analytical) + 1e-10
    )
    
    if rel_error > 1e-5:
        warnings.warn(
            f"Gradient validation failed: relative error = {rel_error:.2e}. "
            "Check your gradient computation.",
            UserWarning
        )