"""Helper functions and convenience utilities for RiemannOpt.

This module provides high-level convenience functions for common optimization tasks,
making it easy to use RiemannOpt without needing to manage optimizer instances directly.
"""

import numpy as np
from typing import Union, Callable, Optional, Dict, Any, Tuple, List, TYPE_CHECKING
import warnings
from numpy.typing import NDArray

if TYPE_CHECKING:
    from . import manifolds, optimizers

# Import at module level to avoid circular imports and ensure proper type checking
try:
    from . import CostFunction
except ImportError:
    CostFunction = None


def _finite_difference_gradient(
    cost_fn: Callable[[NDArray[np.float64]], float],
    point: NDArray[np.float64], 
    epsilon: float = 1e-8
) -> NDArray[np.float64]:
    """Compute gradient using finite differences.
    
    Args:
        cost_fn: Cost function (callable or CostFunction)
        point: Point to compute gradient at
        epsilon: Step size for finite differences
        
    Returns:
        Gradient vector
    """
    n = len(point)
    grad = np.zeros_like(point)
    
    for i in range(n):
        point_plus = point.copy()
        point_minus = point.copy()
        
        point_plus[i] += epsilon
        point_minus[i] -= epsilon
        
        f_plus = cost_fn(point_plus)
        f_minus = cost_fn(point_minus)
        
        grad[i] = (f_plus - f_minus) / (2 * epsilon)
    
    return grad


def optimize(
    manifold: Any,  # Should be a Manifold instance
    cost_function: Union[Callable[[NDArray[np.float64]], float], 
                        Callable[[NDArray[np.float64]], Tuple[float, NDArray[np.float64]]], 
                        "CostFunction"],
    initial_point: NDArray[np.float64],
    optimizer: str = "sgd",
    **kwargs: Any
) -> Dict[str, Any]:
    """Convenience function for optimization on manifolds.
    
    This function provides a simple interface for Riemannian optimization
    without requiring explicit optimizer creation. It handles the creation
    of optimizer instances and manages the optimization loop.
    
    Args:
        manifold: The manifold to optimize on. Should have standard manifold
            methods like `project`, `tangent_projection`, and `retract`.
        cost_function: Cost function to minimize. Can be:
            - Simple callable: `f(x) -> float` (gradient via finite differences)
            - Callable with gradient: `f(x) -> (float, gradient)`
            - CostFunction instance with `value_and_gradient` method
        initial_point: Starting point on the manifold. Will be used as-is
            if already on manifold, otherwise will be projected.
        optimizer: Optimizer type as string (case-insensitive):
            - 'sgd': Stochastic Gradient Descent with optional momentum
            - 'adam': Adaptive Moment Estimation optimizer
            - 'lbfgs': Limited-memory BFGS (quasi-Newton method)
            - 'cg': Conjugate Gradient method
        **kwargs: Optimizer-specific parameters:
            - Common to all:
                - max_iterations (int): Maximum iterations (default: 1000)
                - tolerance (float): Convergence tolerance (default: 1e-6)
            - SGD specific:
                - learning_rate/step_size (float): Step size (default: 0.01)
                - momentum (float): Momentum coefficient (default: 0.0)
            - Adam specific:
                - learning_rate/step_size (float): Step size (default: 0.001)
                - beta1 (float): First moment decay (default: 0.9)
                - beta2 (float): Second moment decay (default: 0.999)
                - epsilon (float): Numerical stability (default: 1e-8)
            - L-BFGS specific:
                - memory_size (int): Number of stored iterations (default: 10)
                - line_search (str): Line search type (default: 'wolfe')
        
    Returns:
        Dictionary with optimization results:
            - 'point': Final optimized point on manifold
            - 'value': Final cost function value
            - 'iterations': Number of iterations performed
            - 'converged': Whether optimization converged successfully
            
    Raises:
        ValueError: If optimizer name is invalid or cost_function is not callable
        TypeError: If optimizer constructor receives unexpected arguments
        
    Example:
        >>> import riemannopt as ro
        >>> import numpy as np
        >>> 
        >>> # Optimize on sphere manifold
        >>> sphere = ro.manifolds.Sphere(10)
        >>> x0 = sphere.random_point()
        >>> 
        >>> # Simple cost function (gradient computed automatically)
        >>> def cost(x):
        ...     return -x[0]  # Maximize first component
        >>> 
        >>> result = ro.optimize(
        ...     sphere, cost, x0, 
        ...     optimizer='sgd',
        ...     learning_rate=0.1,
        ...     max_iterations=100
        ... )
        >>> 
        >>> print(f"Converged: {result['converged']}")
        >>> print(f"Final value: {result['value']:.6f}")
        >>> 
        >>> # Cost function with gradient
        >>> def cost_with_grad(x):
        ...     value = 0.5 * np.sum(x**2)
        ...     grad = x
        ...     return value, grad
        >>> 
        >>> result = ro.optimize(sphere, cost_with_grad, x0, optimizer='adam')
    
    Note:
        This is a simplified wrapper around the more comprehensive
        `riemannopt.core.optimize` function. For advanced features like
        callbacks, logging, and detailed convergence history, use the
        core API directly.
    """
    from . import optimizers
    
    # Import CostFunction locally if not already imported
    global CostFunction
    if CostFunction is None:
        from . import CostFunction
    
    # Convert cost function if needed
    if isinstance(cost_function, CostFunction):
        cost_fn = cost_function
    elif callable(cost_function):
        # Check if the function has a gradient attribute
        gradient_fn = getattr(cost_function, 'gradient', None)
        cost_fn = CostFunction(cost_function, gradient_fn)
    else:
        raise ValueError("cost_function must be callable or CostFunction instance")
    
    # Create optimizer based on string
    optimizer_map = {
        'sgd': optimizers.SGD,
        'adam': optimizers.Adam,
        'lbfgs': optimizers.LBFGS,
        'cg': optimizers.ConjugateGradient,
    }
    
    if optimizer.lower() not in optimizer_map:
        raise ValueError(f"Unknown optimizer '{optimizer}'. Available: {list(optimizer_map.keys())}")
    
    opt_class = optimizer_map[optimizer.lower()]
    
    # Map common parameter names to expected ones
    param_mapping = {
        'learning_rate': 'step_size',
        'lr': 'step_size',
    }
    
    # Create a copy of kwargs with mapped parameters
    mapped_kwargs = {}
    for key, value in kwargs.items():
        mapped_key = param_mapping.get(key, key)
        mapped_kwargs[mapped_key] = value
    
    # Extract optimization parameters
    max_iterations = mapped_kwargs.pop('max_iterations', 1000)
    tolerance = mapped_kwargs.pop('tolerance', 1e-6)
    
    # Create optimizer with appropriate parameters based on type
    if optimizer.lower() in ['sgd', 'adam']:
        # SGD and Adam don't take manifold in constructor
        if optimizer.lower() == 'sgd':
            # Filter parameters for SGD
            sgd_params = {k: v for k, v in mapped_kwargs.items() 
                         if k in ['step_size', 'momentum']}
            opt = opt_class(
                step_size=sgd_params.get('step_size', 0.01),
                momentum=sgd_params.get('momentum', 0.0),
                max_iterations=max_iterations,
                tolerance=tolerance
            )
        else:  # adam
            # Filter parameters for Adam
            adam_params = {k: v for k, v in mapped_kwargs.items() 
                          if k in ['learning_rate', 'beta1', 'beta2', 'epsilon']}
            opt = opt_class(
                learning_rate=adam_params.get('learning_rate', 0.001),
                beta1=adam_params.get('beta1', 0.9),
                beta2=adam_params.get('beta2', 0.999),
                epsilon=adam_params.get('epsilon', 1e-8)
            )
    else:
        # LBFGS and CG take manifold in constructor
        if optimizer.lower() == 'lbfgs':
            opt = opt_class(
                manifold,
                memory_size=mapped_kwargs.get('memory_size', 10),
                _max_iterations=max_iterations,
                _tolerance=tolerance
            )
        else:  # cg
            opt = opt_class(
                manifold,
                variant=mapped_kwargs.get('variant', 'FletcherReeves'),
                _max_iterations=max_iterations,
                _tolerance=tolerance
            )
    
    # For SGD and Adam, we need to manually run the optimization loop
    # since they don't have the manifold set internally
    # All optimizers now use their optimize method
    return opt.optimize(cost_fn, manifold, initial_point)


def gradient_check(
    manifold: Any,  # Should be a Manifold instance
    cost_function: Callable[[NDArray[np.float64]], float],
    point: NDArray[np.float64],
    gradient_function: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    epsilon: float = 1e-8,
    tolerance: float = 1e-5
) -> Dict[str, Any]:
    """Check gradient computation using finite differences.
    
    Args:
        manifold: The manifold
        cost_function: Cost function
        point: Point to check gradient at
        gradient_function: Gradient function (if None, uses finite differences)
        epsilon: Step size for finite differences
        tolerance: Tolerance for gradient check
        
    Returns:
        Dictionary with gradient check results
    """
    from . import CostFunction
    
    # Project point to manifold
    point = manifold.project(point)
    
    if gradient_function is not None:
        # Compare analytical vs numerical gradient
        analytical_grad = gradient_function(point)
        analytical_grad = manifold.tangent_projection(point, analytical_grad)
    else:
        warnings.warn("No gradient function provided, using finite differences only")
        analytical_grad = None
    
    # Compute numerical gradient
    n = len(point)
    numerical_grad = np.zeros_like(point)
    
    for i in range(n):
        # Forward difference
        point_plus = point.copy()
        point_plus[i] += epsilon
        point_plus = manifold.project(point_plus)
        
        point_minus = point.copy()
        point_minus[i] -= epsilon
        point_minus = manifold.project(point_minus)
        
        f_plus = cost_function(point_plus)
        f_minus = cost_function(point_minus)
        
        numerical_grad[i] = (f_plus - f_minus) / (2 * epsilon)
    
    # Project to tangent space
    numerical_grad = manifold.tangent_projection(point, numerical_grad)
    
    result = {
        'numerical_gradient': numerical_grad,
        'point': point,
        'epsilon': epsilon,
    }
    
    if analytical_grad is not None:
        diff = analytical_grad - numerical_grad
        relative_error = np.linalg.norm(diff) / (np.linalg.norm(analytical_grad) + 1e-12)
        
        result.update({
            'analytical_gradient': analytical_grad,
            'difference': diff,
            'absolute_error': np.linalg.norm(diff),
            'relative_error': relative_error,
            'check_passed': relative_error < tolerance
        })
    
    return result


def create_cost_function(
    function: Union[Callable[[NDArray[np.float64]], float],
                   Callable[[NDArray[np.float64]], Tuple[float, NDArray[np.float64]]]],
    gradient: Optional[Callable[[NDArray[np.float64]], NDArray[np.float64]]] = None,
    check_gradient: bool = True,
    manifold: Optional[Any] = None,
    test_point: Optional[NDArray[np.float64]] = None
) -> "CostFunction":
    """Create a CostFunction with optional gradient checking.
    
    Args:
        function: Cost function
        gradient: Gradient function (optional)
        check_gradient: Whether to verify gradient with finite differences
        manifold: Manifold for gradient checking
        test_point: Point for gradient checking
        
    Returns:
        CostFunction instance
    """
    from . import CostFunction
    
    cost_fn = CostFunction(function, gradient)
    
    if check_gradient and gradient is not None and manifold is not None:
        if test_point is None:
            test_point = manifold.random_point()
        
        check_result = gradient_check(manifold, function, test_point, gradient)
        
        if not check_result.get('check_passed', False):
            warnings.warn(
                f"Gradient check failed! Relative error: {check_result['relative_error']:.2e}. "
                "Consider verifying your gradient implementation."
            )
    
    return cost_fn


class OptimizationCallback:
    """Base class for optimization callbacks."""
    
    def __init__(self) -> None:
        self.iteration: int = 0
        self.values: List[float] = []
        self.gradient_norms: List[float] = []
        self.times: List[float] = []
    
    def __call__(self, iteration: int, value: float, gradient_norm: float) -> None:
        """Called at each optimization iteration."""
        import time
        
        self.iteration = iteration
        self.values.append(value)
        self.gradient_norms.append(gradient_norm)
        self.times.append(time.time())
        
        self.on_iteration(iteration, value, gradient_norm)
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float) -> None:
        """Override this method for custom behavior."""
        pass
    
    def reset(self) -> None:
        """Reset callback state."""
        self.iteration = 0
        self.values.clear()
        self.gradient_norms.clear()
        self.times.clear()


class ProgressCallback(OptimizationCallback):
    """Progress callback that prints optimization progress."""
    
    def __init__(self, print_every: int = 10, verbose: bool = True) -> None:
        super().__init__()
        self.print_every: int = print_every
        self.verbose: bool = verbose
        self.start_time: Optional[float] = None
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float) -> None:
        import time
        
        if self.start_time is None:
            self.start_time = time.time()
        
        if self.verbose and (iteration % self.print_every == 0 or iteration == 0):
            elapsed = time.time() - self.start_time
            print(f"Iteration {iteration:4d}: f = {value:.6e}, ||grad|| = {gradient_norm:.6e}, "
                  f"time = {elapsed:.2f}s")


class EarlyStoppingCallback(OptimizationCallback):
    """Early stopping based on improvement criteria."""
    
    def __init__(self, patience: int = 10, min_improvement: float = 1e-8) -> None:
        super().__init__()
        self.patience: int = patience
        self.min_improvement: float = min_improvement
        self.best_value: float = float('inf')
        self.wait: int = 0
        self.should_stop: bool = False
    
    def on_iteration(self, iteration: int, value: float, gradient_norm: float) -> None:
        if value < self.best_value - self.min_improvement:
            self.best_value = value
            self.wait = 0
        else:
            self.wait += 1
            
        if self.wait >= self.patience:
            self.should_stop = True
            print(f"Early stopping at iteration {iteration}: no improvement for {self.patience} iterations")


def plot_convergence(
    result: Dict[str, Any], 
    callback: Optional[OptimizationCallback] = None
) -> None:
    """Plot optimization convergence.
    
    Args:
        result: Optimization result dictionary
        callback: Callback with recorded values (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib not available. Install with: pip install matplotlib")
        return
    
    if callback is not None and hasattr(callback, 'values'):
        values = callback.values
        iterations = list(range(len(values)))
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Cost function values
        ax1.semilogy(iterations, values)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cost Function Value')
        ax1.set_title('Convergence')
        ax1.grid(True)
        
        # Gradient norms
        if hasattr(callback, 'gradient_norms') and callback.gradient_norms:
            ax2.semilogy(iterations, callback.gradient_norms)
            ax2.set_xlabel('Iteration')
            ax2.set_ylabel('Gradient Norm')
            ax2.set_title('Gradient Norm')
            ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
    else:
        print("No callback data available for plotting. "
              "Use ProgressCallback to record optimization history.")


def benchmark_optimizers(
    manifold: Any,  # Should be a Manifold instance
    cost_function: Union[Callable[[NDArray[np.float64]], float], "CostFunction"],
    initial_point: NDArray[np.float64],
    optimizers_config: Optional[Dict[str, Dict[str, Any]]] = None,
    max_iterations: int = 1000
) -> Dict[str, Any]:
    """Benchmark different optimizers on the same problem.
    
    Args:
        manifold: The manifold
        cost_function: Cost function to minimize
        initial_point: Starting point
        optimizers_config: Configuration for each optimizer
        max_iterations: Maximum iterations for each optimizer
        
    Returns:
        Dictionary with benchmarking results
    """
    import time
    from . import CostFunction
    
    if optimizers_config is None:
        optimizers_config = {
            'SGD': {'learning_rate': 0.01},
            'Adam': {'learning_rate': 0.001},
            'LBFGS': {'memory_size': 10},
        }
    
    cost_fn = CostFunction(cost_function) if callable(cost_function) else cost_function
    results = {}
    
    for opt_name, config in optimizers_config.items():
        print(f"Running {opt_name}...")
        
        # Use the same initial point for fair comparison
        x0 = initial_point.copy()
        
        start_time = time.time()
        
        try:
            result = optimize(
                manifold, cost_fn, x0, 
                optimizer=opt_name.lower(),
                max_iterations=max_iterations,
                **config
            )
            
            end_time = time.time()
            
            results[opt_name] = {
                'result': result,
                'time': end_time - start_time,
                'final_value': result['value'],
                'iterations': result['iterations'],
                'converged': result['converged']
            }
            
        except Exception as e:
            print(f"Error with {opt_name}: {e}")
            results[opt_name] = {'error': str(e)}
    
    return results