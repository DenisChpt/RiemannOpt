"""
Helper functions and high-level API for RiemannOpt.

This module provides convenient Python wrappers and utilities that
complement the core Rust implementations.
"""

import numpy as np
from typing import Callable, Dict, Any, Optional, Union, Tuple, List
import time
import warnings

from . import _riemannopt
# Import create_cost_function directly so the helpers can reuse it
from ._riemannopt import create_cost_function as _create_cost_function


def create_cost_function(cost_fn: Callable, 
                        gradient_fn: Optional[Callable] = None,
                        validate_gradient: bool = False,
                        dimension: Optional[Union[int, Tuple[int, int]]] = None) -> '_riemannopt.PyCostFunction':
    """
    Create a cost function wrapper for optimization.
    
    Parameters
    ----------
    cost_fn : callable
        Function that computes the cost. Should accept a numpy array and return a float,
        or return a tuple (cost, gradient) if gradient_fn is None.
    gradient_fn : callable, optional
        Function that computes the gradient. Should accept a numpy array and return 
        a numpy array of the same shape. If None, cost_fn should return (cost, gradient).
    validate_gradient : bool, default=False
        Whether to validate the gradient using finite differences.
    dimension : int or tuple of int, optional
        Problem dimension. For vector problems, pass an int. For matrix problems,
        pass a tuple (rows, cols). If None, will be inferred during first call.
    
    Returns
    -------
    cost_function : PyCostFunction
        Wrapped cost function that can be used with optimizers.
    
    Examples
    --------
    >>> import numpy as np
    >>> import riemannopt as ro
    >>> 
    >>> # Cost function only
    >>> def cost(x):
    ...     return 0.5 * np.sum(x**2)
    >>> cost_fn = ro.create_cost_function(cost)
    >>> 
    >>> # Cost and gradient
    >>> def cost_and_grad(x):
    ...     cost = 0.5 * np.sum(x**2)
    ...     grad = x
    ...     return cost, grad
    >>> cost_fn = ro.create_cost_function(cost_and_grad)
    >>> 
    >>> # Separate cost and gradient functions
    >>> def cost(x):
    ...     return 0.5 * np.sum(x**2)
    >>> def grad(x):
    ...     return x
    >>> cost_fn = ro.create_cost_function(cost, grad, validate_gradient=True)
    """
    return _create_cost_function(
        cost=cost_fn, 
        gradient=gradient_fn, 
        dimension=dimension,
        validate=validate_gradient
    )


def optimize(manifold,
             cost_function, 
             initial_point: np.ndarray,
             optimizer: str = "Adam",
             max_iterations: int = 1000,
             gradient_tolerance: float = 1e-6,
             callback: Optional[Callable] = None,
             **optimizer_kwargs) -> 'OptimizationResult':
    """
    High-level optimization interface.
    
    This function provides a unified interface for all Riemannian optimizers,
    automatically handling manifold types, cost function wrapping, and result conversion.
    
    Parameters
    ----------
    manifold : Manifold
        The manifold on which to optimize.
    cost_function : PyCostFunction or callable
        Cost function to minimize. If callable, will be wrapped automatically.
        The function should accept a numpy array and return either:
        - A scalar cost value
        - A tuple (cost, gradient)
    initial_point : array_like
        Starting point for optimization. Must be on the manifold.
    optimizer : str, default="Adam"
        Name of the optimizer to use. Options: "SGD", "Adam", "LBFGS", 
        "ConjugateGradient", "TrustRegion", "Newton".
    max_iterations : int, default=1000
        Maximum number of optimization iterations.
    gradient_tolerance : float, default=1e-6
        Tolerance for gradient norm convergence criterion.
    callback : callable, optional
        Callback function called at each iteration. Should accept:
        (iteration, point, cost, grad_norm) and return bool (continue?).
    **optimizer_kwargs
        Additional keyword arguments for the optimizer.
    
    Returns
    -------
    result : OptimizationResult
        Optimization result object with attributes:
        - point: Final point (numpy array)
        - value/cost: Final cost value
        - gradient_norm: Final gradient norm
        - converged: Whether the optimization converged
        - iterations: Number of iterations performed
        - time_seconds: Total optimization time
        - termination_reason: Why the optimization stopped
    
    Examples
    --------
    >>> import numpy as np
    >>> import riemannopt as ro
    >>> 
    >>> # Create manifold and cost function
    >>> sphere = ro.manifolds.Sphere(10)
    >>> def cost(x):
    ...     return 0.5 * np.sum(x**2)
    >>> 
    >>> # Optimize
    >>> x0 = sphere.random_point()
    >>> result = ro.optimize(sphere, cost, x0, optimizer="Adam", learning_rate=0.01)
    >>> print(f"Converged: {result.converged}, Final cost: {result.cost:.6f}")
    >>> 
    >>> # With callback
    >>> def callback(iter, x, cost, grad_norm):
    ...     if iter % 10 == 0:
    ...         print(f"Iter {iter}: cost={cost:.4f}")
    ...     return True  # Continue
    >>> 
    >>> result = ro.optimize(sphere, cost, x0, callback=callback)
    """
    # Convert cost function if needed
    if not hasattr(cost_function, 'cost'):
        # Infer dimension from manifold and initial point
        if initial_point.ndim == 1:
            dimension = len(initial_point)
        elif initial_point.ndim == 2:
            dimension = initial_point.shape
        else:
            raise ValueError(f"Unsupported point dimension: {initial_point.ndim}")
        
        # Create cost function wrapper (will auto-detect if it returns gradient)
        cost_function = create_cost_function(cost_function, dimension=dimension)
    
    # Validate initial point is on manifold
    if hasattr(manifold, 'contains'):
        if not manifold.contains(initial_point):
            warnings.warn("Initial point is not on the manifold. Projecting...", UserWarning)
            if hasattr(manifold, 'project'):
                initial_point = manifold.project(initial_point)
            else:
                raise ValueError("Initial point is not on the manifold and no projection available")
    
    # Create optimizer (handle case-insensitive names)
    optimizer_map = {
        'sgd': 'SGD',
        'adam': 'Adam',
        'lbfgs': 'LBFGS',
        'conjugategradient': 'ConjugateGradient',
        'cg': 'ConjugateGradient',
        'trustregion': 'TrustRegion',
        'newton': 'Newton',
        'riemanniannewton': 'Newton'
    }
    
    optimizer_name = optimizer_map.get(optimizer.lower(), optimizer)
    
    try:
        optimizer_class = getattr(_riemannopt.optimizers, optimizer_name)
    except AttributeError:
        available = list(optimizer_map.values())
        raise ValueError(f"Unknown optimizer: {optimizer}. Available: {', '.join(available)}")
    
    # Set default hyperparameters based on optimizer type
    defaults = {
        'SGD': {'learning_rate': 0.01},
        'Adam': {'learning_rate': 0.001, 'beta1': 0.9, 'beta2': 0.999},
        'LBFGS': {'memory_size': 10},
        'ConjugateGradient': {},
        'TrustRegion': {'initial_radius': 1.0},
        'Newton': {'regularization': 1e-6}
    }
    
    # Merge defaults with user kwargs
    final_kwargs = defaults.get(optimizer_name, {}).copy()
    final_kwargs.update(optimizer_kwargs)
    
    opt = optimizer_class(**final_kwargs)
    
    # Prepare optimization parameters
    opt_params = {
        'cost_function': cost_function,
        'initial_point': initial_point,
        'max_iterations': max_iterations,
    }
    
    # Check if the optimizer method accepts gradient_tolerance
    # Only add it if the method signature includes it
    if gradient_tolerance is not None and optimizer_name in ['Adam', 'LBFGS', 'ConjugateGradient', 'TrustRegion', 'Newton']:
        opt_params['gradient_tolerance'] = gradient_tolerance
    
    # Handle callbacks if provided
    callback_wrapper = None
    if callback is not None:
        # Wrap Python callback for Rust
        callback_wrapper = CallbackWrapper(callback)
        opt_params['callback'] = callback_wrapper
    
    # Determine manifold type and call appropriate method
    manifold_name = type(manifold).__name__.lower()
    
    # Try to find the appropriate optimize method
    method_name = f"optimize_{manifold_name}"
    if hasattr(opt, method_name):
        method = getattr(opt, method_name)
        # Add manifold parameter based on manifold type
        opt_params[manifold_name] = manifold
        result = method(**opt_params)
    else:
        # Try generic methods based on point type
        if initial_point.ndim == 1:
            # Vector manifold - assume Sphere for now
            if hasattr(opt, 'optimize_sphere'):
                opt_params['sphere'] = manifold
                result = opt.optimize_sphere(**opt_params)
            else:
                raise ValueError(f"Optimizer {optimizer_name} does not support vector manifolds")
        elif initial_point.ndim == 2:
            # Matrix manifold - try Stiefel first
            if hasattr(opt, 'optimize_stiefel'):
                opt_params['stiefel'] = manifold
                result = opt.optimize_stiefel(**opt_params)
            elif hasattr(opt, 'optimize_matrix'):
                opt_params['matrix_manifold'] = manifold
                result = opt.optimize_matrix(**opt_params)
            else:
                raise ValueError(f"Optimizer {optimizer_name} does not support matrix manifolds")
        else:
            raise ValueError(f"Unsupported point dimension: {initial_point.ndim}")
    
    # The result should be an OptimizationResult object from Rust
    # Add callback history if available
    if callback_wrapper and hasattr(callback_wrapper, 'history'):
        if hasattr(result, 'history'):
            result.history = callback_wrapper.history
    
    return result


class CallbackWrapper:
    """Wrapper to adapt Python callbacks for Rust optimizers."""
    
    def __init__(self, callback):
        self.callback = callback
        self.history = []
        self.should_stop = False
    
    def __call__(self, iteration, point, cost, grad_norm):
        """Called at each iteration by the optimizer."""
        # Store history
        self.history.append({
            'iteration': iteration,
            'cost': cost,
            'grad_norm': grad_norm
        })
        
        # Call user callback
        try:
            result = self.callback(iteration, point, cost, grad_norm)
            # If callback returns False, signal to stop
            if result is False:
                self.should_stop = True
                return False
            return True
        except Exception as e:
            warnings.warn(f"Callback error at iteration {iteration}: {e}", RuntimeWarning)
            return True  # Continue optimization despite callback error


def gradient_check(cost_function, 
                  point: np.ndarray,
                  epsilon: float = 1e-8,
                  tolerance: float = 1e-5) -> Tuple[bool, float]:
    """
    Check gradient accuracy using finite differences.
    
    Parameters
    ----------
    cost_function : PyCostFunction
        Cost function with gradient.
    point : array_like
        Point at which to check the gradient.
    epsilon : float, default=1e-8
        Step size for finite differences.
    tolerance : float, default=1e-5
        Tolerance for gradient accuracy.
    
    Returns
    -------
    is_accurate : bool
        True if gradient is accurate within tolerance.
    max_error : float
        Maximum relative error found.
    """
    # Get analytical gradient
    analytical_grad = cost_function.gradient(point)
    
    # Compute finite difference gradient
    finite_diff_grad = np.zeros_like(point)
    
    for i in range(len(point.flat)):
        point_plus = point.copy()
        point_minus = point.copy()
        
        point_plus.flat[i] += epsilon
        point_minus.flat[i] -= epsilon
        
        cost_plus = cost_function.cost(point_plus)
        cost_minus = cost_function.cost(point_minus)
        
        finite_diff_grad.flat[i] = (cost_plus - cost_minus) / (2 * epsilon)
    
    # Compute relative error
    diff = np.abs(analytical_grad - finite_diff_grad)
    scale = np.maximum(np.abs(analytical_grad), np.abs(finite_diff_grad))
    relative_error = np.divide(diff, scale, out=np.zeros_like(diff), where=scale!=0)
    
    max_error = np.max(relative_error)
    is_accurate = max_error < tolerance
    
    return is_accurate, max_error


# Callback classes for optimization
class OptimizationCallback:
    """Base class for optimization callbacks."""
    
    def __init__(self):
        self.history = []
    
    def __call__(self, iteration: int, point: np.ndarray, cost: float, grad_norm: float) -> bool:
        """
        Called at each optimization iteration.
        
        Parameters
        ----------
        iteration : int
            Current iteration number.
        point : array_like
            Current point.
        cost : float
            Current cost value.
        grad_norm : float
            Current gradient norm.
        
        Returns
        -------
        continue_optimization : bool
            True to continue optimization, False to stop.
        """
        self.history.append({
            'iteration': iteration,
            'point': point.copy(),
            'cost': cost,
            'grad_norm': grad_norm
        })
        return self._callback(iteration, point, cost, grad_norm)
    
    def _callback(self, iteration: int, point: np.ndarray, cost: float, grad_norm: float) -> bool:
        """Override this method in subclasses."""
        return True


class ProgressCallback(OptimizationCallback):
    """Callback that prints optimization progress."""
    
    def __init__(self, print_every: int = 10):
        super().__init__()
        self.print_every = print_every
        self.start_time = None
    
    def _callback(self, iteration: int, point: np.ndarray, cost: float, grad_norm: float) -> bool:
        if self.start_time is None:
            self.start_time = time.time()
        
        if iteration % self.print_every == 0:
            elapsed = time.time() - self.start_time
            print(f"Iter {iteration:4d}: cost = {cost:.6e}, grad_norm = {grad_norm:.3e}, time = {elapsed:.2f}s")
        
        return True


class EarlyStoppingCallback(OptimizationCallback):
    """Callback that implements early stopping based on cost improvement."""
    
    def __init__(self, patience: int = 10, min_improvement: float = 1e-8):
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_cost = float('inf')
        self.patience_counter = 0
    
    def _callback(self, iteration: int, point: np.ndarray, cost: float, grad_norm: float) -> bool:
        if cost < self.best_cost - self.min_improvement:
            self.best_cost = cost
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.patience:
            print(f"Early stopping at iteration {iteration}: no improvement for {self.patience} iterations")
            return False
        
        return True


def plot_convergence(history: List[Dict], 
                    figsize: Tuple[int, int] = (12, 4),
                    log_scale: bool = True):
    """
    Plot optimization convergence history.
    
    Parameters
    ----------
    history : list of dict
        Optimization history from a callback.
    figsize : tuple, default=(12, 4)
        Figure size.
    log_scale : bool, default=True
        Whether to use log scale for y-axis.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plotting")
    
    iterations = [h['iteration'] for h in history]
    costs = [h['cost'] for h in history]
    grad_norms = [h['grad_norm'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot cost
    ax1.plot(iterations, costs, 'b-', linewidth=2)
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cost')
    ax1.set_title('Cost vs Iteration')
    ax1.grid(True, alpha=0.3)
    if log_scale and min(costs) > 0:
        ax1.set_yscale('log')
    
    # Plot gradient norm
    ax2.plot(iterations, grad_norms, 'r-', linewidth=2)
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Gradient Norm')
    ax2.set_title('Gradient Norm vs Iteration')
    ax2.grid(True, alpha=0.3)
    if log_scale and min(grad_norms) > 0:
        ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.show()


def benchmark_optimizers(manifold,
                        cost_function,
                        initial_point: np.ndarray,
                        optimizers: Optional[List[str]] = None,
                        max_iterations: int = 100,
                        n_runs: int = 5) -> Dict[str, Dict]:
    """
    Benchmark multiple optimizers on the same problem.
    
    Parameters
    ----------
    manifold : Manifold
        The manifold for optimization.
    cost_function : PyCostFunction
        Cost function to minimize.
    initial_point : array_like
        Starting point (same for all optimizers).
    optimizers : list of str, optional
        List of optimizer names. If None, uses ["SGD", "Adam", "LBFGS"].
    max_iterations : int, default=100
        Maximum iterations for each run.
    n_runs : int, default=5
        Number of runs for each optimizer (for statistical robustness).
    
    Returns
    -------
    results : dict
        Benchmark results for each optimizer.
    """
    if optimizers is None:
        optimizers = ["SGD", "Adam", "LBFGS"]
    
    results = {}
    
    for opt_name in optimizers:
        print(f"Benchmarking {opt_name}...")
        
        run_results = []
        run_times = []
        
        for run in range(n_runs):
            start_time = time.time()
            
            try:
                result = optimize(
                    manifold, cost_function, initial_point.copy(),
                    optimizer=opt_name,
                    max_iterations=max_iterations
                )
                
                end_time = time.time()
                
                run_results.append(result)
                run_times.append(end_time - start_time)
                
            except Exception as e:
                print(f"  Run {run+1} failed: {e}")
                continue
        
        if run_results:
            # Compute statistics
            final_costs = [r['cost'] for r in run_results]
            iterations = [r['iterations'] for r in run_results]
            converged = [r['converged'] for r in run_results]
            
            results[opt_name] = {
                'mean_cost': np.mean(final_costs),
                'std_cost': np.std(final_costs),
                'mean_iterations': np.mean(iterations),
                'std_iterations': np.std(iterations),
                'mean_time': np.mean(run_times),
                'std_time': np.std(run_times),
                'convergence_rate': np.mean(converged),
                'n_runs': len(run_results)
            }
        else:
            results[opt_name] = {'error': 'All runs failed'}
    
    # Print summary
    print("\nBenchmark Results:")
    print("-" * 80)
    print(f"{'Optimizer':<15} {'Mean Cost':<12} {'Mean Iter':<10} {'Mean Time':<10} {'Conv Rate':<10}")
    print("-" * 80)
    
    for opt_name, result in results.items():
        if 'error' in result:
            print(f"{opt_name:<15} {'FAILED':<12}")
        else:
            print(f"{opt_name:<15} {result['mean_cost']:<12.6f} "
                  f"{result['mean_iterations']:<10.1f} {result['mean_time']:<10.3f} "
                  f"{result['convergence_rate']:<10.1%}")
    
    return results


# Version information
__all__ = [
    'create_cost_function',
    'optimize', 
    'gradient_check',
    'OptimizationCallback',
    'ProgressCallback',
    'EarlyStoppingCallback',
    'plot_convergence',
    'benchmark_optimizers'
]