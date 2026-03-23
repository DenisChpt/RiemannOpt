"""High-level convenience API for RiemannOpt.

Provides Pythonic wrappers around the Rust optimization engine.
"""

from __future__ import annotations

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from . import _riemannopt
from ._riemannopt import create_cost_function as _create_cost_function


# ---------------------------------------------------------------------------
# Cost function helper
# ---------------------------------------------------------------------------

def create_cost_function(
    cost_fn: Callable[..., Any],
    gradient_fn: Optional[Callable[..., Any]] = None,
    validate_gradient: bool = False,
    dimension: Optional[Union[int, Tuple[int, int]]] = None,
) -> _riemannopt.CostFunction:
    """Create a cost function wrapper for optimization.

    Parameters
    ----------
    cost_fn : callable
        Function computing the cost.  Accepts a numpy array, returns either a
        float **or** a ``(float, ndarray)`` tuple when *gradient_fn* is ``None``.
    gradient_fn : callable, optional
        Explicit gradient function ``point -> ndarray``.
    validate_gradient : bool
        Compare the analytical gradient against finite differences.
    dimension : int or (int, int), optional
        Problem dimension; inferred from the first call when omitted.

    Returns
    -------
    CostFunction
        Wrapped cost function usable by any optimizer.
    """
    return _create_cost_function(
        cost=cost_fn,
        gradient=gradient_fn,
        dimension=dimension,
        validate=validate_gradient,
    )


# ---------------------------------------------------------------------------
# High-level optimize
# ---------------------------------------------------------------------------

def optimize(
    manifold: Any,
    cost_function: Any,
    initial_point: np.ndarray,
    optimizer: str = "Adam",
    max_iterations: int = 1000,
    gradient_tolerance: float = 1e-6,
    callback: Optional[Callable[..., Any]] = None,
    **optimizer_kwargs: Any,
) -> Any:
    """Minimize a cost function on a Riemannian manifold.

    Parameters
    ----------
    manifold : Manifold
        Target manifold (e.g. ``riemannopt.manifolds.Sphere(n)``).
    cost_function : CostFunction or callable
        Objective to minimize.  Plain callables are wrapped automatically.
    initial_point : array_like
        Starting point (must lie on *manifold*).
    optimizer : str
        One of ``"SGD"``, ``"Adam"``, ``"LBFGS"``, ``"ConjugateGradient"``
        (or ``"CG"``), ``"TrustRegion"``, ``"Newton"``, ``"NaturalGradient"``.
    max_iterations : int
        Maximum number of iterations.
    gradient_tolerance : float
        Convergence threshold on the Riemannian gradient norm.
    callback : callable, optional
        ``(iteration, point, cost, grad_norm) -> bool``.  Return ``False``
        to stop early.
    **optimizer_kwargs
        Forwarded to the optimizer constructor (e.g. ``learning_rate=0.01``).

    Returns
    -------
    OptimizationResult
        Result object with ``.point``, ``.cost``, ``.converged``, etc.
    """
    # Wrap raw callables
    if not isinstance(cost_function, _riemannopt.CostFunction):
        if initial_point.ndim == 1:
            dim: Any = len(initial_point)
        elif initial_point.ndim == 2:
            dim = initial_point.shape
        else:
            raise ValueError(f"Unsupported point dimension: {initial_point.ndim}")
        cost_function = create_cost_function(cost_function, dimension=dim)

    # Auto-project if off manifold
    if hasattr(manifold, "contains") and not manifold.contains(initial_point):
        warnings.warn(
            "Initial point is not on the manifold. Projecting.", UserWarning, stacklevel=2
        )
        initial_point = manifold.project(initial_point)

    # Resolve optimizer name
    _ALIASES = {
        "sgd": "SGD",
        "adam": "Adam",
        "lbfgs": "LBFGS",
        "conjugategradient": "ConjugateGradient",
        "cg": "ConjugateGradient",
        "trustregion": "TrustRegion",
        "newton": "Newton",
        "naturalgradient": "NaturalGradient",
    }
    optimizer_name = _ALIASES.get(optimizer.lower(), optimizer)

    try:
        optimizer_class = getattr(_riemannopt.optimizers, optimizer_name)
    except AttributeError:
        avail = sorted(set(_ALIASES.values()))
        raise ValueError(
            f"Unknown optimizer: {optimizer!r}. Available: {', '.join(avail)}"
        ) from None

    opt = optimizer_class(**optimizer_kwargs)

    # Wrap plain callback
    cb = CallbackWrapper(callback) if callback is not None else None

    return opt.optimize(
        cost_function=cost_function,
        manifold=manifold,
        initial_point=initial_point,
        max_iterations=max_iterations,
        gradient_tolerance=gradient_tolerance,
        callback=cb,
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class CallbackWrapper:
    """Adapts a plain ``(iter, point, cost, grad_norm) -> bool`` callable for
    the Rust optimizer callback protocol."""

    def __init__(self, fn: Callable[..., Any]) -> None:
        self._fn = fn
        self.history: list[dict[str, Any]] = []

    def __call__(
        self, iteration: int, point: np.ndarray, cost: float, grad_norm: float
    ) -> bool:
        self.history.append(
            {"iteration": iteration, "cost": cost, "grad_norm": grad_norm}
        )
        try:
            result = self._fn(iteration, point, cost, grad_norm)
            return result is not False
        except Exception as exc:
            warnings.warn(
                f"Callback error at iteration {iteration}: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            return True


class OptimizationCallback:
    """Base class for optimization callbacks with automatic history tracking."""

    def __init__(self) -> None:
        self.history: list[dict[str, Any]] = []

    def __call__(
        self, iteration: int, point: np.ndarray, cost: float, grad_norm: float
    ) -> bool:
        self.history.append(
            {
                "iteration": iteration,
                "point": point.copy(),
                "cost": cost,
                "grad_norm": grad_norm,
            }
        )
        return self.on_step(iteration, point, cost, grad_norm)

    def on_step(
        self, iteration: int, point: np.ndarray, cost: float, grad_norm: float
    ) -> bool:
        """Override in subclasses.  Return ``False`` to stop."""
        return True


class ProgressCallback(OptimizationCallback):
    """Print optimization progress every *n* iterations."""

    def __init__(self, every: int = 10) -> None:
        super().__init__()
        self.every = every
        self._t0: Optional[float] = None

    def on_step(
        self, iteration: int, point: np.ndarray, cost: float, grad_norm: float
    ) -> bool:
        if self._t0 is None:
            self._t0 = time.monotonic()
        if iteration % self.every == 0:
            elapsed = time.monotonic() - self._t0
            print(
                f"iter {iteration:5d}  cost {cost: .6e}  "
                f"|grad| {grad_norm:.3e}  [{elapsed:.2f}s]"
            )
        return True


class EarlyStoppingCallback(OptimizationCallback):
    """Stop when cost improvement stalls for *patience* iterations."""

    def __init__(self, patience: int = 10, min_improvement: float = 1e-8) -> None:
        super().__init__()
        self.patience = patience
        self.min_improvement = min_improvement
        self._best = float("inf")
        self._wait = 0

    def on_step(
        self, iteration: int, point: np.ndarray, cost: float, grad_norm: float
    ) -> bool:
        if cost < self._best - self.min_improvement:
            self._best = cost
            self._wait = 0
        else:
            self._wait += 1
        if self._wait >= self.patience:
            return False
        return True


# ---------------------------------------------------------------------------
# Gradient check
# ---------------------------------------------------------------------------

def gradient_check(
    cost_function: _riemannopt.CostFunction,
    point: np.ndarray,
    epsilon: float = 1e-7,
    tolerance: float = 1e-4,
) -> Tuple[bool, float]:
    """Compare the analytical gradient with a central-difference approximation.

    Parameters
    ----------
    cost_function : CostFunction
        Must expose ``.cost()`` and ``.gradient()``.
    point : ndarray
        Evaluation point.
    epsilon : float
        Finite-difference step.
    tolerance : float
        Maximum acceptable relative error.

    Returns
    -------
    ok : bool
    max_relative_error : float
    """
    grad_a = np.asarray(cost_function.gradient(point), dtype=float)
    grad_fd = np.zeros_like(grad_a)

    for idx in range(grad_a.size):
        p_plus = point.copy()
        p_minus = point.copy()
        p_plus.flat[idx] += epsilon
        p_minus.flat[idx] -= epsilon
        grad_fd.flat[idx] = (
            cost_function.cost(p_plus) - cost_function.cost(p_minus)
        ) / (2.0 * epsilon)

    diff = np.abs(grad_a - grad_fd)
    scale = np.maximum(np.abs(grad_a), np.abs(grad_fd))
    rel = np.divide(diff, scale, out=np.zeros_like(diff), where=scale > 0)
    max_err = float(np.max(rel))
    return max_err < tolerance, max_err


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_convergence(
    history: List[Dict[str, Any]],
    figsize: Tuple[int, int] = (12, 4),
    log_scale: bool = True,
) -> None:
    """Plot cost and gradient norm from a callback history list."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib is required for plot_convergence") from None

    iters = [h["iteration"] for h in history]
    costs = [h["cost"] for h in history]
    gnorms = [h.get("grad_norm", 0.0) for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    ax1.plot(iters, costs, "b-", lw=2)
    ax1.set(xlabel="Iteration", ylabel="Cost", title="Cost")
    ax1.grid(True, alpha=0.3)
    if log_scale and min(costs) > 0:
        ax1.set_yscale("log")

    ax2.plot(iters, gnorms, "r-", lw=2)
    ax2.set(xlabel="Iteration", ylabel="|grad|", title="Gradient norm")
    ax2.grid(True, alpha=0.3)
    if log_scale and gnorms and min(gnorms) > 0:
        ax2.set_yscale("log")

    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Benchmarking
# ---------------------------------------------------------------------------

def benchmark_optimizers(
    manifold: Any,
    cost_function: Any,
    initial_point: np.ndarray,
    optimizers: Optional[List[str]] = None,
    max_iterations: int = 200,
    n_runs: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """Run several optimizers on the same problem and compare."""
    if optimizers is None:
        optimizers = ["SGD", "Adam", "ConjugateGradient"]

    results: Dict[str, Dict[str, Any]] = {}
    for name in optimizers:
        times, costs_list, iters_list = [], [], []
        for _ in range(n_runs):
            t0 = time.monotonic()
            try:
                res = optimize(
                    manifold,
                    cost_function,
                    initial_point.copy(),
                    optimizer=name,
                    max_iterations=max_iterations,
                )
                costs_list.append(res.cost)
                iters_list.append(res.iterations)
            except Exception:
                continue
            times.append(time.monotonic() - t0)

        if costs_list:
            results[name] = {
                "mean_cost": float(np.mean(costs_list)),
                "mean_iters": float(np.mean(iters_list)),
                "mean_time": float(np.mean(times)),
                "n_runs": len(costs_list),
            }
        else:
            results[name] = {"error": "all runs failed"}

    return results


__all__ = [
    "create_cost_function",
    "optimize",
    "gradient_check",
    "CallbackWrapper",
    "OptimizationCallback",
    "ProgressCallback",
    "EarlyStoppingCallback",
    "plot_convergence",
    "benchmark_optimizers",
]
