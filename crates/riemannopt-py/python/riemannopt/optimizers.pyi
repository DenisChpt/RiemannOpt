"""Type stubs for RiemannOpt optimizers.

This module provides type hints for all optimizer classes in the RiemannOpt library.
All optimizers now support a unified `optimize` method that works with any manifold.
"""

from typing import Dict, Any, Literal, Optional, Union, Callable, Protocol
import numpy as np
import numpy.typing as npt
from .manifolds import Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, PSDCone, ProductManifold

# Type aliases
ManifoldType = Union[Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, PSDCone, ProductManifold]
PointType = Union[npt.NDArray[np.float64], tuple[npt.NDArray[np.float64], ...]]
CallbackType = Optional[Callable[[int, float, float], bool]]

class CostFunction(Protocol):
    """Protocol for cost functions."""
    
    def cost(self, point: npt.NDArray[np.float64]) -> float: ...
    def gradient(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    def cost_and_gradient(self, point: npt.NDArray[np.float64]) -> tuple[float, npt.NDArray[np.float64]]: ...
    
    @property
    def eval_counts(self) -> Dict[str, int]: ...
    
    def reset_counts(self) -> None: ...

class OptimizationResult:
    """Result from optimization.
    
    Attributes
    ----------
    point : ndarray or tuple of ndarray
        Final point found by the optimizer
    value : float
        Final objective function value
    gradient_norm : float or None
        Norm of the gradient at the final point
    converged : bool
        Whether the optimization converged successfully
    iterations : int
        Number of iterations performed
    function_evals : int
        Number of function evaluations
    gradient_evals : int
        Number of gradient evaluations
    time_seconds : float
        Total optimization time in seconds
    termination_reason : str
        Reason for termination
    history : dict or None
        Optional optimization history
    """
    
    point: PointType
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
    def x(self) -> PointType:
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

class SGD:
    """Riemannian Stochastic Gradient Descent optimizer.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient descent
    momentum : float, default=0.0
        Momentum coefficient (0 for no momentum)
    nesterov : bool, default=False
        Whether to use Nesterov accelerated gradient
    gradient_clip : float or None, default=None
        Maximum gradient norm for clipping
    line_search : bool, default=False
        Whether to use line search
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        nesterov: bool = False,
        gradient_clip: Optional[float] = None,
        line_search: bool = False
    ) -> None: ...
    
    @property
    def learning_rate(self) -> float: ...
    
    @property
    def momentum(self) -> float: ...
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        ...
    
    def optimize(
        self,
        cost_function: CostFunction,
        manifold: ManifoldType,
        initial_point: PointType,
        max_iterations: int,
        gradient_tolerance: Optional[float] = None,
        callback: CallbackType = None,
        target_value: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> OptimizationResult:
        """Optimize on any Riemannian manifold.
        
        Parameters
        ----------
        cost_function : CostFunction
            The cost function to minimize
        manifold : Manifold
            The manifold to optimize on
        initial_point : ndarray or tuple of ndarray
            Starting point on the manifold
        max_iterations : int
            Maximum number of iterations
        gradient_tolerance : float, optional
            Tolerance for gradient norm convergence
        callback : callable, optional
            Callback function called at each iteration
        target_value : float, optional
            Target value to reach
        max_time : float, optional
            Maximum time in seconds
            
        Returns
        -------
        OptimizationResult
            Result object with optimization details
        """
        ...

class Adam:
    """Riemannian Adam optimizer.
    
    Adaptive Moment Estimation (Adam) generalized to Riemannian manifolds.
    
    Parameters
    ----------
    learning_rate : float, default=0.001
        Step size
    beta1 : float, default=0.9
        Exponential decay rate for first moment
    beta2 : float, default=0.999
        Exponential decay rate for second moment
    epsilon : float, default=1e-8
        Small constant for numerical stability
    amsgrad : bool, default=False
        Whether to use AMSGrad variant
    """
    
    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        amsgrad: bool = False
    ) -> None: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...
    
    def optimize(
        self,
        cost_function: CostFunction,
        manifold: ManifoldType,
        initial_point: PointType,
        max_iterations: int,
        gradient_tolerance: Optional[float] = None,
        callback: CallbackType = None,
        target_value: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> OptimizationResult:
        """Optimize using Riemannian Adam."""
        ...

class LBFGS:
    """Riemannian L-BFGS optimizer.
    
    Limited-memory BFGS for Riemannian optimization.
    
    Parameters
    ----------
    memory_size : int, default=10
        Number of vector pairs to store
    max_line_search_iterations : int, default=20
        Maximum iterations for line search
    c1 : float, default=1e-4
        Armijo condition constant
    c2 : float, default=0.9
        Wolfe condition constant
    initial_step_size : float, default=1.0
        Initial step size for line search
    """
    
    def __init__(
        self,
        memory_size: int = 10,
        max_line_search_iterations: int = 20,
        c1: float = 1e-4,
        c2: float = 0.9,
        initial_step_size: float = 1.0
    ) -> None: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...
    
    def optimize(
        self,
        cost_function: CostFunction,
        manifold: ManifoldType,
        initial_point: PointType,
        max_iterations: int,
        gradient_tolerance: Optional[float] = None,
        callback: CallbackType = None,
        target_value: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> OptimizationResult:
        """Optimize using Riemannian L-BFGS."""
        ...

class ConjugateGradient:
    """Riemannian Conjugate Gradient optimizer.
    
    Parameters
    ----------
    method : str, default="FletcherReeves"
        CG variant: "FletcherReeves", "PolakRibiere", "HestenesStiefel", "DaiYuan"
    restart_threshold : float, default=0.1
        Threshold for restarting CG direction
    max_line_search_iterations : int, default=20
        Maximum iterations for line search
    c1 : float, default=1e-4
        Armijo condition constant
    c2 : float, default=0.1
        Wolfe condition constant
    """
    
    def __init__(
        self,
        method: Literal["FletcherReeves", "PolakRibiere", "HestenesStiefel", "DaiYuan"] = "FletcherReeves",
        restart_threshold: float = 0.1,
        max_line_search_iterations: int = 20,
        c1: float = 1e-4,
        c2: float = 0.1
    ) -> None: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...
    
    def optimize(
        self,
        cost_function: CostFunction,
        manifold: ManifoldType,
        initial_point: PointType,
        max_iterations: int,
        gradient_tolerance: Optional[float] = None,
        callback: CallbackType = None,
        target_value: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> OptimizationResult:
        """Optimize using Riemannian Conjugate Gradient."""
        ...

class TrustRegion:
    """Riemannian Trust Region optimizer.
    
    Parameters
    ----------
    initial_radius : float, default=1.0
        Initial trust region radius
    max_radius : float, default=10.0
        Maximum trust region radius
    eta : float, default=0.1
        Threshold for accepting a step
    radius_decrease_factor : float, default=0.25
        Factor for decreasing radius
    radius_increase_factor : float, default=2.0
        Factor for increasing radius
    subproblem_solver : str, default="CG"
        Method for solving subproblem: "CG" or "ExactQR"
    max_subproblem_iterations : int or None, default=None
        Maximum iterations for subproblem solver
    """
    
    def __init__(
        self,
        initial_radius: float = 1.0,
        max_radius: float = 10.0,
        eta: float = 0.1,
        radius_decrease_factor: float = 0.25,
        radius_increase_factor: float = 2.0,
        subproblem_solver: Literal["CG", "ExactQR"] = "CG",
        max_subproblem_iterations: Optional[int] = None
    ) -> None: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...
    
    def optimize(
        self,
        cost_function: CostFunction,
        manifold: ManifoldType,
        initial_point: PointType,
        max_iterations: int,
        gradient_tolerance: Optional[float] = None,
        callback: CallbackType = None,
        target_value: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> OptimizationResult:
        """Optimize using Riemannian Trust Region.
        
        Note: Requires cost function to implement Hessian-vector products.
        """
        ...

class Newton:
    """Riemannian Newton optimizer.
    
    Parameters
    ----------
    regularization : float, default=1e-6
        Regularization for Hessian
    max_cg_iterations : int or None, default=None
        Maximum CG iterations for solving Newton system
    cg_tolerance : float, default=1e-5
        Tolerance for CG solver
    line_search : bool, default=True
        Whether to use line search
    max_line_search_iterations : int, default=20
        Maximum iterations for line search
    """
    
    def __init__(
        self,
        regularization: float = 1e-6,
        max_cg_iterations: Optional[int] = None,
        cg_tolerance: float = 1e-5,
        line_search: bool = True,
        max_line_search_iterations: int = 20
    ) -> None: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...
    
    def optimize(
        self,
        cost_function: CostFunction,
        manifold: ManifoldType,
        initial_point: PointType,
        max_iterations: int,
        gradient_tolerance: Optional[float] = None,
        callback: CallbackType = None,
        target_value: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> OptimizationResult:
        """Optimize using Riemannian Newton method.
        
        Note: Requires cost function to implement Hessian-vector products.
        """
        ...

class NaturalGradient:
    """Natural Gradient optimizer for Riemannian manifolds.
    
    Uses the Fisher information matrix as a preconditioner.
    
    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size
    momentum : float, default=0.0
        Momentum coefficient
    fisher_subsample : int or None, default=None
        Number of samples for Fisher approximation
    damping : float, default=1e-4
        Damping factor for Fisher matrix
    """
    
    def __init__(
        self,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        fisher_subsample: Optional[int] = None,
        damping: float = 1e-4
    ) -> None: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...
    
    def optimize(
        self,
        cost_function: CostFunction,
        manifold: ManifoldType,
        initial_point: PointType,
        max_iterations: int,
        gradient_tolerance: Optional[float] = None,
        callback: CallbackType = None,
        target_value: Optional[float] = None,
        max_time: Optional[float] = None
    ) -> OptimizationResult:
        """Optimize using Natural Gradient."""
        ...

# Deprecated optimizer-specific methods (for backward compatibility)
# These will be removed in a future version

class _DeprecatedOptimizerMethods:
    """Base class with deprecated manifold-specific methods."""
    
    def optimize_sphere(
        self,
        cost_function: CostFunction,
        sphere: Sphere,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult:
        """Deprecated: Use optimize() instead."""
        ...
    
    def optimize_stiefel(
        self,
        cost_function: CostFunction,
        stiefel: Stiefel,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult:
        """Deprecated: Use optimize() instead."""
        ...
    
    def optimize_grassmann(
        self,
        cost_function: CostFunction,
        grassmann: Grassmann,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult:
        """Deprecated: Use optimize() instead."""
        ...
    
    def optimize_spd(
        self,
        cost_function: CostFunction,
        spd: SPD,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult:
        """Deprecated: Use optimize() instead."""
        ...
    
    def optimize_hyperbolic(
        self,
        cost_function: CostFunction,
        hyperbolic: Hyperbolic,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult:
        """Deprecated: Use optimize() instead."""
        ...
    
    def optimize_oblique(
        self,
        cost_function: CostFunction,
        oblique: Oblique,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult:
        """Deprecated: Use optimize() instead."""
        ...
    
    def optimize_psd_cone(
        self,
        cost_function: CostFunction,
        psd_cone: PSDCone,
        initial_point: npt.NDArray[np.float64],
        max_iterations: int,
        gradient_tolerance: Optional[float] = None
    ) -> OptimizationResult:
        """Deprecated: Use optimize() instead."""
        ...