"""Type stubs for RiemannOpt optimizers."""

from typing import Tuple, Dict, Any, Literal, Optional, Union
import numpy as np
import numpy.typing as npt
from .manifolds import Sphere, Stiefel, Grassmann, SPD, Hyperbolic
from . import CostFunction, OptimizationResult

ManifoldType = Union[Sphere, Stiefel, Grassmann, SPD, Hyperbolic]

class SGD:
    """Riemannian Stochastic Gradient Descent optimizer."""
    
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
    """Riemannian Adam optimizer."""
    
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

class LBFGS:
    """Riemannian L-BFGS optimizer."""
    
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

class ConjugateGradient:
    """Riemannian Conjugate Gradient optimizer."""
    
    def __init__(
        self,
        variant: Literal["FletcherReeves", "PolakRibiere", "HestenesStiefel"] = "FletcherReeves",
        restart_threshold: float = 0.1
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

class TrustRegion:
    """Riemannian Trust Region optimizer."""
    
    def __init__(
        self,
        initial_radius: float = 1.0,
        max_radius: float = 10.0,
        eta: float = 0.1,
        radius_decrease_factor: float = 0.25,
        radius_increase_factor: float = 2.0,
        subproblem_solver: str = "CG",
        max_subproblem_iterations: Optional[int] = None
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

class Newton:
    """Riemannian Newton optimizer."""
    
    def __init__(
        self,
        regularization: float = 1e-6,
        max_cg_iterations: Optional[int] = None,
        cg_tolerance: float = 1e-5,
        line_search: bool = True
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