"""Type stubs for RiemannOpt optimizers."""

from typing import Tuple, Dict, Any, Literal
import numpy as np
import numpy.typing as npt
from .manifolds import Sphere, Stiefel, Grassmann, Euclidean, SPD, Hyperbolic
from . import CostFunction

ManifoldType = Sphere | Stiefel | Grassmann | Euclidean | SPD | Hyperbolic

class SGD:
    """Riemannian Stochastic Gradient Descent optimizer."""
    
    def __init__(
        self,
        manifold: ManifoldType,
        learning_rate: float = 0.01,
        momentum: float = 0.0,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> None: ...
    
    def step(
        self,
        cost_function: CostFunction,
        point: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], float]: ...
    
    def optimize(
        self,
        cost_function: CostFunction,
        initial_point: npt.NDArray[np.float64]
    ) -> Dict[str, Any]: ...
    
    @property
    def config(self) -> Dict[str, Any]: ...

class Adam:
    """Riemannian Adam optimizer."""
    
    def __init__(
        self,
        manifold: ManifoldType,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> None: ...

class LBFGS:
    """Riemannian L-BFGS optimizer."""
    
    def __init__(
        self,
        manifold: ManifoldType,
        memory_size: int = 10,
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> None: ...

class ConjugateGradient:
    """Riemannian Conjugate Gradient optimizer."""
    
    def __init__(
        self,
        manifold: ManifoldType,
        variant: Literal["FletcherReeves", "PolakRibiere"] = "FletcherReeves",
        max_iterations: int = 1000,
        tolerance: float = 1e-6
    ) -> None: ...