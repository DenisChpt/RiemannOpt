"""Type definitions for RiemannOpt."""

from typing import Union, Protocol, Callable, Optional, TypeVar, Generic, Dict, Any, Tuple
import numpy as np
from numpy.typing import NDArray, DTypeLike

# Basic types
Scalar = Union[float, np.floating]
Array = NDArray[np.floating]
Point = Array
TangentVector = Array
Gradient = Array

# Function types
CostFunc = Callable[[Point], Scalar]
CostGradFunc = Callable[[Point], Tuple[Scalar, Gradient]]
MetricFunc = Callable[[Point, TangentVector, TangentVector], Scalar]
RetractionFunc = Callable[[Point, TangentVector], Point]

# Generic types
T = TypeVar('T')
ManifoldType = TypeVar('ManifoldType', bound='Manifold')
OptimizerType = TypeVar('OptimizerType', bound='Optimizer')

# Result types
OptimizationResult = Dict[str, Any]


class Manifold(Protocol):
    """Protocol for manifold implementations."""
    
    @property
    def dim(self) -> int:
        """Dimension of the manifold."""
        ...
    
    def project(self, point: Point) -> Point:
        """Project a point onto the manifold."""
        ...
    
    def random_point(self) -> Point:
        """Generate a random point on the manifold."""
        ...
    
    def inner(self, point: Point, v1: TangentVector, v2: TangentVector) -> Scalar:
        """Riemannian inner product."""
        ...
    
    def norm(self, point: Point, vector: TangentVector) -> Scalar:
        """Norm of a tangent vector."""
        ...
    
    def proj(self, point: Point, vector: Array) -> TangentVector:
        """Project vector to tangent space."""
        ...
    
    def retract(self, point: Point, vector: TangentVector) -> Point:
        """Retraction from tangent space to manifold."""
        ...
    
    def egrad2rgrad(self, point: Point, egrad: Array) -> TangentVector:
        """Convert Euclidean gradient to Riemannian gradient."""
        ...


class CostFunction(Protocol):
    """Protocol for cost function implementations."""
    
    def __call__(self, point: Point) -> Scalar:
        """Evaluate cost at point."""
        ...
    
    def gradient(self, point: Point) -> Gradient:
        """Compute gradient at point."""
        ...
    
    def value_and_gradient(self, point: Point) -> Tuple[Scalar, Gradient]:
        """Compute both value and gradient."""
        ...


class Optimizer(Protocol):
    """Protocol for optimizer implementations."""
    
    def step(self, gradient: Gradient, point: Point) -> Point:
        """Perform one optimization step."""
        ...
    
    def reset(self) -> None:
        """Reset optimizer state."""
        ...
    
    def get_config(self) -> Dict[str, Any]:
        """Get optimizer configuration."""
        ...
    
    def set_config(self, **kwargs) -> None:
        """Update optimizer configuration."""
        ...


class Callback(Protocol):
    """Protocol for optimization callbacks."""
    
    def on_optimization_start(self, initial_point: Point) -> None:
        """Called at the start of optimization."""
        ...
    
    def on_iteration_end(
        self, 
        iteration: int, 
        point: Point, 
        cost: Scalar, 
        gradient: Gradient
    ) -> bool:
        """Called at the end of each iteration.
        
        Returns:
            True to continue, False to stop optimization
        """
        ...
    
    def on_optimization_end(self, result: OptimizationResult) -> None:
        """Called at the end of optimization."""
        ...


class LineSearch(Protocol):
    """Protocol for line search implementations."""
    
    def search(
        self,
        manifold: Manifold,
        cost_function: CostFunction,
        point: Point,
        direction: TangentVector,
        initial_step: float = 1.0
    ) -> float:
        """Find step size along direction."""
        ...