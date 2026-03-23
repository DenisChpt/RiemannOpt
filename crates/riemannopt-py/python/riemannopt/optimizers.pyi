"""Type stubs for riemannopt.optimizers — re-exports from the Rust module."""

from ._riemannopt import (
    SGD as SGD,
    Adam as Adam,
    LBFGS as LBFGS,
    ConjugateGradient as ConjugateGradient,
    Newton as Newton,
    TrustRegion as TrustRegion,
    NaturalGradient as NaturalGradient,
    OptimizationResult as OptimizationResult,
    CostFunction as _CostFunction,
)
from typing import Any
from numpy.typing import ArrayLike

def create_optimizer(optimizer_name: str, **kwargs: Any) -> Any: ...
def optimize(
    cost_function: _CostFunction,
    manifold: Any,
    initial_point: ArrayLike,
    optimizer: str = ...,
    max_iterations: int = ...,
    gradient_tolerance: float = ...,
) -> OptimizationResult: ...

__all__: list[str]
