"""Riemannian optimization algorithms.

Re-exports all optimizer classes and factory functions from the Rust native
module so they can be imported with ``from riemannopt.optimizers import Adam``.

Available Optimizers
--------------------
- :class:`SGD` -- Stochastic gradient descent with momentum
- :class:`Adam` -- Adaptive moment estimation
- :class:`LBFGS` -- Limited-memory BFGS
- :class:`ConjugateGradient` -- Nonlinear CG (FR / PR / HS / DY)
- :class:`Newton` -- Riemannian Newton
- :class:`TrustRegion` -- Trust-region with Steihaug-CG
- :class:`NaturalGradient` -- Natural gradient descent

Factory helpers
---------------
- :func:`create_optimizer` -- Create an optimizer by name
- :func:`optimize` -- One-shot optimisation
- :class:`OptimizationResult` -- Result container
"""

from __future__ import annotations

from ._riemannopt import optimizers as _o

SGD = _o.SGD
Adam = _o.Adam
LBFGS = _o.LBFGS
ConjugateGradient = _o.ConjugateGradient
Newton = _o.Newton
TrustRegion = _o.TrustRegion
NaturalGradient = _o.NaturalGradient
OptimizationResult = _o.OptimizationResult
create_optimizer = _o.create_optimizer
optimize = _o.optimize

__all__ = [
    "SGD",
    "Adam",
    "LBFGS",
    "ConjugateGradient",
    "Newton",
    "TrustRegion",
    "NaturalGradient",
    "OptimizationResult",
    "create_optimizer",
    "optimize",
]
