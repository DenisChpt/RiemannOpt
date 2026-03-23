"""Riemannian manifold implementations.

Re-exports all manifold classes from the Rust native module so they can be
imported with ``from riemannopt.manifolds import Sphere``.

Available Manifolds
-------------------
- :class:`Sphere` -- Unit sphere S^{n-1}
- :class:`Stiefel` -- Orthonormal frames St(n, p)
- :class:`Grassmann` -- Subspaces Gr(n, p)
- :class:`SPD` -- Symmetric positive-definite matrices
- :class:`Euclidean` -- Flat R^n
- :class:`Hyperbolic` -- Hyperbolic space H^n
- :class:`Oblique` -- Unit-norm column matrices
- :class:`PSDCone` -- Positive semidefinite cone
- :class:`ProductManifold` -- Cartesian product of manifolds
"""

from __future__ import annotations

from ._riemannopt import manifolds as _m

Sphere = _m.Sphere
Stiefel = _m.Stiefel
Grassmann = _m.Grassmann
SPD = _m.SPD
Euclidean = _m.Euclidean
Hyperbolic = _m.Hyperbolic
Oblique = _m.Oblique
PSDCone = _m.PSDCone
ProductManifold = _m.ProductManifold

__all__ = [
    "Sphere",
    "Stiefel",
    "Grassmann",
    "SPD",
    "Euclidean",
    "Hyperbolic",
    "Oblique",
    "PSDCone",
    "ProductManifold",
]
