"""Type stubs for riemannopt.manifolds — re-exports from the Rust module."""

from ._riemannopt import (
    Sphere as Sphere,
    Stiefel as Stiefel,
    Grassmann as Grassmann,
    SPD as SPD,
    Euclidean as Euclidean,
    Hyperbolic as Hyperbolic,
    Oblique as Oblique,
    PSDCone as PSDCone,
    ProductManifold as ProductManifold,
)

__all__: list[str]
