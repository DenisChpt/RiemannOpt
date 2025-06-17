"""Type stubs for RiemannOpt manifolds."""

from typing import Tuple
import numpy as np
import numpy.typing as npt

class Sphere:
    """Sphere manifold S^{n-1}."""
    
    def __init__(self, dimension: int) -> None: ...
    
    @property
    def dim(self) -> int: ...
    
    @property
    def ambient_dim(self) -> int: ...
    
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    
    def retract(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    
    def tangent_projection(
        self,
        point: npt.NDArray[np.float64],
        vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    
    def exp(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    
    def log(
        self,
        point: npt.NDArray[np.float64],
        other: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    
    def distance(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64]
    ) -> float: ...
    
    def random_point(self) -> npt.NDArray[np.float64]: ...

class Stiefel:
    """Stiefel manifold St(n,p)."""
    
    def __init__(self, n: int, p: int) -> None: ...
    
    @property
    def dim(self) -> int: ...
    
    @property
    def n(self) -> int: ...
    
    @property
    def p(self) -> int: ...
    
    def project(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...
    
    def retract(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    
    def tangent_projection(
        self,
        point: npt.NDArray[np.float64],
        matrix: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]: ...
    
    def random_point(self) -> npt.NDArray[np.float64]: ...

class Grassmann:
    """Grassmann manifold Gr(n,p)."""
    
    def __init__(self, n: int, p: int) -> None: ...
    
    @property
    def dim(self) -> int: ...
    
    @property
    def n(self) -> int: ...
    
    @property
    def p(self) -> int: ...
    
    def project(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class Euclidean:
    """Euclidean manifold R^n."""
    
    def __init__(self, dimension: int) -> None: ...
    
    @property
    def dim(self) -> int: ...
    
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]: ...

class SPD:
    """Symmetric Positive Definite manifold SPD(n)."""
    
    def __init__(self, size: int) -> None: ...
    
    @property
    def dim(self) -> int: ...
    
    @property
    def size(self) -> int: ...

class Hyperbolic:
    """Hyperbolic manifold H^n."""
    
    def __init__(self, dimension: int) -> None: ...
    
    @property
    def dim(self) -> int: ...