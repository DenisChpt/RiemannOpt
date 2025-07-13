"""Type stubs for RiemannOpt manifolds."""

from typing import Tuple
import numpy as np
import numpy.typing as npt

class Sphere:
    """Unit sphere manifold S^{n-1} in R^n."""
    
    def __init__(self, dimension: int) -> None: ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Dimension of the ambient space."""
        ...
    
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project a point onto the manifold."""
        ...
    
    def project_tangent(
        self,
        point: npt.NDArray[np.float64],
        vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Project a vector onto the tangent space."""
        ...
    
    def retract(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Retract a tangent vector to the manifold."""
        ...
    
    def exp(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Exponential map."""
        ...
    
    def log(
        self,
        point: npt.NDArray[np.float64],
        other: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Logarithmic map."""
        ...
    
    def inner(
        self,
        point: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64]
    ) -> float:
        """Riemannian inner product."""
        ...
    
    def norm(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> float:
        """Norm of a tangent vector."""
        ...
    
    def distance(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64]
    ) -> float:
        """Geodesic distance between two points."""
        ...
    
    def random_point(self) -> npt.NDArray[np.float64]:
        """Generate a random point on the manifold."""
        ...
    
    def random_tangent(
        self,
        point: npt.NDArray[np.float64],
        scale: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Generate a random tangent vector."""
        ...
    
    def parallel_transport(
        self,
        from_point: npt.NDArray[np.float64],
        to_point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Parallel transport a tangent vector."""
        ...
    
    def contains(
        self,
        point: npt.NDArray[np.float64],
        atol: float = 1e-10
    ) -> bool:
        """Check if a point is on the manifold."""
        ...
    
    def is_tangent(
        self,
        point: npt.NDArray[np.float64],
        vector: npt.NDArray[np.float64],
        atol: float = 1e-10
    ) -> bool:
        """Check if a vector is in the tangent space."""
        ...

class Stiefel:
    """Stiefel manifold St(n,p) of n×p matrices with orthonormal columns."""
    
    def __init__(self, n: int, p: int) -> None: ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Ambient dimension."""
        ...
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape of matrices (n, p)."""
        ...
    
    @property
    def n(self) -> int:
        """Number of rows."""
        ...
    
    @property
    def p(self) -> int:
        """Number of columns."""
        ...
    
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project a matrix onto the manifold."""
        ...
    
    def project_tangent(
        self,
        point: npt.NDArray[np.float64],
        vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Project a matrix onto the tangent space."""
        ...
    
    def retract(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Retract using QR decomposition."""
        ...
    
    def exp(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Exponential map."""
        ...
    
    def log(
        self,
        point: npt.NDArray[np.float64],
        other: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Logarithmic map."""
        ...
    
    def inner(
        self,
        point: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64]
    ) -> float:
        """Canonical metric inner product."""
        ...
    
    def norm(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> float:
        """Norm of a tangent vector."""
        ...
    
    def distance(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64]
    ) -> float:
        """Geodesic distance."""
        ...
    
    def random_point(self) -> npt.NDArray[np.float64]:
        """Generate a random point."""
        ...
    
    def random_tangent(
        self,
        point: npt.NDArray[np.float64],
        scale: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Generate a random tangent vector."""
        ...
    
    def parallel_transport(
        self,
        from_point: npt.NDArray[np.float64],
        to_point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Parallel transport."""
        ...
    
    def contains(
        self,
        point: npt.NDArray[np.float64],
        atol: float = 1e-10
    ) -> bool:
        """Check if matrix has orthonormal columns."""
        ...
    
    def is_tangent(
        self,
        point: npt.NDArray[np.float64],
        vector: npt.NDArray[np.float64],
        atol: float = 1e-10
    ) -> bool:
        """Check if vector is in tangent space."""
        ...

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

class ProductManifold:
    """Product of multiple manifolds."""
    
    def __init__(self, manifolds: list) -> None: ...
    
    @property
    def manifolds(self) -> list:
        """Component manifolds."""
        ...
    
    @property
    def dim(self) -> int:
        """Sum of component dimensions."""
        ...
    
    def project(self, point: list[npt.NDArray[np.float64]]) -> list[npt.NDArray[np.float64]]:
        """Project each component."""
        ...
    
    def project_tangent(
        self,
        point: list[npt.NDArray[np.float64]],
        vector: list[npt.NDArray[np.float64]]
    ) -> list[npt.NDArray[np.float64]]:
        """Project each tangent component."""
        ...
    
    def retract(
        self,
        point: list[npt.NDArray[np.float64]],
        tangent: list[npt.NDArray[np.float64]]
    ) -> list[npt.NDArray[np.float64]]:
        """Retract each component."""
        ...
    
    def random_point(self) -> list[npt.NDArray[np.float64]]:
        """Random point on product manifold."""
        ...
    
    def random_tangent(
        self,
        point: list[npt.NDArray[np.float64]],
        scale: float = 1.0
    ) -> list[npt.NDArray[np.float64]]:
        """Random tangent on product manifold."""
        ...

class SPD:
    """Symmetric Positive Definite manifold SPD(n)."""
    
    def __init__(self, n: int) -> None: ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension n(n+1)/2."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Ambient dimension n×n."""
        ...
    
    @property
    def n(self) -> int:
        """Size of matrices."""
        ...
    
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project to nearest SPD matrix."""
        ...
    
    def project_tangent(
        self,
        point: npt.NDArray[np.float64],
        vector: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Project to tangent space (symmetric matrices)."""
        ...
    
    def retract(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Retract using matrix exponential."""
        ...
    
    def exp(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Exponential map."""
        ...
    
    def log(
        self,
        point: npt.NDArray[np.float64],
        other: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Logarithmic map."""
        ...
    
    def inner(
        self,
        point: npt.NDArray[np.float64],
        u: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64]
    ) -> float:
        """Affine-invariant metric."""
        ...
    
    def norm(
        self,
        point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> float:
        """Norm under affine-invariant metric."""
        ...
    
    def distance(
        self,
        x: npt.NDArray[np.float64],
        y: npt.NDArray[np.float64]
    ) -> float:
        """Geodesic distance."""
        ...
    
    def random_point(self) -> npt.NDArray[np.float64]:
        """Generate random SPD matrix."""
        ...
    
    def random_tangent(
        self,
        point: npt.NDArray[np.float64],
        scale: float = 1.0
    ) -> npt.NDArray[np.float64]:
        """Generate random symmetric matrix."""
        ...
    
    def parallel_transport(
        self,
        from_point: npt.NDArray[np.float64],
        to_point: npt.NDArray[np.float64],
        tangent: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Parallel transport."""
        ...
    
    def contains(
        self,
        point: npt.NDArray[np.float64],
        atol: float = 1e-10
    ) -> bool:
        """Check if matrix is SPD."""
        ...
    
    def is_tangent(
        self,
        point: npt.NDArray[np.float64],
        vector: npt.NDArray[np.float64],
        atol: float = 1e-10
    ) -> bool:
        """Check if matrix is symmetric."""
        ...

class Hyperbolic:
    """Hyperbolic manifold H^n."""
    
    def __init__(self, dimension: int) -> None: ...
    
    @property
    def dim(self) -> int: ...