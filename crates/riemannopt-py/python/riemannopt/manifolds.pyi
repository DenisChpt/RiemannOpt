"""Type stubs for RiemannOpt manifolds.

This module provides type hints for all manifold classes in the RiemannOpt library.
"""

from typing import Tuple, Union, List, overload
import numpy as np
import numpy.typing as npt

# Type aliases for clarity
VectorArray = npt.NDArray[np.float64]  # Shape: (n,)
MatrixArray = npt.NDArray[np.float64]  # Shape: (n, m)
PointType = Union[VectorArray, MatrixArray]
TangentType = Union[VectorArray, MatrixArray]

class Sphere:
    """Unit sphere manifold S^{n-1} in R^n.
    
    The unit sphere is the set of unit vectors in R^n:
    S^{n-1} = {x ∈ R^n : ||x|| = 1}
    
    Parameters
    ----------
    dimension : int
        Dimension of the ambient space R^n
        
    Attributes
    ----------
    dim : int
        Intrinsic dimension of the manifold (n-1)
    ambient_dim : int
        Dimension of the ambient space (n)
    """
    
    def __init__(self, dimension: int) -> None: ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the manifold (n-1)."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Dimension of the ambient space (n)."""
        ...
    
    def project(self, point: VectorArray) -> VectorArray:
        """Project a point onto the manifold by normalizing.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Point to project
            
        Returns
        -------
        ndarray, shape (n,)
            Normalized point on the sphere
        """
        ...
    
    def project_tangent(
        self,
        point: VectorArray,
        vector: VectorArray
    ) -> VectorArray:
        """Project a vector onto the tangent space at a point.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Point on the manifold
        vector : ndarray, shape (n,)
            Vector to project
            
        Returns
        -------
        ndarray, shape (n,)
            Projected tangent vector
        """
        ...
    
    def retract(
        self,
        point: VectorArray,
        tangent: VectorArray
    ) -> VectorArray:
        """Retract a tangent vector to the manifold.
        
        Uses the exponential map followed by normalization.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Point on the manifold
        tangent : ndarray, shape (n,)
            Tangent vector
            
        Returns
        -------
        ndarray, shape (n,)
            New point on the manifold
        """
        ...
    
    def exp(
        self,
        point: VectorArray,
        tangent: VectorArray
    ) -> VectorArray:
        """Exponential map on the sphere.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Point on the manifold
        tangent : ndarray, shape (n,)
            Tangent vector
            
        Returns
        -------
        ndarray, shape (n,)
            Result of exponential map
        """
        ...
    
    def log(
        self,
        point: VectorArray,
        other: VectorArray
    ) -> VectorArray:
        """Logarithmic map on the sphere.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Base point on the manifold
        other : ndarray, shape (n,)
            Target point on the manifold
            
        Returns
        -------
        ndarray, shape (n,)
            Tangent vector from point to other
        """
        ...
    
    def inner(
        self,
        point: VectorArray,
        u: VectorArray,
        v: VectorArray
    ) -> float:
        """Riemannian inner product of two tangent vectors.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Base point on the manifold
        u : ndarray, shape (n,)
            First tangent vector
        v : ndarray, shape (n,)
            Second tangent vector
            
        Returns
        -------
        float
            Inner product <u, v>_point
        """
        ...
    
    def norm(
        self,
        point: VectorArray,
        tangent: VectorArray
    ) -> float:
        """Norm of a tangent vector.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Base point on the manifold
        tangent : ndarray, shape (n,)
            Tangent vector
            
        Returns
        -------
        float
            Norm ||tangent||_point
        """
        ...
    
    def distance(
        self,
        x: VectorArray,
        y: VectorArray
    ) -> float:
        """Geodesic distance between two points.
        
        Parameters
        ----------
        x : ndarray, shape (n,)
            First point
        y : ndarray, shape (n,)
            Second point
            
        Returns
        -------
        float
            Geodesic distance d(x, y)
        """
        ...
    
    def random_point(self) -> VectorArray:
        """Generate a uniformly random point on the sphere.
        
        Returns
        -------
        ndarray, shape (n,)
            Random point on the sphere
        """
        ...
    
    def random_tangent(
        self,
        point: VectorArray,
        scale: float = 1.0
    ) -> VectorArray:
        """Generate a random tangent vector.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Base point on the manifold
        scale : float, default=1.0
            Scale factor for the tangent vector
            
        Returns
        -------
        ndarray, shape (n,)
            Random tangent vector
        """
        ...
    
    def parallel_transport(
        self,
        from_point: VectorArray,
        to_point: VectorArray,
        tangent: VectorArray
    ) -> VectorArray:
        """Parallel transport a tangent vector along a geodesic.
        
        Parameters
        ----------
        from_point : ndarray, shape (n,)
            Starting point
        to_point : ndarray, shape (n,)
            Ending point
        tangent : ndarray, shape (n,)
            Tangent vector at from_point
            
        Returns
        -------
        ndarray, shape (n,)
            Transported tangent vector at to_point
        """
        ...
    
    def contains(
        self,
        point: VectorArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if a point is on the manifold.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Point to check
        atol : float, default=1e-10
            Absolute tolerance
            
        Returns
        -------
        bool
            True if point is on the sphere
        """
        ...
    
    def is_tangent(
        self,
        point: VectorArray,
        vector: VectorArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if a vector is in the tangent space.
        
        Parameters
        ----------
        point : ndarray, shape (n,)
            Base point on the manifold
        vector : ndarray, shape (n,)
            Vector to check
        atol : float, default=1e-10
            Absolute tolerance
            
        Returns
        -------
        bool
            True if vector is tangent to sphere at point
        """
        ...

class Stiefel:
    """Stiefel manifold St(n,p) of n×p matrices with orthonormal columns.
    
    The Stiefel manifold is the set of n×p matrices with orthonormal columns:
    St(n,p) = {X ∈ R^{n×p} : X^T X = I_p}
    
    Parameters
    ----------
    n : int
        Number of rows
    p : int
        Number of columns (p ≤ n)
        
    Attributes
    ----------
    n : int
        Number of rows
    p : int
        Number of columns
    dim : int
        Intrinsic dimension np - p(p+1)/2
    ambient_dim : int
        Ambient dimension n×p
    shape : tuple
        Shape of matrices (n, p)
    """
    
    def __init__(self, n: int, p: int) -> None: ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension np - p(p+1)/2."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Ambient dimension n×p."""
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
    
    def project(self, point: MatrixArray) -> MatrixArray:
        """Project a matrix onto the Stiefel manifold.
        
        Uses the SVD-based projection.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Matrix to project
            
        Returns
        -------
        ndarray, shape (n, p)
            Matrix with orthonormal columns
        """
        ...
    
    def project_tangent(
        self,
        point: MatrixArray,
        vector: MatrixArray
    ) -> MatrixArray:
        """Project a matrix onto the tangent space.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Point on the manifold
        vector : ndarray, shape (n, p)
            Matrix to project
            
        Returns
        -------
        ndarray, shape (n, p)
            Projected tangent vector
        """
        ...
    
    def retract(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Retract using QR decomposition.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Point on the manifold
        tangent : ndarray, shape (n, p)
            Tangent vector
            
        Returns
        -------
        ndarray, shape (n, p)
            New point on the manifold
        """
        ...
    
    def exp(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Exponential map on the Stiefel manifold.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Point on the manifold
        tangent : ndarray, shape (n, p)
            Tangent vector
            
        Returns
        -------
        ndarray, shape (n, p)
            Result of exponential map
        """
        ...
    
    def log(
        self,
        point: MatrixArray,
        other: MatrixArray
    ) -> MatrixArray:
        """Logarithmic map on the Stiefel manifold.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Base point
        other : ndarray, shape (n, p)
            Target point
            
        Returns
        -------
        ndarray, shape (n, p)
            Tangent vector from point to other
        """
        ...
    
    def inner(
        self,
        point: MatrixArray,
        u: MatrixArray,
        v: MatrixArray
    ) -> float:
        """Canonical metric inner product.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Base point
        u : ndarray, shape (n, p)
            First tangent vector
        v : ndarray, shape (n, p)
            Second tangent vector
            
        Returns
        -------
        float
            Inner product <u, v>_point
        """
        ...
    
    def norm(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> float:
        """Norm of a tangent vector.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Base point
        tangent : ndarray, shape (n, p)
            Tangent vector
            
        Returns
        -------
        float
            Norm ||tangent||_point
        """
        ...
    
    def distance(
        self,
        x: MatrixArray,
        y: MatrixArray
    ) -> float:
        """Geodesic distance between two points.
        
        Parameters
        ----------
        x : ndarray, shape (n, p)
            First point
        y : ndarray, shape (n, p)
            Second point
            
        Returns
        -------
        float
            Geodesic distance
        """
        ...
    
    def random_point(self) -> MatrixArray:
        """Generate a uniformly random point.
        
        Returns
        -------
        ndarray, shape (n, p)
            Random matrix with orthonormal columns
        """
        ...
    
    def random_tangent(
        self,
        point: MatrixArray,
        scale: float = 1.0
    ) -> MatrixArray:
        """Generate a random tangent vector.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Base point
        scale : float, default=1.0
            Scale factor
            
        Returns
        -------
        ndarray, shape (n, p)
            Random tangent vector
        """
        ...
    
    def parallel_transport(
        self,
        from_point: MatrixArray,
        to_point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Parallel transport along a geodesic.
        
        Parameters
        ----------
        from_point : ndarray, shape (n, p)
            Starting point
        to_point : ndarray, shape (n, p)
            Ending point
        tangent : ndarray, shape (n, p)
            Tangent vector at from_point
            
        Returns
        -------
        ndarray, shape (n, p)
            Transported tangent vector
        """
        ...
    
    def contains(
        self,
        point: MatrixArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if matrix has orthonormal columns.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Matrix to check
        atol : float, default=1e-10
            Absolute tolerance
            
        Returns
        -------
        bool
            True if X^T X = I_p
        """
        ...
    
    def is_tangent(
        self,
        point: MatrixArray,
        vector: MatrixArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if vector is in tangent space.
        
        Parameters
        ----------
        point : ndarray, shape (n, p)
            Base point
        vector : ndarray, shape (n, p)
            Vector to check
        atol : float, default=1e-10
            Absolute tolerance
            
        Returns
        -------
        bool
            True if vector is tangent
        """
        ...

class Grassmann:
    """Grassmann manifold Gr(n,p) of p-dimensional subspaces in R^n.
    
    Parameters
    ----------
    n : int
        Dimension of ambient space
    p : int
        Dimension of subspaces
        
    Attributes
    ----------
    n : int
        Ambient dimension
    p : int
        Subspace dimension
    dim : int
        Intrinsic dimension p(n-p)
    """
    
    def __init__(self, n: int, p: int) -> None: ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension p(n-p)."""
        ...
    
    @property
    def n(self) -> int:
        """Ambient dimension."""
        ...
    
    @property
    def p(self) -> int:
        """Subspace dimension."""
        ...
    
    def project(self, matrix: MatrixArray) -> MatrixArray:
        """Project to Grassmann manifold."""
        ...
    
    def project_tangent(
        self,
        point: MatrixArray,
        vector: MatrixArray
    ) -> MatrixArray:
        """Project to tangent space."""
        ...
    
    def retract(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Retraction."""
        ...
    
    def inner(
        self,
        point: MatrixArray,
        u: MatrixArray,
        v: MatrixArray
    ) -> float:
        """Inner product."""
        ...
    
    def norm(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> float:
        """Norm of tangent vector."""
        ...
    
    def random_point(self) -> MatrixArray:
        """Random point."""
        ...
    
    def random_tangent(
        self,
        point: MatrixArray,
        scale: float = 1.0
    ) -> MatrixArray:
        """Random tangent vector."""
        ...

class ProductManifold:
    """Product of multiple manifolds.
    
    Represents M = M1 × M2 × ... × Mn where each Mi is a manifold.
    Points and tangent vectors are represented as tuples.
    
    Parameters
    ----------
    manifolds : list
        List of component manifolds
        
    Attributes
    ----------
    manifolds : list
        Component manifolds
    dim : int
        Sum of component dimensions
    """
    
    def __init__(self, manifolds: List[Union[Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, PSDCone]]) -> None: ...
    
    @property
    def manifolds(self) -> List[Union[Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, PSDCone]]:
        """Component manifolds."""
        ...
    
    @property
    def dim(self) -> int:
        """Sum of component dimensions."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Sum of ambient dimensions."""
        ...
    
    def project(self, point: Tuple[PointType, ...]) -> Tuple[PointType, ...]:
        """Project each component onto its manifold.
        
        Parameters
        ----------
        point : tuple of ndarray
            Tuple of points for each component
            
        Returns
        -------
        tuple of ndarray
            Projected points
        """
        ...
    
    def project_tangent(
        self,
        point: Tuple[PointType, ...],
        vector: Tuple[TangentType, ...]
    ) -> Tuple[TangentType, ...]:
        """Project each tangent component.
        
        Parameters
        ----------
        point : tuple of ndarray
            Base points
        vector : tuple of ndarray
            Tangent vectors
            
        Returns
        -------
        tuple of ndarray
            Projected tangent vectors
        """
        ...
    
    def retract(
        self,
        point: Tuple[PointType, ...],
        tangent: Tuple[TangentType, ...]
    ) -> Tuple[PointType, ...]:
        """Retract each component.
        
        Parameters
        ----------
        point : tuple of ndarray
            Base points
        tangent : tuple of ndarray
            Tangent vectors
            
        Returns
        -------
        tuple of ndarray
            New points
        """
        ...
    
    def inner(
        self,
        point: Tuple[PointType, ...],
        u: Tuple[TangentType, ...],
        v: Tuple[TangentType, ...]
    ) -> float:
        """Sum of component inner products.
        
        Parameters
        ----------
        point : tuple of ndarray
            Base points
        u : tuple of ndarray
            First tangent vectors
        v : tuple of ndarray
            Second tangent vectors
            
        Returns
        -------
        float
            Total inner product
        """
        ...
    
    def norm(
        self,
        point: Tuple[PointType, ...],
        tangent: Tuple[TangentType, ...]
    ) -> float:
        """Norm of product tangent vector.
        
        Parameters
        ----------
        point : tuple of ndarray
            Base points
        tangent : tuple of ndarray
            Tangent vectors
            
        Returns
        -------
        float
            Norm
        """
        ...
    
    def random_point(self) -> Tuple[PointType, ...]:
        """Random point on product manifold.
        
        Returns
        -------
        tuple of ndarray
            Random points for each component
        """
        ...
    
    def random_tangent(
        self,
        point: Tuple[PointType, ...],
        scale: float = 1.0
    ) -> Tuple[TangentType, ...]:
        """Random tangent on product manifold.
        
        Parameters
        ----------
        point : tuple of ndarray
            Base points
        scale : float, default=1.0
            Scale factor
            
        Returns
        -------
        tuple of ndarray
            Random tangent vectors
        """
        ...
    
    def exp(
        self,
        point: Tuple[PointType, ...],
        tangent: Tuple[TangentType, ...]
    ) -> Tuple[PointType, ...]:
        """Exponential map on each component."""
        ...
    
    def log(
        self,
        point: Tuple[PointType, ...],
        other: Tuple[PointType, ...]
    ) -> Tuple[TangentType, ...]:
        """Logarithmic map on each component."""
        ...
    
    def distance(
        self,
        x: Tuple[PointType, ...],
        y: Tuple[PointType, ...]
    ) -> float:
        """Distance on product manifold."""
        ...
    
    def parallel_transport(
        self,
        from_point: Tuple[PointType, ...],
        to_point: Tuple[PointType, ...],
        tangent: Tuple[TangentType, ...]
    ) -> Tuple[TangentType, ...]:
        """Parallel transport on each component."""
        ...
    
    def contains(
        self,
        point: Tuple[PointType, ...],
        atol: float = 1e-10
    ) -> bool:
        """Check if point is on product manifold."""
        ...
    
    def is_tangent(
        self,
        point: Tuple[PointType, ...],
        vector: Tuple[TangentType, ...],
        atol: float = 1e-10
    ) -> bool:
        """Check if vector is tangent."""
        ...

class SPD:
    """Symmetric Positive Definite manifold SPD(n).
    
    The manifold of n×n symmetric positive definite matrices.
    
    Parameters
    ----------
    n : int
        Size of matrices
        
    Attributes
    ----------
    n : int
        Matrix size
    dim : int
        Intrinsic dimension n(n+1)/2
    ambient_dim : int
        Ambient dimension n×n
    """
    
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
    
    def project(self, point: MatrixArray) -> MatrixArray:
        """Project to nearest SPD matrix.
        
        Parameters
        ----------
        point : ndarray, shape (n, n)
            Matrix to project
            
        Returns
        -------
        ndarray, shape (n, n)
            SPD matrix
        """
        ...
    
    def project_tangent(
        self,
        point: MatrixArray,
        vector: MatrixArray
    ) -> MatrixArray:
        """Project to tangent space (symmetric matrices).
        
        Parameters
        ----------
        point : ndarray, shape (n, n)
            SPD matrix
        vector : ndarray, shape (n, n)
            Matrix to project
            
        Returns
        -------
        ndarray, shape (n, n)
            Symmetric matrix
        """
        ...
    
    def retract(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Retract using matrix exponential.
        
        Parameters
        ----------
        point : ndarray, shape (n, n)
            SPD matrix
        tangent : ndarray, shape (n, n)
            Symmetric tangent vector
            
        Returns
        -------
        ndarray, shape (n, n)
            New SPD matrix
        """
        ...
    
    def exp(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Exponential map for affine-invariant metric."""
        ...
    
    def log(
        self,
        point: MatrixArray,
        other: MatrixArray
    ) -> MatrixArray:
        """Logarithmic map for affine-invariant metric."""
        ...
    
    def inner(
        self,
        point: MatrixArray,
        u: MatrixArray,
        v: MatrixArray
    ) -> float:
        """Affine-invariant metric inner product."""
        ...
    
    def norm(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> float:
        """Norm under affine-invariant metric."""
        ...
    
    def distance(
        self,
        x: MatrixArray,
        y: MatrixArray
    ) -> float:
        """Geodesic distance for affine-invariant metric."""
        ...
    
    def random_point(self) -> MatrixArray:
        """Generate random SPD matrix."""
        ...
    
    def random_tangent(
        self,
        point: MatrixArray,
        scale: float = 1.0
    ) -> MatrixArray:
        """Generate random symmetric matrix."""
        ...
    
    def parallel_transport(
        self,
        from_point: MatrixArray,
        to_point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Parallel transport for affine-invariant metric."""
        ...
    
    def contains(
        self,
        point: MatrixArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if matrix is SPD."""
        ...
    
    def is_tangent(
        self,
        point: MatrixArray,
        vector: MatrixArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if matrix is symmetric."""
        ...

class Hyperbolic:
    """Hyperbolic manifold H^n.
    
    The n-dimensional hyperbolic space with constant negative curvature.
    
    Parameters
    ----------
    dimension : int
        Dimension of the hyperbolic space
        
    Attributes
    ----------
    dim : int
        Intrinsic dimension
    ambient_dim : int
        Ambient space dimension (n+1 for Lorentz model)
    """
    
    def __init__(self, dimension: int) -> None: ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Ambient dimension."""
        ...
    
    def project(self, point: VectorArray) -> VectorArray:
        """Project to hyperbolic space."""
        ...
    
    def project_tangent(
        self,
        point: VectorArray,
        vector: VectorArray
    ) -> VectorArray:
        """Project to tangent space."""
        ...
    
    def retract(
        self,
        point: VectorArray,
        tangent: VectorArray
    ) -> VectorArray:
        """Retraction via exponential map."""
        ...
    
    def exp(
        self,
        point: VectorArray,
        tangent: VectorArray
    ) -> VectorArray:
        """Exponential map."""
        ...
    
    def log(
        self,
        point: VectorArray,
        other: VectorArray
    ) -> VectorArray:
        """Logarithmic map."""
        ...
    
    def inner(
        self,
        point: VectorArray,
        u: VectorArray,
        v: VectorArray
    ) -> float:
        """Riemannian inner product."""
        ...
    
    def norm(
        self,
        point: VectorArray,
        tangent: VectorArray
    ) -> float:
        """Norm of tangent vector."""
        ...
    
    def distance(
        self,
        x: VectorArray,
        y: VectorArray
    ) -> float:
        """Hyperbolic distance."""
        ...
    
    def random_point(self) -> VectorArray:
        """Random point."""
        ...
    
    def random_tangent(
        self,
        point: VectorArray,
        scale: float = 1.0
    ) -> VectorArray:
        """Random tangent vector."""
        ...
    
    def parallel_transport(
        self,
        from_point: VectorArray,
        to_point: VectorArray,
        tangent: VectorArray
    ) -> VectorArray:
        """Parallel transport."""
        ...
    
    def contains(
        self,
        point: VectorArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if point is on hyperbolic space."""
        ...
    
    def is_tangent(
        self,
        point: VectorArray,
        vector: VectorArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if vector is tangent."""
        ...

class Oblique:
    """Oblique manifold OB(m,n) of m×n matrices with unit-norm columns.
    
    Parameters
    ----------
    m : int
        Number of rows
    n : int
        Number of columns
    """
    
    def __init__(self, m: int, n: int) -> None: ...
    
    @property
    def m(self) -> int:
        """Number of rows."""
        ...
    
    @property
    def n(self) -> int:
        """Number of columns."""
        ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension (m-1)×n."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Ambient dimension m×n."""
        ...
    
    @property
    def shape(self) -> Tuple[int, int]:
        """Shape (m, n)."""
        ...
    
    def project(self, point: MatrixArray) -> MatrixArray:
        """Normalize each column."""
        ...
    
    def project_tangent(
        self,
        point: MatrixArray,
        vector: MatrixArray
    ) -> MatrixArray:
        """Project to tangent space."""
        ...
    
    def retract(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Retraction by normalization."""
        ...
    
    def inner(
        self,
        point: MatrixArray,
        u: MatrixArray,
        v: MatrixArray
    ) -> float:
        """Inner product."""
        ...
    
    def norm(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> float:
        """Norm."""
        ...
    
    def random_point(self) -> MatrixArray:
        """Random point."""
        ...
    
    def random_tangent(
        self,
        point: MatrixArray,
        scale: float = 1.0
    ) -> MatrixArray:
        """Random tangent."""
        ...

class PSDCone:
    """Positive Semidefinite cone manifold.
    
    The cone of n×n positive semidefinite matrices.
    
    Parameters
    ----------
    n : int
        Size of matrices
    """
    
    def __init__(self, n: int) -> None: ...
    
    @property
    def n(self) -> int:
        """Matrix size."""
        ...
    
    @property
    def dim(self) -> int:
        """Intrinsic dimension n(n+1)/2."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Ambient dimension n×n."""
        ...
    
    def project(self, point: MatrixArray) -> MatrixArray:
        """Project to PSD cone."""
        ...
    
    def project_tangent(
        self,
        point: MatrixArray,
        vector: MatrixArray
    ) -> MatrixArray:
        """Project to tangent cone."""
        ...
    
    def retract(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> MatrixArray:
        """Retraction."""
        ...
    
    def inner(
        self,
        point: MatrixArray,
        u: MatrixArray,
        v: MatrixArray
    ) -> float:
        """Inner product."""
        ...
    
    def norm(
        self,
        point: MatrixArray,
        tangent: MatrixArray
    ) -> float:
        """Norm."""
        ...
    
    def random_point(self) -> MatrixArray:
        """Random PSD matrix."""
        ...
    
    def random_tangent(
        self,
        point: MatrixArray,
        scale: float = 1.0
    ) -> MatrixArray:
        """Random tangent."""
        ...
    
    def contains(
        self,
        point: MatrixArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if matrix is PSD."""
        ...
    
    def is_tangent(
        self,
        point: MatrixArray,
        vector: MatrixArray,
        atol: float = 1e-10
    ) -> bool:
        """Check if vector is in tangent cone."""
        ...