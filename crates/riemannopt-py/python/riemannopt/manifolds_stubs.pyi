"""Type stubs for RiemannOpt manifolds."""

from typing import Optional, Tuple, Union
import numpy as np
import numpy.typing as npt

class Sphere:
    """Unit sphere manifold S^(n-1) = {x ∈ R^n : ||x|| = 1}."""
    
    def __init__(self, dim: int) -> None:
        """Initialize sphere manifold.
        
        Args:
            dim: Ambient dimension (manifold dimension is dim-1)
        """
        ...
    
    @property
    def dim(self) -> int:
        """Manifold dimension (n-1 for S^(n-1))."""
        ...
    
    @property
    def manifold_dim(self) -> int:
        """Alias for dim."""
        ...
    
    @property
    def ambient_dim(self) -> int:
        """Ambient space dimension."""
        ...
    
    def project(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project point onto the sphere.
        
        Args:
            point: Point in ambient space
            
        Returns:
            Projected point on sphere (normalized)
        """
        ...
    
    def tangent_projection(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project vector onto tangent space at point.
        
        Args:
            point: Point on manifold
            vector: Vector in ambient space
            
        Returns:
            Projected vector in tangent space
        """
        ...
    
    def retraction(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Retract from point along tangent vector.
        
        Args:
            point: Point on manifold
            vector: Tangent vector
            
        Returns:
            New point on manifold
        """
        ...
    
    def inner_product(self, point: npt.NDArray[np.float64], u: npt.NDArray[np.float64], v: npt.NDArray[np.float64]) -> float:
        """Riemannian inner product of tangent vectors.
        
        Args:
            point: Point on manifold
            u: First tangent vector
            v: Second tangent vector
            
        Returns:
            Inner product value
        """
        ...
    
    def distance(self, x: npt.NDArray[np.float64], y: npt.NDArray[np.float64]) -> float:
        """Geodesic distance between two points.
        
        Args:
            x: First point on manifold
            y: Second point on manifold
            
        Returns:
            Geodesic distance
        """
        ...
    
    def exp(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Exponential map.
        
        Args:
            point: Point on manifold
            vector: Tangent vector
            
        Returns:
            Exponential of vector at point
        """
        ...
    
    def log(self, point: npt.NDArray[np.float64], target: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Logarithmic map.
        
        Args:
            point: Point on manifold
            target: Target point on manifold
            
        Returns:
            Tangent vector from point to target
        """
        ...
    
    def random_point(self) -> npt.NDArray[np.float64]:
        """Generate random point on manifold."""
        ...
    
    def random_tangent(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Generate random tangent vector at point."""
        ...

class Stiefel:
    """Stiefel manifold St(n,p) = {X ∈ R^(n×p) : X^T X = I_p}."""
    
    def __init__(self, n: int, p: int) -> None:
        """Initialize Stiefel manifold.
        
        Args:
            n: Number of rows
            p: Number of columns (p <= n)
        """
        ...
    
    @property
    def n(self) -> int:
        """Number of rows."""
        ...
    
    @property
    def p(self) -> int:
        """Number of columns."""
        ...
    
    @property
    def ambient_rows(self) -> int:
        """Alias for n."""
        ...
    
    @property
    def ambient_cols(self) -> int:
        """Alias for p."""
        ...
    
    @property
    def dim(self) -> int:
        """Manifold dimension np - p(p+1)/2."""
        ...
    
    @property
    def manifold_dim(self) -> int:
        """Alias for dim."""
        ...
    
    def project(self, matrix: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project matrix onto Stiefel manifold.
        
        Args:
            matrix: Flattened matrix of shape (n*p,)
            
        Returns:
            Projected flattened matrix with orthonormal columns
        """
        ...
    
    def tangent_projection(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Project vector onto tangent space.
        
        Args:
            point: Point on manifold (flattened)
            vector: Vector in ambient space (flattened)
            
        Returns:
            Projected tangent vector (flattened)
        """
        ...
    
    def retraction(self, point: npt.NDArray[np.float64], vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Retract from point along tangent vector.
        
        Args:
            point: Point on manifold (flattened)
            vector: Tangent vector (flattened)
            
        Returns:
            New point on manifold (flattened)
        """
        ...
    
    def random_point(self) -> npt.NDArray[np.float64]:
        """Generate random point on manifold (flattened)."""
        ...
    
    def random_tangent(self, point: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """Generate random tangent vector at point (flattened)."""
        ...