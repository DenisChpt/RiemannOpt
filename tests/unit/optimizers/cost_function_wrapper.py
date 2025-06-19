"""Wrapper for cost functions to handle matrix/vector conversions."""

import numpy as np
import riemannopt

# Access CostFunction through _riemannopt module
CostFunction = riemannopt._riemannopt.CostFunction


class MatrixCostFunctionWrapper:
    """Wraps cost functions that work with matrices to be compatible with vector-based optimizers."""
    
    def __init__(self, cost_fn, grad_fn, shape):
        """
        Initialize the wrapper.
        
        Args:
            cost_fn: Function that takes a matrix and returns a scalar
            grad_fn: Function that takes a matrix and returns a gradient matrix
            shape: Tuple (rows, cols) for the matrix shape
        """
        self.cost_fn = cost_fn
        self.grad_fn = grad_fn
        self.shape = shape
        self.rows, self.cols = shape
    
    def vector_cost_fn(self, x_vec):
        """Cost function that takes a vector."""
        X = x_vec.reshape(self.shape)
        return self.cost_fn(X)
    
    def vector_grad_fn(self, x_vec):
        """Gradient function that takes a vector and returns a vector."""
        X = x_vec.reshape(self.shape)
        grad_matrix = self.grad_fn(X)
        # Ensure gradient is a numpy array
        if not isinstance(grad_matrix, np.ndarray):
            grad_matrix = np.array(grad_matrix)
        return grad_matrix.flatten()
    
    def get_cost_function(self):
        """Return a CostFunction object compatible with optimizers."""
        return CostFunction(self.vector_cost_fn, self.vector_grad_fn)


def create_matrix_cost_function(cost_fn, grad_fn, manifold):
    """
    Create a cost function wrapper based on the manifold type.
    
    Args:
        cost_fn: Cost function that works with the natural representation
        grad_fn: Gradient function that works with the natural representation
        manifold: The manifold object
    
    Returns:
        A CostFunction object
    """
    manifold_name = type(manifold).__name__
    
    # For sphere and hyperbolic, use functions directly (they work with vectors)
    if manifold_name in ['Sphere', 'Hyperbolic']:
        return CostFunction(cost_fn, grad_fn)
    
    # For matrix manifolds, we need to handle the shape
    if manifold_name == 'Stiefel':
        shape = (manifold.n, manifold.p)
        wrapper = MatrixCostFunctionWrapper(cost_fn, grad_fn, shape)
        return wrapper.get_cost_function()
    
    if manifold_name == 'Grassmann':
        shape = (manifold.n, manifold.p)
        wrapper = MatrixCostFunctionWrapper(cost_fn, grad_fn, shape)
        return wrapper.get_cost_function()
    
    if manifold_name == 'SPD':
        shape = (manifold.size, manifold.size)
        wrapper = MatrixCostFunctionWrapper(cost_fn, grad_fn, shape)
        return wrapper.get_cost_function()
    
    if manifold_name == 'Oblique':
        shape = (manifold.n, manifold.p)
        wrapper = MatrixCostFunctionWrapper(cost_fn, grad_fn, shape)
        return wrapper.get_cost_function()
    
    if manifold_name == 'FixedRank':
        shape = (manifold.m, manifold.n)
        wrapper = MatrixCostFunctionWrapper(cost_fn, grad_fn, shape)
        return wrapper.get_cost_function()
    
    if manifold_name == 'PSDCone':
        shape = (manifold.n, manifold.n)
        wrapper = MatrixCostFunctionWrapper(cost_fn, grad_fn, shape)
        return wrapper.get_cost_function()
    
    # Default: assume vector representation
    return CostFunction(cost_fn, grad_fn)