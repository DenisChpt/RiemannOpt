"""Exception classes for RiemannOpt.

This module defines all custom exceptions used throughout the library,
providing clear error messages and proper exception hierarchy.
"""

from typing import Optional, Any, Dict


class RiemannOptError(Exception):
    """Base exception for all RiemannOpt errors.
    
    All custom exceptions in this library inherit from this base class,
    making it easy to catch all library-specific errors.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """Initialize the exception.
        
        Args:
            message: Human-readable error message
            details: Optional dictionary with additional error details
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ManifoldError(RiemannOptError):
    """Errors related to manifold operations."""
    pass


class InvalidPointError(ManifoldError):
    """Raised when a point is not on the manifold.
    
    This error occurs when trying to use a point that doesn't satisfy
    the manifold constraints.
    """
    
    def __init__(self, message: str, point=None, manifold=None, 
                 manifold_name: Optional[str] = None, point_shape: Optional[tuple] = None):
        """Initialize the error.
        
        Args:
            message: Error message
            point: The invalid point (optional)
            manifold: The manifold object (optional)
            manifold_name: Name of the manifold (optional)
            point_shape: Shape of the invalid point (optional)
        """
        details = {}
        if point is not None:
            details['point'] = point
        if manifold is not None:
            details['manifold'] = manifold
        if manifold_name:
            details['manifold_name'] = manifold_name
        if point_shape:
            details['point_shape'] = point_shape
            
        super().__init__(message, details)
        self.point = point
        self.manifold = manifold


class InvalidTangentError(ManifoldError):
    """Raised when a vector is not in the tangent space.
    
    This error occurs when a vector doesn't satisfy the tangent space
    constraints at a given point.
    """
    
    def __init__(self, message: str, vector=None, point=None, manifold=None):
        """Initialize the error.
        
        Args:
            message: Error message
            vector: The invalid vector (optional)
            point: Base point on manifold (optional)
            manifold: The manifold object (optional)
        """
        details = {}
        if vector is not None:
            details['vector'] = vector
        if point is not None:
            details['point'] = point
        if manifold is not None:
            details['manifold'] = manifold
            
        super().__init__(message, details)
        self.vector = vector
        self.point = point
        self.manifold = manifold


class DimensionMismatchError(ManifoldError):
    """Raised when array dimensions don't match expected values.
    
    This is a common error when working with manifolds that require
    specific dimensions.
    """
    
    def __init__(self, message: str, expected_shape=None, actual_shape=None, 
                 array_name: str = "array"):
        """Initialize the error.
        
        Args:
            message: Error message
            expected_shape: Expected dimensions
            actual_shape: Actual dimensions
            array_name: Name of the array for error message
        """
        details = {'array_name': array_name}
        if expected_shape is not None:
            details['expected_shape'] = expected_shape
        if actual_shape is not None:
            details['actual_shape'] = actual_shape
            
        super().__init__(message, details)
        self.expected_shape = expected_shape
        self.actual_shape = actual_shape


class OptimizationError(RiemannOptError):
    """Errors related to optimization."""
    pass


class ConvergenceError(OptimizationError):
    """Raised when optimization fails to converge.
    
    This error indicates that the optimization algorithm couldn't
    find a satisfactory solution within the given constraints.
    """
    
    def __init__(self, message: str, iterations=None, final_value=None,
                 max_iterations=None, gradient_norm=None, cost_change=None):
        """Initialize the error.
        
        Args:
            message: Error message
            iterations: Number of iterations performed
            final_value: Final cost value
            max_iterations: Maximum allowed iterations
            gradient_norm: Final gradient norm
            cost_change: Final cost change
        """
        details = {}
        if iterations is not None:
            details['iterations'] = iterations
        if final_value is not None:
            details['final_value'] = final_value
        if max_iterations is not None:
            details['max_iterations'] = max_iterations
        if gradient_norm is not None:
            details['gradient_norm'] = gradient_norm
        if cost_change is not None:
            details['cost_change'] = cost_change
            
        super().__init__(message, details)
        self.iterations = iterations
        self.final_value = final_value


class LineSearchError(OptimizationError):
    """Raised when line search fails to find acceptable step size.
    
    This error occurs when the line search algorithm cannot find
    a step size that satisfies the descent conditions.
    """
    
    def __init__(self, message: str, step_size=None, method: str = "backtracking",
                 max_iterations: Optional[int] = None):
        """Initialize the error.
        
        Args:
            message: Error message
            step_size: Last attempted step size
            method: Line search method that failed
            max_iterations: Maximum iterations attempted
        """
        details = {'method': method}
        if step_size is not None:
            details['step_size'] = step_size
        if max_iterations is not None:
            details['max_iterations'] = max_iterations
            
        super().__init__(message, details)
        self.step_size = step_size


class InvalidConfigurationError(OptimizationError):
    """Raised when optimizer configuration is invalid.
    
    This error indicates that the provided configuration parameters
    are invalid or incompatible.
    """
    
    def __init__(self, message: str, parameter: Optional[str] = None, 
                 value: Any = None, reason: Optional[str] = None):
        """Initialize the error.
        
        Args:
            message: Error message
            parameter: Name of the invalid parameter
            value: Invalid value
            reason: Explanation of why the value is invalid
        """
        details = {}
        if parameter:
            details['parameter'] = parameter
        if value is not None:
            details['value'] = value
        if reason:
            details['reason'] = reason
            
        super().__init__(message, details)


class NumericalError(RiemannOptError):
    """Raised when numerical issues occur.
    
    This error indicates numerical instability such as overflow,
    underflow, or NaN values.
    """
    
    def __init__(self, message: str, operation: Optional[str] = None, 
                 details: Optional[str] = None):
        """Initialize the error.
        
        Args:
            message: Error message
            operation: Operation that caused the error
            details: Additional details about the error
        """
        error_details = {}
        if operation:
            error_details['operation'] = operation
        if details:
            error_details['details'] = details
            
        super().__init__(message, error_details)


class GradientError(OptimizationError):
    """Raised when gradient computation fails.
    
    This error occurs when the gradient cannot be computed,
    either due to missing implementation or numerical issues.
    """
    
    def __init__(self, message: str, relative_error=None, tolerance=None,
                 function_name: Optional[str] = None, reason: Optional[str] = None):
        """Initialize the error.
        
        Args:
            message: Error message
            relative_error: Relative error in gradient check
            tolerance: Tolerance used for check
            function_name: Name of the function
            reason: Reason for failure
        """
        details = {}
        if relative_error is not None:
            details['relative_error'] = relative_error
        if tolerance is not None:
            details['tolerance'] = tolerance
        if function_name:
            details['function_name'] = function_name
        if reason:
            details['reason'] = reason
            
        super().__init__(message, details)
        self.relative_error = relative_error
        self.tolerance = tolerance


class NotImplementedError(RiemannOptError):
    """Raised when a feature is not yet implemented.
    
    This is used for features that are planned but not yet available.
    """
    
    def __init__(self, feature: str, suggestion: Optional[str] = None):
        """Initialize the error.
        
        Args:
            feature: Description of the missing feature
            suggestion: Optional suggestion for alternatives
        """
        message = f"Feature not implemented: {feature}"
        details = {'feature': feature}
        
        if suggestion:
            message += f". {suggestion}"
            details['suggestion'] = suggestion
        
        super().__init__(message, details)


# Validation helper functions
def validate_point_on_manifold(manifold: Any, point: Any, tolerance: float = 1e-10) -> None:
    """Validate that a point is on the manifold.
    
    Args:
        manifold: The manifold object
        point: Point to validate
        tolerance: Tolerance for constraint satisfaction
        
    Raises:
        InvalidPointError: If point is not on manifold
    """
    if hasattr(manifold, 'check_point') and not manifold.check_point(point, tolerance):
        raise InvalidPointError(
            f"Point is not on {manifold.__class__.__name__}",
            point=point,
            manifold=manifold,
            manifold_name=manifold.__class__.__name__,
            point_shape=point.shape if hasattr(point, 'shape') else None
        )


def validate_tangent_vector(manifold: Any, point: Any, vector: Any, 
                          tolerance: float = 1e-10) -> None:
    """Validate that a vector is in the tangent space.
    
    Args:
        manifold: The manifold object
        point: Base point on manifold
        vector: Vector to validate
        tolerance: Tolerance for constraint satisfaction
        
    Raises:
        InvalidTangentError: If vector is not in tangent space
    """
    if hasattr(manifold, 'check_vector') and not manifold.check_vector(point, vector, tolerance):
        raise InvalidTangentError(
            f"Vector is not in tangent space of {manifold.__class__.__name__}",
            vector=vector,
            point=point,
            manifold=manifold
        )


def handle_rust_error(rust_error_msg: str) -> RiemannOptError:
    """Convert Rust error messages to appropriate Python exceptions.
    
    Args:
        rust_error_msg: Error message from Rust
        
    Returns:
        Appropriate Python exception
    """
    msg_lower = rust_error_msg.lower()
    
    if "invalid point" in msg_lower or "not on manifold" in msg_lower:
        return InvalidPointError(rust_error_msg)
    elif "invalid tangent" in msg_lower or "not in tangent space" in msg_lower:
        return InvalidTangentError(rust_error_msg)
    elif "dimension" in msg_lower and "mismatch" in msg_lower:
        return DimensionMismatchError(rust_error_msg)
    elif "line search" in msg_lower and "failed" in msg_lower:
        return LineSearchError(rust_error_msg)
    elif "convergence" in msg_lower or "max iterations" in msg_lower:
        return ConvergenceError(rust_error_msg)
    elif "numerical" in msg_lower or "instability" in msg_lower:
        return NumericalError(rust_error_msg)
    elif "gradient" in msg_lower:
        return GradientError(rust_error_msg)
    elif "not implemented" in msg_lower:
        return NotImplementedError(rust_error_msg)
    else:
        return RiemannOptError(rust_error_msg)