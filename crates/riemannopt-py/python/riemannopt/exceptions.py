"""Custom exceptions for RiemannOpt.

This module defines Python exceptions that correspond to errors from the Rust core,
providing a rich hierarchy of exceptions for better error handling in Python code.
"""

from typing import Optional, Dict, Any


class RiemannOptError(Exception):
    """Base exception for all RiemannOpt errors.
    
    This is the root of the exception hierarchy. All other RiemannOpt
    exceptions inherit from this class.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ManifoldValidationError(RiemannOptError):
    """Raised when a point or vector fails manifold validation.
    
    This error indicates that a point is not on the manifold or a vector
    is not in the tangent space.
    
    Attributes:
        point_info: Information about the invalid point
        manifold_name: Name of the manifold
        validation_type: Type of validation that failed ('point' or 'tangent')
    """
    
    def __init__(self, message: str, point_info: Optional[str] = None,
                 manifold_name: Optional[str] = None, 
                 validation_type: str = 'point'):
        details = {
            'point_info': point_info,
            'manifold_name': manifold_name,
            'validation_type': validation_type
        }
        super().__init__(message, details)
        self.point_info = point_info
        self.manifold_name = manifold_name
        self.validation_type = validation_type


class OptimizationFailedError(RiemannOptError):
    """Raised when an optimization algorithm fails.
    
    This is a base class for optimization-related errors.
    
    Attributes:
        algorithm: Name of the optimization algorithm
        iteration: Iteration at which the error occurred
        current_cost: Cost value at the time of failure
    """
    
    def __init__(self, message: str, algorithm: Optional[str] = None,
                 iteration: Optional[int] = None,
                 current_cost: Optional[float] = None):
        details = {
            'algorithm': algorithm,
            'iteration': iteration,
            'current_cost': current_cost
        }
        super().__init__(message, details)
        self.algorithm = algorithm
        self.iteration = iteration
        self.current_cost = current_cost


class ConvergenceError(OptimizationFailedError):
    """Raised when an optimization algorithm fails to converge.
    
    This error indicates that the algorithm reached the maximum number
    of iterations without satisfying the convergence criteria.
    
    Attributes:
        tolerance: The convergence tolerance that was not met
        gradient_norm: Final gradient norm
        max_iterations: Maximum iterations allowed
    """
    
    def __init__(self, message: str, tolerance: Optional[float] = None,
                 gradient_norm: Optional[float] = None,
                 max_iterations: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.tolerance = tolerance
        self.gradient_norm = gradient_norm
        self.max_iterations = max_iterations
        self.details.update({
            'tolerance': tolerance,
            'gradient_norm': gradient_norm,
            'max_iterations': max_iterations
        })


class LineSearchError(OptimizationFailedError):
    """Raised when line search fails to find a suitable step size.
    
    Attributes:
        initial_step: Initial step size attempted
        final_step: Final step size (if any)
        num_evaluations: Number of function evaluations performed
    """
    
    def __init__(self, message: str, initial_step: Optional[float] = None,
                 final_step: Optional[float] = None,
                 num_evaluations: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.initial_step = initial_step
        self.final_step = final_step
        self.num_evaluations = num_evaluations
        self.details.update({
            'initial_step': initial_step,
            'final_step': final_step,
            'num_evaluations': num_evaluations
        })


class DimensionMismatchError(RiemannOptError):
    """Raised when array dimensions don't match expected values.
    
    Attributes:
        expected: Expected dimensions
        got: Actual dimensions received
        operation: Operation being performed
    """
    
    def __init__(self, message: str, expected: Optional[tuple] = None,
                 got: Optional[tuple] = None, operation: Optional[str] = None):
        details = {
            'expected': expected,
            'got': got,
            'operation': operation
        }
        super().__init__(message, details)
        self.expected = expected
        self.got = got
        self.operation = operation


class ConfigurationError(RiemannOptError):
    """Raised when there's an error in configuration.
    
    Attributes:
        parameter: Configuration parameter that caused the error
        value: Invalid value provided
        valid_range: Valid range or options for the parameter
    """
    
    def __init__(self, message: str, parameter: Optional[str] = None,
                 value: Any = None, valid_range: Any = None):
        details = {
            'parameter': parameter,
            'value': value,
            'valid_range': valid_range
        }
        super().__init__(message, details)
        self.parameter = parameter
        self.value = value
        self.valid_range = valid_range


class BackendError(RiemannOptError):
    """Raised when there's an error with computational backend.
    
    Attributes:
        backend: Name of the backend (cpu, cuda, etc.)
        reason: Reason for the failure
    """
    
    def __init__(self, message: str, backend: Optional[str] = None,
                 reason: Optional[str] = None):
        details = {
            'backend': backend,
            'reason': reason
        }
        super().__init__(message, details)
        self.backend = backend
        self.reason = reason


# Exception mapping for cleaner error messages
def format_rust_error(error: Exception) -> str:
    """Format a Rust error into a more readable Python error message.
    
    Args:
        error: The original exception from Rust
        
    Returns:
        A formatted error message
    """
    error_str = str(error)
    
    # Remove Rust-specific formatting
    if "Error: " in error_str:
        error_str = error_str.split("Error: ", 1)[-1]
    
    # Clean up common Rust error patterns
    error_str = error_str.replace("\\n", "\n")
    error_str = error_str.strip()
    
    return error_str


__all__ = [
    'RiemannOptError',
    'ManifoldValidationError',
    'OptimizationFailedError',
    'ConvergenceError',
    'LineSearchError',
    'DimensionMismatchError',
    'ConfigurationError',
    'BackendError',
    'format_rust_error'
]