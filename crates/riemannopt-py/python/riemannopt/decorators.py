"""Decorators and utilities for enhanced Python interface."""

import numpy as np
import functools
import warnings
from typing import Callable, Optional, Any, Union
from .exceptions import handle_rust_error, DimensionMismatchError, InvalidPointError


def validate_arrays(*array_args, shapes=None, dtypes=None):
    """Decorator to validate numpy array arguments.
    
    Args:
        *array_args: Names of arguments that should be numpy arrays
        shapes: Expected shapes for each array (optional)
        dtypes: Expected dtypes for each array (optional)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get function signature to map positional args to names
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            for i, arg_name in enumerate(array_args):
                if arg_name in bound_args.arguments:
                    value = bound_args.arguments[arg_name]
                    
                    # Convert to numpy array if needed
                    if not isinstance(value, np.ndarray):
                        try:
                            value = np.asarray(value, dtype=np.float64)
                            bound_args.arguments[arg_name] = value
                        except Exception as e:
                            raise ValueError(f"Cannot convert {arg_name} to numpy array: {e}")
                    
                    # Check dtype
                    if dtypes and i < len(dtypes) and dtypes[i] is not None:
                        if value.dtype != dtypes[i]:
                            value = value.astype(dtypes[i])
                            bound_args.arguments[arg_name] = value
                    
                    # Check shape
                    if shapes and i < len(shapes) and shapes[i] is not None:
                        expected_shape = shapes[i]
                        if value.shape != expected_shape:
                            raise DimensionMismatchError(
                                f"Expected shape {expected_shape} for {arg_name}, got {value.shape}",
                                expected_shape=expected_shape,
                                actual_shape=value.shape
                            )
            
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


def ensure_on_manifold(manifold_arg: str = 'self', point_args: list = None):
    """Decorator to ensure points are on the manifold.
    
    Args:
        manifold_arg: Name of the manifold argument
        point_args: List of point argument names to check
    """
    if point_args is None:
        point_args = ['point']
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            import inspect
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            
            # Get manifold object
            manifold = bound_args.arguments.get(manifold_arg)
            if manifold is None:
                return func(*args, **kwargs)  # Skip if no manifold
            
            # Check and project points if needed
            for point_arg in point_args:
                if point_arg in bound_args.arguments:
                    point = bound_args.arguments[point_arg]
                    
                    if hasattr(manifold, 'check_point_on_manifold'):
                        if not manifold.check_point_on_manifold(point):
                            warnings.warn(f"Point {point_arg} not on manifold, projecting...")
                            projected = manifold.project(point)
                            bound_args.arguments[point_arg] = projected
            
            return func(*bound_args.args, **bound_args.kwargs)
        return wrapper
    return decorator


def handle_rust_exceptions(func):
    """Decorator to convert Rust exceptions to appropriate Python exceptions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Convert Rust errors to Python exceptions
            if hasattr(e, 'args') and e.args:
                error_msg = str(e.args[0]) if e.args else str(e)
                python_exception = handle_rust_error(error_msg)
                raise python_exception from e
            else:
                raise
    return wrapper


def deprecated(reason: str):
    """Mark a function as deprecated."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            warnings.warn(
                f"{func.__name__} is deprecated: {reason}",
                DeprecationWarning,
                stacklevel=2
            )
            return func(*args, **kwargs)
        return wrapper
    return decorator


def cache_result(maxsize: int = 128):
    """Cache function results (useful for expensive manifold operations)."""
    def decorator(func):
        @functools.lru_cache(maxsize=maxsize)
        def wrapper(*args, **kwargs):
            # Convert numpy arrays to tuples for hashing
            hashable_args = []
            hashable_kwargs = {}
            
            for arg in args:
                if isinstance(arg, np.ndarray):
                    hashable_args.append(tuple(arg.flatten()))
                else:
                    hashable_args.append(arg)
            
            for key, value in kwargs.items():
                if isinstance(value, np.ndarray):
                    hashable_kwargs[key] = tuple(value.flatten())
                else:
                    hashable_kwargs[key] = value
            
            return func(*tuple(hashable_args), **hashable_kwargs)
        return wrapper
    return decorator


def time_function(func):
    """Decorator to time function execution."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        
        # Store timing info as function attribute
        wrapper.last_execution_time = end_time - start_time
        
        return result
    return wrapper


def require_gradient(func):
    """Decorator to ensure gradient function is available."""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        if not hasattr(self, 'gradient') or self.gradient is None:
            raise ValueError(f"{func.__name__} requires a gradient function")
        return func(self, *args, **kwargs)
    return wrapper


class property_cached:
    """Cached property decorator for expensive computations."""
    
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
        self.__doc__ = func.__doc__
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        
        value = obj.__dict__.get(self.name, None)
        if value is None:
            value = self.func(obj)
            obj.__dict__[self.name] = value
        return value
    
    def __set__(self, obj, value):
        obj.__dict__[self.name] = value
    
    def __delete__(self, obj):
        obj.__dict__.pop(self.name, None)


def vectorize_manifold_operation(signature: str = "(n),()->(n)"):
    """Vectorize manifold operations to work with batches."""
    def decorator(func):
        vectorized = np.vectorize(func, signature=signature)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check if we need to vectorize
            has_batch = any(
                isinstance(arg, np.ndarray) and arg.ndim > 1 
                for arg in args
            )
            
            if has_batch:
                return vectorized(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        
        return wrapper
    return decorator