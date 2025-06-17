"""Logging configuration for RiemannOpt."""

import logging
import sys
import functools
import time
from typing import Optional, Callable, Any
from .config import get_config

# Cache for loggers
_loggers = {}


def get_logger(name: str) -> logging.Logger:
    """Get a logger for RiemannOpt.
    
    Args:
        name: Logger name (will be prefixed with 'riemannopt.')
        
    Returns:
        Configured logger instance
        
    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting optimization")
    """
    full_name = f"riemannopt.{name}" if not name.startswith("riemannopt") else name
    
    # Return cached logger if available
    if full_name in _loggers:
        return _loggers[full_name]
    
    logger = logging.getLogger(full_name)
    
    # Configure only if not already configured
    if not logger.handlers:
        config = get_config()
        
        # Create handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(config.log_format))
        
        # Add handler and set level
        logger.addHandler(handler)
        logger.setLevel(config.log_level)
        logger.propagate = False
    
    # Cache logger
    _loggers[full_name] = logger
    return logger


def log_function_call(
    logger: Optional[logging.Logger] = None,
    level: int = logging.DEBUG,
    include_args: bool = False,
    include_result: bool = False,
    include_time: bool = True
) -> Callable:
    """Decorator to log function calls.
    
    Args:
        logger: Logger to use (creates one if None)
        level: Log level to use
        include_args: Whether to log function arguments
        include_result: Whether to log function result
        include_time: Whether to log execution time
        
    Example:
        >>> @log_function_call(include_time=True)
        ... def optimize_sphere(x):
        ...     return x / np.linalg.norm(x)
    """
    def decorator(func: Callable) -> Callable:
        nonlocal logger
        if logger is None:
            logger = get_logger(func.__module__)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Build log message
            msg_parts = [f"Calling {func.__name__}"]
            
            if include_args:
                args_str = ", ".join(repr(a) for a in args)
                kwargs_str = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
                all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                if all_args:
                    msg_parts.append(f"({all_args})")
            
            logger.log(level, " ".join(msg_parts))
            
            # Execute function
            start_time = time.perf_counter() if include_time else None
            try:
                result = func(*args, **kwargs)
                
                # Log success
                success_parts = [f"{func.__name__} completed"]
                
                if include_time and start_time is not None:
                    elapsed = time.perf_counter() - start_time
                    success_parts.append(f"in {elapsed:.3f}s")
                
                if include_result:
                    success_parts.append(f"-> {result!r}")
                
                logger.log(level, " ".join(success_parts))
                return result
                
            except Exception as e:
                error_parts = [f"{func.__name__} failed"]
                
                if include_time and start_time is not None:
                    elapsed = time.perf_counter() - start_time
                    error_parts.append(f"after {elapsed:.3f}s")
                
                error_parts.append(f": {type(e).__name__}: {e}")
                logger.error(" ".join(error_parts))
                raise
        
        return wrapper
    return decorator


def log_optimizer_step(logger: logging.Logger) -> Callable:
    """Decorator specifically for optimizer step methods.
    
    Logs iteration number, cost value, and gradient norm.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs) -> Any:
            result = func(self, *args, **kwargs)
            
            # Extract info from result if it's a dict
            if isinstance(result, dict):
                iteration = result.get('iteration', '?')
                cost = result.get('cost', '?')
                grad_norm = result.get('gradient_norm', '?')
                
                logger.debug(
                    f"Step {iteration}: cost={cost:.6e}, |grad|={grad_norm:.6e}"
                )
            
            return result
        return wrapper
    return decorator


class OptimizationLogger:
    """Context manager for optimization logging.
    
    Provides structured logging for optimization runs with
    automatic summary at the end.
    
    Example:
        >>> with OptimizationLogger("sphere_optimization") as opt_logger:
        ...     result = optimizer.optimize(cost_fn, x0)
        ...     opt_logger.log_result(result)
    """
    
    def __init__(self, name: str, level: int = logging.INFO):
        self.logger = get_logger(name)
        self.level = level
        self.start_time = None
        self.metrics = {}
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.logger.log(self.level, "=" * 60)
        self.logger.log(self.level, f"Starting optimization: {self.logger.name}")
        self.logger.log(self.level, "=" * 60)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.perf_counter() - self.start_time
        
        if exc_type is None:
            self.logger.log(self.level, "-" * 60)
            self.logger.log(self.level, "Optimization completed successfully")
            self.logger.log(self.level, f"Total time: {elapsed:.3f}s")
            
            # Log collected metrics
            if self.metrics:
                self.logger.log(self.level, "Summary:")
                for key, value in self.metrics.items():
                    self.logger.log(self.level, f"  {key}: {value}")
        else:
            self.logger.error("-" * 60)
            self.logger.error(f"Optimization failed after {elapsed:.3f}s")
            self.logger.error(f"Error: {exc_type.__name__}: {exc_val}")
        
        self.logger.log(self.level, "=" * 60)
    
    def log_iteration(self, iteration: int, cost: float, gradient_norm: float):
        """Log iteration details."""
        self.logger.debug(
            f"Iteration {iteration}: cost={cost:.6e}, |grad|={gradient_norm:.6e}"
        )
    
    def log_result(self, result: dict):
        """Log optimization result."""
        self.metrics.update({
            'Final cost': f"{result.get('cost', 'N/A'):.6e}",
            'Iterations': result.get('iterations', 'N/A'),
            'Gradient norm': f"{result.get('gradient_norm', 'N/A'):.6e}",
            'Success': result.get('success', False),
        })
    
    def add_metric(self, name: str, value: Any):
        """Add a custom metric to the summary."""
        self.metrics[name] = value