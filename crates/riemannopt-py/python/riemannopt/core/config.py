"""Configuration management for RiemannOpt."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import os
import threading


@dataclass
class RiemannOptConfig:
    """Global configuration for RiemannOpt.
    
    This class manages all configuration options for the library.
    Settings can be modified at runtime and affect global behavior.
    
    Attributes:
        default_dtype: Default data type for arrays ('float32' or 'float64')
        epsilon: Small value for numerical stability
        gradient_tolerance: Tolerance for gradient convergence checks
        cost_tolerance: Tolerance for cost function convergence
        max_iterations: Default maximum iterations for optimizers
        check_gradients: Whether to validate gradients with finite differences
        validate_inputs: Whether to validate input arrays and shapes
        num_threads: Number of threads for parallel operations (None for auto)
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_format: Format string for log messages
        show_progress: Whether to show progress bars (if tqdm available)
        cache_dir: Directory for caching computed values
        random_seed: Random seed for reproducibility (None for no seed)
    """
    
    # Numerical settings
    default_dtype: str = "float64"
    epsilon: float = 1e-10
    gradient_tolerance: float = 1e-6
    cost_tolerance: float = 1e-8
    max_iterations: int = 1000
    
    # Validation settings
    check_gradients: bool = False
    validate_inputs: bool = True
    check_retraction_order: bool = False
    
    # Computation settings
    num_threads: Optional[int] = None
    use_parallel: bool = True
    
    # Logging settings
    log_level: str = "WARNING"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    show_progress: bool = True
    
    # Paths and caching
    cache_dir: Optional[str] = None
    enable_caching: bool = True
    
    # Random settings
    random_seed: Optional[int] = None
    
    @classmethod
    def from_env(cls) -> 'RiemannOptConfig':
        """Create configuration from environment variables.
        
        Environment variables:
            RIEMANNOPT_DTYPE: Default data type
            RIEMANNOPT_CHECK_GRADIENTS: Enable gradient checking
            RIEMANNOPT_NUM_THREADS: Number of threads
            RIEMANNOPT_LOG_LEVEL: Logging level
            RIEMANNOPT_SHOW_PROGRESS: Show progress bars
            RIEMANNOPT_RANDOM_SEED: Random seed
        """
        def parse_bool(value: str) -> bool:
            return value.lower() in ('true', '1', 'yes', 'on')
        
        def parse_int(value: str) -> Optional[int]:
            return int(value) if value and value != '0' else None
        
        return cls(
            default_dtype=os.getenv('RIEMANNOPT_DTYPE', 'float64'),
            check_gradients=parse_bool(os.getenv('RIEMANNOPT_CHECK_GRADIENTS', '')),
            num_threads=parse_int(os.getenv('RIEMANNOPT_NUM_THREADS', '')),
            log_level=os.getenv('RIEMANNOPT_LOG_LEVEL', 'WARNING'),
            show_progress=parse_bool(os.getenv('RIEMANNOPT_SHOW_PROGRESS', 'true')),
            random_seed=parse_int(os.getenv('RIEMANNOPT_RANDOM_SEED', '')),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def update(self, **kwargs) -> None:
        """Update configuration with keyword arguments."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration option: {key}")


# Thread-local storage for context managers
_config_stack = threading.local()

# Global configuration instance
_global_config = RiemannOptConfig.from_env()


def get_config() -> RiemannOptConfig:
    """Get the current configuration.
    
    Returns the context-local configuration if in a context manager,
    otherwise returns the global configuration.
    """
    stack = getattr(_config_stack, 'stack', None)
    if stack:
        return stack[-1]
    return _global_config


def set_config(**kwargs) -> None:
    """Update global configuration.
    
    Args:
        **kwargs: Configuration options to update
        
    Example:
        >>> set_config(check_gradients=True, log_level='DEBUG')
    """
    _global_config.update(**kwargs)


def reset_config() -> None:
    """Reset configuration to defaults."""
    global _global_config
    _global_config = RiemannOptConfig()


class config_context:
    """Context manager for temporary configuration changes.
    
    Example:
        >>> with config_context(check_gradients=True):
        ...     # Gradient checking enabled here
        ...     optimize(...)
        >>> # Original configuration restored
    """
    
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.original_config = None
    
    def __enter__(self):
        if not hasattr(_config_stack, 'stack'):
            _config_stack.stack = []
        
        # Get current config and create a copy
        current = get_config()
        new_config = RiemannOptConfig(**current.to_dict())
        new_config.update(**self.kwargs)
        
        _config_stack.stack.append(new_config)
        return new_config
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        _config_stack.stack.pop()