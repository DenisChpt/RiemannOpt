"""Core functionality for RiemannOpt."""

from .api import optimize, create_cost_function
from .config import get_config, set_config, RiemannOptConfig, config_context
from .logging import get_logger

__all__ = [
    'optimize',
    'create_cost_function',
    'get_config',
    'set_config',
    'RiemannOptConfig',
    'config_context',
    'get_logger',
]