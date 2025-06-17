"""Compatibility layer for optional dependencies."""

import importlib
import warnings
from typing import Optional, Any

def import_optional_dependency(
    name: str, 
    package: Optional[str] = None,
    min_version: Optional[str] = None
) -> Any:
    """Import an optional dependency with informative error.
    
    Args:
        name: Module name to import
        package: Package name for error message (defaults to name)
        min_version: Minimum required version (optional)
        
    Returns:
        Imported module
        
    Raises:
        ImportError: If module cannot be imported with helpful message
    """
    package = package or name
    
    try:
        module = importlib.import_module(name)
    except ImportError as e:
        raise ImportError(
            f"{package} is required for this functionality. "
            f"Install it with: pip install 'riemannopt[{package.lower()}]'"
        ) from e
    
    # Check version if specified
    if min_version and hasattr(module, "__version__"):
        from packaging import version
        if version.parse(module.__version__) < version.parse(min_version):
            raise ImportError(
                f"{package} version {min_version} or higher is required, "
                f"but {module.__version__} is installed."
            )
    
    return module

# Check for optional dependencies at import time
HAS_MATPLOTLIB = False
HAS_TORCH = False
HAS_JAX = False
HAS_TQDM = False

try:
    import matplotlib
    HAS_MATPLOTLIB = True
except ImportError:
    pass

try:
    import torch
    HAS_TORCH = True
except ImportError:
    pass

try:
    import jax
    HAS_JAX = True
except ImportError:
    pass

try:
    import tqdm
    HAS_TQDM = True
except ImportError:
    pass

def check_visualization_deps():
    """Check if visualization dependencies are available."""
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "Visualization requires matplotlib. "
            "Install with: pip install 'riemannopt[viz]'"
        )

def check_torch_deps():
    """Check if PyTorch dependencies are available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch integration requires torch. "
            "Install with: pip install 'riemannopt[torch]'"
        )

def check_jax_deps():
    """Check if JAX dependencies are available."""
    if not HAS_JAX:
        raise ImportError(
            "JAX integration requires jax. "
            "Install with: pip install 'riemannopt[jax]'"
        )