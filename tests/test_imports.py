"""
Central import module for tests.

This module handles the import of riemannopt with proper error handling.
"""

import pytest

try:
    import riemannopt._riemannopt as riemannopt
    HAS_RIEMANNOPT = True
except ImportError:
    HAS_RIEMANNOPT = False
    riemannopt = None

# Skip all tests if riemannopt is not available
if not HAS_RIEMANNOPT:
    pytestmark = pytest.mark.skip(reason="riemannopt module not installed")