"""
Module to handle riemannopt imports for tests.
"""

HAS_RIEMANNOPT = False
riemannopt = None

try:
    import riemannopt
    HAS_RIEMANNOPT = True
except ImportError:
    pass