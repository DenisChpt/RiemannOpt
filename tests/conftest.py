"""Shared fixtures for RiemannOpt Python tests."""

from __future__ import annotations

import numpy as np
import pytest

from riemannopt import create_cost_function
from riemannopt.manifolds import (
    Euclidean,
    Grassmann,
    Hyperbolic,
    Oblique,
    PSDCone,
    SPD,
    Sphere,
    Stiefel,
)


# ---------------------------------------------------------------------------
# Manifold fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sphere() -> Sphere:
    return Sphere(10)


@pytest.fixture
def stiefel() -> Stiefel:
    return Stiefel(6, 3)


@pytest.fixture
def grassmann() -> Grassmann:
    return Grassmann(8, 3)


@pytest.fixture
def spd() -> SPD:
    return SPD(4)


@pytest.fixture
def euclidean() -> Euclidean:
    return Euclidean(10)


@pytest.fixture
def hyperbolic() -> Hyperbolic:
    return Hyperbolic(5)


@pytest.fixture
def oblique() -> Oblique:
    return Oblique(4, 3)


@pytest.fixture
def psd_cone() -> PSDCone:
    return PSDCone(4)


# ---------------------------------------------------------------------------
# Cost function fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rayleigh_cost():
    """Rayleigh quotient x^T A x on S^{n-1}.  Minimum = smallest eigenvalue."""
    n = 10
    A = np.diag(np.arange(1, n + 1, dtype=np.float64))

    def cost(x: np.ndarray) -> float:
        return float(x @ A @ x)

    def grad(x: np.ndarray) -> np.ndarray:
        return 2.0 * A @ x

    return create_cost_function(cost, gradient=grad, dimension=n), A


@pytest.fixture
def trace_cost():
    """Trace minimisation tr(Y^T A Y) on Gr(n, p).  Minimum = sum of p smallest eigenvalues."""
    n, p = 8, 3
    A = np.diag(np.arange(1, n + 1, dtype=np.float64))

    def cost(Y: np.ndarray) -> float:
        return float(np.trace(Y.T @ A @ Y))

    def grad(Y: np.ndarray) -> np.ndarray:
        return 2.0 * A @ Y

    return create_cost_function(cost, gradient=grad, dimension=(n, p)), A, p
