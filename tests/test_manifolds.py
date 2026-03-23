"""Tests for all manifold implementations.

Each manifold must satisfy a common geometric contract plus manifold-specific
constraints.  The parametrised ``TestManifoldContract`` class validates the
generic contract, while dedicated test classes cover manifold-specific behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest

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

# ── helpers ──────────────────────────────────────────────────────────────────

TOL = 1e-10


def _make(manifold):
    """Return (manifold, point, tangent)."""
    x = manifold.random_point()
    v = manifold.random_tangent(x)
    return manifold, x, v


# ── Geometric contract (parametrised) ────────────────────────────────────────

# Manifolds that follow the "standard" API (contains, project, random_tangent(x, scale=...))
_MANIFOLDS = [
    ("Sphere", lambda: Sphere(10)),
    ("Stiefel", lambda: Stiefel(6, 3)),
    ("Grassmann", lambda: Grassmann(8, 3)),
    ("SPD", lambda: SPD(4)),
    ("Oblique", lambda: Oblique(4, 3)),
    ("PSDCone", lambda: PSDCone(4)),
]


@pytest.mark.parametrize("name,factory", _MANIFOLDS, ids=[m[0] for m in _MANIFOLDS])
class TestManifoldContract:
    """Every manifold must pass these generic geometric tests."""

    def test_dim_positive(self, name, factory):
        m = factory()
        assert m.dim > 0

    def test_random_point_on_manifold(self, name, factory):
        m = factory()
        for _ in range(5):
            x = m.random_point()
            assert m.contains(x, atol=TOL), f"{name}: random_point not on manifold"

    def test_project_idempotent(self, name, factory):
        m = factory()
        x = m.random_point()
        xp = m.project(x)
        xpp = m.project(xp)
        np.testing.assert_allclose(xp, xpp, atol=1e-13)

    def test_retract_stays_on_manifold(self, name, factory):
        m = factory()
        x = m.random_point()
        v = m.random_tangent(x)
        for scale in [0.01, 0.1]:
            y = m.retract(x, scale * v)
            assert m.contains(y, atol=1e-6), (
                f"{name}: retract left manifold at scale {scale}"
            )

    def test_zero_retract_is_identity(self, name, factory):
        m = factory()
        x = m.random_point()
        zero = np.zeros_like(x)
        y = m.retract(x, zero)
        np.testing.assert_allclose(y, x, atol=1e-13)

    def test_inner_product_symmetric(self, name, factory):
        m = factory()
        x = m.random_point()
        u = m.random_tangent(x)
        v = m.random_tangent(x)
        assert abs(m.inner(x, u, v) - m.inner(x, v, u)) < 1e-12

    def test_inner_product_positive(self, name, factory):
        m = factory()
        x = m.random_point()
        v = m.random_tangent(x)
        assert m.inner(x, v, v) >= -1e-14

    def test_norm_nonneg(self, name, factory):
        m = factory()
        x = m.random_point()
        v = m.random_tangent(x)
        assert m.norm(x, v) >= -1e-14

    def test_tangent_projection_tangent(self, name, factory):
        m = factory()
        x = m.random_point()
        v_raw = np.random.randn(*x.shape)
        v = m.project_tangent(x, v_raw)
        # Re-projecting should be idempotent
        v2 = m.project_tangent(x, v)
        np.testing.assert_allclose(v, v2, atol=1e-12)


# ── Sphere-specific ─────────────────────────────────────────────────────────

class TestSphere:
    def test_point_has_unit_norm(self, sphere):
        for _ in range(10):
            x = sphere.random_point()
            np.testing.assert_allclose(np.linalg.norm(x), 1.0, atol=1e-14)

    def test_tangent_orthogonal_to_point(self, sphere):
        x = sphere.random_point()
        v = sphere.random_tangent(x)
        assert abs(x @ v) < 1e-12

    def test_exp_stays_on_sphere(self, sphere):
        x = sphere.random_point()
        v = sphere.random_tangent(x)
        y = sphere.exp(x, 0.5 * v)
        np.testing.assert_allclose(np.linalg.norm(y), 1.0, atol=1e-14)

    def test_exp_log_inverse(self, sphere):
        x = sphere.random_point()
        v = sphere.random_tangent(x)
        v_small = 0.01 * v / max(np.linalg.norm(v), 1e-15)
        y = sphere.exp(x, v_small)
        v_recovered = sphere.log(x, y)
        np.testing.assert_allclose(v_small, v_recovered, atol=1e-6)

    def test_distance_symmetric(self, sphere):
        x = sphere.random_point()
        y = sphere.random_point()
        np.testing.assert_allclose(sphere.distance(x, y), sphere.distance(y, x), atol=1e-14)

    def test_parallel_transport_preserves_norm(self, sphere):
        x = sphere.random_point()
        v = sphere.random_tangent(x)
        u = sphere.random_tangent(x)
        y = sphere.exp(x, 0.3 * v / max(np.linalg.norm(v), 1e-15))
        u_t = sphere.parallel_transport(x, y, u)
        np.testing.assert_allclose(np.linalg.norm(u), np.linalg.norm(u_t), atol=1e-11)


# ── Stiefel-specific ────────────────────────────────────────────────────────

class TestStiefel:
    def test_orthonormality(self, stiefel):
        X = stiefel.random_point()
        np.testing.assert_allclose(X.T @ X, np.eye(stiefel.p), atol=1e-13)

    def test_tangent_skew_symmetric(self, stiefel):
        X = stiefel.random_point()
        V = stiefel.random_tangent(X)
        sym = X.T @ V + V.T @ X
        np.testing.assert_allclose(sym, 0, atol=1e-12)

    def test_retract_preserves_orth(self, stiefel):
        X = stiefel.random_point()
        V = stiefel.random_tangent(X)
        Y = stiefel.retract(X, 0.1 * V)
        np.testing.assert_allclose(Y.T @ Y, np.eye(stiefel.p), atol=1e-12)


# ── Grassmann-specific ──────────────────────────────────────────────────────

class TestGrassmann:
    def test_point_has_orthonormal_cols(self, grassmann):
        Y = grassmann.random_point()
        np.testing.assert_allclose(Y.T @ Y, np.eye(grassmann.p), atol=1e-13)

    def test_tangent_horizontal(self, grassmann):
        Y = grassmann.random_point()
        Z = grassmann.random_tangent(Y)
        np.testing.assert_allclose(Y.T @ Z, 0, atol=1e-12)


# ── SPD-specific ────────────────────────────────────────────────────────────

class TestSPD:
    def test_random_point_is_spd(self, spd):
        P = spd.random_point()
        np.testing.assert_allclose(P, P.T, atol=1e-12)
        eigvals = np.linalg.eigvalsh(P)
        assert np.all(eigvals > 0), f"Not positive definite: {eigvals}"

    def test_tangent_is_symmetric(self, spd):
        P = spd.random_point()
        V = spd.random_tangent(P)
        np.testing.assert_allclose(V, V.T, atol=1e-12)


# ── Euclidean-specific ──────────────────────────────────────────────────────

class TestEuclidean:
    def test_retract_is_addition(self, euclidean):
        x = euclidean.random_point()
        v = euclidean.random_tangent(x)
        y = euclidean.retract(x, v)
        np.testing.assert_allclose(y, x + v, atol=1e-14)

    def test_distance_is_norm(self, euclidean):
        x = euclidean.random_point()
        y = euclidean.random_point()
        np.testing.assert_allclose(
            euclidean.distance(x, y), np.linalg.norm(x - y), atol=1e-14
        )

    def test_flat(self, euclidean):
        assert euclidean.is_flat()


# ── Hyperbolic-specific ─────────────────────────────────────────────────────

class TestHyperbolic:
    def test_random_point_inside_ball(self):
        h = Hyperbolic(5)
        x = h.random_point()
        # Poincaré ball model: point should be inside the unit ball
        assert np.linalg.norm(x) < 1.0

    def test_properties(self):
        h = Hyperbolic(5)
        assert h.dim == 5
        assert h.n == 5
        assert h.curvature < 0


# ── Oblique-specific ────────────────────────────────────────────────────────

class TestOblique:
    def test_columns_unit_norm(self, oblique):
        X = oblique.random_point()
        for j in range(X.shape[1]):
            np.testing.assert_allclose(np.linalg.norm(X[:, j]), 1.0, atol=1e-13)
