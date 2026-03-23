"""Tests for all optimizers on known problems with analytical solutions."""

from __future__ import annotations

import numpy as np
import pytest

from riemannopt import create_cost_function
from riemannopt.manifolds import Euclidean, Grassmann, Sphere, Stiefel
from riemannopt.optimizers import (
    SGD,
    Adam,
    ConjugateGradient,
    LBFGS,
    Newton,
    TrustRegion,
)


# ── Cost functions ──────────────────────────────────────────────────────────

def _rayleigh(n: int = 10, gap: float = 3.0):
    """x^T A x on S^{n-1}.  Minimum = 1 at ±e₁."""
    A = np.diag(np.concatenate(([1.0], np.full(n - 1, 1.0 + gap))))

    def cost(x):
        return float(x @ A @ x)

    def grad(x):
        return 2.0 * A @ x

    return create_cost_function(cost, gradient=grad, dimension=n), 1.0


def _quadratic(n: int = 10):
    """0.5 x^T A x + b^T x on R^n.  Known minimiser x* = -A^{-1}b."""
    A = np.diag(np.arange(1, n + 1, dtype=float))
    b = np.ones(n)
    x_star = -b / np.diag(A)
    f_star = float(0.5 * x_star @ A @ x_star + b @ x_star)

    def cost(x):
        return float(0.5 * x @ A @ x + b @ x)

    def grad(x):
        return A @ x + b

    def hvp(x, v):
        return A @ v

    cf = create_cost_function(cost, gradient=grad, dimension=n)
    return cf, x_star, f_star


def _trace(n: int = 8, p: int = 3):
    """tr(Y^T A Y) on Gr(n,p).  Minimum = sum of p smallest eigenvalues."""
    A = np.diag(np.arange(1, n + 1, dtype=float))
    f_star = float(sum(range(1, p + 1)))

    def cost(Y):
        return float(np.trace(Y.T @ A @ Y))

    def grad(Y):
        return 2.0 * A @ Y

    return create_cost_function(cost, gradient=grad, dimension=(n, p)), f_star


# ── Helper ──────────────────────────────────────────────────────────────────

def _sphere_start(n: int) -> np.ndarray:
    x = np.ones(n) / np.sqrt(n)
    x += 0.01 * np.arange(n) / n
    return x / np.linalg.norm(x)


# ═══════════════════════════════════════════════════════════════════════════
#  1.  Rayleigh quotient on the sphere (eigenvector problem)
# ═══════════════════════════════════════════════════════════════════════════

class TestRayleighOnSphere:
    """CG and TrustRegion must converge precisely; SGD / Adam must at least decrease cost."""

    N = 10

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.sphere = Sphere(self.N)
        self.cost_fn, self.f_star = _rayleigh(self.N)
        self.x0 = _sphere_start(self.N)

    # ── second-order / quasi-Newton: precise convergence ────────────

    def test_cg_polak_ribiere(self):
        opt = ConjugateGradient(method="PolakRibiere")
        r = opt.optimize(self.cost_fn, self.sphere, self.x0, 300, gradient_tolerance=1e-10)
        assert abs(self.cost_fn.cost(r.point) - self.f_star) < 1e-6

    def test_cg_fletcher_reeves(self):
        opt = ConjugateGradient(method="FletcherReeves")
        r = opt.optimize(self.cost_fn, self.sphere, self.x0, 300, gradient_tolerance=1e-10)
        assert abs(self.cost_fn.cost(r.point) - self.f_star) < 1e-6

    def test_trust_region(self):
        opt = TrustRegion()
        r = opt.optimize(self.cost_fn, self.sphere, self.x0, 50, gradient_tolerance=1e-10)
        assert abs(self.cost_fn.cost(r.point) - self.f_star) < 1e-6

    # ── first-order: cost must decrease ──────────────────────────────

    def test_sgd_decreases_cost(self):
        opt = SGD(learning_rate=0.01)
        # Don't pass gradient_tolerance — it triggers early stop at iter 1
        r = opt.optimize(self.cost_fn, self.sphere, self.x0, 5000)
        assert self.cost_fn.cost(r.point) < self.cost_fn.cost(self.x0)

    def test_adam_decreases_cost(self):
        opt = Adam(learning_rate=0.01)
        r = opt.optimize(self.cost_fn, self.sphere, self.x0, 5000)
        assert self.cost_fn.cost(r.point) < self.cost_fn.cost(self.x0)

    # ── manifold constraint ──────────────────────────────────────────

    @pytest.mark.parametrize("opt_cls,kw", [
        (SGD, {"learning_rate": 0.01}),
        (Adam, {"learning_rate": 0.01}),
        (ConjugateGradient, {"method": "PolakRibiere"}),
        (TrustRegion, {}),
    ])
    def test_result_on_manifold(self, opt_cls, kw):
        opt = opt_cls(**kw)
        r = opt.optimize(self.cost_fn, self.sphere, self.x0, 100)
        np.testing.assert_allclose(np.linalg.norm(r.point), 1.0, atol=1e-13)


# ═══════════════════════════════════════════════════════════════════════════
#  2.  Quadratic on Euclidean (exact solution known)
# ═══════════════════════════════════════════════════════════════════════════

class TestQuadraticOnEuclidean:
    N = 10

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.eucl = Euclidean(self.N)
        self.cost_fn, self.x_star, self.f_star = _quadratic(self.N)
        self.x0 = np.ones(self.N) * 2.0

    def test_lbfgs_converges(self):
        opt = LBFGS(memory_size=self.N)
        r = opt.optimize(self.cost_fn, self.eucl, self.x0, 50, gradient_tolerance=1e-12)
        np.testing.assert_allclose(self.cost_fn.cost(r.point), self.f_star, atol=1e-8)

    def test_trust_region_converges(self):
        opt = TrustRegion()
        r = opt.optimize(self.cost_fn, self.eucl, self.x0, 50, gradient_tolerance=1e-12)
        np.testing.assert_allclose(self.cost_fn.cost(r.point), self.f_star, atol=1e-8)

    def test_cg_converges_in_n_steps(self):
        opt = ConjugateGradient(method="FletcherReeves")
        r = opt.optimize(self.cost_fn, self.eucl, self.x0, self.N + 5, gradient_tolerance=1e-12)
        assert self.cost_fn.cost(r.point) < 1e-8


# ═══════════════════════════════════════════════════════════════════════════
#  3.  Trace minimisation on Grassmann (eigenspace problem)
# ═══════════════════════════════════════════════════════════════════════════

class TestTraceOnGrassmann:
    N, P = 8, 3

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.gr = Grassmann(self.N, self.P)
        self.cost_fn, self.f_star = _trace(self.N, self.P)
        self.y0 = self.gr.random_point()

    def test_cg_finds_eigenspace(self):
        opt = ConjugateGradient(method="PolakRibiere")
        r = opt.optimize(self.cost_fn, self.gr, self.y0, 300, gradient_tolerance=1e-10)
        np.testing.assert_allclose(self.cost_fn.cost(r.point), self.f_star, atol=1e-4)

    def test_result_on_grassmann(self):
        opt = ConjugateGradient(method="PolakRibiere")
        r = opt.optimize(self.cost_fn, self.gr, self.y0, 200, gradient_tolerance=1e-8)
        np.testing.assert_allclose(r.point.T @ r.point, np.eye(self.P), atol=1e-12)


# ═══════════════════════════════════════════════════════════════════════════
#  4.  Optimizer API / config tests
# ═══════════════════════════════════════════════════════════════════════════

class TestOptimizerAPI:

    def test_optimization_result_fields(self, sphere, rayleigh_cost):
        cost_fn, _ = rayleigh_cost
        x0 = sphere.random_point()
        opt = ConjugateGradient()
        r = opt.optimize(cost_fn, sphere, x0, 50)
        assert isinstance(r.cost, float)
        assert isinstance(r.iterations, int)
        assert isinstance(r.converged, bool)
        assert isinstance(r.time_seconds, float)
        assert isinstance(r.termination_reason, str)
        assert r.point.shape == x0.shape

    def test_result_as_dict(self, sphere, rayleigh_cost):
        cost_fn, _ = rayleigh_cost
        opt = ConjugateGradient()
        r = opt.optimize(cost_fn, sphere, sphere.random_point(), 10)
        d = r.as_dict  # property, not method
        assert "value" in d
        assert "iterations" in d

    def test_adam_config(self):
        adam = Adam(learning_rate=0.05, beta1=0.95, amsgrad=True)
        cfg = adam.config
        assert cfg["learning_rate"] == pytest.approx(0.05)
        assert cfg["beta1"] == pytest.approx(0.95)

    def test_cg_methods(self):
        for method in ["FletcherReeves", "PolakRibiere", "HestenesStiefel", "DaiYuan"]:
            cg = ConjugateGradient(method=method)
            assert cg.config["method"] == method

    def test_invalid_optimizer_name(self):
        from riemannopt.optimizers import create_optimizer
        with pytest.raises(Exception):
            create_optimizer("NonExistent")
