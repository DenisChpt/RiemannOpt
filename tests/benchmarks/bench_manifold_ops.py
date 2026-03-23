"""Benchmarks comparing RiemannOpt manifold operations against pymanopt and geoopt.

Covers: Sphere, Stiefel, Grassmann, SPD.
Operations: retract, project, project_tangent, inner, distance, exp, random_point.

Run with:
    uv run --group bench pytest tests/benchmarks/bench_manifold_ops.py -v --benchmark-columns=mean,stddev,rounds
"""

from __future__ import annotations

import numpy as np
import pytest

# ── RiemannOpt ──────────────────────────────────────────────────────────────
from riemannopt.manifolds import (
    Sphere as ROSphere,
    Stiefel as ROStiefel,
    Grassmann as ROGrassmann,
    SPD as ROSPD,
)

# ── pymanopt ────────────────────────────────────────────────────────────────
pymanopt = pytest.importorskip("pymanopt")
from pymanopt.manifolds import (
    Sphere as PMSphere,
    Stiefel as PMStiefel,
    Grassmann as PMGrassmann,
    PositiveDefinite as PMSPD,
)

# ── geoopt (PyTorch) ────────────────────────────────────────────────────────
geoopt = pytest.importorskip("geoopt")
import torch

torch.set_num_threads(1)  # fair single-thread comparison


# ═══════════════════════════════════════════════════════════════════════════
# Sphere  S^{n-1}
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchSphere:
    N = 1000

    @pytest.fixture(autouse=True)
    def _setup(self):
        # RiemannOpt
        self.ro = ROSphere(self.N)
        self.ro_x = self.ro.random_point()
        self.ro_v = self.ro.random_tangent(self.ro_x)
        self.ro_y = self.ro.random_point()

        # pymanopt
        self.pm = PMSphere(self.N)
        self.pm_x = self.ro_x.copy()
        self.pm_v = self.ro_v.copy()
        self.pm_y = self.ro_y.copy()

        # geoopt
        self.go = geoopt.Sphere()
        self.go_x = torch.from_numpy(self.ro_x.copy())
        self.go_v = torch.from_numpy(self.ro_v.copy())
        self.go_y = torch.from_numpy(self.ro_y.copy())

    # ── retraction ───────────────────────────────────────────────────

    def test_retract_riemannopt(self, benchmark):
        benchmark(self.ro.retract, self.ro_x, self.ro_v)

    def test_retract_pymanopt(self, benchmark):
        benchmark(self.pm.retraction, self.pm_x, self.pm_v)

    def test_retract_geoopt(self, benchmark):
        benchmark(self.go.retr, self.go_x, self.go_v)

    # ── point projection ───────────────────────────────────────────

    def test_project_riemannopt(self, benchmark):
        raw = self.ro_x * 3.0
        benchmark(self.ro.project, raw)

    def test_project_geoopt(self, benchmark):
        raw = self.go_x * 3.0
        benchmark(self.go.projx, raw)

    # ── tangent projection ───────────────────────────────────────────

    def test_project_tangent_riemannopt(self, benchmark):
        benchmark(self.ro.project_tangent, self.ro_x, self.ro_v)

    def test_project_tangent_pymanopt(self, benchmark):
        benchmark(self.pm.to_tangent_space, self.pm_x, self.pm_v)

    def test_project_tangent_geoopt(self, benchmark):
        benchmark(self.go.proju, self.go_x, self.go_v)

    # ── inner product ────────────────────────────────────────────────

    def test_inner_riemannopt(self, benchmark):
        u = self.ro.random_tangent(self.ro_x)
        benchmark(self.ro.inner, self.ro_x, self.ro_v, u)

    def test_inner_pymanopt(self, benchmark):
        u = self.ro.random_tangent(self.ro_x)
        benchmark(self.pm.inner_product, self.pm_x, self.pm_v, u.copy())

    def test_inner_geoopt(self, benchmark):
        u = torch.from_numpy(self.ro.random_tangent(self.ro_x))
        benchmark(self.go.inner, self.go_x, self.go_v, u)

    # ── distance ─────────────────────────────────────────────────────

    def test_distance_riemannopt(self, benchmark):
        benchmark(self.ro.distance, self.ro_x, self.ro_y)

    def test_distance_pymanopt(self, benchmark):
        benchmark(self.pm.dist, self.pm_x, self.pm_y)

    def test_distance_geoopt(self, benchmark):
        benchmark(self.go.dist, self.go_x, self.go_y)

    # ── exp map ──────────────────────────────────────────────────────

    def test_exp_riemannopt(self, benchmark):
        benchmark(self.ro.exp, self.ro_x, 0.1 * self.ro_v)

    def test_exp_pymanopt(self, benchmark):
        benchmark(self.pm.exp, self.pm_x, 0.1 * self.pm_v)

    def test_exp_geoopt(self, benchmark):
        benchmark(self.go.expmap, self.go_x, 0.1 * self.go_v)

    # ── log map ──────────────────────────────────────────────────────

    def test_log_riemannopt(self, benchmark):
        benchmark(self.ro.log, self.ro_x, self.ro_y)

    def test_log_pymanopt(self, benchmark):
        benchmark(self.pm.log, self.pm_x, self.pm_y)

    def test_log_geoopt(self, benchmark):
        benchmark(self.go.logmap, self.go_x, self.go_y)

    # ── random point ─────────────────────────────────────────────────

    def test_random_point_riemannopt(self, benchmark):
        benchmark(self.ro.random_point)

    def test_random_point_pymanopt(self, benchmark):
        benchmark(self.pm.random_point)


# ═══════════════════════════════════════════════════════════════════════════
# Stiefel  St(n, p)
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchStiefel:
    N, P = 100, 10

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.ro = ROStiefel(self.N, self.P)
        self.ro_x = self.ro.random_point()
        self.ro_v = self.ro.random_tangent(self.ro_x)

        self.pm = PMStiefel(self.N, self.P)
        self.pm_x = self.ro_x.copy()
        self.pm_v = self.ro_v.copy()

        self.go = geoopt.Stiefel()
        self.go_x = torch.from_numpy(self.ro_x.copy())
        self.go_v = torch.from_numpy(self.ro_v.copy())

    def test_retract_riemannopt(self, benchmark):
        benchmark(self.ro.retract, self.ro_x, 0.01 * self.ro_v)

    def test_retract_pymanopt(self, benchmark):
        benchmark(self.pm.retraction, self.pm_x, 0.01 * self.pm_v)

    def test_retract_geoopt(self, benchmark):
        benchmark(self.go.retr, self.go_x, 0.01 * self.go_v)

    def test_project_tangent_riemannopt(self, benchmark):
        benchmark(self.ro.project_tangent, self.ro_x, self.ro_v)

    def test_project_tangent_pymanopt(self, benchmark):
        benchmark(self.pm.to_tangent_space, self.pm_x, self.pm_v)

    def test_project_tangent_geoopt(self, benchmark):
        benchmark(self.go.proju, self.go_x, self.go_v)

    def test_exp_riemannopt(self, benchmark):
        benchmark(self.ro.exp, self.ro_x, 0.01 * self.ro_v)

    def test_exp_pymanopt(self, benchmark):
        benchmark(self.pm.exp, self.pm_x, 0.01 * self.pm_v)

    def test_random_point_riemannopt(self, benchmark):
        benchmark(self.ro.random_point)

    def test_random_point_pymanopt(self, benchmark):
        benchmark(self.pm.random_point)

    def test_random_point_geoopt(self, benchmark):
        benchmark(geoopt.Stiefel().random, self.N, self.P)


# ═══════════════════════════════════════════════════════════════════════════
# Grassmann  Gr(n, p)
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchGrassmann:
    N, P = 50, 5

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.ro = ROGrassmann(self.N, self.P)
        self.ro_x = self.ro.random_point()
        self.ro_v = self.ro.random_tangent(self.ro_x)
        self.ro_y = self.ro.random_point()

        self.pm = PMGrassmann(self.N, self.P)
        self.pm_x = self.ro_x.copy()
        self.pm_v = self.ro_v.copy()
        self.pm_y = self.ro_y.copy()

    def test_retract_riemannopt(self, benchmark):
        benchmark(self.ro.retract, self.ro_x, 0.01 * self.ro_v)

    def test_retract_pymanopt(self, benchmark):
        benchmark(self.pm.retraction, self.pm_x, 0.01 * self.pm_v)

    def test_project_tangent_riemannopt(self, benchmark):
        benchmark(self.ro.project_tangent, self.ro_x, self.ro_v)

    def test_project_tangent_pymanopt(self, benchmark):
        benchmark(self.pm.to_tangent_space, self.pm_x, self.pm_v)

    def test_exp_riemannopt(self, benchmark):
        benchmark(self.ro.exp, self.ro_x, 0.01 * self.ro_v)

    def test_exp_pymanopt(self, benchmark):
        benchmark(self.pm.exp, self.pm_x, 0.01 * self.pm_v)

    def test_distance_riemannopt(self, benchmark):
        benchmark(self.ro.distance, self.ro_x, self.ro_y)

    def test_distance_pymanopt(self, benchmark):
        benchmark(self.pm.dist, self.pm_x, self.pm_y)

    def test_random_point_riemannopt(self, benchmark):
        benchmark(self.ro.random_point)

    def test_random_point_pymanopt(self, benchmark):
        benchmark(self.pm.random_point)


# ═══════════════════════════════════════════════════════════════════════════
# SPD  S_++^n
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchSPD:
    N = 10

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.ro = ROSPD(self.N)
        self.ro_x = self.ro.random_point()
        self.ro_v = self.ro.random_tangent(self.ro_x)
        self.ro_y = self.ro.random_point()

        self.pm = PMSPD(self.N)
        self.pm_x = self.ro_x.copy()
        self.pm_v = self.ro_v.copy()
        self.pm_y = self.ro_y.copy()

    def test_retract_riemannopt(self, benchmark):
        benchmark(self.ro.retract, self.ro_x, 0.01 * self.ro_v)

    def test_retract_pymanopt(self, benchmark):
        benchmark(self.pm.retraction, self.pm_x, 0.01 * self.pm_v)

    def test_exp_riemannopt(self, benchmark):
        benchmark(self.ro.exp, self.ro_x, 0.01 * self.ro_v)

    def test_exp_pymanopt(self, benchmark):
        benchmark(self.pm.exp, self.pm_x, 0.01 * self.pm_v)

    def test_distance_riemannopt(self, benchmark):
        benchmark(self.ro.distance, self.ro_x, self.ro_y)

    def test_distance_pymanopt(self, benchmark):
        benchmark(self.pm.dist, self.pm_x, self.pm_y)

    def test_random_point_riemannopt(self, benchmark):
        benchmark(self.ro.random_point)

    def test_random_point_pymanopt(self, benchmark):
        benchmark(self.pm.random_point)

    def test_project_tangent_riemannopt(self, benchmark):
        benchmark(self.ro.project_tangent, self.ro_x, self.ro_v)


# ═══════════════════════════════════════════════════════════════════════════
# Scaling benchmark: vary problem dimension
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestBenchScaling:
    """Measure how each library scales with dimension."""

    @pytest.mark.parametrize("n", [50, 200, 1000, 5000])
    def test_sphere_retract_scaling_riemannopt(self, benchmark, n):
        m = ROSphere(n)
        x = m.random_point()
        v = m.random_tangent(x)
        benchmark(m.retract, x, v)

    @pytest.mark.parametrize("n", [50, 200, 1000, 5000])
    def test_sphere_retract_scaling_pymanopt(self, benchmark, n):
        m = PMSphere(n)
        x = m.random_point()
        v = m.random_tangent_vector(x)
        benchmark(m.retraction, x, v)

    @pytest.mark.parametrize("n", [50, 200, 1000, 5000])
    def test_sphere_retract_scaling_geoopt(self, benchmark, n):
        m = geoopt.Sphere()
        x = torch.randn(n, dtype=torch.float64)
        x = x / x.norm()
        v = torch.randn(n, dtype=torch.float64)
        v = v - x * x.dot(v)
        benchmark(m.retr, x, v)

    @pytest.mark.parametrize("n,p", [(20, 5), (50, 10), (100, 20)])
    def test_stiefel_retract_scaling_riemannopt(self, benchmark, n, p):
        m = ROStiefel(n, p)
        x = m.random_point()
        v = m.random_tangent(x)
        benchmark(m.retract, x, 0.01 * v)

    @pytest.mark.parametrize("n,p", [(20, 5), (50, 10), (100, 20)])
    def test_stiefel_retract_scaling_pymanopt(self, benchmark, n, p):
        m = PMStiefel(n, p)
        x = m.random_point()
        v = m.random_tangent_vector(x)
        benchmark(m.retraction, x, 0.01 * v)

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_spd_exp_scaling_riemannopt(self, benchmark, n):
        m = ROSPD(n)
        x = m.random_point()
        v = m.random_tangent(x)
        benchmark(m.exp, x, 0.01 * v)

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_spd_exp_scaling_pymanopt(self, benchmark, n):
        m = PMSPD(n)
        x = m.random_point()
        v = m.random_tangent_vector(x)
        benchmark(m.exp, x, 0.01 * v)
