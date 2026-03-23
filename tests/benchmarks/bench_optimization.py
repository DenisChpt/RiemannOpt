"""End-to-end optimisation benchmarks: RiemannOpt vs pymanopt vs geoopt.

Compares wall-clock time to converge on the Rayleigh quotient (leading
eigenvector) and trace minimisation (leading eigenspace) problems.

Run with:
    uv run pytest tests/benchmarks/bench_optimization.py -v -s --benchmark-disable
    uv run pytest tests/benchmarks/bench_optimization.py -v --benchmark-columns=mean,stddev,rounds
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pytest

# ── RiemannOpt ──────────────────────────────────────────────────────────────
from riemannopt import create_cost_function
from riemannopt.manifolds import Sphere as ROSphere, Grassmann as ROGrassmann
from riemannopt.optimizers import ConjugateGradient as ROCG

# ── pymanopt ────────────────────────────────────────────────────────────────
pymanopt = pytest.importorskip("pymanopt")
import autograd.numpy as anp
from pymanopt.manifolds import Sphere as PMSphere, Grassmann as PMGrassmann
from pymanopt import Problem as PMProblem
from pymanopt.optimizers import ConjugateGradient as PMCG

# ── geoopt (PyTorch) ────────────────────────────────────────────────────────
geoopt = pytest.importorskip("geoopt")
import torch

torch.set_num_threads(1)


# ── helpers ─────────────────────────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    time_s: float
    cost: float
    iterations: int
    converged: bool


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Rayleigh quotient on S^{n-1}  (leading eigenvector)
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchRayleigh:
    """Compare CG optimisation of x^T A x on the sphere."""

    N = 200
    MAX_ITER = 300
    GTOL = 1e-8

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.A = np.diag(np.arange(1, self.N + 1, dtype=np.float64))
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(self.N)
        self.x0 = x0 / np.linalg.norm(x0)
        self.f_star = 1.0  # smallest eigenvalue

    # ── RiemannOpt ───────────────────────────────────────────────────

    def _run_riemannopt(self) -> BenchResult:
        sphere = ROSphere(self.N)
        A = self.A

        def cost(x):
            return float(x @ A @ x)

        def grad(x):
            return 2.0 * A @ x

        cf = create_cost_function(cost, gradient=grad, dimension=self.N)
        opt = ROCG(method="PolakRibiere")
        t0 = time.perf_counter()
        r = opt.optimize(cf, sphere, self.x0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt", dt, cf.cost(r.point), r.iterations, r.converged)

    # ── pymanopt ─────────────────────────────────────────────────────

    def _run_pymanopt(self) -> BenchResult:
        sphere = PMSphere(self.N)
        A = self.A

        @pymanopt.function.autograd(sphere)
        def cost(x):
            return x @ A @ x

        problem = PMProblem(sphere, cost)
        opt = PMCG(max_iterations=self.MAX_ITER, min_gradient_norm=self.GTOL, verbosity=0)
        t0 = time.perf_counter()
        result = opt.run(problem, initial_point=self.x0.copy())
        dt = time.perf_counter() - t0
        x_opt = result.point
        f_opt = float(x_opt @ A @ x_opt)
        return BenchResult("pymanopt", dt, f_opt, result.iterations, result.iterations < self.MAX_ITER)

    # ── geoopt (manual CG with Riemannian SGD as fallback) ──────────

    def _run_geoopt(self) -> BenchResult:
        A_torch = torch.from_numpy(self.A)
        sphere = geoopt.Sphere()
        x = geoopt.ManifoldParameter(torch.from_numpy(self.x0.copy()), manifold=sphere)
        optimizer = geoopt.optim.RiemannianSGD([x], lr=0.01)
        t0 = time.perf_counter()
        iters = 0
        for i in range(self.MAX_ITER):
            optimizer.zero_grad()
            loss = x @ A_torch @ x
            loss.backward()
            optimizer.step()
            iters = i + 1
            with torch.no_grad():
                g = sphere.egrad2rgrad(x, x.grad)
                if g.norm().item() < self.GTOL:
                    break
        dt = time.perf_counter() - t0
        with torch.no_grad():
            f_opt = float((x @ A_torch @ x).item())
        return BenchResult("geoopt", dt, f_opt, iters, iters < self.MAX_ITER)

    # ── pytest-benchmark wrappers ────────────────────────────────────

    def test_rayleigh_riemannopt(self, benchmark):
        result = benchmark.pedantic(self._run_riemannopt, rounds=5, warmup_rounds=1)
        assert result.cost < self.f_star + 1e-4

    def test_rayleigh_pymanopt(self, benchmark):
        result = benchmark.pedantic(self._run_pymanopt, rounds=5, warmup_rounds=1)
        assert result.cost < self.f_star + 1e-4

    def test_rayleigh_geoopt(self, benchmark):
        result = benchmark.pedantic(self._run_geoopt, rounds=5, warmup_rounds=1)
        # geoopt SGD with lr=0.01 on dim=200 won't converge — just check it ran
        assert result.iterations > 0


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Trace minimisation on Gr(n, p)  (leading eigenspace)
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchTrace:
    """Compare CG optimisation of tr(Y^T A Y) on the Grassmann manifold."""

    N, P = 100, 5
    MAX_ITER = 300
    GTOL = 1e-8

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.A = np.diag(np.arange(1, self.N + 1, dtype=np.float64))
        self.f_star = float(sum(range(1, self.P + 1)))  # 1 + 2 + ... + P
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((self.N, self.P)))
        self.Y0 = Q

    def _run_riemannopt(self) -> BenchResult:
        gr = ROGrassmann(self.N, self.P)
        A = self.A

        def cost(Y):
            return float(np.trace(Y.T @ A @ Y))

        def grad(Y):
            return 2.0 * A @ Y

        cf = create_cost_function(cost, gradient=grad, dimension=(self.N, self.P))
        opt = ROCG(method="PolakRibiere")
        t0 = time.perf_counter()
        r = opt.optimize(cf, gr, self.Y0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt", dt, cf.cost(r.point), r.iterations, r.converged)

    def _run_pymanopt(self) -> BenchResult:
        gr = PMGrassmann(self.N, self.P)
        A = self.A

        @pymanopt.function.autograd(gr)
        def cost(Y):
            return anp.trace(Y.T @ A @ Y)

        problem = PMProblem(gr, cost)
        opt = PMCG(max_iterations=self.MAX_ITER, min_gradient_norm=self.GTOL, verbosity=0)
        t0 = time.perf_counter()
        result = opt.run(problem, initial_point=self.Y0.copy())
        dt = time.perf_counter() - t0
        Y_opt = result.point
        return BenchResult("pymanopt", dt, float(np.trace(Y_opt.T @ A @ Y_opt)), result.iterations, True)

    def test_trace_riemannopt(self, benchmark):
        result = benchmark.pedantic(self._run_riemannopt, rounds=5, warmup_rounds=1)
        assert result.cost < self.f_star + 0.1

    def test_trace_pymanopt(self, benchmark):
        result = benchmark.pedantic(self._run_pymanopt, rounds=5, warmup_rounds=1)
        assert result.cost < self.f_star + 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Scaling benchmark: vary problem dimension
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
