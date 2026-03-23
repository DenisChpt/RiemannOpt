"""End-to-end optimisation benchmarks: RiemannOpt vs pymanopt vs geoopt.

Compares wall-clock time to converge on canonical problems:
  1. Rayleigh quotient on S^{n-1}  (leading eigenvector)
  2. Trace minimisation on Gr(n, p)  (leading eigenspace)
  3. Brockett on St(n, p)           (eigenvalue ordering)
  4. Log-det divergence on SPD(n)   (geometric mean)

Run with:
    uv run --group bench pytest tests/benchmarks/bench_optimization.py -v -s --benchmark-disable
    uv run --group bench pytest tests/benchmarks/bench_optimization.py -v --benchmark-columns=mean,stddev,rounds
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pytest

# ── RiemannOpt ──────────────────────────────────────────────────────────────
from riemannopt import create_cost_function
from riemannopt.manifolds import (
    Sphere as ROSphere,
    Grassmann as ROGrassmann,
    Stiefel as ROStiefel,
    SPD as ROSPD,
)
from riemannopt.optimizers import (
    ConjugateGradient as ROCG,
    TrustRegion as ROTR,
    LBFGS as ROLBFGS,
    Adam as ROAdam,
)

# ── pymanopt ────────────────────────────────────────────────────────────────
pymanopt = pytest.importorskip("pymanopt")
import autograd.numpy as anp
from pymanopt.manifolds import (
    Sphere as PMSphere,
    Grassmann as PMGrassmann,
    Stiefel as PMStiefel,
    PositiveDefinite as PMSPD,
)
from pymanopt import Problem as PMProblem
from pymanopt.optimizers import (
    ConjugateGradient as PMCG,
    TrustRegions as PMTR,
)

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
        self.f_star = 1.0

    # ── RiemannOpt CG ────────────────────────────────────────────────

    def _run_ro_cg(self) -> BenchResult:
        sphere = ROSphere(self.N)
        A = self.A
        cf = create_cost_function(
            lambda x: float(x @ A @ x),
            gradient=lambda x: 2.0 * A @ x,
            dimension=self.N,
        )
        opt = ROCG(method="PolakRibiere")
        t0 = time.perf_counter()
        r = opt.optimize(cf, sphere, self.x0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt-CG", dt, cf.cost(r.point), r.iterations, r.converged)

    # ── RiemannOpt TrustRegion ───────────────────────────────────────

    def _run_ro_tr(self) -> BenchResult:
        sphere = ROSphere(self.N)
        A = self.A
        cf = create_cost_function(
            lambda x: float(x @ A @ x),
            gradient=lambda x: 2.0 * A @ x,
            dimension=self.N,
        )
        opt = ROTR()
        t0 = time.perf_counter()
        r = opt.optimize(cf, sphere, self.x0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt-TR", dt, cf.cost(r.point), r.iterations, r.converged)

    # ── RiemannOpt LBFGS ─────────────────────────────────────────────

    def _run_ro_lbfgs(self) -> BenchResult:
        sphere = ROSphere(self.N)
        A = self.A
        cf = create_cost_function(
            lambda x: float(x @ A @ x),
            gradient=lambda x: 2.0 * A @ x,
            dimension=self.N,
        )
        opt = ROLBFGS()
        t0 = time.perf_counter()
        r = opt.optimize(cf, sphere, self.x0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt-LBFGS", dt, cf.cost(r.point), r.iterations, r.converged)

    # ── pymanopt CG ──────────────────────────────────────────────────

    def _run_pm_cg(self) -> BenchResult:
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
        return BenchResult("pymanopt-CG", dt, float(x_opt @ A @ x_opt), result.iterations, True)

    # ── pymanopt TrustRegions ────────────────────────────────────────

    def _run_pm_tr(self) -> BenchResult:
        sphere = PMSphere(self.N)
        A = self.A

        @pymanopt.function.autograd(sphere)
        def cost(x):
            return x @ A @ x

        problem = PMProblem(sphere, cost)
        opt = PMTR(max_iterations=self.MAX_ITER, min_gradient_norm=self.GTOL, verbosity=0)
        t0 = time.perf_counter()
        result = opt.run(problem, initial_point=self.x0.copy())
        dt = time.perf_counter() - t0
        x_opt = result.point
        return BenchResult("pymanopt-TR", dt, float(x_opt @ A @ x_opt), result.iterations, True)

    # ── geoopt RSGD ──────────────────────────────────────────────────

    def _run_geoopt_rsgd(self) -> BenchResult:
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
        return BenchResult("geoopt-RSGD", dt, f_opt, iters, iters < self.MAX_ITER)

    # ── geoopt RAdam ─────────────────────────────────────────────────

    def _run_geoopt_radam(self) -> BenchResult:
        A_torch = torch.from_numpy(self.A)
        sphere = geoopt.Sphere()
        x = geoopt.ManifoldParameter(torch.from_numpy(self.x0.copy()), manifold=sphere)
        optimizer = geoopt.optim.RiemannianAdam([x], lr=0.01)
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
        return BenchResult("geoopt-RAdam", dt, f_opt, iters, iters < self.MAX_ITER)

    # ── pytest-benchmark wrappers ────────────────────────────────────

    def test_rayleigh_riemannopt_cg(self, benchmark):
        r = benchmark.pedantic(self._run_ro_cg, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1e-4

    def test_rayleigh_riemannopt_tr(self, benchmark):
        r = benchmark.pedantic(self._run_ro_tr, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1e-4

    def test_rayleigh_riemannopt_lbfgs(self, benchmark):
        r = benchmark.pedantic(self._run_ro_lbfgs, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1e-4

    def test_rayleigh_pymanopt_cg(self, benchmark):
        r = benchmark.pedantic(self._run_pm_cg, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1e-4

    def test_rayleigh_pymanopt_tr(self, benchmark):
        r = benchmark.pedantic(self._run_pm_tr, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1e-4

    def test_rayleigh_geoopt_rsgd(self, benchmark):
        r = benchmark.pedantic(self._run_geoopt_rsgd, rounds=5, warmup_rounds=1)
        assert r.iterations > 0

    def test_rayleigh_geoopt_radam(self, benchmark):
        r = benchmark.pedantic(self._run_geoopt_radam, rounds=5, warmup_rounds=1)
        assert r.iterations > 0


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
        self.f_star = float(sum(range(1, self.P + 1)))
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((self.N, self.P)))
        self.Y0 = Q

    def _run_ro_cg(self) -> BenchResult:
        gr = ROGrassmann(self.N, self.P)
        A = self.A
        cf = create_cost_function(
            lambda Y: float(np.trace(Y.T @ A @ Y)),
            gradient=lambda Y: 2.0 * A @ Y,
            dimension=(self.N, self.P),
        )
        opt = ROCG(method="PolakRibiere")
        t0 = time.perf_counter()
        r = opt.optimize(cf, gr, self.Y0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt-CG", dt, cf.cost(r.point), r.iterations, r.converged)

    def _run_ro_tr(self) -> BenchResult:
        gr = ROGrassmann(self.N, self.P)
        A = self.A
        cf = create_cost_function(
            lambda Y: float(np.trace(Y.T @ A @ Y)),
            gradient=lambda Y: 2.0 * A @ Y,
            dimension=(self.N, self.P),
        )
        opt = ROTR()
        t0 = time.perf_counter()
        r = opt.optimize(cf, gr, self.Y0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt-TR", dt, cf.cost(r.point), r.iterations, r.converged)

    def _run_pm_cg(self) -> BenchResult:
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
        return BenchResult("pymanopt-CG", dt, float(np.trace(Y_opt.T @ A @ Y_opt)), result.iterations, True)

    def test_trace_riemannopt_cg(self, benchmark):
        r = benchmark.pedantic(self._run_ro_cg, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 0.1

    def test_trace_riemannopt_tr(self, benchmark):
        r = benchmark.pedantic(self._run_ro_tr, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 0.1

    def test_trace_pymanopt_cg(self, benchmark):
        r = benchmark.pedantic(self._run_pm_cg, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 0.1


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Brockett on St(n, p)
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchBrockett:
    """Compare CG on Brockett cost tr(X^T A X N) on Stiefel."""

    N, P = 30, 5
    MAX_ITER = 300
    GTOL = 1e-8

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.A = np.diag(np.arange(1, self.N + 1, dtype=np.float64))
        self.Nmat = np.diag(np.arange(1, self.P + 1, dtype=np.float64))
        self.f_star = float(sum(i * i for i in range(1, self.P + 1)))
        rng = np.random.default_rng(42)
        Q, _ = np.linalg.qr(rng.standard_normal((self.N, self.P)))
        self.X0 = Q

    def _run_ro_cg(self) -> BenchResult:
        st = ROStiefel(self.N, self.P)
        A, Nmat = self.A, self.Nmat
        cf = create_cost_function(
            lambda X: float(np.trace(X.T @ A @ X @ Nmat)),
            gradient=lambda X: 2.0 * A @ X @ Nmat,
            dimension=(self.N, self.P),
        )
        opt = ROCG(method="PolakRibiere")
        t0 = time.perf_counter()
        r = opt.optimize(cf, st, self.X0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt-CG", dt, cf.cost(r.point), r.iterations, r.converged)

    def _run_pm_cg(self) -> BenchResult:
        st = PMStiefel(self.N, self.P)
        A, Nmat = self.A, self.Nmat

        @pymanopt.function.autograd(st)
        def cost(X):
            return anp.trace(X.T @ A @ X @ Nmat)

        problem = PMProblem(st, cost)
        opt = PMCG(max_iterations=self.MAX_ITER, min_gradient_norm=self.GTOL, verbosity=0)
        t0 = time.perf_counter()
        result = opt.run(problem, initial_point=self.X0.copy())
        dt = time.perf_counter() - t0
        X_opt = result.point
        return BenchResult("pymanopt-CG", dt, float(np.trace(X_opt.T @ A @ X_opt @ Nmat)), result.iterations, True)

    def test_brockett_riemannopt(self, benchmark):
        r = benchmark.pedantic(self._run_ro_cg, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1.0

    def test_brockett_pymanopt(self, benchmark):
        r = benchmark.pedantic(self._run_pm_cg, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Log-det divergence on SPD(n)
# ═══════════════════════════════════════════════════════════════════════════

class TestBenchSPDLogDet:
    """Compare CG on tr(P) - log det(P) on SPD manifold."""

    N = 10
    MAX_ITER = 200
    GTOL = 1e-6

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.f_star = float(self.N)  # minimum at I
        ro_spd = ROSPD(self.N)
        self.P0 = ro_spd.random_point()

    def _run_ro_cg(self) -> BenchResult:
        spd = ROSPD(self.N)
        I = np.eye(self.N)
        cf = create_cost_function(
            lambda P: float(np.trace(P) - np.linalg.slogdet(P)[1]),
            gradient=lambda P: I - np.linalg.inv(P),
            dimension=(self.N, self.N),
        )
        opt = ROCG(method="PolakRibiere")
        t0 = time.perf_counter()
        r = opt.optimize(cf, spd, self.P0.copy(), self.MAX_ITER, gradient_tolerance=self.GTOL)
        dt = time.perf_counter() - t0
        return BenchResult("riemannopt-CG", dt, cf.cost(r.point), r.iterations, r.converged)

    def _run_pm_cg(self) -> BenchResult:
        spd = PMSPD(self.N)
        I = np.eye(self.N)

        @pymanopt.function.autograd(spd)
        def cost(P):
            return anp.trace(P) - anp.linalg.slogdet(P)[1]

        problem = PMProblem(spd, cost)
        opt = PMCG(max_iterations=self.MAX_ITER, min_gradient_norm=self.GTOL, verbosity=0)
        t0 = time.perf_counter()
        result = opt.run(problem, initial_point=self.P0.copy())
        dt = time.perf_counter() - t0
        P_opt = result.point
        f_opt = float(np.trace(P_opt) - np.linalg.slogdet(P_opt)[1])
        return BenchResult("pymanopt-CG", dt, f_opt, result.iterations, True)

    def test_spd_riemannopt(self, benchmark):
        r = benchmark.pedantic(self._run_ro_cg, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1.0

    def test_spd_pymanopt(self, benchmark):
        r = benchmark.pedantic(self._run_pm_cg, rounds=5, warmup_rounds=1)
        assert r.cost < self.f_star + 1.0


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Scaling: full optimisation at increasing dimensions
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestBenchOptScaling:
    """Measure how optimisation time scales with dimension."""

    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_rayleigh_cg_scaling_riemannopt(self, benchmark, n):
        sphere = ROSphere(n)
        A = np.diag(np.arange(1, n + 1, dtype=float))
        cf = create_cost_function(
            lambda x: float(x @ A @ x),
            gradient=lambda x: 2.0 * A @ x,
            dimension=n,
        )
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(n)
        x0 /= np.linalg.norm(x0)

        def run():
            opt = ROCG(method="PolakRibiere")
            return opt.optimize(cf, sphere, x0.copy(), 100, gradient_tolerance=1e-8)

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)

    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_rayleigh_cg_scaling_pymanopt(self, benchmark, n):
        sphere = PMSphere(n)
        A = np.diag(np.arange(1, n + 1, dtype=float))

        @pymanopt.function.autograd(sphere)
        def cost(x):
            return x @ A @ x

        problem = PMProblem(sphere, cost)
        rng = np.random.default_rng(42)
        x0 = rng.standard_normal(n)
        x0 /= np.linalg.norm(x0)

        def run():
            opt = PMCG(max_iterations=100, min_gradient_norm=1e-8, verbosity=0)
            return opt.run(problem, initial_point=x0.copy())

        benchmark.pedantic(run, rounds=3, warmup_rounds=1)
