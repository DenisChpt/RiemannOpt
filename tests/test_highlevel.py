"""Tests for the high-level Python API (helpers.optimize, create_cost_function, etc.)."""

from __future__ import annotations

import numpy as np
import pytest

import riemannopt as ro
from riemannopt.manifolds import Sphere, Stiefel
from riemannopt.helpers import (
    OptimizationCallback,
    ProgressCallback,
    EarlyStoppingCallback,
    gradient_check,
)


class TestCreateCostFunction:
    def test_from_callable(self):
        cf = ro.create_cost_function(lambda x: float(x @ x), dimension=5)
        x = np.ones(5)
        assert cf.cost(x) == pytest.approx(5.0)

    def test_from_cost_and_gradient(self):
        def cost(x):
            return float(0.5 * x @ x)

        def grad(x):
            return x.copy()

        cf = ro.create_cost_function(cost, gradient=grad, dimension=5)
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        c, g = cf.cost_and_gradient(x)
        assert c == pytest.approx(27.5)
        np.testing.assert_allclose(g, x)

    def test_with_explicit_cost_and_gradient(self):
        def cost(x):
            return float(x @ x)

        def grad(x):
            return 2.0 * x

        cf = ro.create_cost_function(cost, gradient=grad, dimension=3)
        x = np.array([1.0, 0.0, 0.0])
        assert cf.cost(x) == pytest.approx(1.0)
        np.testing.assert_allclose(cf.gradient(x), [2.0, 0.0, 0.0])

    def test_eval_counts(self):
        cf = ro.create_cost_function(lambda x: float(x @ x), dimension=3)
        x = np.ones(3)
        cf.cost(x)
        cf.cost(x)
        cf.cost(x)
        counts = cf.eval_counts
        assert counts["cost"] >= 3
        cf.reset_counts()


class TestHighLevelOptimize:
    def test_basic_sphere(self):
        sphere = Sphere(5)

        def cost(x):
            return float(x @ x)

        def grad(x):
            return 2.0 * x

        cf = ro.create_cost_function(cost, gradient=grad, dimension=5)
        x0 = sphere.random_point()
        result = ro.optimize(sphere, cf, x0, optimizer="ConjugateGradient", max_iterations=100)
        assert result.converged or result.iterations == 100

    def test_auto_wrap_callable(self):
        sphere = Sphere(5)
        x0 = sphere.random_point()
        # Pass raw callable — should auto-wrap
        result = ro.optimize(sphere, lambda x: float(x @ x), x0, optimizer="SGD", max_iterations=10)
        assert hasattr(result, "cost")

    def test_case_insensitive_optimizer_name(self):
        sphere = Sphere(5)
        cf = ro.create_cost_function(lambda x: float(x @ x), dimension=5)
        x0 = sphere.random_point()
        # All these should work
        for name in ["sgd", "SGD", "Sgd"]:
            ro.optimize(sphere, cf, x0, optimizer=name, max_iterations=5)

    def test_invalid_optimizer_raises(self):
        sphere = Sphere(5)
        cf = ro.create_cost_function(lambda x: float(x @ x), dimension=5)
        with pytest.raises(ValueError, match="Unknown optimizer"):
            ro.optimize(sphere, cf, sphere.random_point(), optimizer="FooBar")


class TestGradientCheck:
    def test_correct_gradient(self):
        def cost(x):
            return float(0.5 * x @ x)

        def grad(x):
            return x.copy()

        cf = ro.create_cost_function(cost, gradient=grad, dimension=5)
        ok, err = gradient_check(cf, np.ones(5))
        assert ok
        assert err < 1e-4

    def test_wrong_gradient_detected(self):
        def cost(x):
            return float(0.5 * x @ x)

        def bad_grad(x):
            return np.zeros_like(x)  # completely wrong

        cf = ro.create_cost_function(cost, gradient=bad_grad, dimension=5)
        ok, err = gradient_check(cf, np.ones(5))
        assert not ok


class TestCallbacks:
    def test_optimization_callback_collects_history(self):
        sphere = Sphere(5)
        cf = ro.create_cost_function(lambda x: float(x @ x), dimension=5)
        cb = OptimizationCallback()
        # manually call it
        x = sphere.random_point()
        cb(0, x, 1.0, 0.5)
        cb(1, x, 0.9, 0.3)
        assert len(cb.history) == 2
        assert cb.history[0]["cost"] == 1.0

    def test_early_stopping_callback(self):
        cb = EarlyStoppingCallback(patience=3, min_improvement=0.1)
        x = np.zeros(3)
        assert cb(0, x, 10.0, 1.0) is True
        assert cb(1, x, 9.0, 1.0) is True  # improved by 1.0
        assert cb(2, x, 8.99, 1.0) is True  # not enough improvement, wait=1
        assert cb(3, x, 8.98, 1.0) is True  # wait=2
        assert cb(4, x, 8.97, 1.0) is False  # wait=3 → stop
