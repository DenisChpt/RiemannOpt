"""Unit tests for the Newton optimizer."""

import numpy as np
import pytest
import riemannopt

# Access classes through _riemannopt module
Newton = riemannopt._riemannopt.Newton
CostFunction = riemannopt._riemannopt.CostFunction
Sphere = riemannopt._riemannopt.Sphere
Stiefel = riemannopt._riemannopt.Stiefel
SPD = riemannopt._riemannopt.SPD
Oblique = riemannopt._riemannopt.Oblique
FixedRank = riemannopt._riemannopt.FixedRank
PSDCone = riemannopt._riemannopt.PSDCone
Euclidean = riemannopt._riemannopt.Euclidean


class TestNewton:
    """Test cases for the Newton optimizer."""
    
    def test_creation(self):
        """Test Newton optimizer creation."""
        # Default creation
        optimizer = Newton()
        assert optimizer is not None
        
        # Custom parameters
        optimizer = Newton(
            hessian_regularization=1e-6,
            use_gauss_newton=False,
            max_cg_iterations=50,
            cg_tolerance=1e-8,
            max_iterations=50,
            tolerance=1e-8
        )
        assert optimizer is not None
        
        # Invalid parameters
        with pytest.raises(ValueError, match="hessian_regularization must be non-negative"):
            riemannopt.Newton(hessian_regularization=-1.0)
        
        with pytest.raises(ValueError, match="max_cg_iterations must be positive"):
            riemannopt.Newton(max_cg_iterations=0)
        
        with pytest.raises(ValueError, match="cg_tolerance must be positive"):
            riemannopt.Newton(cg_tolerance=0.0)
            
        with pytest.raises(ValueError, match="max_iterations must be positive"):
            riemannopt.Newton(max_iterations=0)
            
        with pytest.raises(ValueError, match="tolerance must be positive"):
            riemannopt.Newton(tolerance=-1e-6)
    
    def test_step_not_supported(self):
        """Test that direct step() calls are not supported."""
        optimizer = Newton()
        manifold = Sphere(10)
        
        x = manifold.random_point()
        grad = manifold.random_tangent(x)
        
        # Newton's step method should raise an error
        with pytest.raises(ValueError, match="step method requires full optimize"):
            optimizer.step(manifold, x, grad)
    
    def test_optimization_on_sphere(self):
        """Test Newton optimization on sphere."""
        n = 10
        manifold = Sphere(n)
        optimizer = Newton(
            max_iterations=100,
            tolerance=1e-6,
            hessian_regularization=1e-4  # Higher regularization for stability
        )
        
        # Cost function: f(x) = -x^T A x (Rayleigh quotient)
        # Minimum is achieved at the eigenvector with largest eigenvalue
        np.random.seed(42)
        A = np.random.randn(n, n)
        A = A + A.T  # Make symmetric
        
        def cost_fn(x):
            return -np.dot(x, A @ x)
        
        def grad_fn(x):
            return -2 * A @ x
        
        cost = CostFunction(cost_fn, grad_fn)
        
        # Initial point
        x0 = manifold.random_point()
        
        # Optimize
        result = optimizer.optimize(cost, manifold, x0)
        
        # Check that we're still on the manifold
        x_opt = result['point']
        # Verify solution is still on the sphere by checking norm
        np.testing.assert_allclose(np.linalg.norm(x_opt), 1.0, rtol=1e-6)
        
        # Check result fields
        assert 'value' in result
        assert 'iterations' in result
        assert 'converged' in result
        assert 'gradient_norm' in result
        assert 'function_evaluations' in result
        assert 'gradient_evaluations' in result
        assert 'duration_seconds' in result
        assert 'termination_reason' in result
    
    def test_optimization_on_stiefel(self):
        """Test Newton optimization on Stiefel manifold."""
        n, p = 10, 3
        manifold = Stiefel(n, p)
        optimizer = Newton(
            max_iterations=50,
            tolerance=1e-5,
            hessian_regularization=1e-3
        )
        
        # Cost function: Procrustes problem
        np.random.seed(123)
        A = np.random.randn(n, p)
        
        def cost_fn(X):
            return -np.trace(A.T @ X)
        
        def grad_fn(X):
            return -A
        
        cost = CostFunction(cost_fn, grad_fn)
        
        # Initial point
        x0 = manifold.random_point()
        
        # Optimize
        result = optimizer.optimize(cost, manifold, x0)
        
        # Check that we're still on the manifold
        x_opt = result['point'].reshape(n, p)
        # Verify solution is still on Stiefel manifold
        # X should satisfy X^T X = I
        XtX = x_opt.T @ x_opt
        np.testing.assert_allclose(XtX, np.eye(p), atol=1e-6)
    
    def test_optimization_on_spd(self):
        """Test Newton optimization on SPD manifold."""
        n = 5
        manifold = SPD(n)
        optimizer = Newton(
            max_iterations=50,
            tolerance=1e-5,
            hessian_regularization=1e-2,
            max_cg_iterations=30
        )
        
        # Cost function: minimize log det + trace
        def cost_fn(X_vec):
            X = X_vec.reshape(n, n)
            sign, logdet = np.linalg.slogdet(X)
            return -logdet + np.trace(X)
        
        def grad_fn(X_vec):
            X = X_vec.reshape(n, n)
            grad = -np.linalg.inv(X) + np.eye(n)
            # Make symmetric
            grad = 0.5 * (grad + grad.T)
            return grad.flatten()
        
        cost = CostFunction(cost_fn, grad_fn)
        
        # Initial point
        x0 = manifold.random_point()
        
        # Optimize
        result = optimizer.optimize(cost, manifold, x0)
        
        # Check that we're still on the manifold
        x_opt = result['point']
        # Manifold-specific validation would go here
        # For now, just check the result exists
        assert x_opt is not None
    
    def test_optimization_on_oblique(self):
        """Test Newton optimization on Oblique manifold."""
        m, n = 5, 3
        manifold = Oblique(m, n)
        optimizer = Newton(
            max_iterations=50,
            tolerance=1e-5,
            hessian_regularization=1e-2
        )
        
        # Cost function: sum of quadratic forms
        np.random.seed(456)
        A = np.random.randn(m, m)
        A = A + A.T  # Make symmetric
        
        def cost_fn(X_vec):
            X = X_vec.reshape(m, n)
            value = 0.0
            for j in range(n):
                value += X[:, j].T @ A @ X[:, j]
            return value
        
        def grad_fn(X_vec):
            X = X_vec.reshape(m, n)
            grad = np.zeros_like(X)
            for j in range(n):
                grad[:, j] = 2 * A @ X[:, j]
            return grad.flatten()
        
        cost = CostFunction(cost_fn, grad_fn)
        
        # Initial point
        x0 = manifold.random_point()
        
        # Optimize
        result = optimizer.optimize(cost, manifold, x0)
        
        # Check that we're still on the manifold
        x_opt = result['point']
        # Manifold-specific validation would go here
        # For now, just check the result exists
        assert x_opt is not None
    
    def test_optimization_on_fixed_rank(self):
        """Test Newton optimization on Fixed-rank manifold."""
        m, n, k = 10, 8, 3
        manifold = FixedRank(m, n, k)
        optimizer = Newton(
            max_iterations=30,
            tolerance=1e-4,
            hessian_regularization=1e-1,  # Higher regularization for fixed-rank
            cg_tolerance=1e-4
        )
        
        # Cost function: matrix completion-like problem
        np.random.seed(789)
        M_target = np.random.randn(m, k) @ np.random.randn(k, n)
        mask = np.random.rand(m, n) > 0.7  # Observe 30% of entries
        
        def cost_fn(X_vec):
            X = X_vec.reshape(m, n)
            diff = (X - M_target) * mask
            return 0.5 * np.sum(diff**2)
        
        def grad_fn(X_vec):
            X = X_vec.reshape(m, n)
            grad = (X - M_target) * mask
            return grad.flatten()
        
        cost = CostFunction(cost_fn, grad_fn)
        
        # Initial point
        x0 = manifold.random_point()
        
        # Optimize
        result = optimizer.optimize(cost, manifold, x0)
        
        # Check that we're still on the manifold
        x_opt = result['point']
        # Manifold-specific validation would go here
        # For now, just check the result exists
        assert x_opt is not None
    
    def test_optimization_on_psd_cone(self):
        """Test Newton optimization on PSD cone manifold."""
        n = 4
        manifold = PSDCone(n)
        optimizer = Newton(
            max_iterations=30,
            tolerance=1e-4,
            hessian_regularization=1e-2
        )
        
        # Cost function: quadratic cost with PSD constraint
        np.random.seed(234)
        Q = np.random.randn(n, n)
        Q = Q + Q.T  # Make symmetric
        
        def cost_fn(X_vec):
            X = X_vec.reshape(n, n)
            return np.trace(Q @ X @ X)
        
        def grad_fn(X_vec):
            X = X_vec.reshape(n, n)
            grad = Q @ X + X @ Q.T
            return grad.flatten()
        
        cost = CostFunction(cost_fn, grad_fn)
        
        # Initial point
        x0 = manifold.random_point()
        
        # Optimize
        result = optimizer.optimize(cost, manifold, x0)
        
        # Check that we're still on the manifold
        x_opt = result['point']
        # Manifold-specific validation would go here
        # For now, just check the result exists
        assert x_opt is not None
    
    def test_cg_parameters(self):
        """Test CG solver parameters."""
        manifold = Sphere(20)
        
        # Test with very few CG iterations
        optimizer = Newton(
            max_cg_iterations=2,  # Very few iterations
            cg_tolerance=1e-10,   # Very tight tolerance (won't be reached)
            max_iterations=20
        )
        
        # Simple quadratic
        def cost_fn(x):
            return 0.5 * np.sum(x**2)
        
        def grad_fn(x):
            return x
        
        cost = CostFunction(cost_fn, grad_fn)
        x0 = manifold.random_point()
        
        # Should still run but maybe not converge well
        result = optimizer.optimize(cost, manifold, x0)
        assert 'iterations' in result
    
    def test_regularization_effect(self):
        """Test effect of Hessian regularization."""
        n = 10
        manifold = Sphere(n)
        
        # Create a poorly conditioned problem
        np.random.seed(567)
        U, _, Vt = np.linalg.svd(np.random.randn(n, n))
        # Create matrix with large condition number
        S = np.diag(np.logspace(0, -6, n))
        A = U @ S @ Vt
        
        def cost_fn(x):
            return 0.5 * x.T @ A @ x
        
        def grad_fn(x):
            return A @ x
        
        cost = CostFunction(cost_fn, grad_fn)
        x0 = manifold.random_point()
        
        # Without regularization (might fail)
        optimizer1 = riemannopt.Newton(
            hessian_regularization=0.0,
            max_iterations=20
        )
        
        # With regularization (should be more stable)
        optimizer2 = riemannopt.Newton(
            hessian_regularization=1e-3,
            max_iterations=20
        )
        
        # Both should run without crashing
        result1 = optimizer1.optimize(cost, manifold, x0)
        result2 = optimizer2.optimize(cost, manifold, x0)
        
        # Regularized version should be more stable
        assert result2['iterations'] <= result1['iterations']
    
    def test_unsupported_manifold(self):
        """Test error on unsupported manifold."""
        optimizer = Newton()
        
        # Create a custom manifold (not supported)
        class CustomManifold:
            pass
        
        manifold = CustomManifold()
        cost = CostFunction(lambda x: 0.0, lambda x: x)
        x0 = np.zeros(5)
        
        with pytest.raises(ValueError, match="Unsupported manifold type"):
            optimizer.optimize(cost, manifold, x0)
    
    def test_euclidean_not_supported(self):
        """Test that Euclidean manifold gives proper error."""
        optimizer = Newton()
        manifold = Euclidean(5)
        
        cost = CostFunction(lambda x: np.sum(x**2), lambda x: 2*x)
        x0 = np.ones(5)
        
        with pytest.raises(ValueError, match="Euclidean manifold not yet supported"):
            optimizer.optimize(cost, manifold, x0)
    
    def test_gauss_newton_mode(self):
        """Test Gauss-Newton mode."""
        optimizer = Newton(use_gauss_newton=True)
        manifold = Sphere(5)
        
        def cost_fn(x):
            return np.sum(x**2)
        
        def grad_fn(x):
            return 2 * x
        
        cost = CostFunction(cost_fn, grad_fn)
        x0 = manifold.random_point()
        
        # Currently not implemented, should fail
        with pytest.raises(ValueError):
            optimizer.optimize(cost, manifold, x0)
    
    def test_representation(self):
        """Test string representation."""
        optimizer = Newton(
            hessian_regularization=1e-6,
            use_gauss_newton=True,
            max_cg_iterations=50,
            cg_tolerance=1e-8,
            max_iterations=100,
            tolerance=1e-7
        )
        
        repr_str = repr(optimizer)
        assert "Newton" in repr_str
        assert "1e-06" in repr_str or "1e-6" in repr_str
        assert "True" in repr_str
        assert "50" in repr_str
        assert "100" in repr_str
        assert "1e-07" in repr_str or "1e-7" in repr_str