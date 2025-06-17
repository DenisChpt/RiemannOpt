"""
Integration tests for complete optimization problems.

This module tests end-to-end optimization scenarios including PCA,
matrix completion, and other real-world applications.
"""

import pytest
import numpy as np
from typing import Tuple, Optional
from conftest import TOLERANCES, riemannopt


class TestPCAOptimization:
    """Test Principal Component Analysis on Stiefel manifold."""
    
    @pytest.mark.integration
    def test_pca_low_rank_data(self, stiefel_factory, sgd_factory):
        """Test PCA on data with known low-rank structure."""
        # Generate low-rank data
        n_samples, n_features = 500, 20
        rank = 3
        
        # True components
        U_true = np.linalg.qr(np.random.randn(n_features, rank))[0]
        
        # Generate data: X = U @ S @ V^T + noise
        S = np.diag([10, 5, 2])  # Clear singular values
        V = np.random.randn(n_samples, rank)
        data = V @ S @ U_true.T
        data += 0.1 * np.random.randn(n_samples, n_features)  # Small noise
        
        # Compute empirical covariance
        data_centered = data - np.mean(data, axis=0)
        C = data_centered.T @ data_centered / n_samples
        
        # Optimize on Stiefel
        stiefel = stiefel_factory(n_features, rank)
        sgd = sgd_factory(step_size=0.001, momentum=0.9)
        
        # Cost function: maximize tr(U^T C U)
        def cost_fn(U):
            return -np.trace(U.T @ C @ U)
        
        def grad_fn(U):
            return -2 * C @ U
        
        # Initial point
        U = stiefel.random_point()
        
        # Optimize
        costs = []
        for i in range(500):
            grad = grad_fn(U)
            U = sgd.step(stiefel, U, grad)
            if i % 50 == 0:
                costs.append(cost_fn(U))
        
        # Check convergence
        assert costs[-1] < costs[0]  # Cost decreased
        
        # Check subspace alignment with true components
        # Compute principal angles
        M = U_true.T @ U
        s = np.linalg.svd(M, compute_uv=False)
        principal_angles = np.arccos(np.clip(s, -1, 1))
        
        # All angles should be small (subspaces align)
        assert np.max(principal_angles) < 0.2  # ~11 degrees tolerance
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pca_convergence_rate(self, stiefel_factory, sgd_factory, adam_factory):
        """Compare convergence rates of different optimizers for PCA."""
        n, p = 50, 5
        
        # Generate data with exponentially decaying eigenvalues
        eigenvalues = np.exp(-np.arange(n) / 5)
        Q = np.linalg.qr(np.random.randn(n, n))[0]
        C = Q @ np.diag(eigenvalues) @ Q.T
        
        # Expected optimal value
        optimal_cost = -np.sum(eigenvalues[:p])
        
        # Test different optimizers
        optimizers = [
            ('SGD', sgd_factory(step_size=0.001)),
            ('SGD+momentum', sgd_factory(step_size=0.001, momentum=0.9)),
            ('Adam', adam_factory(learning_rate=0.01)),
        ]
        
        results = {}
        stiefel = stiefel_factory(n, p)
        
        for name, optimizer in optimizers:
            X = stiefel.random_point()  # Same initial point
            costs = []
            
            for i in range(300):
                grad = -2 * C @ X
                X = optimizer.step(stiefel, X, grad)
                if i % 10 == 0:
                    cost = -np.trace(X.T @ C @ X)
                    costs.append(cost)
            
            results[name] = {
                'costs': costs,
                'final_cost': costs[-1],
                'gap': abs(costs[-1] - optimal_cost)
            }
        
        # All should converge reasonably close to optimum
        for name, result in results.items():
            assert result['gap'] < 0.5, f"{name} did not converge well"
        
        # Momentum should converge faster than vanilla SGD
        assert results['SGD+momentum']['gap'] < results['SGD']['gap']


class TestRayleighQuotientOptimization:
    """Test Rayleigh quotient optimization on sphere."""
    
    @pytest.mark.integration
    def test_generalized_eigenvalue_problem(self, sphere_factory, sgd_factory):
        """Test finding smallest generalized eigenvalue via Rayleigh quotient."""
        n = 30
        
        # Create SPD matrices A and B
        A = np.random.randn(n, n)
        A = A @ A.T + 0.1 * np.eye(n)  # Ensure positive definite
        
        B = np.random.randn(n, n)
        B = B @ B.T + 0.1 * np.eye(n)
        
        # Solve generalized eigenvalue problem A v = λ B v
        eigenvalues, eigenvectors = np.linalg.eig(np.linalg.inv(B) @ A)
        eigenvalues = eigenvalues.real
        min_idx = np.argmin(eigenvalues)
        min_eigenvalue = eigenvalues[min_idx]
        
        # Optimize Rayleigh quotient on sphere
        sphere = sphere_factory(n)
        
        def cost_fn(x):
            return float((x.T @ A @ x) / (x.T @ B @ x))
        
        def grad_fn(x):
            xTBx = x.T @ B @ x
            xTAx = x.T @ A @ x
            return 2 * (A @ x * xTBx - B @ x * xTAx) / (xTBx ** 2)
        
        # Run optimization
        sgd = sgd_factory(step_size=0.01, momentum=0.5)
        x = sphere.random_point()
        
        for _ in range(500):
            grad = grad_fn(x)
            x = sgd.step(sphere, x, grad)
        
        final_cost = cost_fn(x)
        
        # Should find minimum eigenvalue
        assert abs(final_cost - min_eigenvalue) < 0.1


class TestMatrixCompletion:
    """Test low-rank matrix completion on Grassmann manifold."""
    
    @pytest.mark.integration
    def test_matrix_completion_grassmann(self, grassmann_factory, sgd_factory):
        """Test matrix completion using Grassmann manifold optimization."""
        # Problem setup
        m, n = 30, 25
        rank = 5
        
        # Generate true low-rank matrix
        U_true = np.random.randn(m, rank)
        V_true = np.random.randn(n, rank)
        M_true = U_true @ V_true.T
        
        # Observe random entries
        n_observed = int(0.5 * m * n)  # 50% observed
        observed_indices = np.random.choice(m * n, n_observed, replace=False)
        observed_mask = np.zeros((m, n), dtype=bool)
        observed_mask.flat[observed_indices] = True
        
        # Observed values
        M_observed = np.zeros((m, n))
        M_observed[observed_mask] = M_true[observed_mask]
        
        # Optimize on Grassmann manifold
        grassmann = grassmann_factory(m, rank)
        
        def cost_fn(U):
            # Given U, find optimal V by least squares
            V = np.zeros((n, rank))
            for j in range(n):
                mask_j = observed_mask[:, j]
                if np.any(mask_j):
                    U_masked = U[mask_j, :]
                    y_masked = M_observed[mask_j, j]
                    V[j, :] = np.linalg.lstsq(U_masked, y_masked, rcond=None)[0]
            
            # Compute reconstruction error
            M_reconstructed = U @ V.T
            error = np.sum((M_reconstructed[observed_mask] - M_observed[observed_mask])**2)
            return error
        
        def grad_fn(U):
            # Gradient via finite differences (simplified)
            eps = 1e-6
            grad = np.zeros_like(U)
            f0 = cost_fn(U)
            
            for i in range(rank):
                U_perturb = U.copy()
                U_perturb[:, i] += eps
                # Project back to manifold
                U_perturb = grassmann.project(U_perturb)
                f1 = cost_fn(U_perturb)
                grad[:, i] = (f1 - f0) / eps
            
            return grad
        
        # Optimize
        sgd = sgd_factory(step_size=0.001)
        U = grassmann.random_point()
        
        initial_cost = cost_fn(U)
        for _ in range(100):  # Fewer iterations for speed
            grad = grad_fn(U)
            U = sgd.step(grassmann, U, grad)
        
        final_cost = cost_fn(U)
        
        # Cost should decrease
        assert final_cost < initial_cost
        
        # Reconstruction should be reasonable
        # (This is a simple test; real matrix completion needs more sophistication)
        relative_error = final_cost / np.sum(M_observed[observed_mask]**2)
        assert relative_error < 0.5  # Within 50% relative error


class TestManifoldConstrainedOptimization:
    """Test optimization with multiple manifold constraints."""
    
    @pytest.mark.integration
    def test_product_manifold_optimization(self, sphere_factory, stiefel_factory, sgd_factory):
        """Test optimization on product of manifolds."""
        # Product of sphere and Stiefel
        sphere = sphere_factory(10)
        stiefel = stiefel_factory(8, 3)
        
        # Create coupled optimization problem
        A = np.random.randn(10, 10)
        A = A + A.T
        B = np.random.randn(8, 8)
        B = B + B.T
        C = np.random.randn(10, 8)  # Coupling term
        
        def cost_fn(x, Y):
            """Coupled cost function f(x, Y) = x^T A x + tr(Y^T B Y) + x^T C Y 1."""
            return float(x.T @ A @ x + np.trace(Y.T @ B @ Y) + x.T @ C @ Y @ np.ones(3))
        
        def grad_x(x, Y):
            """Gradient with respect to x."""
            return 2 * A @ x + C @ Y @ np.ones(3)
        
        def grad_Y(x, Y):
            """Gradient with respect to Y."""
            return 2 * B @ Y + np.outer(C.T @ x, np.ones(3))
        
        # Alternating optimization
        sgd = sgd_factory(step_size=0.01)
        x = sphere.random_point()
        Y = stiefel.random_point()
        
        costs = []
        for _ in range(100):
            # Update x
            gx = grad_x(x, Y)
            x = sgd.step(sphere, x, gx)
            
            # Update Y
            gY = grad_Y(x, Y)
            Y = sgd.step(stiefel, Y, gY)
            
            costs.append(cost_fn(x, Y))
        
        # Check convergence trend
        assert np.mean(costs[-10:]) < np.mean(costs[:10])


class TestRobustOptimization:
    """Test optimization with outliers and noise."""
    
    @pytest.mark.integration
    def test_robust_pca_on_stiefel(self, stiefel_factory, sgd_factory):
        """Test robust PCA with outliers."""
        n, p = 50, 3
        n_samples = 200
        
        # Generate clean low-rank data
        U_true = np.linalg.qr(np.random.randn(n, p))[0]
        V = np.random.randn(n_samples, p)
        clean_data = V @ np.diag([5, 3, 1]) @ U_true.T
        
        # Add outliers
        n_outliers = 20
        outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
        data = clean_data.copy()
        data[outlier_indices] = np.random.randn(n_outliers, n) * 10
        
        # Robust covariance estimation using Huber loss
        def huber_loss(r, delta=1.0):
            """Huber loss for robust estimation."""
            return np.where(np.abs(r) <= delta,
                           0.5 * r**2,
                           delta * (np.abs(r) - 0.5 * delta))
        
        stiefel = stiefel_factory(n, p)
        
        def robust_cost(U):
            """Robust PCA cost using Huber loss."""
            # Project data onto subspace
            projections = data @ U
            reconstructions = projections @ U.T
            residuals = data - reconstructions
            
            # Huber loss on residuals
            return np.sum(huber_loss(residuals.flatten(), delta=2.0))
        
        def robust_grad(U):
            """Gradient of robust cost."""
            projections = data @ U
            reconstructions = projections @ U.T
            residuals = data - reconstructions
            
            # Huber loss derivative
            huber_weight = np.where(np.abs(residuals) <= 2.0,
                                   residuals,
                                   2.0 * np.sign(residuals))
            
            # Gradient computation
            grad = -2 * data.T @ huber_weight @ U.T @ U + 2 * data.T @ data @ U
            return grad / n_samples
        
        # Optimize
        sgd = sgd_factory(step_size=0.001, momentum=0.9)
        U = stiefel.random_point()
        
        costs = []
        for i in range(300):
            grad = robust_grad(U)
            U = sgd.step(stiefel, U, grad)
            if i % 30 == 0:
                costs.append(robust_cost(U))
        
        # Should converge
        assert costs[-1] < costs[0]
        
        # Check subspace recovery (should be close to true subspace despite outliers)
        M = U_true.T @ U
        s = np.linalg.svd(M, compute_uv=False)
        principal_angles = np.arccos(np.clip(s, -1, 1))
        
        # Should recover subspace reasonably well
        assert np.max(principal_angles) < 0.5  # ~28 degrees tolerance


class TestConstrainedOptimizationApplications:
    """Test real-world applications with manifold constraints."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_orthogonal_procrustes_problem(self, stiefel_factory, sgd_factory):
        """Test orthogonal Procrustes problem for shape alignment."""
        n = 20
        
        # Generate two related point clouds
        X = np.random.randn(n, 3)  # Original shape
        
        # Apply transformation: rotation + reflection + noise
        true_rotation = np.linalg.qr(np.random.randn(3, 3))[0]
        if np.linalg.det(true_rotation) < 0:
            true_rotation[:, 0] *= -1  # Ensure rotation (not reflection)
        
        Y = X @ true_rotation.T + 0.1 * np.random.randn(n, 3)  # Transformed + noise
        
        # Orthogonal Procrustes: min ||Y - X @ R||_F^2 s.t. R^T R = I
        stiefel = stiefel_factory(3, 3)  # Orthogonal group O(3)
        
        def cost_fn(R):
            return np.linalg.norm(Y - X @ R, 'fro')**2
        
        def grad_fn(R):
            return -2 * X.T @ (Y - X @ R)
        
        # Optimize
        sgd = sgd_factory(step_size=0.01, momentum=0.9)
        R = stiefel.random_point()
        
        for _ in range(200):
            grad = grad_fn(R)
            R = sgd.step(stiefel, R, grad)
        
        # Check solution quality
        # R should be close to true_rotation
        alignment = np.trace(R.T @ true_rotation) / 3  # Average cosine
        assert alignment > 0.9  # Good alignment
        
        # Check orthogonality is preserved
        assert np.allclose(R @ R.T, np.eye(3), atol=TOLERANCES['default'])
    
    @pytest.mark.integration
    def test_correlation_matrix_nearest_problem(self, sphere_factory, sgd_factory):
        """Test finding nearest correlation matrix with factor structure."""
        n = 10
        k = 3  # Number of factors
        
        # Generate approximate correlation matrix
        L = np.random.randn(n, k)
        C_approx = L @ L.T
        D = np.diag(1.0 / np.sqrt(np.diag(C_approx)))
        C_approx = D @ C_approx @ D  # Normalize to correlation
        
        # Add noise to make it non-positive definite
        C_noisy = C_approx + 0.1 * np.random.randn(n, n)
        C_noisy = (C_noisy + C_noisy.T) / 2
        
        # Find nearest correlation matrix with factor structure
        # Parameterize as C = L @ L^T with L having unit norm columns
        
        # Each column of L lives on a sphere
        spheres = [sphere_factory(n) for _ in range(k)]
        sgd = sgd_factory(step_size=0.001)
        
        # Initialize factor loadings
        L_cols = [sphere.random_point() for sphere in spheres]
        
        def cost_fn(L_cols):
            L = np.column_stack(L_cols)
            C = L @ L.T
            # Project to correlation matrix
            d = np.sqrt(np.diag(C))
            d[d < 1e-10] = 1.0
            C_corr = C / np.outer(d, d)
            return np.linalg.norm(C_corr - C_noisy, 'fro')**2
        
        # Optimize each factor
        for _ in range(100):
            for j in range(k):
                # Finite difference gradient for column j
                eps = 1e-6
                f0 = cost_fn(L_cols)
                
                L_cols_perturb = L_cols.copy()
                L_cols_perturb[j] = L_cols[j] + eps * spheres[j].random_tangent(L_cols[j])
                L_cols_perturb[j] = spheres[j].project(L_cols_perturb[j])
                f1 = cost_fn(L_cols_perturb)
                
                grad_direction = (f1 - f0) / eps * spheres[j].random_tangent(L_cols[j])
                L_cols[j] = sgd.step(spheres[j], L_cols[j], grad_direction)
        
        # Check result
        L_final = np.column_stack(L_cols)
        C_final = L_final @ L_final.T
        d = np.sqrt(np.diag(C_final))
        C_final_corr = C_final / np.outer(d, d)
        
        # Should be valid correlation matrix
        eigenvalues = np.linalg.eigvalsh(C_final_corr)
        assert np.min(eigenvalues) > -TOLERANCES['numerical']  # Positive semidefinite
        assert np.allclose(np.diag(C_final_corr), 1.0, atol=TOLERANCES['default'])  # Unit diagonal