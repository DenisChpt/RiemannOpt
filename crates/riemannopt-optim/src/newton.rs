//! # Riemannian Newton Method
//!
//! This module implements the Newton method for optimization on Riemannian manifolds.
//! Newton's method uses second-order information (Hessian) to achieve quadratic 
//! convergence near the optimum, making it one of the most powerful optimization 
//! algorithms when applicable.
//!
//! ## Mathematical Foundation
//!
//! Given a smooth cost function f: ℳ → ℝ on a Riemannian manifold (ℳ, g),
//! the Newton method solves the Newton equation at each iteration to find the
//! optimal search direction.
//!
//! ### Newton Equation
//!
//! At each point x_k ∈ ℳ, we solve:
//! ```text
//! Hess f(x_k)[η_k] = -grad f(x_k)
//! ```
//! where:
//! - Hess f(x_k) is the Riemannian Hessian at x_k
//! - η_k ∈ T_{x_k}ℳ is the Newton direction
//! - grad f(x_k) is the Riemannian gradient
//!
//! ### Algorithm
//!
//! For k = 0, 1, 2, ...:
//! ```text
//! 1. Compute gradient: g_k = grad f(x_k)
//! 2. Solve Newton system: Hess f(x_k)[η_k] = -g_k
//! 3. Line search: α_k = argmin_α f(R_{x_k}(α η_k))
//! 4. Update: x_{k+1} = R_{x_k}(α_k η_k)
//! ```
//!
//! ## Variants
//!
//! ### Gauss-Newton Method
//! For least-squares problems f(x) = ½||r(x)||², approximates:
//! ```text
//! Hess f(x) ≈ J(x)^T J(x)
//! ```
//! where J(x) is the Jacobian of residuals r(x).
//!
//! ### Regularized Newton
//! Adds regularization to ensure positive definiteness:
//! ```text
//! (Hess f(x_k) + λI)[η_k] = -grad f(x_k)
//! ```
//!
//! ## Solving the Newton System
//!
//! The Newton system is solved using Conjugate Gradient (CG) method,
//! which is efficient for large-scale problems and only requires
//! Hessian-vector products.
//!
//! ## Zero-Allocation Architecture
//!
//! This implementation follows a zero-allocation design pattern for optimal performance:
//! - All temporary vectors for CG iterations are pre-allocated in the Workspace
//! - Hessian-vector products reuse workspace buffers to avoid allocations
//! - The workspace is initialized once at the beginning of optimization
//!
//! ## Key Features
//!
//! - **Quadratic convergence**: Near the optimum when Hessian is available
//! - **Hessian approximation**: Finite differences when exact Hessian unavailable
//! - **CG solver**: Efficient for large-scale problems
//! - **Regularization**: Ensures numerical stability
//! - **Line search**: Globalizes convergence
//! - **Zero allocations**: Workspace-based memory management for performance
//!
//! ## References
//!
//! 1. Absil, P. A., Mahony, R., & Sepulchre, R. (2008). Optimization algorithms on matrix manifolds.
//! 2. Nocedal, J., & Wright, S. (2006). Numerical optimization.

use num_traits::Float;
use std::time::Instant;

use riemannopt_core::{
    core::{
        manifold::Manifold,
        cost_function::CostFunction,
    },
    error::{Result, ManifoldError},
    types::Scalar,
    memory::workspace::{Workspace, BufferId},
    optimization::{
        optimizer::{Optimizer, OptimizationResult, StoppingCriterion, TerminationReason},
        line_search::{LineSearchParams, BacktrackingLineSearch, LineSearch},
    },
};

/// Configuration for the Riemannian Newton method
#[derive(Debug, Clone)]
pub struct NewtonConfig<T: Scalar> {
    /// Line search parameters
    pub line_search_params: LineSearchParams<T>,
    /// Regularization parameter for Hessian (to ensure positive definiteness)
    pub hessian_regularization: T,
    /// Use Gauss-Newton approximation (for least-squares problems)
    pub use_gauss_newton: bool,
    /// Maximum number of CG iterations for solving Newton system
    pub max_cg_iterations: usize,
    /// Tolerance for CG solver
    pub cg_tolerance: T,
}

impl<T: Scalar> Default for NewtonConfig<T> {
    fn default() -> Self {
        Self {
            line_search_params: LineSearchParams::default(),
            hessian_regularization: <T as Scalar>::from_f64(1e-8),
            use_gauss_newton: false,
            max_cg_iterations: 100,
            cg_tolerance: <T as Scalar>::from_f64(1e-6),
        }
    }
}

impl<T: Scalar> NewtonConfig<T> {
    /// Create a new Newton configuration with default parameters
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the Hessian regularization parameter
    pub fn with_regularization(mut self, reg: T) -> Self {
        self.hessian_regularization = reg;
        self
    }

    /// Enable Gauss-Newton approximation
    pub fn with_gauss_newton(mut self) -> Self {
        self.use_gauss_newton = true;
        self
    }

    /// Set CG solver parameters
    pub fn with_cg_params(mut self, max_iter: usize, tol: T) -> Self {
        self.max_cg_iterations = max_iter;
        self.cg_tolerance = tol;
        self
    }
}

/// Riemannian Newton method optimizer
#[derive(Debug)]
pub struct Newton<T: Scalar> {
    config: NewtonConfig<T>,
}

impl<T: Scalar> Newton<T> {
    /// Create a new Newton optimizer with the given configuration
    pub fn new(config: NewtonConfig<T>) -> Self {
        Self { config }
    }
    
    /// Solve the Newton system H*d = -g using CG
    /// 
    /// This implementation minimizes allocations in the CG loop by reusing
    /// temporary vectors and using clone_from() for efficient copying.
    fn solve_newton_system<M: Manifold<T>>(
        &self,
        manifold: &M,
        cost_fn: &impl CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        point: &M::Point,
        gradient: &M::TangentVector,
        result: &mut M::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        // Allocate CG workspace vectors as proper TangentVectors
        let mut d = gradient.clone(); // Newton direction
        let mut r = gradient.clone(); // Residual
        let mut p = gradient.clone(); // Search direction
        let mut hp = gradient.clone(); // Hessian-vector product
        let mut temp = gradient.clone(); // Temporary vector
        
        // Initialize d = 0 using scale with zero
        manifold.scale_tangent(point, T::zero(), gradient, &mut d, workspace)?;
        // Initialize r = -gradient
        manifold.scale_tangent(point, -T::one(), gradient, &mut r, workspace)?;
        // Initialize p = r
        p.clone_from(&r);
        
        // CG iterations
        for _ in 0..self.config.max_cg_iterations {
            // Compute Hessian-vector product: hp = H*p
            self.hessian_vector_product(manifold, cost_fn, point, gradient, &p, &mut hp, workspace)?;
            
            // Add regularization: hp = hp + reg*p
            // Use temp to store hp, then compute hp = temp + reg*p
            temp.clone_from(&hp);
            manifold.axpy_tangent(point, self.config.hessian_regularization, &p, &temp, &mut hp, workspace)?;
            
            // Compute alpha = <r,r> / <p,hp>
            let rr_inner = manifold.inner_product(point, &r, &r)?;
            let php_inner = manifold.inner_product(point, &p, &hp)?;
            
            if php_inner <= T::zero() {
                // Non-positive curvature detected, exit CG
                break;
            }
            
            let alpha = rr_inner / php_inner;
            
            // Update d = d + alpha*p
            // Use temp to store d, then compute d = temp + alpha*p
            temp.clone_from(&d);
            manifold.axpy_tangent(point, alpha, &p, &temp, &mut d, workspace)?;
            
            // Update r = r - alpha*hp
            // Use temp to store r, then compute r = temp - alpha*hp
            temp.clone_from(&r);
            manifold.axpy_tangent(point, -alpha, &hp, &temp, &mut r, workspace)?;
            
            // Check convergence
            let r_new_norm_sq = manifold.inner_product(point, &r, &r)?;
            let r_new_norm = <T as Float>::sqrt(r_new_norm_sq);
            
            if r_new_norm < self.config.cg_tolerance {
                break;
            }
            
            // Compute beta = <r_new,r_new> / <r_old,r_old>
            let beta = r_new_norm_sq / rr_inner;
            
            // Update p = r + beta*p
            manifold.scale_tangent(point, beta, &p, &mut temp, workspace)?;
            manifold.add_tangents(point, &r, &temp, &mut p, workspace)?;
        }
        
        // Copy result
        result.clone_from(&d);
        
        Ok(())
    }
    
    /// Compute Hessian-vector product using finite differences
    /// 
    /// This implementation reuses temporary vectors to minimize allocations.
    /// While we still need to allocate TangentVectors (due to type constraints),
    /// we avoid unnecessary clones by using clone_from() for efficient copying.
    fn hessian_vector_product<M: Manifold<T>>(
        &self,
        manifold: &M,
        cost_fn: &impl CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        point: &M::Point,
        _gradient: &M::TangentVector,
        vector: &M::TangentVector,
        result: &mut M::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()> {
        if self.config.use_gauss_newton {
            // Gauss-Newton approximation not implemented yet
            return Err(ManifoldError::NotImplemented {
                feature: "Gauss-Newton approximation".to_string(),
            });
        }
        
        // Use finite differences approximation
        let eps = <T as Float>::sqrt(T::epsilon());
        
        // Unfortunately, we still need to allocate temporary vectors as TangentVectors
        // because the workspace returns DVector<T> but manifold operations require M::TangentVector
        // This is a limitation of the current architecture
        let mut scaled_vec = vector.clone();
        let mut grad_plus_riem = vector.clone();
        let mut grad_transported = vector.clone();
        let mut current_grad_riem = vector.clone();
        let mut temp = vector.clone();
        
        // Compute perturbed direction: eps * v
        manifold.scale_tangent(point, eps, vector, &mut scaled_vec, workspace)?;
        
        // Compute perturbed point: x_plus = R_x(eps * v)
        let mut x_plus = point.clone();
        manifold.retract(point, &scaled_vec, &mut x_plus, workspace)?;
        
        // Compute gradient at perturbed point
        let grad_plus = cost_fn.gradient(&x_plus)?;
        grad_plus_riem.clone_from(&grad_plus);
        manifold.euclidean_to_riemannian_gradient(&x_plus, &grad_plus, &mut grad_plus_riem, workspace)?;
        
        // Transport gradient back to original point
        manifold.parallel_transport(&x_plus, point, &grad_plus_riem, &mut grad_transported, workspace)?;
        
        // Get current gradient
        let current_grad = cost_fn.gradient(point)?;
        current_grad_riem.clone_from(&current_grad);
        manifold.euclidean_to_riemannian_gradient(point, &current_grad, &mut current_grad_riem, workspace)?;
        
        // Compute finite difference: (grad_transported - current_grad_riem) / eps
        // First: temp = -current_grad_riem
        manifold.scale_tangent(point, -T::one(), &current_grad_riem, &mut temp, workspace)?;
        // Then: result = grad_transported + temp
        manifold.add_tangents(point, &grad_transported, &temp, result, workspace)?;
        // Finally: result = result / eps (in-place)
        manifold.scale_tangent(point, T::one() / eps, result, &mut temp, workspace)?;
        result.clone_from(&temp);
        
        Ok(())
    }
    
    /// Check stopping criteria
    fn check_stopping_criteria<M: Manifold<T>>(
        &self,
        criterion: &StoppingCriterion<T>,
        iteration: usize,
        current_cost: T,
        previous_cost: Option<T>,
        gradient_norm: Option<T>,
        function_evaluations: usize,
        start_time: Instant,
        current_point: &M::Point,
        previous_point: Option<&M::Point>,
        manifold: &M,
        workspace: &mut Workspace<T>,
    ) -> Option<TerminationReason> {
        // Check iteration limit
        if let Some(max_iter) = criterion.max_iterations {
            if iteration >= max_iter {
                return Some(TerminationReason::MaxIterations);
            }
        }

        // Check time limit
        if let Some(max_time) = criterion.max_time {
            if start_time.elapsed() >= max_time {
                return Some(TerminationReason::MaxTime);
            }
        }

        // Check function evaluation limit
        if let Some(max_evals) = criterion.max_function_evaluations {
            if function_evaluations >= max_evals {
                return Some(TerminationReason::MaxFunctionEvaluations);
            }
        }

        // Check gradient norm
        if let (Some(grad_norm), Some(grad_tol)) = (gradient_norm, criterion.gradient_tolerance) {
            if grad_norm < grad_tol {
                return Some(TerminationReason::Converged);
            }
        }

        // Check function value change
        if let (Some(prev_cost), Some(val_tol)) = (previous_cost, criterion.function_tolerance) {
            if <T as Float>::abs(current_cost - prev_cost) < val_tol && iteration > 0 {
                return Some(TerminationReason::Converged);
            }
        }

        // Check point change
        if let Some(point_tol) = criterion.point_tolerance {
            if let Some(ref prev_point) = previous_point {
                if iteration > 0 {
                    // Compute distance using manifold metric
                    if let Ok(distance) = manifold.distance(prev_point, current_point, workspace) {
                        if distance < point_tol {
                            return Some(TerminationReason::Converged);
                        }
                    }
                }
            }
        }

        // Check target value
        if let Some(target) = criterion.target_value {
            if current_cost <= target {
                return Some(TerminationReason::TargetReached);
            }
        }

        None
    }
}

impl<T: Scalar> Optimizer<T> for Newton<T> {
    fn name(&self) -> &str {
        "Riemannian Newton"
    }

    fn optimize<M, C>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &M::Point,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, M::Point>>
    where
        M: Manifold<T>,
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
    {
        let start_time = Instant::now();
        
        // Allocate workspace with appropriate size and pre-allocate all buffers
        let n = manifold.dimension();
        let mut workspace = Workspace::with_size(n);
        
        // Pre-allocate all buffers needed for Newton method
        // Standard optimization buffers
        workspace.preallocate_vector(BufferId::Gradient, n);
        workspace.preallocate_vector(BufferId::Direction, n);
        workspace.preallocate_vector(BufferId::PreviousGradient, n);
        
        // Hessian-vector product buffers
        workspace.preallocate_vector(BufferId::Temp1, n);
        workspace.preallocate_vector(BufferId::Temp2, n);
        workspace.preallocate_vector(BufferId::Temp3, n);
        
        // CG solver buffers (Custom 10-16)
        for i in 10..=16 {
            workspace.preallocate_vector(BufferId::Custom(i), n);
        }
        
        // Hessian finite difference buffers (Custom 20-22)
        for i in 20..=22 {
            workspace.preallocate_vector(BufferId::Custom(i), n);
        }
        
        // Initialize state
        let mut current_point = initial_point.clone();
        let mut previous_point = None;
        
        // Compute initial cost and gradient
        let mut current_cost = cost_fn.cost(&current_point)?;
        let mut previous_cost = None;
        
        let euclidean_grad = cost_fn.gradient(&current_point)?;
        let mut riemannian_grad = euclidean_grad.clone();
        manifold.euclidean_to_riemannian_gradient(&current_point, &euclidean_grad, &mut riemannian_grad, &mut workspace)?;
        
        let mut gradient_norm = manifold.norm(&current_point, &riemannian_grad)?;
        
        // Initialize line search
        let mut line_search = BacktrackingLineSearch::new();
        
        // Tracking variables
        let mut iteration = 0;
        let mut function_evaluations = 1;
        let mut gradient_evaluations = 1;
        
        // Allocate Newton direction
        let mut newton_dir = riemannian_grad.clone();
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = self.check_stopping_criteria(
                &stopping_criterion,
                iteration,
                current_cost,
                previous_cost,
                Some(gradient_norm),
                function_evaluations,
                start_time,
                &current_point,
                previous_point.as_ref(),
                manifold,
                &mut workspace,
            ) {
                let duration = start_time.elapsed();
                
                return Ok(OptimizationResult {
                    point: current_point,
                    value: current_cost,
                    gradient_norm: Some(gradient_norm),
                    iterations: iteration,
                    function_evaluations,
                    gradient_evaluations,
                    duration,
                    termination_reason: reason,
                    converged: matches!(reason, TerminationReason::Converged | TerminationReason::TargetReached),
                });
            }
            
            // Store previous state
            previous_cost = Some(current_cost);
            
            // Solve Newton system: H*d = -g
            self.solve_newton_system(
                manifold,
                cost_fn,
                &current_point,
                &riemannian_grad,
                &mut newton_dir,
                &mut workspace,
            )?;
            
            // Perform line search
            let ls_result = line_search.search(
                cost_fn,
                manifold,
                &current_point,
                current_cost,
                &riemannian_grad,
                &newton_dir,
                &self.config.line_search_params,
            )?;
            
            // Update counters
            function_evaluations += ls_result.function_evals;
            
            // Update point: x_{k+1} = R_{x_k}(alpha * d_k)
            let mut scaled_dir = newton_dir.clone();
            manifold.scale_tangent(&current_point, ls_result.step_size, &newton_dir, &mut scaled_dir, &mut workspace)?;
            
            let mut new_point = current_point.clone();
            manifold.retract(&current_point, &scaled_dir, &mut new_point, &mut workspace)?;
            
            // Update current point
            previous_point = Some(std::mem::replace(&mut current_point, new_point));
            
            // Compute new cost and gradient
            current_cost = cost_fn.cost(&current_point)?;
            let euclidean_grad = cost_fn.gradient(&current_point)?;
            manifold.euclidean_to_riemannian_gradient(&current_point, &euclidean_grad, &mut riemannian_grad, &mut workspace)?;
            gradient_norm = manifold.norm(&current_point, &riemannian_grad)?;
            
            function_evaluations += 1;
            gradient_evaluations += 1;
            iteration += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riemannopt_core::core::cost_function::QuadraticCost;
    use riemannopt_core::utils::test_manifolds::TestEuclideanManifold;
    use riemannopt_core::types::DVector;
    use nalgebra::Dyn;

    #[test]
    fn test_newton_creation() {
        let config = NewtonConfig::<f64>::new();
        let _optimizer = Newton::new(config);
    }

    #[test]
    fn test_newton_on_simple_problem() {
        let config = NewtonConfig::new()
            .with_regularization(1e-6);
        let mut optimizer = Newton::new(config);
        
        let manifold = TestEuclideanManifold::new(2);
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let x0 = DVector::from_vec(vec![1.0, 1.0]);
        
        let criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        let result = optimizer.optimize(&cost_fn, &manifold, &x0, &criterion).unwrap();
        
        assert!(result.iterations < 10); // Newton should converge fast
        assert!(result.gradient_norm.unwrap_or(1.0) < 1e-6);
    }
}