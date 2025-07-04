//! Riemannian Trust Region optimizer.
//!
//! Trust region methods are robust second-order optimization algorithms that use a
//! local quadratic model of the objective function within a "trust region" where
//! the model is assumed to be accurate. This implementation extends classical
//! trust region methods to Riemannian manifolds.
//!
//! # Algorithm Overview
//!
//! At each iteration, the trust region method:
//! 1. Builds a quadratic model of the objective function in the tangent space
//! 2. Solves a constrained subproblem to find the best step within the trust region
//! 3. Evaluates the actual vs predicted reduction
//! 4. Updates the trust region radius based on the reduction ratio
//! 5. Accepts or rejects the step
//!
//! # Key Features
//!
//! - **Steihaug-CG solver**: Efficient approximate solution of the trust region subproblem
//! - **Adaptive radius**: Automatic trust region size adjustment
//! - **Truncated Newton**: Support for inexact Hessian-vector products
//! - **Robust convergence**: Global convergence guarantees
//! - **Second-order convergence**: Superlinear convergence near the solution
//!
//! # References
//!
//! - Absil et al., "Trust-Region Methods on Riemannian Manifolds" (2007)
//! - Conn et al., "Trust Region Methods" (2000)
//! - Nocedal & Wright, "Numerical Optimization" (2006)

use riemannopt_core::{
    core::{
        manifold::Manifold,
        cost_function::CostFunction,
    },
    error::Result,
    types::Scalar,
    memory::workspace::{Workspace, BufferId},
    optimization::{
        optimizer::{Optimizer, OptimizationResult, StoppingCriterion, TerminationReason},
    },
};
use std::time::Instant;
use std::fmt::Debug;
use num_traits::Float;

/// Configuration for the Trust Region optimizer.
#[derive(Debug, Clone)]
pub struct TrustRegionConfig<T: Scalar> {
    /// Initial trust region radius
    pub initial_radius: T,
    /// Maximum trust region radius
    pub max_radius: T,
    /// Minimum trust region radius (below this, the algorithm terminates)
    pub min_radius: T,
    /// Ratio threshold for accepting a step (eta in literature)
    pub acceptance_ratio: T,
    /// Ratio threshold for increasing the trust region (typically 0.75)
    pub increase_threshold: T,
    /// Ratio threshold for decreasing the trust region (typically 0.25)
    pub decrease_threshold: T,
    /// Factor for increasing the trust region radius (typically 2.0)
    pub increase_factor: T,
    /// Factor for decreasing the trust region radius (typically 0.25)
    pub decrease_factor: T,
    /// Maximum iterations for the CG subproblem solver
    pub max_cg_iterations: Option<usize>,
    /// Tolerance for the CG subproblem solver
    pub cg_tolerance: T,
    /// Whether to use exact Hessian (true) or finite differences (false)
    pub use_exact_hessian: bool,
}

impl<T: Scalar> Default for TrustRegionConfig<T> {
    fn default() -> Self {
        Self {
            initial_radius: <T as Scalar>::from_f64(1.0),
            max_radius: <T as Scalar>::from_f64(10.0),
            min_radius: <T as Scalar>::from_f64(1e-6),
            acceptance_ratio: <T as Scalar>::from_f64(0.1),
            increase_threshold: <T as Scalar>::from_f64(0.75),
            decrease_threshold: <T as Scalar>::from_f64(0.25),
            increase_factor: <T as Scalar>::from_f64(2.0),
            decrease_factor: <T as Scalar>::from_f64(0.25),
            max_cg_iterations: None,
            cg_tolerance: <T as Scalar>::from_f64(1e-6),
            use_exact_hessian: false,
        }
    }
}

impl<T: Scalar> TrustRegionConfig<T> {
    /// Creates a new configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the initial trust region radius.
    pub fn with_initial_radius(mut self, radius: T) -> Self {
        self.initial_radius = radius;
        self
    }

    /// Sets the maximum trust region radius.
    pub fn with_max_radius(mut self, radius: T) -> Self {
        self.max_radius = radius;
        self
    }

    /// Sets the minimum trust region radius.
    pub fn with_min_radius(mut self, radius: T) -> Self {
        self.min_radius = radius;
        self
    }

    /// Sets the acceptance ratio threshold.
    pub fn with_acceptance_ratio(mut self, ratio: T) -> Self {
        self.acceptance_ratio = ratio;
        self
    }

    /// Enables exact Hessian computation.
    pub fn with_exact_hessian(mut self) -> Self {
        self.use_exact_hessian = true;
        self
    }

    /// Sets the maximum CG iterations.
    pub fn with_max_cg_iterations(mut self, max_iter: usize) -> Self {
        self.max_cg_iterations = Some(max_iter);
        self
    }

    /// Sets the CG tolerance.
    pub fn with_cg_tolerance(mut self, tol: T) -> Self {
        self.cg_tolerance = tol;
        self
    }
}

/// Workspace for the Steihaug-CG solver.
#[derive(Debug)]
struct CGWorkspace<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// Solution vector
    s: TV,
    /// Residual vector
    r: TV,
    /// Search direction
    d: TV,
    /// Hessian-vector product
    hd: TV,
    /// Temporary vector
    temp: TV,
    /// Whether we hit the trust region boundary
    boundary_hit: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T, TV> CGWorkspace<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// Creates a new CG workspace with zero vectors.
    fn new(zero_vector: TV) -> Self {
        Self {
            s: zero_vector.clone(),
            r: zero_vector.clone(),
            d: zero_vector.clone(),
            hd: zero_vector.clone(),
            temp: zero_vector,
            boundary_hit: false,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Riemannian Trust Region optimizer.
///
/// This optimizer adapts the classical trust region method to Riemannian manifolds
/// by solving the trust region subproblem in the tangent space and using retraction
/// for updates.
///
/// # Examples
///
/// ```rust,ignore
/// use riemannopt_optim::{TrustRegion, TrustRegionConfig};
/// 
/// // Basic trust region with default parameters
/// let tr: TrustRegion<f64> = TrustRegion::new(TrustRegionConfig::new());
/// 
/// // Trust region with custom parameters
/// let tr_custom = TrustRegion::new(
///     TrustRegionConfig::new()
///         .with_initial_radius(0.5)
///         .with_exact_hessian()
///         .with_max_cg_iterations(20)
/// );
/// ```
#[derive(Debug)]
pub struct TrustRegion<T: Scalar> {
    config: TrustRegionConfig<T>,
}

impl<T: Scalar> TrustRegion<T> {
    /// Creates a new Trust Region optimizer with the given configuration.
    pub fn new(config: TrustRegionConfig<T>) -> Self {
        Self {
            config,
        }
    }

    /// Creates a new Trust Region optimizer with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(TrustRegionConfig::default())
    }

    /// Returns the configuration.
    pub fn config(&self) -> &TrustRegionConfig<T> {
        &self.config
    }
    
    /// Checks stopping criteria internally
    fn check_stopping_criteria<M>(
        &self,
        manifold: &M,
        iteration: usize,
        function_evaluations: usize,
        _gradient_evaluations: usize,
        start_time: Instant,
        current_cost: T,
        previous_cost: Option<T>,
        gradient_norm: Option<T>,
        current_point: &M::Point,
        previous_point: &Option<M::Point>,
        workspace: &mut Workspace<T>,
        criterion: &StoppingCriterion<T>,
        trust_radius: T,
    ) -> Option<TerminationReason>
    where
        M: Manifold<T>,
    {
        // Check if trust region is too small
        if trust_radius < self.config.min_radius {
            return Some(TerminationReason::Converged);
        }

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
    
    /// Computes Hessian-vector product using finite differences.
    fn finite_diff_hessian_vec_product<M, C>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        gradient: &M::TangentVector,
        direction: &M::TangentVector,
        result: &mut M::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        let eps = <T as Float>::sqrt(T::epsilon());
        let norm = <T as Float>::sqrt(
            manifold.inner_product(point, direction, direction)?
        );
        
        if norm < T::epsilon() {
            manifold.scale_tangent(point, T::zero(), gradient, result, workspace)?;
            return Ok(());
        }
        
        let t = eps / norm;
        
        // Create perturbation: t * direction
        let mut perturbation = direction.clone();
        manifold.scale_tangent(point, t, direction, &mut perturbation, workspace)?;
        
        // Compute perturbed point
        let mut perturbed_point = point.clone();
        manifold.retract(point, &perturbation, &mut perturbed_point, workspace)?;
        
        // Compute gradient at perturbed point
        let mut perturbed_euclidean_grad = gradient.clone();
        let _cost = cost_fn.cost_and_gradient(&perturbed_point, workspace, &mut perturbed_euclidean_grad)?;
        
        let mut perturbed_grad = perturbed_euclidean_grad.clone();
        manifold.euclidean_to_riemannian_gradient(&perturbed_point, &perturbed_euclidean_grad, &mut perturbed_grad, workspace)?;
        
        // Transport gradient back
        let mut transported_grad = perturbed_grad.clone();
        manifold.parallel_transport(&perturbed_point, point, &perturbed_grad, &mut transported_grad, workspace)?;
        
        // result = (transported_grad - gradient) / t
        let mut neg_gradient = gradient.clone();
        manifold.scale_tangent(point, -T::one(), gradient, &mut neg_gradient, workspace)?;
        manifold.add_tangents(point, &transported_grad, &neg_gradient, result, workspace)?;
        let result_copy = result.clone();
        manifold.scale_tangent(point, T::one() / t, &result_copy, result, workspace)?;
        
        Ok(())
    }
    
    /// Finds the intersection of the line s + tau*d with the trust region boundary.
    fn boundary_intersection<M>(
        s: &M::TangentVector,
        d: &M::TangentVector,
        radius: T,
        manifold: &M,
        point: &M::Point,
    ) -> Result<(T, T)>
    where
        M: Manifold<T>,
    {
        // Solve ||s + tau*d||^2 = radius^2
        let ss = manifold.inner_product(point, s, s)?;
        let sd = manifold.inner_product(point, s, d)?;
        let dd = manifold.inner_product(point, d, d)?;
        
        let discriminant = sd * sd - dd * (ss - radius * radius);
        
        if discriminant < T::zero() {
            return Err(riemannopt_core::error::ManifoldError::numerical_error(
                "No intersection with trust region boundary"
            ));
        }
        
        let sqrt_disc = <T as Float>::sqrt(discriminant);
        let tau1 = (-sd + sqrt_disc) / dd;
        let tau2 = (-sd - sqrt_disc) / dd;
        
        // Return the positive root
        if tau1 > T::zero() {
            Ok((tau1, tau2))
        } else {
            Ok((tau2, tau1))
        }
    }
    
    /// Solves the trust region subproblem using Steihaug-CG.
    fn solve_subproblem<M, C>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        gradient: &M::TangentVector,
        radius: T,
        cg_workspace: &mut CGWorkspace<T, M::TangentVector>,
        workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        let max_iter = self.config.max_cg_iterations.unwrap_or(manifold.dimension());
        
        // Reset CG vectors
        manifold.scale_tangent(point, T::zero(), gradient, &mut cg_workspace.s, workspace)?;
        
        // r = -gradient
        manifold.scale_tangent(point, -T::one(), gradient, &mut cg_workspace.r, workspace)?;
        // d = r  
        cg_workspace.d = cg_workspace.r.clone();
        
        let mut r_norm_sq = manifold.inner_product(point, &cg_workspace.r, &cg_workspace.r)?;
        
        // Check if gradient is already small
        if <T as Float>::sqrt(r_norm_sq) < self.config.cg_tolerance {
            cg_workspace.boundary_hit = false;
            return Ok(());
        }
        
        cg_workspace.boundary_hit = false;
        
        for _ in 0..max_iter {
            // Compute Hessian-vector product
            if self.config.use_exact_hessian {
                cg_workspace.hd = cost_fn.hessian_vector_product(point, &cg_workspace.d)?;
            } else {
                // Use finite differences for Hessian-vector product
                self.finite_diff_hessian_vec_product(
                    cost_fn, manifold, point, gradient, &cg_workspace.d, &mut cg_workspace.hd, workspace
                )?;
            }
            
            let dhd = manifold.inner_product(point, &cg_workspace.d, &cg_workspace.hd)?;
            
            // Check if we have negative curvature
            if dhd <= T::zero() {
                // Find tau such that ||s + tau*d|| = radius
                let (tau, _) = Self::boundary_intersection(
                    &cg_workspace.s, 
                    &cg_workspace.d, 
                    radius, 
                    manifold, 
                    point,
                )?;
                
                // s += tau * d
                let s_copy = cg_workspace.s.clone();
                manifold.axpy_tangent(point, tau, &cg_workspace.d, &s_copy, &mut cg_workspace.s, workspace)?;
                cg_workspace.boundary_hit = true;
                break;
            }
            
            let alpha = r_norm_sq / dhd;
            
            // Compute s_new = s + alpha * d using temp vector
            manifold.axpy_tangent(point, alpha, &cg_workspace.d, &cg_workspace.s, &mut cg_workspace.temp, workspace)?;
            
            // Check if we would exceed trust region
            let s_new_norm = <T as Float>::sqrt(
                manifold.inner_product(point, &cg_workspace.temp, &cg_workspace.temp)?
            );
            
            if s_new_norm >= radius {
                // Find tau such that ||s + tau*d|| = radius
                let (tau, _) = Self::boundary_intersection(
                    &cg_workspace.s, 
                    &cg_workspace.d, 
                    radius, 
                    manifold, 
                    point,
                )?;
                
                // s += tau * d
                let s_copy = cg_workspace.s.clone();
                manifold.axpy_tangent(point, tau, &cg_workspace.d, &s_copy, &mut cg_workspace.s, workspace)?;
                cg_workspace.boundary_hit = true;
                break;
            }
            
            // Update CG iteration
            cg_workspace.s = cg_workspace.temp.clone(); // s = s_new
            
            // r -= alpha * hd
            let mut scaled_hd = cg_workspace.hd.clone();
            manifold.scale_tangent(point, -alpha, &cg_workspace.hd, &mut scaled_hd, workspace)?;
            let r_copy = cg_workspace.r.clone();
            manifold.add_tangents(point, &r_copy, &scaled_hd, &mut cg_workspace.r, workspace)?;
            
            let r_norm_sq_new = manifold.inner_product(point, &cg_workspace.r, &cg_workspace.r)?;
            
            // Check convergence
            if <T as Float>::sqrt(r_norm_sq_new) < self.config.cg_tolerance {
                break;
            }
            
            let beta = r_norm_sq_new / r_norm_sq;
            // d = r + beta * d
            let d_copy = cg_workspace.d.clone();
            manifold.axpy_tangent(point, beta, &d_copy, &cg_workspace.r, &mut cg_workspace.d, workspace)?;
            r_norm_sq = r_norm_sq_new;
        }
        
        Ok(())
    }
    
    /// Computes the model value at a given step.
    fn model_value<M, C>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        value: T,
        gradient: &M::TangentVector,
        step: &M::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<T>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        let gs = manifold.inner_product(point, gradient, step)?;
        
        // Compute Hessian-vector product
        let mut hs = step.clone();
        if self.config.use_exact_hessian {
            hs = cost_fn.hessian_vector_product(point, step)?;
        } else {
            self.finite_diff_hessian_vec_product(cost_fn, manifold, point, gradient, step, &mut hs, workspace)?;
        }
        
        let shs = manifold.inner_product(point, step, &hs)?;
        
        Ok(value + gs + shs * <T as Scalar>::from_f64(0.5))
    }
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for TrustRegion<T> {
    fn name(&self) -> &str {
        "Riemannian Trust Region"
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
        let n = manifold.dimension();
        let mut workspace = Workspace::with_size(n);
        
        // Pre-allocate workspace buffers
        workspace.get_or_create_vector(BufferId::Gradient, n);
        workspace.get_or_create_vector(BufferId::Direction, n);
        workspace.get_or_create_vector(BufferId::Temp1, n);
        workspace.get_or_create_vector(BufferId::Temp2, n);
        
        // Additional buffers for CG solver
        for i in 0..5 {
            workspace.get_or_create_vector(BufferId::Custom(i as u32), n);
        }
        
        // Initialize state
        let initial_cost = cost_fn.cost(initial_point)?;
        let mut current_point = initial_point.clone();
        let mut previous_point: Option<M::Point> = None;
        let mut current_cost = initial_cost;
        let mut previous_cost: Option<T> = None;
        let mut gradient_norm: Option<T> = None;
        let mut iteration = 0;
        let mut function_evaluations = 1;
        let mut gradient_evaluations = 0;
        let mut trust_radius = self.config.initial_radius;
        
        // Compute initial gradient to get the right type for CG workspace
        let mut euclidean_grad = cost_fn.gradient(&initial_point)?;
        let mut riemannian_grad = euclidean_grad.clone();
        gradient_evaluations += 1;
        
        manifold.euclidean_to_riemannian_gradient(
            &initial_point,
            &euclidean_grad,
            &mut riemannian_grad,
            &mut workspace,
        )?;
        
        // Create zero vector for CG workspace initialization
        let mut zero_vector = riemannian_grad.clone();
        manifold.scale_tangent(&initial_point, T::zero(), &riemannian_grad, &mut zero_vector, &mut workspace)?;
        
        // Initialize CG workspace
        let mut cg_workspace = CGWorkspace::new(zero_vector);
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            let reason = self.check_stopping_criteria(
                manifold,
                iteration,
                function_evaluations,
                gradient_evaluations,
                start_time,
                current_cost,
                previous_cost,
                gradient_norm,
                &current_point,
                &previous_point,
                &mut workspace,
                stopping_criterion,
                trust_radius,
            );
            
            if let Some(reason) = reason {
                return Ok(OptimizationResult::new(
                    current_point,
                    current_cost,
                    iteration,
                    start_time.elapsed(),
                    reason,
                )
                .with_function_evaluations(function_evaluations)
                .with_gradient_evaluations(gradient_evaluations)
                .with_gradient_norm(gradient_norm.unwrap_or(T::zero())));
            }
            
            // Compute gradient at current point
            let _new_cost = cost_fn.cost_and_gradient(
                &current_point,
                &mut workspace,
                &mut euclidean_grad,
            )?;
            function_evaluations += 1;
            gradient_evaluations += 1;
            
            // Convert to Riemannian gradient
            manifold.euclidean_to_riemannian_gradient(
                &current_point,
                &euclidean_grad,
                &mut riemannian_grad,
                &mut workspace,
            )?;
            
            // Compute gradient norm
            let grad_norm_squared = manifold.inner_product(
                &current_point,
                &riemannian_grad,
                &riemannian_grad,
            )?;
            let grad_norm = <T as Float>::sqrt(grad_norm_squared);
            gradient_norm = Some(grad_norm);
            
            // Solve trust region subproblem
            self.solve_subproblem(
                cost_fn,
                manifold,
                &current_point,
                &riemannian_grad,
                trust_radius,
                &mut cg_workspace,
                &mut workspace,
            )?;
            
            // Get step from CG workspace
            let step = &cg_workspace.s;
            
            // Compute predicted reduction
            let model_current = current_cost;
            let model_step = self.model_value(
                cost_fn,
                manifold,
                &current_point,
                current_cost,
                &riemannian_grad,
                step,
                &mut workspace,
            )?;
            let predicted_reduction = model_current - model_step;
            
            // Compute trial point
            let mut trial_point = current_point.clone();
            manifold.retract(&current_point, step, &mut trial_point, &mut workspace)?;
            
            let trial_value = cost_fn.cost(&trial_point)?;
            function_evaluations += 1;
            
            // Compute actual reduction
            let actual_reduction = current_cost - trial_value;
            
            // Compute reduction ratio
            let ratio = if <T as Float>::abs(predicted_reduction) > T::epsilon() {
                actual_reduction / predicted_reduction
            } else {
                T::zero()
            };
            
            // Accept or reject the step
            if ratio >= self.config.acceptance_ratio {
                // Accept the step
                previous_point = Some(current_point.clone());
                current_point = trial_point;
                previous_cost = Some(current_cost);
                current_cost = trial_value;
                iteration += 1;
            }
            
            // Update trust region radius
            if ratio < self.config.decrease_threshold {
                // Poor agreement: shrink trust region
                trust_radius *= self.config.decrease_factor;
            } else if ratio > self.config.increase_threshold && cg_workspace.boundary_hit {
                // Good agreement and hit boundary: expand trust region
                trust_radius = <T as Float>::min(
                    trust_radius * self.config.increase_factor,
                    self.config.max_radius,
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    

    #[test]
    fn test_trust_region_config() {
        let config = TrustRegionConfig::<f64>::new()
            .with_initial_radius(0.5)
            .with_max_radius(5.0)
            .with_exact_hessian()
            .with_max_cg_iterations(20)
            .with_cg_tolerance(1e-8);
        
        assert_eq!(config.initial_radius, 0.5);
        assert_eq!(config.max_radius, 5.0);
        assert!(config.use_exact_hessian);
        assert_eq!(config.max_cg_iterations, Some(20));
        assert_eq!(config.cg_tolerance, 1e-8);
    }
    
    #[test]
    fn test_trust_region_builder() {
        let tr = TrustRegion::<f64>::with_default_config();
        assert_eq!(tr.name(), "Riemannian Trust Region");
    }
}