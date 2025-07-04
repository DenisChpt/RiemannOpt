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
        optimizer::{Optimizer, OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
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

/// Internal state for Trust Region optimizer.
#[derive(Debug)]
struct TrustRegionInternalState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    workspace: Workspace<T>,
    iteration: usize,
    
    /// Current trust region radius
    radius: T,
    /// Previous model value (kept for potential future use)
    #[allow(dead_code)]
    model_value: Option<T>,
    /// Number of rejected steps in a row
    consecutive_rejections: usize,
    /// CG workspace vectors for zero-allocation
    cg_s: TV,      // Solution vector
    cg_r: TV,      // Residual vector  
    cg_d: TV,      // Search direction
    cg_hd: TV,     // Hessian-vector product
    cg_temp: TV,   // Temporary vector
    boundary_hit: bool,
}

impl<T, TV> TrustRegionInternalState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// Creates a new trust region state.
    fn new(n: usize, initial_radius: T, zero_vector: TV) -> Self {
        let mut workspace = Workspace::with_size(n);
        
        // Pre-allocate workspace buffers
        workspace.get_or_create_vector(BufferId::Gradient, n);
        workspace.get_or_create_vector(BufferId::Direction, n);
        workspace.get_or_create_vector(BufferId::Temp1, n);
        workspace.get_or_create_vector(BufferId::Temp2, n);
        
        Self {
            workspace,
            iteration: 0,
            radius: initial_radius,
            model_value: None,
            consecutive_rejections: 0,
            cg_s: zero_vector.clone(),
            cg_r: zero_vector.clone(),
            cg_d: zero_vector.clone(),
            cg_hd: zero_vector.clone(),
            cg_temp: zero_vector,
            boundary_hit: false,
        }
    }

    /// Updates the trust region radius based on the reduction ratio.
    fn update_radius(&mut self, ratio: T, config: &TrustRegionConfig<T>) {
        if ratio < config.decrease_threshold {
            // Poor agreement: shrink trust region
            self.radius *= config.decrease_factor;
            self.consecutive_rejections += 1;
        } else if ratio > config.increase_threshold {
            // Good agreement: expand trust region
            self.radius = <T as Float>::min(
                self.radius * config.increase_factor,
                config.max_radius,
            );
            self.consecutive_rejections = 0;
        } else {
            // Moderate agreement: keep radius unchanged
            self.consecutive_rejections = 0;
        }
    }
    
    fn update_iteration(&mut self) {
        self.iteration += 1;
    }
}

/// Steihaug-CG solver for the trust region subproblem.
///
/// Solves the subproblem:
/// min_s  m(s) = f + <g, s> + 0.5 <s, H s>
/// s.t.   ||s|| <= Delta
///
/// where s is in the tangent space.
struct SteiahugCG;

impl SteiahugCG {
    /// Solves the trust region subproblem using truncated conjugate gradient.
    ///
    /// The solution is written to the internal state vectors.
    fn solve_with_internal_state<T, C, M>(
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        gradient: &M::TangentVector,
        radius: T,
        config: &TrustRegionConfig<T>,
        internal_state: &mut TrustRegionInternalState<T, M::TangentVector>,
    ) -> Result<()>
    where
        T: Scalar,
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {

        let max_iter = config.max_cg_iterations.unwrap_or(manifold.dimension());
        
        let workspace = &mut internal_state.workspace;
        
        // Reset CG vectors
        internal_state.cg_s = gradient.clone();
        manifold.scale_tangent(point, T::zero(), gradient, &mut internal_state.cg_s, workspace)?;
        
        // r = -gradient
        manifold.scale_tangent(point, -T::one(), gradient, &mut internal_state.cg_r, workspace)?;
        // d = r  
        internal_state.cg_d = internal_state.cg_r.clone();
        
        let mut r_norm_sq = manifold.inner_product(point, &internal_state.cg_r, &internal_state.cg_r)?;
        
        // Check if gradient is already small
        if <T as Float>::sqrt(r_norm_sq) < config.cg_tolerance {
            internal_state.boundary_hit = false;
            return Ok(());
        }
        
        internal_state.boundary_hit = false;
        
        for _ in 0..max_iter {
            // Compute Hessian-vector product
            if config.use_exact_hessian {
                internal_state.cg_hd = cost_fn.hessian_vector_product(point, &internal_state.cg_d)?;
            } else {
                // Use finite differences for Hessian-vector product
                Self::finite_diff_hessian_vec_product::<T, C, M>(
                    cost_fn, manifold, point, gradient, &internal_state.cg_d, &mut internal_state.cg_hd, workspace
                )?;
            }
            
            let dhd = manifold.inner_product(point, &internal_state.cg_d, &internal_state.cg_hd)?;
            
            // Check if we have negative curvature
            if dhd <= T::zero() {
                // Find tau such that ||s + tau*d|| = radius
                let (tau, _) = Self::boundary_intersection::<T, M>(
                    &internal_state.cg_s, 
                    &internal_state.cg_d, 
                    radius, 
                    manifold, 
                    point,
                    workspace
                )?;
                
                // s += tau * d
                let s_copy = internal_state.cg_s.clone();
                manifold.axpy_tangent(point, tau, &internal_state.cg_d, &s_copy, &mut internal_state.cg_s, workspace)?;
                internal_state.boundary_hit = true;
                break;
            }
            
            let alpha = r_norm_sq / dhd;
            
            // Compute s_new = s + alpha * d using temp vector
            manifold.axpy_tangent(point, alpha, &internal_state.cg_d, &internal_state.cg_s, &mut internal_state.cg_temp, workspace)?;
            
            // Check if we would exceed trust region
            let s_new_norm = <T as Float>::sqrt(
                manifold.inner_product(point, &internal_state.cg_temp, &internal_state.cg_temp)?
            );
            
            if s_new_norm >= radius {
                // Find tau such that ||s + tau*d|| = radius
                let (tau, _) = Self::boundary_intersection::<T, M>(
                    &internal_state.cg_s, 
                    &internal_state.cg_d, 
                    radius, 
                    manifold, 
                    point,
                    workspace
                )?;
                
                // s += tau * d
                let s_copy = internal_state.cg_s.clone();
                manifold.axpy_tangent(point, tau, &internal_state.cg_d, &s_copy, &mut internal_state.cg_s, workspace)?;
                internal_state.boundary_hit = true;
                break;
            }
            
            // Update CG iteration
            internal_state.cg_s = internal_state.cg_temp.clone(); // s = s_new
            
            // r -= alpha * hd
            let mut scaled_hd = internal_state.cg_hd.clone();
            manifold.scale_tangent(point, -alpha, &internal_state.cg_hd, &mut scaled_hd, workspace)?;
            let r_copy = internal_state.cg_r.clone();
            manifold.add_tangents(point, &r_copy, &scaled_hd, &mut internal_state.cg_r, workspace)?;
            
            let r_norm_sq_new = manifold.inner_product(point, &internal_state.cg_r, &internal_state.cg_r)?;
            
            // Check convergence
            if <T as Float>::sqrt(r_norm_sq_new) < config.cg_tolerance {
                break;
            }
            
            let beta = r_norm_sq_new / r_norm_sq;
            // d = r + beta * d
            let d_copy = internal_state.cg_d.clone();
            manifold.axpy_tangent(point, beta, &d_copy, &internal_state.cg_r, &mut internal_state.cg_d, workspace)?;
            r_norm_sq = r_norm_sq_new;
        }
        
        Ok(())
    }
    
    /// Computes Hessian-vector product using finite differences.
    fn finite_diff_hessian_vec_product<T, C, M>(
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        gradient: &M::TangentVector,
        direction: &M::TangentVector,
        result: &mut M::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        T: Scalar,
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
        let mut perturbed_grad = gradient.clone();
        
        let _cost = cost_fn.cost_and_gradient(&perturbed_point, workspace, &mut perturbed_euclidean_grad)?;
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
    fn boundary_intersection<T, M>(
        s: &M::TangentVector,
        d: &M::TangentVector,
        radius: T,
        manifold: &M,
        point: &M::Point,
        _workspace: &mut Workspace<T>,
    ) -> Result<(T, T)>
    where
        T: Scalar,
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
}

/// Trust Region optimizer for Riemannian manifolds.
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

    /// Returns the configuration.
    pub fn config(&self) -> &TrustRegionConfig<T> {
        &self.config
    }
    
    /// Returns the optimizer name.
    pub fn name(&self) -> &str {
        "Riemannian Trust Region"
    }

    /// Computes the model value at a given step.
    fn model_value<C, M>(
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
            SteiahugCG::finite_diff_hessian_vec_product::<T, C, M>(cost_fn, manifold, point, gradient, step, &mut hs, workspace)?;
        }
        
        let shs = manifold.inner_product(point, step, &hs)?;
        
        Ok(value + gs + shs * <T as Scalar>::from_f64(0.5))
    }

    /// Performs a single optimization step.
    fn step_with_state<C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, M::Point, M::TangentVector>,
        internal_state: &mut TrustRegionInternalState<T, M::TangentVector>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        // Create gradient buffer
        let mut euclidean_grad = match &state.gradient {
            Some(g) => g.clone(),
            None => {
                // For the first iteration, compute the gradient
                cost_fn.gradient(&state.point)?
            }
        };
        let mut riemannian_grad = euclidean_grad.clone();
        let mut new_point = state.point.clone();
        
        // Compute gradient
        {
            let workspace = &mut internal_state.workspace;
            let _cost = cost_fn.cost_and_gradient(&state.point, workspace, &mut euclidean_grad)?;
            state.function_evaluations += 1;
            state.gradient_evaluations += 1;
            
            // Convert to Riemannian gradient
            manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad, &mut riemannian_grad, workspace)?;
        }
        
        // Compute gradient norm
        let grad_norm_squared = manifold.inner_product(&state.point, &riemannian_grad, &riemannian_grad)?;
        let grad_norm = <T as Float>::sqrt(grad_norm_squared);
        state.gradient_norm = Some(grad_norm);
        
        // Store gradient in state
        state.gradient = Some(riemannian_grad.clone());
        
        // Check if radius is too small
        if internal_state.radius < self.config.min_radius {
            return Err(riemannopt_core::error::ManifoldError::numerical_error(
                "Trust region radius below minimum threshold"
            ));
        }
        
        // Solve trust region subproblem
        SteiahugCG::solve_with_internal_state::<T, C, M>(
            cost_fn,
            manifold,
            &state.point,
            &riemannian_grad,
            internal_state.radius,
            &self.config,
            internal_state,
        )?;
        
        // Get step from internal state
        let step = &internal_state.cg_s;
        let _boundary_hit = internal_state.boundary_hit;
        
        // Compute predicted reduction
        let model_current = state.value;
        let model_step = {
            let workspace = &mut internal_state.workspace;
            self.model_value(
                cost_fn,
                manifold,
                &state.point,
                state.value,
                &riemannian_grad,
                step,
                workspace,
            )?
        };
        let predicted_reduction = model_current - model_step;
        
        // Compute trial point
        {
            let workspace = &mut internal_state.workspace;
            manifold.retract(&state.point, step, &mut new_point, workspace)?;
        }
        let trial_value = cost_fn.cost(&new_point)?;
        state.function_evaluations += 1;
        
        // Compute actual reduction
        let actual_reduction = state.value - trial_value;
        
        // Compute reduction ratio
        let ratio = if <T as Float>::abs(predicted_reduction) > T::epsilon() {
            actual_reduction / predicted_reduction
        } else {
            T::zero()
        };
        
        // Accept or reject the step
        if ratio >= self.config.acceptance_ratio {
            // Accept the step
            state.point = new_point;
            state.value = trial_value;
            state.iteration += 1;
            internal_state.consecutive_rejections = 0;
        } else {
            // Reject the step
            internal_state.consecutive_rejections += 1;
        }
        
        // Update trust region radius
        internal_state.update_radius(ratio, &self.config);
        
        // Update internal state iteration
        internal_state.update_iteration();
        
        Ok(())
    }

    /// Optimizes the given cost function on the manifold.
    pub fn optimize<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &M::Point,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, M::Point>>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        let start_time = Instant::now();
        
        // Initialize optimization state
        let initial_cost = cost_fn.cost(initial_point)?;
        let mut state = OptimizerState::new(
            initial_point.clone(),
            initial_cost
        );
        
        // Create internal state
        let n = manifold.dimension();
        let zero_vector = {
            // Create a zero tangent vector by getting the gradient and scaling it to zero
            let mut workspace_temp = Workspace::with_size(n);
            let euclidean_grad = cost_fn.gradient(initial_point)?;
            let mut temp = euclidean_grad.clone();
            
            // Convert to Riemannian gradient
            manifold.euclidean_to_riemannian_gradient(initial_point, &euclidean_grad, &mut temp, &mut workspace_temp)?;
            
            // Scale it to zero to get a proper zero tangent vector
            let mut zero_tangent = temp.clone();
            manifold.scale_tangent(initial_point, T::zero(), &temp, &mut zero_tangent, &mut workspace_temp)?;
            zero_tangent
        };
        
        let mut internal_state = TrustRegionInternalState::<T, M::TangentVector>::new(
            n, 
            self.config.initial_radius,
            zero_vector
        );
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = ConvergenceChecker::check(&state, manifold, stopping_criterion)? {
                return Ok(OptimizationResult::new(
                    state.point.clone(),
                    state.value,
                    state.iteration,
                    start_time.elapsed(),
                    reason,
                )
                .with_function_evaluations(state.function_evaluations)
                .with_gradient_evaluations(state.gradient_evaluations)
                .with_gradient_norm(state.gradient_norm.unwrap_or(T::zero())));
            }
            
            // Check if trust region is too small
            if internal_state.radius < self.config.min_radius {
                return Ok(OptimizationResult::new(
                    state.point.clone(),
                    state.value,
                    state.iteration,
                    start_time.elapsed(),
                    riemannopt_core::optimization::optimizer::TerminationReason::Converged,
                )
                .with_function_evaluations(state.function_evaluations)
                .with_gradient_evaluations(state.gradient_evaluations)
                .with_gradient_norm(state.gradient_norm.unwrap_or(T::zero())));
            }
            
            // Perform one optimization step
            self.step_with_state(cost_fn, manifold, &mut state, &mut internal_state)?;
        }
    }

    /// Performs a single optimization step.
    pub fn step<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        // Create temporary internal state
        let n = manifold.dimension();
        let zero_vector = {
            // Create a zero tangent vector by getting the gradient and scaling it to zero
            let mut workspace_temp = Workspace::with_size(n);
            let euclidean_grad = cost_fn.gradient(&state.point)?;
            let mut temp = euclidean_grad.clone();
            
            // Convert to Riemannian gradient
            manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad, &mut temp, &mut workspace_temp)?;
            
            // Scale it to zero to get a proper zero tangent vector
            let mut zero_tangent = temp.clone();
            manifold.scale_tangent(&state.point, T::zero(), &temp, &mut zero_tangent, &mut workspace_temp)?;
            zero_tangent
        };
        
        let mut internal_state = TrustRegionInternalState::<T, M::TangentVector>::new(
            n,
            self.config.initial_radius,
            zero_vector
        );
        
        // Delegate to internal implementation
        self.step_with_state(cost_fn, manifold, state, &mut internal_state)
    }
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for TrustRegion<T> {
    fn name(&self) -> &str {
        "Riemannian Trust Region"
    }
    
    fn optimize<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &M::Point,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, M::Point>>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        self.optimize(cost_fn, manifold, initial_point, stopping_criterion)
    }
    
    fn step<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        self.step(cost_fn, manifold, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riemannopt_core::{
        core::cost_function::CostFunction,
        error::Result,
        memory::workspace::Workspace,
        optimization::optimizer::StoppingCriterion,
        types::{DVector, Scalar},
    };
    use riemannopt_manifolds::Sphere;
    use std::fmt::Debug;
    
    /// Simple quadratic cost function for testing
    #[derive(Debug)]
    struct QuadraticCost<T: Scalar> {
        matrix: DVector<T>,
    }
    
    impl<T: Scalar> QuadraticCost<T> {
        fn new(dim: usize) -> Self {
            // Create a diagonal matrix with eigenvalues 1, 2, ..., dim
            let mut matrix = DVector::zeros(dim);
            for i in 0..dim {
                matrix[i] = <T as Scalar>::from_f64((i + 1) as f64);
            }
            Self { matrix }
        }
    }
    
    impl<T: Scalar> CostFunction<T> for QuadraticCost<T> {
        type Point = DVector<T>;
        type TangentVector = DVector<T>;
    
        fn cost(&self, x: &Self::Point) -> Result<T> {
            // f(x) = sum_i matrix[i] * x[i]^2
            let mut cost = T::zero();
            for i in 0..x.len() {
                cost = cost + self.matrix[i] * x[i] * x[i];
            }
            Ok(cost)
        }
    
        fn cost_and_gradient(
            &self,
            x: &Self::Point,
            _workspace: &mut Workspace<T>,
            gradient: &mut Self::TangentVector,
        ) -> Result<T> {
            // grad f(x) = 2 * diag(matrix) * x
            let mut cost = T::zero();
            for i in 0..x.len() {
                cost = cost + self.matrix[i] * x[i] * x[i];
                gradient[i] = <T as Scalar>::from_f64(2.0) * self.matrix[i] * x[i];
            }
            Ok(cost)
        }
        
        fn hessian_vector_product(
            &self,
            _x: &Self::Point,
            v: &Self::TangentVector,
        ) -> Result<Self::TangentVector> {
            // Hv = 2 * diag(matrix) * v
            let mut result = v.clone();
            for i in 0..v.len() {
                result[i] = <T as Scalar>::from_f64(2.0) * self.matrix[i] * v[i];
            }
            Ok(result)
        }
        
        fn gradient_fd_alloc(&self, x: &Self::Point) -> Result<Self::TangentVector> {
            // Fallback to gradient computation
            let mut gradient = DVector::zeros(x.len());
            self.cost_and_gradient(x, &mut Workspace::new(), &mut gradient)?;
            Ok(gradient)
        }
    }

    #[test]
    fn test_trust_region_config() {
        let config = TrustRegionConfig::<f64>::new()
            .with_initial_radius(0.5)
            .with_max_radius(5.0)
            .with_exact_hessian();
        
        assert_eq!(config.initial_radius, 0.5);
        assert_eq!(config.max_radius, 5.0);
        assert!(config.use_exact_hessian);
    }

    #[test]
    fn test_trust_region_state_update() {
        let config = TrustRegionConfig::<f64>::default();
        let zero_vector = DVector::zeros(2);
        let mut state = TrustRegionInternalState::<f64, DVector<f64>>::new(2, 1.0, zero_vector);
        
        // Test radius decrease
        state.update_radius(0.1, &config);
        assert_eq!(state.radius, 0.25);
        assert_eq!(state.consecutive_rejections, 1);
        
        // Test radius increase
        state.radius = 1.0;
        state.update_radius(0.9, &config);
        assert_eq!(state.radius, 2.0);
        assert_eq!(state.consecutive_rejections, 0);
        
        // Test radius unchanged
        state.radius = 1.0;
        state.update_radius(0.5, &config);
        assert_eq!(state.radius, 1.0);
        assert_eq!(state.consecutive_rejections, 0);
    }

    #[test]
    fn test_trust_region_on_sphere() -> Result<()> {
        let dim = 5;
        let sphere = Sphere::<f64>::new(dim)?;
        let cost_fn = QuadraticCost::new(dim);
        
        // Initial point on sphere - start from a non-critical point
        let mut initial_point = DVector::zeros(dim);
        initial_point[0] = 0.5;
        initial_point[dim-1] = 0.5;
        initial_point /= initial_point.norm(); // Normalize to be on sphere
        
        let config = TrustRegionConfig::new()
            .with_initial_radius(0.5)
            .with_max_cg_iterations(10);
        
        let mut optimizer = TrustRegion::new(config);
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        let result = optimizer.optimize(
            &cost_fn,
            &sphere,
            &initial_point,
            &stopping_criterion,
        )?;
        
        // Check basic convergence properties
        assert!(result.value.is_finite());
        assert!(result.gradient_norm.unwrap_or(0.0) < 1e-3);
        assert!(result.iterations < 100);
        
        Ok(())
    }

    #[test]
    fn test_basic_trust_region_setup() {
        // Note: SteiahugCG::solve was removed, this test checks basic setup
        let config = TrustRegionConfig::<f64>::new()
            .with_cg_tolerance(1e-10)
            .with_initial_radius(0.5);
        
        assert_eq!(config.cg_tolerance, 1e-10);
        assert_eq!(config.initial_radius, 0.5);
        
        let optimizer = TrustRegion::<f64>::new(config);
        assert_eq!(optimizer.name(), "Riemannian Trust Region");
    }


    #[test]
    fn test_trust_region_with_custom_config() -> Result<()> {
        let dim = 5;
        let sphere = Sphere::<f64>::new(dim)?;
        let cost_fn = QuadraticCost::new(dim);
        
        // Initial point on sphere
        let mut initial_point = DVector::zeros(dim);
        initial_point[0] = 1.0;
        
        let config = TrustRegionConfig::new()
            .with_initial_radius(0.1)
            .with_max_cg_iterations(5)
            .with_cg_tolerance(1e-4);
        
        let mut optimizer = TrustRegion::new(config);
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(20)
            .with_gradient_tolerance(1e-3);
        
        let result = optimizer.optimize(
            &cost_fn,
            &sphere,
            &initial_point,
            &stopping_criterion,
        )?;
        
        // Basic sanity checks
        assert!(result.value.is_finite());
        assert!(result.gradient_norm.unwrap_or(0.0).is_finite());
        
        Ok(())
    }
}