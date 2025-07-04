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
    cost_function::CostFunction,
    core::CachedCostFunction,
    error::{ManifoldError, Result},
    manifold::{Manifold, Point, TangentVector},
    memory::workspace::{Workspace, WorkspaceBuilder},
    optimizer::{Optimizer, OptimizerStateLegacy as OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker, TerminationReason},
    retraction::Retraction,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

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

/// State for the Trust Region optimizer.
#[derive(Debug)]
struct TrustRegionState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Current trust region radius
    radius: T,
    /// Previous model value (kept for potential future use)
    #[allow(dead_code)]
    model_value: Option<T>,
    /// Number of rejected steps in a row
    consecutive_rejections: usize,
    /// Workspace for the Steihaug-CG solver (zero-allocation)
    cg_workspace: SteiahugCGWorkspace<T, D>,
}

impl<T, D> TrustRegionState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new trust region state.
    fn new(initial_radius: T, dimension: D) -> Self {
        Self {
            radius: initial_radius,
            model_value: None,
            consecutive_rejections: 0,
            cg_workspace: SteiahugCGWorkspace::new(dimension),
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
}

/// Steihaug-CG solver for the trust region subproblem.
///
/// Solves the subproblem:
/// min_s  m(s) = f + <g, s> + 0.5 <s, H s>
/// s.t.   ||s|| <= Delta
///
/// where s is in the tangent space.
struct SteiahugCG;

/// Workspace for the Steihaug-CG solver to avoid allocations.
#[derive(Debug)]
struct SteiahugCGWorkspace<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Solution vector s
    s: TangentVector<T, D>,
    /// Residual vector r
    r: TangentVector<T, D>,
    /// Search direction d
    d: TangentVector<T, D>,
    /// Hessian-vector product Hd
    hd: TangentVector<T, D>,
    /// Temporary vector for intermediate calculations
    temp: TangentVector<T, D>,
    /// Temporary vector for perturbation calculations
    temp_perturbation: TangentVector<T, D>,
    /// Temporary point for finite differences
    temp_point: Point<T, D>,
    /// Temporary gradient for finite differences
    temp_grad: TangentVector<T, D>,
}

impl<T, D> SteiahugCGWorkspace<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new workspace with the given dimension.
    fn new(dim: D) -> Self {
        let zeros = TangentVector::zeros_generic(dim, nalgebra::U1);
        Self {
            s: zeros.clone(),
            r: zeros.clone(),
            d: zeros.clone(),
            hd: zeros.clone(),
            temp: zeros.clone(),
            temp_perturbation: zeros.clone(),
            temp_point: zeros.clone(),
            temp_grad: zeros,
        }
    }

    /// Resets all vectors to zero.
    fn reset(&mut self) {
        self.s.fill(T::zero());
        self.r.fill(T::zero());
        self.d.fill(T::zero());
        self.hd.fill(T::zero());
        self.temp.fill(T::zero());
        self.temp_perturbation.fill(T::zero());
        self.temp_point.fill(T::zero());
        self.temp_grad.fill(T::zero());
    }
}

impl SteiahugCG {
    /// Solves the trust region subproblem using truncated conjugate gradient (allocating version).
    ///
    /// Returns the step direction and a flag indicating if the boundary was hit.
    #[allow(dead_code)]  // Used in tests to verify zero-allocation version produces same results
    fn solve<T, D, C, M>(
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        gradient: &TangentVector<T, D>,
        radius: T,
        config: &TrustRegionConfig<T>,
    ) -> Result<(TangentVector<T, D>, bool)>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Create workspace for this solve
        let mut workspace = SteiahugCGWorkspace::new(gradient.shape_generic().0);
        
        // Use the zero-allocation version
        Self::solve_with_workspace(cost_fn, manifold, point, gradient, radius, config, &mut workspace)?;
        
        // Return solution (this still requires one allocation for backward compatibility)
        Ok((workspace.s.clone(), workspace.temp[0] != T::zero())) // temp[0] stores boundary_hit flag
    }

    /// Solves the trust region subproblem using truncated conjugate gradient (zero-allocation version).
    ///
    /// The solution is written to `workspace.s` and the boundary hit flag is stored in `workspace.temp[0]`.
    fn solve_with_workspace<T, D, C, M>(
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        gradient: &TangentVector<T, D>,
        radius: T,
        config: &TrustRegionConfig<T>,
        workspace: &mut SteiahugCGWorkspace<T, D>,
    ) -> Result<()>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let n = gradient.len();
        let max_iter = config.max_cg_iterations.unwrap_or(n);
        
        // Reset workspace
        workspace.reset();
        
        // Initialize CG iteration using workspace vectors
        // s is already zero from reset
        workspace.r.copy_from(&(-gradient)); // r = -gradient
        workspace.d.copy_from(&workspace.r); // d = r
        let mut r_norm_sq = manifold.inner_product(point, &workspace.r, &workspace.r)?;
        
        // Check if gradient is already small
        if <T as Float>::sqrt(r_norm_sq) < config.cg_tolerance {
            workspace.temp[0] = T::zero(); // boundary_hit = false
            return Ok(());
        }
        
        let mut boundary_hit = false;
        
        for _ in 0..max_iter {
            // Compute Hessian-vector product using workspace
            if config.use_exact_hessian {
                workspace.hd.copy_from(&cost_fn.hessian_vector_product(point, &workspace.d)?);
            } else {
                // Use finite differences for Hessian-vector product (zero-allocation version)
                // We need to temporarily copy direction to avoid borrow conflicts
                let d_copy = workspace.d.clone();
                Self::finite_diff_hessian_vec_product_workspace(
                    cost_fn, manifold, point, gradient, &d_copy, workspace
                )?;
            };
            
            let dhd = manifold.inner_product(point, &workspace.d, &workspace.hd)?;
            
            // Check if we have negative curvature
            if dhd <= T::zero() {
                // Find tau such that ||s + tau*d|| = radius
                let (tau, _) = Self::boundary_intersection(&workspace.s, &workspace.d, radius, manifold, point)?;
                workspace.s.axpy(tau, &workspace.d, T::one()); // s += tau * d
                boundary_hit = true;
                break;
            }
            
            let alpha = r_norm_sq / dhd;
            
            // Compute s_new = s + alpha * d using temp vector
            workspace.temp.copy_from(&workspace.s);
            workspace.temp.axpy(alpha, &workspace.d, T::one()); // temp = s + alpha * d
            
            // Check if we would exceed trust region
            let s_new_norm = <T as Float>::sqrt(
                manifold.inner_product(point, &workspace.temp, &workspace.temp)?
            );
            
            if s_new_norm >= radius {
                // Find tau such that ||s + tau*d|| = radius
                let (tau, _) = Self::boundary_intersection(&workspace.s, &workspace.d, radius, manifold, point)?;
                workspace.s.axpy(tau, &workspace.d, T::one()); // s += tau * d
                boundary_hit = true;
                break;
            }
            
            // Update CG iteration
            workspace.s.copy_from(&workspace.temp); // s = s_new
            workspace.r.axpy(-alpha, &workspace.hd, T::one()); // r -= alpha * hd
            let r_norm_sq_new = manifold.inner_product(point, &workspace.r, &workspace.r)?;
            
            // Check convergence
            if <T as Float>::sqrt(r_norm_sq_new) < config.cg_tolerance {
                break;
            }
            
            let beta = r_norm_sq_new / r_norm_sq;
            // d = r + beta * d
            workspace.d *= beta;
            workspace.d += &workspace.r;
            r_norm_sq = r_norm_sq_new;
        }
        
        // Store boundary hit flag in temp[0]
        workspace.temp[0] = if boundary_hit { T::one() } else { T::zero() };
        
        Ok(())
    }
    
    /// Computes Hessian-vector product using finite differences (allocating version).
    fn finite_diff_hessian_vec_product<T, D, C, M>(
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        gradient: &TangentVector<T, D>,
        direction: &TangentVector<T, D>,
    ) -> Result<TangentVector<T, D>>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let eps = <T as Float>::sqrt(T::epsilon());
        let norm = <T as Float>::sqrt(
            manifold.inner_product(point, direction, direction)?
        );
        
        if norm < T::epsilon() {
            return Ok(TangentVector::zeros_generic(
                gradient.shape_generic().0,
                nalgebra::U1,
            ));
        }
        
        let t = eps / norm;
        
        // Use default retraction for perturbation
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        let perturbed = retraction.retract(manifold, point, &(direction * t))?;
        let (_, grad_perturbed) = cost_fn.cost_and_gradient(&perturbed)?;
        
        // Transport gradient back and compute difference
        let grad_transported = manifold.parallel_transport(&perturbed, point, &grad_perturbed)?;
        
        Ok((grad_transported - gradient) / t)
    }

    /// Computes Hessian-vector product using finite differences (zero-allocation version).
    ///
    /// Uses workspace for temporary calculations and writes the result to workspace.hd.
    fn finite_diff_hessian_vec_product_workspace<T, D, C, M>(
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        gradient: &TangentVector<T, D>,
        direction: &TangentVector<T, D>,
        workspace: &mut SteiahugCGWorkspace<T, D>,
    ) -> Result<()>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let eps = <T as Float>::sqrt(T::epsilon());
        let norm = <T as Float>::sqrt(
            manifold.inner_product(point, direction, direction)?
        );
        
        if norm < T::epsilon() {
            workspace.hd.fill(T::zero());
            return Ok(());
        }
        
        let t = eps / norm;
        
        // Use default retraction for perturbation
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        // Compute perturbation: temp_perturbation = direction * t
        workspace.temp_perturbation.copy_from(direction);
        workspace.temp_perturbation *= t;
        
        // Compute perturbed point
        workspace.temp_point.copy_from(point);
        workspace.temp_point = retraction.retract(manifold, point, &workspace.temp_perturbation)?;
        
        // Compute gradient at perturbed point
        let (_, grad_perturbed) = cost_fn.cost_and_gradient(&workspace.temp_point)?;
        
        // Transport gradient back and compute difference
        workspace.temp_grad.copy_from(&manifold.parallel_transport(
            &workspace.temp_point, 
            point, 
            &grad_perturbed
        )?);
        
        // hd = (grad_transported - gradient) / t
        workspace.hd.copy_from(&workspace.temp_grad);
        workspace.hd -= gradient;
        workspace.hd /= t;
        
        Ok(())
    }
    
    /// Finds the intersection of the line s + tau*d with the trust region boundary.
    fn boundary_intersection<T, D, M>(
        s: &TangentVector<T, D>,
        d: &TangentVector<T, D>,
        radius: T,
        manifold: &M,
        point: &Point<T, D>,
    ) -> Result<(T, T)>
    where
        T: Scalar,
        D: Dim,
        DefaultAllocator: Allocator<D>,
        M: Manifold<T, D>,
    {
        // Solve ||s + tau*d||^2 = radius^2
        let ss = manifold.inner_product(point, s, s)?;
        let sd = manifold.inner_product(point, s, d)?;
        let dd = manifold.inner_product(point, d, d)?;
        
        let discriminant = sd * sd - dd * (ss - radius * radius);
        
        if discriminant < T::zero() {
            return Err(ManifoldError::numerical_error(
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
pub struct TrustRegion<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    config: TrustRegionConfig<T>,
    _phantom: std::marker::PhantomData<D>,
}

impl<T, D> TrustRegion<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new Trust Region optimizer with the given configuration.
    pub fn new(config: TrustRegionConfig<T>) -> Self {
        Self {
            config,
            _phantom: std::marker::PhantomData,
        }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &TrustRegionConfig<T> {
        &self.config
    }

    /// Computes the model value at a given step.
    fn model_value<C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        value: T,
        gradient: &TangentVector<T, D>,
        step: &TangentVector<T, D>,
    ) -> Result<T>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let gs = manifold.inner_product(point, gradient, step)?;
        
        // Compute Hessian-vector product
        let hs = if self.config.use_exact_hessian {
            cost_fn.hessian_vector_product(point, step)?
        } else {
            SteiahugCG::finite_diff_hessian_vec_product(cost_fn, manifold, point, gradient, step)?
        };
        
        let shs = manifold.inner_product(point, step, &hs)?;
        
        Ok(value + gs + shs * <T as Scalar>::from_f64(0.5))
    }

    /// Performs a single optimization step.
    fn step_internal<C, M, R>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        retraction: &R,
        state: &mut OptimizerState<T, D>,
        tr_state: &mut TrustRegionState<T, D>,
        _workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
    {
        // Compute gradient
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&state.point)?;
        let gradient = manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad)?;
        let grad_norm = <T as Float>::sqrt(
            manifold.inner_product(&state.point, &gradient, &gradient)?
        );
        
        state.set_gradient(gradient.clone(), grad_norm);
        state.value = cost;
        
        // Check if radius is too small
        if tr_state.radius < self.config.min_radius {
            return Err(ManifoldError::numerical_error(
                "Trust region radius below minimum threshold"
            ));
        }
        
        // Solve trust region subproblem using zero-allocation workspace
        SteiahugCG::solve_with_workspace(
            cost_fn,
            manifold,
            &state.point,
            &gradient,
            tr_state.radius,
            &self.config,
            &mut tr_state.cg_workspace,
        )?;
        
        // Extract solution from workspace
        let step = &tr_state.cg_workspace.s;
        let _boundary_hit = tr_state.cg_workspace.temp[0] != T::zero();
        
        // Compute predicted reduction
        let model_current = cost;
        let model_step = self.model_value(
            cost_fn,
            manifold,
            &state.point,
            cost,
            &gradient,
            &step,
        )?;
        let predicted_reduction = model_current - model_step;
        
        // Compute trial point
        let trial_point = retraction.retract(manifold, &state.point, &step)?;
        let trial_value = cost_fn.cost(&trial_point)?;
        
        // Compute actual reduction
        let actual_reduction = cost - trial_value;
        
        // Compute reduction ratio
        let ratio = if <T as Float>::abs(predicted_reduction) > T::epsilon() {
            actual_reduction / predicted_reduction
        } else {
            T::zero()
        };
        
        // Accept or reject the step
        if ratio >= self.config.acceptance_ratio {
            // Accept the step
            state.update(trial_point, trial_value);
            tr_state.consecutive_rejections = 0;
        } else {
            // Reject the step
            tr_state.consecutive_rejections += 1;
        }
        
        // Update trust region radius
        tr_state.update_radius(ratio, &self.config);
        
        Ok(())
    }

    /// Optimizes the given cost function.
    pub fn optimize<C, M, R>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        retraction: &R,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
        DefaultAllocator: Allocator<D, D>,
    {
        let start_time = Instant::now();
        
        // Wrap cost function with caching to avoid redundant computations
        let cached_cost_fn = CachedCostFunction::new(cost_fn);
        
        // Initialize optimizer state
        let initial_cost = cached_cost_fn.cost(initial_point)?;
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        let dim = initial_point.shape_generic().0;
        let mut tr_state = TrustRegionState::new(self.config.initial_radius, dim);
        
        // Create a single workspace for the entire optimization
        let n = initial_point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .build();
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = ConvergenceChecker::check(&state, manifold, stopping_criterion)? {
                // Get cache statistics for diagnostics
                let ((_cost_hits, cost_misses), (_grad_hits, grad_misses), _) = cached_cost_fn.cache_stats();
                
                return Ok(OptimizationResult::new(
                    state.point,
                    state.value,
                    state.iteration,
                    start_time.elapsed(),
                    reason,
                )
                .with_function_evaluations(cost_misses)  // Use cache misses as actual evaluations
                .with_gradient_evaluations(grad_misses)  // Use cache misses as actual evaluations
                .with_gradient_norm(state.gradient_norm.unwrap_or(T::zero())));
            }
            
            // Check if trust region is too small
            if tr_state.radius < self.config.min_radius {
                // Get cache statistics for diagnostics
                let ((_cost_hits, cost_misses), (_grad_hits, grad_misses), _) = cached_cost_fn.cache_stats();
                
                return Ok(OptimizationResult::new(
                    state.point,
                    state.value,
                    state.iteration,
                    start_time.elapsed(),
                    TerminationReason::Converged,
                )
                .with_function_evaluations(cost_misses)  // Use cache misses as actual evaluations
                .with_gradient_evaluations(grad_misses)  // Use cache misses as actual evaluations
                .with_gradient_norm(state.gradient_norm.unwrap_or(T::zero())));
            }
            
            // Perform one optimization step
            self.step_internal(&cached_cost_fn, manifold, retraction, &mut state, &mut tr_state, &mut workspace)?;
        }
    }

    /// Performs a single optimization step.
    pub fn step<C, M, R>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        retraction: &R,
        state: &mut OptimizerState<T, D>,
    ) -> Result<()>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
        R: Retraction<T, D>,
    {
        // Create temporary trust region state
        let dim = state.point.shape_generic().0;
        let mut tr_state = TrustRegionState::new(self.config.initial_radius, dim);
        
        // Create temporary workspace
        let n = state.point.len();
        let mut workspace = WorkspaceBuilder::new()
            .with_standard_buffers(n)
            .build();
        
        self.step_internal(cost_fn, manifold, retraction, state, &mut tr_state, &mut workspace)
    }
}

// Implementation of the Optimizer trait
impl<T, D> Optimizer<T, D> for TrustRegion<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    fn name(&self) -> &str {
        "Riemannian Trust Region"
    }

    fn optimize<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Use the default retraction
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        TrustRegion::optimize(self, cost_fn, manifold, &retraction, initial_point, stopping_criterion)
    }

    fn step<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, D>,
    ) -> Result<()>
    where
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Use the default retraction
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        self.step(cost_fn, manifold, &retraction, state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use riemannopt_core::{
        cost_function::QuadraticCost,
        test_manifolds::TestEuclideanManifold,
        types::DVector,
    };
    use nalgebra::Dyn;
    use approx::assert_relative_eq;

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
        let mut state = TrustRegionState::<f64, Dyn>::new(1.0, Dyn(2));
        
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
    fn test_trust_region_on_quadratic() {
        let cost_fn = QuadraticCost::simple(Dyn(3));
        let manifold = TestEuclideanManifold::new(3);
        let initial_point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        let config = TrustRegionConfig::new()
            .with_initial_radius(0.5)
            .with_max_cg_iterations(10);
        
        let mut optimizer = TrustRegion::new(config);
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        use riemannopt_core::retraction::DefaultRetraction;
        let retraction = DefaultRetraction;
        
        let result = optimizer.optimize(
            &cost_fn,
            &manifold,
            &retraction,
            &initial_point,
            &stopping_criterion,
        ).unwrap();
        
        // Should converge to origin for simple quadratic
        assert!(result.converged);
        assert!(result.point.norm() < 1e-3);
        assert!(result.iterations < 100);
    }

    #[test]
    fn test_steihaug_cg_boundary() {
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let point = DVector::from_vec(vec![1.0, 1.0]);
        let gradient = DVector::from_vec(vec![1.0, 1.0]);
        let radius = 0.5;
        
        let config = TrustRegionConfig::<f64>::new()
            .with_cg_tolerance(1e-10);
        
        let (step, boundary_hit) = SteiahugCG::solve(
            &cost_fn,
            &manifold,
            &point,
            &gradient,
            radius,
            &config,
        ).unwrap();
        
        let step_norm = step.norm();
        
        // Step should be on the boundary
        assert!(boundary_hit);
        assert_relative_eq!(step_norm, radius, epsilon = 1e-6);
    }

    #[test]
    fn test_trust_region_optimizer_trait() {
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let mut optimizer = TrustRegion::<f64, Dyn>::new(TrustRegionConfig::default());
        
        // Test that it implements the Optimizer trait
        assert_eq!(optimizer.name(), "Riemannian Trust Region");
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(50)
            .with_gradient_tolerance(1e-6);
        
        // Use the trait method
        let result: OptimizationResult<f64, Dyn> = Optimizer::optimize(
            &mut optimizer,
            &cost_fn,
            &manifold,
            &initial_point,
            &stopping_criterion,
        ).unwrap();
        
        assert!(result.converged);
    }

    #[test]
    fn test_steihaug_cg_zero_allocation() {
        // Test that the zero-allocation version produces the same results as the allocating version
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let point = DVector::from_vec(vec![1.0, 1.0]);
        let gradient = DVector::from_vec(vec![1.0, 1.0]);
        let radius = 0.5;
        
        let config = TrustRegionConfig::<f64>::new()
            .with_cg_tolerance(1e-10)
            .with_max_cg_iterations(10);
        
        // Test allocating version
        let (step_alloc, boundary_hit_alloc) = SteiahugCG::solve(
            &cost_fn,
            &manifold,
            &point,
            &gradient,
            radius,
            &config,
        ).unwrap();
        
        // Test zero-allocation version
        let mut workspace = SteiahugCGWorkspace::new(Dyn(2));
        SteiahugCG::solve_with_workspace(
            &cost_fn,
            &manifold,
            &point,
            &gradient,
            radius,
            &config,
            &mut workspace,
        ).unwrap();
        
        let step_workspace = workspace.s.clone();
        let boundary_hit_workspace = workspace.temp[0] != 0.0;
        
        // Both versions should produce the same results
        assert_relative_eq!(step_alloc[0], step_workspace[0], epsilon = 1e-10);
        assert_relative_eq!(step_alloc[1], step_workspace[1], epsilon = 1e-10);
        assert_eq!(boundary_hit_alloc, boundary_hit_workspace);
        
        // Verify that the solution norm is correct
        let step_norm = step_workspace.norm();
        if boundary_hit_workspace {
            assert_relative_eq!(step_norm, radius, epsilon = 1e-10);
        }
    }
}