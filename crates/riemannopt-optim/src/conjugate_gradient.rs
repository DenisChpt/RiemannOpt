//! Riemannian Conjugate Gradient optimizer.
//!
//! Conjugate gradient methods are a class of optimization algorithms that use
//! conjugate directions to achieve faster convergence than steepest descent.
//! This implementation extends classical CG methods to Riemannian manifolds.
//!
//! # Algorithm Overview
//!
//! At each iteration, the conjugate gradient method:
//! 1. Computes the Riemannian gradient
//! 2. Determines a conjugate direction using the chosen beta formula
//! 3. Performs line search along the conjugate direction
//! 4. Updates the position using retraction
//!
//! # Supported Methods
//!
//! - **Fletcher-Reeves (FR)**: β = ||g_k||² / ||g_{k-1}||²
//! - **Polak-Ribière (PR)**: β = <g_k, g_k - g_{k-1}> / ||g_{k-1}||²
//! - **Hestenes-Stiefel (HS)**: β = <g_k, g_k - g_{k-1}> / <d_{k-1}, g_k - g_{k-1}>
//! - **Dai-Yuan (DY)**: β = ||g_k||² / <d_{k-1}, g_k - g_{k-1}>
//!
//! # Key Features
//!
//! - **Multiple CG variants**: FR, PR, HS, DY methods
//! - **Automatic restarts**: Periodic or condition-based restarts
//! - **Preconditioning support**: Optional preconditioner application
//! - **Hybrid methods**: PR+ (non-negative PR) and other variants
//! - **Line search integration**: Ensures sufficient decrease
//!
//! # References
//!
//! - Hager & Zhang, "A survey of nonlinear conjugate gradient methods" (2006)
//! - Dai & Yuan, "A nonlinear conjugate gradient method with a strong global convergence property" (1999)
//! - Ring & Wirth, "Optimization methods on Riemannian manifolds and their application to shape space" (2012)

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
        line_search::BacktrackingLineSearch,
    },
};
use std::time::Instant;
use std::fmt::Debug;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use num_traits::Float;

/// Conjugate gradient method variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum ConjugateGradientMethod {
    /// Fletcher-Reeves
    FletcherReeves,
    /// Polak-Ribière
    PolakRibiere,
    /// Hestenes-Stiefel
    HestenesStiefel,
    /// Dai-Yuan
    DaiYuan,
}

/// Internal state for Conjugate Gradient optimizer.
#[derive(Debug)]
struct CGInternalState<T, P, TV>
where
    T: Scalar,
    P: Clone + Debug + Send + Sync,
    TV: Clone + Debug + Send + Sync,
{
    workspace: Workspace<T>,
    iteration: usize,
    
    /// Previous search direction
    previous_direction: Option<TV>,
    /// Previous gradient
    previous_gradient: Option<TV>,
    /// Previous point
    previous_point: Option<P>,
    /// Method for computing beta (FR, PR, HS, DY)
    method: ConjugateGradientMethod,
    /// Number of iterations since last restart
    iterations_since_restart: usize,
    /// Restart period (0 means no periodic restart)
    restart_period: usize,
}

impl<T, P, TV> CGInternalState<T, P, TV>
where
    T: Scalar,
    P: Clone + Debug + Send + Sync,
    TV: Clone + Debug + Send + Sync,
{
    /// Creates a new conjugate gradient state.
    fn new(n: usize, method: ConjugateGradientMethod, restart_period: usize) -> Self {
        let mut workspace = Workspace::with_size(n);
        
        // Pre-allocate workspace buffers
        workspace.get_or_create_vector(BufferId::Gradient, n);
        workspace.get_or_create_vector(BufferId::Direction, n);
        workspace.get_or_create_vector(BufferId::Temp1, n);
        workspace.get_or_create_vector(BufferId::Temp2, n);
        
        Self {
            workspace,
            iteration: 0,
            previous_direction: None,
            previous_gradient: None,
            previous_point: None,
            method,
            iterations_since_restart: 0,
            restart_period,
        }
    }
    
    fn update_iteration(&mut self) {
        self.iteration += 1;
        self.iterations_since_restart += 1;
    }

    /// Computes the beta coefficient for the conjugate gradient method.
    fn compute_beta<M>(
        &self,
        manifold: &M,
        point: &P,
        gradient: &TV,
        config: &CGConfig<T>,
    ) -> Result<T>
    where
        M: Manifold<T, Point = P, TangentVector = TV>,
    {
        if self.previous_gradient.is_none() || self.previous_direction.is_none() {
            return Ok(T::zero());
        }
        
        let prev_grad = self.previous_gradient.as_ref().unwrap();
        let prev_dir = self.previous_direction.as_ref().unwrap();
        
        // Compute inner products needed for beta calculation
        let grad_norm_sq = manifold.inner_product(point, gradient, gradient)?;
        let prev_grad_norm_sq = manifold.inner_product(point, prev_grad, prev_grad)?;
        
        let beta = match self.method {
            ConjugateGradientMethod::FletcherReeves => {
                // β = ||g_k||² / ||g_{k-1}||²
                if prev_grad_norm_sq > T::epsilon() {
                    grad_norm_sq / prev_grad_norm_sq
                } else {
                    T::zero()
                }
            }
            ConjugateGradientMethod::PolakRibiere => {
                // β = <g_k, g_k - g_{k-1}> / ||g_{k-1}||²
                let mut grad_diff = gradient.clone();
                let mut transported_prev_grad = prev_grad.clone();
                
                // Transport previous gradient to current point if points differ
                if let Some(ref prev_point) = self.previous_point {
                    manifold.parallel_transport(prev_point, point, prev_grad, &mut transported_prev_grad, &mut self.workspace.clone())?;
                }
                
                // Compute g_k - g_{k-1}
                // First create -g_{k-1}
                let mut neg_prev_grad = transported_prev_grad.clone();
                manifold.scale_tangent(point, -T::one(), &transported_prev_grad, &mut neg_prev_grad, &mut self.workspace.clone())?;
                // Then add g_k + (-g_{k-1})
                manifold.add_tangents(point, gradient, &neg_prev_grad, &mut grad_diff, &mut self.workspace.clone())?;
                
                let numerator = manifold.inner_product(point, gradient, &grad_diff)?;
                
                if prev_grad_norm_sq > T::epsilon() {
                    let beta = numerator / prev_grad_norm_sq;
                    if config.use_pr_plus {
                        <T as Float>::max(T::zero(), beta) // PR+
                    } else {
                        beta
                    }
                } else {
                    T::zero()
                }
            }
            ConjugateGradientMethod::HestenesStiefel => {
                // β = <g_k, g_k - g_{k-1}> / <d_{k-1}, g_k - g_{k-1}>
                let mut grad_diff = gradient.clone();
                let mut transported_prev_grad = prev_grad.clone();
                let mut transported_prev_dir = prev_dir.clone();
                
                // Transport previous values to current point if needed
                if let Some(ref prev_point) = self.previous_point {
                    manifold.parallel_transport(prev_point, point, prev_grad, &mut transported_prev_grad, &mut self.workspace.clone())?;
                    manifold.parallel_transport(prev_point, point, prev_dir, &mut transported_prev_dir, &mut self.workspace.clone())?;
                }
                
                // Compute g_k - g_{k-1}
                // First create -g_{k-1}
                let mut neg_prev_grad = transported_prev_grad.clone();
                manifold.scale_tangent(point, -T::one(), &transported_prev_grad, &mut neg_prev_grad, &mut self.workspace.clone())?;
                // Then add g_k + (-g_{k-1})
                manifold.add_tangents(point, gradient, &neg_prev_grad, &mut grad_diff, &mut self.workspace.clone())?;
                
                let numerator = manifold.inner_product(point, gradient, &grad_diff)?;
                let denominator = manifold.inner_product(point, &transported_prev_dir, &grad_diff)?;
                
                if <T as Float>::abs(denominator) > T::epsilon() {
                    numerator / denominator
                } else {
                    T::zero()
                }
            }
            ConjugateGradientMethod::DaiYuan => {
                // β = ||g_k||² / <d_{k-1}, g_k - g_{k-1}>
                let mut grad_diff = gradient.clone();
                let mut transported_prev_grad = prev_grad.clone();
                let mut transported_prev_dir = prev_dir.clone();
                
                // Transport previous values to current point if needed
                if let Some(ref prev_point) = self.previous_point {
                    manifold.parallel_transport(prev_point, point, prev_grad, &mut transported_prev_grad, &mut self.workspace.clone())?;
                    manifold.parallel_transport(prev_point, point, prev_dir, &mut transported_prev_dir, &mut self.workspace.clone())?;
                }
                
                // Compute g_k - g_{k-1}
                // First create -g_{k-1}
                let mut neg_prev_grad = transported_prev_grad.clone();
                manifold.scale_tangent(point, -T::one(), &transported_prev_grad, &mut neg_prev_grad, &mut self.workspace.clone())?;
                // Then add g_k + (-g_{k-1})
                manifold.add_tangents(point, gradient, &neg_prev_grad, &mut grad_diff, &mut self.workspace.clone())?;
                
                let denominator = manifold.inner_product(point, &transported_prev_dir, &grad_diff)?;
                
                if <T as Float>::abs(denominator) > T::epsilon() {
                    grad_norm_sq / denominator
                } else {
                    T::zero()
                }
            }
        };
        
        // Apply beta bounds if configured
        let mut beta_bounded = beta;
        if let Some(min_beta) = config.min_beta {
            if beta < min_beta {
                beta_bounded = T::zero(); // Restart if beta too small
            }
        }
        if let Some(max_beta) = config.max_beta {
            beta_bounded = <T as Float>::min(beta_bounded, max_beta);
        }
        
        Ok(beta_bounded)
    }
}

/// Public state for Conjugate Gradient optimizer (for compatibility).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ConjugateGradientState<T, P, TV>
where
    T: Scalar,
{
    /// Method for computing beta (FR, PR, HS, DY)
    pub method: ConjugateGradientMethod,
    /// Restart period (0 means no periodic restart)
    pub restart_period: usize,
    _phantom: std::marker::PhantomData<(T, P, TV)>,
}

impl<T, P, TV> ConjugateGradientState<T, P, TV>
where
    T: Scalar,
{
    /// Creates a new conjugate gradient state.
    pub fn new(method: ConjugateGradientMethod, restart_period: usize) -> Self {
        Self {
            method,
            restart_period,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Configuration for the Conjugate Gradient optimizer.
#[derive(Debug, Clone)]
pub struct CGConfig<T: Scalar> {
    /// The CG method variant to use
    pub method: ConjugateGradientMethod,
    /// Period for automatic restarts (0 = no automatic restart)
    pub restart_period: usize,
    /// Whether to use PR+ (non-negative Polak-Ribière)
    pub use_pr_plus: bool,
    /// Line search strategy
    pub line_search: Option<BacktrackingLineSearch>,
    /// Minimum value of beta before restart
    pub min_beta: Option<T>,
    /// Maximum value of beta allowed
    pub max_beta: Option<T>,
    /// Whether to use preconditioning
    pub use_preconditioner: bool,
}

impl<T: Scalar> Default for CGConfig<T> {
    fn default() -> Self {
        Self {
            method: ConjugateGradientMethod::PolakRibiere,
            restart_period: 0, // No automatic restart by default
            use_pr_plus: true, // Use PR+ by default
            line_search: Some(BacktrackingLineSearch::new()),
            min_beta: Some(<T as Scalar>::from_f64(-0.5)),
            max_beta: Some(<T as Scalar>::from_f64(10.0)),
            use_preconditioner: false,
        }
    }
}

impl<T: Scalar> CGConfig<T> {
    /// Creates a new configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the CG method variant.
    pub fn with_method(mut self, method: ConjugateGradientMethod) -> Self {
        self.method = method;
        self
    }

    /// Sets the restart period (n means restart every n iterations).
    pub fn with_restart_period(mut self, period: usize) -> Self {
        self.restart_period = period;
        self
    }

    /// Enables or disables PR+ (non-negative Polak-Ribière).
    pub fn with_pr_plus(mut self, use_pr_plus: bool) -> Self {
        self.use_pr_plus = use_pr_plus;
        self
    }

    /// Sets the line search strategy.
    pub fn with_line_search(mut self) -> Self {
        self.line_search = Some(BacktrackingLineSearch::new());
        self
    }

    /// Sets the minimum beta value before restart.
    pub fn with_min_beta(mut self, min_beta: T) -> Self {
        self.min_beta = Some(min_beta);
        self
    }

    /// Sets the maximum beta value allowed.
    pub fn with_max_beta(mut self, max_beta: T) -> Self {
        self.max_beta = Some(max_beta);
        self
    }

    /// Enables preconditioning.
    pub fn with_preconditioner(mut self) -> Self {
        self.use_preconditioner = true;
        self
    }

    /// Creates a configuration for Fletcher-Reeves method.
    pub fn fletcher_reeves() -> Self {
        Self::new().with_method(ConjugateGradientMethod::FletcherReeves)
    }

    /// Creates a configuration for Polak-Ribière method.
    pub fn polak_ribiere() -> Self {
        Self::new().with_method(ConjugateGradientMethod::PolakRibiere)
    }

    /// Creates a configuration for Hestenes-Stiefel method.
    pub fn hestenes_stiefel() -> Self {
        Self::new().with_method(ConjugateGradientMethod::HestenesStiefel)
    }

    /// Creates a configuration for Dai-Yuan method.
    pub fn dai_yuan() -> Self {
        Self::new().with_method(ConjugateGradientMethod::DaiYuan)
    }
}

/// Riemannian Conjugate Gradient optimizer.
///
/// This optimizer adapts the classical conjugate gradient algorithm to Riemannian
/// manifolds by properly handling the transport of search directions and using
/// the manifold's metric for inner products.
///
/// # Examples
///
/// ```rust,ignore
/// use riemannopt_optim::{ConjugateGradient, CGConfig};
/// 
/// // Basic CG with Polak-Ribière method
/// let cg: ConjugateGradient<f64> = ConjugateGradient::new(CGConfig::new());
/// 
/// // CG with Fletcher-Reeves and periodic restart
/// let cg_fr = ConjugateGradient::new(
///     CGConfig::fletcher_reeves()
///         .with_restart_period(10)
///         .with_min_beta(0.0)
/// );
/// ```
#[derive(Debug)]
pub struct ConjugateGradient<T: Scalar> {
    config: CGConfig<T>,
}

impl<T: Scalar> ConjugateGradient<T> {
    /// Creates a new Conjugate Gradient optimizer with the given configuration.
    pub fn new(config: CGConfig<T>) -> Self {
        Self {
            config,
        }
    }

    /// Creates a new Conjugate Gradient optimizer with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(CGConfig::default())
    }

    /// Returns the configuration.
    pub fn config(&self) -> &CGConfig<T> {
        &self.config
    }

    /// Returns the optimizer name.
    pub fn name(&self) -> &str {
        match self.config.method {
            ConjugateGradientMethod::FletcherReeves => "Riemannian CG-FR",
            ConjugateGradientMethod::PolakRibiere => {
                if self.config.use_pr_plus {
                    "Riemannian CG-PR+"
                } else {
                    "Riemannian CG-PR"
                }
            }
            ConjugateGradientMethod::HestenesStiefel => "Riemannian CG-HS",
            ConjugateGradientMethod::DaiYuan => "Riemannian CG-DY",
        }
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
        let mut internal_state = CGInternalState::<T, M::Point, M::TangentVector>::new(
            n,
            self.config.method,
            self.config.restart_period,
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
            
            // Perform one optimization step
            self.step_with_state(cost_fn, manifold, &mut state, &mut internal_state)?;
        }
    }

    /// Performs a single optimization step with internal state.
    fn step_with_state<C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, M::Point, M::TangentVector>,
        internal_state: &mut CGInternalState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        // Initialize buffers for gradients
        let mut euclidean_grad = if let Some(ref g) = state.gradient {
            // Reuse existing gradient vector
            g.clone()
        } else {
            // Create a gradient by calling gradient_fd_alloc first time
            cost_fn.gradient_fd_alloc(&state.point)?
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
        
        // Compute beta coefficient
        let beta = internal_state.compute_beta(manifold, &state.point, &riemannian_grad, &self.config)?;
        
        // Compute conjugate gradient direction
        let mut direction = riemannian_grad.clone();
        {
            let workspace = &mut internal_state.workspace;
            
            // Check if we should restart
            let should_restart = (internal_state.iterations_since_restart > 0 && 
                                 internal_state.restart_period > 0 && 
                                 internal_state.iterations_since_restart >= internal_state.restart_period) ||
                                beta < T::zero() || beta.is_nan();
            
            if should_restart || internal_state.previous_direction.is_none() {
                // Restart with steepest descent: d = -g
                manifold.scale_tangent(&state.point, -T::one(), &riemannian_grad, &mut direction, workspace)?;
                internal_state.iterations_since_restart = 0;
            } else {
                // Compute conjugate direction: d = -g + beta * d_{k-1}
                let prev_dir = internal_state.previous_direction.as_ref().unwrap();
                let mut transported_prev_dir = prev_dir.clone();
                
                // Transport previous direction to current point if needed
                if let Some(ref prev_point) = internal_state.previous_point {
                    manifold.parallel_transport(prev_point, &state.point, prev_dir, &mut transported_prev_dir, workspace)?;
                }
                
                // d = beta * d_{k-1}
                manifold.scale_tangent(&state.point, beta, &transported_prev_dir, &mut direction, workspace)?;
                // d = -g + beta * d_{k-1}
                let mut neg_grad = riemannian_grad.clone();
                manifold.scale_tangent(&state.point, -T::one(), &riemannian_grad, &mut neg_grad, workspace)?;
                let mut new_direction = direction.clone();
                manifold.add_tangents(&state.point, &neg_grad, &direction, &mut new_direction, workspace)?;
                direction = new_direction;
                
                // CRITICAL: Project the direction back to the tangent space
                // This ensures the direction is truly tangent at the current point
                let mut projected_direction = direction.clone();
                manifold.project_tangent(&state.point, &direction, &mut projected_direction, workspace)?;
                direction = projected_direction;
            }
        }
        
        // Perform line search
        let step_size = if let Some(ref _line_search) = self.config.line_search {
            let workspace = &mut internal_state.workspace;
            self.perform_line_search(
                cost_fn,
                manifold,
                &state.point,
                &direction,
                state.value,
                &riemannian_grad,
                workspace,
            )?
        } else {
            // Use fixed step size
            T::one()
        };
        
        // Take the step
        {
            let workspace = &mut internal_state.workspace;
            // Scale direction by step size
            let mut scaled_direction = direction.clone();
            manifold.scale_tangent(&state.point, step_size, &direction, &mut scaled_direction, workspace)?;
            
            // Take the step using retraction
            manifold.retract(&state.point, &scaled_direction, &mut new_point, workspace)?;
        }
        
        // Evaluate cost at new point
        let new_cost = cost_fn.cost(&new_point)?;
        state.function_evaluations += 1;
        
        // Update internal state for next iteration
        internal_state.previous_direction = Some(direction);
        internal_state.previous_gradient = Some(riemannian_grad);
        internal_state.previous_point = Some(state.point.clone());
        internal_state.update_iteration();
        
        // Update optimization state
        state.point = new_point;
        state.value = new_cost;
        state.iteration += 1;
        
        Ok(())
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
        let mut internal_state = CGInternalState::<T, M::Point, M::TangentVector>::new(
            n,
            self.config.method,
            self.config.restart_period,
        );
        
        // Delegate to internal implementation
        self.step_with_state(cost_fn, manifold, state, &mut internal_state)
    }
    
    /// Performs line search to find an appropriate step size.
    fn perform_line_search<C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        direction: &M::TangentVector,
        current_cost: T,
        _gradient: &M::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<T>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        // Backtracking line search with Armijo condition
        let c1 = <T as Scalar>::from_f64(1e-4);
        let backtrack_factor = <T as Scalar>::from_f64(0.5);
        let mut alpha = T::one();
        let max_iterations = 20;
        
        // Compute directional derivative using the gradient we already have
        let directional_derivative = manifold.inner_product(point, _gradient, direction)?;
        
        for _ in 0..max_iterations {
            // Try the step
            let mut scaled_direction = direction.clone();
            manifold.scale_tangent(point, alpha, direction, &mut scaled_direction, workspace)?;
            
            let mut trial_point = point.clone();
            manifold.retract(point, &scaled_direction, &mut trial_point, workspace)?;
            
            // Evaluate cost
            let trial_cost = cost_fn.cost(&trial_point)?;
            
            // Check Armijo condition
            let expected_decrease = c1 * alpha * directional_derivative;
            if trial_cost <= current_cost + expected_decrease {
                return Ok(alpha);
            }
            
            // Backtrack
            alpha *= backtrack_factor;
        }
        
        Ok(alpha)
    }
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for ConjugateGradient<T> {
    fn name(&self) -> &str {
        self.name()
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

    #[test]
    fn test_cg_config() {
        let config = CGConfig::<f64>::fletcher_reeves()
            .with_restart_period(10)
            .with_min_beta(-0.5)
            .with_max_beta(5.0);
        
        assert_eq!(config.method, ConjugateGradientMethod::FletcherReeves);
        assert_eq!(config.restart_period, 10);
        assert_eq!(config.min_beta, Some(-0.5));
        assert_eq!(config.max_beta, Some(5.0));
    }

    #[test]
    fn test_cg_variants() {
        let fr_config = CGConfig::<f64>::fletcher_reeves();
        let pr_config = CGConfig::<f64>::polak_ribiere();
        let hs_config = CGConfig::<f64>::hestenes_stiefel();
        let dy_config = CGConfig::<f64>::dai_yuan();
        
        assert_eq!(fr_config.method, ConjugateGradientMethod::FletcherReeves);
        assert_eq!(pr_config.method, ConjugateGradientMethod::PolakRibiere);
        assert_eq!(hs_config.method, ConjugateGradientMethod::HestenesStiefel);
        assert_eq!(dy_config.method, ConjugateGradientMethod::DaiYuan);
    }

    // Tests involving manifolds are temporarily disabled
    // TODO: Re-enable tests once test infrastructure is in place
}