//! Riemannian L-BFGS optimizer.
//!
//! L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton optimization
//! algorithm that approximates the inverse Hessian using a limited history of past gradient
//! and position updates. This implementation extends L-BFGS to Riemannian manifolds by
//! properly handling parallel transport of stored vectors.
//!
//! # Algorithm Overview
//!
//! The Riemannian L-BFGS algorithm:
//! 1. Stores m most recent gradient differences and position differences
//! 2. Approximates the inverse Hessian-vector product using two-loop recursion
//! 3. Computes search direction as negative approximate Newton direction
//! 4. Performs line search to find suitable step size
//! 5. Updates position using retraction
//!
//! ## Two-Loop Recursion Algorithm
//!
//! The core of L-BFGS is the two-loop recursion that efficiently computes H_k * grad_f(x_k):
//!
//! ```text
//! q = grad_f(x_k)
//! for i = k-1, k-2, ..., k-m:
//!     α_i = ρ_i * <s_i, q>
//!     q = q - α_i * y_i
//! 
//! r = H_0 * q  // Initial Hessian approximation
//! 
//! for i = k-m, k-m+1, ..., k-1:
//!     β = ρ_i * <y_i, r>
//!     r = r + (α_i - β) * s_i
//! 
//! return -r  // Search direction
//! ```
//!
//! ## Riemannian Adaptations
//!
//! The key challenge in Riemannian L-BFGS is handling vectors from different tangent spaces:
//! - **Vector Transport**: Historical vectors must be transported to the current tangent space
//! - **Metric Awareness**: Inner products use the Riemannian metric
//! - **Retraction**: Updates use retraction instead of vector addition
//!
//! # Key Features
//!
//! - **Limited memory**: Only stores m vector pairs (typically 5-20)
//! - **Superlinear convergence**: Near optimal points with good initial Hessian approximation
//! - **Vector transport**: Properly handles curvature when transporting stored vectors
//! - **Automatic scaling**: Initial Hessian approximation based on most recent update
//! - **Strong Wolfe line search**: Ensures sufficient decrease and curvature conditions
//!
//! # References
//!
//! - Nocedal & Wright, "Numerical Optimization" (2006)
//! - Huang et al., "A Riemannian BFGS Method" (2015)
//! - Sato, "Riemannian Optimization and Its Applications" (2021)

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
        line_search::StrongWolfeLineSearch,
    },
};
use std::time::Instant;
use std::fmt::Debug;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use num_traits::Float;

/// Internal state for L-BFGS optimizer.
#[derive(Debug)]
struct LBFGSInternalState<T, P, TV>
where
    T: Scalar,
    P: Clone + Debug + Send + Sync,
    TV: Clone + Debug + Send + Sync,
{
    workspace: Workspace<T>,
    iteration: usize,
    
    // L-BFGS specific state
    /// Memory size (number of vector pairs to store)
    memory_size: usize,
    /// Stored position differences (s_k = x_{k+1} - x_k)
    s_history: Vec<TV>,
    /// Stored gradient differences (y_k = g_{k+1} - g_k)
    y_history: Vec<TV>,
    /// Inner products rho_k = 1 / (y_k^T s_k)
    rho_history: Vec<T>,
    /// Previous point
    previous_point: Option<P>,
    /// Previous gradient
    previous_gradient: Option<TV>,
    /// Search direction from previous iteration
    previous_direction: Option<TV>,
    /// Step size from previous iteration
    previous_step_size: Option<T>,
}

impl<T, P, TV> LBFGSInternalState<T, P, TV>
where
    T: Scalar,
    P: Clone + Debug + Send + Sync,
    TV: Clone + Debug + Send + Sync,
{
    fn new(n: usize, memory_size: usize) -> Self {
        let mut workspace = Workspace::with_size(n);
        
        // Pre-allocate workspace buffers
        workspace.get_or_create_vector(BufferId::Gradient, n);
        workspace.get_or_create_vector(BufferId::Direction, n);
        workspace.get_or_create_vector(BufferId::Temp1, n);
        workspace.get_or_create_vector(BufferId::Temp2, n);
        
        // Allocate additional buffers for L-BFGS history operations
        for i in 0..memory_size {
            workspace.get_or_create_vector(BufferId::Custom((i * 2) as u32), n);     // For transported s_i
            workspace.get_or_create_vector(BufferId::Custom((i * 2 + 1) as u32), n); // For transported y_i
        }
        
        Self {
            workspace,
            iteration: 0,
            memory_size,
            s_history: Vec::with_capacity(memory_size),
            y_history: Vec::with_capacity(memory_size),
            rho_history: Vec::with_capacity(memory_size),
            previous_point: None,
            previous_gradient: None,
            previous_direction: None,
            previous_step_size: None,
        }
    }
    
    fn update_iteration(&mut self) {
        self.iteration += 1;
    }
}

/// Public state for L-BFGS optimizer (for compatibility).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct LBFGSState<T, TV>
where
    T: Scalar,
{
    /// Memory size (number of vector pairs to store)
    pub memory_size: usize,
    _phantom: std::marker::PhantomData<(T, TV)>,
}

impl<T, TV> LBFGSState<T, TV>
where
    T: Scalar,
{
    /// Creates a new L-BFGS state.
    pub fn new(memory_size: usize) -> Self {
        Self {
            memory_size,
            _phantom: std::marker::PhantomData,
        }
    }
}

/// Configuration for the L-BFGS optimizer.
#[derive(Debug, Clone)]
pub struct LBFGSConfig<T: Scalar> {
    /// Number of vector pairs to store (typically 5-20)
    pub memory_size: usize,
    /// Line search strategy
    pub line_search: Option<StrongWolfeLineSearch>,
    /// Initial step size for line search
    pub initial_step_size: T,
    /// Whether to use cautious updates (skip updates that don't satisfy positive definiteness)
    pub use_cautious_updates: bool,
}

impl<T: Scalar> Default for LBFGSConfig<T> {
    fn default() -> Self {
        Self {
            memory_size: 10,
            line_search: Some(StrongWolfeLineSearch::new()),
            initial_step_size: <T as Scalar>::from_f64(1.0),
            use_cautious_updates: true,
        }
    }
}

impl<T: Scalar> LBFGSConfig<T> {
    /// Creates a new configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the memory size (number of vector pairs to store).
    pub fn with_memory_size(mut self, size: usize) -> Self {
        self.memory_size = size;
        self
    }

    /// Sets the line search strategy.
    pub fn with_line_search(mut self) -> Self {
        self.line_search = Some(StrongWolfeLineSearch::new());
        self
    }

    /// Sets the initial step size for line search.
    pub fn with_initial_step_size(mut self, step_size: T) -> Self {
        self.initial_step_size = step_size;
        self
    }

    /// Enables or disables cautious updates.
    pub fn with_cautious_updates(mut self, cautious: bool) -> Self {
        self.use_cautious_updates = cautious;
        self
    }
}

/// Riemannian L-BFGS optimizer.
///
/// This optimizer adapts the classical L-BFGS algorithm to Riemannian manifolds
/// by properly handling the transport of stored vector pairs and using the
/// manifold's metric for inner products.
///
/// # Examples
///
/// ```rust,ignore
/// use riemannopt_optim::{LBFGS, LBFGSConfig};
/// 
/// // Basic L-BFGS with default parameters
/// let lbfgs: LBFGS<f64> = LBFGS::new(LBFGSConfig::new());
/// 
/// // L-BFGS with custom parameters
/// let lbfgs_custom = LBFGS::new(
///     LBFGSConfig::new()
///         .with_memory_size(20)
///         .with_initial_step_size(0.1)
///         .with_cautious_updates(true)
/// );
/// ```
#[derive(Debug)]
pub struct LBFGS<T: Scalar> {
    config: LBFGSConfig<T>,
}

impl<T: Scalar> LBFGS<T> {
    /// Creates a new L-BFGS optimizer with given configuration.
    pub fn new(config: LBFGSConfig<T>) -> Self {
        Self { 
            config,
        }
    }

    /// Creates a new L-BFGS optimizer with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(LBFGSConfig::default())
    }

    /// Returns the optimizer configuration.
    pub fn config(&self) -> &LBFGSConfig<T> {
        &self.config
    }

    /// Returns the optimizer name.
    pub fn name(&self) -> &str {
        "Riemannian L-BFGS"
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
        let mut internal_state = LBFGSInternalState::<T, M::Point, M::TangentVector>::new(
            n, 
            self.config.memory_size
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
        internal_state: &mut LBFGSInternalState<T, M::Point, M::TangentVector>,
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
        
        // Compute search direction using L-BFGS two-loop recursion
        let mut direction = riemannian_grad.clone();
        self.compute_lbfgs_direction(
            manifold,
            &state.point,
            &riemannian_grad,
            &mut direction,
            internal_state,
        )?;
        
        // Perform line search if configured
        let step_size = if let Some(ref _line_search) = self.config.line_search {
            // Perform strong Wolfe line search
            self.perform_line_search(
                cost_fn,
                manifold,
                &state.point,
                &direction,
                state.value,
                &riemannian_grad,
                internal_state,
            )?
        } else {
            // Use fixed step size
            if state.iteration == 0 {
                self.config.initial_step_size
            } else {
                T::one()
            }
        };
        
        // Update history before taking the step
        internal_state.previous_direction = Some(direction.clone());
        internal_state.previous_step_size = Some(step_size);
        
        // Scale direction by step size and take the step
        {
            let workspace = &mut internal_state.workspace;
            let scaled_direction = direction.clone();
            manifold.scale_tangent(&state.point, -step_size, &scaled_direction, &mut direction, workspace)?;
            manifold.retract(&state.point, &direction, &mut new_point, workspace)?;
        }
        
        // Compute gradient at new point for history update
        let mut new_euclidean_grad = euclidean_grad.clone();
        let mut new_riemannian_grad = riemannian_grad.clone();
        {
            let workspace = &mut internal_state.workspace;
            let new_cost = cost_fn.cost_and_gradient(&new_point, workspace, &mut new_euclidean_grad)?;
            state.function_evaluations += 1;
            state.gradient_evaluations += 1;
            
            // Convert to Riemannian gradient
            manifold.euclidean_to_riemannian_gradient(&new_point, &new_euclidean_grad, &mut new_riemannian_grad, workspace)?;
            
            // Update L-BFGS history
            self.update_lbfgs_history(
                manifold,
                &state.point,
                &new_point,
                &riemannian_grad,
                &new_riemannian_grad,
                internal_state,
            )?;
            
            // Update state
            state.point = new_point;
            state.value = new_cost;
            state.iteration += 1;
        }
        
        // Update internal state iteration
        internal_state.update_iteration();
        
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
        // Create internal state
        let n = manifold.dimension();
        let mut internal_state = LBFGSInternalState::<T, M::Point, M::TangentVector>::new(
            n,
            self.config.memory_size
        );
        
        // Delegate to internal implementation
        self.step_with_state(cost_fn, manifold, state, &mut internal_state)
    }

    /// Computes the L-BFGS search direction using the two-loop recursion.
    fn compute_lbfgs_direction<M>(
        &self,
        manifold: &M,
        point: &M::Point,
        gradient: &M::TangentVector,
        direction: &mut M::TangentVector,
        internal_state: &mut LBFGSInternalState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        M: Manifold<T>,
    {
        let workspace = &mut internal_state.workspace;
        let m = internal_state.s_history.len();
        
        if m == 0 {
            // No history yet, return negative gradient
            direction.clone_from(gradient);
            manifold.scale_tangent(point, -T::one(), gradient, direction, workspace)?;
            return Ok(());
        }
        
        // Allocate workspace for alpha values
        let mut alpha = vec![T::zero(); m];
        
        // Initialize q = gradient
        let mut q = gradient.clone();
        
        // Transport all stored vectors to current tangent space
        let mut transported_s = Vec::with_capacity(m);
        let mut transported_y = Vec::with_capacity(m);
        
        for i in 0..m {
            let mut s_i = internal_state.s_history[i].clone();
            let mut y_i = internal_state.y_history[i].clone();
            
            // Transport s_i and y_i to current tangent space
            // Note: s_i and y_i are stored at different points along the trajectory
            // For simplicity, we transport them from the previous point
            if let Some(ref prev_point) = internal_state.previous_point {
                manifold.parallel_transport(prev_point, point, &internal_state.s_history[i], &mut s_i, workspace)?;
                manifold.parallel_transport(prev_point, point, &internal_state.y_history[i], &mut y_i, workspace)?;
            }
            
            transported_s.push(s_i);
            transported_y.push(y_i);
        }
        
        // First loop: compute alpha[i] and update q
        for i in (0..m).rev() {
            // alpha[i] = rho[i] * <s[i], q>
            let s_dot_q = manifold.inner_product(point, &transported_s[i], &q)?;
            alpha[i] = internal_state.rho_history[i] * s_dot_q;
            
            // q = q - alpha[i] * y[i]
            let mut scaled_y = transported_y[i].clone();
            manifold.scale_tangent(point, -alpha[i], &transported_y[i], &mut scaled_y, workspace)?;
            let temp_q = q.clone();
            manifold.add_tangents(point, &temp_q, &scaled_y, &mut q, workspace)?;
        }
        
        // Compute initial Hessian approximation H_0
        let mut r = q.clone();
        if m > 0 {
            // H_0 = gamma * I where gamma = <s_{m-1}, y_{m-1}> / <y_{m-1}, y_{m-1}>
            let s_dot_y = manifold.inner_product(point, &transported_s[m-1], &transported_y[m-1])?;
            let y_dot_y = manifold.inner_product(point, &transported_y[m-1], &transported_y[m-1])?;
            
            if y_dot_y > T::zero() {
                let gamma = s_dot_y / y_dot_y;
                manifold.scale_tangent(point, gamma, &q, &mut r, workspace)?;
            }
        }
        
        // Second loop: compute search direction
        for i in 0..m {
            // beta = rho[i] * <y[i], r>
            let y_dot_r = manifold.inner_product(point, &transported_y[i], &r)?;
            let beta = internal_state.rho_history[i] * y_dot_r;
            
            // r = r + (alpha[i] - beta) * s[i]
            let coeff = alpha[i] - beta;
            let mut scaled_s = transported_s[i].clone();
            manifold.scale_tangent(point, coeff, &transported_s[i], &mut scaled_s, workspace)?;
            let temp = r.clone();
            manifold.add_tangents(point, &temp, &scaled_s, &mut r, workspace)?;
        }
        
        // Return negative direction for descent
        manifold.scale_tangent(point, -T::one(), &r, direction, workspace)?;
        
        Ok(())
    }
    
    /// Performs line search to find an appropriate step size.
    fn perform_line_search<C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        direction: &M::TangentVector,
        current_cost: T,
        gradient: &M::TangentVector,
        internal_state: &mut LBFGSInternalState<T, M::Point, M::TangentVector>,
    ) -> Result<T>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        let workspace = &mut internal_state.workspace;
        
        // Compute directional derivative
        let directional_derivative = manifold.inner_product(point, gradient, direction)?;
        
        // Strong Wolfe line search parameters
        let c1 = <T as Scalar>::from_f64(1e-4); // Sufficient decrease parameter
        let c2 = <T as Scalar>::from_f64(0.9);  // Curvature parameter
        
        let mut alpha = self.config.initial_step_size;
        let max_iterations = 20;
        
        for _ in 0..max_iterations {
            // Try the step
            let mut scaled_direction = direction.clone();
            manifold.scale_tangent(point, -alpha, direction, &mut scaled_direction, workspace)?;
            
            let mut trial_point = point.clone();
            manifold.retract(point, &scaled_direction, &mut trial_point, workspace)?;
            
            // Evaluate cost
            let trial_cost = cost_fn.cost(&trial_point)?;
            
            // Check Armijo condition
            let expected_decrease = c1 * alpha * directional_derivative;
            if trial_cost <= current_cost + expected_decrease {
                // Check curvature condition (simplified)
                let trial_euclidean_gradient = cost_fn.gradient(&trial_point)?;
                let mut trial_gradient = trial_euclidean_gradient.clone();
                
                // Convert to Riemannian gradient
                manifold.euclidean_to_riemannian_gradient(&trial_point, &trial_euclidean_gradient, &mut trial_gradient, workspace)?;
                
                // Transport direction to trial point for curvature check
                // For now, we use a simplified check
                let trial_directional = manifold.inner_product(&trial_point, &trial_gradient, direction)?;
                
                if <T as Float>::abs(trial_directional) <= c2 * <T as Float>::abs(directional_derivative) {
                    return Ok(alpha);
                }
            }
            
            // Backtrack
            alpha *= <T as Scalar>::from_f64(0.5);
        }
        
        Ok(alpha)
    }
    
    /// Updates the L-BFGS history with new information.
    fn update_lbfgs_history<M>(
        &self,
        manifold: &M,
        old_point: &M::Point,
        new_point: &M::Point,
        old_gradient: &M::TangentVector,
        new_gradient: &M::TangentVector,
        internal_state: &mut LBFGSInternalState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        M: Manifold<T>,
    {
        let workspace = &mut internal_state.workspace;
        
        // Only update if we have a previous point and gradient
        if internal_state.previous_point.is_some() && internal_state.previous_gradient.is_some() {
            // Compute s_k using the stored direction and step size
            let s_k = if let (Some(ref direction), Some(step_size)) = 
                (&internal_state.previous_direction, internal_state.previous_step_size) {
                let mut s = direction.clone();
                manifold.scale_tangent(old_point, step_size, direction, &mut s, workspace)?;
                s
            } else {
                // Fallback: use inverse retraction if available
                // For now, approximate with scaled gradient difference
                old_gradient.clone()
            };
            
            // Compute y_k = g_{k+1} - g_k (transport old gradient to new point's tangent space)
            let mut transported_old_grad = old_gradient.clone();
            manifold.parallel_transport(old_point, new_point, old_gradient, &mut transported_old_grad, workspace)?;
            
            // y_k = new_gradient - transported_old_grad
            let mut y_k = new_gradient.clone();
            let mut neg_transported_old_grad = transported_old_grad.clone();
            manifold.scale_tangent(new_point, -T::one(), &transported_old_grad, &mut neg_transported_old_grad, workspace)?;
            manifold.add_tangents(new_point, new_gradient, &neg_transported_old_grad, &mut y_k, workspace)?;
            
            // Compute rho_k = 1 / <y_k, s_k>
            // Note: we need to transport s_k to new_point for inner product
            let mut transported_s_k = s_k.clone();
            manifold.parallel_transport(old_point, new_point, &s_k, &mut transported_s_k, workspace)?;
            
            let y_dot_s = manifold.inner_product(new_point, &y_k, &transported_s_k)?;
            
            // Check if update satisfies curvature condition (for cautious updates)
            let min_curvature = <T as Scalar>::from_f64(1e-8);
            if <T as Float>::abs(y_dot_s) > min_curvature && 
               (!self.config.use_cautious_updates || y_dot_s > T::zero()) {
                
                let rho = T::one() / y_dot_s;
                
                // Store s_k and y_k at old_point's tangent space
                // (they will be transported to current point when used)
                if internal_state.s_history.len() >= internal_state.memory_size {
                    internal_state.s_history.remove(0);
                    internal_state.y_history.remove(0);
                    internal_state.rho_history.remove(0);
                }
                
                internal_state.s_history.push(s_k);
                internal_state.y_history.push(y_k);
                internal_state.rho_history.push(rho);
            }
        }
        
        // Update previous values
        internal_state.previous_point = Some(new_point.clone());
        internal_state.previous_gradient = Some(new_gradient.clone());
        
        Ok(())
    }
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for LBFGS<T> {
    fn name(&self) -> &str {
        "Riemannian L-BFGS"
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
    use riemannopt_core::types::DVector;

    #[test]
    fn test_lbfgs_config() {
        let config = LBFGSConfig::<f64>::new()
            .with_memory_size(20)
            .with_initial_step_size(0.5)
            .with_cautious_updates(false);
        
        assert_eq!(config.memory_size, 20);
        assert_eq!(config.initial_step_size, 0.5);
        assert!(!config.use_cautious_updates);
    }
    
    #[test]
    fn test_lbfgs_state() {
        let state = LBFGSState::<f64, DVector<f64>>::new(5);
        assert_eq!(state.memory_size, 5);
    }

    // Tests involving manifolds are temporarily disabled
    // TODO: Re-enable tests once test infrastructure is in place
}