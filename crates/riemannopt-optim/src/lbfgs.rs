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
        optimizer::{Optimizer, OptimizationResult, StoppingCriterion, TerminationReason},
        line_search::LineSearchParams,
    },
};
use std::time::Instant;
use std::fmt::Debug;
use num_traits::Float;
use std::collections::VecDeque;

/// Storage for one iteration's L-BFGS data.
#[derive(Debug)]
pub struct LBFGSHistoryEntry<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// Position difference vector s_k (in tangent space at x_k)
    s: TV,
    /// Gradient difference vector y_k (in tangent space at x_{k+1})
    y: TV,
    /// Inner product rho_k = 1 / <y_k, s_k>
    rho: T,
    /// Iteration number when this entry was created
    _iteration: usize,
}

/// State for L-BFGS history management.
#[derive(Debug)]
pub struct LBFGSState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// History entries stored in chronological order
    pub history: VecDeque<LBFGSHistoryEntry<T, TV>>,
    /// Maximum number of entries to store
    pub memory_size: usize,
    /// Whether to use cautious updates
    pub use_cautious_updates: bool,
}

impl<T, TV> LBFGSState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// Creates a new L-BFGS state.
    pub fn new(memory_size: usize) -> Self {
        Self {
            history: VecDeque::with_capacity(memory_size),
            memory_size,
            use_cautious_updates: true,
        }
    }
    
    /// Adds a new entry to the history.
    pub fn add_entry(&mut self, s: TV, y: TV, rho: T, iteration: usize) {
        if self.history.len() >= self.memory_size {
            self.history.pop_front();
        }
        self.history.push_back(LBFGSHistoryEntry { s, y, rho, _iteration: iteration });
    }
    
    /// Clears the history.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// Configuration for the L-BFGS optimizer.
#[derive(Debug, Clone)]
pub struct LBFGSConfig<T: Scalar> {
    /// Number of vector pairs to store (typically 5-20)
    pub memory_size: usize,
    /// Initial step size for line search
    pub initial_step_size: T,
    /// Whether to use cautious updates (skip updates that don't satisfy positive definiteness)
    pub use_cautious_updates: bool,
    /// Line search parameters
    pub line_search_params: LineSearchParams<T>,
}

impl<T: Scalar> Default for LBFGSConfig<T> {
    fn default() -> Self {
        Self {
            memory_size: 10,
            initial_step_size: <T as Scalar>::from_f64(1.0),
            use_cautious_updates: true,
            line_search_params: LineSearchParams::default(),
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
    
    /// Sets custom line search parameters.
    pub fn with_line_search_params(mut self, params: LineSearchParams<T>) -> Self {
        self.line_search_params = params;
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
        _workspace: &mut Workspace<T>,
        criterion: &StoppingCriterion<T>,
    ) -> Option<TerminationReason>
    where
        M: Manifold<T>,
    {
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
                    if let Ok(distance) = manifold.distance(prev_point, current_point) {
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
    
    /// Computes the L-BFGS search direction using the two-loop recursion.
    fn compute_lbfgs_direction_inplace<M>(
        &self,
        manifold: &M,
        current_point: &M::Point,
        gradient: &M::TangentVector,
        lbfgs_state: &LBFGSState<T, M::TangentVector>,
        direction: &mut M::TangentVector,
        _workspace: &mut Workspace<T>,
        point_history: &Vec<M::Point>,
    ) -> Result<()>
    where
        M: Manifold<T>,
    {
        let m = lbfgs_state.history.len();
        
        if m == 0 {
            // No history yet, return negative gradient
            manifold.scale_tangent(current_point, -T::one(), gradient, direction)?;
            return Ok(());
        }
        
        // Allocate workspace for alpha values
        let mut alpha = vec![T::zero(); m];
        
        // Initialize q = gradient
        let mut q = gradient.clone();
        
        // First loop: compute alpha[i] and update q
        for i in (0..m).rev() {
            let entry = &lbfgs_state.history[i];
            
            // Transport s_i and y_i to current tangent space
            let history_idx = point_history.len() - m + i;
            if history_idx < point_history.len() - 1 {
                // Transport from the point where the entry was created
                let mut transported_s = entry.s.clone();
                let mut transported_y = entry.y.clone();
                
                // s was in tangent space at x_k, transport to current point
                manifold.parallel_transport(
                    &point_history[history_idx],
                    current_point,
                    &entry.s,
                    &mut transported_s,
                )?;
                
                // y was in tangent space at x_{k+1}, transport to current point
                manifold.parallel_transport(
                    &point_history[history_idx + 1],
                    current_point,
                    &entry.y,
                    &mut transported_y,
                )?;
                
                // alpha[i] = rho[i] * <s[i], q>
                let s_dot_q = manifold.inner_product(current_point, &transported_s, &q)?;
                alpha[i] = entry.rho * s_dot_q;
                
                // q = q - alpha[i] * y[i]
                let mut scaled_y = transported_y.clone();
                manifold.scale_tangent(current_point, -alpha[i], &transported_y, &mut scaled_y)?;
                let temp_q = q.clone();
                let mut temp_add = q.clone();
                manifold.add_tangents(current_point, &temp_q, &scaled_y, &mut q, &mut temp_add)?;
            } else {
                // For the most recent entry, vectors are already in current tangent space
                let s_dot_q = manifold.inner_product(current_point, &entry.s, &q)?;
                alpha[i] = entry.rho * s_dot_q;
                
                let mut scaled_y = entry.y.clone();
                manifold.scale_tangent(current_point, -alpha[i], &entry.y, &mut scaled_y)?;
                let temp_q = q.clone();
                let mut temp_add = q.clone();
                manifold.add_tangents(current_point, &temp_q, &scaled_y, &mut q, &mut temp_add)?;
            }
        }
        
        // Compute initial Hessian approximation H_0
        let mut r = q.clone();
        if m > 0 {
            // Use the most recent entry for scaling
            let last_entry = &lbfgs_state.history[m - 1];
            
            // Transport vectors to current point if needed
            let history_idx = point_history.len() - 1;
            if history_idx > 0 && history_idx < point_history.len() {
                let mut transported_s = last_entry.s.clone();
                let mut transported_y = last_entry.y.clone();
                
                manifold.parallel_transport(
                    &point_history[history_idx - 1],
                    current_point,
                    &last_entry.s,
                    &mut transported_s,
                )?;
                
                manifold.parallel_transport(
                    &point_history[history_idx],
                    current_point,
                    &last_entry.y,
                    &mut transported_y,
                )?;
                
                // H_0 = gamma * I where gamma = <s_{m-1}, y_{m-1}> / <y_{m-1}, y_{m-1}>
                let s_dot_y = manifold.inner_product(current_point, &transported_s, &transported_y)?;
                let y_dot_y = manifold.inner_product(current_point, &transported_y, &transported_y)?;
                
                if y_dot_y > T::zero() {
                    let gamma = s_dot_y / y_dot_y;
                    manifold.scale_tangent(current_point, gamma, &q, &mut r)?;
                }
            } else {
                // Use vectors as they are (already in current tangent space)
                let s_dot_y = manifold.inner_product(current_point, &last_entry.s, &last_entry.y)?;
                let y_dot_y = manifold.inner_product(current_point, &last_entry.y, &last_entry.y)?;
                
                if y_dot_y > T::zero() {
                    let gamma = s_dot_y / y_dot_y;
                    manifold.scale_tangent(current_point, gamma, &q, &mut r)?;
                }
            }
        }
        
        // Second loop: compute search direction
        for i in 0..m {
            let entry = &lbfgs_state.history[i];
            
            // Transport vectors if needed
            let history_idx = point_history.len() - m + i;
            if history_idx < point_history.len() - 1 {
                let mut transported_s = entry.s.clone();
                let mut transported_y = entry.y.clone();
                
                manifold.parallel_transport(
                    &point_history[history_idx],
                    current_point,
                    &entry.s,
                    &mut transported_s,
                )?;
                
                manifold.parallel_transport(
                    &point_history[history_idx + 1],
                    current_point,
                    &entry.y,
                    &mut transported_y,
                )?;
                
                // beta = rho[i] * <y[i], r>
                let y_dot_r = manifold.inner_product(current_point, &transported_y, &r)?;
                let beta = entry.rho * y_dot_r;
                
                // r = r + (alpha[i] - beta) * s[i]
                let coeff = alpha[i] - beta;
                let mut scaled_s = transported_s.clone();
                manifold.scale_tangent(current_point, coeff, &transported_s, &mut scaled_s)?;
                let temp_r = r.clone();
                let mut temp_add = r.clone();
                manifold.add_tangents(current_point, &temp_r, &scaled_s, &mut r, &mut temp_add)?;
            } else {
                // Use vectors as they are
                let y_dot_r = manifold.inner_product(current_point, &entry.y, &r)?;
                let beta = entry.rho * y_dot_r;
                
                let coeff = alpha[i] - beta;
                let mut scaled_s = entry.s.clone();
                manifold.scale_tangent(current_point, coeff, &entry.s, &mut scaled_s)?;
                let temp_r = r.clone();
                let mut temp_add = r.clone();
                manifold.add_tangents(current_point, &temp_r, &scaled_s, &mut r, &mut temp_add)?;
            }
        }
        
        // Return negative direction for descent
        manifold.scale_tangent(current_point, -T::one(), &r, direction)?;
        
        Ok(())
    }
    
    /// Performs line search to find an appropriate step size.
    fn perform_line_search<M, C>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        direction: &M::TangentVector,
        current_cost: T,
        gradient: &M::TangentVector,
        _workspace: &mut Workspace<T>,
        function_evaluations: &mut usize,
        gradient_evaluations: &mut usize,
    ) -> Result<T>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        // Compute directional derivative
        let directional_derivative = manifold.inner_product(point, gradient, direction)?;
        
        // Use line search parameters from config
        let c1 = self.config.line_search_params.c1;
        let c2 = self.config.line_search_params.c2;
        let max_iterations = self.config.line_search_params.max_iterations;
        
        let mut alpha = self.config.initial_step_size;
        let alpha_min = <T as Scalar>::from_f64(1e-10);
        let alpha_max = <T as Scalar>::from_f64(10.0);
        
        for _iter in 0..max_iterations {
            // Try the step
            let mut scaled_direction = direction.clone();
            manifold.scale_tangent(point, alpha, direction, &mut scaled_direction)?;
            
            let mut trial_point = point.clone();
            manifold.retract(point, &scaled_direction, &mut trial_point)?;
            
            // Evaluate cost
            let trial_cost = cost_fn.cost(&trial_point)?;
            *function_evaluations += 1;
            
            // Check Armijo condition
            let expected_decrease = c1 * alpha * directional_derivative;
            if trial_cost <= current_cost + expected_decrease {
                // Check curvature condition
                let mut euclidean_grad = gradient.clone();
                let _trial_gradient_cost = cost_fn.cost_and_gradient(&trial_point, _workspace, &mut euclidean_grad)?;
                *function_evaluations += 1;
                *gradient_evaluations += 1;
                
                let mut trial_gradient = euclidean_grad.clone();
                manifold.euclidean_to_riemannian_gradient(
                    &trial_point,
                    &euclidean_grad,
                    &mut trial_gradient,
                )?;
                
                // Transport direction to trial point for curvature check
                let mut transported_direction = direction.clone();
                manifold.parallel_transport(point, &trial_point, direction, &mut transported_direction)?;
                
                let trial_directional = manifold.inner_product(&trial_point, &trial_gradient, &transported_direction)?;
                
                if <T as Float>::abs(trial_directional) <= c2 * <T as Float>::abs(directional_derivative) {
                    return Ok(alpha);
                }
                
                // If curvature condition not satisfied, increase alpha (zoom phase would go here)
                if trial_directional < directional_derivative {
                    alpha = <T as Float>::min(alpha * <T as Scalar>::from_f64(2.0), alpha_max);
                } else {
                    // Curvature condition violated in other direction, reduce alpha
                    alpha *= <T as Scalar>::from_f64(0.5);
                }
            } else {
                // Armijo condition not satisfied, reduce alpha
                alpha *= <T as Scalar>::from_f64(0.5);
            }
            
            if alpha < alpha_min {
                break;
            }
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
        direction: &M::TangentVector,
        step_size: T,
        lbfgs_state: &mut LBFGSState<T, M::TangentVector>,
        _workspace: &mut Workspace<T>,
        iteration: usize,
    ) -> Result<()>
    where
        M: Manifold<T>,
    {
        // Compute s_k = step_size * direction (in tangent space at old_point)
        let mut s_k = direction.clone();
        manifold.scale_tangent(old_point, step_size, direction, &mut s_k)?;
        
        // Transport old gradient to new point's tangent space
        let mut transported_old_grad = old_gradient.clone();
        manifold.parallel_transport(old_point, new_point, old_gradient, &mut transported_old_grad)?;
        
        // Compute y_k = new_gradient - transported_old_grad (in tangent space at new_point)
        let mut y_k = new_gradient.clone();
        let mut neg_transported_old_grad = transported_old_grad.clone();
        manifold.scale_tangent(new_point, -T::one(), &transported_old_grad, &mut neg_transported_old_grad)?;
        let mut temp_add = y_k.clone();
        manifold.add_tangents(new_point, new_gradient, &neg_transported_old_grad, &mut y_k, &mut temp_add)?;
        
        // Transport s_k to new_point for inner product computation
        let mut transported_s_k = s_k.clone();
        manifold.parallel_transport(old_point, new_point, &s_k, &mut transported_s_k)?;
        
        // Compute rho_k = 1 / <y_k, s_k>
        let y_dot_s = manifold.inner_product(new_point, &y_k, &transported_s_k)?;
        
        // Check if update satisfies curvature condition
        let min_curvature = <T as Scalar>::from_f64(1e-8);
        if <T as Float>::abs(y_dot_s) > min_curvature && 
           (!self.config.use_cautious_updates || y_dot_s > T::zero()) {
            
            let rho = T::one() / y_dot_s;
            
            // Add the new entry (s_k stored at old_point, y_k stored at new_point)
            lbfgs_state.add_entry(s_k, y_k, rho, iteration);
        }
        
        Ok(())
    }
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for LBFGS<T> {
    fn name(&self) -> &str {
        "Riemannian L-BFGS"
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
        workspace.preallocate_vector(BufferId::Gradient, n);
        workspace.preallocate_vector(BufferId::Direction, n);
        workspace.preallocate_vector(BufferId::Temp1, n);
        workspace.preallocate_vector(BufferId::Temp2, n);
        
        // Additional buffers for L-BFGS operations
        for i in 0..self.config.memory_size {
            workspace.preallocate_vector(BufferId::Custom((i * 2) as u32), n);
            workspace.preallocate_vector(BufferId::Custom((i * 2 + 1) as u32), n);
        }
        
        // Initialize state
        let initial_cost = cost_fn.cost(initial_point)?;
        let mut lbfgs_state = LBFGSState::new(self.config.memory_size);
        lbfgs_state.use_cautious_updates = self.config.use_cautious_updates;
        
        let mut current_point = initial_point.clone();
        let mut previous_point: Option<M::Point> = None;
        let mut current_cost = initial_cost;
        let mut previous_cost: Option<T> = None;
        let mut gradient_norm: Option<T> = None;
        let mut iteration = 0;
        let mut function_evaluations = 1;
        let mut gradient_evaluations = 0;
        
        // Keep track of point history for transport operations
        let mut point_history: Vec<M::Point> = Vec::new();
        point_history.push(initial_point.clone());
        
        // Compute initial gradient to get the right type
        let mut euclidean_grad = cost_fn.gradient(&initial_point)?;
        let mut riemannian_grad = euclidean_grad.clone();
        gradient_evaluations += 1;
        
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
            )?;
            
            // Compute gradient norm
            let grad_norm_squared = manifold.inner_product(
                &current_point,
                &riemannian_grad,
                &riemannian_grad,
            )?;
            let grad_norm = <T as Float>::sqrt(grad_norm_squared);
            gradient_norm = Some(grad_norm);
            
            // Compute L-BFGS direction
            let mut direction = riemannian_grad.clone();
            self.compute_lbfgs_direction_inplace(
                manifold,
                &current_point,
                &riemannian_grad,
                &lbfgs_state,
                &mut direction,
                &mut workspace,
                &point_history,
            )?;
            
            // Perform line search
            let step_size = self.perform_line_search(
                cost_fn,
                manifold,
                &current_point,
                &direction,
                current_cost,
                &riemannian_grad,
                &mut workspace,
                &mut function_evaluations,
                &mut gradient_evaluations,
            )?;
            
            // Take the step using retraction
            let mut scaled_direction = direction.clone();
            manifold.scale_tangent(
                &current_point,
                step_size,
                &direction,
                &mut scaled_direction,
            )?;
            
            let mut new_point = current_point.clone();
            manifold.retract(
                &current_point,
                &scaled_direction,
                &mut new_point,
            )?;
            
            // Compute gradient at new point for history update
            let mut new_euclidean_grad = euclidean_grad.clone();
            let mut new_riemannian_grad = riemannian_grad.clone();
            
            let new_cost_verify = cost_fn.cost_and_gradient(
                &new_point,
                &mut workspace,
                &mut new_euclidean_grad,
            )?;
            function_evaluations += 1;
            gradient_evaluations += 1;
            
            manifold.euclidean_to_riemannian_gradient(
                &new_point,
                &new_euclidean_grad,
                &mut new_riemannian_grad,
            )?;
            
            // Update L-BFGS history
            if iteration > 0 || previous_point.is_some() {
                self.update_lbfgs_history(
                    manifold,
                    &current_point,
                    &new_point,
                    &riemannian_grad,
                    &new_riemannian_grad,
                    &direction,
                    step_size,
                    &mut lbfgs_state,
                    &mut workspace,
                    iteration,
                )?;
            }
            
            // Update state
            previous_point = Some(current_point.clone());
            current_point = new_point;
            previous_cost = Some(current_cost);
            current_cost = new_cost_verify;
            iteration += 1;
            
            // Update point history
            point_history.push(current_point.clone());
            
            // Keep point history size bounded
            if point_history.len() > self.config.memory_size + 2 {
                point_history.remove(0);
            }
        }
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
        assert_eq!(state.history.len(), 0);
    }
}