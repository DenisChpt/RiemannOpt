//! # Riemannian Stochastic Gradient Descent (SGD)
//!
//! This module implements the Stochastic Gradient Descent optimizer adapted for
//! Riemannian manifolds. SGD is the most fundamental optimization algorithm,
//! here extended to handle the non-Euclidean geometry of manifolds through
//! retraction operations and proper handling of the Riemannian metric.
//!
//! ## Mathematical Foundation
//!
//! Given a smooth cost function f: ℳ → ℝ on a Riemannian manifold (ℳ, g),
//! Riemannian SGD iteratively minimizes f by following steepest descent directions
//! adapted to the manifold geometry.
//!
//! ### Basic Algorithm
//! For k = 0, 1, 2, ...:
//! ```text
//! 1. Compute Riemannian gradient: ξ_k = grad f(x_k)
//! 2. Choose step size: α_k > 0
//! 3. Update: x_{k+1} = R_{x_k}(-α_k ξ_k)
//! ```
//! where R_{x_k} is a retraction at x_k.
//!
//! ### Momentum Variants
//!
//! #### Classical Momentum
//! ```text
//! v_0 = 0
//! v_{k+1} = β v_k + grad f(x_k)
//! x_{k+1} = R_{x_k}(-α_k v_{k+1})
//! ```
//! where β ∈ [0,1) is the momentum coefficient.
//!
//! #### Nesterov Acceleration
//! ```text
//! v_0 = 0
//! y_k = R_{x_k}(β v_k)                    # Lookahead step
//! v_{k+1} = β v_k + grad f(y_k)              # Update momentum
//! x_{k+1} = R_{x_k}(-α_k v_{k+1})           # Take step
//! ```

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
        step_size::StepSizeSchedule,
        line_search::BacktrackingLineSearch,
    },
};
use num_traits::Float;
use std::time::Instant;
use std::fmt::Debug;

/// Momentum method for Riemannian SGD.
///
/// Momentum methods accelerate convergence by incorporating information from
/// previous gradients. On Riemannian manifolds, this requires careful handling
/// of momentum vectors as they move between different tangent spaces.
#[derive(Debug, Clone)]
pub enum MomentumMethod<T: Scalar> {
    /// No momentum - pure gradient descent.
    None,
    
    /// Classical momentum (Heavy Ball method).
    Classical {
        /// Momentum coefficient β ∈ [0,1). Higher values give more momentum.
        coefficient: T,
    },
    
    /// Nesterov Accelerated Gradient (NAG).
    Nesterov {
        /// Momentum coefficient β ∈ [0,1). Should be close to 1 for best acceleration.
        coefficient: T,
    },
}

/// Configuration for the Riemannian SGD optimizer.
///
/// This struct encapsulates all hyperparameters and options for SGD optimization.
/// It provides a builder pattern for easy configuration.
#[derive(Debug, Clone)]
pub struct SGDConfig<T: Scalar> {
    /// Step size schedule controlling how α_k evolves over iterations.
    pub step_size: StepSizeSchedule<T>,
    
    /// Momentum method for acceleration.
    pub momentum: MomentumMethod<T>,
    
    /// Gradient clipping threshold to prevent exploding gradients.
    pub gradient_clip: Option<T>,
    
    /// Line search strategy for adaptive step size selection.
    pub line_search: Option<BacktrackingLineSearch>,
}

impl<T: Scalar> Default for SGDConfig<T> {
    fn default() -> Self {
        Self {
            step_size: StepSizeSchedule::Constant(<T as Scalar>::from_f64(0.01)),
            momentum: MomentumMethod::None,
            gradient_clip: None,
            line_search: None,
        }
    }
}

impl<T: Scalar> SGDConfig<T> {
    /// Creates a new SGD configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Sets the step size schedule.
    pub fn with_step_size(mut self, schedule: StepSizeSchedule<T>) -> Self {
        self.step_size = schedule;
        self
    }
    
    /// Sets constant step size.
    pub fn with_constant_step_size(mut self, step_size: T) -> Self {
        self.step_size = StepSizeSchedule::Constant(step_size);
        self
    }
    
    /// Sets exponential decay schedule.
    pub fn with_exponential_decay(mut self, initial: T, decay_rate: T) -> Self {
        self.step_size = StepSizeSchedule::ExponentialDecay { initial, decay_rate };
        self
    }
    
    /// Sets momentum method.
    pub fn with_momentum(mut self, momentum: MomentumMethod<T>) -> Self {
        self.momentum = momentum;
        self
    }
    
    /// Sets classical momentum.
    pub fn with_classical_momentum(mut self, coefficient: T) -> Self {
        self.momentum = MomentumMethod::Classical { coefficient };
        self
    }
    
    /// Sets Nesterov momentum.
    pub fn with_nesterov_momentum(mut self, coefficient: T) -> Self {
        self.momentum = MomentumMethod::Nesterov { coefficient };
        self
    }
    
    /// Sets gradient clipping threshold.
    pub fn with_gradient_clip(mut self, threshold: T) -> Self {
        self.gradient_clip = Some(threshold);
        self
    }
    
    /// Sets line search strategy.
    pub fn with_line_search(mut self, _max_iterations: usize) -> Self {
        self.line_search = Some(BacktrackingLineSearch::new());
        self
    }
}

/// Internal state for momentum computation
#[derive(Debug)]
pub struct MomentumState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// Current momentum vector
    pub momentum_vector: Option<TV>,
    /// Momentum coefficient
    pub coefficient: Option<T>,
    /// Whether using Nesterov momentum
    pub is_nesterov: bool,
}

impl<T, TV> MomentumState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    fn new(momentum: &MomentumMethod<T>) -> Self {
        let (coefficient, is_nesterov) = match momentum {
            MomentumMethod::None => (None, false),
            MomentumMethod::Classical { coefficient } => (Some(*coefficient), false),
            MomentumMethod::Nesterov { coefficient } => (Some(*coefficient), true),
        };
        
        Self {
            momentum_vector: None,
            coefficient,
            is_nesterov,
        }
    }
}

/// Riemannian Stochastic Gradient Descent optimizer.
///
/// This optimizer implements SGD adapted for Riemannian manifolds, providing
/// the foundation for optimization on curved spaces with proper handling of
/// the manifold geometry through retractions and Riemannian gradients.
#[derive(Debug)]
pub struct SGD<T: Scalar> {
    config: SGDConfig<T>,
}

impl<T: Scalar> SGD<T> {
    /// Creates a new SGD optimizer with the given configuration.
    pub fn new(config: SGDConfig<T>) -> Self {
        Self {
            config,
        }
    }
    
    /// Creates an SGD optimizer with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(SGDConfig::default())
    }
    
    /// Returns the optimizer configuration.
    pub fn config(&self) -> &SGDConfig<T> {
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
    
    /// Computes the search direction based on the momentum configuration (in-place).
    fn compute_momentum_direction_inplace<M>(
        &self,
        manifold: &M,
        current_point: &M::Point,
        previous_point: &Option<M::Point>,
        momentum_state: &mut MomentumState<T, M::TangentVector>,
        gradient: &mut M::TangentVector,
        _workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        M: Manifold<T>,
    {
        if momentum_state.coefficient.is_none() {
            // No momentum - gradient is already the direction
            return Ok(());
        }
        
        let coefficient = momentum_state.coefficient.unwrap();
        
        // Handle momentum
        match &mut momentum_state.momentum_vector {
            None => {
                // First iteration - initialize momentum with gradient
                momentum_state.momentum_vector = Some(gradient.clone());
            }
            Some(momentum) => {
                // Transport momentum from previous point to current point if needed
                if let Some(ref prev_point) = previous_point {
                    // Parallel transport the momentum vector
                    let mut transported_momentum = momentum.clone();
                    manifold.parallel_transport(
                        prev_point,
                        current_point,
                        momentum,
                        &mut transported_momentum,
                    )?;
                    *momentum = transported_momentum;
                }
                
                if momentum_state.is_nesterov {
                    // Nesterov momentum
                    // Update momentum: m = β * m + gradient
                    let mut temp = momentum.clone();
                    let mut temp2 = momentum.clone(); // Additional temp for add_tangents
                    manifold.scale_tangent(current_point, coefficient, momentum, &mut temp)?;
                    manifold.add_tangents(current_point, &temp, gradient, momentum, &mut temp2)?;
                } else {
                    // Classical momentum
                    // Update momentum: m = β * m + (1-β) * gradient
                    let mut temp = momentum.clone();
                    let mut temp2 = momentum.clone(); // Additional temp for add_tangents
                    manifold.scale_tangent(current_point, coefficient, momentum, &mut temp)?;
                    let mut scaled_grad = gradient.clone();
                    manifold.scale_tangent(current_point, T::one() - coefficient, gradient, &mut scaled_grad)?;
                    manifold.add_tangents(current_point, &temp, &scaled_grad, momentum, &mut temp2)?;
                }
                
                // Direction is the momentum
                gradient.clone_from(momentum);
            }
        }
        
        Ok(())
    }
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for SGD<T> {
    fn name(&self) -> &str {
        "Riemannian SGD"
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
        
        if !matches!(self.config.momentum, MomentumMethod::None) {
            workspace.preallocate_vector(BufferId::Momentum, n);
            if matches!(self.config.momentum, MomentumMethod::Nesterov { .. }) {
                workspace.preallocate_vector(BufferId::Temp2, n);
            }
        }
        
        // Initialize state
        let initial_cost = cost_fn.cost(initial_point)?;
        let mut momentum_state = MomentumState::new(&self.config.momentum);
        let mut current_point = initial_point.clone();
        let mut previous_point: Option<M::Point> = None;
        let mut current_cost = initial_cost;
        let mut previous_cost: Option<T> = None;
        let mut gradient_norm: Option<T> = None;
        let mut iteration = 0;
        let mut function_evaluations = 1;
        let mut gradient_evaluations = 0;
        
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
            let new_cost = cost_fn.cost_and_gradient(
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
            
            // Compute search direction with momentum
            self.compute_momentum_direction_inplace(
                manifold,
                &current_point,
                &previous_point,
                &mut momentum_state,
                &mut riemannian_grad,
                &mut workspace,
            )?;
            
            // Determine step size
            let base_step_size = self.config.step_size.get_step_size(iteration);
            
            // Apply gradient clipping if needed
            let step_size = if let Some(threshold) = self.config.gradient_clip {
                if grad_norm > threshold {
                    base_step_size * (threshold / grad_norm)
                } else {
                    base_step_size
                }
            } else {
                base_step_size
            };
            
            // Scale direction by negative step size (descent direction)
            let mut search_direction = riemannian_grad.clone();
            manifold.scale_tangent(
                &current_point,
                -step_size,
                &riemannian_grad,
                &mut search_direction,
            )?;
            
            // TODO: Implement line search if configured
            
            // Take the step using retraction
            let mut new_point = current_point.clone();
            manifold.retract(
                &current_point,
                &search_direction,
                &mut new_point,
            )?;
            
            // Update state
            previous_point = Some(std::mem::replace(&mut current_point, new_point));
            previous_cost = Some(current_cost);
            current_cost = new_cost;
            iteration += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use riemannopt_core::types::DVector;

    #[test]
    fn test_step_size_schedules() {
        // Test constant step size
        let constant = StepSizeSchedule::Constant(0.1);
        assert_relative_eq!(constant.get_step_size(0), 0.1, epsilon = 1e-10);
        assert_relative_eq!(constant.get_step_size(100), 0.1, epsilon = 1e-10);
        
        // Test exponential decay
        let exp_decay = StepSizeSchedule::ExponentialDecay {
            initial: 1.0,
            decay_rate: 0.9,
        };
        assert_relative_eq!(exp_decay.get_step_size(0), 1.0, epsilon = 1e-10);
        assert!(exp_decay.get_step_size(10) < 1.0);
        
        // Test polynomial decay
        let poly_decay = StepSizeSchedule::PolynomialDecay {
            initial: 1.0,
            decay_rate: 1.0,
            power: 1.0,
        };
        assert_relative_eq!(poly_decay.get_step_size(0), 1.0, epsilon = 1e-10);
        assert_relative_eq!(poly_decay.get_step_size(1), 0.5, epsilon = 1e-10);
    }
    
    #[test]
    fn test_sgd_config() {
        // Test config builder
        let config = SGDConfig::<f64>::new()
            .with_constant_step_size(0.01)
            .with_classical_momentum(0.9)
            .with_gradient_clip(1.0);
        
        assert!(matches!(config.step_size, StepSizeSchedule::Constant(v) if v == 0.01));
        assert!(matches!(config.momentum, MomentumMethod::Classical { coefficient } if coefficient == 0.9));
        assert_eq!(config.gradient_clip, Some(1.0));
    }
    
    #[test]
    fn test_momentum_state() {
        // Test state initialization for different momentum types
        type TestTangent = DVector<f64>;
        
        let state_none = MomentumState::<f64, TestTangent>::new(&MomentumMethod::None);
        assert!(state_none.momentum_vector.is_none());
        assert!(!state_none.is_nesterov);
        assert!(state_none.coefficient.is_none());
        
        let state_classical = MomentumState::<f64, TestTangent>::new(&MomentumMethod::Classical { coefficient: 0.9 });
        assert_eq!(state_classical.coefficient, Some(0.9));
        assert!(!state_classical.is_nesterov);
        
        let state_nesterov = MomentumState::<f64, TestTangent>::new(&MomentumMethod::Nesterov { coefficient: 0.9 });
        assert_eq!(state_nesterov.coefficient, Some(0.9));
        assert!(state_nesterov.is_nesterov);
    }
}