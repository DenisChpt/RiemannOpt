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
        optimizer::{Optimizer, OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
        step_size::StepSizeSchedule,
        line_search::BacktrackingLineSearch,
    },
};
use num_traits::Float;
use std::time::Instant;
use std::fmt::Debug;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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

/// Internal state for SGD optimizer
#[derive(Debug)]
struct SGDInternalState<T, P, TV> 
where
    T: Scalar,
    P: Clone + Debug + Send + Sync,
    TV: Clone + Debug + Send + Sync,
{
    workspace: Workspace<T>,
    iteration: usize,
    // Store previous point for parallel transport in momentum methods
    previous_point: Option<P>,
    // Store momentum vector
    momentum_vector: Option<TV>,
    momentum_coefficient: Option<T>,
    is_nesterov: bool,
}

impl<T, P, TV> SGDInternalState<T, P, TV>
where
    T: Scalar,
    P: Clone + Debug + Send + Sync,
    TV: Clone + Debug + Send + Sync,
{
    fn new(n: usize, momentum: &MomentumMethod<T>) -> Self {
        let mut workspace = Workspace::with_size(n);
        
        // Pre-allocate workspace buffers
        workspace.get_or_create_vector(BufferId::Gradient, n);
        workspace.get_or_create_vector(BufferId::Direction, n);
        workspace.get_or_create_vector(BufferId::Temp1, n);
        
        if !matches!(momentum, MomentumMethod::None) {
            workspace.get_or_create_vector(BufferId::Momentum, n);
            if matches!(momentum, MomentumMethod::Nesterov { .. }) {
                workspace.get_or_create_vector(BufferId::Temp2, n);
            }
        }
        
        let (momentum_coefficient, is_nesterov) = match momentum {
            MomentumMethod::None => (None, false),
            MomentumMethod::Classical { coefficient } => (Some(*coefficient), false),
            MomentumMethod::Nesterov { coefficient } => (Some(*coefficient), true),
        };
        
        Self {
            workspace,
            iteration: 0,
            previous_point: None,
            momentum_vector: None,
            momentum_coefficient,
            is_nesterov,
        }
    }
    
    #[allow(dead_code)]
    fn workspace_mut(&mut self) -> &mut Workspace<T> {
        &mut self.workspace
    }
    
    fn update_iteration(&mut self) {
        self.iteration += 1;
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
    
    /// Returns the optimizer name.
    pub fn name(&self) -> &str {
        "Riemannian SGD"
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
        let mut internal_state = SGDInternalState::<T, M::Point, M::TangentVector>::new(n, &self.config.momentum);
        
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
        internal_state: &mut SGDInternalState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>,
    {
        // Create gradient buffers - in practice these would be reused from the state
        let mut euclidean_grad = match &state.gradient {
            Some(g) => g.clone(),
            None => {
                // For the first iteration, compute the gradient
                cost_fn.gradient(&state.point)?
            }
        };
        let mut riemannian_grad = euclidean_grad.clone();
        let mut new_point = state.point.clone();
        
        // Get iteration before borrowing workspace
        let iteration = internal_state.iteration;
        
        // Compute gradient at current point (in-place)
        let workspace = &mut internal_state.workspace;
        let _cost = cost_fn.cost_and_gradient(&state.point, workspace, &mut euclidean_grad)?;
        state.function_evaluations += 1;
        state.gradient_evaluations += 1;
        
        // Convert to Riemannian gradient (in-place)
        manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad, &mut riemannian_grad, workspace)?;
        
        // Compute gradient norm
        let grad_norm_squared = manifold.inner_product(&state.point, &riemannian_grad, &riemannian_grad)?;
        let grad_norm = <T as Float>::sqrt(grad_norm_squared);
        state.gradient_norm = Some(grad_norm);
        
        // Store gradient in state for convergence checking
        state.gradient = Some(riemannian_grad.clone());
        
        // Compute search direction based on momentum configuration
        self.compute_momentum_direction_inplace(
            manifold,
            &state.point,
            &mut riemannian_grad,
            internal_state,
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
        let workspace = &mut internal_state.workspace;
        manifold.scale_tangent(&state.point, -step_size, &riemannian_grad, &mut search_direction, workspace)?;
        
        // Perform line search if configured
        if let Some(ref _line_search) = self.config.line_search {
            // TODO: Implement line search with new API
        }
        
        // Take the step using retraction
        manifold.retract(&state.point, &search_direction, &mut new_point, workspace)?;
        
        // Update internal state for momentum
        if internal_state.momentum_coefficient.is_some() {
            internal_state.previous_point = Some(state.point.clone());
        }
        
        // Update state
        state.point = new_point;
        state.value = cost_fn.cost(&state.point)?;
        state.function_evaluations += 1;
        state.iteration += 1;
        
        // Update optimizer state iteration count
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
        // Create internal state if this is the first call
        let n = manifold.dimension();
        let mut internal_state = SGDInternalState::<T, M::Point, M::TangentVector>::new(n, &self.config.momentum);
        
        // Delegate to the internal implementation
        self.step_with_state(cost_fn, manifold, state, &mut internal_state)
    }
    
    
    /// Computes the search direction based on the momentum configuration (in-place).
    fn compute_momentum_direction_inplace<M>(
        &self,
        manifold: &M,
        point: &M::Point,
        gradient: &mut M::TangentVector,
        internal_state: &mut SGDInternalState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        M: Manifold<T>,
    {
        if internal_state.momentum_coefficient.is_none() {
            // No momentum - gradient is already the direction
            return Ok(());
        }
        
        let coefficient = internal_state.momentum_coefficient.unwrap();
        let workspace = &mut internal_state.workspace;
        
        // Handle momentum
        match &mut internal_state.momentum_vector {
            None => {
                // First iteration - initialize momentum with gradient
                internal_state.momentum_vector = Some(gradient.clone());
            }
            Some(momentum) => {
                // Transport momentum from previous point to current point if needed
                if let Some(prev_point) = &internal_state.previous_point {
                    // Parallel transport the momentum vector
                    let mut transported_momentum = momentum.clone();
                    manifold.parallel_transport(
                        prev_point,
                        point,
                        momentum,
                        &mut transported_momentum,
                        workspace,
                    )?;
                    *momentum = transported_momentum;
                }
                
                if internal_state.is_nesterov {
                    // Nesterov momentum
                    // For simplified version, we compute gradient at current point
                    // TODO: Implement full Nesterov with lookahead gradient
                    
                    // Update momentum: m = β * m + gradient
                    let mut temp = momentum.clone();
                    manifold.scale_tangent(point, coefficient, momentum, &mut temp, workspace)?;
                    manifold.add_tangents(point, &temp, gradient, momentum, workspace)?;
                } else {
                    // Classical momentum
                    // Update momentum: m = β * m + (1-β) * gradient
                    let mut temp = momentum.clone();
                    manifold.scale_tangent(point, coefficient, momentum, &mut temp, workspace)?;
                    let mut scaled_grad = gradient.clone();
                    manifold.scale_tangent(point, T::one() - coefficient, gradient, &mut scaled_grad, workspace)?;
                    manifold.add_tangents(point, &temp, &scaled_grad, momentum, workspace)?;
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
    fn test_sgd_internal_state() {
        // Test state initialization for different momentum types
        type TestPoint = DVector<f64>;
        type TestTangent = DVector<f64>;
        
        let state_none = SGDInternalState::<f64, TestPoint, TestTangent>::new(10, &MomentumMethod::None);
        assert!(state_none.momentum_vector.is_none());
        assert!(!state_none.is_nesterov);
        
        let state_classical = SGDInternalState::<f64, TestPoint, TestTangent>::new(10, &MomentumMethod::Classical { coefficient: 0.9 });
        assert_eq!(state_classical.momentum_coefficient, Some(0.9));
        assert!(!state_classical.is_nesterov);
        
        let state_nesterov = SGDInternalState::<f64, TestPoint, TestTangent>::new(10, &MomentumMethod::Nesterov { coefficient: 0.9 });
        assert_eq!(state_nesterov.momentum_coefficient, Some(0.9));
        assert!(state_nesterov.is_nesterov);
    }
}