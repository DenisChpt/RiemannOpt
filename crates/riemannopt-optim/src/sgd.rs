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
    memory::workspace::Workspace,
    optimization::{
        optimizer::{Optimizer, OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
        optimizer_state::{OptimizerStateData, OptimizerStateWithData},
        step_size::StepSizeSchedule,
        line_search::BacktrackingLineSearch,
    },
};
use num_traits::Float;
use std::marker::PhantomData;
use std::time::Instant;
use std::fmt::Debug;
use std::collections::HashMap;
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

/// Riemannian Stochastic Gradient Descent optimizer.
///
/// This optimizer implements SGD adapted for Riemannian manifolds, providing
/// the foundation for optimization on curved spaces with proper handling of
/// the manifold geometry through retractions and Riemannian gradients.
#[derive(Debug)]
pub struct SGD<T: Scalar, M: Manifold<T>> {
    config: SGDConfig<T>,
    state: Option<OptimizerStateWithData<T, M::Point, M::TangentVector>>,
    _phantom: PhantomData<M>,
}

impl<T: Scalar, M: Manifold<T>> SGD<T, M> 
where
    M::TangentVector: 'static,
    M::Point: 'static,
{
    /// Creates a new SGD optimizer with the given configuration.
    pub fn new(config: SGDConfig<T>) -> Self {
        Self { 
            config,
            state: None,
            _phantom: PhantomData,
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
    
    /// Initializes the optimizer state if needed.
    fn ensure_state_initialized(&mut self, manifold: &M) {
        if self.state.is_none() {
            let n = manifold.dimension();
            let workspace = Workspace::with_size(n);
            
            // Create state data based on momentum configuration
            let state_data: Box<dyn OptimizerStateData<T, M::TangentVector>> = 
                match &self.config.momentum {
                    MomentumMethod::None => {
                        Box::new(DummyState::<T, M::TangentVector>::new())
                    }
                    MomentumMethod::Classical { coefficient } => {
                        Box::new(MomentumState::<T, M::TangentVector>::new(*coefficient, false))
                    }
                    MomentumMethod::Nesterov { coefficient } => {
                        Box::new(MomentumState::<T, M::TangentVector>::new(*coefficient, true))
                    }
                };
            
            self.state = Some(OptimizerStateWithData::new(workspace, state_data));
        }
    }
    
    /// Optimizes the given cost function on the manifold.
    pub fn optimize<C>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &M::Point,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, M::Point>>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
    {
        let start_time = Instant::now();
        
        // Ensure state is initialized
        self.ensure_state_initialized(manifold);
        
        // Initialize optimization state
        let initial_cost = cost_fn.cost(initial_point)?;
        let mut state = OptimizerState::new(
            initial_point.clone(),
            initial_cost
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
            self.step(cost_fn, manifold, &mut state)?;
        }
    }

    /// Performs a single optimization step.
    pub fn step<C>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
    {
        // Ensure state is initialized
        self.ensure_state_initialized(manifold);
        
        // Compute gradient at current point
        let gradient = cost_fn.gradient(&state.point)?;
        state.gradient_evaluations += 1;
        
        // Compute gradient norm and store in state
        let grad_norm_squared = manifold.inner_product(&state.point, &gradient, &gradient)?;
        let grad_norm = <T as Float>::sqrt(grad_norm_squared);
        state.gradient_norm = Some(grad_norm);
        
        // Store gradient in state for convergence checking
        state.gradient = Some(gradient.clone());
        
        // Compute search direction based on momentum configuration
        let direction = self.compute_momentum_direction(
            manifold,
            &state.point,
            &gradient,
        )?;
        
        // Determine step size
        let base_step_size = self.config.step_size.get_step_size(state.iteration);
        
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
        let mut scaled_direction = direction.clone();
        {
            let workspace = self.state.as_mut().unwrap().workspace_mut();
            manifold.scale_tangent(
                &state.point,
                -step_size,
                &direction,
                &mut scaled_direction,
                workspace,
            )?;
        }
        
        // Perform line search if configured
        let final_step_size = if let Some(ref _line_search) = self.config.line_search {
            // Use a static method to avoid borrowing conflicts
            Self::perform_line_search_static(
                cost_fn,
                manifold,
                &state.point,
                &scaled_direction,
                state.value,
                grad_norm,
                step_size,
                self.state.as_mut().unwrap().workspace_mut(),
            )?
        } else {
            step_size
        };
        
        // Take the step using retraction
        let mut new_point = state.point.clone();
        {
            let workspace = self.state.as_mut().unwrap().workspace_mut();
            if final_step_size != step_size {
                // Re-scale direction if line search changed step size
                manifold.scale_tangent(
                    &state.point,
                    -final_step_size,
                    &direction,
                    &mut scaled_direction,
                    workspace,
                )?;
            }
            manifold.retract(&state.point, &scaled_direction, &mut new_point, workspace)?;
        }
        
        // Evaluate cost at new point
        let new_cost = cost_fn.cost(&new_point)?;
        state.function_evaluations += 1;
        
        // Update state
        state.update(new_point, new_cost);
        
        // Update optimizer state iteration count
        if let Some(opt_state) = self.state.as_mut() {
            opt_state.update_iteration();
        }
        
        Ok(())
    }
    
    /// Performs line search to find optimal step size.
    #[allow(dead_code)]
    fn perform_line_search<C>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        direction: &M::TangentVector,
        current_cost: T,
        grad_norm: T,
        initial_step_size: T,
        workspace: &mut Workspace<T>,
    ) -> Result<T>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
    {
        let _line_search = self.config.line_search.as_ref().unwrap();
        let mut step_size = initial_step_size;
        let armijo_constant = <T as Scalar>::from_f64(0.0001);
        
        // Compute expected decrease
        let expected_decrease = -grad_norm * grad_norm * step_size * armijo_constant;
        
        // BacktrackingLineSearch doesn't have exposed parameters, use default max iterations
        let max_iterations = 20;
        for _ in 0..max_iterations {
            // Try the step
            let mut trial_point = point.clone();
            manifold.retract(point, direction, &mut trial_point, workspace)?;
            
            // Evaluate cost
            let trial_cost = cost_fn.cost(&trial_point)?;
            
            // Check Armijo condition
            if trial_cost <= current_cost + expected_decrease {
                return Ok(step_size);
            }
            
            // Backtrack with default contraction factor
            step_size *= <T as Scalar>::from_f64(0.5);
        }
        
        // Return the final step size even if Armijo condition not met
        Ok(step_size)
    }
    
    /// Static version of perform_line_search to avoid borrowing conflicts.
    fn perform_line_search_static<C>(
        cost_fn: &C,
        manifold: &M,
        point: &M::Point,
        direction: &M::TangentVector,
        current_cost: T,
        grad_norm: T,
        initial_step_size: T,
        workspace: &mut Workspace<T>,
    ) -> Result<T>
    where
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
    {
        let mut step_size = initial_step_size;
        let armijo_constant = <T as Scalar>::from_f64(0.0001);
        
        // Compute expected decrease
        let expected_decrease = -grad_norm * grad_norm * step_size * armijo_constant;
        
        // BacktrackingLineSearch doesn't have exposed parameters, use default max iterations
        let max_iterations = 20;
        for _ in 0..max_iterations {
            // Try the step
            let mut trial_point = point.clone();
            manifold.retract(point, direction, &mut trial_point, workspace)?;
            
            // Evaluate cost
            let trial_cost = cost_fn.cost(&trial_point)?;
            
            // Check Armijo condition
            if trial_cost <= current_cost + expected_decrease {
                return Ok(step_size);
            }
            
            // Backtrack with default contraction factor
            step_size *= <T as Scalar>::from_f64(0.5);
        }
        
        // Return the final step size even if Armijo condition not met
        Ok(step_size)
    }
    
    /// Computes the search direction based on the momentum configuration.
    fn compute_momentum_direction(
        &mut self,
        manifold: &M,
        point: &M::Point,
        gradient: &M::TangentVector,
    ) -> Result<M::TangentVector> {
        match &self.config.momentum {
            MomentumMethod::None => {
                // Pure gradient descent
                Ok(gradient.clone())
            }
            MomentumMethod::Classical { coefficient } | MomentumMethod::Nesterov { coefficient } => {
                // For Phase 3, implement simplified momentum
                // TODO: Implement full momentum with state tracking and parallel transport
                
                // For now, apply simple exponential smoothing
                let mut direction = gradient.clone();
                
                // In a full implementation, we would:
                // 1. Access the MomentumState from self.state
                // 2. Transport previous momentum to current tangent space
                // 3. Update momentum: m = β * transported_m + (1-β) * gradient
                // 4. For Nesterov: compute lookahead point and gradient
                
                // Apply simple scaling as placeholder
                let workspace = self.state.as_mut().unwrap().workspace_mut();
                manifold.scale_tangent(
                    point,
                    T::one() - *coefficient * <T as Scalar>::from_f64(0.5),
                    &gradient,
                    &mut direction,
                    workspace,
                )?;
                
                Ok(direction)
            }
        }
    }
}

// Implementation of the Optimizer trait from core
// Note: We implement for a specific manifold type M
impl<T: Scalar, M: Manifold<T>> Optimizer<T> for SGD<T, M> {
    fn name(&self) -> &str {
        "Riemannian SGD"
    }
    
    fn optimize<C, MF>(
        &mut self,
        _cost_fn: &C,
        _manifold: &MF,
        _initial_point: &MF::Point,
        _stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, MF::Point>>
    where
        C: CostFunction<T, Point = MF::Point, TangentVector = MF::TangentVector>,
        MF: Manifold<T>,
    {
        // This is a limitation of the current design - we can't easily handle
        // different manifold types M and MF. For now, we assume they're compatible.
        // A proper solution would require redesigning the trait or using dynamic dispatch.
        
        // Create a new typed optimizer for the specific manifold
        // This is not ideal but works around the type system limitations
        Err(riemannopt_core::error::ManifoldError::not_implemented(
            "Generic manifold optimization not yet implemented"
        ))
    }
    
    fn step<C, MF>(
        &mut self,
        _cost_fn: &C,
        _manifold: &MF,
        _state: &mut OptimizerState<T, MF::Point, MF::TangentVector>,
    ) -> Result<()>
    where
        C: CostFunction<T, Point = MF::Point, TangentVector = MF::TangentVector>,
        MF: Manifold<T>,
    {
        // Similar limitation as optimize - we need the manifold types to match
        Err(riemannopt_core::error::ManifoldError::not_implemented(
            "Generic manifold step not yet implemented"
        ))
    }
}

/// Dummy state for SGD without momentum
#[derive(Debug, Clone)]
struct DummyState<T: Scalar, TV> {
    _phantom: PhantomData<(T, TV)>,
}

impl<T: Scalar, TV> DummyState<T, TV> {
    fn new() -> Self {
        Self { _phantom: PhantomData }
    }
}

impl<T: Scalar, TV: Clone + Debug + Send + Sync + 'static> OptimizerStateData<T, TV> for DummyState<T, TV> {
    fn clone_box(&self) -> Box<dyn OptimizerStateData<T, TV>> {
        Box::new(self.clone())
    }
    
    fn optimizer_name(&self) -> &str {
        "SGD"
    }
    
    fn reset(&mut self) {}
    
    fn summary(&self) -> HashMap<String, String> {
        HashMap::new()
    }
    
    fn update_iteration(&mut self, _iteration: usize) {}
}

/// State for gradient descent with momentum.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct MomentumState<T, TV>
where
    T: Scalar,
{
    /// Momentum vector
    pub momentum: Option<TV>,

    /// Momentum coefficient (typically 0.9)
    pub beta: T,

    /// Whether to use Nesterov acceleration
    pub nesterov: bool,
}

impl<T, TV> MomentumState<T, TV>
where
    T: Scalar,
    TV: Clone,
{
    /// Creates a new momentum state.
    pub fn new(beta: T, nesterov: bool) -> Self {
        Self {
            momentum: None,
            beta,
            nesterov,
        }
    }

    /// Updates the momentum vector.
    pub fn update_momentum(&mut self, gradient: &TV) 
    where
        TV: std::ops::MulAssign<T> + for<'a> std::ops::AddAssign<TV>,
    {
        match &mut self.momentum {
            Some(m) => {
                // m = beta * m + (1 - beta) * gradient
                *m *= self.beta;
                let mut grad_scaled = gradient.clone();
                grad_scaled *= T::one() - self.beta;
                *m += grad_scaled;
            }
            None => {
                self.momentum = Some(gradient.clone());
            }
        }
    }

    /// Gets the search direction based on the current momentum.
    pub fn get_direction(&self, gradient: &TV) -> TV 
    where
        TV: std::ops::Add<Output = TV> + std::ops::Mul<T, Output = TV>,
    {
        match (&self.momentum, self.nesterov) {
            (Some(m), true) => {
                // Nesterov: direction = gradient + beta * momentum
                gradient.clone() + m.clone() * self.beta
            }
            (Some(m), false) => {
                // Classical momentum: use momentum directly
                m.clone()
            }
            (None, _) => gradient.clone(),
        }
    }

    /// Gets a reference to the search direction to avoid cloning when possible.
    pub fn get_direction_ref<'a>(&'a self, gradient: &'a TV) -> Option<&'a TV> {
        match (&self.momentum, self.nesterov) {
            (Some(m), false) => Some(m),
            (None, _) => Some(gradient),
            _ => None, // Nesterov requires computation
        }
    }
}

impl<T, TV> OptimizerStateData<T, TV> for MomentumState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync + 'static,
{
    fn clone_box(&self) -> Box<dyn OptimizerStateData<T, TV>> {
        Box::new(self.clone())
    }
    
    fn optimizer_name(&self) -> &str {
        if self.nesterov {
            "Nesterov Momentum"
        } else {
            "Classical Momentum"
        }
    }

    fn reset(&mut self) {
        self.momentum = None;
    }

    fn summary(&self) -> HashMap<String, String> {
        let mut summary = HashMap::new();
        summary.insert("beta".to_string(), format!("{}", self.beta));
        summary.insert("nesterov".to_string(), self.nesterov.to_string());
        summary.insert(
            "has_momentum".to_string(),
            self.momentum.is_some().to_string(),
        );
        summary
    }

    fn update_iteration(&mut self, _iteration: usize) {
        // Momentum doesn't change with iteration by default
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use riemannopt_core::types::DVector;

    // Tests are temporarily disabled until test manifolds are available
    // TODO: Re-enable tests once test infrastructure is in place
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
    fn test_momentum_state() {
        let mut state = MomentumState::<f64, DVector<f64>>::new(0.9, false);
        assert_eq!(state.optimizer_name(), "Classical Momentum");

        let gradient = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        state.update_momentum(&gradient);

        assert!(state.momentum.is_some());
        let summary = state.summary();
        assert_eq!(summary.get("beta").unwrap(), "0.9");
        assert_eq!(summary.get("has_momentum").unwrap(), "true");

        state.reset();
        assert!(state.momentum.is_none());
    }

    #[test]
    fn test_nesterov_momentum() {
        let mut state = MomentumState::<f64, DVector<f64>>::new(0.9, true);
        assert_eq!(state.optimizer_name(), "Nesterov Momentum");

        let gradient = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        state.update_momentum(&gradient);

        assert!(state.momentum.is_some());
        let summary = state.summary();
        assert_eq!(summary.get("beta").unwrap(), "0.9");
        assert_eq!(summary.get("nesterov").unwrap(), "true");
    }
}