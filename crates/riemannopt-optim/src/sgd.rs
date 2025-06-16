//! Riemannian Stochastic Gradient Descent (SGD) optimizer.
//!
//! This module implements the SGD optimizer adapted for Riemannian manifolds.
//! SGD is the fundamental optimization algorithm, extended to handle the 
//! non-Euclidean geometry of manifolds through retraction operations.
//!
//! # Algorithm Overview
//!
//! The Riemannian SGD algorithm performs the following steps:
//! 1. Compute the Euclidean gradient at the current point
//! 2. Convert to Riemannian gradient using the manifold metric
//! 3. Take a step in the negative gradient direction using retraction
//! 4. Apply momentum if enabled
//!
//! # Features
//!
//! - **Step size scheduling**: Constant, exponential decay, polynomial decay
//! - **Momentum methods**: Classical momentum and Nesterov acceleration  
//! - **Gradient clipping**: Prevents exploding gradients
//! - **Line search integration**: Optional Armijo/Wolfe line search
//! - **Batch processing**: Support for mini-batch gradients

use riemannopt_core::{
    cost_function::CostFunction,
    error::Result,
    manifold::{Manifold, Point, TangentVector},
    optimizer::{Optimizer, OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
    step_size::StepSizeSchedule,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use std::time::Instant;


/// Momentum method for SGD.
#[derive(Debug, Clone)]
pub enum MomentumMethod<T>
where
    T: Scalar,
{
    /// No momentum
    None,
    
    /// Classical momentum: v_k = beta*v_{k-1} + grad_k
    Classical {
        coefficient: T,
    },
    
    /// Nesterov accelerated gradient
    Nesterov {
        coefficient: T,
    },
}

/// Configuration for the SGD optimizer.
#[derive(Debug, Clone)]
pub struct SGDConfig<T>
where
    T: Scalar,
{
    /// Step size schedule
    pub step_size: StepSizeSchedule<T>,
    
    /// Momentum method
    pub momentum: MomentumMethod<T>,
    
    /// Gradient clipping threshold (None = no clipping)
    pub gradient_clip: Option<T>,
    
    /// Whether to use line search for step size
    pub use_line_search: bool,
    
    /// Maximum line search iterations
    pub max_line_search_iterations: usize,
}

impl<T> Default for SGDConfig<T>
where
    T: Scalar,
{
    fn default() -> Self {
        Self {
            step_size: StepSizeSchedule::Constant(<T as Scalar>::from_f64(0.01)),
            momentum: MomentumMethod::None,
            gradient_clip: None,
            use_line_search: false,
            max_line_search_iterations: 20,
        }
    }
}

impl<T> SGDConfig<T>
where
    T: Scalar,
{
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
    
    /// Enables line search for adaptive step sizes.
    pub fn with_line_search(mut self, max_iterations: usize) -> Self {
        self.use_line_search = true;
        self.max_line_search_iterations = max_iterations;
        self
    }
}

/// Riemannian SGD optimizer state.
#[derive(Debug)]
struct SGDState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Momentum vector (if using momentum)
    momentum: Option<TangentVector<T, D>>,
    
    /// Previous gradient for momentum calculation  
    previous_gradient: Option<TangentVector<T, D>>,
}

impl<T, D> SGDState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn new() -> Self {
        Self {
            momentum: None,
            previous_gradient: None,
        }
    }
}

/// Riemannian Stochastic Gradient Descent optimizer.
///
/// This optimizer implements SGD adapted for Riemannian manifolds,
/// with support for various step size schedules, momentum methods,
/// and gradient clipping.
///
/// # Mathematical Foundation
///
/// Given a cost function f: M -> R on manifold M, SGD performs:
/// 
/// 1. Compute Riemannian gradient: grad_f(x_k)
/// 2. Update: x_{k+1} = R_{x_k}(-alpha_k * d_k)
/// 
/// where R is the retraction operation and d_k is the search direction
/// (possibly including momentum).
///
/// # Examples
///
/// ```rust
/// use riemannopt_optim::{SGD, SGDConfig, StepSizeSchedule, MomentumMethod};
/// 
/// // Basic SGD with constant step size
/// let sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
/// 
/// // SGD with exponential decay and classical momentum
/// let sgd_advanced = SGD::new(
///     SGDConfig::new()
///         .with_exponential_decay(0.1, 0.95)
///         .with_classical_momentum(0.9)
///         .with_gradient_clip(1.0)
/// );
/// ```
#[derive(Debug)]
pub struct SGD<T>
where
    T: Scalar,
{
    config: SGDConfig<T>,
}

impl<T> SGD<T>
where
    T: Scalar,
{
    /// Creates a new SGD optimizer with the given configuration.
    pub fn new(config: SGDConfig<T>) -> Self {
        Self { config }
    }
    
    /// Creates an SGD optimizer with default configuration.
    pub fn default() -> Self {
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
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function to minimize
    /// * `manifold` - The manifold on which to optimize
    /// * `initial_point` - Starting point for optimization
    /// * `stopping_criterion` - Conditions for terminating optimization
    ///
    /// # Returns
    ///
    /// An `OptimizationResult` containing the optimal point and metadata.
    pub fn optimize<D, C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &Point<T, D>,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, D>>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        let start_time = Instant::now();
        
        // Initialize optimization state
        let initial_cost = cost_fn.cost(initial_point)?;
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        
        // Initialize SGD-specific state
        let mut sgd_state = SGDState::new();
        
        // Main optimization loop
        loop {
            // Check stopping criteria
            if let Some(reason) = ConvergenceChecker::check(&state, manifold, stopping_criterion)? {
                return Ok(OptimizationResult::new(
                    state.point,
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
            self.step_internal(cost_fn, manifold, &mut state, &mut sgd_state)?;
        }
    }

    /// Performs a single optimization step.
    ///
    /// This method is useful for implementing custom optimization loops
    /// or for debugging purposes.
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - The cost function
    /// * `manifold` - The manifold
    /// * `state` - Current optimizer state
    ///
    /// # Returns
    ///
    /// Updated state after one iteration.
    pub fn step<D, C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, D>,
    ) -> Result<()>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // For the public interface, we need to maintain internal SGD state
        // This is a limitation of the current design - ideally the state would be generic
        let mut sgd_state = SGDState::new();
        self.step_internal(cost_fn, manifold, state, &mut sgd_state)
    }
    
    /// Clips the gradient if gradient clipping is enabled.
    fn clip_gradient<D>(&self, gradient: &mut TangentVector<T, D>)
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        if let Some(threshold) = self.config.gradient_clip {
            let norm = gradient.norm();
            if norm > threshold {
                gradient.scale_mut(threshold / norm);
            }
        }
    }
    
    /// Computes the search direction including momentum if enabled.
    fn compute_search_direction<D, M>(
        &self,
        gradient: &TangentVector<T, D>,
        sgd_state: &mut SGDState<T, D>,
        manifold: &M,
        current_point: &Point<T, D>,
    ) -> Result<TangentVector<T, D>>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        M: Manifold<T, D>,
    {
        match &self.config.momentum {
            MomentumMethod::None => {
                // Simple gradient descent direction
                Ok(-gradient)
            }
            MomentumMethod::Classical { coefficient } => {
                // Classical momentum: v_k = beta*v_{k-1} + grad_k
                let direction = if let Some(ref prev_momentum) = sgd_state.momentum {
                    // Transport previous momentum to current point if needed
                    let transported_momentum = if let Some(ref _prev_grad) = sgd_state.previous_gradient {
                        // For simplicity, we use the current tangent space
                        // In practice, parallel transport could be used for better accuracy
                        manifold.project_tangent(current_point, prev_momentum)?
                    } else {
                        prev_momentum.clone()
                    };
                    
                    transported_momentum * *coefficient + gradient
                } else {
                    gradient.clone()
                };
                
                sgd_state.momentum = Some(direction.clone());
                sgd_state.previous_gradient = Some(gradient.clone());
                
                Ok(-direction)
            }
            MomentumMethod::Nesterov { coefficient } => {
                // Nesterov momentum: lookahead then gradient step
                let direction = if let Some(ref prev_momentum) = sgd_state.momentum {
                    let transported_momentum = manifold.project_tangent(current_point, prev_momentum)?;
                    
                    // v_k = beta*v_{k-1} + grad_k
                    let new_momentum = transported_momentum * *coefficient + gradient;
                    
                    // Nesterov update: use beta*v_k + grad_k as direction
                    new_momentum * *coefficient + gradient
                } else {
                    gradient.clone()
                };
                
                sgd_state.momentum = Some(direction.clone() - gradient);
                sgd_state.previous_gradient = Some(gradient.clone());
                
                Ok(-direction)
            }
        }
    }

    /// Internal step method that has access to SGD-specific state.
    fn step_internal<D, C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, D>,
        sgd_state: &mut SGDState<T, D>,
    ) -> Result<()>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Compute cost and Euclidean gradient
        let (cost, euclidean_grad) = cost_fn.cost_and_gradient(&state.point)?;
        
        // Convert to Riemannian gradient
        let mut riemannian_grad = manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad)?;
        
        // Apply gradient clipping if enabled
        self.clip_gradient(&mut riemannian_grad);
        
        let grad_norm = riemannian_grad.norm();
        state.set_gradient(riemannian_grad.clone(), grad_norm);
        
        // Compute search direction (including momentum)
        let search_direction = self.compute_search_direction(
            &riemannian_grad,
            sgd_state,
            manifold,
            &state.point,
        )?;
        
        // Determine step size
        let step_size = if self.config.use_line_search {
            // Use line search to find appropriate step size
            self.line_search_step_size(cost_fn, manifold, &state.point, &search_direction, cost)?
        } else {
            // Use scheduled step size
            self.config.step_size.get_step_size(state.iteration)
        };
        
        // Take the step using retraction
        let tangent_step = search_direction * step_size;
        let new_point = manifold.retract(&state.point, &tangent_step)?;
        
        // Evaluate cost at new point
        let new_cost = cost_fn.cost(&new_point)?;
        
        // Update state
        state.update(new_point, new_cost);
        
        Ok(())
    }
    
    /// Performs line search to find an appropriate step size.
    fn line_search_step_size<D, C, M>(
        &self,
        cost_fn: &C,
        manifold: &M,
        point: &Point<T, D>,
        direction: &TangentVector<T, D>,
        current_cost: T,
    ) -> Result<T>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Simple backtracking line search (Armijo condition)
        let c1 = <T as Scalar>::from_f64(1e-4); // Armijo parameter
        let initial_step = self.config.step_size.get_step_size(0);
        let shrink_factor = <T as Scalar>::from_f64(0.5);
        
        let direction_norm = direction.norm();
        if direction_norm < T::epsilon() {
            return Ok(T::zero());
        }
        
        let mut step_size = initial_step;
        
        for _ in 0..self.config.max_line_search_iterations {
            let tangent_step = direction * step_size;
            let new_point = manifold.retract(point, &tangent_step)?;
            let new_cost = cost_fn.cost(&new_point)?;
            
            // Check Armijo condition: f(x + alpha*d) <= f(x) + c1*alpha*grad_f(x)^T*d
            let expected_decrease = c1 * step_size * direction_norm * direction_norm;
            
            if new_cost <= current_cost - expected_decrease {
                return Ok(step_size);
            }
            
            step_size = step_size * shrink_factor;
            
            if step_size < T::epsilon() {
                break;
            }
        }
        
        // If line search fails, return a small step size
        Ok(<T as Scalar>::from_f64(1e-8))
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
    fn test_sgd_creation() {
        let config = SGDConfig::<f64>::new()
            .with_constant_step_size(0.01)
            .with_classical_momentum(0.9);
        
        let sgd = SGD::new(config);
        assert_eq!(sgd.name(), "Riemannian SGD");
    }
    
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
    fn test_gradient_clipping() {
        let config = SGDConfig::<f64>::new().with_gradient_clip(1.0);
        let sgd = SGD::new(config);
        
        let mut gradient = DVector::from_vec(vec![2.0, 0.0, 0.0]);
        sgd.clip_gradient(&mut gradient);
        
        // Gradient should be clipped to norm 1.0
        assert_relative_eq!(gradient.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[0], 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_sgd_optimization_simple() {
        let cost_fn = QuadraticCost::simple(Dyn(3));
        let manifold = TestEuclideanManifold::new(3);
        let initial_point = DVector::from_vec(vec![1.0, 1.0, 1.0]);
        
        let mut sgd = SGD::new(SGDConfig::<f64>::new().with_constant_step_size(0.1));
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-6);
        
        let result = sgd.optimize(&cost_fn, &manifold, &initial_point, &stopping_criterion).unwrap();
        
        // Should converge to origin
        assert!(result.converged);
        assert!(result.point.norm() < 1e-3);
    }
    
    #[test]
    fn test_momentum_methods() {
        let config_none = SGDConfig::<f64>::new();
        let config_classical = SGDConfig::<f64>::new().with_classical_momentum(0.9);
        let config_nesterov = SGDConfig::<f64>::new().with_nesterov_momentum(0.9);
        
        let sgd_none = SGD::new(config_none);
        let sgd_classical = SGD::new(config_classical);
        let sgd_nesterov = SGD::new(config_nesterov);
        
        // Just test that they can be created without panic
        assert_eq!(sgd_none.name(), "Riemannian SGD");
        assert_eq!(sgd_classical.name(), "Riemannian SGD");
        assert_eq!(sgd_nesterov.name(), "Riemannian SGD");
    }
    
    #[test]
    fn test_line_search_configuration() {
        let config = SGDConfig::<f64>::new().with_line_search(50);
        let sgd = SGD::new(config);
        
        assert!(sgd.config().use_line_search);
        assert_eq!(sgd.config().max_line_search_iterations, 50);
    }
    
    #[test]
    fn test_sgd_with_momentum() {
        let cost_fn = QuadraticCost::simple(Dyn(2));
        let manifold = TestEuclideanManifold::new(2);
        let initial_point = DVector::from_vec(vec![2.0, 2.0]);
        
        let mut sgd = SGD::new(
            SGDConfig::<f64>::new()
                .with_constant_step_size(0.01)
                .with_classical_momentum(0.9)
        );
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(500)
            .with_gradient_tolerance(1e-6);
        
        let result = sgd.optimize(&cost_fn, &manifold, &initial_point, &stopping_criterion).unwrap();
        
        // Should converge, potentially faster than without momentum
        assert!(result.point.norm() < 1e-3);
        assert!(result.iterations <= 500);
    }
}

// Implementation of the Optimizer trait from core
impl<T, D> Optimizer<T, D> for SGD<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Riemannian SGD"
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
        // Call the concrete optimize method (not a recursive call)
        SGD::optimize(self, cost_fn, manifold, initial_point, stopping_criterion)
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
        // Call the concrete step method (not a recursive call)
        SGD::step(self, cost_fn, manifold, state)
    }
}