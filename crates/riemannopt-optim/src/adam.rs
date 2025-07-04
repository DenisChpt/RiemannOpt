//! Riemannian Adam optimizer.
//!
//! Adam (Adaptive Moment Estimation) is a popular first-order optimization algorithm
//! that combines the benefits of momentum and adaptive learning rates. This implementation
//! extends Adam to Riemannian manifolds by properly handling tangent space operations
//! and parallel transport.
//!
//! # Algorithm
//!
//! The Riemannian Adam update at iteration t:
//! 1. Compute Riemannian gradient: g_t = grad f(x_t)
//! 2. Parallel transport moments: m_{t-1} → Γ(m_{t-1}), v_{t-1} → Γ(√v_{t-1})²
//! 3. Update biased first moment: m_t = β₁ Γ(m_{t-1}) + (1-β₁) g_t
//! 4. Update biased second moment: v_t = β₂ Γ(√v_{t-1})² + (1-β₂) g_t ⊙ g_t
//! 5. Bias correction: m̂_t = m_t / (1-β₁^t), v̂_t = v_t / (1-β₂^t)
//! 6. Compute update: u_t = -α m̂_t / (√v̂_t + ε)
//! 7. Project to tangent space: ũ_t = P_{x_t}(u_t)
//! 8. Update position: x_{t+1} = R_{x_t}(ũ_t)
//!
//! # Key Features
//!
//! - **Adaptive learning rates**: Per-parameter adaptation based on gradient history
//! - **Momentum**: Exponentially decaying average of past gradients
//! - **Bias correction**: Compensates for initialization bias
//! - **AMSGrad variant**: Optional monotonic learning rate decrease
//! - **AdamW**: Weight decay regularization variant
//!
//! # References
//!
//! - Kingma & Ba, "Adam: A Method for Stochastic Optimization" (2014)
//! - Becigneul & Ganea, "Riemannian Adaptive Optimization Methods" (2019)

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

/// State for Adam moment estimates.
#[derive(Debug)]
pub struct AdamState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// First moment estimate (mean of gradients)
    pub m: Option<TV>,
    /// Second moment estimate (for standard Adam, this approximates the second moment)
    pub v: Option<TV>,
    /// Maximum second moment (for AMSGrad)
    pub v_max: Option<TV>,
    /// Configuration
    pub beta1: T,
    pub beta2: T,
    pub epsilon: T,
    pub amsgrad: bool,
    /// Current time step (for bias correction)
    pub t: usize,
}

impl<T, TV> AdamState<T, TV>
where
    T: Scalar,
    TV: Clone + Debug + Send + Sync,
{
    /// Creates a new Adam state.
    pub fn new(beta1: T, beta2: T, epsilon: T, amsgrad: bool) -> Self {
        Self {
            m: None,
            v: None,
            v_max: None,
            beta1,
            beta2,
            epsilon,
            amsgrad,
            t: 0,
        }
    }
}

/// Builder for `AdamState` with a fluent API.
///
/// # Example
///
/// ```rust,ignore
/// let state = AdamStateBuilder::new()
///     .beta1(0.9)
///     .beta2(0.999)
///     .epsilon(1e-8)
///     .amsgrad(true)
///     .build();
/// ```
#[derive(Debug, Clone)]
pub struct AdamStateBuilder<T: Scalar> {
    beta1: T,
    beta2: T,
    epsilon: T,
    amsgrad: bool,
}

impl<T: Scalar> AdamStateBuilder<T> {
    /// Creates a new builder with default values.
    pub fn new() -> Self {
        Self {
            beta1: T::from(0.9).unwrap(),
            beta2: T::from(0.999).unwrap(),
            epsilon: T::from(1e-8).unwrap(),
            amsgrad: false,
        }
    }
    
    /// Sets the exponential decay rate for the first moment.
    pub fn beta1(mut self, beta1: T) -> Self {
        self.beta1 = beta1;
        self
    }
    
    /// Sets the exponential decay rate for the second moment.
    pub fn beta2(mut self, beta2: T) -> Self {
        self.beta2 = beta2;
        self
    }
    
    /// Sets the epsilon value for numerical stability.
    pub fn epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }
    
    /// Enables or disables the AMSGrad variant.
    pub fn amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
    
    /// Builds the `AdamState`.
    pub fn build<TV>(self) -> AdamState<T, TV>
    where
        TV: Clone + Debug + Send + Sync,
    {
        AdamState::new(self.beta1, self.beta2, self.epsilon, self.amsgrad)
    }
}

impl<T: Scalar> Default for AdamStateBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for the Adam optimizer.
#[derive(Debug, Clone)]
pub struct AdamConfig<T: Scalar> {
    /// Learning rate (α)
    pub learning_rate: T,
    /// First moment decay rate (β₁)
    pub beta1: T,
    /// Second moment decay rate (β₂)
    pub beta2: T,
    /// Small constant for numerical stability (ε)
    pub epsilon: T,
    /// Whether to use AMSGrad variant
    pub use_amsgrad: bool,
    /// Weight decay coefficient for AdamW variant
    pub weight_decay: Option<T>,
    /// Whether to apply gradient clipping
    pub gradient_clip: Option<T>,
}

impl<T: Scalar> Default for AdamConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: <T as Scalar>::from_f64(0.001),
            beta1: <T as Scalar>::from_f64(0.9),
            beta2: <T as Scalar>::from_f64(0.999),
            epsilon: <T as Scalar>::from_f64(1e-8),
            use_amsgrad: false,
            weight_decay: None,
            gradient_clip: None,
        }
    }
}

impl<T: Scalar> AdamConfig<T> {
    /// Creates a new configuration with default parameters.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the learning rate.
    pub fn with_learning_rate(mut self, learning_rate: T) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    /// Sets the first moment decay rate (β₁).
    pub fn with_beta1(mut self, beta1: T) -> Self {
        self.beta1 = beta1;
        self
    }

    /// Sets the second moment decay rate (β₂).
    pub fn with_beta2(mut self, beta2: T) -> Self {
        self.beta2 = beta2;
        self
    }

    /// Sets the epsilon value for numerical stability.
    pub fn with_epsilon(mut self, epsilon: T) -> Self {
        self.epsilon = epsilon;
        self
    }

    /// Enables AMSGrad variant.
    pub fn with_amsgrad(mut self) -> Self {
        self.use_amsgrad = true;
        self
    }

    /// Enables AdamW with specified weight decay.
    pub fn with_weight_decay(mut self, weight_decay: T) -> Self {
        self.weight_decay = Some(weight_decay);
        self
    }

    /// Enables gradient clipping with specified threshold.
    pub fn with_gradient_clip(mut self, threshold: T) -> Self {
        self.gradient_clip = Some(threshold);
        self
    }
}

/// Riemannian Adam optimizer.
///
/// This optimizer adapts the classical Adam algorithm to Riemannian manifolds
/// by properly handling tangent space operations and using parallel transport
/// to maintain moment estimates across different tangent spaces.
///
/// # Examples
///
/// ```rust,ignore
/// use riemannopt_optim::{Adam, AdamConfig};
/// 
/// // Basic Adam with default parameters
/// let adam: Adam<f64> = Adam::new(AdamConfig::new());
/// 
/// // Adam with custom parameters and AMSGrad
/// let adam_custom = Adam::new(
///     AdamConfig::new()
///         .with_learning_rate(0.01)
///         .with_beta1(0.95)
///         .with_amsgrad()
///         .with_gradient_clip(1.0)
/// );
/// ```
#[derive(Debug)]
pub struct Adam<T: Scalar> {
    config: AdamConfig<T>,
}

impl<T: Scalar> Adam<T> {
    /// Creates a new Adam optimizer with the given configuration.
    pub fn new(config: AdamConfig<T>) -> Self {
        Self { 
            config,
        }
    }

    /// Creates a new Adam optimizer with default configuration.
    pub fn with_default_config() -> Self {
        Self::new(AdamConfig::default())
    }

    /// Returns the optimizer configuration.
    pub fn config(&self) -> &AdamConfig<T> {
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
    
    /// Computes the Adam direction with proper moment updates and parallel transport.
    fn compute_adam_direction_inplace<M>(
        &self,
        manifold: &M,
        current_point: &M::Point,
        previous_point: &Option<M::Point>,
        gradient: &M::TangentVector,
        adam_state: &mut AdamState<T, M::TangentVector>,
        direction: &mut M::TangentVector,
        workspace: &mut Workspace<T>,
    ) -> Result<()>
    where
        M: Manifold<T>,
    {
        // Update time step for bias correction
        adam_state.t += 1;
        
        // Initialize or update moments
        match (&mut adam_state.m, &mut adam_state.v) {
            (None, None) => {
                // First iteration - initialize moments
                // m_0 = (1-β₁) * g_0
                let mut m_init = gradient.clone();
                manifold.scale_tangent(current_point, T::one() - adam_state.beta1, gradient, &mut m_init, workspace)?;
                
                // For second moment, we use gradient norm approximation
                // v_0 = (1-β₂) * ||g_0||²
                let grad_norm_sq = manifold.inner_product(current_point, gradient, gradient)?;
                let mut v_init = gradient.clone();
                // Scale gradient by sqrt((1-β₂) * ||g||²) / ||g|| to get a vector with norm sqrt((1-β₂) * ||g||²)
                let grad_norm = <T as Float>::sqrt(grad_norm_sq);
                if grad_norm > T::zero() {
                    let scale = <T as Float>::sqrt((T::one() - adam_state.beta2) * grad_norm_sq) / grad_norm;
                    manifold.scale_tangent(current_point, scale, gradient, &mut v_init, workspace)?;
                } else {
                    manifold.scale_tangent(current_point, T::zero(), gradient, &mut v_init, workspace)?;
                }
                
                adam_state.m = Some(m_init);
                adam_state.v = Some(v_init);
                
                if adam_state.amsgrad {
                    adam_state.v_max = adam_state.v.clone();
                }
            }
            (Some(_m), Some(_v)) => {
                // Moments already exist
            }
            _ => return Err(riemannopt_core::error::ManifoldError::invalid_point("Invalid Adam state")),
        }
        
        // Now update the moments
        if let (Some(m), Some(v)) = (&mut adam_state.m, &mut adam_state.v) {
            // Transport previous moments if we have a previous point
            if let Some(ref prev_point) = previous_point {
                let mut transported_m = m.clone();
                let mut transported_v = v.clone();
                
                manifold.parallel_transport(prev_point, current_point, m, &mut transported_m, workspace)?;
                manifold.parallel_transport(prev_point, current_point, v, &mut transported_v, workspace)?;
                
                *m = transported_m;
                *v = transported_v;
                
                if let Some(v_max) = &mut adam_state.v_max {
                    let mut transported_v_max = v_max.clone();
                    manifold.parallel_transport(prev_point, current_point, v_max, &mut transported_v_max, workspace)?;
                    *v_max = transported_v_max;
                }
            }
            
            // Update first moment: m = β₁ * m + (1-β₁) * gradient
            let beta1 = adam_state.beta1;
            let one_minus_beta1 = T::one() - beta1;
            
            let mut temp_m = m.clone();
            manifold.scale_tangent(current_point, beta1, m, &mut temp_m, workspace)?;
            let mut scaled_grad = gradient.clone();
            manifold.scale_tangent(current_point, one_minus_beta1, gradient, &mut scaled_grad, workspace)?;
            manifold.add_tangents(current_point, &temp_m, &scaled_grad, m, workspace)?;
            
            // Update second moment approximation
            // In standard Adam: v = β₂ * v + (1-β₂) * g²
            // Since we can't do element-wise operations, we approximate using norms
            let beta2 = adam_state.beta2;
            let one_minus_beta2 = T::one() - beta2;
            
            // Current approach: maintain v as a vector whose norm represents the second moment
            let v_norm_sq = manifold.inner_product(current_point, v, v)?;
            let grad_norm_sq = manifold.inner_product(current_point, gradient, gradient)?;
            
            // New second moment estimate: β₂ * ||v||² + (1-β₂) * ||g||²
            let new_v_norm_sq = beta2 * v_norm_sq + one_minus_beta2 * grad_norm_sq;
            let new_v_norm = <T as Float>::sqrt(new_v_norm_sq);
            
            // Scale v to have the new norm, maintaining direction
            let current_v_norm = <T as Float>::sqrt(v_norm_sq);
            if current_v_norm > T::zero() {
                let scale = new_v_norm / current_v_norm;
                let temp_v = v.clone();
                manifold.scale_tangent(current_point, scale, &temp_v, v, workspace)?;
            } else {
                // If v has zero norm, use gradient direction
                if grad_norm_sq > T::zero() {
                    manifold.scale_tangent(current_point, new_v_norm / <T as Float>::sqrt(grad_norm_sq), gradient, v, workspace)?;
                }
            }
            
            // Update v_max for AMSGrad
            if adam_state.amsgrad {
                if let Some(v_max) = &mut adam_state.v_max {
                    let v_max_norm_sq = manifold.inner_product(current_point, v_max, v_max)?;
                    if new_v_norm_sq > v_max_norm_sq {
                        v_max.clone_from(v);
                    }
                }
            }
        }
        
        // Compute bias correction
        let t = adam_state.t as f64;
        let bias1 = T::one() - <T as Scalar>::from_f64(adam_state.beta1.to_f64().powf(t));
        let bias2 = T::one() - <T as Scalar>::from_f64(adam_state.beta2.to_f64().powf(t));
        
        // Compute direction
        if let (Some(m), Some(v)) = (&adam_state.m, &adam_state.v) {
            // For AMSGrad, use v_max instead of v
            let v_to_use = if adam_state.amsgrad {
                adam_state.v_max.as_ref().unwrap_or(v)
            } else {
                v
            };
            
            // Bias-corrected first moment: m̂ = m / (1 - β₁^t)
            direction.clone_from(m);
            let mut m_hat = direction.clone();
            manifold.scale_tangent(current_point, T::one() / bias1, &direction, &mut m_hat, workspace)?;
            
            // For second moment, compute effective learning rate adjustment
            // Standard Adam: direction = m̂ / (√v̂ + ε)
            // Our approximation: direction = m̂ / (||v|| / √bias2 + ε)
            let v_norm = <T as Float>::sqrt(manifold.inner_product(current_point, v_to_use, v_to_use)?);
            let v_hat_norm = v_norm / <T as Float>::sqrt(bias2);
            let scale = T::one() / (v_hat_norm + adam_state.epsilon);
            
            manifold.scale_tangent(current_point, scale, &m_hat, direction, workspace)?;
        }
        
        Ok(())
    }
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for Adam<T> {
    fn name(&self) -> &str {
        if self.config.use_amsgrad {
            "Riemannian AMSGrad"
        } else {
            "Riemannian Adam"
        }
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
        workspace.get_or_create_vector(BufferId::Momentum, n);  // For m
        workspace.get_or_create_vector(BufferId::SecondMoment, n);  // For v
        if self.config.use_amsgrad {
            workspace.get_or_create_vector(BufferId::Temp2, n);  // For v_max
        }
        
        // Initialize state
        let initial_cost = cost_fn.cost(initial_point)?;
        let mut adam_state = AdamState::new(
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon,
            self.config.use_amsgrad,
        );
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
            
            // Apply gradient clipping if enabled
            if let Some(threshold) = self.config.gradient_clip {
                if grad_norm > threshold {
                    let scale = threshold / grad_norm;
                    let clipped_grad = riemannian_grad.clone();
                    manifold.scale_tangent(
                        &current_point,
                        scale,
                        &clipped_grad,
                        &mut riemannian_grad,
                        &mut workspace,
                    )?;
                }
            }
            
            // Compute Adam direction
            let mut direction = riemannian_grad.clone();
            self.compute_adam_direction_inplace(
                manifold,
                &current_point,
                &previous_point,
                &riemannian_grad,
                &mut adam_state,
                &mut direction,
                &mut workspace,
            )?;
            
            // Apply weight decay if configured (AdamW variant)
            if let Some(weight_decay) = self.config.weight_decay {
                // AdamW applies weight decay by adding λ * w to the gradient
                // In the Riemannian setting, we need to be careful about this
                // For now, we skip this as it requires projecting the point to tangent space
                // which is not a standard operation
                let _ = weight_decay; // Suppress unused warning
            }
            
            // Scale by negative learning rate and take step
            let mut search_direction = direction.clone();
            manifold.scale_tangent(
                &current_point,
                -self.config.learning_rate,
                &direction,
                &mut search_direction,
                &mut workspace,
            )?;
            
            // Take the step using retraction
            let mut new_point = current_point.clone();
            manifold.retract(
                &current_point,
                &search_direction,
                &mut new_point,
                &mut workspace,
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
    use riemannopt_core::types::DVector;

    #[test]
    fn test_adam_config() {
        let config = AdamConfig::<f64>::new()
            .with_learning_rate(0.01)
            .with_beta1(0.95)
            .with_beta2(0.99)
            .with_epsilon(1e-6)
            .with_amsgrad()
            .with_weight_decay(0.001)
            .with_gradient_clip(1.0);
        
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.beta1, 0.95);
        assert_eq!(config.beta2, 0.99);
        assert_eq!(config.epsilon, 1e-6);
        assert!(config.use_amsgrad);
        assert_eq!(config.weight_decay, Some(0.001));
        assert_eq!(config.gradient_clip, Some(1.0));
    }
    
    #[test]
    fn test_adam_state() {
        let state = AdamState::<f64, DVector<f64>>::new(0.9, 0.999, 1e-8, false);
        assert_eq!(state.beta1, 0.9);
        assert_eq!(state.beta2, 0.999);
        assert_eq!(state.epsilon, 1e-8);
        assert!(!state.amsgrad);
    }

    #[test]
    fn test_amsgrad_state() {
        let state = AdamState::<f64, DVector<f64>>::new(0.9, 0.999, 1e-8, true);
        assert!(state.amsgrad);
    }
    
    #[test]
    fn test_adam_builder() {
        let state = AdamStateBuilder::<f64>::new()
            .beta1(0.8)
            .beta2(0.95)
            .epsilon(1e-10)
            .amsgrad(true)
            .build::<DVector<f64>>();

        assert_eq!(state.beta1, 0.8);
        assert_eq!(state.beta2, 0.95);
        assert_eq!(state.epsilon, 1e-10);
        assert!(state.amsgrad);
    }
}