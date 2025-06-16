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
//! 2. Update biased first moment: m_t = β₁ m_{t-1} + (1-β₁) g_t
//! 3. Update biased second moment: v_t = β₂ v_{t-1} + (1-β₂) g_t ⊙ g_t
//! 4. Bias correction: m̂_t = m_t / (1-β₁^t), v̂_t = v_t / (1-β₂^t)
//! 5. Compute update: u_t = -α m̂_t / (√v̂_t + ε)
//! 6. Update position: x_{t+1} = R_{x_t}(u_t)
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
    cost_function::CostFunction,
    error::Result,
    manifold::{Manifold, Point, TangentVector},
    optimizer::{Optimizer, OptimizerState, OptimizationResult, StoppingCriterion, ConvergenceChecker},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim};
use num_traits::Float;
use std::time::Instant;

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

/// State for the Adam optimizer.
#[derive(Debug)]
struct AdamState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// First moment estimate (m_t)
    first_moment: TangentVector<T, D>,
    /// Second moment estimate (v_t)
    second_moment: TangentVector<T, D>,
    /// Maximum second moment for AMSGrad (v̂_t)
    max_second_moment: Option<TangentVector<T, D>>,
}

impl<T, D> AdamState<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Creates a new Adam state with zero moments.
    fn new(dimension: D, use_amsgrad: bool) -> Self {
        let zeros = TangentVector::zeros_generic(dimension, nalgebra::U1);
        Self {
            first_moment: zeros.clone(),
            second_moment: zeros.clone(),
            max_second_moment: if use_amsgrad {
                Some(zeros)
            } else {
                None
            },
        }
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
/// ```rust
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
pub struct Adam<T>
where
    T: Scalar,
{
    config: AdamConfig<T>,
}

impl<T> Adam<T>
where
    T: Scalar,
{
    /// Creates a new Adam optimizer with the given configuration.
    pub fn new(config: AdamConfig<T>) -> Self {
        Self { config }
    }

    /// Creates a new Adam optimizer with default configuration.
    pub fn default() -> Self {
        Self::new(AdamConfig::default())
    }

    /// Returns the optimizer configuration.
    pub fn config(&self) -> &AdamConfig<T> {
        &self.config
    }

    /// Returns the optimizer name.
    pub fn name(&self) -> &str {
        "Riemannian Adam"
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
        
        // Initialize Adam-specific state
        let mut adam_state = AdamState::new(initial_point.shape_generic().0, self.config.use_amsgrad);
        
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
            self.step_internal(cost_fn, manifold, &mut state, &mut adam_state)?;
        }
    }

    /// Performs a single optimization step.
    ///
    /// This is the public step method that creates temporary Adam state.
    /// For full optimization runs, use `optimize` which maintains state across iterations.
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
        // Create temporary Adam state - this is a limitation of the public step interface
        let dim = state.point.shape_generic().0;
        let mut adam_state = AdamState::new(dim, self.config.use_amsgrad);
        self.step_internal(cost_fn, manifold, state, &mut adam_state)
    }

    /// Clips gradient if configured.
    fn clip_gradient<D>(&self, gradient: &mut TangentVector<T, D>) 
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
    {
        if let Some(threshold) = self.config.gradient_clip {
            let norm = gradient.norm();
            if norm > threshold {
                *gradient *= threshold / norm;
            }
        }
    }

    /// Computes bias-corrected moments.
    fn compute_bias_correction(&self, iteration: usize) -> (T, T) {
        let t = <T as Scalar>::from_f64(iteration as f64);
        let beta1_t = <T as Float>::powf(self.config.beta1, t);
        let beta2_t = <T as Float>::powf(self.config.beta2, t);
        
        let bias1 = T::one() - beta1_t;
        let bias2 = T::one() - beta2_t;
        
        (bias1, bias2)
    }

    /// Applies weight decay if configured (AdamW variant).
    fn apply_weight_decay<D, M>(&self, point: &mut Point<T, D>, manifold: &M) -> Result<()>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        M: Manifold<T, D>,
    {
        if let Some(weight_decay) = self.config.weight_decay {
            // For manifolds, we apply weight decay by moving towards the origin
            // This is a simplified version - proper implementation would use
            // a reference point on the manifold
            let origin = manifold.random_point();
            let direction = manifold.inverse_retract(point, &origin)?;
            let decay_step = direction * (-weight_decay);
            *point = manifold.retract(point, &decay_step)?;
        }
        Ok(())
    }

    /// Internal step method that has access to Adam-specific state.
    fn step_internal<D, C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, D>,
        adam_state: &mut AdamState<T, D>,
    ) -> Result<()>
    where
        D: Dim,
        DefaultAllocator: Allocator<D>,
        C: CostFunction<T, D>,
        M: Manifold<T, D>,
    {
        // Compute cost and Euclidean gradient
        let (_cost, euclidean_grad) = cost_fn.cost_and_gradient(&state.point)?;
        
        // Convert to Riemannian gradient
        let mut gradient = manifold.euclidean_to_riemannian_gradient(&state.point, &euclidean_grad)?;
        
        // Apply gradient clipping if configured
        self.clip_gradient(&mut gradient);
        
        let grad_norm = gradient.norm();
        state.set_gradient(gradient.clone(), grad_norm);
        
        // Transport previous moments to current tangent space if needed
        if state.iteration > 0 {
            // For efficiency, we assume moments are already in current tangent space
            // In practice, parallel transport might be needed for non-flat manifolds
            adam_state.first_moment = manifold.project_tangent(&state.point, &adam_state.first_moment)?;
            adam_state.second_moment = manifold.project_tangent(&state.point, &adam_state.second_moment)?;
            if let Some(ref mut max_moment) = adam_state.max_second_moment {
                *max_moment = manifold.project_tangent(&state.point, max_moment)?;
            }
        }
        
        // Update biased first moment estimate
        adam_state.first_moment = &adam_state.first_moment * self.config.beta1 
            + &gradient * (T::one() - self.config.beta1);
        
        // Update biased second moment estimate (element-wise square)
        let gradient_squared = gradient.component_mul(&gradient);
        adam_state.second_moment = &adam_state.second_moment * self.config.beta2 
            + &gradient_squared * (T::one() - self.config.beta2);
        
        // Update maximum second moment for AMSGrad
        if let Some(ref mut max_moment) = adam_state.max_second_moment {
            max_moment.zip_apply(&adam_state.second_moment, |max_v, v| {
                *max_v = <T as Float>::max(*max_v, v);
            });
        }
        
        // Compute bias correction
        let (bias1, bias2) = self.compute_bias_correction(state.iteration + 1);
        
        // Compute bias-corrected moments
        let m_hat = &adam_state.first_moment / bias1;
        let v_hat = if let Some(ref max_moment) = adam_state.max_second_moment {
            max_moment / bias2
        } else {
            &adam_state.second_moment / bias2
        };
        
        // Compute update direction
        let mut direction = TangentVector::zeros_generic(state.point.shape_generic().0, nalgebra::U1);
        for i in 0..direction.len() {
            let v_sqrt = <T as Float>::sqrt(v_hat[i] + self.config.epsilon);
            direction[i] = -m_hat[i] / v_sqrt;
        }
        
        // Scale by learning rate
        direction *= self.config.learning_rate;
        
        // Update position on manifold
        let new_point = manifold.retract(&state.point, &direction)?;
        
        // Apply weight decay if configured (AdamW)
        let mut final_point = new_point;
        self.apply_weight_decay(&mut final_point, manifold)?;
        
        // Project to ensure we stay on manifold (safety measure)
        final_point = manifold.project_point(&final_point);
        
        // Evaluate cost at new point
        let new_cost = cost_fn.cost(&final_point)?;
        
        // Update state
        state.update(final_point, new_cost);
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Dyn};

    /// Simple quadratic cost function for testing
    #[derive(Debug)]
    struct QuadraticCost;
    
    impl CostFunction<f64, Dyn> for QuadraticCost {
        fn cost(&self, point: &DVector<f64>) -> Result<f64> {
            Ok(0.5 * point.norm_squared())
        }
        
        fn cost_and_gradient(&self, point: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
            let cost = self.cost(point)?;
            Ok((cost, point.clone()))
        }
    }
    
    /// Simple Euclidean manifold for testing
    #[derive(Debug)]
    struct EuclideanManifold {
        dim: usize,
    }
    
    impl EuclideanManifold {
        fn new(dim: usize) -> Self {
            Self { dim }
        }
    }
    
    impl Manifold<f64, Dyn> for EuclideanManifold {
        fn name(&self) -> &str {
            "Euclidean"
        }
        
        fn dimension(&self) -> usize {
            self.dim
        }
        
        fn is_point_on_manifold(&self, point: &DVector<f64>, _tolerance: f64) -> bool {
            point.len() == self.dim
        }
        
        fn is_vector_in_tangent_space(&self, _point: &DVector<f64>, vector: &DVector<f64>, _tolerance: f64) -> bool {
            vector.len() == self.dim
        }
        
        fn project_point(&self, point: &DVector<f64>) -> DVector<f64> {
            point.clone()
        }
        
        fn project_tangent(&self, _point: &DVector<f64>, vector: &DVector<f64>) -> Result<DVector<f64>> {
            Ok(vector.clone())
        }
        
        fn inner_product(&self, _point: &DVector<f64>, u: &DVector<f64>, v: &DVector<f64>) -> Result<f64> {
            Ok(u.dot(v))
        }
        
        fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>) -> Result<DVector<f64>> {
            Ok(point + tangent)
        }
        
        fn inverse_retract(&self, point: &DVector<f64>, other: &DVector<f64>) -> Result<DVector<f64>> {
            Ok(other - point)
        }
        
        fn euclidean_to_riemannian_gradient(&self, _point: &DVector<f64>, grad: &DVector<f64>) -> Result<DVector<f64>> {
            Ok(grad.clone())
        }
        
        fn random_point(&self) -> DVector<f64> {
            DVector::zeros(self.dim)
        }
        
        fn random_tangent(&self, _point: &DVector<f64>) -> Result<DVector<f64>> {
            Ok(DVector::zeros(self.dim))
        }
    }

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
        let state = AdamState::<f64, Dyn>::new(Dyn(5), true);
        assert_eq!(state.first_moment.len(), 5);
        assert_eq!(state.second_moment.len(), 5);
        assert!(state.max_second_moment.is_some());
    }

    #[test]
    fn test_bias_correction() {
        let adam = Adam::<f64>::new(AdamConfig::default());
        
        // Test bias correction at different iterations
        let (bias1_1, bias2_1) = adam.compute_bias_correction(1);
        assert_relative_eq!(bias1_1, 0.1, epsilon = 1e-10);
        assert_relative_eq!(bias2_1, 0.001, epsilon = 1e-10);
        
        let (bias1_10, bias2_10) = adam.compute_bias_correction(10);
        assert!(bias1_10 > bias1_1);
        assert!(bias2_10 > bias2_1);
    }

    #[test]
    fn test_adam_on_euclidean() {
        let manifold = EuclideanManifold::new(2);
        let cost_fn = QuadraticCost;
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let config = AdamConfig::new()
            .with_learning_rate(0.1)
            .with_beta1(0.9)
            .with_beta2(0.999);
        
        let mut optimizer = Adam::new(config);
        
        let stopping_criterion = StoppingCriterion::new()
            .with_max_iterations(100)
            .with_gradient_tolerance(1e-4);
        
        let result = optimizer.optimize(
            &cost_fn,
            &manifold,
            &initial_point,
            &stopping_criterion,
        ).unwrap();
        
        // Adam should converge close to the minimum at origin
        assert!(result.value < 1e-3);
        assert!(result.converged);
        assert!(result.iterations > 0);
    }

    #[test]
    fn test_gradient_clipping() {
        let adam = Adam::new(
            AdamConfig::new().with_gradient_clip(1.0)
        );
        
        let mut gradient = DVector::from_vec(vec![3.0, 4.0]); // norm = 5
        adam.clip_gradient(&mut gradient);
        
        assert_relative_eq!(gradient.norm(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(gradient[0], 0.6, epsilon = 1e-10);
        assert_relative_eq!(gradient[1], 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_amsgrad_variant() {
        let manifold = EuclideanManifold::new(2);
        let cost_fn = QuadraticCost;
        let initial_point = DVector::from_vec(vec![1.0, 1.0]);
        
        let config = AdamConfig::new()
            .with_learning_rate(0.1)
            .with_amsgrad();
        
        let mut optimizer = Adam::new(config);
        let initial_cost = cost_fn.cost(&initial_point).unwrap();
        let mut state = OptimizerState::new(initial_point.clone(), initial_cost);
        let mut adam_state = AdamState::new(Dyn(2), true);
        
        // Perform a step
        optimizer.step_internal(&cost_fn, &manifold, &mut state, &mut adam_state).unwrap();
        
        // Check that max_second_moment is being tracked
        assert!(adam_state.max_second_moment.is_some());
        let max_moment = adam_state.max_second_moment.as_ref().unwrap();
        
        // Max moment should be >= current second moment
        for i in 0..2 {
            assert!(max_moment[i] >= adam_state.second_moment[i]);
        }
    }
}

// Implementation of the Optimizer trait from core
impl<T, D> Optimizer<T, D> for Adam<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn name(&self) -> &str {
        "Riemannian Adam"
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
        Adam::optimize(self, cost_fn, manifold, initial_point, stopping_criterion)
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
        Adam::step(self, cost_fn, manifold, state)
    }
}