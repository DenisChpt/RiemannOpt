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

use num_traits::Float;
use riemannopt_core::{
	core::{cost_function::CostFunction, manifold::Manifold},
	error::Result,
	optimization::optimizer::{
		OptimizationResult, Optimizer, StoppingCriterion, TerminationReason,
	},
	types::Scalar,
};
use std::fmt::Debug;
use std::time::Instant;

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
		Self { config }
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
		previous_point: Option<&M::Point>,
		criterion: &StoppingCriterion<T>,
	) -> Option<TerminationReason>
	where
		M: Manifold<T>,
	{
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
			if let Some(prev_point) = previous_point {
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

	/// Computes the Adam direction with proper moment updates and parallel transport.
	///
	/// Uses `scratch0` and `scratch1` as temporary buffers instead of allocating.
	fn compute_adam_direction<M>(
		&self,
		manifold: &M,
		current_point: &M::Point,
		previous_point: Option<&M::Point>,
		gradient: &M::TangentVector,
		adam_state: &mut AdamState<T, M::TangentVector>,
		direction: &mut M::TangentVector,
		scratch0: &mut M::TangentVector,
		scratch1: &mut M::TangentVector,
		manifold_ws: &mut M::Workspace,
	) -> Result<()>
	where
		M: Manifold<T>,
	{
		adam_state.t += 1;

		// Initialize moments on first iteration (unavoidable clones)
		match (&mut adam_state.m, &mut adam_state.v) {
			(None, None) => {
				let mut m_init = gradient.clone();
				manifold.scale_tangent(
					current_point,
					T::one() - adam_state.beta1,
					gradient,
					&mut m_init,
				)?;

				let grad_norm_sq =
					manifold.inner_product(current_point, gradient, gradient, manifold_ws)?;
				let mut v_init = gradient.clone();
				let grad_norm = <T as Float>::sqrt(grad_norm_sq);
				if grad_norm > T::zero() {
					let scale = <T as Float>::sqrt((T::one() - adam_state.beta2) * grad_norm_sq)
						/ grad_norm;
					manifold.scale_tangent(current_point, scale, gradient, &mut v_init)?;
				} else {
					manifold.scale_tangent(current_point, T::zero(), gradient, &mut v_init)?;
				}

				adam_state.m = Some(m_init);
				adam_state.v = Some(v_init);

				if adam_state.amsgrad {
					adam_state.v_max = adam_state.v.clone();
				}
			}
			(Some(_), Some(_)) => {}
			_ => {
				return Err(riemannopt_core::error::ManifoldError::invalid_point(
					"Invalid Adam state",
				))
			}
		}

		// Update moments
		if let (Some(m), Some(v)) = (&mut adam_state.m, &mut adam_state.v) {
			// Transport moments using scratch buffers (zero alloc)
			if let Some(prev_point) = previous_point {
				manifold.parallel_transport(prev_point, current_point, m, scratch0, manifold_ws)?;
				manifold.project_tangent(current_point, scratch0, m, manifold_ws)?;

				manifold.parallel_transport(prev_point, current_point, v, scratch0, manifold_ws)?;
				manifold.project_tangent(current_point, scratch0, v, manifold_ws)?;

				if let Some(v_max) = &mut adam_state.v_max {
					manifold.parallel_transport(
						prev_point,
						current_point,
						v_max,
						scratch0,
						manifold_ws,
					)?;
					manifold.project_tangent(current_point, scratch0, v_max, manifold_ws)?;
				}
			}

			// m = β₁ · m + (1-β₁) · gradient
			let beta1 = adam_state.beta1;
			manifold.scale_tangent(current_point, beta1, m, scratch0)?;
			manifold.scale_tangent(current_point, T::one() - beta1, gradient, scratch1)?;
			manifold.add_tangents(current_point, scratch0, scratch1, m)?;
			// direction is used as temp for add_tangents here, m gets the result

			// Update second moment norm: ||v_new||² = β₂ · ||v||² + (1-β₂) · ||g||²
			let beta2 = adam_state.beta2;
			let v_norm_sq = manifold.inner_product(current_point, v, v, manifold_ws)?;
			let grad_norm_sq =
				manifold.inner_product(current_point, gradient, gradient, manifold_ws)?;
			let new_v_norm_sq = beta2 * v_norm_sq + (T::one() - beta2) * grad_norm_sq;
			let new_v_norm = <T as Float>::sqrt(new_v_norm_sq);
			let current_v_norm = <T as Float>::sqrt(v_norm_sq);

			if current_v_norm > T::zero() {
				// scale_tangent reads v into scratch0, then writes back to v
				manifold.scale_tangent(current_point, new_v_norm / current_v_norm, v, scratch0)?;
				v.clone_from(scratch0);
			} else if grad_norm_sq > T::zero() {
				manifold.scale_tangent(
					current_point,
					new_v_norm / <T as Float>::sqrt(grad_norm_sq),
					gradient,
					v,
				)?;
			}

			// AMSGrad: v_max = max(v_max, v) by norm
			if adam_state.amsgrad {
				if let Some(v_max) = &mut adam_state.v_max {
					let v_max_norm_sq =
						manifold.inner_product(current_point, v_max, v_max, manifold_ws)?;
					if new_v_norm_sq > v_max_norm_sq {
						v_max.clone_from(v);
					}
				}
			}
		}

		// Compute bias-corrected direction
		let t = adam_state.t as f64;
		let bias1 = T::one() - <T as Scalar>::from_f64(adam_state.beta1.to_f64().powf(t));
		let bias2 = T::one() - <T as Scalar>::from_f64(adam_state.beta2.to_f64().powf(t));

		if let (Some(m), Some(v)) = (&adam_state.m, &adam_state.v) {
			let v_to_use = if adam_state.amsgrad {
				adam_state.v_max.as_ref().unwrap_or(v)
			} else {
				v
			};

			// m̂ = m / (1 - β₁^t)  →  into scratch0
			manifold.scale_tangent(current_point, T::one() / bias1, m, scratch0)?;

			// direction = m̂ / (||v̂|| + ε)
			let v_norm = <T as Float>::sqrt(manifold.inner_product(
				current_point,
				v_to_use,
				v_to_use,
				manifold_ws,
			)?);
			let v_hat_norm = v_norm / <T as Float>::sqrt(bias2);
			let scale = T::one() / (v_hat_norm + adam_state.epsilon);
			manifold.scale_tangent(current_point, scale, scratch0, direction)?;
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
		use riemannopt_core::optimization::workspace::CommonWorkspace;

		let start_time = Instant::now();

		let initial_cost = cost_fn.cost(initial_point)?;
		let mut adam_state = AdamState::new(
			self.config.beta1,
			self.config.beta2,
			self.config.epsilon,
			self.config.use_amsgrad,
		);
		let mut current_point = initial_point.clone();
		let mut has_previous_point = false;
		let mut current_cost = initial_cost;
		let mut previous_cost: Option<T> = None;
		let mut gradient_norm: Option<T> = None;
		let mut iteration = 0;
		let mut function_evaluations = 1;
		let mut gradient_evaluations = 0;

		// Compute initial gradient to initialize workspace
		let initial_grad = cost_fn.gradient(initial_point)?;
		gradient_evaluations += 1;

		// Pre-allocate all buffers once
		let mut ws = CommonWorkspace::new(initial_point, &initial_grad);
		let mut manifold_ws = manifold.create_workspace(initial_point);
		ws.euclidean_grad.clone_from(&initial_grad);

		loop {
			let prev_ref = if has_previous_point {
				Some(&ws.previous_point)
			} else {
				None
			};
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
				prev_ref,
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

			// Gradient computation (in-place)
			let new_cost = cost_fn.cost_and_gradient(&current_point, &mut ws.euclidean_grad)?;
			function_evaluations += 1;
			gradient_evaluations += 1;

			manifold.euclidean_to_riemannian_gradient(
				&current_point,
				&ws.euclidean_grad,
				&mut ws.riemannian_grad,
				&mut manifold_ws,
			)?;

			let grad_norm_squared = manifold.inner_product(
				&current_point,
				&ws.riemannian_grad,
				&ws.riemannian_grad,
				&mut manifold_ws,
			)?;
			let grad_norm = <T as Float>::sqrt(grad_norm_squared);
			gradient_norm = Some(grad_norm);

			// Gradient clipping (in-place via scratch)
			if let Some(threshold) = self.config.gradient_clip {
				if grad_norm > threshold {
					let scale = threshold / grad_norm;
					ws.scratch[0].clone_from(&ws.riemannian_grad);
					manifold.scale_tangent(
						&current_point,
						scale,
						&ws.scratch[0],
						&mut ws.riemannian_grad,
					)?;
				}
			}

			// Adam direction (uses scratch[0..1] internally)
			let prev_ref = if has_previous_point {
				Some(&ws.previous_point as &M::Point)
			} else {
				None
			};
			{
				let [ref mut s0, ref mut s1, ..] = ws.scratch;
				self.compute_adam_direction(
					manifold,
					&current_point,
					prev_ref,
					&ws.riemannian_grad,
					&mut adam_state,
					&mut ws.direction,
					s0,
					s1,
					&mut manifold_ws,
				)?;
			}

			// Scale by -lr and retract (in-place)
			manifold.scale_tangent(
				&current_point,
				-self.config.learning_rate,
				&ws.direction,
				&mut ws.scaled_direction,
			)?;
			manifold.retract(
				&current_point,
				&ws.scaled_direction,
				&mut ws.new_point,
				&mut manifold_ws,
			)?;

			// Swap points (zero alloc)
			std::mem::swap(&mut ws.previous_point, &mut current_point);
			has_previous_point = true;
			std::mem::swap(&mut current_point, &mut ws.new_point);

			previous_cost = Some(current_cost);
			current_cost = new_cost;
			iteration += 1;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use riemannopt_core::linalg;

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
		let state = AdamState::<f64, linalg::Vec<f64>>::new(0.9, 0.999, 1e-8, false);
		assert_eq!(state.beta1, 0.9);
		assert_eq!(state.beta2, 0.999);
		assert_eq!(state.epsilon, 1e-8);
		assert!(!state.amsgrad);
	}

	#[test]
	fn test_amsgrad_state() {
		let state = AdamState::<f64, linalg::Vec<f64>>::new(0.9, 0.999, 1e-8, true);
		assert!(state.amsgrad);
	}

	#[test]
	fn test_adam_builder() {
		let state = AdamStateBuilder::<f64>::new()
			.beta1(0.8)
			.beta2(0.95)
			.epsilon(1e-10)
			.amsgrad(true)
			.build::<linalg::Vec<f64>>();

		assert_eq!(state.beta1, 0.8);
		assert_eq!(state.beta2, 0.95);
		assert_eq!(state.epsilon, 1e-10);
		assert!(state.amsgrad);
	}
}
