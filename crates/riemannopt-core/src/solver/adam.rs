//! Riemannian Adam optimizer.
//!
//! Adam (Adaptive Moment Estimation) is a popular first-order optimization algorithm.
//! This implementation extends Adam to Riemannian manifolds by properly handling
//! tangent space operations and parallel transport, avoiding coordinate-wise
//! operations that break manifold invariance.
//!
//! # Algorithm
//!
//! The Riemannian Adam update at iteration t:
//! 1. Compute Riemannian gradient: g_t = grad f(x_t)
//! 2. Parallel transport moments: m_{t-1} → Γ(m_{t-1}), v_{t-1} → Γ(v_{t-1})
//! 3. Update biased first moment: m_t = β₁ Γ(m_{t-1}) + (1-β₁) g_t
//! 4. Update second moment norm tracking: ‖v_t‖² = β₂ ‖Γ(v_{t-1})‖² + (1-β₂) ‖g_t‖²
//! 5. Bias correction: m̂_t = m_t / (1-β₁^t), ‖v̂_t‖ = ‖v_t‖ / √(1-β₂^t)
//! 6. Compute update: u_t = m̂_t / (‖v̂_t‖ + ε)
//! 7. Update position: x_{t+1} = R_{x_t}(-α u_t)

use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

use crate::{
	manifold::Manifold,
	problem::Problem,
	solver::{Solver, SolverResult, StoppingCriterion, TerminationReason},
	types::Scalar,
};

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
			gradient_clip: None,
		}
	}
}

impl<T: Scalar> AdamConfig<T> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_learning_rate(mut self, learning_rate: T) -> Self {
		self.learning_rate = learning_rate;
		self
	}

	pub fn with_beta1(mut self, beta1: T) -> Self {
		self.beta1 = beta1;
		self
	}

	pub fn with_beta2(mut self, beta2: T) -> Self {
		self.beta2 = beta2;
		self
	}

	pub fn with_epsilon(mut self, epsilon: T) -> Self {
		self.epsilon = epsilon;
		self
	}

	pub fn with_amsgrad(mut self) -> Self {
		self.use_amsgrad = true;
		self
	}

	pub fn with_gradient_clip(mut self, threshold: T) -> Self {
		self.gradient_clip = Some(threshold);
		self
	}
}

/// Riemannian Adam optimizer.
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

	pub fn config(&self) -> &AdamConfig<T> {
		&self.config
	}
}

impl<T: Scalar> Solver<T> for Adam<T> {
	fn name(&self) -> &str {
		if self.config.use_amsgrad {
			"Riemannian AMSGrad"
		} else {
			"Riemannian Adam"
		}
	}

	fn solve<M, P>(
		&mut self,
		problem: &P,
		manifold: &M,
		initial_point: &M::Point,
		stopping_criterion: &StoppingCriterion<T>,
	) -> SolverResult<T, M::Point>
	where
		M: Manifold<T>,
		P: Problem<T, M>,
	{
		let start_time = Instant::now();

		// ════════════════════════════════════════════════════════════════════
		// 1. Memory Allocation (Cold Path)
		// ════════════════════════════════════════════════════════════════════
		let mut current_point = initial_point.clone();
		let mut previous_point = manifold.allocate_point();
		let mut candidate_point = manifold.allocate_point();

		let mut gradient = manifold.allocate_tangent();
		let mut m = manifold.allocate_tangent();
		let mut v = manifold.allocate_tangent();
		let mut v_max = manifold.allocate_tangent(); // Used only if amsgrad=true
		let mut direction = manifold.allocate_tangent();

		// Scratch buffers to avoid any allocation during math operations
		let mut scratch0 = manifold.allocate_tangent();

		let mut prob_ws = problem.create_workspace(manifold, &current_point);
		let mut man_ws = manifold.create_workspace(&current_point);

		// ════════════════════════════════════════════════════════════════════
		// 2. Initialization
		// ════════════════════════════════════════════════════════════════════
		let mut current_cost = problem.cost_and_gradient(
			manifold,
			&current_point,
			&mut gradient,
			&mut prob_ws,
			&mut man_ws,
		);
		let mut grad_norm = manifold.norm(&current_point, &gradient, &mut man_ws);

		let mut iteration = 0;
		let mut fn_evals = 1;
		let mut grad_evals = 1;
		let mut termination = TerminationReason::MaxIterations;

		let grad_tol = stopping_criterion
			.gradient_tolerance
			.unwrap_or(T::DEFAULT_GRADIENT_TOLERANCE);
		let max_iter = stopping_criterion.max_iterations.unwrap_or(usize::MAX);

		if grad_norm <= grad_tol {
			termination = TerminationReason::Converged;
		}

		let mut v_max_norm_sq = T::zero(); // Track v_max scalar norm efficiently

		// ════════════════════════════════════════════════════════════════════
		// 3. Optimization Loop (Hot Path - Zero Allocation)
		// ════════════════════════════════════════════════════════════════════
		while termination == TerminationReason::MaxIterations && iteration < max_iter {
			// -- A. Gradient Clipping --
			if let Some(threshold) = self.config.gradient_clip {
				if grad_norm > threshold {
					let scale = threshold / grad_norm;
					manifold.scale_tangent(scale, &mut gradient);
					grad_norm = threshold; // update norm to reflect clipping
				}
			}

			let grad_norm_sq = grad_norm * grad_norm;
			let t = (iteration + 1) as f64;

			// -- B. Transport Moments from previous point (if iteration > 0) --
			if iteration > 0 {
				manifold.parallel_transport(
					&previous_point,
					&current_point,
					&m,
					&mut scratch0,
					&mut man_ws,
				);
				m.clone_from(&scratch0);

				manifold.parallel_transport(
					&previous_point,
					&current_point,
					&v,
					&mut scratch0,
					&mut man_ws,
				);
				v.clone_from(&scratch0);

				if self.config.use_amsgrad {
					manifold.parallel_transport(
						&previous_point,
						&current_point,
						&v_max,
						&mut scratch0,
						&mut man_ws,
					);
					v_max.clone_from(&scratch0);
				}
			} else {
				// Initialize moments on first step
				m.clone_from(&gradient);
				v.clone_from(&gradient);
				if self.config.use_amsgrad {
					v_max.clone_from(&gradient);
					v_max_norm_sq = grad_norm_sq;
				}
			}

			// -- C. Update Moments --
			let beta1 = self.config.beta1;
			let beta2 = self.config.beta2;

			if iteration == 0 {
				// Iteration 0 initialization scaling
				manifold.scale_tangent(T::one() - beta1, &mut m);

				let scale = if grad_norm > T::zero() {
					<T as Float>::sqrt((T::one() - beta2) * grad_norm_sq) / grad_norm
				} else {
					T::zero()
				};
				manifold.scale_tangent(scale, &mut v);
			} else {
				// m_t = β₁ * m_{t-1} + (1-β₁) * g_t
				manifold.scale_tangent(beta1, &mut m);
				scratch0.clone_from(&gradient);
				manifold.scale_tangent(T::one() - beta1, &mut scratch0);
				manifold.add_tangents(&mut m, &scratch0);

				// Update second moment norm tracking
				let v_norm_sq = manifold.inner_product(&current_point, &v, &v, &mut man_ws);
				let new_v_norm_sq = beta2 * v_norm_sq + (T::one() - beta2) * grad_norm_sq;
				let new_v_norm = <T as Float>::sqrt(new_v_norm_sq);
				let current_v_norm = <T as Float>::sqrt(v_norm_sq);

				if current_v_norm > T::zero() {
					let scale = new_v_norm / current_v_norm;
					manifold.scale_tangent(scale, &mut v);
				} else if grad_norm > T::zero() {
					v.clone_from(&gradient);
					manifold.scale_tangent(new_v_norm / grad_norm, &mut v);
				}

				// AMSGrad variant update
				if self.config.use_amsgrad && new_v_norm_sq > v_max_norm_sq {
					v_max.clone_from(&v);
					v_max_norm_sq = new_v_norm_sq;
				}
			}

			// -- D. Bias Correction and Search Direction --
			let bias1 = T::one() - <T as Scalar>::from_f64(self.config.beta1.to_f64().powf(t));
			let bias2 = T::one() - <T as Scalar>::from_f64(self.config.beta2.to_f64().powf(t));

			let v_to_use = if self.config.use_amsgrad { &v_max } else { &v };

			let v_norm = manifold.norm(&current_point, v_to_use, &mut man_ws);
			let v_hat_norm = v_norm / <T as Float>::sqrt(bias2);

			let m_hat_scale = T::one() / bias1;
			let step_scale = T::one() / (v_hat_norm + self.config.epsilon);

			// direction = (m / bias1) / (v_hat_norm + epsilon)
			direction.clone_from(&m);
			manifold.scale_tangent(m_hat_scale * step_scale, &mut direction);

			// -- E. Retraction (Update Position) --
			// step = -learning_rate * direction
			manifold.scale_tangent(-self.config.learning_rate, &mut direction);

			previous_point.clone_from(&current_point);
			manifold.retract(
				&current_point,
				&direction,
				&mut candidate_point,
				&mut man_ws,
			);
			current_point.clone_from(&candidate_point);

			// -- F. Evaluate New State --
			let prev_cost = current_cost;
			current_cost = problem.cost_and_gradient(
				manifold,
				&current_point,
				&mut gradient,
				&mut prob_ws,
				&mut man_ws,
			);
			fn_evals += 1;
			grad_evals += 1;
			grad_norm = manifold.norm(&current_point, &gradient, &mut man_ws);
			iteration += 1;

			// -- G. Stopping Criteria Check --
			if grad_norm <= grad_tol {
				termination = TerminationReason::Converged;
			} else if let Some(val_tol) = stopping_criterion.function_tolerance {
				if <T as Float>::abs(current_cost - prev_cost) < val_tol {
					termination = TerminationReason::Converged;
				}
			} else if let Some(target) = stopping_criterion.target_value {
				if current_cost <= target {
					termination = TerminationReason::TargetReached;
				}
			} else if let Some(max_time) = stopping_criterion.max_time {
				if start_time.elapsed() >= max_time {
					termination = TerminationReason::MaxTime;
				}
			} else if let Some(max_fn) = stopping_criterion.max_function_evaluations {
				if fn_evals >= max_fn {
					termination = TerminationReason::MaxFunctionEvaluations;
				}
			}
		}

		// ════════════════════════════════════════════════════════════════════
		// 4. Return Results
		// ════════════════════════════════════════════════════════════════════
		SolverResult::new(
			current_point,
			current_cost,
			iteration,
			start_time.elapsed(),
			termination,
		)
		.with_gradient_norm(grad_norm)
		.with_function_evaluations(fn_evals)
		.with_gradient_evaluations(grad_evals)
	}
}
