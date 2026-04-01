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
//! v_{k+1} = β Γ(v_k) + grad f(x_{k+1})
//! x_{k+1} = R_{x_k}(-α_k v_{k+1})
//! ```
//! where β ∈ [0,1) is the momentum coefficient, and Γ is the parallel transport.

use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

use crate::{
	manifold::Manifold, preconditioner::Preconditioner, problem::Problem, solver::{Solver, SolverResult, StoppingCriterion, TerminationReason}, types::Scalar
};

// ════════════════════════════════════════════════════════════════════════════
//  Configuration
// ════════════════════════════════════════════════════════════════════════════

/// Step size schedules for SGD.
#[derive(Debug, Clone)]
pub enum StepSizeSchedule<T: Scalar> {
	/// Constant step size α_k = α
	Constant(T),
	/// Exponential decay α_k = α_0 * decay^k
	ExponentialDecay { initial: T, decay_rate: T },
	/// Polynomial decay (often 1/sqrt(k) or 1/k in stochastic settings)
	PolynomialDecay { initial: T, decay_rate: T, power: T },
}

impl<T: Scalar> StepSizeSchedule<T> {
	#[inline]
	pub fn get_step_size(&self, iteration: usize) -> T {
		match self {
			Self::Constant(c) => *c,
			Self::ExponentialDecay {
				initial,
				decay_rate,
			} => {
				let dr = (*decay_rate).to_f64();
				*initial * <T as Scalar>::from_f64(dr.powi(iteration as i32))
			}
			Self::PolynomialDecay {
				initial,
				decay_rate,
				power,
			} => {
				let base =
					(T::one() + *decay_rate * <T as Scalar>::from_f64(iteration as f64)).to_f64();
				let exp = (*power).to_f64();
				*initial / <T as Scalar>::from_f64(base.powf(exp))
			}
		}
	}
}

/// Momentum method for Riemannian SGD.
#[derive(Debug, Clone, Copy, PartialEq)]
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
#[derive(Debug, Clone)]
pub struct SGDConfig<T: Scalar> {
	/// Step size schedule controlling how α_k evolves over iterations.
	pub step_size: StepSizeSchedule<T>,
	/// Momentum method for acceleration.
	pub momentum: MomentumMethod<T>,
	/// Gradient clipping threshold to prevent exploding gradients.
	pub gradient_clip: Option<T>,
}

impl<T: Scalar> Default for SGDConfig<T> {
	fn default() -> Self {
		Self {
			step_size: StepSizeSchedule::Constant(<T as Scalar>::from_f64(0.01)),
			momentum: MomentumMethod::None,
			gradient_clip: None,
		}
	}
}

impl<T: Scalar> SGDConfig<T> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_step_size(mut self, schedule: StepSizeSchedule<T>) -> Self {
		self.step_size = schedule;
		self
	}

	pub fn with_constant_step_size(mut self, step_size: T) -> Self {
		self.step_size = StepSizeSchedule::Constant(step_size);
		self
	}

	pub fn with_momentum(mut self, momentum: MomentumMethod<T>) -> Self {
		self.momentum = momentum;
		self
	}

	pub fn with_gradient_clip(mut self, threshold: T) -> Self {
		self.gradient_clip = Some(threshold);
		self
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Solver Implementation
// ════════════════════════════════════════════════════════════════════════════

/// Riemannian Stochastic Gradient Descent optimizer.
#[derive(Debug)]
pub struct SGD<T: Scalar> {
	config: SGDConfig<T>,
}

impl<T: Scalar> SGD<T> {
	pub fn new(config: SGDConfig<T>) -> Self {
		Self { config }
	}

	pub fn with_default_config() -> Self {
		Self::new(SGDConfig::default())
	}
}

impl<T: Scalar> Solver<T> for SGD<T> {
	fn name(&self) -> &str {
		match self.config.momentum {
			MomentumMethod::None => "Riemannian SGD",
			MomentumMethod::Classical { .. } => "Riemannian SGD with Momentum",
			MomentumMethod::Nesterov { .. } => "Riemannian SGD with Nesterov Momentum",
		}
	}

	fn solve<M, P, Pre>(
		&mut self,
		problem: &P,
		manifold: &M,
		initial_point: &M::Point,
		_: &mut Pre,
		stopping_criterion: &StoppingCriterion<T>,
	) -> SolverResult<T, M::Point>
	where
		M: Manifold<T>,
		P: Problem<T, M>,
		Pre: Preconditioner<T, M>
	{
		let start_time = Instant::now();

		// ════════════════════════════════════════════════════════════════════
		// 1. Memory Allocation (Cold Path)
		// ════════════════════════════════════════════════════════════════════
		let mut current_point = initial_point.clone();
		let mut previous_point = manifold.allocate_point();
		// No candidate_point — we retract directly into current_point
		// after swapping it into previous_point.

		let mut gradient = manifold.allocate_tangent();
		let mut momentum = manifold.allocate_tangent();
		let mut direction = manifold.allocate_tangent();
		// No scratch0 — `direction` doubles as the transport scratch
		// buffer (it is written before being read in every branch).

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

		// Zero-initialize momentum
		manifold.scale_tangent(T::zero(), &mut momentum);

		// ════════════════════════════════════════════════════════════════════
		// 3. Optimization Loop (Hot Path — Zero Allocation)
		// ════════════════════════════════════════════════════════════════════
		while termination == TerminationReason::MaxIterations && iteration < max_iter {
			// -- A. Gradient Clipping --
			if let Some(threshold) = self.config.gradient_clip {
				if grad_norm > threshold {
					let scale = threshold / grad_norm;
					manifold.scale_tangent(scale, &mut gradient);
				}
			}

			// -- B. Compute Search Direction (with Momentum) --
			match self.config.momentum {
				MomentumMethod::None => {
					manifold.copy_tangent(&mut direction, &gradient);
				}
				MomentumMethod::Classical { coefficient } => {
					if iteration > 0 {
						// Transport momentum to current tangent space.
						// `direction` is used as the transport scratch buffer —
						// it will be overwritten with the final direction below.
						manifold.parallel_transport(
							&previous_point,
							&current_point,
							&momentum,
							&mut direction,
							&mut man_ws,
						);
						// Re-project for numerical safety
						manifold.project_tangent(
							&current_point,
							&direction,
							&mut momentum,
							&mut man_ws,
						);
					} else {
						manifold.scale_tangent(T::zero(), &mut momentum);
					}

					// v_t = β · v_{t-1} + g_t
					manifold.scale_tangent(coefficient, &mut momentum);
					manifold.add_tangents(&mut momentum, &gradient);

					// direction = v_t
					manifold.copy_tangent(&mut direction, &momentum);
				}
				MomentumMethod::Nesterov { coefficient } => {
					if iteration > 0 {
						// Transport momentum (using direction as scratch)
						manifold.parallel_transport(
							&previous_point,
							&current_point,
							&momentum,
							&mut direction,
							&mut man_ws,
						);
						manifold.project_tangent(
							&current_point,
							&direction,
							&mut momentum,
							&mut man_ws,
						);
					} else {
						manifold.scale_tangent(T::zero(), &mut momentum);
					}

					// v_t = β · v_{t-1} + g_t
					manifold.scale_tangent(coefficient, &mut momentum);
					manifold.add_tangents(&mut momentum, &gradient);

					// d_t = g_t + β · v_t   (Nesterov lookahead)
					manifold.copy_tangent(&mut direction, &gradient);
					manifold.axpy_tangent(coefficient, &momentum, &mut direction);
				}
			};

			// -- C. Retract (Update Position) --
			let step_size = self.config.step_size.get_step_size(iteration);
			manifold.scale_tangent(-step_size, &mut direction);

			// Swap current → previous (O(1)), then retract directly
			// into current_point.  Eliminates candidate_point.
			std::mem::swap(&mut previous_point, &mut current_point);
			manifold.retract(&previous_point, &direction, &mut current_point, &mut man_ws);

			// -- D. Evaluate New State --
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

			// -- E. Stopping Criteria Check --
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
		.with_function_evaluations(fn_evals)
		.with_gradient_evaluations(grad_evals)
		.with_gradient_norm(grad_norm)
	}
}
