//! # Riemannian Natural Gradient Solver
//!
//! This module implements the Natural Gradient method for optimization on Riemannian
//! manifolds. Natural gradient descent uses the Fisher information matrix to define
//! a more natural geometry for the parameter space, leading to updates that are
//! invariant to reparametrization and often converge faster than standard gradient descent.
//!
//! ## Mathematical Foundation
//!
//! Given a smooth cost function f: ℳ → ℝ on a Riemannian manifold (ℳ, g),
//! the natural gradient method uses the Fisher information matrix to rescale
//! the gradient according to the local geometry.
//!
//! At each point x_k ∈ ℳ:
//! ```text
//! η_k = (F(x_k) + λI)^{-1} grad f(x_k)
//! ```
//! where:
//! - F(x_k) is the Fisher information matrix at x_k
//! - λ is the damping factor
//! - grad f(x_k) is the Riemannian gradient
//! - η_k is the natural gradient direction

use num_traits::Float;
use std::time::Instant;

use crate::{
	manifold::Manifold,
	problem::Problem,
	solver::{Solver, SolverResult, StoppingCriterion, TerminationReason},
	types::Scalar,
};

/// Fisher matrix approximation methods
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FisherApproximation {
	/// Full Fisher information matrix
	Full,
	/// Diagonal approximation (efficient for large-scale)
	Diagonal,
	/// Identity matrix (reduces to standard steepest descent)
	Identity,
	/// Empirical Fisher from mini-batch
	Empirical,
}

/// Configuration for the Natural Gradient solver
#[derive(Debug, Clone)]
pub struct NaturalGradientConfig<T: Scalar> {
	/// Learning rate / step size (α)
	pub learning_rate: T,
	/// Fisher approximation method
	pub fisher_approximation: FisherApproximation,
	/// Damping factor for numerical stability (λ)
	pub damping: T,
	/// Update frequency for Fisher matrix (recompute every N iterations)
	pub fisher_update_freq: usize,
	/// Number of samples for empirical Fisher estimation
	pub fisher_num_samples: usize,
}

impl<T: Scalar> Default for NaturalGradientConfig<T> {
	fn default() -> Self {
		Self {
			learning_rate: <T as Scalar>::from_f64(0.01),
			fisher_approximation: FisherApproximation::Identity, // Safe default until F is implemented in Problem
			damping: <T as Scalar>::from_f64(1e-4),
			fisher_update_freq: 10,
			fisher_num_samples: 100,
		}
	}
}

impl<T: Scalar> NaturalGradientConfig<T> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_learning_rate(mut self, lr: T) -> Self {
		self.learning_rate = lr;
		self
	}

	pub fn with_fisher_approximation(mut self, method: FisherApproximation) -> Self {
		self.fisher_approximation = method;
		self
	}

	pub fn with_damping(mut self, damping: T) -> Self {
		self.damping = damping;
		self
	}

	pub fn with_fisher_update_freq(mut self, freq: usize) -> Self {
		self.fisher_update_freq = freq;
		self
	}
}

/// Riemannian Natural Gradient solver
#[derive(Debug)]
pub struct NaturalGradient<T: Scalar> {
	config: NaturalGradientConfig<T>,
}

impl<T: Scalar> NaturalGradient<T> {
	pub fn new(config: NaturalGradientConfig<T>) -> Self {
		Self { config }
	}

	pub fn with_default_config() -> Self {
		Self::new(NaturalGradientConfig::default())
	}

	/// Apply Fisher information matrix inverse to gradient: result = (F + λI)^{-1} grad
	fn apply_fisher_inverse<M: Manifold<T>>(
		&self,
		manifold: &M,
		_point: &M::Point,
		gradient: &M::TangentVector,
		result: &mut M::TangentVector,
	) {
		match self.config.fisher_approximation {
			FisherApproximation::Identity => {
				// F = I, so (I + λI)^{-1} g = (1 / (1 + λ)) g
				let scale = T::one() / (T::one() + self.config.damping);
				result.clone_from(gradient);
				manifold.scale_tangent(scale, result);
			}
			FisherApproximation::Diagonal => {
				// TODO: Requires Problem trait to expose Fisher diagonal.
				// Fallback to damped identity.
				let scale = T::one() / (T::one() + self.config.damping);
				result.clone_from(gradient);
				manifold.scale_tangent(scale, result);
			}
			FisherApproximation::Full | FisherApproximation::Empirical => {
				// TODO: Requires Problem trait to solve Fisher system (like a Hessian-Vector Product).
				// Fallback to damped identity.
				let scale = T::one() / (T::one() + self.config.damping);
				result.clone_from(gradient);
				manifold.scale_tangent(scale, result);
			}
		}
	}
}

impl<T: Scalar> Solver<T> for NaturalGradient<T> {
	fn name(&self) -> &str {
		"Riemannian Natural Gradient"
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
		let mut candidate_point = manifold.allocate_point();

		let mut gradient = manifold.allocate_tangent();
		let mut natural_gradient = manifold.allocate_tangent();
		let mut direction = manifold.allocate_tangent();

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

		// ════════════════════════════════════════════════════════════════════
		// 3. Optimization Loop (Hot Path - Zero Allocation)
		// ════════════════════════════════════════════════════════════════════
		while termination == TerminationReason::MaxIterations && iteration < max_iter {
			// -- A. Compute Natural Gradient --
			// η_k = (F_k + λI)^{-1} g_k
			self.apply_fisher_inverse(manifold, &current_point, &gradient, &mut natural_gradient);

			// -- B. Compute Search Direction --
			// d_k = -α_k η_k
			direction.clone_from(&natural_gradient);
			manifold.scale_tangent(-self.config.learning_rate, &mut direction);

			// -- C. Retract to new point --
			// x_{k+1} = R_{x_k}(d_k)
			manifold.retract(
				&current_point,
				&direction,
				&mut candidate_point,
				&mut man_ws,
			);

			// Swap points
			std::mem::swap(&mut current_point, &mut candidate_point);

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
