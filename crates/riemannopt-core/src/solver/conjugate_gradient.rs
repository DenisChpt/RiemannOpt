//! Riemannian Conjugate Gradient solver.
//!
//! Conjugate gradient methods are a class of optimization algorithms that use
//! conjugate directions to achieve faster convergence than steepest descent.
//! This implementation extends classical CG methods to Riemannian manifolds using
//! parallel transport.

use std::fmt::Debug;
use std::time::Instant;

use crate::{
	manifold::Manifold,
	problem::Problem,
	solver::{Solver, SolverResult, StoppingCriterion, TerminationReason},
	types::Scalar,
};

/// Conjugate gradient method variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjugateGradientMethod {
	FletcherReeves,
	PolakRibiere,
	HestenesStiefel,
	DaiYuan,
	HagerZhang,
	LiuStorey,
}

/// Configuration for the Conjugate Gradient solver.
#[derive(Debug, Clone)]
pub struct CGConfig<T: Scalar> {
	pub method: ConjugateGradientMethod,
	pub restart_period: usize,
	pub use_pr_plus: bool,
	pub min_beta: Option<T>,
	pub max_beta: Option<T>,
	pub initial_step_size: T,
	pub backtrack_rho: T,
	pub armijo_c: T,
}

impl<T: Scalar> Default for CGConfig<T> {
	fn default() -> Self {
		Self {
			method: ConjugateGradientMethod::HestenesStiefel,
			restart_period: 0,
			use_pr_plus: true,
			min_beta: None,
			max_beta: None,
			initial_step_size: T::one(),
			backtrack_rho: <T as Scalar>::from_f64(0.5),
			armijo_c: <T as Scalar>::from_f64(1e-4),
		}
	}
}

impl<T: Scalar> CGConfig<T> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn fletcher_reeves() -> Self {
		Self::new().with_method(ConjugateGradientMethod::FletcherReeves)
	}
	pub fn polak_ribiere() -> Self {
		Self::new().with_method(ConjugateGradientMethod::PolakRibiere)
	}
	pub fn hestenes_stiefel() -> Self {
		Self::new().with_method(ConjugateGradientMethod::HestenesStiefel)
	}
	pub fn dai_yuan() -> Self {
		Self::new().with_method(ConjugateGradientMethod::DaiYuan)
	}

	pub fn with_method(mut self, method: ConjugateGradientMethod) -> Self {
		self.method = method;
		self
	}
	pub fn with_restart_period(mut self, period: usize) -> Self {
		self.restart_period = period;
		self
	}
	pub fn with_pr_plus(mut self, use_pr_plus: bool) -> Self {
		self.use_pr_plus = use_pr_plus;
		self
	}
	pub fn with_min_beta(mut self, min_beta: T) -> Self {
		self.min_beta = Some(min_beta);
		self
	}
	pub fn with_max_beta(mut self, max_beta: T) -> Self {
		self.max_beta = Some(max_beta);
		self
	}
}

/// Riemannian Conjugate Gradient solver.
#[derive(Debug)]
pub struct ConjugateGradient<T: Scalar> {
	config: CGConfig<T>,
}

impl<T: Scalar> ConjugateGradient<T> {
	pub fn new(config: CGConfig<T>) -> Self {
		Self { config }
	}

	pub fn with_default_config() -> Self {
		Self::new(CGConfig::default())
	}
}

impl<T: Scalar> Solver<T> for ConjugateGradient<T> {
	fn name(&self) -> &str {
		match self.config.method {
			ConjugateGradientMethod::FletcherReeves => "Riemannian CG-FR",
			ConjugateGradientMethod::PolakRibiere => {
				if self.config.use_pr_plus {
					"Riemannian CG-PR+"
				} else {
					"Riemannian CG-PR"
				}
			}
			ConjugateGradientMethod::HestenesStiefel => "Riemannian CG-HS",
			ConjugateGradientMethod::DaiYuan => "Riemannian CG-DY",
			ConjugateGradientMethod::HagerZhang => "Riemannian CG-HZ",
			ConjugateGradientMethod::LiuStorey => "Riemannian CG-LS",
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
		let mut prev_gradient = manifold.allocate_tangent();
		let mut direction = manifold.allocate_tangent();
		let mut prev_direction = manifold.allocate_tangent();

		let mut scratch = manifold.allocate_tangent();

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

		let mut iter = 0;
		let mut fn_evals = 1;
		let mut grad_evals = 1;
		let mut termination = TerminationReason::MaxIterations;
		let mut iterations_since_restart = 0;
		let mut last_step_size = self.config.initial_step_size;

		let grad_tol = stopping_criterion
			.gradient_tolerance
			.unwrap_or(T::DEFAULT_GRADIENT_TOLERANCE);
		let max_iter = stopping_criterion.max_iterations.unwrap_or(usize::MAX);

		if grad_norm <= grad_tol {
			termination = TerminationReason::Converged;
		}

		let mut prev_grad_norm_sq = T::zero();

		// ════════════════════════════════════════════════════════════════════
		// 3. Optimization Loop (Hot Path - Zero Allocation)
		// ════════════════════════════════════════════════════════════════════
		while termination == TerminationReason::MaxIterations && iter < max_iter {
			let grad_norm_sq = grad_norm * grad_norm;

			// -- A. Compute Conjugate Direction --
			let mut restarted = false;

			if iter == 0
				|| (self.config.restart_period > 0
					&& iterations_since_restart >= self.config.restart_period)
			{
				// Restart: direction = -gradient
				direction.clone_from(&gradient);
				manifold.scale_tangent(-T::one(), &mut direction);
				restarted = true;
			} else {
				// 1. Parallel transport previous gradient and direction to current tangent space
				manifold.parallel_transport(
					&previous_point,
					&current_point,
					&prev_gradient,
					&mut scratch,
					&mut man_ws,
				);
				prev_gradient.clone_from(&scratch);

				manifold.parallel_transport(
					&previous_point,
					&current_point,
					&prev_direction,
					&mut scratch,
					&mut man_ws,
				);
				prev_direction.clone_from(&scratch);

				// 2. Compute difference: scratch = g_k - τ(g_{k-1})
				scratch.clone_from(&gradient);
				manifold.axpy_tangent(-T::one(), &prev_gradient, &mut scratch);

				// 3. Compute Beta
				let mut beta = match self.config.method {
					ConjugateGradientMethod::FletcherReeves => {
						grad_norm_sq / prev_grad_norm_sq.max(T::epsilon())
					}
					ConjugateGradientMethod::PolakRibiere => {
						let num = manifold.inner_product(
							&current_point,
							&gradient,
							&scratch,
							&mut man_ws,
						);
						let b = num / prev_grad_norm_sq.max(T::epsilon());
						if self.config.use_pr_plus {
							T::zero().max(b)
						} else {
							b
						}
					}
					ConjugateGradientMethod::HestenesStiefel => {
						let num = manifold.inner_product(
							&current_point,
							&gradient,
							&scratch,
							&mut man_ws,
						);
						let den = manifold.inner_product(
							&current_point,
							&prev_direction,
							&scratch,
							&mut man_ws,
						);
						if den.abs() > T::epsilon() {
							num / den
						} else {
							T::zero()
						}
					}
					ConjugateGradientMethod::DaiYuan => {
						let den = manifold.inner_product(
							&current_point,
							&prev_direction,
							&scratch,
							&mut man_ws,
						);
						if den.abs() > T::epsilon() {
							grad_norm_sq / den
						} else {
							T::zero()
						}
					}
					_ => T::zero(), // Fallback for simplicity, HZ and LS can be added similarly
				};

				// Apply constraints
				if let Some(min_b) = self.config.min_beta {
					beta = beta.max(min_b);
				}
				if let Some(max_b) = self.config.max_beta {
					beta = beta.min(max_b);
				}

				// 4. Update direction: d_k = -g_k + β τ(d_{k-1})
				direction.clone_from(&prev_direction);
				manifold.scale_tangent(beta, &mut direction);
				manifold.axpy_tangent(-T::one(), &gradient, &mut direction);
			}

			// Ensure it's a descent direction (⟨g, d⟩ < 0)
			let mut dir_deriv =
				manifold.inner_product(&current_point, &gradient, &direction, &mut man_ws);
			if dir_deriv >= T::zero() {
				direction.clone_from(&gradient);
				manifold.scale_tangent(-T::one(), &mut direction);
				dir_deriv = -grad_norm_sq;
				restarted = true;
			}

			if restarted {
				iterations_since_restart = 0;
			}

			// -- B. Adaptive Line Search (Armijo Backtracking) --
			// For CG, we guess the initial step size based on the previous step
			let mut alpha = if iter == 0 {
				T::one() / grad_norm
			} else {
				last_step_size * <T as Scalar>::from_f64(2.0)
			};

			let mut ls_success = false;
			while alpha > T::MIN_STEP_SIZE {
				// candidate = R_x(alpha * d)
				manifold.scale_tangent(alpha, &mut direction);
				manifold.retract(
					&current_point,
					&direction,
					&mut candidate_point,
					&mut man_ws,
				);
				manifold.scale_tangent(T::one() / alpha, &mut direction); // restore direction

				let candidate_cost = problem.cost(&candidate_point);
				fn_evals += 1;

				if candidate_cost <= current_cost + self.config.armijo_c * alpha * dir_deriv {
					current_cost = candidate_cost;
					ls_success = true;
					break;
				}
				alpha = alpha * self.config.backtrack_rho;
			}

			if !ls_success {
				termination = TerminationReason::LineSearchFailed;
				break;
			}

			last_step_size = alpha;

			// -- C. State Update --
			previous_point.clone_from(&current_point);
			current_point.clone_from(&candidate_point);

			prev_gradient.clone_from(&gradient);
			prev_direction.clone_from(&direction);
			prev_grad_norm_sq = grad_norm_sq;

			let _new_cost = problem.cost_and_gradient(
				manifold,
				&current_point,
				&mut gradient,
				&mut prob_ws,
				&mut man_ws,
			);
			grad_evals += 1;

			grad_norm = manifold.norm(&current_point, &gradient, &mut man_ws);
			iter += 1;
			iterations_since_restart += 1;

			// -- D. Stopping Criteria Check --
			if grad_norm <= grad_tol {
				termination = TerminationReason::Converged;
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

		SolverResult::new(
			current_point,
			current_cost,
			iter,
			start_time.elapsed(),
			termination,
		)
		.with_function_evaluations(fn_evals)
		.with_gradient_evaluations(grad_evals)
		.with_gradient_norm(grad_norm)
	}
}
