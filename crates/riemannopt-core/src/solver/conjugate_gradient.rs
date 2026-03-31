//! Riemannian Conjugate Gradient solver.
//!
//! Conjugate gradient methods are a class of optimization algorithms that use
//! conjugate directions to achieve faster convergence than steepest descent.
//! This implementation extends classical CG methods to Riemannian manifolds
//! using parallel transport.
//!
//! # Line Search
//!
//! Uses an adaptive line search inspired by Manopt/Pymanopt:
//! - The initial step size is normalized by `‖d‖` (direction-invariant).
//! - After the first iteration, the previous accepted step is reused.
//! - If accepted on first try or after many backtracks, the initial step
//!   is doubled for the next iteration (stay aggressive).
//! - If accepted after exactly one backtrack, the initial step is kept
//!   (we're in the right ballpark).
//! - The sufficient decrease parameter defaults to 0.5 (stronger than
//!   the standard Armijo 1e-4), which ensures high-quality steps that
//!   preserve conjugacy.

use num_traits::Float;
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

	// ── Line search parameters ───────────────────────────────────────
	/// Sufficient decrease parameter (Armijo c₁).
	///
	/// Default: 0.5 (matches Pymanopt's AdaptiveLineSearcher).
	/// This is more restrictive than the standard 1e-4, which forces
	/// higher-quality steps that preserve CG conjugacy.
	/// Use 1e-4 for L-BFGS or steepest descent.
	pub sufficient_decrease: T,
	/// Step size contraction factor (ρ).
	pub contraction_factor: T,
	/// Maximum number of line search evaluations per iteration.
	pub ls_max_iterations: usize,
	/// Initial step size for the first iteration (before adaptation).
	pub initial_step_size: T,
}

impl<T: Scalar> Default for CGConfig<T> {
	fn default() -> Self {
		Self {
			method: ConjugateGradientMethod::HestenesStiefel,
			restart_period: 0,
			use_pr_plus: true,
			min_beta: None,
			max_beta: None,
			sufficient_decrease: <T as Scalar>::from_f64(0.5),
			contraction_factor: <T as Scalar>::from_f64(0.5),
			ls_max_iterations: 10,
			initial_step_size: T::one(),
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
	pub fn with_sufficient_decrease(mut self, c: T) -> Self {
		self.sufficient_decrease = c;
		self
	}
	pub fn with_contraction_factor(mut self, rho: T) -> Self {
		self.contraction_factor = rho;
		self
	}
	pub fn with_ls_max_iterations(mut self, max: usize) -> Self {
		self.ls_max_iterations = max;
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

		let grad_tol = stopping_criterion
			.gradient_tolerance
			.unwrap_or(T::DEFAULT_GRADIENT_TOLERANCE);
		let max_iter = stopping_criterion.max_iterations.unwrap_or(usize::MAX);

		if grad_norm <= grad_tol {
			termination = TerminationReason::Converged;
		}

		let mut prev_grad_norm_sq = T::zero();

		// ── Adaptive line search state ───────────────────────────────────
		// Tracks the previous accepted α to reuse as the next initial guess.
		// `None` means first iteration (use 1/‖d‖ as initial step).
		let mut ls_prev_alpha: Option<T> = None;

		// ════════════════════════════════════════════════════════════════════
		// 3. Optimization Loop (Hot Path — Zero Allocation)
		// ════════════════════════════════════════════════════════════════════
		while termination == TerminationReason::MaxIterations && iter < max_iter {
			let grad_norm_sq = grad_norm * grad_norm;

			// -- A. Compute Conjugate Direction --
			let mut restarted = false;

			if iter == 0
				|| (self.config.restart_period > 0
					&& iterations_since_restart >= self.config.restart_period)
			{
				// Restart: direction = −gradient
				manifold.copy_tangent(&mut direction, &gradient);
				manifold.scale_tangent(-T::one(), &mut direction);
				restarted = true;
			} else {
				// 1. Transport previous gradient and direction to current
				//    tangent space. Swap avoids a memcpy after each transport.
				manifold.parallel_transport(
					&previous_point,
					&current_point,
					&prev_gradient,
					&mut scratch,
					&mut man_ws,
				);
				std::mem::swap(&mut prev_gradient, &mut scratch);

				manifold.parallel_transport(
					&previous_point,
					&current_point,
					&prev_direction,
					&mut scratch,
					&mut man_ws,
				);
				std::mem::swap(&mut prev_direction, &mut scratch);

				// 2. scratch = g_k − τ(g_{k-1})
				manifold.copy_tangent(&mut scratch, &gradient);
				manifold.axpy_tangent(-T::one(), &prev_gradient, &mut scratch);

				// 3. Compute β
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
					ConjugateGradientMethod::HagerZhang => {
						let diff_dot_d = manifold.inner_product(
							&current_point,
							&scratch,
							&prev_direction,
							&mut man_ws,
						);
						if diff_dot_d.abs() <= T::epsilon() {
							T::zero()
						} else {
							let diff_norm_sq = manifold.inner_product(
								&current_point,
								&scratch,
								&scratch,
								&mut man_ws,
							);
							let grad_dot_d = manifold.inner_product(
								&current_point,
								&gradient,
								&prev_direction,
								&mut man_ws,
							);
							let grad_dot_diff = manifold.inner_product(
								&current_point,
								&gradient,
								&scratch,
								&mut man_ws,
							);
							let two = T::one() + T::one();
							let num = grad_dot_diff - two * diff_norm_sq * grad_dot_d / diff_dot_d;
							num / diff_dot_d
						}
					}
					ConjugateGradientMethod::LiuStorey => {
						let num = manifold.inner_product(
							&current_point,
							&gradient,
							&scratch,
							&mut man_ws,
						);
						let den = -manifold.inner_product(
							&previous_point,
							&prev_gradient,
							&prev_direction,
							&mut man_ws,
						);
						if den.abs() > T::epsilon() {
							T::zero().max(num / den)
						} else {
							T::zero()
						}
					}
				};

				// Apply constraints
				if let Some(min_b) = self.config.min_beta {
					beta = beta.max(min_b);
				}
				if let Some(max_b) = self.config.max_beta {
					beta = beta.min(max_b);
				}

				// 4. d_k = β · τ(d_{k-1}) − g_k
				manifold.copy_tangent(&mut direction, &prev_direction);
				manifold.scale_tangent(beta, &mut direction);
				manifold.axpy_tangent(-T::one(), &gradient, &mut direction);
			}

			// Ensure descent direction (⟨g, d⟩ < 0)
			let mut dir_deriv =
				manifold.inner_product(&current_point, &gradient, &direction, &mut man_ws);
			if dir_deriv >= T::zero() {
				manifold.copy_tangent(&mut direction, &gradient);
				manifold.scale_tangent(-T::one(), &mut direction);
				dir_deriv = -grad_norm_sq;
				restarted = true;
			}

			if restarted {
				iterations_since_restart = 0;
			}

			// ══════════════════════════════════════════════════════════════
			// B. Adaptive Line Search
			// ══════════════════════════════════════════════════════════════
			// Compute ‖d‖ for direction-invariant step sizing.
			let dir_norm = manifold.norm(&current_point, &direction, &mut man_ws);

			// Initial step: reuse previous α, or 1/‖d‖ on first iteration.
			let mut alpha = match ls_prev_alpha {
				Some(prev) => prev,
				None => {
					if dir_norm > T::epsilon() {
						self.config.initial_step_size / dir_norm
					} else {
						self.config.initial_step_size
					}
				}
			};

			// Scale+shrink: scale direction once, shrink in-place per trial.
			// One copy instead of one per trial.
			manifold.copy_tangent(&mut scratch, &direction);
			manifold.scale_tangent(alpha, &mut scratch);

			let mut ls_success = false;
			let mut ls_evals: usize = 0;

			while ls_evals < self.config.ls_max_iterations {
				manifold.retract(&current_point, &scratch, &mut candidate_point, &mut man_ws);
				let candidate_cost = problem.cost(&candidate_point, &mut prob_ws, &mut man_ws);
				fn_evals += 1;
				ls_evals += 1;

				if candidate_cost
					<= current_cost + self.config.sufficient_decrease * alpha * dir_deriv
				{
					current_cost = candidate_cost;
					ls_success = true;
					break;
				}

				// Shrink in-place
				manifold.scale_tangent(self.config.contraction_factor, &mut scratch);
				alpha *= self.config.contraction_factor;
			}

			// If no decrease found at all, reject step.
			if !ls_success {
				// Check if we at least found SOME decrease (even if not
				// satisfying the sufficient decrease condition).
				let last_cost = problem.cost(&candidate_point, &mut prob_ws, &mut man_ws);
				if last_cost < current_cost {
					current_cost = last_cost;
				} else {
					termination = TerminationReason::LineSearchFailed;
					break;
				}
			}

			// ── Adapt step size for next iteration ──────────────────────
			// Strategy from Manopt/Pymanopt:
			// - 2 evals (exactly 1 backtrack): we're in the right ballpark → keep α.
			// - 1 eval or 3+ evals: either we were too conservative or α got
			//   shrunk a lot → double α so we try further next time.
			ls_prev_alpha = Some(if ls_evals == 2 { alpha } else { alpha + alpha });

			// -- C. State Update --
			std::mem::swap(&mut previous_point, &mut current_point);
			std::mem::swap(&mut current_point, &mut candidate_point);

			std::mem::swap(&mut prev_gradient, &mut gradient);
			std::mem::swap(&mut prev_direction, &mut direction);
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
			} else if let Some(val_tol) = stopping_criterion.function_tolerance {
				if <T as Float>::abs(current_cost - _new_cost) < val_tol {
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
