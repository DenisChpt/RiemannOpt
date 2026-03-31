//! # Riemannian Newton Method
//!
//! This module implements the Newton method for optimization on Riemannian manifolds.
//! Newton's method uses second-order information (Hessian) to achieve quadratic
//! convergence near the optimum, making it one of the most powerful optimization
//! algorithms when applicable.
//!
//! ## Mathematical Foundation
//!
//! At each point x_k ∈ ℳ, we solve the Newton equation:
//! ```text
//! Hess f(x_k)[η_k] = -grad f(x_k)
//! ```
//! The system is solved approximately using a linear Conjugate Gradient (tCG)
//! method, which only requires Hessian-vector products.

use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

use crate::{
	manifold::Manifold,
	problem::Problem,
	solver::{Solver, SolverResult, StoppingCriterion, TerminationReason},
	types::Scalar,
};

/// Configuration for the Riemannian Newton method
#[derive(Debug, Clone)]
pub struct NewtonConfig<T: Scalar> {
	/// Regularization parameter for Hessian (to ensure positive definiteness)
	pub hessian_regularization: T,
	/// Maximum number of CG iterations for solving Newton system
	pub max_cg_iterations: usize,
	/// Tolerance for CG solver relative to gradient norm
	pub cg_tolerance_factor: T,
	/// Armijo line search decrease parameter (c1)
	pub armijo_c: T,
	/// Line search backtracking step multiplier (rho)
	pub backtrack_rho: T,
}

impl<T: Scalar> Default for NewtonConfig<T> {
	fn default() -> Self {
		Self {
			hessian_regularization: <T as Scalar>::from_f64(1e-8),
			max_cg_iterations: 100,
			cg_tolerance_factor: <T as Scalar>::from_f64(0.1),
			armijo_c: <T as Scalar>::from_f64(1e-4),
			backtrack_rho: <T as Scalar>::from_f64(0.5),
		}
	}
}

impl<T: Scalar> NewtonConfig<T> {
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_regularization(mut self, reg: T) -> Self {
		self.hessian_regularization = reg;
		self
	}

	pub fn with_cg_params(mut self, max_iter: usize, tol_factor: T) -> Self {
		self.max_cg_iterations = max_iter;
		self.cg_tolerance_factor = tol_factor;
		self
	}
}

/// Riemannian Newton method optimizer
#[derive(Debug)]
pub struct Newton<T: Scalar> {
	config: NewtonConfig<T>,
}

impl<T: Scalar> Newton<T> {
	pub fn new(config: NewtonConfig<T>) -> Self {
		Self { config }
	}

	pub fn with_default_config() -> Self {
		Self::new(NewtonConfig::default())
	}

	/// Solves the Newton system Hess_f(x)[d] = −grad_f(x) using Truncated
	/// Conjugate Gradient (tCG).
	///
	/// All inner-loop updates use `axpy_tangent` (y ← y + α·x) so no
	/// temporary buffer is needed — the former `cg_temp` is eliminated.
	///
	/// # Returns
	/// Whether the returned direction is guaranteed to be a descent direction.
	fn solve_newton_system<M, P>(
		&self,
		problem: &P,
		manifold: &M,
		point: &M::Point,
		gradient: &M::TangentVector,
		grad_norm: T,
		direction: &mut M::TangentVector,
		cg_r: &mut M::TangentVector,
		cg_p: &mut M::TangentVector,
		cg_hp: &mut M::TangentVector,
		prob_ws: &mut P::Workspace,
		man_ws: &mut M::Workspace,
	) -> bool
	where
		M: Manifold<T>,
		P: Problem<T, M>,
	{
		// Target residual norm
		let target_res_norm = <T as Float>::min(
			grad_norm * self.config.cg_tolerance_factor,
			grad_norm * grad_norm,
		);

		// d = 0
		manifold.scale_tangent(T::zero(), direction);

		// r = −grad
		manifold.copy_tangent(cg_r, gradient);
		manifold.scale_tangent(-T::one(), cg_r);

		// p = r
		manifold.copy_tangent(cg_p, cg_r);

		let mut rr = manifold.inner_product(point, cg_r, cg_r, man_ws);

		let reg = self.config.hessian_regularization;

		for cg_iter in 0..self.config.max_cg_iterations {
			// hp = Hess(p) + reg · p
			problem
				.riemannian_hessian_vector_product(manifold, point, cg_p, cg_hp, prob_ws, man_ws);
			manifold.axpy_tangent(reg, cg_p, cg_hp);

			let php = manifold.inner_product(point, cg_p, cg_hp, man_ws);

			// Negative curvature → stop
			if php <= T::zero() {
				if cg_iter == 0 {
					// p is −grad; take it as steepest-descent fallback.
					manifold.copy_tangent(direction, cg_p);
				}
				// else: current `direction` is already a valid descent direction
				return true;
			}

			let alpha = rr / php;

			// d ← d + α · p
			manifold.axpy_tangent(alpha, cg_p, direction);

			// r ← r − α · hp
			manifold.axpy_tangent(-alpha, cg_hp, cg_r);

			let new_rr = manifold.inner_product(point, cg_r, cg_r, man_ws);

			// Convergence check
			if new_rr.sqrt() <= target_res_norm {
				break;
			}

			let beta = new_rr / rr;

			// p ← r + β · p   (= β·p then p += r)
			manifold.scale_tangent(beta, cg_p);
			manifold.add_tangents(cg_p, cg_r);

			rr = new_rr;
		}

		true
	}
}

impl<T: Scalar> Solver<T> for Newton<T> {
	fn name(&self) -> &str {
		"Riemannian Newton (tCG)"
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
		let mut direction = manifold.allocate_tangent();

		// tCG buffers.  After tCG returns, cg_r is reused as the
		// line-search scratch buffer (avoiding a separate allocation).
		let mut cg_r = manifold.allocate_tangent();
		let mut cg_p = manifold.allocate_tangent();
		let mut cg_hp = manifold.allocate_tangent();

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
		// 3. Optimization Loop (Hot Path — Zero Allocation)
		// ════════════════════════════════════════════════════════════════════
		while termination == TerminationReason::MaxIterations && iteration < max_iter {
			// -- A. Solve Newton System --
			self.solve_newton_system(
				problem,
				manifold,
				&current_point,
				&gradient,
				grad_norm,
				&mut direction,
				&mut cg_r,
				&mut cg_p,
				&mut cg_hp,
				&mut prob_ws,
				&mut man_ws,
			);

			// Fallback to steepest descent if tCG produced a non-descent direction
			let mut dir_deriv =
				manifold.inner_product(&current_point, &gradient, &direction, &mut man_ws);
			if dir_deriv >= T::zero() {
				manifold.copy_tangent(&mut direction, &gradient);
				manifold.scale_tangent(-T::one(), &mut direction);
				dir_deriv = -(grad_norm * grad_norm);
			}

			// -- B. Line Search (Armijo Backtracking) --
			// Reuse cg_r as scratch for the scaled direction (tCG is done).
			let scratch = &mut cg_r;
			let mut alpha = T::one();
			let mut ls_success = false;

			while alpha > T::MIN_STEP_SIZE {
				manifold.copy_tangent(scratch, &direction);
				manifold.scale_tangent(alpha, scratch);
				manifold.retract(&current_point, scratch, &mut candidate_point, &mut man_ws);

				let candidate_cost = problem.cost(&candidate_point, &mut prob_ws, &mut man_ws);
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

			// -- C. State Update --
			// O(1) swap instead of O(n·p) memcpy.
			std::mem::swap(&mut current_point, &mut candidate_point);

			let prev_cost = current_cost;
			current_cost = problem.cost_and_gradient(
				manifold,
				&current_point,
				&mut gradient,
				&mut prob_ws,
				&mut man_ws,
			);
			grad_evals += 1;

			grad_norm = manifold.norm(&current_point, &gradient, &mut man_ws);
			iteration += 1;

			// -- D. Stopping Criteria Check --
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
