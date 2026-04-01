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
//!
//! ## Preconditioning
//!
//! The inner tCG solver supports preconditioning via the preconditioned CG
//! recurrence:
//! ```text
//!   z = P⁻¹ r,   β = ⟨r_new, z_new⟩ / ⟨r_old, z_old⟩,   p = z + β·p
//! ```
//! Passing [`IdentityPreconditioner`] recovers the original algorithm with
//! zero overhead.
//!
//! [`IdentityPreconditioner`]: crate::preconditioner::IdentityPreconditioner

use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

use crate::{
	manifold::Manifold,
	preconditioner::Preconditioner,
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

	/// Solves the Newton system Hess_f(x)[d] = −grad_f(x) using preconditioned
	/// Truncated Conjugate Gradient (tCG).
	///
	/// When `use_pre` is `true`, the CG recurrence uses z = P⁻¹r:
	///   β = ⟨r_new, z_new⟩ / ⟨r_old, z_old⟩,   p = z + β·p
	///
	/// When `use_pre` is `false` (IdentityPreconditioner), z is never
	/// touched and the standard CG recurrence is used. The flag is a
	/// compile-time constant after monomorphisation.
	///
	/// # Returns
	/// Whether the returned direction is guaranteed to be a descent direction.
	#[allow(clippy::too_many_arguments)]
	fn solve_newton_system<M, P, Pre>(
		&self,
		problem: &P,
		manifold: &M,
		point: &M::Point,
		gradient: &M::TangentVector,
		grad_norm: T,
		direction: &mut M::TangentVector,
		cg_r: &mut M::TangentVector,
		cg_z: &mut M::TangentVector,
		cg_p: &mut M::TangentVector,
		cg_hp: &mut M::TangentVector,
		use_pre: bool,
		preconditioner: &mut Pre,
		pre_ws: &mut Pre::Workspace,
		prob_ws: &mut P::Workspace,
		man_ws: &mut M::Workspace,
	) -> bool
	where
		M: Manifold<T>,
		P: Problem<T, M>,
		Pre: Preconditioner<T, M>,
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

		// p = P⁻¹r (or r when unpreconditioned)
		let mut r_dot_z = if use_pre {
			preconditioner.apply(manifold, point, cg_r, cg_z, pre_ws, man_ws);
			manifold.copy_tangent(cg_p, cg_z);
			manifold.inner_product(point, cg_r, cg_z, man_ws)
		} else {
			manifold.copy_tangent(cg_p, cg_r);
			manifold.inner_product(point, cg_r, cg_r, man_ws)
		};

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
					// p is −P⁻¹grad (or −grad); take it as fallback.
					manifold.copy_tangent(direction, cg_p);
				}
				return true;
			}

			let alpha = r_dot_z / php;

			// d ← d + α · p
			manifold.axpy_tangent(alpha, cg_p, direction);

			// r ← r − α · hp
			manifold.axpy_tangent(-alpha, cg_hp, cg_r);

			// ── Preconditioned residual & convergence check ──────────
			let (r_next_norm_sq, r_dot_z_new) = if use_pre {
				preconditioner.apply(manifold, point, cg_r, cg_z, pre_ws, man_ws);
				let rr = manifold.inner_product(point, cg_r, cg_r, man_ws);
				let rz = manifold.inner_product(point, cg_r, cg_z, man_ws);
				(rr, rz)
			} else {
				let rr = manifold.inner_product(point, cg_r, cg_r, man_ws);
				(rr, rr)
			};

			if r_next_norm_sq.sqrt() <= target_res_norm {
				break;
			}

			let beta = r_dot_z_new / r_dot_z;

			// p ← z + β·p  (or r + β·p when unpreconditioned)
			manifold.scale_tangent(beta, cg_p);
			if use_pre {
				manifold.add_tangents(cg_p, cg_z);
			} else {
				manifold.add_tangents(cg_p, cg_r);
			}

			r_dot_z = r_dot_z_new;
		}

		true
	}
}

impl<T: Scalar> Solver<T> for Newton<T> {
	fn name(&self) -> &str {
		"Riemannian Newton (tCG)"
	}

	fn solve<M, P, Pre>(
		&mut self,
		problem: &P,
		manifold: &M,
		initial_point: &M::Point,
		preconditioner: &mut Pre,
		stopping_criterion: &StoppingCriterion<T>,
	) -> SolverResult<T, M::Point>
	where
		M: Manifold<T>,
		P: Problem<T, M>,
		Pre: Preconditioner<T, M>,
	{
		let start_time = Instant::now();
		let use_pre = !preconditioner.is_identity();

		// ════════════════════════════════════════════════════════════════════
		// 1. Memory Allocation (Cold Path)
		// ════════════════════════════════════════════════════════════════════
		let mut current_point = initial_point.clone();
		let mut candidate_point = manifold.allocate_point();

		let mut gradient = manifold.allocate_tangent();
		let mut direction = manifold.allocate_tangent();

		// tCG buffers. After tCG returns, cg_r is reused as line-search
		// scratch and cg_z doubles as old_gradient storage for preconditioner
		// update — zero extra allocation.
		let mut cg_r = manifold.allocate_tangent();
		let mut cg_z = manifold.allocate_tangent();
		let mut cg_p = manifold.allocate_tangent();
		let mut cg_hp = manifold.allocate_tangent();

		let mut prob_ws = problem.create_workspace(manifold, &current_point);
		let mut man_ws = manifold.create_workspace(&current_point);
		let mut pre_ws = preconditioner.create_workspace(manifold, &current_point);

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
			// -- A. Solve Newton System (preconditioned tCG) --
			self.solve_newton_system(
				problem,
				manifold,
				&current_point,
				&gradient,
				grad_norm,
				&mut direction,
				&mut cg_r,
				&mut cg_z,
				&mut cg_p,
				&mut cg_hp,
				use_pre,
				preconditioner,
				&mut pre_ws,
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
				alpha *= self.config.backtrack_rho;
			}

			if !ls_success {
				termination = TerminationReason::LineSearchFailed;
				break;
			}

			// -- C. State Update --
			// Save old gradient into cg_z for preconditioner update.
			if use_pre {
				manifold.copy_tangent(&mut cg_z, &gradient);
			}

			// O(1) swap. After swap: candidate_point = old point.
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

			// Update preconditioner with new curvature pair.
			// The actual step is alpha · direction, but direction lives in
			// T_{old_point}. We use `scratch` (= cg_r) which held the scaled
			// step (alpha · direction) from the last line-search iteration.
			if use_pre {
				// Reconstruct the accepted step: scratch = alpha · direction.
				// scratch was left at the accepted alpha from the line search.
				preconditioner.update(
					manifold,
					&candidate_point, // x_old (after swap)
					&current_point,   // x_new
					scratch,          // step = α·direction ∈ T_{x_old}
					&cg_z,            // grad_old (saved above)
					&gradient,        // grad_new
					&mut pre_ws,
					&mut man_ws,
				);
			}

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
