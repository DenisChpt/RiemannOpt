//! Riemannian L-BFGS solver.
//!
//! L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton
//! optimization algorithm. On Riemannian manifolds, it requires transporting
//! all stored vectors to the current tangent space after each step.

use std::collections::VecDeque;
use std::fmt::Debug;
use std::time::Instant;

use crate::{
	manifold::Manifold,
	problem::Problem,
	solver::{Solver, SolverResult, StoppingCriterion, TerminationReason},
	types::Scalar,
};

/// Storage for one iteration's L-BFGS data.
#[derive(Debug)]
pub struct LBFGSHistoryEntry<T, TV>
where
	T: Scalar,
	TV: Clone + Debug + Send + Sync,
{
	pub s: TV,
	pub y: TV,
	pub rho: T,
}

/// Configuration for the L-BFGS solver.
#[derive(Debug, Clone)]
pub struct LBFGSConfig<T: Scalar> {
	pub memory_size: usize,
	pub initial_step_size: T,
	pub use_cautious_updates: bool,
	pub backtrack_rho: T,
	pub armijo_c: T,
}

impl<T: Scalar> Default for LBFGSConfig<T> {
	fn default() -> Self {
		Self {
			memory_size: 10,
			initial_step_size: T::one(),
			use_cautious_updates: true,
			backtrack_rho: <T as Scalar>::from_f64(0.5),
			armijo_c: <T as Scalar>::from_f64(1e-4),
		}
	}
}

impl<T: Scalar> LBFGSConfig<T> {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn with_memory_size(mut self, size: usize) -> Self {
		self.memory_size = size;
		self
	}
	pub fn with_initial_step_size(mut self, step_size: T) -> Self {
		self.initial_step_size = step_size;
		self
	}
	pub fn with_cautious_updates(mut self, cautious: bool) -> Self {
		self.use_cautious_updates = cautious;
		self
	}
}

/// Riemannian L-BFGS solver.
#[derive(Debug)]
pub struct LBFGS<T: Scalar> {
	config: LBFGSConfig<T>,
}

impl<T: Scalar> LBFGS<T> {
	pub fn new(config: LBFGSConfig<T>) -> Self {
		Self { config }
	}

	pub fn with_default_config() -> Self {
		Self::new(LBFGSConfig::default())
	}

	/// Computes the L-BFGS search direction using the two-loop recursion.
	///
	/// Uses `axpy_tangent` (fused y ← y + α·x) to avoid all temporary
	/// copies in the inner loops.  The only remaining copy is the final
	/// project_tangent call (which needs distinct input/output buffers).
	fn compute_direction<M>(
		&self,
		manifold: &M,
		current_point: &M::Point,
		gradient: &M::TangentVector,
		history: &VecDeque<LBFGSHistoryEntry<T, M::TangentVector>>,
		direction: &mut M::TangentVector,
		scratch: &mut M::TangentVector,
		alphas: &mut [T],
		manifold_ws: &mut M::Workspace,
	) where
		M: Manifold<T>,
	{
		let m = history.len();

		if m == 0 {
			manifold.copy_tangent(direction, gradient);
			manifold.scale_tangent(-T::one(), direction);
			return;
		}

		// q ← gradient  (direction serves as q)
		manifold.copy_tangent(direction, gradient);

		// ── First loop (backward) ────────────────────────────────────────
		for i in (0..m).rev() {
			let entry = &history[i];
			let s_dot_q = manifold.inner_product(current_point, &entry.s, direction, manifold_ws);
			alphas[i] = entry.rho * s_dot_q;
			manifold.axpy_tangent(-alphas[i], &entry.y, direction);
		}

		// ── Initial Hessian scaling ──────────────────────────────────────
		let last = &history[m - 1];
		let s_dot_y = manifold.inner_product(current_point, &last.s, &last.y, manifold_ws);
		let y_dot_y = manifold.inner_product(current_point, &last.y, &last.y, manifold_ws);

		let gamma = if y_dot_y > T::EPSILON {
			s_dot_y / y_dot_y
		} else {
			T::one()
		};
		manifold.scale_tangent(gamma, direction);

		// ── Second loop (forward) ────────────────────────────────────────
		for i in 0..m {
			let entry = &history[i];
			let y_dot_r = manifold.inner_product(current_point, &entry.y, direction, manifold_ws);
			let beta = entry.rho * y_dot_r;
			manifold.axpy_tangent(alphas[i] - beta, &entry.s, direction);
		}

		// direction = −r
		manifold.scale_tangent(-T::one(), direction);

		// Re-project for numerical safety (needs distinct src / dst)
		manifold.copy_tangent(scratch, direction);
		manifold.project_tangent(current_point, scratch, direction, manifold_ws);
	}
}

impl<T: Scalar> Solver<T> for LBFGS<T> {
	fn name(&self) -> &str {
		"Riemannian L-BFGS"
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
		let mut old_gradient = manifold.allocate_tangent();
		let mut direction = manifold.allocate_tangent();

		let mut scratch = manifold.allocate_tangent();
		let mut s_k = manifold.allocate_tangent();
		let mut y_k = manifold.allocate_tangent();

		let mut prob_ws = problem.create_workspace(manifold, &current_point);
		let mut man_ws = manifold.create_workspace(&current_point);

		let mut history: VecDeque<LBFGSHistoryEntry<T, M::TangentVector>> =
			VecDeque::with_capacity(self.config.memory_size);
		let mut alphas = vec![T::zero(); self.config.memory_size];

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
		while termination == TerminationReason::MaxIterations && iter < max_iter {
			// -- A. Compute Direction (two-loop recursion) --
			self.compute_direction(
				manifold,
				&current_point,
				&gradient,
				&history,
				&mut direction,
				&mut scratch,
				&mut alphas,
				&mut man_ws,
			);

			// Ensure descent direction
			let dir_deriv =
				manifold.inner_product(&current_point, &gradient, &direction, &mut man_ws);
			let dir_deriv = if dir_deriv >= T::zero() {
				// Fallback to steepest descent
				history.clear();
				manifold.copy_tangent(&mut direction, &gradient);
				manifold.scale_tangent(-T::one(), &mut direction);
				-(grad_norm * grad_norm)
			} else {
				dir_deriv
			};

			// -- B. Line Search (Armijo Backtracking) --
			// Scale direction in-place rather than copy+scale each trial.
			// After the loop, direction = alpha_final · d_original.
			let mut alpha = self.config.initial_step_size;
			let mut ls_success = false;

			manifold.scale_tangent(alpha, &mut direction);

			while alpha > T::MIN_STEP_SIZE {
				manifold.retract(
					&current_point,
					&direction,
					&mut candidate_point,
					&mut man_ws,
				);

				let candidate_cost = problem.cost(&candidate_point, &mut prob_ws, &mut man_ws);
				fn_evals += 1;

				if candidate_cost <= current_cost + self.config.armijo_c * alpha * dir_deriv {
					current_cost = candidate_cost;
					ls_success = true;
					break;
				}

				// Shrink direction in-place: direction *= ρ
				manifold.scale_tangent(self.config.backtrack_rho, &mut direction);
				alpha *= self.config.backtrack_rho;
			}

			if !ls_success {
				termination = TerminationReason::LineSearchFailed;
				break;
			}

			// -- C. Compute s_k = alpha · direction (at current_point) --
			manifold.copy_tangent(&mut s_k, &direction);

			// -- D. Transport history vectors to candidate_point --
			// Uses swap to move the transported result into the entry
			// without a redundant memcpy.
			for entry in history.iter_mut() {
				manifold.parallel_transport(
					&current_point,
					&candidate_point,
					&entry.s,
					&mut scratch,
					&mut man_ws,
				);
				std::mem::swap(&mut entry.s, &mut scratch);

				manifold.parallel_transport(
					&current_point,
					&candidate_point,
					&entry.y,
					&mut scratch,
					&mut man_ws,
				);
				std::mem::swap(&mut entry.y, &mut scratch);
			}

			// Transport s_k to candidate_point
			manifold.parallel_transport(
				&current_point,
				&candidate_point,
				&s_k,
				&mut scratch,
				&mut man_ws,
			);
			std::mem::swap(&mut s_k, &mut scratch);

			// Transport current gradient to candidate_point → old_gradient
			manifold.parallel_transport(
				&current_point,
				&candidate_point,
				&gradient,
				&mut old_gradient,
				&mut man_ws,
			);

			// -- E. Move to candidate_point --
			std::mem::swap(&mut current_point, &mut candidate_point);

			// -- F. Compute New Gradient --
			let _new_cost = problem.cost_and_gradient(
				manifold,
				&current_point,
				&mut gradient,
				&mut prob_ws,
				&mut man_ws,
			);
			grad_evals += 1;
			grad_norm = manifold.norm(&current_point, &gradient, &mut man_ws);

			// -- G. Compute y_k = g_{k+1} − τ(g_k) and Update History --
			manifold.copy_tangent(&mut y_k, &gradient);
			manifold.axpy_tangent(-T::one(), &old_gradient, &mut y_k);

			let s_dot_y = manifold.inner_product(&current_point, &s_k, &y_k, &mut man_ws);
			let s_norm_sq = manifold.inner_product(&current_point, &s_k, &s_k, &mut man_ws);

			let accepted = if self.config.use_cautious_updates {
				let threshold = <T as Scalar>::from_f64(1e-6) * grad_norm;
				s_dot_y / s_norm_sq >= threshold
			} else {
				s_dot_y > T::EPSILON
			};

			if accepted {
				let s_norm = s_norm_sq.sqrt();
				if s_norm > T::EPSILON {
					let inv_s = T::one() / s_norm;
					manifold.scale_tangent(inv_s, &mut s_k);
					manifold.scale_tangent(inv_s, &mut y_k);

					let normalized_s_dot_y =
						manifold.inner_product(&current_point, &s_k, &y_k, &mut man_ws);

					if normalized_s_dot_y > T::EPSILON {
						let rho = T::one() / normalized_s_dot_y;

						if history.len() >= self.config.memory_size {
							let mut old = history.pop_front().unwrap();
							std::mem::swap(&mut old.s, &mut s_k);
							std::mem::swap(&mut old.y, &mut y_k);
							old.rho = rho;
							history.push_back(old);
						} else {
							history.push_back(LBFGSHistoryEntry {
								s: s_k.clone(),
								y: y_k.clone(),
								rho,
							});
						}
					}
				}
			}

			iter += 1;

			// -- H. Stopping Criteria --
			if grad_norm <= grad_tol {
				termination = TerminationReason::Converged;
			} else if let Some(val_tol) = stopping_criterion.function_tolerance {
				if <T as num_traits::Float>::abs(current_cost - _new_cost) < val_tol {
					termination = TerminationReason::Converged;
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
