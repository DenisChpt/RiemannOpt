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
	fn compute_direction<M>(
		&self,
		manifold: &M,
		current_point: &M::Point,
		gradient: &M::TangentVector,
		history: &VecDeque<LBFGSHistoryEntry<T, M::TangentVector>>,
		direction: &mut M::TangentVector,
		scratch0: &mut M::TangentVector,
		scratch1: &mut M::TangentVector,
		alphas: &mut [T],
		manifold_ws: &mut M::Workspace,
	) where
		M: Manifold<T>,
	{
		let m = history.len();

		if m == 0 {
			direction.clone_from(gradient);
			manifold.scale_tangent(-T::one(), direction);
			return;
		}

		// q lives in `direction` initially
		direction.clone_from(gradient);

		// First loop (backward)
		for i in (0..m).rev() {
			let entry = &history[i];
			let s_dot_q = manifold.inner_product(current_point, &entry.s, direction, manifold_ws);
			alphas[i] = entry.rho * s_dot_q;

			// q = q - α_i * y_i
			scratch0.clone_from(&entry.y);
			manifold.scale_tangent(-alphas[i], scratch0);
			manifold.add_tangents(direction, scratch0); // direction += scratch0
		}

		// r = H_0 * q
		let last_entry = &history[m - 1];
		let s_dot_y =
			manifold.inner_product(current_point, &last_entry.s, &last_entry.y, manifold_ws);
		let y_dot_y =
			manifold.inner_product(current_point, &last_entry.y, &last_entry.y, manifold_ws);

		let gamma = if y_dot_y > T::EPSILON {
			s_dot_y / y_dot_y
		} else {
			T::one()
		};
		manifold.scale_tangent(gamma, direction);

		// Second loop (forward)
		for i in 0..m {
			let entry = &history[i];
			let y_dot_r = manifold.inner_product(current_point, &entry.y, direction, manifold_ws);
			let beta = entry.rho * y_dot_r;
			let coeff = alphas[i] - beta;

			// r = r + (α_i - β) * s_i
			scratch1.clone_from(&entry.s);
			manifold.scale_tangent(coeff, scratch1);
			manifold.add_tangents(direction, scratch1); // direction += scratch1
		}

		// Return -r
		manifold.scale_tangent(-T::one(), direction);

		// Project to tangent space for numerical safety
		scratch0.clone_from(direction);
		manifold.project_tangent(current_point, scratch0, direction, manifold_ws);
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

		let mut scratch0 = manifold.allocate_tangent();
		let mut scratch1 = manifold.allocate_tangent();
		let mut s_k = manifold.allocate_tangent();
		let mut y_k = manifold.allocate_tangent();

		let mut prob_ws = problem.create_workspace(manifold, &current_point);
		let mut man_ws = manifold.create_workspace(&current_point);

		let mut history: VecDeque<LBFGSHistoryEntry<T, M::TangentVector>> =
			VecDeque::with_capacity(self.config.memory_size);
		let mut alphas = vec![T::zero(); self.config.memory_size]; // Pre-allocated alpha buffer

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
		// 3. Optimization Loop (Hot Path - Zero Allocation)
		// ════════════════════════════════════════════════════════════════════
		while termination == TerminationReason::MaxIterations && iter < max_iter {
			// -- A. Compute Direction --
			self.compute_direction(
				manifold,
				&current_point,
				&gradient,
				&history,
				&mut direction,
				&mut scratch0,
				&mut scratch1,
				&mut alphas,
				&mut man_ws,
			);

			// Ensure descent direction
			let dir_deriv =
				manifold.inner_product(&current_point, &gradient, &direction, &mut man_ws);
			let dir_deriv = if dir_deriv >= T::zero() {
				// Fallback to Steepest Descent if not a descent direction
				history.clear();
				direction.clone_from(&gradient);
				manifold.scale_tangent(-T::one(), &mut direction);
				-(grad_norm * grad_norm)
			} else {
				dir_deriv
			};

			// -- B. Line Search (Armijo Backtracking) --
			let mut alpha = self.config.initial_step_size;
			let mut ls_success = false;

			while alpha > T::MIN_STEP_SIZE {
				manifold.scale_tangent(alpha, &mut direction);
				manifold.retract(
					&current_point,
					&direction,
					&mut candidate_point,
					&mut man_ws,
				);
				manifold.scale_tangent(T::one() / alpha, &mut direction); // restore

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

			// -- C. Compute s_k and transport history --
			manifold.scale_tangent(alpha, &mut direction);
			s_k.clone_from(&direction); // s_k lives at current_point

			// Transport all history vectors from current_point to candidate_point
			for entry in history.iter_mut() {
				manifold.parallel_transport(
					&current_point,
					&candidate_point,
					&entry.s,
					&mut scratch0,
					&mut man_ws,
				);
				entry.s.clone_from(&scratch0);

				manifold.parallel_transport(
					&current_point,
					&candidate_point,
					&entry.y,
					&mut scratch0,
					&mut man_ws,
				);
				entry.y.clone_from(&scratch0);
			}

			// Transport s_k itself to candidate_point
			manifold.parallel_transport(
				&current_point,
				&candidate_point,
				&s_k,
				&mut scratch0,
				&mut man_ws,
			);
			s_k.clone_from(&scratch0);

			// Transport old gradient to compute y_k
			manifold.parallel_transport(
				&current_point,
				&candidate_point,
				&gradient,
				&mut old_gradient,
				&mut man_ws,
			);

			// Update position
			current_point.clone_from(&candidate_point);

			// -- D. Compute New Gradient --
			let _new_cost_verify = problem.cost_and_gradient(
				manifold,
				&current_point,
				&mut gradient,
				&mut prob_ws,
				&mut man_ws,
			);
			grad_evals += 1;
			grad_norm = manifold.norm(&current_point, &gradient, &mut man_ws);

			// -- E. Compute y_k and Update History --
			// y_k = g_{k+1} - τ(g_k)
			y_k.clone_from(&gradient);
			manifold.scale_tangent(-T::one(), &mut old_gradient);
			manifold.add_tangents(&mut y_k, &old_gradient);

			let s_dot_y = manifold.inner_product(&current_point, &s_k, &y_k, &mut man_ws);
			let s_norm_sq = manifold.inner_product(&current_point, &s_k, &s_k, &mut man_ws);

			let accepted = if self.config.use_cautious_updates {
				let threshold = <T as Scalar>::from_f64(1e-6) * grad_norm;
				s_dot_y / s_norm_sq >= threshold
			} else {
				s_dot_y > T::EPSILON
			};

			if accepted {
				// Normalize s and y to prevent numerical overflow in the history
				let s_norm = s_norm_sq.sqrt();
				if s_norm > T::EPSILON {
					let inv_s = T::one() / s_norm;
					manifold.scale_tangent(inv_s, &mut s_k);
					manifold.scale_tangent(inv_s, &mut y_k);

					let normalized_s_dot_y =
						manifold.inner_product(&current_point, &s_k, &y_k, &mut man_ws);
					if normalized_s_dot_y > T::EPSILON {
						if history.len() >= self.config.memory_size {
							history.pop_front();
						}
						history.push_back(LBFGSHistoryEntry {
							s: s_k.clone(),
							y: y_k.clone(),
							rho: T::one() / normalized_s_dot_y,
						});
					}
				}
			}

			iter += 1;

			// -- F. Stopping Criteria --
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
