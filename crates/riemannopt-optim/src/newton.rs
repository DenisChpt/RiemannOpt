//! # Riemannian Newton Method
//!
//! This module implements the Newton method for optimization on Riemannian manifolds.
//! Newton's method uses second-order information (Hessian) to achieve quadratic
//! convergence near the optimum, making it one of the most powerful optimization
//! algorithms when applicable.
//!
//! ## Mathematical Foundation
//!
//! Given a smooth cost function f: ℳ → ℝ on a Riemannian manifold (ℳ, g),
//! the Newton method solves the Newton equation at each iteration to find the
//! optimal search direction.
//!
//! ### Newton Equation
//!
//! At each point x_k ∈ ℳ, we solve:
//! ```text
//! Hess f(x_k)[η_k] = -grad f(x_k)
//! ```
//! where:
//! - Hess f(x_k) is the Riemannian Hessian at x_k
//! - η_k ∈ T_{x_k}ℳ is the Newton direction
//! - grad f(x_k) is the Riemannian gradient
//!
//! ### Algorithm
//!
//! For k = 0, 1, 2, ...:
//! ```text
//! 1. Compute gradient: g_k = grad f(x_k)
//! 2. Solve Newton system: Hess f(x_k)[η_k] = -g_k
//! 3. Line search: α_k = argmin_α f(R_{x_k}(α η_k))
//! 4. Update: x_{k+1} = R_{x_k}(α_k η_k)
//! ```
//!
//! ## Variants
//!
//! ### Gauss-Newton Method
//! For least-squares problems f(x) = ½||r(x)||², approximates:
//! ```text
//! Hess f(x) ≈ J(x)^T J(x)
//! ```
//! where J(x) is the Jacobian of residuals r(x).
//!
//! ### Regularized Newton
//! Adds regularization to ensure positive definiteness:
//! ```text
//! (Hess f(x_k) + λI)[η_k] = -grad f(x_k)
//! ```
//!
//! ## Solving the Newton System
//!
//! The Newton system is solved using Conjugate Gradient (CG) method,
//! which is efficient for large-scale problems and only requires
//! Hessian-vector products.
//!
//! ## Zero-Allocation Architecture
//!
//! This implementation follows a zero-allocation design pattern for optimal performance:
//! - All temporary vectors for CG iterations are pre-allocated in the Workspace
//! - Hessian-vector products reuse workspace buffers to avoid allocations
//! - The workspace is initialized once at the beginning of optimization
//!
//! ## Key Features
//!
//! - **Quadratic convergence**: Near the optimum when Hessian is available
//! - **Hessian approximation**: Finite differences when exact Hessian unavailable
//! - **CG solver**: Efficient for large-scale problems
//! - **Regularization**: Ensures numerical stability
//! - **Line search**: Globalizes convergence
//! - **Zero allocations**: Workspace-based memory management for performance
//!
//! ## References
//!
//! 1. Absil, P. A., Mahony, R., & Sepulchre, R. (2008). Optimization algorithms on matrix manifolds.
//! 2. Nocedal, J., & Wright, S. (2006). Numerical optimization.

use num_traits::Float;
use std::time::Instant;

use riemannopt_core::{
	core::{cost_function::CostFunction, manifold::Manifold},
	error::{ManifoldError, Result},
	optimization::{
		line_search::{BacktrackingLineSearch, LineSearch, LineSearchParams},
		optimizer::{OptimizationResult, Optimizer, StoppingCriterion, TerminationReason},
	},
	types::Scalar,
};

/// Configuration for the Riemannian Newton method
#[derive(Debug, Clone)]
pub struct NewtonConfig<T: Scalar> {
	/// Line search parameters
	pub line_search_params: LineSearchParams<T>,
	/// Regularization parameter for Hessian (to ensure positive definiteness)
	pub hessian_regularization: T,
	/// Use Gauss-Newton approximation (for least-squares problems)
	pub use_gauss_newton: bool,
	/// Maximum number of CG iterations for solving Newton system
	pub max_cg_iterations: usize,
	/// Tolerance for CG solver
	pub cg_tolerance: T,
	/// Whether to use exact Hessian (via cost_fn.hessian_vector_product + ehess2rhess)
	/// or finite differences (default: true)
	pub use_exact_hessian: bool,
}

impl<T: Scalar> Default for NewtonConfig<T> {
	fn default() -> Self {
		Self {
			line_search_params: LineSearchParams::default(),
			hessian_regularization: <T as Scalar>::from_f64(1e-8),
			use_gauss_newton: false,
			max_cg_iterations: 100,
			cg_tolerance: <T as Scalar>::from_f64(1e-6),
			use_exact_hessian: true,
		}
	}
}

impl<T: Scalar> NewtonConfig<T> {
	/// Create a new Newton configuration with default parameters
	pub fn new() -> Self {
		Self::default()
	}

	/// Set the Hessian regularization parameter
	pub fn with_regularization(mut self, reg: T) -> Self {
		self.hessian_regularization = reg;
		self
	}

	/// Enable Gauss-Newton approximation
	pub fn with_gauss_newton(mut self) -> Self {
		self.use_gauss_newton = true;
		self
	}

	/// Set CG solver parameters
	pub fn with_cg_params(mut self, max_iter: usize, tol: T) -> Self {
		self.max_cg_iterations = max_iter;
		self.cg_tolerance = tol;
		self
	}
}

/// Riemannian Newton method optimizer
#[derive(Debug)]
pub struct Newton<T: Scalar> {
	config: NewtonConfig<T>,
}

impl<T: Scalar> Newton<T> {
	/// Create a new Newton optimizer with the given configuration
	pub fn new(config: NewtonConfig<T>) -> Self {
		Self { config }
	}

	/// Solve the Newton system H*d = -g using CG.
	///
	/// CG buffers (d, r, p, hp, temp, temp2) are passed in pre-allocated
	/// to avoid 6 clones per Newton iteration.
	fn solve_newton_system<M: Manifold<T>>(
		&self,
		manifold: &M,
		cost_fn: &impl CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		point: &M::Point,
		gradient: &M::TangentVector,
		euclidean_grad: &M::TangentVector,
		result: &mut M::TangentVector,
		// Pre-allocated CG buffers
		cg_d: &mut M::TangentVector,
		cg_r: &mut M::TangentVector,
		cg_p: &mut M::TangentVector,
		cg_hp: &mut M::TangentVector,
		cg_temp: &mut M::TangentVector,
		cg_temp2: &mut M::TangentVector,
		// Pre-allocated FD HVP buffers
		fd_buf0: &mut M::TangentVector,
		fd_buf1: &mut M::TangentVector,
		fd_buf2: &mut M::TangentVector,
		fd_point: &mut M::Point,
		manifold_ws: &mut M::Workspace,
	) -> Result<()> {
		// Initialize d = 0, r = -gradient, p = r
		manifold.scale_tangent(point, T::zero(), gradient, cg_d)?;
		manifold.scale_tangent(point, -T::one(), gradient, cg_r)?;
		cg_p.clone_from(cg_r);

		for cg_iter in 0..self.config.max_cg_iterations {
			// hp = H*p
			self.hessian_vector_product(
				manifold,
				cost_fn,
				point,
				gradient,
				euclidean_grad,
				cg_p,
				cg_hp,
				fd_buf0,
				fd_buf1,
				fd_buf2,
				fd_point,
				manifold_ws,
			)?;

			// hp = hp + reg*p  (regularization)
			cg_temp.clone_from(cg_hp);
			manifold.axpy_tangent(
				point,
				self.config.hessian_regularization,
				cg_p,
				cg_temp,
				cg_hp,
				cg_temp2,
			)?;

			let rr_inner = manifold.inner_product(point, cg_r, cg_r, manifold_ws)?;
			let php_inner = manifold.inner_product(point, cg_p, cg_hp, manifold_ws)?;

			if php_inner <= T::zero() {
				if cg_iter == 0 {
					cg_d.clone_from(cg_p);
				}
				break;
			}

			let alpha = rr_inner / php_inner;

			// d = d + alpha*p
			cg_temp.clone_from(cg_d);
			manifold.axpy_tangent(point, alpha, cg_p, cg_temp, cg_d, cg_temp2)?;

			// r = r - alpha*hp
			cg_temp.clone_from(cg_r);
			manifold.axpy_tangent(point, -alpha, cg_hp, cg_temp, cg_r, cg_temp2)?;

			let r_new_norm_sq = manifold.inner_product(point, cg_r, cg_r, manifold_ws)?;
			if <T as Float>::sqrt(r_new_norm_sq) < self.config.cg_tolerance {
				break;
			}

			let beta = r_new_norm_sq / rr_inner;

			// p = r + beta*p
			manifold.scale_tangent(point, beta, cg_p, cg_temp2)?;
			manifold.add_tangents(point, cg_r, cg_temp2, cg_p)?;
		}

		result.clone_from(cg_d);
		Ok(())
	}

	/// Compute Riemannian Hessian-vector product.
	///
	/// When `use_exact_hessian` is true, uses the cost function's exact Euclidean
	/// HVP and converts it to Riemannian via `manifold.euclidean_to_riemannian_hessian`.
	/// Otherwise, falls back to finite differences using pre-allocated buffers.
	fn hessian_vector_product<M: Manifold<T>>(
		&self,
		manifold: &M,
		cost_fn: &impl CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		point: &M::Point,
		riemannian_grad: &M::TangentVector,
		euclidean_grad: &M::TangentVector,
		vector: &M::TangentVector,
		result: &mut M::TangentVector,
		// Pre-allocated FD buffers (unused in exact-hessian path)
		fd_buf0: &mut M::TangentVector,
		fd_buf1: &mut M::TangentVector,
		fd_buf2: &mut M::TangentVector,
		fd_point: &mut M::Point,
		manifold_ws: &mut M::Workspace,
	) -> Result<()> {
		if self.config.use_gauss_newton {
			return Err(ManifoldError::NotImplemented {
				feature: "Gauss-Newton approximation".to_string(),
			});
		}

		if self.config.use_exact_hessian {
			let ehvp = cost_fn.hessian_vector_product(point, vector)?;
			manifold.euclidean_to_riemannian_hessian(
				point,
				euclidean_grad,
				&ehvp,
				vector,
				result,
				manifold_ws,
			)?;
			return Ok(());
		}

		// Finite differences on Riemannian gradient (zero alloc via pre-allocated buffers)
		let eps = T::fd_epsilon();

		// fd_buf0 = eps * v
		manifold.scale_tangent(point, eps, vector, fd_buf0)?;

		// fd_point = R_x(eps * v)
		manifold.retract(point, fd_buf0, fd_point, manifold_ws)?;

		// fd_buf0 = euclidean gradient at perturbed point (in-place)
		cost_fn.cost_and_gradient(fd_point, fd_buf0)?;

		// fd_buf1 = Riemannian gradient at perturbed point
		manifold.euclidean_to_riemannian_gradient(fd_point, fd_buf0, fd_buf1, manifold_ws)?;

		// fd_buf2 = τ(grad_plus_riem) transported back to original point
		manifold.parallel_transport(fd_point, point, fd_buf1, fd_buf2, manifold_ws)?;

		// result = (grad_transported - riemannian_grad) / eps
		// fd_buf0 = -riemannian_grad
		manifold.scale_tangent(point, -T::one(), riemannian_grad, fd_buf0)?;
		// result = grad_transported + (-riemannian_grad)
		manifold.add_tangents(point, fd_buf2, fd_buf0, result)?;
		// fd_buf0 = result / eps
		manifold.scale_tangent(point, T::one() / eps, result, fd_buf0)?;

		// Project back to tangent space
		manifold.project_tangent(point, fd_buf0, result, manifold_ws)?;

		Ok(())
	}

	/// Check stopping criteria
	fn check_stopping_criteria<M: Manifold<T>>(
		&self,
		criterion: &StoppingCriterion<T>,
		iteration: usize,
		current_cost: T,
		previous_cost: Option<T>,
		gradient_norm: Option<T>,
		function_evaluations: usize,
		start_time: Instant,
		current_point: &M::Point,
		previous_point: Option<&M::Point>,
		manifold: &M,
	) -> Option<TerminationReason> {
		// Check iteration limit
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
}

impl<T: Scalar> Optimizer<T> for Newton<T> {
	fn name(&self) -> &str {
		"Riemannian Newton"
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

		// Initialize state
		let mut current_point = initial_point.clone();
		let mut has_previous_point = false;

		// Compute initial cost and gradient
		let mut current_cost = cost_fn.cost(&current_point)?;
		let mut previous_cost = None;

		let initial_grad = cost_fn.gradient(&current_point)?;

		// Pre-allocate all buffers once — zero allocations from here on
		let mut ws = CommonWorkspace::new(initial_point, &initial_grad);
		let mut manifold_ws = manifold.create_workspace(initial_point);
		ws.euclidean_grad.clone_from(&initial_grad);

		manifold.euclidean_to_riemannian_gradient(
			&current_point,
			&ws.euclidean_grad,
			&mut ws.riemannian_grad,
			&mut manifold_ws,
		)?;

		let mut gradient_norm =
			manifold.norm(&current_point, &ws.riemannian_grad, &mut manifold_ws)?;

		// Initialize line search
		let mut line_search = BacktrackingLineSearch::new();

		// Pre-allocate CG solver + FD HVP buffers (reused across Newton iterations)
		let mut cg_d = initial_grad.clone();
		let mut cg_r = initial_grad.clone();
		let mut cg_p = initial_grad.clone();
		let mut cg_hp = initial_grad.clone();
		let mut cg_temp = initial_grad.clone();
		let mut cg_temp2 = initial_grad.clone();
		let mut fd_buf0 = initial_grad.clone();
		let mut fd_buf1 = initial_grad.clone();
		let mut fd_buf2 = initial_grad.clone();
		let mut fd_point = initial_point.clone();

		// Tracking variables
		let mut iteration = 0;
		let mut function_evaluations = 1;
		let mut gradient_evaluations = 1;

		// Main optimization loop
		loop {
			// Check stopping criteria
			let prev_ref = if has_previous_point {
				Some(&ws.previous_point)
			} else {
				None
			};
			if let Some(reason) = self.check_stopping_criteria(
				stopping_criterion,
				iteration,
				current_cost,
				previous_cost,
				Some(gradient_norm),
				function_evaluations,
				start_time,
				&current_point,
				prev_ref,
				manifold,
			) {
				return Ok(OptimizationResult::new(
					current_point,
					current_cost,
					iteration,
					start_time.elapsed(),
					reason,
				)
				.with_function_evaluations(function_evaluations)
				.with_gradient_evaluations(gradient_evaluations)
				.with_gradient_norm(gradient_norm));
			}

			// Store previous cost
			previous_cost = Some(current_cost);

			// Solve Newton system: H*d = -g (pre-allocated CG + FD buffers)
			self.solve_newton_system(
				manifold,
				cost_fn,
				&current_point,
				&ws.riemannian_grad,
				&ws.euclidean_grad,
				&mut ws.direction,
				&mut cg_d,
				&mut cg_r,
				&mut cg_p,
				&mut cg_hp,
				&mut cg_temp,
				&mut cg_temp2,
				&mut fd_buf0,
				&mut fd_buf1,
				&mut fd_buf2,
				&mut fd_point,
				&mut manifold_ws,
			)?;

			// Perform line search
			let ls_result = line_search.search(
				cost_fn,
				manifold,
				&current_point,
				current_cost,
				&ws.riemannian_grad,
				&ws.direction,
				&self.config.line_search_params,
				&mut manifold_ws,
			)?;

			// Update counters
			function_evaluations += ls_result.function_evals;

			// Update point: x_{k+1} = R_{x_k}(alpha * d_k)
			manifold.scale_tangent(
				&current_point,
				ls_result.step_size,
				&ws.direction,
				&mut ws.scaled_direction,
			)?;

			manifold.retract(
				&current_point,
				&ws.scaled_direction,
				&mut ws.new_point,
				&mut manifold_ws,
			)?;

			// Swap: previous ← current, current ← new
			std::mem::swap(&mut ws.previous_point, &mut current_point);
			has_previous_point = true;
			std::mem::swap(&mut current_point, &mut ws.new_point);

			// Compute new cost and gradient (in-place)
			current_cost = cost_fn.cost_and_gradient(&current_point, &mut ws.euclidean_grad)?;
			manifold.euclidean_to_riemannian_gradient(
				&current_point,
				&ws.euclidean_grad,
				&mut ws.riemannian_grad,
				&mut manifold_ws,
			)?;
			gradient_norm = manifold.norm(&current_point, &ws.riemannian_grad, &mut manifold_ws)?;

			function_evaluations += 1;
			gradient_evaluations += 1;
			iteration += 1;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use riemannopt_core::core::cost_function::QuadraticCost;
	use riemannopt_core::linalg::{self, VectorOps, VectorView};
	use riemannopt_core::utils::test_manifolds::TestEuclideanManifold;

	#[test]
	fn test_newton_creation() {
		let config = NewtonConfig::<f64>::new();
		let _optimizer = Newton::new(config);
	}

	#[test]
	fn test_newton_on_simple_problem() {
		let config = NewtonConfig::new().with_regularization(1e-6);
		let mut optimizer = Newton::new(config);

		let manifold = TestEuclideanManifold::new(2);
		let cost_fn = QuadraticCost::simple(2);
		let x0 = linalg::Vec::<f64>::from_slice(&[1.0, 1.0]);

		let criterion = StoppingCriterion::new()
			.with_max_iterations(100)
			.with_gradient_tolerance(1e-6);

		let result = optimizer
			.optimize(&cost_fn, &manifold, &x0, &criterion)
			.unwrap();

		assert!(result.iterations < 10); // Newton should converge fast
		assert!(result.gradient_norm.unwrap_or(1.0) < 1e-6);
	}
}
