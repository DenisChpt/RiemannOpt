//! # Riemannian Trust Region Solver
//!
//! Trust region methods are robust second-order optimization algorithms that use a
//! local quadratic model of the objective function within a "trust region" where
//! the model is assumed to be accurate.
//!
//! At each iteration, the solver computes a step by approximately minimizing the
//! quadratic model:
//! ```text
//! m(s) = f(x) + ⟨grad f(x), s⟩ + ½⟨s, Hess f(x)[s]⟩
//! ```
//! subject to ‖s‖ ≤ Δ, using the Steihaug Truncated Conjugate Gradient (tCG) method.

use num_traits::Float;
use std::fmt::Debug;
use std::time::Instant;

use crate::{
	manifold::Manifold,
	problem::Problem,
	solver::{Solver, SolverResult, StoppingCriterion, TerminationReason},
	types::Scalar,
};

/// Configuration for the Trust Region solver.
#[derive(Debug, Clone)]
pub struct TrustRegionConfig<T: Scalar> {
	pub initial_radius: T,
	pub max_radius: T,
	pub min_radius: T,
	pub acceptance_ratio: T,
	pub increase_threshold: T,
	pub decrease_threshold: T,
	pub increase_factor: T,
	pub decrease_factor: T,
	pub max_cg_iterations: Option<usize>,
	/// TCG linear convergence target (kappa)
	pub kappa: T,
	/// TCG superlinear convergence exponent (theta)
	pub theta: T,
}

impl<T: Scalar> Default for TrustRegionConfig<T> {
	fn default() -> Self {
		Self {
			initial_radius: <T as Scalar>::from_f64(1.0),
			max_radius: <T as Scalar>::from_f64(10.0),
			min_radius: <T as Scalar>::from_f64(1e-6),
			acceptance_ratio: <T as Scalar>::from_f64(0.1),
			increase_threshold: <T as Scalar>::from_f64(0.75),
			decrease_threshold: <T as Scalar>::from_f64(0.25),
			increase_factor: <T as Scalar>::from_f64(2.0),
			decrease_factor: <T as Scalar>::from_f64(0.25),
			max_cg_iterations: None,
			kappa: <T as Scalar>::from_f64(0.1),
			theta: T::one(),
		}
	}
}

impl<T: Scalar> TrustRegionConfig<T> {
	pub fn new() -> Self {
		Self::default()
	}
	pub fn with_initial_radius(mut self, radius: T) -> Self {
		self.initial_radius = radius;
		self
	}
	pub fn with_max_radius(mut self, radius: T) -> Self {
		self.max_radius = radius;
		self
	}
	pub fn with_min_radius(mut self, radius: T) -> Self {
		self.min_radius = radius;
		self
	}
	pub fn with_acceptance_ratio(mut self, ratio: T) -> Self {
		self.acceptance_ratio = ratio;
		self
	}
	pub fn with_max_cg_iterations(mut self, max_iter: usize) -> Self {
		self.max_cg_iterations = Some(max_iter);
		self
	}
}

/// Riemannian Trust Region optimizer.
#[derive(Debug)]
pub struct TrustRegion<T: Scalar> {
	config: TrustRegionConfig<T>,
}

impl<T: Scalar> TrustRegion<T> {
	pub fn new(config: TrustRegionConfig<T>) -> Self {
		Self { config }
	}

	pub fn with_default_config() -> Self {
		Self::new(TrustRegionConfig::default())
	}

	/// Finds the positive root tau such that ||s + tau * p|| = radius.
	fn boundary_intersection<M>(
		&self,
		s: &M::TangentVector,
		p: &M::TangentVector,
		radius: T,
		manifold: &M,
		point: &M::Point,
		manifold_ws: &mut M::Workspace,
	) -> T
	where
		M: Manifold<T>,
	{
		// ||s + tau*p||^2 = radius^2
		// ||p||^2 tau^2 + 2<s,p> tau + ||s||^2 - radius^2 = 0
		let a = manifold.inner_product(point, p, p, manifold_ws);
		let b = <T as Scalar>::from_f64(2.0) * manifold.inner_product(point, s, p, manifold_ws);
		let c = manifold.inner_product(point, s, s, manifold_ws) - radius * radius;

		let discriminant = b * b - <T as Scalar>::from_f64(4.0) * a * c;
		let sqrt_disc = <T as Float>::sqrt(<T as Float>::max(T::zero(), discriminant));

		// We only care about the positive root since we step forward
		(-b + sqrt_disc) / (<T as Scalar>::from_f64(2.0) * a)
	}

	/// Solves the trust region subproblem using Steihaug Truncated-CG (tCG).
	///
	/// # Returns
	/// A boolean indicating if the trust region boundary was hit.
	fn solve_tcg<M, P>(
		&self,
		problem: &P,
		manifold: &M,
		point: &M::Point,
		gradient: &M::TangentVector,
		grad_norm: T,
		radius: T,
		s: &mut M::TangentVector,    // Step output
		r: &mut M::TangentVector,    // Residual buffer
		p: &mut M::TangentVector,    // CG direction buffer
		hp: &mut M::TangentVector,   // Hessian-vector product buffer
		temp: &mut M::TangentVector, // Scratch buffer
		prob_ws: &mut P::Workspace,
		man_ws: &mut M::Workspace,
	) -> bool
	where
		M: Manifold<T>,
		P: Problem<T, M>,
	{
		let max_iter = self
			.config
			.max_cg_iterations
			.unwrap_or_else(|| manifold.dimension());

		// Target residual norm: ||r_0|| * min(||r_0||^theta, kappa)
		let target_norm = grad_norm
			* <T as Float>::min(
				<T as Float>::powf(grad_norm, self.config.theta),
				self.config.kappa,
			);

		// s = 0
		manifold.scale_tangent(T::zero(), s);
		// r = grad
		r.clone_from(gradient);
		// p = -r
		p.clone_from(r);
		manifold.scale_tangent(-T::one(), p);

		let mut r_norm_sq = manifold.inner_product(point, r, r, man_ws);

		for _ in 0..max_iter {
			// Hp = Hess(p)
			problem.riemannian_hessian_vector_product(manifold, point, p, hp, prob_ws, man_ws);

			let kappa_val = manifold.inner_product(point, p, hp, man_ws);

			// Negative curvature detected: intersect with boundary and stop
			if kappa_val <= T::zero() {
				let tau = self.boundary_intersection(s, p, radius, manifold, point, man_ws);
				manifold.axpy_tangent(tau, p, s); // s = s + tau*p
				return true;
			}

			let alpha = r_norm_sq / kappa_val;

			// s_next = s + alpha * p
			temp.clone_from(s);
			manifold.axpy_tangent(alpha, p, temp);

			// Check if step exceeds trust region radius
			let s_next_norm = manifold.norm(point, temp, man_ws);
			if s_next_norm >= radius {
				let tau = self.boundary_intersection(s, p, radius, manifold, point, man_ws);
				manifold.axpy_tangent(tau, p, s);
				return true;
			}

			s.clone_from(temp);

			// r_next = r + alpha * Hp
			manifold.axpy_tangent(alpha, hp, r);

			let r_next_norm_sq = manifold.inner_product(point, r, r, man_ws);
			if <T as Float>::sqrt(r_next_norm_sq) <= target_norm {
				return false; // Interior convergence
			}

			let beta = r_next_norm_sq / r_norm_sq;

			// p = -r_next + beta * p
			temp.clone_from(r);
			manifold.scale_tangent(-T::one(), temp); // temp = -r
			manifold.scale_tangent(beta, p);
			manifold.add_tangents(p, temp); // p = -r + beta * p

			r_norm_sq = r_next_norm_sq;
		}

		false // Reached max iterations without hitting boundary
	}
}

impl<T: Scalar> Solver<T> for TrustRegion<T> {
	fn name(&self) -> &str {
		"Riemannian Trust Region"
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
		let mut step_s = manifold.allocate_tangent();

		// tCG Buffers
		let mut cg_r = manifold.allocate_tangent();
		let mut cg_p = manifold.allocate_tangent();
		let mut cg_hp = manifold.allocate_tangent();
		let mut cg_temp = manifold.allocate_tangent();

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

		// Dimension-aware initial trust region radius
		let dim_f = manifold.dimension() as f64;
		let typical_dist = <T as Scalar>::from_f64(dim_f.sqrt());
		let mut trust_radius = <T as Float>::max(
			self.config.initial_radius,
			typical_dist / <T as Scalar>::from_f64(8.0),
		);
		let max_radius = <T as Float>::max(self.config.max_radius, typical_dist);

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
			// Check minimum trust region radius
			if trust_radius < self.config.min_radius {
				termination = TerminationReason::Converged;
				break;
			}

			// -- A. Solve Trust Region Subproblem (tCG) --
			let boundary_hit = self.solve_tcg(
				problem,
				manifold,
				&current_point,
				&gradient,
				grad_norm,
				trust_radius,
				&mut step_s,
				&mut cg_r,
				&mut cg_p,
				&mut cg_hp,
				&mut cg_temp,
				&mut prob_ws,
				&mut man_ws,
			);

			// -- B. Compute Predicted Reduction --
			// m(0) - m(s) = - (<g, s> + 0.5 <s, Hs>)
			// We need Hs, so we do one final HVP
			problem.riemannian_hessian_vector_product(
				manifold,
				&current_point,
				&step_s,
				&mut cg_hp, // Use cg_hp to store Hs
				&mut prob_ws,
				&mut man_ws,
			);

			let g_dot_s = manifold.inner_product(&current_point, &gradient, &step_s, &mut man_ws);
			let s_dot_hs = manifold.inner_product(&current_point, &step_s, &mut cg_hp, &mut man_ws);
			let predicted_reduction = -(g_dot_s + <T as Scalar>::from_f64(0.5) * s_dot_hs);

			// -- C. Evaluate Actual Reduction --
			manifold.retract(&current_point, &step_s, &mut candidate_point, &mut man_ws);
			let trial_cost = problem.cost(&candidate_point);
			fn_evals += 1;

			let actual_reduction = current_cost - trial_cost;

			// -- D. Compute Ratio (rho) --
			// Small regularization to avoid division by zero or numerical artifacts
			let reg = <T as Float>::max(T::one(), <T as Float>::abs(current_cost))
				* T::epsilon()
				* <T as Scalar>::from_f64(1e3);

			let rho = if <T as Float>::abs(predicted_reduction) > reg {
				actual_reduction / predicted_reduction
			} else if actual_reduction >= T::zero() {
				// Near convergence, both reductions ~ 0. Accept step to avoid getting stuck.
				<T as Scalar>::from_f64(0.75)
			} else {
				T::zero()
			};

			// -- E. Accept or Reject Step --
			if rho >= self.config.acceptance_ratio {
				// Step accepted
				current_point.clone_from(&candidate_point);
				current_cost = trial_cost;

				// We only need to compute gradient if step is accepted
				let _ = problem.cost_and_gradient(
					manifold,
					&current_point,
					&mut gradient,
					&mut prob_ws,
					&mut man_ws,
				);
				grad_evals += 1;
				grad_norm = manifold.norm(&current_point, &gradient, &mut man_ws);
			}

			// -- F. Update Trust Region Radius --
			if rho < self.config.decrease_threshold {
				// Poor model prediction: shrink trust region
				trust_radius = trust_radius * self.config.decrease_factor;
			} else if rho > self.config.increase_threshold && boundary_hit {
				// Excellent prediction & hit boundary: expand trust region
				trust_radius =
					<T as Float>::min(trust_radius * self.config.increase_factor, max_radius);
			}

			iteration += 1;

			// -- G. Stopping Criteria Check --
			if grad_norm <= grad_tol {
				termination = TerminationReason::Converged;
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
