//! Core solver traits and types for Riemannian optimization.
//!
//! This module provides the foundational abstractions for solving optimization
//! problems on Riemannian manifolds. It defines mathematically rigorous interfaces
//! and data structures that enable efficient numerical optimization while maintaining
//! manifold constraints.
//!
//! # Mathematical Foundation
//!
//! Riemannian optimization seeks to minimize smooth functions f: ℳ → ℝ where ℳ is
//! a Riemannian manifold. The key insight is to leverage the manifold's geometric
//! structure through:
//!
//! - **Riemannian gradient**: grad f ∈ T_x ℳ obtained by projecting ∇f onto T_x ℳ
//! - **Retraction maps**: R_x: T_x ℳ → ℳ for moving along tangent directions
//! - **Vector transport**: Moving tangent vectors between different tangent spaces
//!
//! # Preconditioning
//!
//! Every solver accepts a preconditioner through the unified [`Solver::solve`]
//! interface. A preconditioner P ≈ Hess f reduces the condition number of
//! inner iterative sub-solvers from κ(H) to κ(P⁻¹H).
//!
//! Second-order solvers (Trust Region, Newton-CG, Conjugate Gradient) use
//! the preconditioner in their inner loops. First-order solvers (SGD, Adam)
//! are free to ignore it.
//!
//! When no preconditioning is desired, pass [`IdentityPreconditioner`] — a
//! zero-sized type that the compiler eliminates entirely:
//!
//! ```ignore
//! use riemannopt_core::preconditioner::IdentityPreconditioner;
//!
//! // No preconditioning:
//! let result = solver.solve(&prob, &mfld, &x0, &mut IdentityPreconditioner, &stop);
//!
//! // With preconditioning:
//! let mut pre = LbfgsPreconditioner::new(10);
//! let result = solver.solve(&prob, &mfld, &x0, &mut pre, &stop);
//! ```

pub mod adam;
pub mod conjugate_gradient;
pub mod lbfgs;
pub mod natural_gradient;
pub mod newton;
pub mod sgd;
pub mod trust_region;

pub use adam::{Adam, AdamConfig};
pub use conjugate_gradient::{CGConfig, ConjugateGradient};
pub use lbfgs::{LBFGSConfig, LBFGS};
pub use newton::{Newton, NewtonConfig};
pub use sgd::{MomentumMethod, SGDConfig, SGD};
pub use trust_region::{TrustRegion, TrustRegionConfig};

use crate::{manifold::Manifold, preconditioner::Preconditioner, problem::Problem, types::Scalar};
use std::fmt::Debug;
use std::time::Duration;

// ════════════════════════════════════════════════════════════════════════════
//  Solver trait
// ════════════════════════════════════════════════════════════════════════════

/// Universal interface for optimization solvers on Riemannian manifolds.
///
/// A single [`solve`](Self::solve) method handles both preconditioned and
/// unpreconditioned optimization. Solvers that benefit from preconditioning
/// use `Pre` in their inner loops; solvers that do not simply ignore it.
///
/// Passing [`IdentityPreconditioner`] produces zero overhead — the compiler
/// monomorphises the ZST away entirely. Solvers may additionally check
/// [`pre.is_identity()`](Preconditioner::is_identity) to skip buffer
/// allocation when no preconditioning is active.
///
/// [`IdentityPreconditioner`]: crate::preconditioner::IdentityPreconditioner
pub trait Solver<T>: Debug
where
	T: Scalar,
{
	/// Returns a human-readable name identifying the solver algorithm.
	fn name(&self) -> &str;

	/// Minimizes f: ℳ → ℝ on the given Riemannian manifold.
	///
	/// # Arguments
	///
	/// * `problem`            — the objective function and its derivatives
	/// * `manifold`           — the Riemannian manifold ℳ
	/// * `initial_point`      — starting point x₀ ∈ ℳ
	/// * `preconditioner`     — P⁻¹ ≈ H⁻¹ applied in inner iterative solvers.
	///   Pass `&mut IdentityPreconditioner` when no preconditioning is desired.
	/// * `stopping_criterion` — convergence / budget limits
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
		Pre: Preconditioner<T, M>;
}

// ════════════════════════════════════════════════════════════════════════════
//  Result
// ════════════════════════════════════════════════════════════════════════════

/// Comprehensive result of a Riemannian optimization run.
#[derive(Debug, Clone)]
pub struct SolverResult<T, P>
where
	T: Scalar,
{
	/// The final point xₖ ∈ ℳ found by the solver
	pub point: P,

	/// The objective function value f(xₖ) at the final point
	pub value: T,

	/// The Riemannian gradient norm ||grad f(xₖ)||_g
	pub gradient_norm: Option<T>,

	/// Total number of major optimization iterations performed
	pub iterations: usize,

	/// Total number of objective function evaluations f(x)
	pub function_evaluations: usize,

	/// Total number of gradient evaluations grad f(x)
	pub gradient_evaluations: usize,

	/// Wall-clock time elapsed during optimization
	pub duration: Duration,

	/// Mathematical or computational reason for algorithm termination
	pub termination_reason: TerminationReason,

	/// True if first-order necessary conditions for optimality are satisfied
	pub converged: bool,
}

impl<T, P> SolverResult<T, P>
where
	T: Scalar,
{
	pub fn new(
		point: P,
		value: T,
		iterations: usize,
		duration: Duration,
		termination_reason: TerminationReason,
	) -> Self {
		let converged = matches!(
			termination_reason,
			TerminationReason::Converged | TerminationReason::TargetReached
		);

		Self {
			point,
			value,
			gradient_norm: None,
			iterations,
			function_evaluations: 0,
			gradient_evaluations: 0,
			duration,
			termination_reason,
			converged,
		}
	}

	pub fn with_gradient_norm(mut self, norm: T) -> Self {
		self.gradient_norm = Some(norm);
		self
	}

	pub fn with_function_evaluations(mut self, count: usize) -> Self {
		self.function_evaluations = count;
		self
	}

	pub fn with_gradient_evaluations(mut self, count: usize) -> Self {
		self.gradient_evaluations = count;
		self
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Termination & stopping criteria
// ════════════════════════════════════════════════════════════════════════════

/// Mathematical and computational reasons for solver termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
	Converged,
	TargetReached,
	MaxIterations,
	MaxTime,
	MaxFunctionEvaluations,
	LineSearchFailed,
	NumericalError,
	UserTerminated,
	CallbackRequest,
}

/// Mathematical and computational stopping criteria for solvers.
#[derive(Debug, Clone)]
pub struct StoppingCriterion<T>
where
	T: Scalar,
{
	pub max_iterations: Option<usize>,
	pub max_time: Option<Duration>,
	pub max_function_evaluations: Option<usize>,
	pub gradient_tolerance: Option<T>,
	pub function_tolerance: Option<T>,
	pub point_tolerance: Option<T>,
	pub target_value: Option<T>,
}

impl<T> Default for StoppingCriterion<T>
where
	T: Scalar,
{
	fn default() -> Self {
		Self {
			max_iterations: Some(1000),
			max_time: None,
			max_function_evaluations: None,
			gradient_tolerance: Some(<T as Scalar>::from_f64(1e-6)),
			function_tolerance: None,
			point_tolerance: None,
			target_value: None,
		}
	}
}

impl<T> StoppingCriterion<T>
where
	T: Scalar,
{
	pub fn new() -> Self {
		Self::default()
	}

	pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
		self.max_iterations = Some(max_iter);
		self
	}

	pub fn with_max_time(mut self, max_time: Duration) -> Self {
		self.max_time = Some(max_time);
		self
	}

	pub fn with_gradient_tolerance(mut self, tol: T) -> Self {
		self.gradient_tolerance = Some(tol);
		self
	}

	pub fn with_function_tolerance(mut self, tol: T) -> Self {
		self.function_tolerance = Some(tol);
		self
	}

	pub fn with_point_tolerance(mut self, tol: T) -> Self {
		self.point_tolerance = Some(tol);
		self
	}

	pub fn with_target_value(mut self, target: T) -> Self {
		self.target_value = Some(target);
		self
	}
}
