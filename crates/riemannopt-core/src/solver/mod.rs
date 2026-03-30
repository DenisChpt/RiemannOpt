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
//! # Optimization Framework
//!
//! The solving process follows this general structure:
//!
//! 1. **Initialization**: Start with x₀ ∈ ℳ
//! 2. **Gradient computation**: Compute grad f(xₖ) ∈ T_{xₖ} ℳ
//! 3. **Search direction**: Determine ηₖ ∈ T_{xₖ} ℳ (e.g., -grad f(xₖ))
//! 4. **Line search**: Find step size αₖ > 0
//! 5. **Retraction**: Update xₖ₊₁ = R_{xₖ}(αₖ ηₖ)
//! 6. **Convergence**: Check stopping criteria
//!
//! # Key Components
//!
//! ## Core Abstractions
//! - **Solver trait**: Universal interface for all optimization algorithms
//! - **SolverResult**: Complete optimization outcome with metadata
//!
//! ## Convergence Control
//! - **StoppingCriterion**: Mathematical conditions for algorithm termination:
//!   - **Gradient norm**: ||grad f(x)||_g < ε_grad (first-order optimality)
//!   - **Function change**: |f(xₖ) - f(xₖ₋₁)| < ε_f (stationarity)
//!   - **Point change**: d_ℳ(xₖ, xₖ₋₁) < ε_x (convergence in manifold metric)
//!
//! ## Termination Analysis
//! - **Converged**: First-order necessary conditions satisfied
//! - **TargetReached**: Objective value below specified threshold
//! - **MaxIterations**: Computational budget exhausted
//! - **LineSearchFailed**: Unable to find adequate step size

pub mod adam;
pub mod conjugate_gradient;
pub mod lbfgs;
pub mod natural_gradient;
pub mod newton;
pub mod sgd;
pub mod trust_region;

// Re-export solver types
pub use adam::{Adam, AdamConfig};
pub use conjugate_gradient::{CGConfig, ConjugateGradient};
pub use lbfgs::{LBFGSConfig, LBFGS};
pub use newton::{Newton, NewtonConfig};
pub use sgd::{MomentumMethod, SGDConfig, SGD};
pub use trust_region::{TrustRegion, TrustRegionConfig};

use crate::{manifold::Manifold, problem::Problem, types::Scalar};
use std::fmt::Debug;
use std::time::Duration;

/// Comprehensive result of a Riemannian optimization run.
///
/// This structure encapsulates all relevant information from a solver process,
/// including the solution, convergence diagnostics, and computational statistics.
/// It provides both the mathematical result and practical metadata needed for
/// analysis and debugging.
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
	/// Creates a new solver result.
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

/// Universal interface for optimization solvers on Riemannian manifolds.
pub trait Solver<T>: Debug
where
	T: Scalar,
{
	/// Returns a human-readable name identifying the solver algorithm.
	fn name(&self) -> &str;

	/// Minimizes the objective function f: ℳ → ℝ on the given Riemannian manifold.
	fn solve<M, P>(
		&mut self,
		problem: &P,
		manifold: &M,
		initial_point: &M::Point,
		stopping_criterion: &StoppingCriterion<T>,
	) -> SolverResult<T, M::Point>
	where
		M: Manifold<T>,
		P: Problem<T, M>;
}
