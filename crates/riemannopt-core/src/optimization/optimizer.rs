//! Core optimizer traits and types for Riemannian optimization.
//!
//! This module provides the foundational abstractions for implementing optimization
//! algorithms on Riemannian manifolds. It defines mathematically rigorous interfaces
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
//! The optimization process follows this general structure:
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
//! - **Optimizer trait**: Universal interface for all optimization algorithms
//! - **OptimizationResult**: Complete optimization outcome with metadata
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
//!
//! # Implementation Guidelines
//!
//! When implementing optimization algorithms:
//!
//! ## Mathematical Rigor
//! - Ensure all operations respect manifold constraints
//! - Use appropriate metrics for distance computations
//! - Implement numerically stable algorithms
//!
//! ## Computational Efficiency
//! - Leverage workspace-based memory management
//! - Minimize manifold constraint violations
//! - Use efficient retraction and transport operations
//!
//! ## Convergence Properties
//! - Prove or empirically verify convergence rates
//! - Handle degenerate cases (singularities, boundary conditions)
//! - Provide meaningful termination diagnostics
//!
//! # Examples
//!
//! ## Basic Gradient Descent
//!
//! ```rust,no_run
//! # use riemannopt_core::prelude::*;
//! # struct GradientDescent { step_size: f64 }
//! # impl Optimizer<f64> for GradientDescent {
//! #   fn name(&self) -> &str { "Gradient Descent" }
//! #   fn optimize<C, M>(&mut self, cost_fn: &C, manifold: &M, 
//! #                     initial_point: &M::Point, criterion: &StoppingCriterion<f64>) 
//! #                     -> Result<OptimizationResult<f64, M::Point>> 
//! #   where C: CostFunction<f64>, M: Manifold<f64> { todo!() }
//! # }
//! # fn main() -> Result<()> {
//! let mut optimizer = GradientDescent { step_size: 0.01 };
//! let criterion = StoppingCriterion::new()
//!     .with_gradient_tolerance(1e-6)
//!     .with_max_iterations(1000);
//! 
//! let result = optimizer.optimize(&cost_fn, &manifold, &x0, &criterion)?;
//! 
//! if result.converged {
//!     println!("Converged to optimal point with value {}", result.value);
//!     println!("Gradient norm: {:?}", result.gradient_norm);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ## Custom Stopping Criteria
//!
//! ```rust,no_run
//! # use riemannopt_core::prelude::*;
//! # use std::time::Duration;
//! let strict_criterion = StoppingCriterion::new()
//!     .with_gradient_tolerance(1e-12)      // Very tight gradient tolerance
//!     .with_function_tolerance(1e-15)      // Minimal function change
//!     .with_point_tolerance(1e-12)         // Tight point convergence
//!     .with_max_time(Duration::from_secs(300))  // 5-minute time limit
//!     .with_target_value(-100.0);          // Stop if f(x) ≤ -100
//! ```

use crate::{
    core::{manifold::Manifold, cost_function::CostFunction},
    error::Result,
    types::Scalar,
};
use std::fmt::Debug;
use std::time::Duration;

/// Comprehensive result of a Riemannian optimization run.
///
/// This structure encapsulates all relevant information from an optimization process,
/// including the solution, convergence diagnostics, and computational statistics.
/// It provides both the mathematical result and practical metadata needed for
/// analysis and debugging.
///
/// # Mathematical Interpretation
///
/// - **point**: The final iterate xₖ ∈ ℳ found by the algorithm
/// - **value**: The objective function value f(xₖ) at the final point
/// - **gradient_norm**: ||grad f(xₖ)||_g measuring first-order optimality
///
/// # Convergence Analysis
///
/// The `converged` flag indicates whether first-order necessary conditions
/// for optimality are satisfied within the specified tolerance. This typically
/// means ||grad f(x)||_g < ε_grad, but may include additional criteria.
///
/// # Computational Diagnostics
///
/// - **Function evaluations**: Total calls to f(x)
/// - **Gradient evaluations**: Total calls to grad f(x) computation
/// - **Duration**: Wall-clock time for the optimization process
/// - **Iterations**: Number of major optimization steps performed
#[derive(Debug, Clone)]
pub struct OptimizationResult<T, P>
where
    T: Scalar,
{
    /// The final point xₖ ∈ ℳ found by the optimizer
    pub point: P,

    /// The objective function value f(xₖ) at the final point
    pub value: T,

    /// The Riemannian gradient norm ||grad f(xₖ)||_g (if computed)
    /// This is the primary measure of first-order optimality
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

impl<T, P> OptimizationResult<T, P>
where
    T: Scalar,
{
    /// Creates a new optimization result.
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

    /// Sets the gradient norm at the optimal point.
    pub fn with_gradient_norm(mut self, norm: T) -> Self {
        self.gradient_norm = Some(norm);
        self
    }

    /// Sets the function evaluation count.
    pub fn with_function_evaluations(mut self, count: usize) -> Self {
        self.function_evaluations = count;
        self
    }

    /// Sets the gradient evaluation count.
    pub fn with_gradient_evaluations(mut self, count: usize) -> Self {
        self.gradient_evaluations = count;
        self
    }
}

/// Mathematical and computational reasons for optimization algorithm termination.
///
/// These reasons provide insight into why the optimization process stopped,
/// enabling proper interpretation of results and debugging of convergence issues.
///
/// # Mathematical Termination
/// - **Converged**: First-order optimality conditions satisfied
/// - **TargetReached**: Objective value below user-specified threshold
///
/// # Computational Limits
/// - **MaxIterations**: Iteration budget exhausted
/// - **MaxTime**: Wall-clock time limit exceeded
/// - **MaxFunctionEvaluations**: Function evaluation budget exhausted
///
/// # Algorithmic Failures
/// - **LineSearchFailed**: Unable to find acceptable step size
/// - **NumericalError**: Numerical instability or invalid operations
///
/// # External Control
/// - **UserTerminated**: Manual termination requested
/// - **CallbackRequest**: Callback function requested early termination
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerminationReason {
    /// First-order necessary conditions satisfied: ||grad f(x)||_g < ε_grad
    Converged,
    /// Objective function value reached user-specified target: f(x) ≤ f_target
    TargetReached,
    /// Maximum iteration count exhausted without convergence
    MaxIterations,
    /// Wall-clock time limit exceeded
    MaxTime,
    /// Function evaluation budget exhausted
    MaxFunctionEvaluations,
    /// Line search failed to find step satisfying Armijo or Wolfe conditions
    LineSearchFailed,
    /// Numerical error: NaN, infinity, or manifold constraint violation
    NumericalError,
    /// User manually requested termination
    UserTerminated,
    /// Progress callback function requested early termination
    CallbackRequest,
}

/// Mathematical and computational stopping criteria for optimization algorithms.
///
/// This structure defines the conditions under which an optimization algorithm
/// should terminate. It combines mathematical convergence criteria with practical
/// computational limits to ensure robust algorithm behavior.
///
/// # Mathematical Convergence Criteria
///
/// ## First-Order Optimality
/// - **gradient_tolerance**: ||grad f(x)||_g < ε_grad
///   Tests whether the Riemannian gradient norm satisfies first-order
///   necessary conditions for optimality.
///
/// ## Progress-Based Criteria
/// - **function_tolerance**: |f(xₖ) - f(xₖ₋₁)| < ε_f
///   Detects stagnation in objective function improvement.
/// - **point_tolerance**: d_ℳ(xₖ, xₖ₋₁) < ε_x
///   Measures convergence in the manifold's intrinsic metric.
///
/// ## Target-Based Termination
/// - **target_value**: f(x) ≤ f_target
///   Stops when objective reaches satisfactory level.
///
/// # Computational Resource Limits
///
/// - **max_iterations**: Upper bound on major algorithm iterations
/// - **max_time**: Wall-clock time budget for optimization
/// - **max_function_evaluations**: Limit on f(x) evaluations
///
/// # Usage Guidelines
///
/// ## Setting Tolerances
/// - Use gradient_tolerance ≈ √machine_epsilon for well-conditioned problems
/// - Set function_tolerance ≈ machine_epsilon for smooth functions
/// - Choose point_tolerance based on manifold geometry and application needs
///
/// ## Resource Planning
/// - Set conservative iteration limits for exploratory runs
/// - Use time limits for real-time applications
/// - Monitor evaluation counts for expensive function evaluations
#[derive(Debug, Clone)]
pub struct StoppingCriterion<T>
where
    T: Scalar,
{
    /// Maximum number of major optimization iterations
    /// Prevents infinite loops and provides computational budget control
    pub max_iterations: Option<usize>,

    /// Maximum wall-clock time for optimization process
    /// Essential for real-time applications with strict timing constraints
    pub max_time: Option<Duration>,

    /// Maximum number of objective function evaluations f(x)
    /// Useful when function evaluation is computationally expensive
    pub max_function_evaluations: Option<usize>,

    /// Tolerance for Riemannian gradient norm: ||grad f(x)||_g < ε_grad
    /// Primary test for first-order necessary optimality conditions
    pub gradient_tolerance: Option<T>,

    /// Tolerance for objective function change: |f(xₖ) - f(xₖ₋₁)| < ε_f
    /// Detects practical convergence when progress stagnates
    pub function_tolerance: Option<T>,

    /// Tolerance for point change in manifold metric: d_ℳ(xₖ, xₖ₋₁) < ε_x
    /// Measures convergence in the manifold's natural geometry
    pub point_tolerance: Option<T>,

    /// Target objective value: stop when f(x) ≤ f_target
    /// Enables early termination when satisfactory solution is found
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
            function_tolerance: Some(<T as Scalar>::from_f64(1e-9)),
            point_tolerance: Some(<T as Scalar>::from_f64(1e-9)),
            target_value: None,
        }
    }
}

impl<T> StoppingCriterion<T>
where
    T: Scalar,
{
    /// Creates a new stopping criterion with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum number of iterations.
    pub fn with_max_iterations(mut self, max_iter: usize) -> Self {
        self.max_iterations = Some(max_iter);
        self
    }

    /// Sets the maximum optimization time.
    pub fn with_max_time(mut self, max_time: Duration) -> Self {
        self.max_time = Some(max_time);
        self
    }

    /// Sets the gradient tolerance.
    pub fn with_gradient_tolerance(mut self, tol: T) -> Self {
        self.gradient_tolerance = Some(tol);
        self
    }

    /// Sets the function value change tolerance.
    pub fn with_function_tolerance(mut self, tol: T) -> Self {
        self.function_tolerance = Some(tol);
        self
    }

    /// Sets the point change tolerance.
    pub fn with_point_tolerance(mut self, tol: T) -> Self {
        self.point_tolerance = Some(tol);
        self
    }

    /// Sets the target objective value.
    pub fn with_target_value(mut self, target: T) -> Self {
        self.target_value = Some(target);
        self
    }
}

/// Universal interface for optimization algorithms on Riemannian manifolds.
///
/// This trait defines the mathematical and computational contract that all
/// Riemannian optimization algorithms must satisfy. It provides a unified
/// interface for minimizing smooth functions f: ℳ → ℝ where ℳ is a
/// Riemannian manifold.
///
/// # Mathematical Framework
///
/// Implementations must respect the Riemannian structure:
/// - Use Riemannian gradients grad f ∈ T_x ℳ instead of Euclidean gradients
/// - Employ retraction maps R_x: T_x ℳ → ℳ for updates
/// - Maintain manifold constraints throughout optimization
/// - Leverage manifold geometry for improved convergence
///
/// # Algorithm Requirements
///
/// ## Convergence Properties
/// - Prove or empirically demonstrate convergence to stationary points
/// - Handle degenerate cases (rank deficiency, boundary conditions)
/// - Provide meaningful termination diagnostics
///
/// ## Numerical Stability
/// - Use numerically stable algorithms (QR, SVD, Cholesky)
/// - Avoid explicit matrix inversions when possible
/// - Handle near-singular cases gracefully
///
/// ## Computational Efficiency
/// - Minimize manifold constraint projections
/// - Leverage workspace-based memory management
/// - Use efficient retraction and transport operations
///
/// # Implementation Guidelines
///
/// ## State Management
/// The optimizer should maintain its own internal state including workspace
/// and algorithm-specific data structures. Each instance is specialized for
/// a specific manifold type and maintains full ownership of its state.
///
/// ## Error Handling
/// Return appropriate errors for:
/// - Manifold constraint violations
/// - Numerical instabilities (NaN, infinity)
/// - Line search failures
/// - Resource exhaustion
///
/// ## Generic Design
/// The trait itself is not generic over manifold types. Instead, methods
/// are generic to allow flexibility while maintaining zero-allocation efficiency.
/// Each optimizer instance is created for a specific manifold type.
pub trait Optimizer<T>: Debug
where
    T: Scalar,
{
    /// Returns a human-readable name identifying the optimization algorithm.
    ///
    /// This is used for logging, debugging, and result reporting.
    /// Examples: "Steepest Descent", "Conjugate Gradient", "LBFGS", "Trust Region"
    fn name(&self) -> &str;

    /// Minimizes the objective function f: ℳ → ℝ on the given Riemannian manifold.
    ///
    /// This is the primary interface for running optimization algorithms.
    /// It performs the complete optimization process from initialization
    /// to convergence, returning comprehensive results.
    ///
    /// # Mathematical Process
    ///
    /// 1. **Initialize**: Start from x₀ ∈ ℳ (initial_point)
    /// 2. **Iterate**: Perform optimization steps until convergence
    /// 3. **Terminate**: Stop when stopping criteria are satisfied
    /// 4. **Report**: Return final point xₖ and optimization metadata
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - Objective function f: ℳ → ℝ to minimize
    /// * `manifold` - Riemannian manifold ℳ with metric structure
    /// * `initial_point` - Starting point x₀ ∈ ℳ for optimization
    /// * `stopping_criterion` - Mathematical and computational termination conditions
    ///
    /// # Returns
    ///
    /// Complete `OptimizationResult` containing:
    /// - Final point xₖ ∈ ℳ and objective value f(xₖ)
    /// - Convergence diagnostics (gradient norm, iteration count)
    /// - Computational statistics (function evaluations, runtime)
    /// - Termination reason and convergence status
    ///
    /// # Errors
    ///
    /// Returns errors for:
    /// - Invalid initial point (not on manifold)
    /// - Numerical instabilities during optimization
    /// - Manifold operation failures
    /// - Resource exhaustion without progress
    fn optimize<M, C>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &M::Point,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, M::Point>>
    where
        M: Manifold<T>,
        C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>;
}
