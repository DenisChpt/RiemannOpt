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
//! - **OptimizerState**: Mutable state tracking during optimization
//!
//! ## Convergence Control 
//! - **StoppingCriterion**: Mathematical conditions for algorithm termination:
//!   - **Gradient norm**: ||grad f(x)||_g < ε_grad (first-order optimality)
//!   - **Function change**: |f(xₖ) - f(xₖ₋₁)| < ε_f (stationarity)
//!   - **Point change**: d_ℳ(xₖ, xₖ₋₁) < ε_x (convergence in manifold metric)
//! - **ConvergenceChecker**: Logic for evaluating termination conditions
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
//! #   fn step<C, M>(&mut self, cost_fn: &C, manifold: &M, 
//! #                 state: &mut OptimizerState<f64, M::Point, M::TangentVector>) -> Result<()>
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
    core::manifold::Manifold,
    error::Result,
    memory::workspace::Workspace,
    types::Scalar,
};
use std::fmt::Debug;
use std::time::{Duration, Instant};

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

/// Dynamic state information maintained during Riemannian optimization.
///
/// This structure tracks the evolving state of an optimization algorithm,
/// maintaining both the current mathematical state (point, value, gradient)
/// and computational metadata (iteration counts, timing information).
///
/// # Mathematical State
///
/// - **Current state**: (point, value, gradient) representing current iterate
/// - **Previous state**: Historical information for convergence analysis
/// - **Gradient information**: Both vector and norm for optimality testing
///
/// # Computational Tracking
///
/// - **Iteration counting**: Major algorithm steps
/// - **Evaluation counting**: Function and gradient computation costs
/// - **Timing information**: Wall-clock performance measurement
///
/// # State Evolution
///
/// The state evolves through the optimization process:
/// 1. **Initialization**: Set initial point and function value
/// 2. **Gradient computation**: Add gradient and norm information
/// 3. **Iteration update**: Move to new point, preserve history
/// 4. **Convergence testing**: Use current and historical data
#[derive(Clone, Debug)]
pub struct OptimizerState<T, P, TV>
where
    T: Scalar,
{
    /// Current iterate xₖ ∈ ℳ on the manifold
    pub point: P,

    /// Current objective function value f(xₖ)
    pub value: T,

    /// Current Riemannian gradient grad f(xₖ) ∈ T_{xₖ} ℳ (if computed)
    pub gradient: Option<TV>,

    /// Riemannian gradient norm ||grad f(xₖ)||_g for optimality testing
    pub gradient_norm: Option<T>,

    /// Previous iterate xₖ₋₁ for convergence analysis
    pub previous_point: Option<P>,

    /// Previous objective value f(xₖ₋₁) for progress monitoring
    pub previous_value: Option<T>,

    /// Current major iteration number k
    pub iteration: usize,

    /// Total number of objective function evaluations f(x)
    pub function_evaluations: usize,

    /// Total number of gradient evaluations grad f(x)
    pub gradient_evaluations: usize,

    /// Optimization start time for duration calculation
    pub start_time: Instant,
}

impl<T, P, TV> OptimizerState<T, P, TV>
where
    T: Scalar,
    P: Clone,
    TV: Clone,
{
    /// Creates a new optimizer state.
    pub fn new(point: P, value: T) -> Self {
        Self {
            point,
            value,
            gradient: None,
            gradient_norm: None,
            previous_point: None,
            previous_value: None,
            iteration: 0,
            function_evaluations: 1,
            gradient_evaluations: 0,
            start_time: Instant::now(),
        }
    }

    /// Updates the state with a new point and value.
    pub fn update(&mut self, point: P, value: T) {
        self.previous_point = Some(std::mem::replace(&mut self.point, point));
        self.previous_value = Some(self.value);
        self.value = value;
        self.iteration += 1;
        self.function_evaluations += 1;
    }

    /// Sets the current gradient.
    pub fn set_gradient(&mut self, gradient: TV, norm: T) {
        self.gradient = Some(gradient);
        self.gradient_norm = Some(norm);
        self.gradient_evaluations += 1;
    }

    /// Computes the change in objective value.
    pub fn value_change(&self) -> Option<T> {
        self.previous_value
            .map(|prev| <T as num_traits::Float>::abs(self.value - prev))
    }

    /// Computes the distance between current and previous points.
    pub fn point_change<M>(&self, _manifold: &M) -> Result<Option<T>> 
    where 
        M: Manifold<T>,
        M::Point: std::borrow::Borrow<P>,
    {
        match &self.previous_point {
            Some(_prev) => {
                let _workspace = Workspace::<T>::new();
                // This will need special handling since we can't directly use distance
                // For now, return None - this will be fixed when we update Manifold trait
                Ok(None)
            },
            None => Ok(None),
        }
    }

    /// Gets the elapsed time since optimization started.
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
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
/// The optimizer should maintain internal state separately from the
/// `OptimizerState` parameter, which is used for algorithm communication.
///
/// ## Error Handling
/// Return appropriate errors for:
/// - Manifold constraint violations
/// - Numerical instabilities (NaN, infinity)
/// - Line search failures
/// - Resource exhaustion
///
/// ## Generic Design
/// The trait is generic over cost function and manifold types to enable:
/// - Type-safe manifold operations
/// - Efficient memory layouts
/// - Compile-time optimization
/// - Flexible algorithm composition
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
    fn optimize<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        initial_point: &M::Point,
        stopping_criterion: &StoppingCriterion<T>,
    ) -> Result<OptimizationResult<T, M::Point>>
    where
        C: crate::cost_function::CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>;

    /// Performs a single major iteration of the optimization algorithm.
    ///
    /// This method enables fine-grained control over the optimization process,
    /// allowing custom termination logic, progress monitoring, and algorithm
    /// composition. It updates the provided state in-place.
    ///
    /// # Mathematical Operation
    ///
    /// Typically performs one iteration of the form:
    /// 1. Compute search direction ηₖ ∈ T_{xₖ} ℳ
    /// 2. Determine step size αₖ > 0 (possibly via line search)
    /// 3. Update point: xₖ₊₁ = R_{xₖ}(αₖ ηₖ)
    /// 4. Evaluate f(xₖ₊₁) and update gradient information
    ///
    /// # State Updates
    ///
    /// The method updates the state with:
    /// - New point and objective value
    /// - New gradient and gradient norm (if computed)
    /// - Incremented iteration and evaluation counters
    /// - Previous point/value for convergence analysis
    ///
    /// # Arguments
    ///
    /// * `cost_fn` - Objective function f: ℳ → ℝ
    /// * `manifold` - Riemannian manifold ℳ
    /// * `state` - Current optimization state (modified in-place)
    ///
    /// # Errors
    ///
    /// Returns errors for:
    /// - Line search failure
    /// - Numerical instabilities
    /// - Manifold constraint violations
    /// - Invalid state (NaN values, etc.)
    ///
    /// # Usage
    ///
    /// ```rust,no_run
    /// # use riemannopt_core::prelude::*;
    /// # fn example() -> Result<()> {
    /// let mut state = OptimizerState::new(initial_point, initial_value);
    /// 
    /// for iteration in 0..max_iterations {
    ///     optimizer.step(&cost_fn, &manifold, &mut state)?;
    ///     
    ///     // Custom convergence check
    ///     if let Some(grad_norm) = state.gradient_norm {
    ///         if grad_norm < tolerance {
    ///             break;
    ///         }
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    fn step<C, M>(
        &mut self,
        cost_fn: &C,
        manifold: &M,
        state: &mut OptimizerState<T, M::Point, M::TangentVector>,
    ) -> Result<()>
    where
        C: crate::cost_function::CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
        M: Manifold<T>;
}

/// Mathematical convergence analysis for Riemannian optimization algorithms.
///
/// This utility provides rigorous testing of convergence conditions for
/// optimization algorithms on Riemannian manifolds. It implements both
/// standard and strict convergence criteria based on established
/// optimization theory.
///
/// # Convergence Theory
///
/// ## First-Order Conditions
/// For unconstrained optimization on manifolds, a point x* is stationary if:
/// ||grad f(x*)||_g = 0
///
/// In practice, we test: ||grad f(x)||_g < ε_grad
///
/// ## Progress-Based Criteria
/// - **Function progress**: |f(xₖ) - f(xₖ₋₁)| < ε_f
/// - **Point progress**: d_ℳ(xₖ, xₖ₋₁) < ε_x using manifold distance
///
/// # Testing Strategies
///
/// ## Standard Testing (`check`)
/// Tests each criterion independently - algorithm terminates when
/// ANY condition is satisfied.
///
/// ## Strict Testing (`check_strict`)
/// Requires MULTIPLE criteria to be satisfied simultaneously,
/// providing higher confidence in convergence quality.
pub struct ConvergenceChecker;

impl ConvergenceChecker {
    /// Tests all stopping criteria and returns the first satisfied condition.
    ///
    /// This method implements standard convergence testing where the algorithm
    /// terminates as soon as ANY criterion is satisfied. It checks criteria
    /// in order of computational cost (cheapest first).
    ///
    /// # Testing Order
    ///
    /// 1. **Iteration limit**: iteration ≥ max_iterations
    /// 2. **Time limit**: elapsed_time ≥ max_time
    /// 3. **Evaluation limit**: evaluations ≥ max_evaluations
    /// 4. **Gradient norm**: ||grad f(x)||_g < ε_grad
    /// 5. **Function change**: |f(xₖ) - f(xₖ₋₁)| < ε_f
    /// 6. **Point change**: d_ℳ(xₖ, xₖ₋₁) < ε_x
    /// 7. **Target value**: f(x) ≤ f_target
    ///
    /// # Arguments
    ///
    /// * `state` - Current optimization state with point, value, and gradients
    /// * `manifold` - Riemannian manifold for distance computations
    /// * `criterion` - Stopping criteria configuration
    ///
    /// # Returns
    ///
    /// - `Some(TerminationReason)` if any criterion is satisfied
    /// - `None` if optimization should continue
    ///
    /// # Mathematical Notes
    ///
    /// - Gradient norm uses the Riemannian metric: ||·||_g
    /// - Point distance uses manifold metric: d_ℳ(·,·)
    /// - Function change uses absolute difference: |f(xₖ) - f(xₖ₋₁)|
    ///
    /// # Errors
    ///
    /// Returns errors for:
    /// - Manifold distance computation failures
    /// - Invalid state (NaN values)
    /// - Numerical instabilities in distance calculation
    pub fn check<T, P, TV, M>(
        state: &OptimizerState<T, P, TV>,
        manifold: &M,
        criterion: &StoppingCriterion<T>,
    ) -> Result<Option<TerminationReason>>
    where
        T: Scalar,
        M: Manifold<T>,
        M::Point: std::borrow::Borrow<P>,
        P: Clone,
        TV: Clone,
    {
        // Check iteration limit
        if let Some(max_iter) = criterion.max_iterations {
            if state.iteration >= max_iter {
                return Ok(Some(TerminationReason::MaxIterations));
            }
        }

        // Check time limit
        if let Some(max_time) = criterion.max_time {
            if state.elapsed() >= max_time {
                return Ok(Some(TerminationReason::MaxTime));
            }
        }

        // Check function evaluation limit
        if let Some(max_evals) = criterion.max_function_evaluations {
            if state.function_evaluations >= max_evals {
                return Ok(Some(TerminationReason::MaxFunctionEvaluations));
            }
        }

        // Check gradient norm
        if let (Some(grad_norm), Some(grad_tol)) =
            (state.gradient_norm, criterion.gradient_tolerance)
        {
            if grad_norm < grad_tol {
                return Ok(Some(TerminationReason::Converged));
            }
        }

        // Check function value change
        if let (Some(val_change), Some(val_tol)) =
            (state.value_change(), criterion.function_tolerance)
        {
            if val_change < val_tol && state.iteration > 0 {
                return Ok(Some(TerminationReason::Converged));
            }
        }

        // Check point change
        if let Some(point_tol) = criterion.point_tolerance {
            if let Some(point_change) = state.point_change(manifold)? {
                if point_change < point_tol && state.iteration > 0 {
                    return Ok(Some(TerminationReason::Converged));
                }
            }
        }

        // Check target value
        if let Some(target) = criterion.target_value {
            if state.value <= target {
                return Ok(Some(TerminationReason::TargetReached));
            }
        }

        Ok(None)
    }

    /// Tests convergence using strict multi-criteria analysis.
    ///
    /// This method implements rigorous convergence testing where multiple
    /// criteria must be satisfied simultaneously. It provides higher confidence
    /// in solution quality at the cost of potentially longer optimization times.
    ///
    /// # Multi-Criteria Logic
    ///
    /// The algorithm has converged if ALL required criteria are satisfied:
    /// - **Gradient criterion** (if `require_gradient`): ||grad f(x)||_g < ε_grad
    /// - **Progress criteria** (if `require_value_and_point`):
    ///   - Function progress: |f(xₖ) - f(xₖ₋₁)| < ε_f
    ///   - Point progress: d_ℳ(xₖ, xₖ₋₁) < ε_x
    ///
    /// # Usage Guidelines
    ///
    /// ## Conservative Convergence
    /// ```rust,no_run
    /// # use riemannopt_core::prelude::*;
    /// # fn example() -> Result<Option<TerminationReason>> {
    /// // Require both gradient and progress criteria
    /// let result = ConvergenceChecker::check_strict(
    ///     &state, &manifold, &criterion,
    ///     true,  // require_gradient
    ///     true   // require_value_and_point
    /// )?;
    /// # Ok(result)
    /// # }
    /// ```
    ///
    /// ## Gradient-Only Convergence
    /// ```rust,no_run
    /// # use riemannopt_core::prelude::*;
    /// # fn example() -> Result<Option<TerminationReason>> {
    /// // Require only gradient criterion (for smooth functions)
    /// let result = ConvergenceChecker::check_strict(
    ///     &state, &manifold, &criterion,
    ///     true,  // require_gradient
    ///     false  // skip progress criteria
    /// )?;
    /// # Ok(result)
    /// # }
    /// ```
    ///
    /// # Arguments
    ///
    /// * `state` - Current optimization state
    /// * `manifold` - Riemannian manifold for geometric operations
    /// * `criterion` - Stopping criteria configuration
    /// * `require_gradient` - Whether gradient norm must satisfy tolerance
    /// * `require_value_and_point` - Whether both function and point progress must satisfy tolerances
    ///
    /// # Returns
    ///
    /// - `Some(TerminationReason::Converged)` if all required criteria satisfied
    /// - `Some(other_reason)` for non-convergence termination (limits exceeded)
    /// - `None` if optimization should continue
    pub fn check_strict<T, P, TV, M>(
        state: &OptimizerState<T, P, TV>,
        manifold: &M,
        criterion: &StoppingCriterion<T>,
        require_gradient: bool,
        require_value_and_point: bool,
    ) -> Result<Option<TerminationReason>>
    where
        T: Scalar,
        M: Manifold<T>,
        M::Point: std::borrow::Borrow<P>,
        P: Clone,
        TV: Clone,
    {
        // First check non-convergence termination conditions
        let basic_check = Self::check(state, manifold, criterion)?;
        if let Some(reason) = basic_check {
            if !matches!(reason, TerminationReason::Converged) {
                return Ok(Some(reason));
            }
        }

        let mut converged = true;

        // Check gradient criterion if required
        if require_gradient {
            if let (Some(grad_norm), Some(grad_tol)) =
                (state.gradient_norm, criterion.gradient_tolerance)
            {
                converged &= grad_norm < grad_tol;
            } else {
                converged = false;
            }
        }

        // Check value and point change if required
        if require_value_and_point && state.iteration > 0 {
            let value_ok = if let (Some(val_change), Some(val_tol)) =
                (state.value_change(), criterion.function_tolerance)
            {
                val_change < val_tol
            } else {
                false
            };

            let point_ok = if let Some(point_tol) = criterion.point_tolerance {
                if let Some(point_change) = state.point_change(manifold)? {
                    point_change < point_tol
                } else {
                    false
                }
            } else {
                false
            };

            converged &= value_ok && point_ok;
        }

        if converged && state.iteration > 0 {
            Ok(Some(TerminationReason::Converged))
        } else {
            Ok(None)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::DVector;
    use std::time::Duration;

    #[test]
    fn test_optimization_result() {
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let result = OptimizationResult::<f64, DVector<f64>>::new(
            point.clone(),
            0.5,
            100,
            Duration::from_secs(1),
            TerminationReason::Converged,
        );

        assert_eq!(result.point, point);
        assert_eq!(result.value, 0.5);
        assert_eq!(result.iterations, 100);
        assert!(result.converged);
        assert_eq!(result.termination_reason, TerminationReason::Converged);
    }

    #[test]
    fn test_stopping_criterion() {
        let criterion = StoppingCriterion::<f64>::new()
            .with_max_iterations(500)
            .with_gradient_tolerance(1e-8)
            .with_function_tolerance(1e-10);

        assert_eq!(criterion.max_iterations, Some(500));
        assert_eq!(criterion.gradient_tolerance, Some(1e-8));
        assert_eq!(criterion.function_tolerance, Some(1e-10));
    }

    #[test]
    fn test_optimizer_state() {
        let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
        let mut state = OptimizerState::<f64, DVector<f64>, DVector<f64>>::new(point.clone(), 1.0);

        assert_eq!(state.iteration, 0);
        assert_eq!(state.function_evaluations, 1);
        assert_eq!(state.gradient_evaluations, 0);

        let new_point = DVector::from_vec(vec![0.9, 0.1, 0.0]);
        state.update(new_point.clone(), 0.9);

        assert_eq!(state.iteration, 1);
        assert_eq!(state.point, new_point);
        assert_eq!(state.value, 0.9);
        assert_eq!(state.previous_value, Some(1.0));
        assert!((state.value_change().unwrap() - 0.1f64).abs() < 1e-10);
    }

    // Tests temporairement commentés - seront mis à jour après l'implémentation complète
    // #[test]
    // fn test_convergence_checker_iterations() {
    //     let point = DVector::from_vec(vec![1.0, 0.0, 0.0]);
    //     let mut state = OptimizerState::<f64, DVector<f64>, DVector<f64>>::new(point, 1.0);
    //     state.iteration = 1000;

    //     let criterion = StoppingCriterion::new().with_max_iterations(1000);

    //     let manifold = MinimalTestManifold::new(3);
    //     let result = ConvergenceChecker::check(&state, &manifold, &criterion).unwrap();
    //     assert_eq!(result, Some(TerminationReason::MaxIterations));
    // }

    // #[test]
    // fn test_convergence_checker_gradient() {
    //     // TODO: Mettre à jour après avoir implémenté MockManifold avec les types associés
    // }
}
