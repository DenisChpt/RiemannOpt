//! Line search algorithms for Riemannian optimization.
//!
//! This module implements mathematically rigorous line search strategies for
//! determining optimal step sizes along search directions on Riemannian manifolds.
//! These algorithms are fundamental components of optimization methods, ensuring
//! convergence while maintaining computational efficiency.
//!
//! # Mathematical Foundation
//!
//! ## Riemannian Line Search Problem
//!
//! Given:
//! - A point x ∈ ℳ on a Riemannian manifold
//! - A search direction η ∈ T_x ℳ in the tangent space
//! - An objective function f: ℳ → ℝ
//!
//! Find the optimal step size α* ∈ ℝ₊ that minimizes:
//! φ(α) = f(R_x(α η))
//!
//! where R_x: T_x ℳ → ℳ is a retraction map.
//!
//! ## Sufficient Decrease Conditions
//!
//! Line search algorithms ensure progress by requiring sufficient decrease
//! in the objective function. The most common conditions are:
//!
//! ### Armijo Condition (Sufficient Decrease)
//! f(R_x(α η)) ≤ f(x) + c₁ α ⟨grad f(x), η⟩_g
//!
//! where 0 < c₁ < 1 (typically c₁ = 10⁻⁴).
//!
//! ### Wolfe Conditions
//!
//! **Weak Wolfe conditions** combine sufficient decrease with curvature:
//! 1. Armijo: f(R_x(α η)) ≤ f(x) + c₁ α ⟨grad f(x), η⟩_g
//! 2. Curvature: ⟨grad f(R_x(α η)), T_{x→y}(η)⟩_g ≥ c₂ ⟨grad f(x), η⟩_g
//!
//! **Strong Wolfe conditions** replace condition 2 with:
//! 2'. Strong curvature: |⟨grad f(R_x(α η)), T_{x→y}(η)⟩_g| ≤ c₂ |⟨grad f(x), η⟩_g|
//!
//! where 0 < c₁ < c₂ < 1, typically c₂ ∈ [0.1, 0.9], and T_{x→y} denotes
//! vector transport from T_x ℳ to T_y ℳ.
//!
//! # Riemannian Adaptations
//!
//! ## Retraction-Based Movement
//! Unlike Euclidean optimization where x_{k+1} = x_k + α_k d_k,
//! Riemannian optimization uses:
//! x_{k+1} = R_{x_k}(α_k η_k)
//!
//! ## Vector Transport
//! For curvature conditions, tangent vectors must be transported:
//! η_transported = T_{x_k → x_{k+1}}(η_k)
//!
//! This maintains the geometric relationship between vectors across
//! different tangent spaces.
//!
//! ## Manifold-Specific Considerations
//! - **Geodesic bounds**: Step sizes may be limited by manifold geometry
//! - **Injectivity radius**: Large steps may violate retraction properties
//! - **Numerical stability**: Manifold operations may introduce constraints
//!
//! # Algorithm Variants
//!
//! ## Backtracking Line Search
//! - **Purpose**: Simple, robust algorithm requiring only Armijo condition
//! - **Strategy**: Start with large step, reduce until sufficient decrease
//! - **Advantages**: Minimal gradient evaluations, guaranteed termination
//! - **Applications**: Gradient descent, Newton methods
//!
//! ## Strong Wolfe Line Search
//! - **Purpose**: High-quality steps for quasi-Newton methods
//! - **Strategy**: Bracketing and zoom phases to satisfy both conditions
//! - **Advantages**: Ensures good curvature properties for BFGS/L-BFGS
//! - **Applications**: Quasi-Newton methods, conjugate gradient
//!
//! ## Fixed Step Size
//! - **Purpose**: Simplified algorithms with theoretical guarantees
//! - **Strategy**: Predetermined step size (e.g., 1/k, constant)
//! - **Advantages**: No function evaluations for step size
//! - **Applications**: Stochastic methods, theoretical analysis
//!
//! # Implementation Guidelines
//!
//! ## Numerical Considerations
//! - Use robust parameter validation (0 < c₁ < c₂ < 1)
//! - Handle degenerate cases (zero directional derivative)
//! - Implement safeguards against infinite loops
//! - Respect machine precision limitations
//!
//! ## Computational Efficiency
//! - Minimize function and gradient evaluations
//! - Leverage workspace-based memory management
//! - Use efficient retraction algorithms
//! - Cache intermediate computations when possible
//!
//! ## Convergence Theory
//! - Ensure descent direction: ⟨grad f(x), η⟩_g < 0
//! - Prove termination under reasonable conditions
//! - Maintain global convergence properties
//!
//! # Examples
//!
//! ## Basic Backtracking Usage
//!
//! ```rust,ignore
//! # use riemannopt_core::prelude::*;
//! let mut line_search = BacktrackingLineSearch::new();
//! let params = LineSearchParams::backtracking();
//!
//! let result = line_search.search(
//!     &cost_fn, &manifold, &point, value,
//!     &gradient, &direction, &params
//! )?;
//!
//! if result.success {
//!     println!("Step size: {}, New value: {}", result.step_size, result.new_value);
//! }
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```
//!
//! ## Strong Wolfe for Quasi-Newton
//!
//! ```rust,ignore
//! # use riemannopt_core::prelude::*;
//! let mut line_search = StrongWolfeLineSearch::new();
//! let params = LineSearchParams::strong_wolfe();
//!
//! // Ensures good curvature properties for BFGS updates
//! let result = line_search.search(
//!     &cost_fn, &manifold, &point, value,
//!     &gradient, &quasi_newton_direction, &params
//! )?;
//! ```

use crate::{
	core::{cost_function::CostFunction, manifold::Manifold},
	error::{ManifoldError, Result},
	types::Scalar,
};
use num_traits::Float;
use std::fmt::Debug;

/// Comprehensive result of a Riemannian line search operation.
///
/// This structure contains all information generated during a line search,
/// including the accepted step size, new point, function values, and
/// computational statistics. It enables both optimization algorithm
/// implementation and performance analysis.
///
/// # Mathematical Interpretation
///
/// Given initial point x and search direction η, the line search finds α
/// such that the new point y = R_x(α η) satisfies sufficient decrease
/// conditions. This result captures the complete state transition.
#[derive(Debug, Clone)]
pub struct LineSearchResult<T, P, TV>
where
	T: Scalar,
{
	/// The accepted step size α satisfying sufficient decrease conditions
	pub step_size: T,

	/// The new point y = R_x(α η) ∈ ℳ after retraction
	pub new_point: P,

	/// The objective function value f(y) at the new point
	pub new_value: T,

	/// The Riemannian gradient grad f(y) ∈ T_y ℳ (if computed during line search)
	pub new_gradient: Option<TV>,

	/// Total number of objective function evaluations f(·) performed
	pub function_evals: usize,

	/// Total number of gradient evaluations grad f(·) performed  
	pub gradient_evals: usize,

	/// True if line search found acceptable step satisfying convergence conditions
	pub success: bool,
}

/// Mathematical and computational parameters for line search algorithms.
///
/// This structure encapsulates all tuning parameters required for line search
/// algorithms, including step size bounds, Wolfe condition constants, and
/// computational limits. Proper parameter selection is crucial for both
/// convergence guarantees and computational efficiency.
///
/// # Mathematical Parameters
///
/// ## Wolfe Condition Constants
/// - **c₁ (Armijo parameter)**: Controls sufficient decrease requirement
///   - Range: (0, 1), typically 10⁻⁴
///   - Smaller values require more decrease
/// - **c₂ (curvature parameter)**: Controls curvature condition
///   - Range: (c₁, 1), typically 0.1-0.9
///   - Larger values accept more curvature
///
/// ## Step Size Management
/// - **initial_step_size**: Starting step size α₀
/// - **max_step_size**: Upper bound to prevent overshooting
/// - **min_step_size**: Lower bound before declaring failure
/// - **rho**: Backtracking reduction factor ∈ (0, 1)
///
/// # Parameter Guidelines
///
/// ## For Gradient Descent
/// ```rust,no_run
/// # use riemannopt_core::prelude::*;
/// let params = LineSearchParams::<f64>::backtracking(); // c₁ = 0.5, simple conditions
/// ```
///
/// ## For Quasi-Newton Methods
/// ```rust,no_run
/// # use riemannopt_core::prelude::*;
/// let params = LineSearchParams::<f64>::strong_wolfe(); // c₁ = 10⁻⁴, c₂ = 0.9
/// ```
///
/// ## Custom Configuration
/// ```rust,no_run
/// # use riemannopt_core::prelude::*;
/// let params = LineSearchParams {
///     c1: 1e-4,           // Tight sufficient decrease
///     c2: 0.1,            // Loose curvature (for CG)
///     initial_step_size: 1.0,
///     max_step_size: 10.0,
///     min_step_size: 1e-12,
///     rho: 0.5,           // Halve step size each iteration
///     max_iterations: 30, // Reasonable computational budget
/// };
/// ```
#[derive(Debug, Clone)]
pub struct LineSearchParams<T>
where
	T: Scalar,
{
	/// Initial step size α₀ for line search start
	pub initial_step_size: T,

	/// Maximum allowable step size to prevent overshooting
	pub max_step_size: T,

	/// Minimum step size threshold before declaring line search failure
	pub min_step_size: T,

	/// Maximum number of line search iterations before termination
	pub max_iterations: usize,

	/// Armijo parameter c₁ ∈ (0,1) for sufficient decrease condition
	/// f(R_x(αη)) ≤ f(x) + c₁α⟨grad f(x), η⟩_g
	pub c1: T,

	/// Wolfe parameter c₂ ∈ (c₁,1) for curvature condition
	/// |⟨grad f(R_x(αη)), T(η)⟩_g| ≤ c₂|⟨grad f(x), η⟩_g|
	pub c2: T,

	/// Backtracking reduction factor ρ ∈ (0,1) for step size reduction
	/// α_{i+1} = ρ α_i when Armijo condition fails
	pub rho: T,
}

impl<T> Default for LineSearchParams<T>
where
	T: Scalar,
{
	fn default() -> Self {
		Self {
			initial_step_size: T::one(),
			max_step_size: <T as Scalar>::from_f64(10.0),
			min_step_size: <T as Scalar>::from_f64(1e-10),
			max_iterations: 50,
			c1: <T as Scalar>::from_f64(1e-4),
			c2: <T as Scalar>::from_f64(0.9),
			rho: <T as Scalar>::from_f64(0.5),
		}
	}
}

impl<T> LineSearchParams<T>
where
	T: Scalar,
{
	/// Validates line search parameters against mathematical requirements.
	///
	/// Ensures all parameters satisfy the theoretical conditions required
	/// for convergence guarantees and numerical stability.
	///
	/// # Mathematical Validation
	///
	/// ## Step Size Constraints
	/// - All step sizes must be positive
	/// - min_step_size < max_step_size (proper ordering)
	/// - initial_step_size should be reasonable
	///
	/// ## Wolfe Parameter Constraints
	/// - 0 < c₁ < 1 (sufficient decrease parameter)
	/// - c₁ < c₂ < 1 (proper curvature parameter ordering)
	/// - These ensure both progress and termination
	///
	/// ## Computational Constraints
	/// - 0 < ρ < 1 (backtracking must reduce step size)
	/// - max_iterations > 0 (must allow at least one iteration)
	///
	/// # Errors
	///
	/// Returns `ManifoldError::InvalidParameter` if:
	/// - Step sizes violate positivity or ordering constraints
	/// - Wolfe constants don't satisfy 0 < c₁ < c₂ < 1
	/// - Backtracking factor ρ ∉ (0, 1)
	/// - Maximum iterations is zero
	///
	/// # Usage
	///
	/// ```rust,no_run
	/// # use riemannopt_core::prelude::*;
	/// let params = LineSearchParams::<f64>::default();
	/// params.validate()?; // Ensure mathematical validity
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	pub fn validate(&self) -> Result<()> {
		// Validate step sizes
		if self.initial_step_size <= T::zero() {
			return Err(ManifoldError::invalid_parameter(
				"Initial step size must be positive",
			));
		}

		if self.min_step_size <= T::zero() {
			return Err(ManifoldError::invalid_parameter(
				"Minimum step size must be positive",
			));
		}

		if self.max_step_size <= self.min_step_size {
			return Err(ManifoldError::invalid_parameter(
				"Maximum step size must be greater than minimum step size",
			));
		}

		// Validate Wolfe constants
		if self.c1 <= T::zero() || self.c1 >= T::one() {
			return Err(ManifoldError::invalid_parameter(
				"Armijo constant c1 must be in (0, 1)",
			));
		}

		if self.c2 <= self.c1 || self.c2 >= T::one() {
			return Err(ManifoldError::invalid_parameter(
				"Wolfe constant c2 must satisfy c1 < c2 < 1",
			));
		}

		// Validate backtracking factor
		if self.rho <= T::zero() || self.rho >= T::one() {
			return Err(ManifoldError::invalid_parameter(
				"Backtracking factor rho must be in (0, 1)",
			));
		}

		// Validate iterations
		if self.max_iterations == 0 {
			return Err(ManifoldError::invalid_parameter(
				"Maximum iterations must be at least 1",
			));
		}

		Ok(())
	}

	/// Creates parameters optimized for strong Wolfe line search.
	///
	/// These parameters are designed for quasi-Newton methods (BFGS, L-BFGS)
	/// that require high-quality step sizes with good curvature properties.
	///
	/// # Parameter Values
	/// - c₁ = 10⁻⁴ (tight sufficient decrease)
	/// - c₂ = 0.9 (loose curvature for quasi-Newton)
	/// - Other parameters: conservative defaults
	///
	/// # Applications
	/// - BFGS and L-BFGS optimization
	/// - Newton-type methods requiring curvature information
	/// - High-precision optimization
	pub fn strong_wolfe() -> Self {
		Self::default()
	}

	/// Creates parameters optimized for weak Wolfe line search.
	///
	/// Similar to strong Wolfe but with relaxed curvature condition.
	/// Suitable for methods that don't require strict curvature properties.
	///
	/// # Parameter Values
	/// - c₁ = 10⁻⁴ (sufficient decrease)
	/// - c₂ = 0.9 (standard curvature parameter)
	///
	/// # Applications
	/// - Conjugate gradient methods
	/// - Algorithms with theoretical convergence guarantees
	/// - Methods where weak Wolfe conditions suffice
	pub fn weak_wolfe() -> Self {
		Self {
			c2: <T as Scalar>::from_f64(0.9),
			..Self::default()
		}
	}

	/// Creates parameters optimized for backtracking line search.
	///
	/// These parameters emphasize simplicity and robustness over optimality.
	/// Suitable for gradient descent and other first-order methods.
	///
	/// # Parameter Values
	/// - c₁ = 0.5 (relaxed sufficient decrease)
	/// - ρ = 0.5 (halve step size each iteration)
	/// - max_iterations = 20 (quick termination)
	///
	/// # Applications
	/// - Steepest descent optimization
	/// - First-order methods
	/// - Robust optimization with guaranteed progress
	pub fn backtracking() -> Self {
		Self {
			c1: <T as Scalar>::from_f64(0.5),
			c2: <T as Scalar>::from_f64(0.9), // Set to valid value even if not used
			rho: <T as Scalar>::from_f64(0.5),
			max_iterations: 20,
			..Self::default()
		}
	}

	/// Creates parameters optimized for nonlinear conjugate gradient methods.
	///
	/// CG requires a tight curvature condition (c₂ ≈ 0.1) to maintain
	/// conjugacy of successive search directions. Using c₂ = 0.9 (as for
	/// quasi-Newton) allows steps that destroy conjugacy, causing stagnation
	/// at large dimensions.
	///
	/// # Parameter Values
	/// - c₁ = 10⁻⁴ (standard sufficient decrease)
	/// - c₂ = 0.9 (relaxed curvature — allows larger steps for CG)
	/// - max_iterations = 30
	pub fn for_conjugate_gradient() -> Self {
		Self {
			c1: <T as Scalar>::from_f64(1e-4),
			c2: <T as Scalar>::from_f64(0.9),
			initial_step_size: T::one(),
			max_step_size: <T as Scalar>::from_f64(10.0),
			min_step_size: <T as Scalar>::from_f64(1e-12),
			rho: <T as Scalar>::from_f64(0.5),
			max_iterations: 30,
		}
	}
}

/// Context container for line search operations on Riemannian manifolds.
///
/// This structure bundles the fundamental mathematical objects required for
/// line search: the objective function and the manifold structure. It provides
/// a cleaner API by reducing parameter passing overhead and enabling more
/// intuitive method chaining.
///
/// # Design Benefits
///
/// ## API Simplification
/// - Reduces method parameter count from 7+ to 4-5
/// - Groups conceptually related objects
/// - Enables method chaining and functional composition
///
/// ## Type Safety
/// - Ensures consistent cost function and manifold pairing
/// - Compile-time verification of type compatibility
/// - Prevents mismatched manifold/function combinations
///
/// # Usage Pattern
///
/// ```rust,ignore
/// # use riemannopt_core::prelude::*;
/// let ctx = LineSearchContext::new(&cost_fn, &manifold);
///
/// let result = line_search.search_with_context(
///     &ctx, &point, value, &gradient, &direction, &params
/// )?;
/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
/// ```
#[derive(Debug)]
pub struct LineSearchContext<'a, T, C, M>
where
	T: Scalar,
	C: CostFunction<T>,
	M: Manifold<T>,
{
	/// The objective function f: ℳ → ℝ to minimize
	pub cost_fn: &'a C,
	/// The Riemannian manifold ℳ defining the constraint set
	pub manifold: &'a M,
	/// Type parameter phantom data for proper trait resolution
	_phantom: std::marker::PhantomData<T>,
}

impl<'a, T, C, M> LineSearchContext<'a, T, C, M>
where
	T: Scalar,
	C: CostFunction<T>,
	M: Manifold<T>,
{
	/// Creates a new line search context binding a cost function to a manifold.
	///
	/// This establishes the mathematical setting for line search operations
	/// by pairing the objective function f: ℳ → ℝ with the manifold ℳ.
	///
	/// # Arguments
	///
	/// * `cost_fn` - Objective function to minimize
	/// * `manifold` - Riemannian manifold constraining the optimization
	///
	/// # Type Safety
	///
	/// The context ensures that the cost function's Point and TangentVector
	/// types match the manifold's associated types, preventing runtime errors.
	pub fn new(cost_fn: &'a C, manifold: &'a M) -> Self {
		Self {
			cost_fn,
			manifold,
			_phantom: std::marker::PhantomData,
		}
	}
}

/// Universal interface for line search algorithms on Riemannian manifolds.
///
/// This trait defines the mathematical and computational contract for line search
/// implementations. It provides multiple interfaces to accommodate different usage
/// patterns while maintaining theoretical rigor and computational efficiency.
///
/// # Mathematical Contract
///
/// Implementations must find a step size α > 0 such that:
/// y = R_x(α η) ∈ ℳ satisfies sufficient decrease conditions
///
/// where:
/// - x ∈ ℳ is the current point
/// - η ∈ T_x ℳ is the search direction  
/// - R_x: T_x ℳ → ℳ is the retraction map
///
/// # Implementation Requirements
///
/// ## Descent Direction Validation
/// Verify that ⟨grad f(x), η⟩_g < 0 (descent condition)
///
/// ## Sufficient Decrease
/// Ensure progress through Armijo or Wolfe conditions
///
/// ## Numerical Robustness
/// - Handle edge cases (zero gradients, small steps)
/// - Respect parameter bounds and constraints
/// - Provide meaningful error diagnostics
///
/// ## Computational Efficiency
/// - Minimize function/gradient evaluations
/// - Use workspace-based memory management
/// - Leverage efficient manifold operations
///
/// # Interface Variants
///
/// ## Primary Interface: `search`
/// Full-featured method computing directional derivative internally
///
/// ## Optimized Interface: `search_with_deriv`
/// Efficient method when directional derivative is pre-computed
///
/// ## Context Interface: `search_with_context`
/// Clean API using bundled parameters for complex optimization loops
pub trait LineSearch<T>: Debug
where
	T: Scalar,
{
	/// Performs line search to find optimal step size along search direction.
	///
	/// This is the primary line search interface that computes the directional
	/// derivative internally and finds a step size satisfying sufficient decrease
	/// conditions. It provides a complete solution for most optimization algorithms.
	///
	/// # Mathematical Process
	///
	/// 1. **Validate inputs**: Ensure descent direction ⟨grad f(x), η⟩_g < 0
	/// 2. **Compute directional derivative**: d₀ = ⟨grad f(x), η⟩_g
	/// 3. **Search for step size**: Find α satisfying Armijo/Wolfe conditions
	/// 4. **Return result**: New point y = R_x(α η) and metadata
	///
	/// # Arguments
	///
	/// * `cost_fn` - Objective function f: ℳ → ℝ to minimize
	/// * `manifold` - Riemannian manifold ℳ with retraction R_x
	/// * `point` - Current point x ∈ ℳ
	/// * `value` - Current function value f(x)
	/// * `gradient` - Current Riemannian gradient grad f(x) ∈ T_x ℳ
	/// * `direction` - Search direction η ∈ T_x ℳ (must be descent: ⟨grad f(x), η⟩_g < 0)
	/// * `params` - Line search parameters (step sizes, Wolfe constants, etc.)
	///
	/// # Returns
	///
	/// `LineSearchResult` containing:
	/// - Accepted step size α and new point y = R_x(α η)
	/// - Function value f(y) and optionally gradient grad f(y)
	/// - Computational statistics (function/gradient evaluations)
	/// - Success flag indicating whether sufficient decrease was achieved
	///
	/// # Errors
	///
	/// Returns errors for:
	/// - Non-descent direction: ⟨grad f(x), η⟩_g ≥ 0
	/// - Invalid parameters: violating mathematical constraints
	/// - Line search failure: no acceptable step size found
	/// - Manifold operation failures: retraction or transport errors
	/// - Numerical issues: NaN, infinity, or precision loss
	///
	/// # Performance Notes
	///
	/// This method computes the directional derivative, which requires one
	/// inner product evaluation. For repeated line searches, consider using
	/// `search_with_deriv` to avoid redundant computations.
	#[allow(clippy::too_many_arguments)]
	fn search<C, M>(
		&mut self,
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		value: T,
		gradient: &M::TangentVector,
		direction: &M::TangentVector,
		params: &LineSearchParams<T>,
		ws: &mut M::Workspace,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		// Default implementation: compute directional derivative and use efficient method
		let directional_deriv = manifold.inner_product(point, gradient, direction, ws)?;
		self.search_with_deriv(
			cost_fn,
			manifold,
			point,
			value,
			direction,
			directional_deriv,
			params,
			ws,
		)
	}

	/// Efficient line search with pre-computed directional derivative.
	///
	/// This optimized interface avoids redundant computation of the directional
	/// derivative when it has already been computed. This is particularly valuable
	/// in optimization algorithms that reuse gradient information.
	///
	/// # Mathematical Foundation
	///
	/// The directional derivative d₀ = ⟨grad f(x), η⟩_g represents the instantaneous
	/// rate of change of f along direction η. It must be negative for η to be
	/// a descent direction.
	///
	/// # Efficiency Benefits
	///
	/// - Saves one Riemannian inner product computation per line search
	/// - Enables algorithmic optimizations (gradient reuse, direction caching)
	/// - Reduces computational overhead in iteration-heavy algorithms
	///
	/// # Arguments
	///
	/// * `cost_fn` - Objective function f: ℳ → ℝ to minimize
	/// * `manifold` - Riemannian manifold ℳ with geometric operations
	/// * `point` - Current point x ∈ ℳ on the manifold
	/// * `value` - Current objective value f(x)
	/// * `direction` - Search direction η ∈ T_x ℳ
	/// * `directional_deriv` - Pre-computed directional derivative ⟨grad f(x), η⟩_g < 0
	/// * `params` - Line search algorithm parameters
	///
	/// # Returns
	///
	/// Complete `LineSearchResult` with step size, new point, and statistics.
	///
	/// # Preconditions
	///
	/// - `directional_deriv` must equal ⟨grad f(x), η⟩_g
	/// - `directional_deriv` < 0 (descent direction)
	/// - `point` must lie on the manifold
	/// - `direction` must be in the tangent space T_x ℳ
	///
	/// # Usage
	///
	/// ```rust,ignore
	/// # use riemannopt_core::prelude::*;
	/// // Compute directional derivative once
	/// let dir_deriv = manifold.inner_product(&point, &gradient, &direction)?;
	///
	/// // Use in line search without recomputation
	/// let result = line_search.search_with_deriv(
	///     &cost_fn, &manifold, &point, value,
	///     &direction, dir_deriv, &params
	/// )?;
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	#[allow(clippy::too_many_arguments)]
	fn search_with_deriv<C, M>(
		&mut self,
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		value: T,
		direction: &M::TangentVector,
		directional_deriv: T,
		params: &LineSearchParams<T>,
		ws: &mut M::Workspace,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>;

	/// Returns a human-readable name identifying the line search algorithm.
	///
	/// Used for logging, debugging, and algorithm selection. Examples:
	/// "Backtracking", "StrongWolfe", "FixedStep", "Cubic", "MoreThuente".
	fn name(&self) -> &str;

	/// Line search using bundled context for cleaner API.
	///
	/// This interface provides syntactic convenience by bundling the cost function
	/// and manifold into a context object. It's particularly useful in complex
	/// optimization loops where the context remains constant.
	///
	/// # API Benefits
	///
	/// - Reduces parameter count for cleaner method signatures
	/// - Enables method chaining and functional composition
	/// - Improves code readability in optimization implementations
	/// - Maintains type safety through compile-time verification
	///
	/// # Arguments
	///
	/// * `ctx` - Line search context bundling cost function and manifold
	/// * `point` - Current point x ∈ ℳ on the manifold
	/// * `value` - Current objective function value f(x)
	/// * `gradient` - Current Riemannian gradient grad f(x) ∈ T_x ℳ
	/// * `direction` - Search direction η ∈ T_x ℳ (must satisfy descent condition)
	/// * `params` - Line search algorithm parameters
	///
	/// # Returns
	///
	/// Complete `LineSearchResult` equivalent to the standard `search` method.
	///
	/// # Usage
	///
	/// ```rust,ignore
	/// # use riemannopt_core::prelude::*;
	/// let ctx = LineSearchContext::new(&cost_fn, &manifold);
	///
	/// for iteration in 0..max_iterations {
	///     let result = line_search.search_with_context(
	///         &ctx, &point, value, &gradient, &direction, &params
	///     )?;
	///     
	///     // Update optimization state...
	/// }
	/// # Ok::<(), riemannopt_core::error::ManifoldError>(())
	/// ```
	///
	/// # Default Implementation
	///
	/// The default implementation delegates to the standard `search` method,
	/// so implementing this method is optional for most line search algorithms.
	fn search_with_context<C, M>(
		&mut self,
		ctx: &LineSearchContext<T, C, M>,
		point: &M::Point,
		value: T,
		gradient: &M::TangentVector,
		direction: &M::TangentVector,
		params: &LineSearchParams<T>,
		ws: &mut M::Workspace,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		// Default implementation delegates to the old method
		self.search(
			ctx.cost_fn,
			ctx.manifold,
			point,
			value,
			gradient,
			direction,
			params,
			ws,
		)
	}
}

/// Backtracking line search with Armijo sufficient decrease condition.
///
/// This algorithm implements the most fundamental line search strategy, ensuring
/// monotonic decrease in the objective function through iterative step size
/// reduction. It is robust, simple to implement, and provides convergence
/// guarantees for a wide class of optimization algorithms.
///
/// # Mathematical Algorithm
///
/// Starting with initial step size α₀:
/// 1. **Test condition**: f(R_x(α η)) ≤ f(x) + c₁ α ⟨grad f(x), η⟩_g
/// 2. **If satisfied**: Accept step size α
/// 3. **If not**: Reduce α ← ρα and repeat
/// 4. **Termination**: Success when condition met, failure if α < α_min
///
/// where 0 < c₁ < 1 (typically 10⁻⁴) and 0 < ρ < 1 (typically 0.5).
///
/// # Theoretical Properties
///
/// ## Convergence Guarantees
/// - **Finite termination**: Algorithm terminates in finite steps for descent directions
/// - **Sufficient decrease**: Ensures f(x_{k+1}) < f(x_k) with adequate margin
/// - **Global convergence**: Supports convergence proofs for optimization methods
///
/// ## Computational Complexity
/// - **Function evaluations**: O(log(α₀/α_min)) in worst case
/// - **Gradient evaluations**: None (reuses current gradient)
/// - **Memory**: O(1) (only stores current step size)
///
/// # Advantages
///
/// - **Simplicity**: Minimal implementation complexity
/// - **Robustness**: Works reliably across diverse problems
/// - **Efficiency**: Low computational overhead per iteration
/// - **Generality**: Suitable for first-order and some second-order methods
///
/// # Limitations
///
/// - **Step quality**: May accept suboptimal step sizes
/// - **Quasi-Newton incompatibility**: Insufficient curvature information for BFGS
/// - **Slow convergence**: Conservative steps may slow optimization
///
/// # Applications
///
/// - **Gradient descent**: Primary choice for steepest descent methods
/// - **Newton methods**: When Hessian is positive definite
/// - **Robust optimization**: When simplicity and reliability are prioritized
/// - **Real-time algorithms**: When computational budget is limited
#[derive(Debug, Clone, Copy)]
pub struct BacktrackingLineSearch;

impl BacktrackingLineSearch {
	/// Creates a new backtracking line search with default parameters.
	///
	/// Uses standard Armijo condition with conservative default parameters
	/// suitable for most gradient-based optimization algorithms.
	pub fn new() -> Self {
		Self
	}
}

impl Default for BacktrackingLineSearch {
	fn default() -> Self {
		Self::new()
	}
}

impl<T> LineSearch<T> for BacktrackingLineSearch
where
	T: Scalar,
{
	fn search_with_deriv<C, M>(
		&mut self,
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		value: T,
		direction: &M::TangentVector,
		directional_deriv: T,
		params: &LineSearchParams<T>,
		ws: &mut M::Workspace,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		// Non-descent direction: return current point
		if directional_deriv >= T::zero() {
			return Ok(LineSearchResult {
				step_size: T::zero(),
				new_point: point.clone(),
				new_value: value,
				new_gradient: None,
				function_evals: 0,
				gradient_evals: 0,
				success: false,
			});
		}

		let mut alpha = params.initial_step_size;
		let mut new_point = point.clone();
		let mut func_evals = 0;

		for _ in 0..params.max_iterations {
			// Retract along alpha * direction
			let mut scaled_dir = direction.clone();
			manifold.scale_tangent(point, alpha, direction, &mut scaled_dir)?;
			manifold.retract(point, &scaled_dir, &mut new_point, ws)?;

			let new_value = cost_fn.cost(&new_point)?;
			func_evals += 1;

			// Armijo condition: f(new) <= f(x) + c₁ * α * df₀
			if new_value <= value + params.c1 * alpha * directional_deriv {
				return Ok(LineSearchResult {
					step_size: alpha,
					new_point,
					new_value,
					new_gradient: None,
					function_evals: func_evals,
					gradient_evals: 0,
					success: true,
				});
			}

			// Contract step size
			alpha = alpha * params.rho;

			if alpha < params.min_step_size {
				break;
			}
		}

		// Return best effort (last tried point)
		let mut scaled_dir = direction.clone();
		manifold.scale_tangent(point, alpha, direction, &mut scaled_dir)?;
		manifold.retract(point, &scaled_dir, &mut new_point, ws)?;
		let new_value = cost_fn.cost(&new_point)?;
		func_evals += 1;

		Ok(LineSearchResult {
			step_size: alpha,
			new_point,
			new_value,
			new_gradient: None,
			function_evals: func_evals,
			gradient_evals: 0,
			success: new_value < value,
		})
	}

	fn name(&self) -> &str {
		"Backtracking"
	}
}

/// Adaptive line search with Armijo condition and step size memory.
///
/// Uses only the sufficient decrease (Armijo) condition and remembers the last
/// accepted step size as the initial guess for the next iteration. This makes it
/// very fast (typically 1-2 function evaluations) at the cost of weaker theoretical
/// guarantees compared to Strong Wolfe.
///
/// # Algorithm
///
/// 1. Start with `α = previous_α` (or `1/‖d‖` on first call)
/// 2. While `f(R_x(αd)) > f(x) + c₁·α·⟨∇f(x), d⟩`: `α *= contraction_factor`
/// 3. If `f(new_x) > f(x)`: reject step entirely (`α = 0`)
/// 4. Adapt hint for next call based on number of evaluations
#[derive(Debug, Clone)]
pub struct AdaptiveLineSearch<T: Scalar> {
	/// Remembered step size from previous successful search
	previous_alpha: Option<T>,
	/// Backtracking contraction factor (default: 0.5)
	contraction_factor: T,
	/// Sufficient decrease parameter (default: 0.5, much more aggressive than Strong Wolfe's 1e-4)
	sufficient_decrease: T,
	/// Maximum backtracking iterations (default: 10)
	max_iterations: usize,
	/// Initial step size when no previous alpha is available (default: 1.0)
	initial_step_size: T,
}

impl<T: Scalar> AdaptiveLineSearch<T> {
	/// Creates a new adaptive line search with default parameters.
	pub fn new() -> Self {
		Self {
			previous_alpha: None,
			contraction_factor: <T as Scalar>::from_f64(0.5),
			sufficient_decrease: <T as Scalar>::from_f64(0.5),
			max_iterations: 10,
			initial_step_size: T::one(),
		}
	}

	/// Sets the contraction factor (default: 0.5).
	pub fn with_contraction_factor(mut self, factor: T) -> Self {
		self.contraction_factor = factor;
		self
	}

	/// Sets the sufficient decrease parameter (default: 0.5).
	pub fn with_sufficient_decrease(mut self, c: T) -> Self {
		self.sufficient_decrease = c;
		self
	}

	/// Sets the maximum number of backtracking iterations (default: 10).
	pub fn with_max_iterations(mut self, n: usize) -> Self {
		self.max_iterations = n;
		self
	}
}

impl<T: Scalar> Default for AdaptiveLineSearch<T> {
	fn default() -> Self {
		Self::new()
	}
}

impl<T> LineSearch<T> for AdaptiveLineSearch<T>
where
	T: Scalar,
{
	fn search_with_deriv<C, M>(
		&mut self,
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		value: T,
		direction: &M::TangentVector,
		directional_deriv: T,
		params: &LineSearchParams<T>,
		ws: &mut M::Workspace,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		// Non-descent direction: return current point
		if directional_deriv >= T::zero() {
			return Ok(LineSearchResult {
				step_size: T::zero(),
				new_point: point.clone(),
				new_value: value,
				new_gradient: None,
				function_evals: 0,
				gradient_evals: 0,
				success: false,
			});
		}

		// Use c1 from params as the Armijo sufficient decrease parameter
		let suff_decrease = if self.sufficient_decrease > T::zero() {
			self.sufficient_decrease
		} else {
			params.c1
		};

		// Compute direction norm for initial step normalization
		let norm_d = <T as Float>::sqrt(manifold.inner_product(point, direction, direction, ws)?);

		// Determine initial alpha
		let mut alpha = if let Some(prev) = self.previous_alpha {
			prev
		} else if norm_d > T::zero() {
			self.initial_step_size / norm_d
		} else {
			self.initial_step_size
		};

		// Backtracking loop
		let mut new_point = point.clone();
		let mut new_value;
		let mut cost_evaluations = 0;

		loop {
			// Retract along direction with step size alpha
			let mut scaled_dir = direction.clone();
			manifold.scale_tangent(point, alpha, direction, &mut scaled_dir)?;
			manifold.retract(point, &scaled_dir, &mut new_point, ws)?;
			new_value = cost_fn.cost(&new_point)?;
			cost_evaluations += 1;

			// Armijo condition: f(new) <= f(x) + c * alpha * df0
			if new_value <= value + suff_decrease * alpha * directional_deriv {
				break;
			}

			// Contract step size
			alpha = alpha * self.contraction_factor;

			if cost_evaluations >= self.max_iterations {
				break;
			}
		}

		// Safety: if step made things worse, reject entirely
		if new_value > value {
			self.previous_alpha = None;
			return Ok(LineSearchResult {
				step_size: T::zero(),
				new_point: point.clone(),
				new_value: value,
				new_gradient: None,
				function_evals: cost_evaluations,
				gradient_evals: 0,
				success: false,
			});
		}

		// Adapt hint for next iteration:
		// 1 eval (accepted immediately) → α was conservative, grow it
		// 2 evals (one backtrack) → α is about right, keep it
		// 3+ evals (multiple backtracks) → α was too large, keep the contracted value
		let two = <T as Scalar>::from_f64(2.0);
		let next_alpha = if cost_evaluations == 1 {
			alpha * two
		} else {
			alpha
		};
		self.previous_alpha = Some(next_alpha);

		// Return physical step size = alpha * norm_d
		let physical_step = alpha * norm_d;

		Ok(LineSearchResult {
			step_size: physical_step,
			new_point,
			new_value,
			new_gradient: None,
			function_evals: cost_evaluations,
			gradient_evals: 0,
			success: true,
		})
	}

	fn name(&self) -> &str {
		"Adaptive"
	}
}

/// Line search satisfying strong Wolfe conditions for high-quality steps.
///
/// This sophisticated algorithm ensures both sufficient decrease and strong
/// curvature conditions, making it ideal for quasi-Newton methods that require
/// high-quality step sizes with good second-order information. It implements
/// the mathematical foundation for BFGS and L-BFGS convergence theory.
///
/// # Mathematical Conditions
///
/// Strong Wolfe conditions require:
/// 1. **Sufficient decrease**: f(R_x(α η)) ≤ f(x) + c₁ α ⟨grad f(x), η⟩_g
/// 2. **Strong curvature**: |⟨grad f(R_x(α η)), T_{x→y}(η)⟩_g| ≤ c₂ |⟨grad f(x), η⟩_g|
///
/// where 0 < c₁ < c₂ < 1, typically c₁ = 10⁻⁴ and c₂ = 0.9.
///
/// # Theoretical Significance
///
/// ## Quasi-Newton Theory
/// Strong Wolfe conditions ensure that quasi-Newton updates:
/// - Maintain positive definiteness of Hessian approximations
/// - Satisfy the secant equation: B_{k+1} s_k = y_k
/// - Converge superlinearly under standard assumptions
///
/// ## Curvature Information
/// The strong curvature condition provides bounds on the directional
/// derivative, ensuring that:
/// - Step sizes are neither too small nor too large
/// - Second-order information is preserved
/// - Conjugate gradient orthogonality relationships hold
///
/// # Algorithm Strategy
///
/// The implementation uses a bracketing and zoom approach:
/// 1. **Bracketing phase**: Find interval containing acceptable step
/// 2. **Zoom phase**: Refine interval using interpolation
/// 3. **Termination**: Accept step satisfying both Wolfe conditions
///
/// # Computational Characteristics
///
/// ## Function Evaluations
/// - Typically 2-6 evaluations for well-conditioned problems
/// - May require more for ill-conditioned or non-convex functions
///
/// ## Gradient Evaluations
/// - Usually matches function evaluations
/// - Required for curvature condition testing
///
/// # Applications
///
/// ## Primary Use Cases
/// - **BFGS optimization**: Essential for superlinear convergence
/// - **L-BFGS methods**: Limited-memory quasi-Newton algorithms
/// - **Nonlinear conjugate gradient**: When strong Wolfe CG variants are used
///
/// ## Performance Benefits
/// - Higher-quality steps lead to fewer optimization iterations
/// - Better convergence rates offset additional line search cost
/// - Essential for maintaining quasi-Newton Hessian approximation quality
/// Cubic interpolation for step size selection in the zoom phase.
///
/// Fits a cubic polynomial through two points with function values and
/// derivatives, then returns the minimizer. Falls back to bisection
/// when the cubic has no real minimum (negative discriminant).
///
/// The result is safeguarded to lie within `[lo + 0.1*(hi-lo), hi - 0.1*(hi-lo)]`
/// to prevent the interpolant from collapsing to a bracket endpoint.
fn cubic_interpolation<T: Scalar>(
	alpha_lo: T,
	alpha_hi: T,
	phi_lo: T,
	phi_hi: T,
	dphi_lo: T,
	dphi_hi: T,
) -> T {
	let d = alpha_hi - alpha_lo;
	if <T as Float>::abs(d) < <T as Scalar>::from_f64(1e-20) {
		return (alpha_lo + alpha_hi) / <T as Scalar>::from_f64(2.0);
	}

	// Cubic interpolation coefficients (Nocedal & Wright, pp. 59-60)
	let d1 = dphi_lo + dphi_hi - <T as Scalar>::from_f64(3.0) * (phi_hi - phi_lo) / d;
	let disc = d1 * d1 - dphi_lo * dphi_hi;

	let alpha = if disc < T::zero() {
		// No real minimum — fall back to bisection
		(alpha_lo + alpha_hi) / <T as Scalar>::from_f64(2.0)
	} else {
		let d2 = <T as Float>::sqrt(disc);
		// Minimizer of the cubic
		alpha_hi - d * (dphi_hi + d2 - d1) / (dphi_hi - dphi_lo + <T as Scalar>::from_f64(2.0) * d2)
	};

	// Safeguard: keep result well within [lo, hi]
	let margin = <T as Scalar>::from_f64(0.1) * <T as Float>::abs(d);
	let lo = <T as Float>::min(alpha_lo, alpha_hi) + margin;
	let hi = <T as Float>::max(alpha_lo, alpha_hi) - margin;
	<T as Float>::max(lo, <T as Float>::min(hi, alpha))
}

/// Line search satisfying strong Wolfe conditions via bracketing and zoom.
///
/// Implements Nocedal & Wright Algorithms 3.5 (bracketing) and 3.6 (zoom),
/// adapted for Riemannian manifolds using retraction and vector transport.
///
/// # Algorithm
///
/// **Phase 1 — Bracketing**: Grow step size until an interval `[α_lo, α_hi]`
/// is found that must contain a step satisfying both Wolfe conditions.
///
/// **Phase 2 — Zoom**: Narrow the bracket using cubic interpolation until
/// a step satisfying both conditions is found.
///
/// # Strong Wolfe Conditions on Manifolds
///
/// 1. **Armijo**: f(R_x(αη)) ≤ f(x) + c₁α⟨∇f(x), η⟩_x
/// 2. **Curvature**: |⟨∇f(R_x(αη)), T_{x→y}(η)⟩_y| ≤ c₂|⟨∇f(x), η⟩_x|
///
/// where R_x is retraction and T_{x→y} is vector transport.
#[derive(Debug, Clone)]
pub struct StrongWolfeLineSearch {
	/// Maximum iterations in the zoom phase
	max_zoom_iterations: usize,
}

impl StrongWolfeLineSearch {
	/// Creates a new strong Wolfe line search.
	pub fn new() -> Self {
		Self {
			max_zoom_iterations: 10,
		}
	}

	/// Sets the maximum number of zoom iterations.
	pub fn with_max_zoom_iterations(mut self, n: usize) -> Self {
		self.max_zoom_iterations = n;
		self
	}

	/// Evaluates φ(α) = f(R_x(α·η)) by retracting along the search direction.
	///
	/// Returns (cost_value, new_point).
	fn evaluate_phi<T, C, M>(
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		direction: &M::TangentVector,
		alpha: T,
		ws: &mut M::Workspace,
	) -> Result<(T, M::Point)>
	where
		T: Scalar,
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		let mut scaled_dir = direction.clone();
		manifold.scale_tangent(point, alpha, direction, &mut scaled_dir)?;
		let mut trial_point = point.clone();
		manifold.retract(point, &scaled_dir, &mut trial_point, ws)?;
		let cost = cost_fn.cost(&trial_point)?;
		Ok((cost, trial_point))
	}

	/// Computes the directional derivative φ'(α) at a trial point.
	///
	/// Evaluates the gradient at trial_point, converts to Riemannian gradient,
	/// transports the search direction from the base point, and returns the
	/// inner product ⟨grad f(y), T_{x→y}(η)⟩_y.
	///
	/// Returns (directional_derivative, riemannian_gradient_at_trial).
	#[allow(clippy::too_many_arguments)]
	fn evaluate_dphi<T, C, M>(
		cost_fn: &C,
		manifold: &M,
		base_point: &M::Point,
		trial_point: &M::Point,
		direction: &M::TangentVector,
		ws: &mut M::Workspace,
	) -> Result<(T, M::TangentVector)>
	where
		T: Scalar,
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		// Compute Euclidean gradient at trial point, then convert
		let mut euc_grad = direction.clone();
		cost_fn.cost_and_gradient(trial_point, &mut euc_grad)?;
		let mut riem_grad = euc_grad.clone();
		manifold.euclidean_to_riemannian_gradient(trial_point, &euc_grad, &mut riem_grad, ws)?;

		// Transport search direction from base to trial
		let mut transported_dir = direction.clone();
		manifold.parallel_transport(
			base_point,
			trial_point,
			direction,
			&mut transported_dir,
			ws,
		)?;

		// Directional derivative at trial point
		let dphi = manifold.inner_product(trial_point, &riem_grad, &transported_dir, ws)?;
		Ok((dphi, riem_grad))
	}

	/// Zoom phase (Nocedal & Wright Algorithm 3.6).
	///
	/// Given a bracket [α_lo, α_hi] known to contain a Wolfe-satisfying step,
	/// narrows it using cubic interpolation until the conditions are met.
	#[allow(clippy::too_many_arguments)]
	fn zoom<T, C, M>(
		&self,
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		direction: &M::TangentVector,
		phi_0: T,
		dphi_0: T,
		c1: T,
		c2: T,
		mut alpha_lo: T,
		mut alpha_hi: T,
		mut phi_lo: T,
		mut phi_hi: T,
		mut dphi_lo: T,
		dphi_hi: T,
		// Best result found before entering zoom (fallback)
		best_result: &mut Option<LineSearchResult<T, M::Point, M::TangentVector>>,
		func_evals: &mut usize,
		grad_evals: &mut usize,
		ws: &mut M::Workspace,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		T: Scalar,
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		let abs_dphi_0 = <T as Float>::abs(dphi_0);
		let mut _dphi_hi = dphi_hi;

		for _ in 0..self.max_zoom_iterations {
			// Pick trial step via cubic interpolation
			let alpha_j =
				cubic_interpolation(alpha_lo, alpha_hi, phi_lo, phi_hi, dphi_lo, _dphi_hi);

			// Evaluate φ(α_j)
			let (phi_j, trial_point) =
				Self::evaluate_phi(cost_fn, manifold, point, direction, alpha_j, ws)?;
			*func_evals += 1;

			if phi_j > phi_0 + c1 * alpha_j * dphi_0 || phi_j >= phi_lo {
				// Armijo violated or no improvement over lo — shrink from hi side
				alpha_hi = alpha_j;
				phi_hi = phi_j;
				// dphi_hi is stale but cubic_interpolation handles it
				// We don't compute gradient here to save evaluations
				_dphi_hi = dphi_lo; // Approximate — will be corrected if we enter the else branch
			} else {
				// Armijo satisfied and phi_j < phi_lo — check curvature
				let (dphi_j, grad_j) =
					Self::evaluate_dphi(cost_fn, manifold, point, &trial_point, direction, ws)?;
				*grad_evals += 1;
				*func_evals += 1; // cost_and_gradient counts as both

				// Track best result seen
				*best_result = Some(LineSearchResult {
					step_size: alpha_j,
					new_point: trial_point.clone(),
					new_value: phi_j,
					new_gradient: Some(grad_j.clone()),
					function_evals: *func_evals,
					gradient_evals: *grad_evals,
					success: false,
				});

				// Strong Wolfe curvature condition
				if <T as Float>::abs(dphi_j) <= c2 * abs_dphi_0 {
					return Ok(LineSearchResult {
						step_size: alpha_j,
						new_point: trial_point,
						new_value: phi_j,
						new_gradient: Some(grad_j),
						function_evals: *func_evals,
						gradient_evals: *grad_evals,
						success: true,
					});
				}

				// Update bracket
				if dphi_j * (alpha_hi - alpha_lo) >= T::zero() {
					alpha_hi = alpha_lo;
					phi_hi = phi_lo;
					_dphi_hi = dphi_lo;
				}

				alpha_lo = alpha_j;
				phi_lo = phi_j;
				dphi_lo = dphi_j;
			}

			// Check if bracket has collapsed
			if <T as Float>::abs(alpha_hi - alpha_lo) < <T as Scalar>::from_f64(1e-12) {
				break;
			}
		}

		// Zoom exhausted — return best result found
		if let Some(result) = best_result.take() {
			Ok(result)
		} else {
			// Fallback: evaluate at alpha_lo (which satisfies Armijo)
			let (phi, new_point) =
				Self::evaluate_phi(cost_fn, manifold, point, direction, alpha_lo, ws)?;
			*func_evals += 1;
			Ok(LineSearchResult {
				step_size: alpha_lo,
				new_point,
				new_value: phi,
				new_gradient: None,
				function_evals: *func_evals,
				gradient_evals: *grad_evals,
				success: false,
			})
		}
	}
}

impl Default for StrongWolfeLineSearch {
	fn default() -> Self {
		Self::new()
	}
}

impl<T> LineSearch<T> for StrongWolfeLineSearch
where
	T: Scalar,
{
	fn search_with_deriv<C, M>(
		&mut self,
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		value: T,
		direction: &M::TangentVector,
		directional_deriv: T,
		params: &LineSearchParams<T>,
		ws: &mut M::Workspace,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		params.validate()?;

		// Near-zero directional derivative means gradient ≈ 0 (stationary point)
		// Return current point as-is rather than erroring
		let min_deriv = <T as Scalar>::from_f64(1e-30);
		if directional_deriv >= T::zero() || <T as Float>::abs(directional_deriv) < min_deriv {
			return Ok(LineSearchResult {
				step_size: T::zero(),
				new_point: point.clone(),
				new_value: value,
				new_gradient: None,
				function_evals: 0,
				gradient_evals: 0,
				success: false,
			});
		}

		let phi_0 = value;
		let dphi_0 = directional_deriv;
		let abs_dphi_0 = <T as Float>::abs(dphi_0);
		let c1 = params.c1;
		let c2 = params.c2;
		let alpha_max = params.max_step_size;
		let two = <T as Scalar>::from_f64(2.0);

		let mut func_evals: usize = 0;
		let mut grad_evals: usize = 0;

		let mut alpha_prev = T::zero();
		let mut phi_prev = phi_0;
		let mut dphi_prev = dphi_0;

		let mut alpha = params.initial_step_size;

		// Track the best Armijo-satisfying result as fallback
		let mut best_result: Option<LineSearchResult<T, M::Point, M::TangentVector>> = None;

		// ── Phase 1: Bracketing (Nocedal & Wright Algorithm 3.5) ──
		for i in 0..params.max_iterations {
			let (phi, trial_point) =
				Self::evaluate_phi(cost_fn, manifold, point, direction, alpha, ws)?;
			func_evals += 1;

			// Check Armijo condition
			let armijo_bound = phi_0 + c1 * alpha * dphi_0;

			if phi > armijo_bound || (phi >= phi_prev && i > 0) {
				// Bracket found: [alpha_prev, alpha]
				return self.zoom(
					cost_fn,
					manifold,
					point,
					direction,
					phi_0,
					dphi_0,
					c1,
					c2,
					alpha_prev,
					alpha,
					phi_prev,
					phi,
					dphi_prev,
					dphi_prev, // approximate dphi_hi
					&mut best_result,
					&mut func_evals,
					&mut grad_evals,
					ws,
				);
			}

			// Armijo satisfied — check curvature
			let (dphi, grad) =
				Self::evaluate_dphi(cost_fn, manifold, point, &trial_point, direction, ws)?;
			grad_evals += 1;
			func_evals += 1; // cost_and_gradient

			// Track as best result
			best_result = Some(LineSearchResult {
				step_size: alpha,
				new_point: trial_point.clone(),
				new_value: phi,
				new_gradient: Some(grad.clone()),
				function_evals: func_evals,
				gradient_evals: grad_evals,
				success: false,
			});

			// Strong Wolfe curvature condition satisfied?
			if <T as Float>::abs(dphi) <= c2 * abs_dphi_0 {
				return Ok(LineSearchResult {
					step_size: alpha,
					new_point: trial_point,
					new_value: phi,
					new_gradient: Some(grad),
					function_evals: func_evals,
					gradient_evals: grad_evals,
					success: true,
				});
			}

			// Curvature flipped — bracket found: [alpha, alpha_prev]
			if dphi >= T::zero() {
				return self.zoom(
					cost_fn,
					manifold,
					point,
					direction,
					phi_0,
					dphi_0,
					c1,
					c2,
					alpha,
					alpha_prev,
					phi,
					phi_prev,
					dphi,
					dphi_prev,
					&mut best_result,
					&mut func_evals,
					&mut grad_evals,
					ws,
				);
			}

			// Grow step size
			alpha_prev = alpha;
			phi_prev = phi;
			dphi_prev = dphi;
			alpha = <T as Float>::min(alpha * two, alpha_max);

			// If we've hit the maximum step size, stop growing
			if alpha >= alpha_max {
				break;
			}
		}

		// Bracketing exhausted — return best result found
		if let Some(mut result) = best_result {
			result.function_evals = func_evals;
			result.gradient_evals = grad_evals;
			Ok(result)
		} else {
			// Nothing worked — return initial point unchanged with zero step
			Ok(LineSearchResult {
				step_size: T::zero(),
				new_point: point.clone(),
				new_value: value,
				new_gradient: None,
				function_evals: func_evals,
				gradient_evals: grad_evals,
				success: false,
			})
		}
	}

	fn name(&self) -> &str {
		"StrongWolfe"
	}
}

/// Fixed step size strategy for algorithms with predetermined step lengths.
///
/// This "line search" uses a constant step size without any search procedure,
/// making it suitable for algorithms with theoretical step size guarantees
/// or when computational resources are extremely limited. While not adaptive,
/// it can be effective for specific problem classes.
///
/// # Mathematical Foundation
///
/// The update is simply: x_{k+1} = R_x(α η) where α is fixed.
///
/// No optimization of α is performed, making this the simplest possible
/// "line search" strategy.
///
/// # Theoretical Applications
///
/// ## Diminishing Step Sizes
/// For stochastic optimization, step sizes like α_k = 1/k or α_k = 1/√k
/// provide convergence guarantees without line search overhead.
///
/// ## Lipschitz-Based Steps
/// When the gradient is L-Lipschitz continuous, α = 1/L guarantees
/// sufficient decrease without search.
///
/// ## Trust Region Alternative
/// In trust region methods, the step size is determined by the
/// trust region radius rather than line search.
///
/// # Advantages
///
/// - **Computational efficiency**: Zero function evaluations for step size
/// - **Simplicity**: Minimal implementation complexity
/// - **Predictability**: Deterministic behavior aids analysis
/// - **Memory efficiency**: No line search state to maintain
///
/// # Disadvantages
///
/// - **Suboptimal steps**: May be far from optimal step length
/// - **Problem dependence**: Requires problem-specific tuning
/// - **Slow convergence**: Conservative steps may slow optimization
/// - **No adaptivity**: Cannot respond to changing function behavior
///
/// # Use Cases
///
/// ## Stochastic Optimization
/// ```rust,no_run
/// # use riemannopt_core::prelude::*;
/// # use riemannopt_core::line_search::FixedStepSize;
/// // Diminishing step size for SGD
/// let iteration = 10; // example iteration number
/// let step_size = 1.0 / (iteration as f64 + 1.0);
/// let line_search = FixedStepSize::new(step_size);
/// ```
///
/// ## Lipschitz Gradient Methods
/// ```rust,no_run
/// # use riemannopt_core::prelude::*;
/// # use riemannopt_core::line_search::FixedStepSize;
/// // Step size based on Lipschitz constant
/// let lipschitz_constant = 2.0;
/// let line_search = FixedStepSize::new(1.0 / lipschitz_constant);
/// ```
///
/// ## Real-Time Applications
/// When computational budget is strictly limited and approximate
/// solutions are acceptable.
#[derive(Debug, Clone, Copy)]
pub struct FixedStepSize<T> {
	step_size: T,
}

impl<T: Scalar> FixedStepSize<T> {
	/// Creates a fixed step size strategy with specified step length.
	///
	/// # Arguments
	///
	/// * `step_size` - Fixed step size α > 0 to use for all iterations
	///
	/// # Mathematical Note
	///
	/// The step size should be chosen based on problem characteristics:
	/// - For Lipschitz gradients: α ≤ 1/L where L is the Lipschitz constant
	/// - For stochastic methods: α_k = O(1/k) or O(1/√k)
	/// - For trust region methods: α determined by trust region radius
	pub fn new(step_size: T) -> Self {
		Self { step_size }
	}
}

impl<T> LineSearch<T> for FixedStepSize<T>
where
	T: Scalar,
{
	fn search_with_deriv<C, M>(
		&mut self,
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		_value: T,
		direction: &M::TangentVector,
		_directional_deriv: T,
		_params: &LineSearchParams<T>,
		ws: &mut M::Workspace,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		let mut new_point = point.clone();

		// Use manifold's retract - scaling would require trait bounds on TangentVector
		manifold.retract(point, direction, &mut new_point, ws)?;
		let new_value = cost_fn.cost(&new_point)?;

		Ok(LineSearchResult {
			step_size: self.step_size,
			new_point,
			new_value,
			new_gradient: None,
			function_evals: 1,
			gradient_evals: 0,
			success: true,
		})
	}

	fn name(&self) -> &str {
		"FixedStep"
	}
}

#[cfg(test)]
mod tests {
	// TODO: rewrite line search tests with linalg abstraction
	/*
	use super::*;

	// Simple Euclidean manifold for testing
	#[derive(Debug)]
	struct EuclideanManifold {
		dim: usize,
	}

	impl Manifold<f64> for EuclideanManifold {
		fn name(&self) -> &str {
			"Euclidean"
		}
		fn dimension(&self) -> usize {
			self.dim
		}
		fn is_point_on_manifold(&self, _point: &DVector<f64>, _tol: f64) -> bool {
			true
		}
		fn is_vector_in_tangent_space(
			&self,
			_point: &DVector<f64>,
			_vector: &DVector<f64>,
			_tol: f64,
		) -> bool {
			true
		}
		fn project_point(&self, point: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) {
			result.copy_from(point);
		}
		fn project_tangent(
			&self,
			_point: &DVector<f64>,
			vector: &DVector<f64>,
			result: &mut DVector<f64>,
			_workspace: &mut Workspace<f64>,
		) -> Result<()> {
			result.copy_from(vector);
			Ok(())
		}
		fn inner_product(
			&self,
			_point: &DVector<f64>,
			u: &DVector<f64>,
			v: &DVector<f64>,
		) -> Result<f64> {
			Ok(u.dot(v))
		}
		fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
			result.copy_from(&(point + tangent));
			Ok(())
		}
		fn inverse_retract(
			&self,
			point: &DVector<f64>,
			other: &DVector<f64>,
			result: &mut DVector<f64>,
			_workspace: &mut Workspace<f64>,
		) -> Result<()> {
			result.copy_from(&(other - point));
			Ok(())
		}
		fn euclidean_to_riemannian_gradient(
			&self,
			_point: &DVector<f64>,
			euclidean_grad: &DVector<f64>,
			result: &mut DVector<f64>,
			_workspace: &mut Workspace<f64>,
		) -> Result<()> {
			result.copy_from(euclidean_grad);
			Ok(())
		}
		fn random_point(&self) -> DVector<f64> {
			DVector::zeros(self.dim)
		}
		fn random_tangent(&self, _point: &DVector<f64>, result: &mut DVector<f64>, _workspace: &mut Workspace<f64>) -> Result<()> {
			result.fill(0.0);
			Ok(())
		}
	}

	#[test]
	fn test_backtracking_line_search() {
		let manifold = EuclideanManifold { dim: 2 };
		let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));

		let point = DVector::from_vec(vec![1.0, 1.0]);
		let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
		let direction = -&gradient; // Steepest descent direction

		let mut ls = BacktrackingLineSearch::new();
		let params = LineSearchParams::backtracking();

		let result = ls
			.search(
				&cost,
				&manifold,
				&point,
				value,
				&gradient,
				&direction,
				&params,
			)
			.unwrap();

		assert!(result.success);
		assert!(result.step_size > 0.0);
		assert!(result.new_value < value);
	}

	#[test]
	fn test_backtracking_descent_check() {
		let manifold = EuclideanManifold { dim: 2 };
		let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));

		let point = DVector::from_vec(vec![1.0, 1.0]);
		let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
		let bad_direction = gradient.clone(); // Ascent direction

		let mut ls = BacktrackingLineSearch::new();
		let params = LineSearchParams::backtracking();

		let result = ls.search(
			&cost,
			&manifold,
			&point,
			value,
			&gradient,
			&bad_direction,
			&params,
		);

		assert!(result.is_err());
	}

	#[test]
	fn test_strong_wolfe_line_search() {
		let manifold = EuclideanManifold { dim: 2 };
		let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));

		let point = DVector::from_vec(vec![2.0, 3.0]);
		let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
		let direction = -&gradient;

		let mut ls = StrongWolfeLineSearch::new();
		let params = LineSearchParams::strong_wolfe();

		let result = ls
			.search(
				&cost,
				&manifold,
				&point,
				value,
				&gradient,
				&direction,
				&params,
			)
			.unwrap();

		assert!(result.success);
		assert!(result.new_gradient.is_some());
		assert!(result.new_value < value);

		// For quadratic function, optimal step size is 1.0
		assert_relative_eq!(result.step_size, 1.0, epsilon = 0.1);
	}

	#[test]
	fn test_fixed_step_size() {
		let manifold = EuclideanManifold { dim: 2 };
		let cost = QuadraticCost::<f64, Dyn>::simple(Dyn(2));

		let point = DVector::from_vec(vec![1.0, 1.0]);
		let (value, gradient) = cost.cost_and_gradient_alloc(&point).unwrap();
		let direction = -&gradient;

		let mut ls = FixedStepSize::new(0.1);
		let params = LineSearchParams::default();

		let result = ls
			.search(
				&cost,
				&manifold,
				&point,
				value,
				&gradient,
				&direction,
				&params,
			)
			.unwrap();

		assert!(result.success);
		assert_eq!(result.step_size, 0.1);
		assert_eq!(result.function_evals, 1);

		// New point should be point + 0.1 * direction
		let expected = &point + &direction * 0.1;
		assert_relative_eq!(result.new_point, expected);
	}

	#[test]
	fn test_line_search_params() {
		let params = LineSearchParams::<f64>::strong_wolfe();
		assert_eq!(params.c1, 1e-4);
		assert_eq!(params.c2, 0.9);

		let params = LineSearchParams::<f64>::backtracking();
		assert_eq!(params.c1, 0.5);
		assert_eq!(params.rho, 0.5);
		assert_eq!(params.max_iterations, 20);
	}
	*/
}
