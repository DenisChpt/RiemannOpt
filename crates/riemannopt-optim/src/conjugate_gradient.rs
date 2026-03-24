//! Riemannian Conjugate Gradient optimizer.
//!
//! Conjugate gradient methods are a class of optimization algorithms that use
//! conjugate directions to achieve faster convergence than steepest descent.
//! This implementation extends classical CG methods to Riemannian manifolds.
//!
//! # Algorithm Overview
//!
//! At each iteration, the conjugate gradient method:
//! 1. Computes the Riemannian gradient
//! 2. Determines a conjugate direction using the chosen beta formula
//! 3. Performs line search along the conjugate direction
//! 4. Updates the position using retraction
//!
//! # Supported Methods
//!
//! - **Fletcher-Reeves (FR)**: β = ||g_k||² / ||g_{k-1}||²
//! - **Polak-Ribière (PR)**: β = <g_k, g_k - g_{k-1}> / ||g_{k-1}||²
//! - **Hestenes-Stiefel (HS)**: β = <g_k, g_k - g_{k-1}> / <d_{k-1}, g_k - g_{k-1}>
//! - **Dai-Yuan (DY)**: β = ||g_k||² / <d_{k-1}, g_k - g_{k-1}>
//!
//! # Key Features
//!
//! - **Multiple CG variants**: FR, PR, HS, DY methods
//! - **Automatic restarts**: Periodic or condition-based restarts
//! - **Preconditioning support**: Optional preconditioner application
//! - **Hybrid methods**: PR+ (non-negative PR) and other variants
//! - **Line search integration**: Ensures sufficient decrease
//!
//! # References
//!
//! - Hager & Zhang, "A survey of nonlinear conjugate gradient methods" (2006)
//! - Dai & Yuan, "A nonlinear conjugate gradient method with a strong global convergence property" (1999)
//! - Ring & Wirth, "Optimization methods on Riemannian manifolds and their application to shape space" (2012)

use num_traits::Float;
use riemannopt_core::{
	core::{cost_function::CostFunction, manifold::Manifold},
	error::Result,
	optimization::{
		line_search::{LineSearch, LineSearchParams, LineSearchResult, StrongWolfeLineSearch},
		optimizer::{OptimizationResult, Optimizer, StoppingCriterion, TerminationReason},
	},
	types::Scalar,
};
use std::fmt::Debug;
use std::time::Instant;

/// Conjugate gradient method variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConjugateGradientMethod {
	/// Fletcher-Reeves
	FletcherReeves,
	/// Polak-Ribière
	PolakRibiere,
	/// Hestenes-Stiefel
	HestenesStiefel,
	/// Dai-Yuan
	DaiYuan,
}

/// State for conjugate gradient optimization.
#[derive(Debug)]
pub struct ConjugateGradientState<T, TV>
where
	T: Scalar,
	TV: Clone + Debug + Send + Sync,
{
	_phantom: std::marker::PhantomData<T>,
	/// Previous search direction
	pub previous_direction: Option<TV>,
	/// Previous gradient
	pub previous_gradient: Option<TV>,
	/// Previous gradient norm squared ‖g_k‖² computed BEFORE transport
	/// (used as denominator in beta formulas to avoid transport-error accumulation)
	pub previous_gradient_norm_sq: Option<T>,
	/// Method for computing beta (FR, PR, HS, DY)
	pub method: ConjugateGradientMethod,
	/// Number of iterations since last restart
	pub iterations_since_restart: usize,
	/// Restart period (0 means no periodic restart)
	pub restart_period: usize,
	/// Last accepted step size (for adaptive initial step)
	pub last_step_size: Option<T>,
}

impl<T, TV> ConjugateGradientState<T, TV>
where
	T: Scalar,
	TV: Clone + Debug + Send + Sync,
{
	/// Creates a new conjugate gradient state.
	pub fn new(method: ConjugateGradientMethod, restart_period: usize) -> Self {
		Self {
			previous_direction: None,
			previous_gradient: None,
			previous_gradient_norm_sq: None,
			method,
			iterations_since_restart: 0,
			restart_period,
			last_step_size: None,
			_phantom: std::marker::PhantomData,
		}
	}

	/// Resets the state (for restarts).
	pub fn reset(&mut self) {
		self.previous_direction = None;
		self.previous_gradient = None;
		self.previous_gradient_norm_sq = None;
		self.iterations_since_restart = 0;
		// Keep last_step_size across restarts
	}
}

/// Configuration for the Conjugate Gradient optimizer.
#[derive(Debug, Clone)]
pub struct CGConfig<T: Scalar> {
	/// The CG method variant to use
	pub method: ConjugateGradientMethod,
	/// Period for automatic restarts (0 = no automatic restart)
	pub restart_period: usize,
	/// Whether to use PR+ (non-negative Polak-Ribière)
	pub use_pr_plus: bool,
	/// Minimum value of beta before restart
	pub min_beta: Option<T>,
	/// Maximum value of beta allowed
	pub max_beta: Option<T>,
	/// Line search parameters
	pub line_search_params: LineSearchParams<T>,
}

impl<T: Scalar> Default for CGConfig<T> {
	fn default() -> Self {
		Self {
			method: ConjugateGradientMethod::HestenesStiefel,
			restart_period: 0, // No automatic restart by default
			use_pr_plus: true, // Use max(0, beta) safeguard for HS too
			min_beta: None,
			max_beta: None,
			line_search_params: LineSearchParams::for_conjugate_gradient(),
		}
	}
}

impl<T: Scalar> CGConfig<T> {
	/// Creates a new configuration with default parameters.
	pub fn new() -> Self {
		Self::default()
	}

	/// Sets the CG method variant.
	pub fn with_method(mut self, method: ConjugateGradientMethod) -> Self {
		self.method = method;
		self
	}

	/// Sets the restart period (n means restart every n iterations).
	pub fn with_restart_period(mut self, period: usize) -> Self {
		self.restart_period = period;
		self
	}

	/// Enables or disables PR+ (non-negative Polak-Ribière).
	pub fn with_pr_plus(mut self, use_pr_plus: bool) -> Self {
		self.use_pr_plus = use_pr_plus;
		self
	}

	/// Sets the minimum beta value before restart.
	pub fn with_min_beta(mut self, min_beta: T) -> Self {
		self.min_beta = Some(min_beta);
		self
	}

	/// Sets the maximum beta value allowed.
	pub fn with_max_beta(mut self, max_beta: T) -> Self {
		self.max_beta = Some(max_beta);
		self
	}

	/// Sets custom line search parameters.
	pub fn with_line_search_params(mut self, params: LineSearchParams<T>) -> Self {
		self.line_search_params = params;
		self
	}

	/// Creates a configuration for Fletcher-Reeves method.
	pub fn fletcher_reeves() -> Self {
		Self::new().with_method(ConjugateGradientMethod::FletcherReeves)
	}

	/// Creates a configuration for Polak-Ribière method.
	pub fn polak_ribiere() -> Self {
		Self::new().with_method(ConjugateGradientMethod::PolakRibiere)
	}

	/// Creates a configuration for Hestenes-Stiefel method.
	pub fn hestenes_stiefel() -> Self {
		Self::new().with_method(ConjugateGradientMethod::HestenesStiefel)
	}

	/// Creates a configuration for Dai-Yuan method.
	pub fn dai_yuan() -> Self {
		Self::new().with_method(ConjugateGradientMethod::DaiYuan)
	}
}

/// Riemannian Conjugate Gradient optimizer.
///
/// This optimizer adapts the classical conjugate gradient algorithm to Riemannian
/// manifolds by properly handling the transport of search directions and using
/// the manifold's metric for inner products.
///
/// # Examples
///
/// ```rust,ignore
/// use riemannopt_optim::{ConjugateGradient, CGConfig};
///
/// // Basic CG with Polak-Ribière method
/// let cg: ConjugateGradient<f64> = ConjugateGradient::new(CGConfig::new());
///
/// // CG with Fletcher-Reeves and periodic restart
/// let cg_fr = ConjugateGradient::new(
///     CGConfig::fletcher_reeves()
///         .with_restart_period(10)
///         .with_min_beta(0.0)
/// );
/// ```
#[derive(Debug)]
pub struct ConjugateGradient<T: Scalar> {
	config: CGConfig<T>,
}

impl<T: Scalar> ConjugateGradient<T> {
	/// Creates a new Conjugate Gradient optimizer with the given configuration.
	pub fn new(config: CGConfig<T>) -> Self {
		Self { config }
	}

	/// Creates a new Conjugate Gradient optimizer with default configuration.
	pub fn with_default_config() -> Self {
		Self::new(CGConfig::default())
	}

	/// Returns the configuration.
	pub fn config(&self) -> &CGConfig<T> {
		&self.config
	}

	/// Checks stopping criteria internally
	fn check_stopping_criteria<M>(
		&self,
		manifold: &M,
		iteration: usize,
		function_evaluations: usize,
		_gradient_evaluations: usize,
		start_time: Instant,
		current_cost: T,
		previous_cost: Option<T>,
		gradient_norm: Option<T>,
		current_point: &M::Point,
		previous_point: &Option<M::Point>,
		criterion: &StoppingCriterion<T>,
	) -> Option<TerminationReason>
	where
		M: Manifold<T>,
	{
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
			if let Some(ref prev_point) = previous_point {
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

	/// Computes the beta coefficient for the conjugate gradient method.
	///
	/// Computes β and the new conjugate direction in a single pass.
	///
	/// Transports `prev_dir` and `prev_grad` once and uses them for both β
	/// computation and direction construction, halving the transport cost
	/// compared to separate compute_beta + direction_update.
	///
	/// Writes the new direction into `direction`. Returns whether a restart occurred.
	fn compute_direction<M>(
		&self,
		manifold: &M,
		current_point: &M::Point,
		previous_point: &M::Point,
		gradient: &M::TangentVector,
		grad_norm_sq: T,
		cg_state: &ConjugateGradientState<T, M::TangentVector>,
		direction: &mut M::TangentVector,
	) -> Result<bool>
	where
		M: Manifold<T>,
	{
		if cg_state.previous_gradient.is_none() || cg_state.previous_direction.is_none() {
			manifold.scale_tangent(current_point, -T::one(), gradient, direction)?;
			return Ok(true);
		}

		let prev_grad = cg_state.previous_gradient.as_ref().unwrap();
		let prev_dir = cg_state.previous_direction.as_ref().unwrap();
		let prev_grad_norm_sq = cg_state
			.previous_gradient_norm_sq
			.unwrap_or_else(T::epsilon);

		// Transport previous gradient to current tangent space
		let mut transported_prev_grad = gradient.clone(); // reuse shape
		if manifold
			.parallel_transport(previous_point, current_point, prev_grad, &mut transported_prev_grad)
			.is_err()
		{
			manifold.scale_tangent(current_point, -T::one(), gradient, direction)?;
			return Ok(true);
		}

		// Transport previous direction to current tangent space (stored in `direction`)
		if manifold
			.parallel_transport(previous_point, current_point, prev_dir, direction)
			.is_err()
		{
			manifold.scale_tangent(current_point, -T::one(), gradient, direction)?;
			return Ok(true);
		}
		// `direction` now holds τ(d_{k-1})

		// grad_diff = g_k - τ(g_{k-1})
		let mut grad_diff = gradient.clone();
		manifold.scale_tangent(
			current_point,
			-T::one(),
			&transported_prev_grad,
			&mut grad_diff,
		)?;
		let neg_transported = grad_diff.clone();
		manifold.add_tangents(
			current_point,
			gradient,
			&neg_transported,
			&mut grad_diff,
			&mut transported_prev_grad, // reuse as temp
		)?;

		// Compute beta (direction holds τ(d_{k-1}))
		let beta = match cg_state.method {
			ConjugateGradientMethod::FletcherReeves => {
				if prev_grad_norm_sq > T::epsilon() {
					grad_norm_sq / prev_grad_norm_sq
				} else {
					T::zero()
				}
			}
			ConjugateGradientMethod::PolakRibiere => {
				let num = manifold.inner_product(current_point, gradient, &grad_diff)?;
				if prev_grad_norm_sq > T::epsilon() {
					let b = num / prev_grad_norm_sq;
					if self.config.use_pr_plus {
						<T as Float>::max(T::zero(), b)
					} else {
						b
					}
				} else {
					T::zero()
				}
			}
			ConjugateGradientMethod::HestenesStiefel => {
				let num = manifold.inner_product(current_point, gradient, &grad_diff)?;
				let den = manifold.inner_product(current_point, direction, &grad_diff)?;
				let tiny = T::from(100.0).unwrap()
					* T::epsilon()
					* <T as Float>::sqrt(
						manifold.inner_product(current_point, direction, direction)?
							* manifold.inner_product(current_point, &grad_diff, &grad_diff)?,
					);
				if <T as Float>::abs(den) > tiny {
					<T as Float>::max(T::zero(), num / den)
				} else {
					T::zero()
				}
			}
			ConjugateGradientMethod::DaiYuan => {
				let den = manifold.inner_product(current_point, direction, &grad_diff)?;
				let tiny = T::from(100.0).unwrap()
					* T::epsilon()
					* <T as Float>::sqrt(
						manifold.inner_product(current_point, direction, direction)?
							* manifold.inner_product(current_point, &grad_diff, &grad_diff)?,
					);
				if <T as Float>::abs(den) > tiny {
					grad_norm_sq / den
				} else {
					T::zero()
				}
			}
		};

		// Apply beta bounds
		let mut beta = beta;
		if let Some(min_b) = self.config.min_beta {
			if beta < min_b {
				beta = T::zero();
			}
		}
		if let Some(max_b) = self.config.max_beta {
			beta = <T as Float>::min(beta, max_b);
		}

		// Check for restart conditions
		if beta <= T::zero()
			|| beta.is_nan()
			|| (cg_state.restart_period > 0
				&& cg_state.iterations_since_restart >= cg_state.restart_period)
		{
			manifold.scale_tangent(current_point, -T::one(), gradient, direction)?;
			return Ok(true);
		}

		// Build d = -g + β * τ(d_{k-1})
		// direction holds τ(d_{k-1}), scale by β
		let transported_dir = direction.clone();
		manifold.scale_tangent(current_point, beta, &transported_dir, direction)?;
		// Add -gradient
		manifold.scale_tangent(current_point, -T::one(), gradient, &mut grad_diff)?;
		let scaled_dir = direction.clone();
		let mut temp = transported_prev_grad; // reuse already-consumed buffer
		manifold.add_tangents(current_point, &grad_diff, &scaled_dir, direction, &mut temp)?;

		// Project to tangent space for numerical safety
		manifold.project_tangent(current_point, &direction.clone(), direction)?;

		Ok(false)
	}

	/// Performs Strong Wolfe line search to find an appropriate step size.
	///
	/// Uses an adaptive initial step size:
	/// - First iteration: `α₀ = 1 / ‖d‖` to normalize the step
	/// - Subsequent iterations: reuse the previous accepted step × 2
	///
	/// Returns the full `LineSearchResult` so the optimizer can reuse the
	/// new point, cost value, and gradient (avoiding redundant evaluations).
	fn perform_line_search<M, C>(
		&self,
		cost_fn: &C,
		manifold: &M,
		point: &M::Point,
		direction: &M::TangentVector,
		current_cost: T,
		gradient: &M::TangentVector,
		function_evaluations: &mut usize,
		gradient_evaluations: &mut usize,
		cg_state: &ConjugateGradientState<T, M::TangentVector>,
	) -> Result<LineSearchResult<T, M::Point, M::TangentVector>>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		// Compute directional derivative: φ'(0) = ⟨grad f, d⟩
		let directional_derivative = manifold.inner_product(point, gradient, direction)?;

		// Adaptive initial step size:
		// First iteration: normalize by direction norm to avoid overshooting
		// Subsequent iterations: double the last accepted step (optimistic)
		let dir_norm_sq = manifold.inner_product(point, direction, direction)?;
		let dir_norm = <T as Float>::sqrt(dir_norm_sq);
		let initial_step = if let Some(prev_alpha) = cg_state.last_step_size {
			let two = <T as Scalar>::from_f64(2.0);
			<T as Float>::min(
				prev_alpha * two,
				self.config.line_search_params.max_step_size,
			)
		} else if dir_norm > T::zero() {
			T::one() / dir_norm
		} else {
			T::one()
		};

		// Override initial step size in params
		let mut params = self.config.line_search_params.clone();
		params.initial_step_size = initial_step;

		let mut ls = StrongWolfeLineSearch::new();
		let result = ls.search_with_deriv(
			cost_fn,
			manifold,
			point,
			current_cost,
			direction,
			directional_derivative,
			&params,
		)?;

		*function_evaluations += result.function_evals;
		*gradient_evaluations += result.gradient_evals;

		Ok(result)
	}
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for ConjugateGradient<T> {
	fn name(&self) -> &str {
		match self.config.method {
			ConjugateGradientMethod::FletcherReeves => "Riemannian CG-FR",
			ConjugateGradientMethod::PolakRibiere => {
				if self.config.use_pr_plus {
					"Riemannian CG-PR+"
				} else {
					"Riemannian CG-PR"
				}
			}
			ConjugateGradientMethod::HestenesStiefel => "Riemannian CG-HS",
			ConjugateGradientMethod::DaiYuan => "Riemannian CG-DY",
		}
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
		let start_time = Instant::now();

		// Initialize state
		let initial_cost = cost_fn.cost(initial_point)?;
		let mut cg_state = ConjugateGradientState::<T, M::TangentVector>::new(
			self.config.method,
			self.config.restart_period,
		);

		let mut current_point = initial_point.clone();
		let mut previous_point: Option<M::Point> = None;
		let mut current_cost = initial_cost;
		let mut previous_cost: Option<T> = None;
		let mut gradient_norm: Option<T> = None;
		let mut iteration = 0;
		let mut function_evaluations = 1;
		let mut gradient_evaluations = 0;

		// Compute initial gradient to get the right type
		let mut euclidean_grad = cost_fn.gradient(&initial_point)?;
		let mut riemannian_grad = euclidean_grad.clone();
		gradient_evaluations += 1;

		// When the line search returns a gradient at the new point, we reuse it
		let mut has_reusable_gradient = false;

		// Main optimization loop
		loop {
			// Check stopping criteria
			let reason = self.check_stopping_criteria(
				manifold,
				iteration,
				function_evaluations,
				gradient_evaluations,
				start_time,
				current_cost,
				previous_cost,
				gradient_norm,
				&current_point,
				&previous_point,
				stopping_criterion,
			);

			if let Some(reason) = reason {
				return Ok(OptimizationResult::new(
					current_point,
					current_cost,
					iteration,
					start_time.elapsed(),
					reason,
				)
				.with_function_evaluations(function_evaluations)
				.with_gradient_evaluations(gradient_evaluations)
				.with_gradient_norm(gradient_norm.unwrap_or(T::zero())));
			}

			// Compute gradient at current point (skip if line search already provided it)
			if !has_reusable_gradient {
				let _new_cost =
					cost_fn.cost_and_gradient(&current_point, &mut euclidean_grad)?;
				function_evaluations += 1;
				gradient_evaluations += 1;

				// Convert to Riemannian gradient
				manifold.euclidean_to_riemannian_gradient(
					&current_point,
					&euclidean_grad,
					&mut riemannian_grad,
				)?;
			}
			// Reset flag — will be set again if next line search provides gradient
			has_reusable_gradient = false;

			// Compute gradient norm
			let grad_norm_squared =
				manifold.inner_product(&current_point, &riemannian_grad, &riemannian_grad)?;
			let grad_norm = <T as Float>::sqrt(grad_norm_squared);
			gradient_norm = Some(grad_norm);

			// Compute β and conjugate direction in one pass (single transport)
			let mut direction = riemannian_grad.clone();
			let restarted = if iteration > 0 && previous_point.is_some() {
				self.compute_direction(
					manifold,
					&current_point,
					previous_point.as_ref().unwrap(),
					&riemannian_grad,
					grad_norm_squared,
					&cg_state,
					&mut direction,
				)?
			} else {
				manifold.scale_tangent(
					&current_point,
					-T::one(),
					&riemannian_grad,
					&mut direction,
				)?;
				true
			};

			if restarted {
				cg_state.iterations_since_restart = 0;
			}

			// Safeguard: ensure descent direction
			let directional_derivative =
				manifold.inner_product(&current_point, &riemannian_grad, &direction)?;
			if directional_derivative >= T::zero() {
				manifold.scale_tangent(
					&current_point,
					-T::one(),
					&riemannian_grad,
					&mut direction,
				)?;
				cg_state.iterations_since_restart = 0;
			}

			// Perform line search (returns full result with new point, cost, gradient)
			let ls_result = self.perform_line_search(
				cost_fn,
				manifold,
				&current_point,
				&direction,
				current_cost,
				&riemannian_grad,
				&mut function_evaluations,
				&mut gradient_evaluations,
				&cg_state,
			)?;
			// Detect stagnation: if the step size falls below machine epsilon
			// relative to the direction norm, the line search can no longer make
			// meaningful progress. Terminate early instead of wasting iterations.
			let min_step = <T as Scalar>::from_f64(1e-10);
			if ls_result.step_size < min_step && iteration > 0 {
				return Ok(OptimizationResult::new(
					current_point,
					current_cost,
					iteration,
					start_time.elapsed(),
					TerminationReason::Converged,
				)
				.with_function_evaluations(function_evaluations)
				.with_gradient_evaluations(gradient_evaluations)
				.with_gradient_norm(grad_norm));
			}

			// Only store positive step sizes — a zero step (failed line search)
			// would cause initial_step_size = 0 on next iteration, crashing the line search.
			if ls_result.step_size > T::zero() {
				cg_state.last_step_size = Some(ls_result.step_size);
			}

			// Update state — reuse results from line search
			cg_state.previous_direction = Some(direction);
			cg_state.previous_gradient = Some(riemannian_grad.clone());
			// Store gradient norm BEFORE transport — used as denominator in beta
			// to avoid transport-error accumulation.
			cg_state.previous_gradient_norm_sq = Some(grad_norm_squared);
			cg_state.iterations_since_restart += 1;

			previous_point = Some(current_point.clone());
			current_point = ls_result.new_point;
			previous_cost = Some(current_cost);
			current_cost = ls_result.new_value;

			// If the line search computed the gradient, reuse it for the next iteration
			if let Some(ls_grad) = ls_result.new_gradient {
				riemannian_grad = ls_grad;
				has_reusable_gradient = true;
			}

			iteration += 1;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_cg_config() {
		let config = CGConfig::<f64>::fletcher_reeves()
			.with_restart_period(10)
			.with_min_beta(-0.5)
			.with_max_beta(5.0);

		assert!(matches!(
			config.method,
			ConjugateGradientMethod::FletcherReeves
		));
		assert_eq!(config.restart_period, 10);
		assert_eq!(config.min_beta, Some(-0.5));
		assert_eq!(config.max_beta, Some(5.0));
	}

	#[test]
	fn test_cg_variants() {
		let fr_config = CGConfig::<f64>::fletcher_reeves();
		let pr_config = CGConfig::<f64>::polak_ribiere();
		let hs_config = CGConfig::<f64>::hestenes_stiefel();
		let dy_config = CGConfig::<f64>::dai_yuan();

		assert!(matches!(
			fr_config.method,
			ConjugateGradientMethod::FletcherReeves
		));
		assert!(matches!(
			pr_config.method,
			ConjugateGradientMethod::PolakRibiere
		));
		assert!(matches!(
			hs_config.method,
			ConjugateGradientMethod::HestenesStiefel
		));
		assert!(matches!(dy_config.method, ConjugateGradientMethod::DaiYuan));
	}

	#[test]
	fn test_cg_builder() {
		let cg = ConjugateGradient::<f64>::with_default_config();
		assert_eq!(cg.name(), "Riemannian CG-HS");
	}
}
