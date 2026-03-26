//! Riemannian L-BFGS optimizer.
//!
//! L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton optimization
//! algorithm that approximates the inverse Hessian using a limited history of past gradient
//! and position updates. This implementation extends L-BFGS to Riemannian manifolds by
//! transporting all stored vectors to the current tangent space after each step.
//!
//! # Algorithm Overview
//!
//! The Riemannian L-BFGS algorithm:
//! 1. Stores m most recent gradient differences and position differences
//! 2. After each step, transports ALL stored vectors to the new tangent space
//! 3. Approximates the inverse Hessian-vector product using two-loop recursion
//! 4. Computes search direction as negative approximate Newton direction
//! 5. Performs line search to find suitable step size
//! 6. Updates position using retraction
//!
//! ## Two-Loop Recursion Algorithm
//!
//! Since all stored vectors are maintained in the current tangent space,
//! the recursion is the standard L-BFGS two-loop (no transport needed):
//!
//! ```text
//! q = grad_f(x_k)
//! for i = k-1, k-2, ..., k-m:
//!     α_i = ρ_i * <s_i, q>
//!     q = q - α_i * y_i
//!
//! r = H_0 * q  // Initial Hessian approximation
//!
//! for i = k-m, k-m+1, ..., k-1:
//!     β = ρ_i * <y_i, r>
//!     r = r + (α_i - β) * s_i
//!
//! return -r  // Search direction
//! ```
//!
//! ## Riemannian Adaptations
//!
//! The key adaptations for Riemannian manifolds:
//! - **Eager transport**: After each iteration, ALL stored (s_i, y_i) pairs are
//!   transported from the old tangent space to the new tangent space
//! - **No transport in recursion**: The two-loop recursion operates purely in the
//!   current tangent space, avoiding inconsistency errors
//! - **Cautious updates**: Pairs are only stored if they satisfy a curvature condition
//! - **Normalization**: s_k and y_k are normalized by ‖s_k‖ before storage
//!
//! # Key Features
//!
//! - **Limited memory**: Only stores m vector pairs (typically 5-20)
//! - **Superlinear convergence**: Near optimal points with good initial Hessian approximation
//! - **Automatic scaling**: Initial Hessian approximation based on most recent update
//! - **Strong Wolfe line search**: Ensures sufficient decrease and curvature conditions
//!
//! # References
//!
//! - Nocedal & Wright, "Numerical Optimization" (2006)
//! - Huang et al., "A Riemannian BFGS Method" (2015)
//! - Sato, "Riemannian Optimization and Its Applications" (2021)
//! - Boumal, "An Introduction to Optimization on Smooth Manifolds" (2023)

use num_traits::Float;
use riemannopt_core::{
	core::{cost_function::CostFunction, manifold::Manifold},
	error::Result,
	optimization::{
		line_search::LineSearchParams,
		optimizer::{OptimizationResult, Optimizer, StoppingCriterion, TerminationReason},
	},
	types::Scalar,
};
use std::collections::VecDeque;
use std::fmt::Debug;
use std::time::Instant;

/// Storage for one iteration's L-BFGS data.
///
/// All vectors are maintained in the current tangent space. After each iteration,
/// all stored entries are transported to the new tangent space.
#[derive(Debug)]
pub struct LBFGSHistoryEntry<T, TV>
where
	T: Scalar,
	TV: Clone + Debug + Send + Sync,
{
	/// Position difference vector s_k (normalized, in current tangent space)
	s: TV,
	/// Gradient difference vector y_k (normalized, in current tangent space)
	y: TV,
	/// Inner product rho_k = 1 / <y_k, s_k>
	rho: T,
	/// Iteration number when this entry was created
	_iteration: usize,
}

/// State for L-BFGS history management.
#[derive(Debug)]
pub struct LBFGSState<T, TV>
where
	T: Scalar,
	TV: Clone + Debug + Send + Sync,
{
	/// History entries stored in chronological order
	pub history: VecDeque<LBFGSHistoryEntry<T, TV>>,
	/// Maximum number of entries to store
	pub memory_size: usize,
	/// Whether to use cautious updates
	pub use_cautious_updates: bool,
}

impl<T, TV> LBFGSState<T, TV>
where
	T: Scalar,
	TV: Clone + Debug + Send + Sync,
{
	/// Creates a new L-BFGS state.
	pub fn new(memory_size: usize) -> Self {
		Self {
			history: VecDeque::with_capacity(memory_size),
			memory_size,
			use_cautious_updates: true,
		}
	}

	/// Adds a new entry to the history.
	pub fn add_entry(&mut self, s: TV, y: TV, rho: T, iteration: usize) {
		if self.history.len() >= self.memory_size {
			self.history.pop_front();
		}
		self.history.push_back(LBFGSHistoryEntry {
			s,
			y,
			rho,
			_iteration: iteration,
		});
	}

	/// Clears the history.
	pub fn clear(&mut self) {
		self.history.clear();
	}
}

/// Configuration for the L-BFGS optimizer.
#[derive(Debug, Clone)]
pub struct LBFGSConfig<T: Scalar> {
	/// Number of vector pairs to store (typically 5-20)
	pub memory_size: usize,
	/// Initial step size for line search
	pub initial_step_size: T,
	/// Whether to use cautious updates (skip updates that don't satisfy positive definiteness)
	pub use_cautious_updates: bool,
	/// Line search parameters
	pub line_search_params: LineSearchParams<T>,
}

impl<T: Scalar> Default for LBFGSConfig<T> {
	fn default() -> Self {
		Self {
			memory_size: 30,
			initial_step_size: <T as Scalar>::from_f64(1.0),
			use_cautious_updates: true,
			line_search_params: LineSearchParams::default(),
		}
	}
}

impl<T: Scalar> LBFGSConfig<T> {
	/// Creates a new configuration with default parameters.
	pub fn new() -> Self {
		Self::default()
	}

	/// Sets the memory size (number of vector pairs to store).
	pub fn with_memory_size(mut self, size: usize) -> Self {
		self.memory_size = size;
		self
	}

	/// Sets the initial step size for line search.
	pub fn with_initial_step_size(mut self, step_size: T) -> Self {
		self.initial_step_size = step_size;
		self
	}

	/// Enables or disables cautious updates.
	pub fn with_cautious_updates(mut self, cautious: bool) -> Self {
		self.use_cautious_updates = cautious;
		self
	}

	/// Sets custom line search parameters.
	pub fn with_line_search_params(mut self, params: LineSearchParams<T>) -> Self {
		self.line_search_params = params;
		self
	}
}

/// Riemannian L-BFGS optimizer.
///
/// This optimizer adapts the classical L-BFGS algorithm to Riemannian manifolds
/// by eagerly transporting all stored vector pairs to the current tangent space
/// after each iteration.
///
/// # Examples
///
/// ```rust,ignore
/// use riemannopt_optim::{LBFGS, LBFGSConfig};
///
/// // Basic L-BFGS with default parameters
/// let lbfgs: LBFGS<f64> = LBFGS::new(LBFGSConfig::new());
///
/// // L-BFGS with custom parameters
/// let lbfgs_custom = LBFGS::new(
///     LBFGSConfig::new()
///         .with_memory_size(20)
///         .with_initial_step_size(0.1)
///         .with_cautious_updates(true)
/// );
/// ```
#[derive(Debug)]
pub struct LBFGS<T: Scalar> {
	config: LBFGSConfig<T>,
}

impl<T: Scalar> LBFGS<T> {
	/// Creates a new L-BFGS optimizer with given configuration.
	pub fn new(config: LBFGSConfig<T>) -> Self {
		Self { config }
	}

	/// Creates a new L-BFGS optimizer with default configuration.
	pub fn with_default_config() -> Self {
		Self::new(LBFGSConfig::default())
	}

	/// Returns the optimizer configuration.
	pub fn config(&self) -> &LBFGSConfig<T> {
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
		previous_point: Option<&M::Point>,
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

	/// Computes the L-BFGS search direction using the two-loop recursion.
	///
	/// All stored (s_i, y_i) vectors are already in the current tangent space,
	/// so no transport is needed during the recursion.
	/// Computes the L-BFGS direction using the two-loop recursion.
	///
	/// Uses `scratch0` and `scratch1` as temporary buffers and `alpha_buf`
	/// as pre-allocated storage for the alpha coefficients.
	/// Zero heap allocations — all work happens in pre-allocated buffers.
	fn compute_lbfgs_direction<M>(
		&self,
		manifold: &M,
		current_point: &M::Point,
		gradient: &M::TangentVector,
		lbfgs_state: &LBFGSState<T, M::TangentVector>,
		direction: &mut M::TangentVector,
		scratch0: &mut M::TangentVector,
		scratch1: &mut M::TangentVector,
		alpha_buf: &mut Vec<T>,
		manifold_ws: &mut M::Workspace,
	) -> Result<()>
	where
		M: Manifold<T>,
	{
		let m = lbfgs_state.history.len();

		if m == 0 {
			manifold.scale_tangent(current_point, -T::one(), gradient, direction)?;
			return Ok(());
		}

		// Reuse pre-allocated alpha buffer (no heap alloc if capacity suffices)
		alpha_buf.clear();
		alpha_buf.resize(m, T::zero());

		// q lives in `direction` throughout; scratch0/scratch1 are temporaries.
		direction.clone_from(gradient);

		// First loop (backward): q ← q - α_i · y_i
		for i in (0..m).rev() {
			let entry = &lbfgs_state.history[i];
			let s_dot_q =
				manifold.inner_product(current_point, &entry.s, direction, manifold_ws)?;
			alpha_buf[i] = entry.rho * s_dot_q;

			// scratch0 = -α_i · y_i
			manifold.scale_tangent(current_point, -alpha_buf[i], &entry.y, scratch0)?;
			// scratch1 = q + scratch0
			manifold.add_tangents(current_point, &*direction, scratch0, scratch1)?;
			// q ← scratch1 (in-place copy, no alloc)
			direction.clone_from(scratch1);
		}

		// r = γ · q  →  scratch0
		{
			let last_entry = &lbfgs_state.history[m - 1];
			let s_dot_y =
				manifold.inner_product(current_point, &last_entry.s, &last_entry.y, manifold_ws)?;
			let y_dot_y =
				manifold.inner_product(current_point, &last_entry.y, &last_entry.y, manifold_ws)?;

			if y_dot_y > T::zero() {
				let gamma = s_dot_y / y_dot_y;
				manifold.scale_tangent(current_point, gamma, direction, scratch0)?;
			} else {
				scratch0.clone_from(direction);
			}
		}

		// Second loop (forward): r ← r + (α_i - β) · s_i
		// r lives in scratch0; direction and scratch1 are temporaries.
		for i in 0..m {
			let entry = &lbfgs_state.history[i];
			let y_dot_r = manifold.inner_product(current_point, &entry.y, scratch0, manifold_ws)?;
			let beta = entry.rho * y_dot_r;
			let coeff = alpha_buf[i] - beta;

			// scratch1 = coeff · s_i
			manifold.scale_tangent(current_point, coeff, &entry.s, scratch1)?;
			// direction = r + scratch1
			manifold.add_tangents(current_point, &*scratch0, scratch1, direction)?;
			// r ← direction (in-place copy, no alloc)
			scratch0.clone_from(direction);
		}

		// direction = -r, then project for numerical safety
		manifold.scale_tangent(current_point, -T::one(), scratch0, scratch1)?;
		manifold.project_tangent(current_point, scratch1, direction, manifold_ws)?;

		Ok(())
	}

	/// Performs Strong Wolfe line search.
	///
	/// L-BFGS requires Wolfe conditions for good Hessian approximation quality,
	/// especially on curved manifolds.
	/// Performs Strong Wolfe line search using pre-allocated buffers.
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
		// Pre-allocated line search buffers
		ls_scaled_dir: &mut M::TangentVector,
		ls_trial_point: &mut M::Point,
		ls_trial_grad: &mut M::TangentVector,
		ls_trial_riem_grad: &mut M::TangentVector,
		ls_transported_dir: &mut M::TangentVector,
		manifold_ws: &mut M::Workspace,
	) -> Result<T>
	where
		C: CostFunction<T, Point = M::Point, TangentVector = M::TangentVector>,
		M: Manifold<T>,
	{
		let directional_derivative =
			manifold.inner_product(point, gradient, direction, manifold_ws)?;

		let c1 = self.config.line_search_params.c1;
		let c2 = self.config.line_search_params.c2;
		let max_iterations = self.config.line_search_params.max_iterations;

		let mut alpha = self.config.initial_step_size;
		let alpha_min = <T as Scalar>::from_f64(1e-10);
		let alpha_max = <T as Scalar>::from_f64(10.0);

		for _iter in 0..max_iterations {
			manifold.scale_tangent(point, alpha, direction, ls_scaled_dir)?;
			manifold.retract(point, ls_scaled_dir, ls_trial_point, manifold_ws)?;

			let trial_cost = cost_fn.cost(ls_trial_point)?;
			*function_evaluations += 1;

			let expected_decrease = c1 * alpha * directional_derivative;
			if trial_cost <= current_cost + expected_decrease {
				// Armijo satisfied — check curvature condition
				let _trial_gradient_cost =
					cost_fn.cost_and_gradient(ls_trial_point, ls_trial_grad)?;
				*function_evaluations += 1;
				*gradient_evaluations += 1;

				manifold.euclidean_to_riemannian_gradient(
					ls_trial_point,
					ls_trial_grad,
					ls_trial_riem_grad,
					manifold_ws,
				)?;

				manifold.parallel_transport(
					point,
					ls_trial_point,
					direction,
					ls_transported_dir,
					manifold_ws,
				)?;

				let trial_directional = manifold.inner_product(
					ls_trial_point,
					ls_trial_riem_grad,
					ls_transported_dir,
					manifold_ws,
				)?;

				if <T as Float>::abs(trial_directional)
					<= c2 * <T as Float>::abs(directional_derivative)
				{
					return Ok(alpha);
				}

				if trial_directional < directional_derivative {
					alpha = <T as Float>::min(alpha * <T as Scalar>::from_f64(2.0), alpha_max);
				} else {
					alpha *= <T as Scalar>::from_f64(0.5);
				}
			} else {
				alpha *= <T as Scalar>::from_f64(0.5);
			}

			if alpha < alpha_min {
				break;
			}
		}

		Ok(alpha)
	}

	/// Updates the L-BFGS history with new information.
	///
	/// After computing the new (s_k, y_k) pair in the new tangent space,
	/// transports ALL previously stored vectors to the new tangent space.
	/// This ensures all stored vectors are always in the same (current) tangent space.
	/// Updates the L-BFGS history with new information.
	///
	/// Uses `scratch0` and `scratch1` as temporary buffers.
	/// The only unavoidable allocations are the 2 vectors that enter the history
	/// (the history must own them).
	fn update_lbfgs_history<M>(
		&self,
		manifold: &M,
		old_point: &M::Point,
		new_point: &M::Point,
		old_gradient: &M::TangentVector,
		new_gradient: &M::TangentVector,
		direction: &M::TangentVector,
		step_size: T,
		grad_norm: T,
		lbfgs_state: &mut LBFGSState<T, M::TangentVector>,
		iteration: usize,
		scratch0: &mut M::TangentVector,
		scratch1: &mut M::TangentVector,
		manifold_ws: &mut M::Workspace,
	) -> Result<bool>
	where
		M: Manifold<T>,
	{
		// scratch0 = s_k = step_size · direction
		manifold.scale_tangent(old_point, step_size, direction, scratch0)?;

		// scratch1 = τ(s_k) = transported s_k to new tangent space
		manifold.parallel_transport(old_point, new_point, scratch0, scratch1, manifold_ws)?;
		// Save transported_s_k for later — we'll need it after the history transport loop.
		// Use scratch0 for it (s_k is no longer needed at old_point).
		std::mem::swap(scratch0, scratch1);
		// Now: scratch0 = transported_s_k, scratch1 = s_k (old, will be reused)

		// scratch1 = τ(old_gradient) = transported old gradient
		manifold.parallel_transport(old_point, new_point, old_gradient, scratch1, manifold_ws)?;

		// y_k = new_gradient - τ(old_gradient)
		// scratch1 = -τ(old_gradient) then y_k = new_gradient + scratch1
		// But we need a place for y_k. Use `direction` idea... no, direction is &.
		// We need to build y_k somewhere. Negate scratch1 in-place:
		// scratch1 currently = τ(old_grad). Scale to -τ(old_grad):
		// Need a temp to scale... use the transported_s_k (scratch0) as temp? No, we need it.
		// Actually: we have add_tangents which computes result = v1 + v2.
		// We want y_k = new_gradient - scratch1. If we negate scratch1 first:
		// Can't negate in-place (scale_tangent needs different input/output).
		// Use a clone_from trick: save scratch1, negate into itself via a copy.
		// Actually simplest: compute y_k into a local. This IS the entry that goes into history,
		// so it needs to be owned anyway.
		// y_k = new_gradient - τ(old_grad)
		// Use y_k as temp: negate scratch1 into y_k, then add_tangents into scratch1,
		// then move scratch1 → y_k. This avoids cloning scratch1.
		let mut y_k = new_gradient.clone(); // Will be moved into history
		manifold.scale_tangent(new_point, -T::one(), scratch1, &mut y_k)?;
		// y_k = -τ(old_grad), scratch1 = τ(old_grad)
		// Now: y_k_final = new_gradient + y_k = new_gradient - τ(old_grad)
		manifold.add_tangents(new_point, new_gradient, &y_k, scratch1)?;
		std::mem::swap(&mut y_k, scratch1);
		// y_k = new_gradient - τ(old_grad), scratch1 is free

		// Metrics on transported_s_k (scratch0) and y_k
		let s_k_norm_sq = manifold.inner_product(new_point, scratch0, scratch0, manifold_ws)?;
		let s_k_norm = <T as Float>::sqrt(s_k_norm_sq);

		if s_k_norm < <T as Scalar>::from_f64(1e-15) {
			return Ok(false);
		}

		let s_dot_y = manifold.inner_product(new_point, &y_k, scratch0, manifold_ws)?;

		let pair_accepted = if self.config.use_cautious_updates {
			let cautious_threshold = <T as Scalar>::from_f64(1e-6) * grad_norm;
			let curvature_ratio = s_dot_y / s_k_norm_sq;
			curvature_ratio >= cautious_threshold
		} else {
			let min_curvature = <T as Scalar>::from_f64(1e-8);
			<T as Float>::abs(s_dot_y) > min_curvature && s_dot_y > T::zero()
		};

		// Transport ALL history entries using swap pattern.
		// scratch1 is free; use it and y_k-temp area for swap buffers.
		// But y_k is needed later if pair_accepted. Use scratch1 for s-transport
		// and a dedicated buf for y-transport. We can reuse scratch0 AFTER saving
		// transported_s_k into a temporary owned vector (which becomes the entry).
		// Actually: transported_s_k (scratch0) will be moved into history.
		// We need it after the loop. So we can't use scratch0 as transport buf.
		// Use scratch1 for both s and y transport (sequentially):
		for entry in lbfgs_state.history.iter_mut() {
			// Transport s
			if manifold
				.parallel_transport(old_point, new_point, &entry.s, scratch1, manifold_ws)
				.is_err()
			{
				lbfgs_state.clear();
				return Ok(false);
			}
			std::mem::swap(&mut entry.s, scratch1);

			// Transport y (reuse scratch1 which now holds old entry.s, overwritten)
			if manifold
				.parallel_transport(old_point, new_point, &entry.y, scratch1, manifold_ws)
				.is_err()
			{
				lbfgs_state.clear();
				return Ok(false);
			}
			std::mem::swap(&mut entry.y, scratch1);

			let new_s_dot_y = manifold.inner_product(new_point, &entry.y, &entry.s, manifold_ws)?;
			if <T as Float>::abs(new_s_dot_y) < <T as Scalar>::from_f64(1e-15) {
				lbfgs_state.clear();
				return Ok(false);
			}
			entry.rho = T::one() / new_s_dot_y;
		}

		if !pair_accepted {
			return Ok(false);
		}

		// Normalize and add to history. scratch0 = transported_s_k.
		// Scale in-place: scratch1 = inv_norm · scratch0, then move scratch1 into entry.
		let inv_s_norm = T::one() / s_k_norm;
		manifold.scale_tangent(new_point, inv_s_norm, scratch0, scratch1)?;
		// Normalized s is in scratch1 — clone for history ownership
		let normalized_s = scratch1.clone();
		// Normalize y_k → scratch0, then clone for history ownership
		manifold.scale_tangent(new_point, inv_s_norm, &y_k, scratch0)?;
		let normalized_y = scratch0.clone();

		let normalized_s_dot_y =
			manifold.inner_product(new_point, &normalized_y, &normalized_s, manifold_ws)?;

		if <T as Float>::abs(normalized_s_dot_y) < <T as Scalar>::from_f64(1e-15) {
			return Ok(false);
		}

		let rho = T::one() / normalized_s_dot_y;
		lbfgs_state.add_entry(normalized_s, normalized_y, rho, iteration);

		Ok(true)
	}
}

// Implementation of the Optimizer trait from core
impl<T: Scalar> Optimizer<T> for LBFGS<T> {
	fn name(&self) -> &str {
		"Riemannian L-BFGS"
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

		let initial_cost = cost_fn.cost(initial_point)?;
		let mut lbfgs_state = LBFGSState::new(self.config.memory_size);
		lbfgs_state.use_cautious_updates = self.config.use_cautious_updates;

		let mut current_point = initial_point.clone();
		let mut has_previous_point = false;
		let mut current_cost = initial_cost;
		let mut previous_cost: Option<T> = None;
		let mut gradient_norm: Option<T> = None;
		let mut iteration = 0;
		let mut function_evaluations = 1;
		let mut gradient_evaluations = 0;

		let minstepsize = <T as Scalar>::from_f64(1e-10);
		let mut ultimatum = false;

		let initial_grad = cost_fn.gradient(initial_point)?;
		gradient_evaluations += 1;

		let mut ws = CommonWorkspace::new(initial_point, &initial_grad);
		let mut manifold_ws = manifold.create_workspace(initial_point);
		ws.euclidean_grad.clone_from(&initial_grad);

		// Extra buffers for L-BFGS: gradient at new point + alpha coefficients
		let mut new_euclidean_grad = initial_grad.clone();
		let mut new_riemannian_grad = initial_grad.clone();
		let mut alpha_buf: Vec<T> = Vec::with_capacity(self.config.memory_size);

		loop {
			let prev_ref = if has_previous_point {
				Some(&ws.previous_point)
			} else {
				None
			};
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
				prev_ref,
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

			let _new_cost = cost_fn.cost_and_gradient(&current_point, &mut ws.euclidean_grad)?;
			function_evaluations += 1;
			gradient_evaluations += 1;

			manifold.euclidean_to_riemannian_gradient(
				&current_point,
				&ws.euclidean_grad,
				&mut ws.riemannian_grad,
				&mut manifold_ws,
			)?;

			let grad_norm_squared = manifold.inner_product(
				&current_point,
				&ws.riemannian_grad,
				&ws.riemannian_grad,
				&mut manifold_ws,
			)?;
			let grad_norm = <T as Float>::sqrt(grad_norm_squared);
			gradient_norm = Some(grad_norm);

			// Compute L-BFGS direction (zero alloc via scratch + alpha_buf)
			{
				let [ref mut s0, ref mut s1, ..] = ws.scratch;
				self.compute_lbfgs_direction(
					manifold,
					&current_point,
					&ws.riemannian_grad,
					&lbfgs_state,
					&mut ws.direction,
					s0,
					s1,
					&mut alpha_buf,
					&mut manifold_ws,
				)?;
			}

			// Line search (zero alloc via workspace buffers)
			let step_size = self.perform_line_search(
				cost_fn,
				manifold,
				&current_point,
				&ws.direction,
				current_cost,
				&ws.riemannian_grad,
				&mut function_evaluations,
				&mut gradient_evaluations,
				&mut ws.scaled_direction,
				&mut ws.ls_trial_point,
				&mut ws.ls_trial_grad,
				&mut ws.ls_trial_riem_grad,
				&mut ws.ls_transported_dir,
				&mut manifold_ws,
			)?;

			if step_size < minstepsize {
				if !ultimatum {
					lbfgs_state.clear();
					ultimatum = true;
					continue;
				} else {
					return Ok(OptimizationResult::new(
						current_point,
						current_cost,
						iteration,
						start_time.elapsed(),
						TerminationReason::Converged,
					)
					.with_function_evaluations(function_evaluations)
					.with_gradient_evaluations(gradient_evaluations)
					.with_gradient_norm(gradient_norm.unwrap_or(T::zero())));
				}
			} else {
				ultimatum = false;
			}

			// Scale direction and retract (in-place)
			manifold.scale_tangent(
				&current_point,
				step_size,
				&ws.direction,
				&mut ws.scaled_direction,
			)?;
			manifold.retract(
				&current_point,
				&ws.scaled_direction,
				&mut ws.new_point,
				&mut manifold_ws,
			)?;

			// Gradient at new point (reuse buffers)
			let new_cost_verify =
				cost_fn.cost_and_gradient(&ws.new_point, &mut new_euclidean_grad)?;
			function_evaluations += 1;
			gradient_evaluations += 1;

			manifold.euclidean_to_riemannian_gradient(
				&ws.new_point,
				&new_euclidean_grad,
				&mut new_riemannian_grad,
				&mut manifold_ws,
			)?;

			// Update history (uses scratch[0..1] as transport buffers)
			{
				let [ref mut s0, ref mut s1, ..] = ws.scratch;
				let _updated = self.update_lbfgs_history(
					manifold,
					&current_point,
					&ws.new_point,
					&ws.riemannian_grad,
					&new_riemannian_grad,
					&ws.direction,
					step_size,
					grad_norm,
					&mut lbfgs_state,
					iteration,
					s0,
					s1,
					&mut manifold_ws,
				)?;
			}

			// Swap points (zero alloc)
			std::mem::swap(&mut ws.previous_point, &mut current_point);
			has_previous_point = true;
			std::mem::swap(&mut current_point, &mut ws.new_point);

			previous_cost = Some(current_cost);
			current_cost = new_cost_verify;
			iteration += 1;
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use riemannopt_core::linalg;

	#[test]
	fn test_lbfgs_config() {
		let config = LBFGSConfig::<f64>::new()
			.with_memory_size(20)
			.with_initial_step_size(0.5)
			.with_cautious_updates(false);

		assert_eq!(config.memory_size, 20);
		assert_eq!(config.initial_step_size, 0.5);
		assert!(!config.use_cautious_updates);
	}

	#[test]
	fn test_lbfgs_state() {
		let state = LBFGSState::<f64, linalg::Vec<f64>>::new(5);
		assert_eq!(state.memory_size, 5);
		assert_eq!(state.history.len(), 0);
	}
}
