//! Pre-allocated workspace buffers for zero-allocation optimization loops.
//!
//! # Overview
//!
//! The [`CommonWorkspace`] provides reusable buffers for the gradient pipeline,
//! point management, line search, and general scratch space. It is generic over
//! the manifold's `Point` and `TangentVector` types and is instantiated once at
//! the start of [`Optimizer::optimize()`] by cloning prototype values.
//!
//! Per-optimizer workspaces (defined in `riemannopt-optim`) embed a
//! `CommonWorkspace` and add algorithm-specific buffers (momentum, moments,
//! LBFGS history transport, CG solver state, …).
//!
//! # Design Rationale
//!
//! - **Lives in `riemannopt-core`** so that manifold methods can later accept
//!   workspace scratch buffers without introducing circular dependencies.
//! - **Invisible to the user** — created and consumed entirely within
//!   `optimize()`.
//! - **Fixed-size scratch array** (`[TV; 4]`) covers the deepest temporary
//!   chain (LBFGS two-loop recursion needs ≤ 3 simultaneous temporaries).

use std::fmt::Debug;

/// Pre-allocated buffers shared by all first-order (and most second-order)
/// Riemannian optimizers.
///
/// All buffers are initialised once via [`CommonWorkspace::new`] and reused
/// across iterations through `copy_from` and `std::mem::swap`.
pub struct CommonWorkspace<P, TV> {
	// ── Gradient pipeline ───────────────────────────────────────────────
	/// Buffer for the Euclidean gradient returned by `CostFunction::cost_and_gradient`.
	pub euclidean_grad: TV,
	/// Buffer for the Riemannian gradient after `euclidean_to_riemannian_gradient`.
	pub riemannian_grad: TV,

	// ── Direction pipeline ──────────────────────────────────────────────
	/// Optimizer-computed descent direction (e.g. −∇f, conjugate direction, LBFGS direction).
	pub direction: TV,
	/// `direction` scaled by the step size, ready for retraction.
	pub scaled_direction: TV,

	// ── Point buffers ───────────────────────────────────────────────────
	/// Result of `manifold.retract(current_point, scaled_direction, &mut new_point)`.
	pub new_point: P,
	/// Previous iterate, used for parallel transport.  Populated via `std::mem::swap`.
	pub previous_point: P,

	// ── Scratch tangent vectors ─────────────────────────────────────────
	/// General-purpose scratch buffers for `add_tangents`, `scale_tangent` chains,
	/// and any other operation that needs an anonymous temporary.
	///
	/// Four slots are sufficient for the deepest nesting in the codebase
	/// (LBFGS two-loop recursion).
	pub scratch: [TV; 4],

	// ── Line-search buffers ─────────────────────────────────────────────
	/// Trial point evaluated during line search.
	pub ls_trial_point: P,
	/// Euclidean gradient at the trial point.
	pub ls_trial_grad: TV,
	/// Riemannian gradient at the trial point.
	pub ls_trial_riem_grad: TV,
	/// Direction transported to the trial point (for curvature condition).
	pub ls_transported_dir: TV,
}

impl<P, TV> CommonWorkspace<P, TV>
where
	P: Clone + Debug + Send + Sync,
	TV: Clone + Debug + Send + Sync,
{
	/// Creates a workspace by cloning prototype values.
	///
	/// Call this once at the start of `optimize()` using the initial point
	/// and a tangent-vector prototype (e.g. the initial Euclidean gradient).
	///
	/// # Cost
	///
	/// Performs exactly **3 + 4 + 4 = 11 `Point` clones** and
	/// **2 + 4 + 4 + 4 = 14 `TangentVector` clones** — a one-time O(1) cost
	/// that replaces the O(iterations × buffers_per_iter) clones in the
	/// current code.
	pub fn new(proto_point: &P, proto_tangent: &TV) -> Self {
		Self {
			euclidean_grad: proto_tangent.clone(),
			riemannian_grad: proto_tangent.clone(),
			direction: proto_tangent.clone(),
			scaled_direction: proto_tangent.clone(),
			new_point: proto_point.clone(),
			previous_point: proto_point.clone(),
			scratch: [
				proto_tangent.clone(),
				proto_tangent.clone(),
				proto_tangent.clone(),
				proto_tangent.clone(),
			],
			ls_trial_point: proto_point.clone(),
			ls_trial_grad: proto_tangent.clone(),
			ls_trial_riem_grad: proto_tangent.clone(),
			ls_transported_dir: proto_tangent.clone(),
		}
	}
}

impl<P: Debug, TV: Debug> Debug for CommonWorkspace<P, TV> {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("CommonWorkspace")
			.field("scratch_slots", &self.scratch.len())
			.finish()
	}
}
