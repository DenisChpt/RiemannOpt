//! Limited-memory BFGS preconditioner for Riemannian solvers.
//!
//! Maintains a circular buffer of `m` curvature pairs (sᵢ, yᵢ) and applies
//! the standard two-loop recursion to approximate H⁻¹ v without ever forming
//! the dense inverse Hessian.
//!
//! # Memory
//!
//! O(m × dim) where `m` is the memory depth and `dim` is the manifold
//! dimension. All tangent vectors are pre-allocated once in the workspace;
//! the hot path (`apply`, `update`) performs **zero heap allocation**.
//!
//! # Riemannian Transport
//!
//! After each accepted step x_old → x_new, all stored pairs are
//! parallel-transported from T_{x_old} ℳ to T_{x_new} ℳ so that the
//! two-loop recursion operates in a single tangent space. This is O(m)
//! transport calls per accepted step — amortised against the O(dim)
//! Hessian-vector products saved by better conditioning.

use std::fmt::Debug;

use crate::{manifold::Manifold, types::Scalar};

use super::Preconditioner;

// ════════════════════════════════════════════════════════════════════════════
//  Configuration
// ════════════════════════════════════════════════════════════════════════════

/// Limited-memory BFGS preconditioner.
///
/// Stores only configuration; all mutable state lives in
/// [`LbfgsWorkspace`], allocated once by the solver.
///
/// # Example
///
/// ```ignore
/// let pre = LbfgsPreconditioner::new(10);  // 10 curvature pairs
/// let solver = TrustRegion::new(config).with_preconditioner(pre);
/// ```
#[derive(Debug, Clone)]
pub struct LbfgsPreconditioner {
	/// Maximum number of stored (s, y) curvature pairs.
	memory: usize,
	/// Initial inverse Hessian scaling strategy.
	/// When `None`, uses the Shanno–Phua auto-scaling:
	///   γ = ⟨s_latest, y_latest⟩ / ⟨y_latest, y_latest⟩
	/// When `Some(γ)`, uses the fixed value γ for H₀ = γ I.
	initial_scale: Option<f64>,
}

impl LbfgsPreconditioner {
	/// Creates a new L-BFGS preconditioner with the given memory depth.
	///
	/// A memory of 5–20 is typical. Larger values give a better Hessian
	/// approximation at the cost of O(m × dim) storage and O(m) inner
	/// products per `apply`.
	#[inline]
	pub fn new(memory: usize) -> Self {
		assert!(memory > 0, "L-BFGS memory must be at least 1");
		Self {
			memory,
			initial_scale: None,
		}
	}

	/// Overrides the automatic Shanno–Phua scaling with a fixed γ.
	///
	/// H₀ = γ I is used as the initial inverse Hessian in the two-loop
	/// recursion. Set to `1.0` for the identity (no initial scaling).
	#[inline]
	pub fn with_initial_scale(mut self, gamma: f64) -> Self {
		self.initial_scale = Some(gamma);
		self
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Workspace (all mutable state + scratch buffers)
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated workspace for [`LbfgsPreconditioner`].
///
/// Contains the curvature pair ring buffer and scratch buffers for the
/// two-loop recursion. Allocated once; zero heap allocation thereafter.
pub struct LbfgsWorkspace<TV, T> {
	// ── Ring buffer of curvature pairs ───────────────────────────────
	/// Step vectors sᵢ = T_{x_old→x_new}(step).
	s_buf: Vec<TV>,
	/// Gradient difference vectors yᵢ = grad_new − T_{x_old→x_new}(grad_old).
	y_buf: Vec<TV>,
	/// Cached ρᵢ = 1 / ⟨yᵢ, sᵢ⟩.
	rho: Vec<T>,

	// ── Ring buffer bookkeeping ──────────────────────────────────────
	/// Index of the *next* write position (newest = head − 1 mod cap).
	head: usize,
	/// Number of valid pairs currently stored (≤ capacity).
	len: usize,

	// ── Scratch for two-loop recursion (no allocation in apply) ──────
	/// α coefficients (one per stored pair).
	alpha: Vec<T>,
	/// Scratch tangent vector for the running `q` in the recursion.
	q: TV,

	// ── Scratch for update (parallel transport) ──────────────────────
	/// Temporary buffer for transporting vectors.
	transport_tmp: TV,

	/// Current H₀ scaling factor γ.
	gamma: T,
}

/// # Safety
///
/// `Send`/`Sync` are safe because `TV` (= `M::TangentVector`) is required
/// to be `Send + Sync` by the `Manifold` trait, and `T: Scalar` is `Send + Sync`.
unsafe impl<TV: Send, T: Send> Send for LbfgsWorkspace<TV, T> {}
unsafe impl<TV: Sync, T: Sync> Sync for LbfgsWorkspace<TV, T> {}

// `Default` is required by the trait bound but never actually used —
// the solver always calls `create_workspace` which produces a properly
// sized instance. This impl creates a dummy that will be overwritten.
impl<TV, T: Default> Default for LbfgsWorkspace<TV, T> {
	fn default() -> Self {
		Self {
			s_buf: Vec::new(),
			y_buf: Vec::new(),
			rho: Vec::new(),
			head: 0,
			len: 0,
			alpha: Vec::new(),
			// SAFETY: these vecs are empty so q / transport_tmp are never
			// accessed before create_workspace replaces this instance.
			q: unsafe { std::mem::zeroed() },
			transport_tmp: unsafe { std::mem::zeroed() },
			gamma: T::default(),
		}
	}
}

impl<TV, T: Copy> LbfgsWorkspace<TV, T> {
	/// Returns the ring buffer capacity (= preconditioner memory).
	#[inline]
	fn capacity(&self) -> usize {
		self.s_buf.len()
	}

	/// Maps a logical index `0..len` (0 = oldest) to a physical ring index.
	#[inline]
	fn ring_index(&self, logical: usize) -> usize {
		let cap = self.capacity();
		// oldest is at (head - len) mod cap
		(self.head + cap - self.len + logical) % cap
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Preconditioner impl
// ════════════════════════════════════════════════════════════════════════════

impl<T: Scalar, M: Manifold<T>> Preconditioner<T, M> for LbfgsPreconditioner {
	type Workspace = LbfgsWorkspace<M::TangentVector, T>;

	fn create_workspace(&self, manifold: &M, _proto_point: &M::Point) -> Self::Workspace {
		let m = self.memory;
		LbfgsWorkspace {
			s_buf: (0..m).map(|_| manifold.allocate_tangent()).collect(),
			y_buf: (0..m).map(|_| manifold.allocate_tangent()).collect(),
			rho: vec![T::zero(); m],
			head: 0,
			len: 0,
			alpha: vec![T::zero(); m],
			q: manifold.allocate_tangent(),
			transport_tmp: manifold.allocate_tangent(),
			gamma: T::one(),
		}
	}

	/// Applies the L-BFGS two-loop recursion: result ← H⁻¹ v.
	///
	/// When no curvature pairs are stored yet (cold start), falls back to
	/// result ← γ v (scaled identity).
	///
	/// **Zero allocation.** All work uses pre-allocated workspace buffers.
	fn apply(
		&self,
		manifold: &M,
		point: &M::Point,
		v: &M::TangentVector,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		man_ws: &mut M::Workspace,
	) {
		let m = ws.len;

		// ── Cold start: H₀ = γ I ────────────────────────────────────
		if m == 0 {
			manifold.copy_tangent(result, v);
			manifold.scale_tangent(ws.gamma, result);
			return;
		}

		// ── First loop: q ← v, then walk newest → oldest ────────────
		// q lives in ws.q to avoid allocation.
		manifold.copy_tangent(&mut ws.q, v);

		for i in (0..m).rev() {
			let idx = ws.ring_index(i);
			// αᵢ = ρᵢ ⟨sᵢ, q⟩
			let a = ws.rho[idx]
				* manifold.inner_product(point, &ws.s_buf[idx], &ws.q, man_ws);
			ws.alpha[idx] = a;
			// q ← q − αᵢ yᵢ
			manifold.axpy_tangent(-a, &ws.y_buf[idx], &mut ws.q);
		}

		// ── Initial scaling: result ← γ q ────────────────────────────
		manifold.copy_tangent(result, &ws.q);
		manifold.scale_tangent(ws.gamma, result);

		// ── Second loop: walk oldest → newest ────────────────────────
		for i in 0..m {
			let idx = ws.ring_index(i);
			// β = ρᵢ ⟨yᵢ, result⟩
			let beta = ws.rho[idx]
				* manifold.inner_product(point, &ws.y_buf[idx], result, man_ws);
			// result ← result + (αᵢ − β) sᵢ
			manifold.axpy_tangent(ws.alpha[idx] - beta, &ws.s_buf[idx], result);
		}
	}

	/// Stores a new curvature pair and transports all existing pairs.
	///
	/// **Transport cost**: O(len) parallel transport calls.
	/// **Zero allocation**: all tangent vectors are pre-allocated in workspace.
	fn update(
		&mut self,
		manifold: &M,
		old_point: &M::Point,
		new_point: &M::Point,
		step: &M::TangentVector,
		old_gradient: &M::TangentVector,
		new_gradient: &M::TangentVector,
		ws: &mut Self::Workspace,
		man_ws: &mut M::Workspace,
	) {
		// ── 1. Transport existing pairs to T_{x_new} ℳ ──────────────
		for i in 0..ws.len {
			let idx = ws.ring_index(i);

			// s_i ← T_{old→new}(s_i)
			manifold.parallel_transport(
				old_point,
				new_point,
				&ws.s_buf[idx],
				&mut ws.transport_tmp,
				man_ws,
			);
			manifold.copy_tangent(&mut ws.s_buf[idx], &ws.transport_tmp);

			// y_i ← T_{old→new}(y_i)
			manifold.parallel_transport(
				old_point,
				new_point,
				&ws.y_buf[idx],
				&mut ws.transport_tmp,
				man_ws,
			);
			manifold.copy_tangent(&mut ws.y_buf[idx], &ws.transport_tmp);
		}

		// ── 2. Compute new pair at x_new ─────────────────────────────
		let slot = ws.head;

		// s_new = T_{old→new}(step)
		manifold.parallel_transport(
			old_point,
			new_point,
			step,
			&mut ws.s_buf[slot],
			man_ws,
		);

		// y_new = grad_new − T_{old→new}(grad_old)
		manifold.parallel_transport(
			old_point,
			new_point,
			old_gradient,
			&mut ws.transport_tmp,
			man_ws,
		);
		manifold.copy_tangent(&mut ws.y_buf[slot], new_gradient);
		manifold.axpy_tangent(
			-T::one(),
			&ws.transport_tmp,
			&mut ws.y_buf[slot],
		);

		// ρ = 1 / ⟨y, s⟩ (with safeguard against near-zero denominator)
		let sy = manifold.inner_product(
			new_point,
			&ws.y_buf[slot],
			&ws.s_buf[slot],
			man_ws,
		);

		let threshold = T::epsilon() * <T as Scalar>::from_f64(1e6);
		if sy > threshold {
			ws.rho[slot] = T::one() / sy;

			// ── 3. Auto-scale γ (Shanno–Phua) ────────────────────────
			match self.initial_scale {
				Some(fixed) => {
					ws.gamma = <T as Scalar>::from_f64(fixed);
				}
				None => {
					let yy = manifold.inner_product(
						new_point,
						&ws.y_buf[slot],
						&ws.y_buf[slot],
						man_ws,
					);
					if yy > T::zero() {
						ws.gamma = sy / yy;
					}
				}
			}

			// ── 4. Advance ring buffer ───────────────────────────────
			ws.head = (ws.head + 1) % ws.capacity();
			if ws.len < ws.capacity() {
				ws.len += 1;
			}
		}
		// If sy ≤ threshold, we skip this pair (it would produce a
		// non-positive-definite update). The ring buffer is unchanged.
	}

	fn reset(&mut self) {
		// We can't touch the workspace from here (no access to ws),
		// but the solver is expected to call create_workspace again
		// or manually clear the workspace. The trait's reset semantics
		// are: "the next apply should behave as if no update had occurred".
		// Solvers should zero ws.len after calling reset.
		//
		// A more robust approach: solvers call ws.len = 0 directly,
		// or we add a `reset_workspace` method. For now, this is a
		// signal to the solver.
	}

	#[inline]
	fn is_identity(&self) -> bool {
		false
	}
}

impl LbfgsWorkspace<(), ()> {
	// This impl block exists purely to anchor doc-links.
}

/// Resets the workspace state, discarding all stored curvature pairs.
///
/// This is a free function because [`Preconditioner::reset`] does not
/// receive a workspace reference. Solvers should call this after
/// `preconditioner.reset()` or directly when they need a fresh start.
#[inline]
pub fn reset_lbfgs_workspace<TV, T: Scalar>(ws: &mut LbfgsWorkspace<TV, T>) {
	ws.head = 0;
	ws.len = 0;
	ws.gamma = T::one();
}