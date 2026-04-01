//! Jacobi (diagonal) preconditioner.
//!
//! Applies P⁻¹ v = v ⊘ d element-wise, where d is a vector of positive
//! diagonal scaling factors approximating diag(Hess f). This is the cheapest
//! non-trivial preconditioner:
//!
//! - **O(n) storage** — one tangent vector for the diagonal.
//! - **O(n) apply** — a single element-wise division (SIMD-friendly).
//! - **No transport** — the diagonal is recomputed or fixed, not transported.
//!
//! # When to use
//!
//! - **Product manifolds** where components have different scales.
//! - **Diagonally dominant Hessians** (SPD, PSD cone, least-squares).
//! - **Large-scale problems** where even L-BFGS memory (O(m × n)) is too much.
//!
//! # Diagonal source
//!
//! Two modes are supported:
//!
//! 1. **Fixed diagonal** — provided at construction, never changes.
//!    Use when the Hessian structure is known a priori (e.g. regularised
//!    least-squares where diag(H) ≈ diag(JᵀJ) + λI).
//!
//! 2. **Dynamic diagonal** — recomputed via a user callback after each
//!    accepted step. The callback receives the new point and writes the
//!    diagonal into a pre-allocated buffer. Zero allocation in the hot path.

use std::fmt::{self, Debug};

use crate::{manifold::Manifold, types::Scalar};

use super::Preconditioner;
use crate::linalg::AsElementWise;

// ════════════════════════════════════════════════════════════════════════════
//  Diagonal source
// ════════════════════════════════════════════════════════════════════════════

/// Strategy for obtaining the preconditioner diagonal.
enum DiagonalSource<T: Scalar, M: Manifold<T>> {
	/// Fixed diagonal, set once at construction. The `Option` is `Some`
	/// until `create_workspace` drains it into the workspace (one-shot move).
	Fixed(std::cell::UnsafeCell<Option<M::TangentVector>>),
	/// Recomputed after each accepted step via a user-provided closure.
	///
	/// Signature: `fn(manifold, point, diagonal_out, manifold_ws)`
	///
	/// The closure must write strictly positive values into `diagonal_out`.
	Dynamic(
		Box<
			dyn Fn(&M, &M::Point, &mut M::TangentVector, &mut M::Workspace)
				+ Send
				+ Sync,
		>,
	),
}

/// # Safety
///
/// `Fixed` wraps an `UnsafeCell` only to allow a one-shot drain in
/// `create_workspace` (called exactly once, single-threaded, before the
/// solver loop). The `UnsafeCell` is never accessed concurrently.
unsafe impl<T: Scalar, M: Manifold<T>> Send for DiagonalSource<T, M> {}
unsafe impl<T: Scalar, M: Manifold<T>> Sync for DiagonalSource<T, M> {}

impl<T: Scalar, M: Manifold<T>> Debug for DiagonalSource<T, M> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		match self {
			Self::Fixed(_) => write!(f, "Fixed"),
			Self::Dynamic(_) => write!(f, "Dynamic(<closure>)"),
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Preconditioner struct
// ════════════════════════════════════════════════════════════════════════════

/// Jacobi (diagonal) preconditioner.
///
/// Stores the diagonal source strategy. The actual diagonal vector lives
/// in [`JacobiWorkspace`] to support manifold-generic tangent vector types.
pub struct JacobiPreconditioner<T: Scalar, M: Manifold<T>> {
	source: DiagonalSource<T, M>,
	/// Floor value: diag entries below this are clamped to avoid division
	/// by near-zero. Default: `T::epsilon() * 1e6`.
	floor: T,
}

impl<T: Scalar, M: Manifold<T>> Debug for JacobiPreconditioner<T, M> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		f.debug_struct("JacobiPreconditioner")
			.field("source", &self.source)
			.field("floor", &self.floor)
			.finish()
	}
}

impl<T: Scalar, M: Manifold<T>> JacobiPreconditioner<T, M> {
	/// Creates a Jacobi preconditioner with a fixed diagonal.
	///
	/// `diagonal` is a tangent vector whose entries are the positive scaling
	/// factors d. The preconditioner applies P⁻¹ v = v ⊘ d element-wise.
	///
	/// The diagonal is moved into the workspace on the first (and only)
	/// call to `create_workspace`.
	pub fn fixed(diagonal: M::TangentVector) -> Self {
		Self {
			source: DiagonalSource::Fixed(std::cell::UnsafeCell::new(Some(diagonal))),
			floor: T::epsilon() * <T as Scalar>::from_f64(1e6),
		}
	}

	/// Creates a Jacobi preconditioner with a dynamic diagonal callback.
	///
	/// `compute_diagonal` is called after each accepted step (and once at
	/// initialisation) to fill the diagonal buffer with strictly positive
	/// scaling factors.
	///
	/// # Callback signature
	///
	/// ```ignore
	/// fn(manifold: &M, point: &M::Point, diag_out: &mut M::TangentVector, man_ws: &mut M::Workspace)
	/// ```
	pub fn dynamic<F>(compute_diagonal: F) -> Self
	where
		F: Fn(&M, &M::Point, &mut M::TangentVector, &mut M::Workspace) + Send + Sync + 'static,
	{
		Self {
			source: DiagonalSource::Dynamic(Box::new(compute_diagonal)),
			floor: T::epsilon() * <T as Scalar>::from_f64(1e6),
		}
	}

	/// Sets the minimum floor for diagonal entries.
	///
	/// Entries below this value are clamped to prevent division-by-zero
	/// or near-zero in `apply`. Default: `T::epsilon() * 1e6`.
	#[inline]
	pub fn with_floor(mut self, floor: T) -> Self {
		self.floor = floor;
		self
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Workspace
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated workspace for [`JacobiPreconditioner`].
///
/// Contains the diagonal vector and a scratch buffer. O(2n) memory total.
pub struct JacobiWorkspace<TV> {
	/// The diagonal scaling vector d (all entries > 0).
	diagonal: TV,
	/// Whether the diagonal has been initialised (either from fixed input
	/// or from the first dynamic callback call).
	initialised: bool,
}

unsafe impl<TV: Send> Send for JacobiWorkspace<TV> {}
unsafe impl<TV: Sync> Sync for JacobiWorkspace<TV> {}

impl<TV> Default for JacobiWorkspace<TV> {
	fn default() -> Self {
		Self {
			// Will be replaced by create_workspace before any use.
			diagonal: unsafe { std::mem::zeroed() },
			initialised: false,
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Preconditioner impl
// ════════════════════════════════════════════════════════════════════════════

impl<T: Scalar, M: Manifold<T>> Preconditioner<T, M> for JacobiPreconditioner<T, M>
where
	M::TangentVector: AsElementWise<T>,
{
	type Workspace = JacobiWorkspace<M::TangentVector>;

	fn create_workspace(&self, manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let mut diagonal = manifold.allocate_tangent();

		match &self.source {
			DiagonalSource::Fixed(cell) => {
				// One-shot drain: move the stored diagonal into the workspace.
				// SAFETY: create_workspace is called exactly once, before the
				// solver loop, on a single thread.
				let stored = unsafe { &mut *cell.get() };
				if let Some(d) = stored.take() {
					diagonal = d;
				}
				// Apply floor clamping to the user-provided diagonal.
				diagonal.ew_clamp_min(self.floor);
			}
			DiagonalSource::Dynamic(compute) => {
				let mut man_ws = manifold.create_workspace(proto_point);
				compute(manifold, proto_point, &mut diagonal, &mut man_ws);
				diagonal.ew_clamp_min(self.floor);
			}
		}

		JacobiWorkspace {
			diagonal,
			initialised: true,
		}
	}

	/// Applies P⁻¹ v = v ⊘ d element-wise.
	///
	/// Single-pass, SIMD-friendly: iterates over contiguous slices.
	/// **Zero allocation.**
	fn apply(
		&self,
		manifold: &M,
		_point: &M::Point,
		v: &M::TangentVector,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		_man_ws: &mut M::Workspace,
	) {
		debug_assert!(ws.initialised, "Jacobi diagonal not initialised");
		// result ← v (copy)
		manifold.copy_tangent(result, v);
		// result ← result ⊘ diagonal (element-wise divide in-place)
		result.ew_div_assign(&ws.diagonal);
	}

	/// Recomputes the diagonal for dynamic mode after an accepted step.
	#[allow(unused_variables)]
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
		if let DiagonalSource::Dynamic(compute) = &self.source {
			compute(manifold, new_point, &mut ws.diagonal, man_ws);
			ws.diagonal.ew_clamp_min(self.floor);
			ws.initialised = true;
		}
		// Fixed: diagonal never changes.
	}

	fn reset(&mut self) {
		// Nothing to do on self; the solver should re-call create_workspace
		// or the dynamic callback will refresh on the next update.
	}

	#[inline]
	fn is_identity(&self) -> bool {
		false
	}
}