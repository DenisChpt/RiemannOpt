//! Identity preconditioner (P = I).
//!
//! This is the **zero-cost default**. When monomorphised as the default type
//! parameter `Pre = IdentityPreconditioner`, every method either:
//! - inlines to a single `memcpy` ([`apply`] → `copy_tangent`), or
//! - is a no-op eliminated by the compiler ([`update`], [`reset`]).
//!
//! Solvers that check [`is_identity`](super::Preconditioner::is_identity)
//! can skip the `apply` call and the `z` buffer allocation entirely,
//! producing *exactly* the same code as the un-preconditioned algorithm.

use std::fmt::Debug;

use crate::{manifold::Manifold, types::Scalar};

use super::Preconditioner;

/// Identity preconditioner: P⁻¹ v = v.
///
/// Zero-sized type — adds no memory overhead to solver structs.
#[derive(Debug, Clone, Copy, Default)]
pub struct IdentityPreconditioner;

impl IdentityPreconditioner {
	#[inline]
	pub fn new() -> Self {
		Self
	}
}

impl<T: Scalar, M: Manifold<T>> Preconditioner<T, M> for IdentityPreconditioner {
	/// No workspace needed — ZST, eliminated at compile time.
	type Workspace = ();

	#[inline]
	fn apply(
		&self,
		manifold: &M,
		_point: &M::Point,
		v: &M::TangentVector,
		result: &mut M::TangentVector,
		_ws: &mut Self::Workspace,
		_man_ws: &mut M::Workspace,
	) {
		// Single memcpy. Solvers that check is_identity() skip even this.
		manifold.copy_tangent(result, v);
	}

	#[inline]
	fn is_identity(&self) -> bool {
		true
	}
}
