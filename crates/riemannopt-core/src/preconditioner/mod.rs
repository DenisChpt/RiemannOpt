//! Preconditioners for Riemannian solvers.
//!
//! A preconditioner approximates the inverse Hessian P⁻¹ ≈ H⁻¹ and is used
//! to improve the spectral conditioning of the truncated CG subproblem in
//! trust region, Newton-CG, and conjugate gradient solvers.
//!
//! # Design Principles
//!
//! 1. **Zero allocation** — `apply` writes into a caller-provided `&mut` buffer.
//!    All scratch lives in `Preconditioner::Workspace`, allocated once.
//! 2. **Infallible** — no `Result`. A preconditioner must always produce output.
//! 3. **Riemannian-native** — operates on tangent vectors in T_x ℳ.
//! 4. **Zero-cost default** — [`IdentityPreconditioner`] compiles to a single
//!    `memcpy` (or nothing, if the caller aliases the buffers via the manifold).
//!    When used as a generic default, the compiler monomorphises it away entirely.
//! 5. **Backend-agnostic** — generic over `M: Manifold<T>`.
//!
//! # Preconditioned tCG
//!
//! In the standard truncated CG, the iteration uses:
//! ```text
//!   β = ⟨r_new, r_new⟩ / ⟨r_old, r_old⟩
//!   p = −r + β·p
//! ```
//!
//! With a preconditioner P, this becomes:
//! ```text
//!   z = P⁻¹ r
//!   β = ⟨r_new, z_new⟩ / ⟨r_old, z_old⟩
//!   p = −z + β·p
//! ```
//!
//! Mathematically equivalent to CG on P^{−½} H P^{−½}, which has condition
//! number κ(P⁻¹H) ≪ κ(H) when P ≈ H.

pub mod identity;
pub mod lbfgs;
pub mod jacobi;

pub use identity::IdentityPreconditioner;
pub use lbfgs::LbfgsPreconditioner;
pub use jacobi::JacobiPreconditioner;

pub use crate::linalg::AsElementWise;

use crate::{manifold::Manifold, types::Scalar};
use std::fmt::Debug;

// ════════════════════════════════════════════════════════════════════════════
//  Core trait
// ════════════════════════════════════════════════════════════════════════════

/// A preconditioner for Riemannian optimization.
///
/// Computes P⁻¹ v for a tangent vector v ∈ T_x ℳ, where P approximates
/// the Riemannian Hessian Hess f(x). The quality of the approximation
/// determines the conditioning improvement in iterative sub-solvers.
///
/// # Type Parameters
///
/// * `T` — scalar type (f32 or f64)
/// * `M` — the manifold on which the preconditioner operates
///
/// # Lifetime
///
/// Preconditioners may accumulate state across iterations (e.g. L-BFGS
/// curvature pairs). Call [`update`](Self::update) after each accepted step
/// and [`reset`](Self::reset) to clear accumulated state.
pub trait Preconditioner<T: Scalar, M: Manifold<T>>: Debug + Send + Sync {
	/// Opaque workspace for intermediate computations.
	/// Use `()` if none needed (e.g. [`IdentityPreconditioner`]).
	type Workspace: Default + Send + Sync;

	/// Allocates a workspace sized for the given prototype point.
	///
	/// Called once by the solver before the optimisation loop.
	#[inline]
	fn create_workspace(&self, _manifold: &M, _proto_point: &M::Point) -> Self::Workspace {
		Self::Workspace::default()
	}

	/// Applies the preconditioner: result ← P⁻¹ v.
	///
	/// Both `v` and `result` live in T_x ℳ. The implementation **must not**
	/// allocate and **must not** read `result` before writing (it may contain
	/// garbage).
	///
	/// # Arguments
	///
	/// * `manifold`  — the ambient manifold (for inner products / norms)
	/// * `point`     — the current base point x ∈ ℳ
	/// * `v`         — input tangent vector v ∈ T_x ℳ
	/// * `result`    — output buffer for P⁻¹ v  (pre-allocated by caller)
	/// * `ws`        — preconditioner workspace
	/// * `man_ws`    — manifold workspace
	fn apply(
		&self,
		manifold: &M,
		point: &M::Point,
		v: &M::TangentVector,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		man_ws: &mut M::Workspace,
	);

	/// Updates internal state after an accepted step.
	///
	/// Called by the solver when x_old → x_new is accepted. Implementations
	/// that accumulate curvature information (L-BFGS pairs, etc.) should
	/// store the transported (s, y) pair here.
	///
	/// The solver guarantees:
	/// - `step` ∈ T_{x_old} ℳ  (the accepted step before retraction)
	/// - `old_gradient` ∈ T_{x_old} ℳ
	/// - `new_gradient` ∈ T_{x_new} ℳ
	///
	/// Implementations that need both gradients in the *same* tangent space
	/// must parallel-transport internally (using `manifold.parallel_transport`).
	///
	/// Default: no-op. Stateless preconditioners (Identity, Jacobi) need
	/// not override.
	#[inline]
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
		// Stateless: nothing to do.
	}

	/// Clears all accumulated internal state.
	///
	/// Called when the solver needs a fresh start (e.g. after a restart in
	/// CG, or on a user-triggered reset). The next `apply` should behave
	/// as if no prior `update` calls had occurred.
	///
	/// Default: no-op.
	#[inline]
	fn reset(&mut self) {
		// Stateless: nothing to do.
	}

	/// Returns `true` if this preconditioner is the identity (P = I).
	///
	/// Solvers may use this to skip the `apply` call entirely and avoid
	/// allocating the extra `z` buffer when no preconditioning is active.
	/// This enables **zero overhead** for the un-preconditioned path.
	///
	/// Default: `false`.
	#[inline]
	fn is_identity(&self) -> bool {
		false
	}
}