//! Optimization problem interface for Riemannian manifolds.
//!
//! This module defines the [`Problem`] trait — the bridge between a cost
//! function and the manifold geometry. Every method is infallible, in-place,
//! and takes an opaque `Workspace` to avoid heap allocations in the hot loop.
//!
//! # Design Principles
//!
//! 1. **Zero allocation** — all outputs are written into caller-provided `&mut` buffers.
//! 2. **Infallible** — no `Result`. If the point is off-manifold, that's a solver bug
//!    caught by `debug_assert!`, not a runtime error on the fast path.
//! 3. **Riemannian-native** — `riemannian_gradient` returns a vector in T_x ℳ,
//!    never a raw Euclidean gradient. The projection is the problem's responsibility.
//! 4. **Backend-agnostic** — generic over `M: Manifold<T>`, inheriting the backend
//!    from the manifold's associated types.
//! 5. **Shared workspace** — intermediate buffers live in `Problem::Workspace`,
//!    allocated once by the solver and reused every iteration.


pub mod euclidean;
pub mod fixed_rank;
pub mod grassmann;
pub mod hyperbolic;
pub mod oblique;
pub mod product;
pub mod psd_cone;
pub mod sphere;
pub mod spd;
pub mod stiefel;

use crate::{manifold::Manifold, types::Scalar};
use std::fmt::Debug;
use std::sync::atomic::{AtomicUsize, Ordering};

// ════════════════════════════════════════════════════════════════════════════
//  Core trait
// ════════════════════════════════════════════════════════════════════════════

/// An optimization problem f: ℳ → ℝ on a Riemannian manifold.
///
/// # Type Parameters
///
/// * `T` — scalar type (f32 or f64)
/// * `M` — the manifold on which the problem is defined
pub trait Problem<T: Scalar, M: Manifold<T>>: Debug {
	/// Opaque workspace for intermediate computations (e.g. Euclidean
	/// gradient buffer before projection). Use `()` if none needed.
	type Workspace: Default + Send + Sync;

	/// Allocates a workspace sized for the given prototype point.
	///
	/// Called once by the solver before the optimization loop.
	#[inline]
	fn create_workspace(&self, _manifold: &M, _proto_point: &M::Point) -> Self::Workspace {
		Self::Workspace::default()
	}

	/// Evaluates the objective function f(x).
	fn cost(&self, point: &M::Point) -> T;

	/// Computes the Riemannian gradient grad f(x) ∈ T_x ℳ in-place.
	///
	/// The result **must** be a valid tangent vector. If the underlying
	/// computation yields a Euclidean gradient, this method must project
	/// it via `manifold.euclidean_to_riemannian_gradient`.
	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	);

	/// Evaluates both cost and Riemannian gradient.
	///
	/// Override this when cost and gradient share intermediate computations
	/// (e.g. the matrix-vector product A·x in a quadratic).
	#[inline]
	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		let c = self.cost(point);
		self.riemannian_gradient(manifold, point, gradient, ws, manifold_ws);
		c
	}

	/// Computes the Riemannian Hessian-vector product Hess f(x)[ξ] in-place.
	///
	/// Required only by second-order solvers (trust-region, Newton).
	/// Default panics at runtime.
	#[inline]
	fn riemannian_hessian_vector_product(
		&self,
		_manifold: &M,
		_point: &M::Point,
		_vector: &M::TangentVector,
		_result: &mut M::TangentVector,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) {
		unimplemented!(
			"Hessian-vector product not implemented. Override for second-order solvers."
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Counting wrapper (profiling / debugging)
// ════════════════════════════════════════════════════════════════════════════

/// Transparent wrapper that counts cost, gradient, and Hessian evaluations.
///
/// Useful for benchmarking and verifying that solvers don't waste evaluations.
#[derive(Debug)]
pub struct CountingProblem<P> {
	/// The underlying problem.
	pub inner: P,
	/// Number of cost evaluations.
	pub cost_count: AtomicUsize,
	/// Number of gradient evaluations.
	pub gradient_count: AtomicUsize,
	/// Number of Hessian-vector product evaluations.
	pub hessian_count: AtomicUsize,
}

impl<P> CountingProblem<P> {
	/// Wraps an existing problem with evaluation counters.
	pub fn new(inner: P) -> Self {
		Self {
			inner,
			cost_count: AtomicUsize::new(0),
			gradient_count: AtomicUsize::new(0),
			hessian_count: AtomicUsize::new(0),
		}
	}

	/// Resets all counters to zero.
	pub fn reset_counts(&self) {
		self.cost_count.store(0, Ordering::Relaxed);
		self.gradient_count.store(0, Ordering::Relaxed);
		self.hessian_count.store(0, Ordering::Relaxed);
	}

	/// Returns (cost_evals, gradient_evals, hessian_evals).
	pub fn counts(&self) -> (usize, usize, usize) {
		(
			self.cost_count.load(Ordering::Relaxed),
			self.gradient_count.load(Ordering::Relaxed),
			self.hessian_count.load(Ordering::Relaxed),
		)
	}
}

impl<T, M, P> Problem<T, M> for CountingProblem<P>
where
	T: Scalar,
	M: Manifold<T>,
	P: Problem<T, M>,
{
	type Workspace = P::Workspace;

	#[inline]
	fn create_workspace(&self, manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		self.inner.create_workspace(manifold, proto_point)
	}

	#[inline]
	fn cost(&self, point: &M::Point) -> T {
		self.cost_count.fetch_add(1, Ordering::Relaxed);
		self.inner.cost(point)
	}

	#[inline]
	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		self.gradient_count.fetch_add(1, Ordering::Relaxed);
		self.inner
			.riemannian_gradient(manifold, point, result, ws, manifold_ws);
	}

	#[inline]
	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		self.cost_count.fetch_add(1, Ordering::Relaxed);
		self.gradient_count.fetch_add(1, Ordering::Relaxed);
		self.inner
			.cost_and_gradient(manifold, point, gradient, ws, manifold_ws)
	}

	#[inline]
	fn riemannian_hessian_vector_product(
		&self,
		manifold: &M,
		point: &M::Point,
		vector: &M::TangentVector,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		self.hessian_count.fetch_add(1, Ordering::Relaxed);
		self.inner
			.riemannian_hessian_vector_product(manifold, point, vector, result, ws, manifold_ws);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Quadratic cost (testing & eigenvalue problems)
// ════════════════════════════════════════════════════════════════════════════

use crate::linalg::{LinAlgBackend, MatrixOps, VectorOps, VectorView};
use std::marker::PhantomData;

/// Quadratic cost f(x) = ½ x^T A x + b^T x + c.
///
/// On the sphere, minimizing this is equivalent to finding the smallest
/// eigenvector of A (Rayleigh quotient). Generic over the backend.
#[derive(Debug, Clone)]
pub struct QuadraticCost<T: Scalar, B: LinAlgBackend<T>> {
	/// Symmetric matrix A.
	pub a: B::Matrix,
	/// Linear term b.
	pub b: B::Vector,
	/// Constant term c.
	pub c: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> QuadraticCost<T, B> {
	/// Creates a new quadratic cost f(x) = ½ x^T A x + b^T x + c.
	pub fn new(a: B::Matrix, b: B::Vector, c: T) -> Self {
		Self {
			a,
			b,
			c,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`QuadraticCost`]: buffers for Euclidean gradient and HVP.
pub struct QuadraticWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Buffer for the Euclidean gradient (A·x + b).
	pub egrad: B::Vector,
	/// Buffer for the Euclidean Hessian-vector product (A·ξ).
	pub ehvp: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for QuadraticWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: <B::Vector as VectorOps<T>>::zeros(0),
			ehvp: <B::Vector as VectorOps<T>>::zeros(0),
			_phantom: PhantomData,
		}
	}
}

// SAFETY: B::Vector: Send + Sync is guaranteed by VectorOps: Send + Sync
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for QuadraticWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for QuadraticWorkspace<T, B> where B::Vector: Sync {}

impl<T, B, M> Problem<T, M> for QuadraticCost<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = QuadraticWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		QuadraticWorkspace {
			egrad: B::Vector::zeros(n),
			ehvp: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	#[inline]
	fn cost(&self, point: &M::Point) -> T {
		// f(x) = ½ x^T A x + b^T x + c
		// Reuse nothing here — cost-only calls are rare in practice.
		// For the hot path, use cost_and_gradient which shares A·x.
		let ax = self.a.mat_vec(point);
		let quad = point.dot(&ax) * <T as Scalar>::from_f64(0.5);
		let lin = self.b.dot(point);
		quad + lin + self.c
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// 1. Euclidean gradient: ∇f(x) = A·x + b → ws.egrad  (zero alloc)
		self.a.mat_vec_into(point, &mut ws.egrad);
		ws.egrad.add_assign(&self.b);

		// 2. Project to tangent space → result
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		// A·x → ws.egrad  (shared between cost and gradient, zero alloc)
		self.a.mat_vec_into(point, &mut ws.egrad);

		// Cost: ½ x^T (A·x) + b^T x + c
		let cost = point.dot(&ws.egrad) * <T as Scalar>::from_f64(0.5) + self.b.dot(point) + self.c;

		// Euclidean gradient: A·x + b (in-place on ws.egrad)
		ws.egrad.add_assign(&self.b);

		// Project to tangent space
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);

		cost
	}

	fn riemannian_hessian_vector_product(
		&self,
		manifold: &M,
		point: &M::Point,
		vector: &M::TangentVector,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		// Euclidean HVP: ∇²f · ξ = A · ξ → ws.ehvp  (zero alloc)
		self.a.mat_vec_into(vector, &mut ws.ehvp);

		// Euclidean gradient (needed for curvature correction) → ws.egrad
		self.a.mat_vec_into(point, &mut ws.egrad);
		ws.egrad.add_assign(&self.b);

		// Convert to Riemannian HVP (includes Weingarten / curvature correction)
		manifold.euclidean_to_riemannian_hessian(
			point,
			&ws.egrad,
			&ws.ehvp,
			vector,
			result,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Derivative checker (validation utility)
// ════════════════════════════════════════════════════════════════════════════

/// Utility for validating gradient implementations against finite differences.
///
/// Unlike naive ambient-space FD, this uses retraction along tangent directions
/// to stay on the manifold: grad f(x) · ξ ≈ (f(R_x(hξ)) − f(R_x(−hξ))) / 2h.
pub struct DerivativeChecker;

impl DerivativeChecker {
	/// Checks the directional derivative along a random tangent vector.
	///
	/// Returns `(passes, relative_error)`.
	pub fn check_gradient<T, M, P>(
		problem: &P,
		manifold: &M,
		point: &M::Point,
		tol: T,
	) -> (bool, T)
	where
		T: Scalar + num_traits::Float,
		M: Manifold<T>,
		M::TangentVector: VectorOps<T>,
		P: Problem<T, M>,
	{
		let h = T::FD_CENTRAL_STEP;

		// Compute analytical gradient
		let mut grad = manifold.allocate_tangent();
		let mut ws = problem.create_workspace(manifold, point);
		let mut mws = manifold.create_workspace(point);
		problem.riemannian_gradient(manifold, point, &mut grad, &mut ws, &mut mws);

		// Random tangent direction
		let mut xi = manifold.allocate_tangent();
		manifold.random_tangent(point, &mut xi);

		// Analytical directional derivative: ⟨grad f, ξ⟩
		let analytical = manifold.inner_product(point, &grad, &xi, &mut mws);

		// Numerical directional derivative via retraction
		let mut point_plus = manifold.allocate_point();
		let mut point_minus = manifold.allocate_point();

		// h·ξ
		manifold.scale_tangent(h, &mut xi);
		manifold.retract(point, &xi, &mut point_plus, &mut mws);

		// −h·ξ
		manifold.scale_tangent(-T::one(), &mut xi);
		manifold.retract(point, &xi, &mut point_minus, &mut mws);

		let f_plus = problem.cost(&point_plus);
		let f_minus = problem.cost(&point_minus);
		let numerical = (f_plus - f_minus) / (h + h);

		// Relative error
		let denom = T::one().max(analytical.abs().max(numerical.abs()));
		let error = (analytical - numerical).abs() / denom;

		(error < tol, error)
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_counting_problem_counters() {
		// Verify counters initialize and reset correctly
		#[derive(Debug)]
		struct DummyProblem;

		// CountingProblem wraps any type
		let counting = CountingProblem::new(DummyProblem);
		assert_eq!(counting.counts(), (0, 0, 0));

		counting.cost_count.fetch_add(3, Ordering::Relaxed);
		counting.gradient_count.fetch_add(2, Ordering::Relaxed);
		counting.hessian_count.fetch_add(1, Ordering::Relaxed);
		assert_eq!(counting.counts(), (3, 2, 1));

		counting.reset_counts();
		assert_eq!(counting.counts(), (0, 0, 0));
	}
}
