//! Optimization problem interface for Riemannian manifolds.
//!
//! This module defines the [`Problem`] trait — the bridge between a cost
//! function and the manifold geometry. Every method is infallible, in-place,
//! and takes an opaque `Workspace` to avoid heap allocations in the hot loop.
//!
//! # Design Principles
//!
//! 1. **Zero allocation** — all outputs are written into caller-provided `&mut` buffers.
//! 2. **Infallible** — no `Result`. If the point is off-manifold, that's a solver bug.
//! 3. **Riemannian-native** — `riemannian_gradient` returns a vector in T_x ℳ.
//! 4. **Backend-agnostic** — generic over `M: Manifold<T>`.
//! 5. **Shared workspace** — intermediate buffers live in `Problem::Workspace`,
//!    allocated once by the solver and reused every iteration.

pub mod euclidean;
pub mod fixed_rank;
pub mod grassmann;
pub mod hyperbolic;
pub mod oblique;
pub mod product;
pub mod psd_cone;
pub mod spd;
pub mod sphere;
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
	/// Opaque workspace for intermediate computations.
	/// Use `()` if none needed.
	type Workspace: Default + Send + Sync;

	/// Allocates a workspace sized for the given prototype point.
	///
	/// Called once by the solver before the optimization loop.
	#[inline]
	fn create_workspace(&self, _manifold: &M, _proto_point: &M::Point) -> Self::Workspace {
		Self::Workspace::default()
	}

	/// Evaluates the objective function f(x). **Zero allocation** when
	/// workspaces are properly sized.
	fn cost(&self, point: &M::Point, ws: &mut Self::Workspace, manifold_ws: &mut M::Workspace)
		-> T;

	/// Computes the Riemannian gradient grad f(x) ∈ T_x ℳ in-place.
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
	/// Override when cost and gradient share intermediate computations.
	#[inline]
	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		let c = self.cost(point, ws, manifold_ws);
		self.riemannian_gradient(manifold, point, gradient, ws, manifold_ws);
		c
	}

	/// Computes the Riemannian Hessian-vector product Hess f(x)[ξ] in-place.
	///
	/// Required only by second-order solvers.
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
//  Counting wrapper
// ════════════════════════════════════════════════════════════════════════════

/// Transparent wrapper that counts cost, gradient, and Hessian evaluations.
#[derive(Debug)]
pub struct CountingProblem<P> {
	pub inner: P,
	pub cost_count: AtomicUsize,
	pub gradient_count: AtomicUsize,
	pub hessian_count: AtomicUsize,
}

impl<P> CountingProblem<P> {
	pub fn new(inner: P) -> Self {
		Self {
			inner,
			cost_count: AtomicUsize::new(0),
			gradient_count: AtomicUsize::new(0),
			hessian_count: AtomicUsize::new(0),
		}
	}

	pub fn reset_counts(&self) {
		self.cost_count.store(0, Ordering::Relaxed);
		self.gradient_count.store(0, Ordering::Relaxed);
		self.hessian_count.store(0, Ordering::Relaxed);
	}

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
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		self.cost_count.fetch_add(1, Ordering::Relaxed);
		self.inner.cost(point, ws, manifold_ws)
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
		self.inner.riemannian_hessian_vector_product(
			manifold,
			point,
			vector,
			result,
			ws,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Quadratic cost
// ════════════════════════════════════════════════════════════════════════════

use crate::linalg::{LinAlgBackend, MatrixOps, VectorOps, VectorView};
use std::marker::PhantomData;

/// Quadratic cost f(x) = ½ xᵀAx + bᵀx + c.
#[derive(Debug, Clone)]
pub struct QuadraticCost<T: Scalar, B: LinAlgBackend<T>> {
	pub a: B::Matrix,
	pub b: B::Vector,
	pub c: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> QuadraticCost<T, B> {
	pub fn new(a: B::Matrix, b: B::Vector, c: T) -> Self {
		Self {
			a,
			b,
			c,
			_phantom: PhantomData,
		}
	}
}

/// Workspace for [`QuadraticCost`].
pub struct QuadraticWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// Buffer for A·x (reused by cost and gradient).
	pub ax: B::Vector,
	/// Buffer for the Euclidean gradient (A·x + b).
	pub egrad: B::Vector,
	/// Buffer for the Euclidean HVP (A·ξ).
	pub ehvp: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for QuadraticWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ax: <B::Vector as VectorOps<T>>::zeros(0),
			egrad: <B::Vector as VectorOps<T>>::zeros(0),
			ehvp: <B::Vector as VectorOps<T>>::zeros(0),
			_phantom: PhantomData,
		}
	}
}

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
			ax: B::Vector::zeros(n),
			egrad: B::Vector::zeros(n),
			ehvp: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	#[inline]
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		// A·x → ws.ax  (zero alloc, reusable by cost_and_gradient)
		self.a.mat_vec_into(point, &mut ws.ax);
		let quad = point.dot(&ws.ax) * <T as Scalar>::from_f64(0.5);
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
		// ∇f = A·x + b
		self.a.mat_vec_into(point, &mut ws.egrad);
		ws.egrad.add_assign(&self.b);
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
		// A·x → ws.egrad  (shared)
		self.a.mat_vec_into(point, &mut ws.egrad);

		// Cost: ½ xᵀ(A·x) + bᵀx + c
		let cost = point.dot(&ws.egrad) * <T as Scalar>::from_f64(0.5) + self.b.dot(point) + self.c;

		// Gradient: A·x + b
		ws.egrad.add_assign(&self.b);
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
		// A·ξ → ws.ehvp
		self.a.mat_vec_into(vector, &mut ws.ehvp);

		// Euclidean gradient (for curvature correction)
		self.a.mat_vec_into(point, &mut ws.egrad);
		ws.egrad.add_assign(&self.b);

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
//  Derivative checker
// ════════════════════════════════════════════════════════════════════════════

/// Validates gradient implementations against finite differences on the manifold.
pub struct DerivativeChecker;

impl DerivativeChecker {
	/// Checks the directional derivative along a random tangent vector.
	///
	/// Returns `(passes, relative_error)`.
	pub fn check_gradient<T, M, P>(problem: &P, manifold: &M, point: &M::Point, tol: T) -> (bool, T)
	where
		T: Scalar + num_traits::Float,
		M: Manifold<T>,
		M::TangentVector: VectorOps<T>,
		P: Problem<T, M>,
	{
		let h = T::FD_CENTRAL_STEP;

		let mut grad = manifold.allocate_tangent();
		let mut ws = problem.create_workspace(manifold, point);
		let mut mws = manifold.create_workspace(point);
		problem.riemannian_gradient(manifold, point, &mut grad, &mut ws, &mut mws);

		let mut xi = manifold.allocate_tangent();
		manifold.random_tangent(point, &mut xi);

		// Analytical: ⟨grad f, ξ⟩
		let analytical = manifold.inner_product(point, &grad, &xi, &mut mws);

		// Numerical: (f(R(hξ)) − f(R(−hξ))) / 2h
		let mut point_plus = manifold.allocate_point();
		let mut point_minus = manifold.allocate_point();

		manifold.scale_tangent(h, &mut xi);
		manifold.retract(point, &xi, &mut point_plus, &mut mws);

		manifold.scale_tangent(-T::one(), &mut xi);
		manifold.retract(point, &xi, &mut point_minus, &mut mws);

		let f_plus = problem.cost(&point_plus, &mut ws, &mut mws);
		let f_minus = problem.cost(&point_minus, &mut ws, &mut mws);
		let numerical = (f_plus - f_minus) / (h + h);

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
		#[derive(Debug)]
		struct DummyProblem;

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
