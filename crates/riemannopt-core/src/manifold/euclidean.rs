//! # Euclidean Manifold ℝ^n
//!
//! The Euclidean manifold is the standard n-dimensional vector space equipped
//! with the usual Euclidean metric. While trivial as a manifold, it provides
//! a baseline for optimization algorithms and enables unconstrained optimization
//! within the Riemannian framework.
//!
//! ## Mathematical Definition
//!
//! ```text
//! ℝ^n = {x = (x₁, ..., xₙ) : xᵢ ∈ ℝ}
//! ```
//!
//! ## Geometric Structure
//!
//! All Riemannian operations are trivial:
//! - **Tangent space**: T_x ℝ^n = ℝ^n for all x
//! - **Metric**: ⟨u, v⟩_x = u^T v (standard inner product)
//! - **Projection**: P_x(v) = v (identity)
//! - **Retraction**: R_x(v) = x + v (addition)
//! - **Exponential map**: exp_x(v) = x + v
//! - **Logarithmic map**: log_x(y) = y − x
//! - **Parallel transport**: Γ_{x→y}(v) = v (identity)
//!
//! ## Properties
//!
//! - Dimension: n
//! - Curvature: Flat (zero everywhere)
//! - Complete metric space, simply connected
//! - Geodesics: straight lines
//!
//! ## Computational Complexity
//!
//! | Operation          | Time   | Space |
//! |--------------------|--------|-------|
//! | All operations     | O(n)   | O(1)  |

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};

use crate::{
	linalg::{LinAlgBackend, VectorOps, VectorView},
	manifold::Manifold,
	types::Scalar,
};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

/// The Euclidean manifold ℝ^n.
///
/// Generic over scalar type `T` and linear algebra backend `B`.
#[derive(Clone)]
pub struct Euclidean<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	n: usize,
	_phantom: PhantomData<(T, B)>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for Euclidean<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Euclidean(ℝ^{})", self.n)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Euclidean<T, B> {
	/// Creates a new Euclidean manifold of dimension n.
	///
	/// # Panics
	///
	/// Panics if `n == 0`.
	#[inline]
	pub fn new(n: usize) -> Self {
		assert!(n > 0, "Euclidean manifold requires dimension > 0");
		Self {
			n,
			_phantom: PhantomData,
		}
	}

	/// Returns the dimension of the space.
	#[inline]
	pub fn dim(&self) -> usize {
		self.n
	}
}

impl<T, B> Manifold<T> for Euclidean<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Vector;
	type TangentVector = B::Vector;
	type Workspace = ();

	#[inline]
	fn name(&self) -> &str {
		"Euclidean"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.n
	}

	#[inline]
	fn is_point_on_manifold(&self, point: &Self::Point, _tol: T) -> bool {
		VectorView::len(point) == self.n
	}

	#[inline]
	fn is_vector_in_tangent_space(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		_tol: T,
	) -> bool {
		VectorView::len(vector) == self.n
	}

	#[inline]
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		result.copy_from(point);
	}

	/// P_x(v) = v  (identity — every vector is tangent in ℝ^n)
	#[inline]
	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		result.copy_from(vector);
	}

	/// ⟨u, v⟩_x = u^T v
	#[inline]
	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut (),
	) -> T {
		u.dot(v)
	}

	/// ‖v‖_x = ‖v‖₂
	#[inline]
	fn norm(&self, _point: &Self::Point, vector: &Self::TangentVector, _ws: &mut ()) -> T {
		vector.norm()
	}

	/// R_x(v) = x + v
	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		_ws: &mut (),
	) {
		result.copy_from(point);
		result.add_assign(tangent);
	}

	/// log_x(y) = y − x
	#[inline]
	fn inverse_retract(
		&self,
		x: &Self::Point,
		y: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		result.copy_from(y);
		result.sub_assign(x);
	}

	/// grad f = ∇f  (identity projection)
	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		_point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		result.copy_from(euclidean_grad);
	}

	/// Γ_{x→y}(v) = v  (flat space — identity transport)
	#[inline]
	fn parallel_transport(
		&self,
		_from: &Self::Point,
		_to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut (),
	) {
		result.copy_from(vector);
	}

	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;
		for i in 0..self.n {
			*result.get_mut(i) = <T as Scalar>::from_f64(normal.sample(&mut rng));
		}
	}

	fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector) {
		// In ℝ^n every vector is tangent — just generate a random unit vector
		self.random_point(result);
		let norm = result.norm();
		if norm > T::zero() {
			result.div_scalar_mut(norm);
		}
	}

	/// d(x, y) = ‖y − x‖₂
	///
	/// Not on the hot path — allocates one temporary vector to avoid
	/// the catastrophic cancellation of ‖x‖² + ‖y‖² − 2⟨x,y⟩ when x ≈ y.
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let mut diff: B::Vector = VectorOps::zeros(self.n);
		diff.copy_from(y);
		diff.sub_assign(x);
		diff.norm()
	}

	#[inline]
	fn has_exact_exp_log(&self) -> bool {
		true
	}

	#[inline]
	fn is_flat(&self) -> bool {
		true
	}

	// ════════════════════════════════════════════════════════════════════════
	// Vector ops — trivial in ℝ^n
	// ════════════════════════════════════════════════════════════════════════

	#[inline]
	fn scale_tangent(&self, scalar: T, v: &mut Self::TangentVector) {
		v.scale_mut(scalar);
	}

	#[inline]
	fn add_tangents(&self, v1: &mut Self::TangentVector, v2: &Self::TangentVector) {
		v1.add_assign(v2);
	}

	#[inline]
	fn axpy_tangent(&self, alpha: T, x: &Self::TangentVector, y: &mut Self::TangentVector) {
		y.axpy(alpha, x, T::one());
	}

	#[inline]
	fn allocate_point(&self) -> Self::Point {
		VectorOps::zeros(self.n)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		VectorOps::zeros(self.n)
	}
}
