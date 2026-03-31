//! # Product Manifold M₁ × M₂
//!
//! This module provides a generic product manifold implementation.
//!
//! By representing points and tangent vectors as Rust tuples `(M1::Point, M2::Point)`,
//! we achieve perfect static dispatch, zero-cost component separation, and the
//! ability to mix heterogeneous manifolds (e.g., vectors and matrices) without
//! any heap allocation or concatenation overhead.

use crate::{manifold::Manifold, types::Scalar};
use num_traits::Float;

/// A product manifold M₁ × M₂.
///
/// Combines two manifolds into a single manifold. Points, tangent vectors,
/// and workspaces are represented as tuples of the underlying components.
#[derive(Debug, Clone)]
pub struct Product<M1, M2> {
	pub manifold1: M1,
	pub manifold2: M2,
}

impl<M1, M2> Product<M1, M2> {
	/// Creates a new static product manifold M₁ × M₂.
	#[inline]
	pub fn new(manifold1: M1, manifold2: M2) -> Self {
		Self {
			manifold1,
			manifold2,
		}
	}

	#[inline]
	pub fn components(&self) -> (&M1, &M2) {
		(&self.manifold1, &self.manifold2)
	}
}

impl<T, M1, M2> Manifold<T> for Product<M1, M2>
where
	T: Scalar + Float,
	M1: Manifold<T>,
	M2: Manifold<T>,
{
	// The ultimate zero-cost abstraction: just use tuples!
	type Point = (M1::Point, M2::Point);
	type TangentVector = (M1::TangentVector, M2::TangentVector);
	type Workspace = (M1::Workspace, M2::Workspace);

	#[inline]
	fn create_workspace(&self, proto: &Self::Point) -> Self::Workspace {
		(
			self.manifold1.create_workspace(&proto.0),
			self.manifold2.create_workspace(&proto.1),
		)
	}

	#[inline]
	fn name(&self) -> &str {
		"Product" // For dynamic display, one might format both names, but static &str is fine here
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.manifold1.dimension() + self.manifold2.dimension()
	}

	#[inline]
	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		self.manifold1.is_point_on_manifold(&point.0, tol)
			&& self.manifold2.is_point_on_manifold(&point.1, tol)
	}

	#[inline]
	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		self.manifold1
			.is_vector_in_tangent_space(&point.0, &vector.0, tol)
			&& self
				.manifold2
				.is_vector_in_tangent_space(&point.1, &vector.1, tol)
	}

	#[inline]
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		self.manifold1.project_point(&point.0, &mut result.0);
		self.manifold2.project_point(&point.1, &mut result.1);
	}

	#[inline]
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.manifold1
			.project_tangent(&point.0, &vector.0, &mut result.0, &mut ws.0);
		self.manifold2
			.project_tangent(&point.1, &vector.1, &mut result.1, &mut ws.1);
	}

	#[inline]
	fn inner_product(
		&self,
		point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> T {
		let ip1 = self
			.manifold1
			.inner_product(&point.0, &u.0, &v.0, &mut ws.0);
		let ip2 = self
			.manifold2
			.inner_product(&point.1, &u.1, &v.1, &mut ws.1);
		ip1 + ip2
	}

	#[inline]
	fn norm(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> T {
		// ||(u,v)||^2 = ||u||^2 + ||v||^2
		let n1 = self.manifold1.norm(&point.0, &vector.0, &mut ws.0);
		let n2 = self.manifold2.norm(&point.1, &vector.1, &mut ws.1);
		<T as Float>::sqrt(n1 * n1 + n2 * n2)
	}

	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		self.manifold1
			.retract(&point.0, &tangent.0, &mut result.0, &mut ws.0);
		self.manifold2
			.retract(&point.1, &tangent.1, &mut result.1, &mut ws.1);
	}

	#[inline]
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.manifold1
			.inverse_retract(&point.0, &other.0, &mut result.0, &mut ws.0);
		self.manifold2
			.inverse_retract(&point.1, &other.1, &mut result.1, &mut ws.1);
	}

	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		egrad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.manifold1.euclidean_to_riemannian_gradient(
			&point.0,
			&egrad.0,
			&mut result.0,
			&mut ws.0,
		);
		self.manifold2.euclidean_to_riemannian_gradient(
			&point.1,
			&egrad.1,
			&mut result.1,
			&mut ws.1,
		);
	}

	#[inline]
	fn euclidean_to_riemannian_hessian(
		&self,
		point: &Self::Point,
		egrad: &Self::TangentVector,
		ehvp: &Self::TangentVector,
		tangent_vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.manifold1.euclidean_to_riemannian_hessian(
			&point.0,
			&egrad.0,
			&ehvp.0,
			&tangent_vector.0,
			&mut result.0,
			&mut ws.0,
		);
		self.manifold2.euclidean_to_riemannian_hessian(
			&point.1,
			&egrad.1,
			&ehvp.1,
			&tangent_vector.1,
			&mut result.1,
			&mut ws.1,
		);
	}

	#[inline]
	fn parallel_transport(
		&self,
		from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.manifold1
			.parallel_transport(&from.0, &to.0, &vector.0, &mut result.0, &mut ws.0);
		self.manifold2
			.parallel_transport(&from.1, &to.1, &vector.1, &mut result.1, &mut ws.1);
	}

	#[inline]
	fn random_point(&self, result: &mut Self::Point) {
		self.manifold1.random_point(&mut result.0);
		self.manifold2.random_point(&mut result.1);
	}

	#[inline]
	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) {
		self.manifold1.random_tangent(&point.0, &mut result.0);
		self.manifold2.random_tangent(&point.1, &mut result.1);
	}

	#[inline]
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let d1 = self.manifold1.distance(&x.0, &y.0);
		let d2 = self.manifold2.distance(&x.1, &y.1);
		<T as Float>::sqrt(d1 * d1 + d2 * d2)
	}

	#[inline]
	fn has_exact_exp_log(&self) -> bool {
		self.manifold1.has_exact_exp_log() && self.manifold2.has_exact_exp_log()
	}

	#[inline]
	fn is_flat(&self) -> bool {
		self.manifold1.is_flat() && self.manifold2.is_flat()
	}

	// ════════════════════════════════════════════════════════════════════════
	// Vector ops
	// ════════════════════════════════════════════════════════════════════════

	#[inline]
	fn scale_tangent(&self, scalar: T, v: &mut Self::TangentVector) {
		self.manifold1.scale_tangent(scalar, &mut v.0);
		self.manifold2.scale_tangent(scalar, &mut v.1);
	}

	#[inline]
	fn add_tangents(&self, v1: &mut Self::TangentVector, v2: &Self::TangentVector) {
		self.manifold1.add_tangents(&mut v1.0, &v2.0);
		self.manifold2.add_tangents(&mut v1.1, &v2.1);
	}

	#[inline]
	fn axpy_tangent(&self, alpha: T, x: &Self::TangentVector, y: &mut Self::TangentVector) {
		self.manifold1.axpy_tangent(alpha, &x.0, &mut y.0);
		self.manifold2.axpy_tangent(alpha, &x.1, &mut y.1);
	}

	#[inline]
	fn allocate_point(&self) -> Self::Point {
		(
			self.manifold1.allocate_point(),
			self.manifold2.allocate_point(),
		)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		(
			self.manifold1.allocate_tangent(),
			self.manifold2.allocate_tangent(),
		)
	}
}

/// Creates a static product manifold with type inference.
#[inline]
pub fn product<M1, M2>(manifold1: M1, manifold2: M2) -> Product<M1, M2> {
	Product::new(manifold1, manifold2)
}
