//! Optimization problems on the hyperbolic manifold ℍⁿ (Poincaré ball model).
//!
//! # Problems
//!
//! - [`PoincareEmbedding`] — Graph/tree embedding in hyperbolic space
//! - [`HyperbolicLogisticRegression`] — Logistic regression using geodesic distance

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, RealScalar, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Poincaré Embedding
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct PoincareEmbedding<T: Scalar, B: LinAlgBackend<T>> {
	pub positive_targets: Vec<B::Vector>,
	pub negative_targets: Vec<B::Vector>,
	pub alpha: T,
	pub curvature: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> PoincareEmbedding<T, B> {
	pub fn new(
		positive_targets: Vec<B::Vector>,
		negative_targets: Vec<B::Vector>,
		alpha: T,
	) -> Self {
		Self {
			positive_targets,
			negative_targets,
			alpha,
			curvature: -T::one(),
			_phantom: PhantomData,
		}
	}

	fn poincare_distance(&self, u: &B::Vector, v: &B::Vector) -> T {
		let n = VectorView::len(u);
		let mut diff_sq = T::zero();
		for i in 0..n {
			let d = u.get(i) - v.get(i);
			diff_sq = diff_sq + d * d;
		}
		let u_sq = u.norm_squared();
		let v_sq = v.norm_squared();
		let denom = (T::one() - u_sq) * (T::one() - v_sq);
		let two = <T as Scalar>::from_f64(2.0);
		let arg = T::one() + two * diff_sq / denom.max(T::EPSILON);
		arg.max(T::one() + T::EPSILON).acosh()
	}

	fn distance_sq_egrad(&self, u: &B::Vector, v: &B::Vector, result: &mut B::Vector) {
		let n = VectorView::len(u);
		let u_sq = u.norm_squared();
		let v_sq = v.norm_squared();
		let two = <T as Scalar>::from_f64(2.0);
		let four = <T as Scalar>::from_f64(4.0);

		let alpha_u = T::one() - u_sq;
		let alpha_v = T::one() - v_sq;

		let mut diff_sq = T::zero();
		for i in 0..n {
			let d = u.get(i) - v.get(i);
			diff_sq = diff_sq + d * d;
		}

		let denom = alpha_u * alpha_v;
		let arg = T::one() + two * diff_sq / denom.max(T::EPSILON);
		let dist = arg.max(T::one() + T::EPSILON).acosh();

		if dist < T::EPSILON {
			result.fill(T::zero());
			return;
		}

		let inv_sinh_d = T::one() / (arg * arg - T::one()).max(T::EPSILON).sqrt();
		let scale = two * dist * inv_sinh_d * four / (alpha_u * alpha_u * alpha_v).max(T::EPSILON);

		for i in 0..n {
			let ui = u.get(i);
			let vi = v.get(i);
			let darg_dui = (ui - vi) * alpha_u + diff_sq * ui;
			*result.get_mut(i) = *result.get_mut(i) + scale * darg_dui;
		}
	}
}

pub struct PoincareWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	temp_grad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for PoincareWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			temp_grad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for PoincareWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for PoincareWorkspace<T, B> where B::Vector: Sync {}

impl<T, B, M> Problem<T, M> for PoincareEmbedding<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = PoincareWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		PoincareWorkspace {
			egrad: B::Vector::zeros(n),
			temp_grad: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	fn cost(
		&self,
		point: &M::Point,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let mut cost = T::zero();
		for v in &self.positive_targets {
			let d = self.poincare_distance(point, v);
			cost = cost + d * d;
		}
		if self.alpha > T::zero() {
			for v in &self.negative_targets {
				let d = self.poincare_distance(point, v);
				let d_sq = d * d;
				cost = cost - self.alpha * ((-d_sq).exp() + T::EPSILON).ln();
			}
		}
		cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		ws.egrad.fill(T::zero());
		for v in &self.positive_targets {
			self.distance_sq_egrad(point, v, &mut ws.egrad);
		}
		if self.alpha > T::zero() {
			for v in &self.negative_targets {
				let d = self.poincare_distance(point, v);
				let d_sq = d * d;
				let weight = -self.alpha * (-d_sq).exp() / ((-d_sq).exp() + T::EPSILON);
				ws.temp_grad.fill(T::zero());
				self.distance_sq_egrad(point, v, &mut ws.temp_grad);
				ws.egrad.axpy(weight, &ws.temp_grad, T::one());
			}
		}
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Hyperbolic Logistic Regression
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct HyperbolicLogisticRegression<T: Scalar, B: LinAlgBackend<T>> {
	pub data: Vec<B::Vector>,
	pub labels: Vec<T>,
	pub normal: B::Vector,
	pub lambda: T,
	inv_m: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> HyperbolicLogisticRegression<T, B> {
	pub fn new(data: Vec<B::Vector>, labels: Vec<T>, normal: B::Vector, lambda: T) -> Self {
		debug_assert_eq!(data.len(), labels.len());
		let m = data.len();
		Self {
			data,
			labels,
			normal,
			lambda,
			inv_m: T::one() / <T as RealScalar>::from_usize(m),
			_phantom: PhantomData,
		}
	}
}

pub struct HyperbolicLRWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for HyperbolicLRWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for HyperbolicLRWorkspace<T, B> where
	B::Vector: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for HyperbolicLRWorkspace<T, B> where
	B::Vector: Sync
{
}

#[inline]
fn softplus<T: Scalar>(t: T) -> T {
	let abs_t = t.abs();
	abs_t.max(T::zero()) + (T::one() + (-abs_t).exp()).ln()
}

#[inline]
fn sigmoid<T: Scalar>(t: T) -> T {
	if t >= T::zero() {
		T::one() / (T::one() + (-t).exp())
	} else {
		let e = t.exp();
		e / (T::one() + e)
	}
}

impl<T, B, M> Problem<T, M> for HyperbolicLogisticRegression<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = HyperbolicLRWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		HyperbolicLRWorkspace {
			egrad: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	fn cost(
		&self,
		point: &M::Point,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let n = VectorView::len(point);
		let half = <T as Scalar>::from_f64(0.5);
		let mut loss = T::zero();
		for (xi, &yi) in self.data.iter().zip(&self.labels) {
			let mut score = T::zero();
			for k in 0..n {
				score = score + (xi.get(k) - point.get(k)) * self.normal.get(k);
			}
			loss = loss + softplus(-yi * score);
		}
		self.inv_m * loss + half * self.lambda * point.norm_squared()
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let n = VectorView::len(point);
		ws.egrad.fill(T::zero());
		for (xi, &yi) in self.data.iter().zip(&self.labels) {
			let mut score = T::zero();
			for k in 0..n {
				score = score + (xi.get(k) - point.get(k)) * self.normal.get(k);
			}
			let weight = -yi * sigmoid(-yi * score);
			ws.egrad.axpy(self.inv_m * weight, &self.normal, T::one());
		}
		ws.egrad.axpy(self.lambda, point, T::one());
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}
