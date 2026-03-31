//! Optimization problems on the unit sphere S^{n-1}.
//!
//! # Problems
//!
//! - [`RayleighQuotient`] — Dominant eigenvector (min/max eigenvalue)
//! - [`MaxCutSphere`] — Max-Cut SDP relaxation (Burer-Monteiro, rank-1)
//! - [`SphericalKMeans`] — Spherical K-Means clustering

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Rayleigh Quotient
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct RayleighQuotient<T: Scalar, B: LinAlgBackend<T>> {
	pub a: B::Matrix,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> RayleighQuotient<T, B> {
	pub fn new(a: B::Matrix) -> Self {
		debug_assert_eq!(a.nrows(), a.ncols(), "A must be square");
		Self {
			a,
			_phantom: PhantomData,
		}
	}
}

pub struct RayleighWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	ax: B::Vector,
	axi: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for RayleighWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ax: B::Vector::zeros(0),
			axi: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RayleighWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RayleighWorkspace<T, B> where B::Vector: Sync {}

impl<T, B, M> Problem<T, M> for RayleighQuotient<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = RayleighWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = VectorView::len(proto_point);
		RayleighWorkspace {
			ax: B::Vector::zeros(n),
			axi: B::Vector::zeros(n),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — uses ws.ax for A·x.
	#[inline]
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		self.a.mat_vec_into(point, &mut ws.ax);
		point.dot(&ws.ax)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		self.a.mat_vec_into(point, &mut ws.ax);
		ws.ax.scale_mut(<T as Scalar>::from_f64(2.0));
		manifold.euclidean_to_riemannian_gradient(point, &ws.ax, result, manifold_ws);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		self.a.mat_vec_into(point, &mut ws.ax);
		let cost = point.dot(&ws.ax);
		ws.ax.scale_mut(<T as Scalar>::from_f64(2.0));
		manifold.euclidean_to_riemannian_gradient(point, &ws.ax, gradient, manifold_ws);
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
		let two = <T as Scalar>::from_f64(2.0);
		self.a.mat_vec_into(point, &mut ws.ax);
		ws.ax.scale_mut(two);
		self.a.mat_vec_into(vector, &mut ws.axi);
		ws.axi.scale_mut(two);
		manifold.euclidean_to_riemannian_hessian(
			point,
			&ws.ax,
			&ws.axi,
			vector,
			result,
			manifold_ws,
		);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Max-Cut (Burer-Monteiro, rank-1 relaxation)
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MaxCutSphere<T: Scalar, B: LinAlgBackend<T>> {
	inner: RayleighQuotient<T, B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> MaxCutSphere<T, B> {
	pub fn from_laplacian(laplacian: B::Matrix) -> Self {
		Self {
			inner: RayleighQuotient::new(laplacian),
		}
	}

	pub fn from_adjacency(w: B::Matrix) -> Self {
		let n = w.nrows();
		debug_assert_eq!(n, w.ncols(), "Adjacency matrix must be square");
		let ones = B::Vector::from_fn(n, |_| T::one());
		let degree = w.mat_vec(&ones);
		let mut laplacian = B::Matrix::from_diagonal(&degree);
		laplacian.sub_assign(&w);
		Self::from_laplacian(laplacian)
	}
}

impl<T, B, M> Problem<T, M> for MaxCutSphere<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = RayleighWorkspace<T, B>;

	fn create_workspace(&self, manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		<RayleighQuotient<T, B> as Problem<T, M>>::create_workspace(
			&self.inner,
			manifold,
			proto_point,
		)
	}

	#[inline]
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		<RayleighQuotient<T, B> as Problem<T, M>>::cost(&self.inner, point, ws, manifold_ws)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		<RayleighQuotient<T, B> as Problem<T, M>>::riemannian_gradient(
			&self.inner,
			manifold,
			point,
			result,
			ws,
			manifold_ws,
		);
	}

	fn cost_and_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		gradient: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) -> T {
		<RayleighQuotient<T, B> as Problem<T, M>>::cost_and_gradient(
			&self.inner,
			manifold,
			point,
			gradient,
			ws,
			manifold_ws,
		)
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
		<RayleighQuotient<T, B> as Problem<T, M>>::riemannian_hessian_vector_product(
			&self.inner,
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
//  Spherical K-Means
// ════════════════════════════════════════════════════════════════════════════

pub struct SphericalKMeansWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for SphericalKMeansWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for SphericalKMeansWorkspace<T, B> where
	B::Vector: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for SphericalKMeansWorkspace<T, B> where
	B::Vector: Sync
{
}

#[derive(Debug, Clone)]
pub struct SphericalKMeans<T: Scalar, B: LinAlgBackend<T>> {
	data_sum: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> SphericalKMeans<T, B> {
	pub fn new(data: &B::Matrix) -> Self {
		let n = data.ncols();
		let m = data.nrows();
		let ones = B::Vector::from_fn(m, |_| T::one());
		let dt = data.transpose_to_owned();
		let data_sum = dt.mat_vec(&ones);
		debug_assert_eq!(VectorView::len(&data_sum), n);
		Self {
			data_sum,
			_phantom: PhantomData,
		}
	}

	pub fn from_data_sum(data_sum: B::Vector) -> Self {
		Self {
			data_sum,
			_phantom: PhantomData,
		}
	}
}

impl<T, B, M> Problem<T, M> for SphericalKMeans<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = SphericalKMeansWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		SphericalKMeansWorkspace {
			egrad: B::Vector::zeros(VectorView::len(proto_point)),
			_phantom: PhantomData,
		}
	}

	/// Pure dot product — no allocation needed.
	#[inline]
	fn cost(
		&self,
		point: &M::Point,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		-self.data_sum.dot(point)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		ws.egrad.copy_from(&self.data_sum);
		ws.egrad.scale_mut(-T::one());
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
		let cost = -self.data_sum.dot(point);
		ws.egrad.copy_from(&self.data_sum);
		ws.egrad.scale_mut(-T::one());
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		cost
	}
}
