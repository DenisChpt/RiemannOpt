//! Optimization problems on the oblique manifold OB(n,p).
//!
//! # Problems
//!
//! - [`DictionaryLearning`] — Sparse dictionary learning
//! - [`ObliqueICA`] — Non-orthogonal Independent Component Analysis
//! - [`PhaseRetrieval`] — Signal recovery from magnitude measurements

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Dictionary Learning
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct DictionaryLearning<T: Scalar, B: LinAlgBackend<T>> {
	yst: B::Matrix,
	sst: B::Matrix,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> DictionaryLearning<T, B> {
	pub fn new(y: &B::Matrix, codes: &B::Matrix) -> Self {
		let n = y.nrows();
		let p = codes.nrows();
		let mut yst = B::Matrix::zeros(n, p);
		yst.gemm_bt(T::one(), y.as_view(), codes.as_view(), T::zero());
		let mut sst = B::Matrix::zeros(p, p);
		sst.gemm_bt(T::one(), codes.as_view(), codes.as_view(), T::zero());
		Self {
			yst,
			sst,
			_phantom: PhantomData,
		}
	}
}

pub struct DictLearnWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Matrix,
	ehvp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for DictLearnWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for DictLearnWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for DictLearnWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M> Problem<T, M> for DictionaryLearning<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = DictLearnWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		DictLearnWorkspace {
			egrad: B::Matrix::zeros(n, p),
			ehvp: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — uses ws.egrad for DSSᵀ.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let half = <T as Scalar>::from_f64(0.5);
		ws.egrad
			.gemm(T::one(), point.as_view(), self.sst.as_view(), T::zero());
		let quad = half * point.frobenius_dot(&ws.egrad);
		let lin = point.frobenius_dot(&self.yst);
		quad - lin
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		ws.egrad
			.gemm(T::one(), point.as_view(), self.sst.as_view(), T::zero());
		ws.egrad.sub_assign(&self.yst);
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
		let half = <T as Scalar>::from_f64(0.5);
		ws.egrad
			.gemm(T::one(), point.as_view(), self.sst.as_view(), T::zero());
		let quad = half * point.frobenius_dot(&ws.egrad);
		let lin = point.frobenius_dot(&self.yst);
		let cost = quad - lin;
		ws.egrad.sub_assign(&self.yst);
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
		ws.ehvp
			.gemm(T::one(), vector.as_view(), self.sst.as_view(), T::zero());
		ws.egrad
			.gemm(T::one(), point.as_view(), self.sst.as_view(), T::zero());
		ws.egrad.sub_assign(&self.yst);
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
//  Oblique ICA
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct ObliqueICA<T: Scalar, B: LinAlgBackend<T>> {
	pub data: B::Matrix,
	pub contrast: super::stiefel::ICAContrast,
	inv_m: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> ObliqueICA<T, B> {
	pub fn new(data: B::Matrix, contrast: super::stiefel::ICAContrast) -> Self {
		let m = data.ncols();
		Self {
			data,
			contrast,
			inv_m: T::one() / <T as RealScalar>::from_usize(m),
			_phantom: PhantomData,
		}
	}
}

pub struct ObliqueICAWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	wtx: B::Matrix,
	egrad: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for ObliqueICAWorkspace<T, B> {
	fn default() -> Self {
		Self {
			wtx: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for ObliqueICAWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for ObliqueICAWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M> Problem<T, M> for ObliqueICA<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = ObliqueICAWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let p = proto_point.ncols();
		let m = self.data.ncols();
		let n = proto_point.nrows();
		ObliqueICAWorkspace {
			wtx: B::Matrix::zeros(p, m),
			egrad: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — uses ws.wtx for WᵀX.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let p = point.ncols();
		let m = self.data.ncols();
		ws.wtx
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());
		let mut total = T::zero();
		for k in 0..p {
			for j in 0..m {
				total += self.contrast.g(ws.wtx.get(k, j));
			}
		}
		-self.inv_m * total
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let p = point.ncols();
		let m = self.data.ncols();
		ws.wtx
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());
		for k in 0..p {
			for j in 0..m {
				let s = ws.wtx.get(k, j);
				*ws.wtx.get_mut(k, j) = self.contrast.g_prime(s);
			}
		}
		ws.egrad.gemm_bt(
			-self.inv_m,
			self.data.as_view(),
			ws.wtx.as_view(),
			T::zero(),
		);
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
		let p = point.ncols();
		let m = self.data.ncols();
		ws.wtx
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());
		let mut total = T::zero();
		for k in 0..p {
			for j in 0..m {
				let s = ws.wtx.get(k, j);
				total += self.contrast.g(s);
				*ws.wtx.get_mut(k, j) = self.contrast.g_prime(s);
			}
		}
		let cost = -self.inv_m * total;
		ws.egrad.gemm_bt(
			-self.inv_m,
			self.data.as_view(),
			ws.wtx.as_view(),
			T::zero(),
		);
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		cost
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Phase Retrieval
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct PhaseRetrieval<T: Scalar, B: LinAlgBackend<T>> {
	pub measurements: B::Matrix,
	pub intensities: B::Vector,
	inv_m: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> PhaseRetrieval<T, B> {
	pub fn new(measurements: B::Matrix, intensities: B::Vector) -> Self {
		let m = measurements.nrows();
		debug_assert_eq!(VectorView::len(&intensities), m);
		Self {
			measurements,
			intensities,
			inv_m: T::one() / <T as RealScalar>::from_usize(m),
			_phantom: PhantomData,
		}
	}
}

pub struct PhaseRetrievalWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	ax: B::Matrix,
	egrad: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for PhaseRetrievalWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ax: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for PhaseRetrievalWorkspace<T, B> where
	B::Matrix: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for PhaseRetrievalWorkspace<T, B> where
	B::Matrix: Sync
{
}

impl<T, B, M> Problem<T, M> for PhaseRetrieval<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = PhaseRetrievalWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let m = self.measurements.nrows();
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		PhaseRetrievalWorkspace {
			ax: B::Matrix::zeros(m, p),
			egrad: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — uses ws.ax for AX.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let m = self.measurements.nrows();
		let p = point.ncols();
		ws.ax.gemm(
			T::one(),
			self.measurements.as_view(),
			point.as_view(),
			T::zero(),
		);
		let quarter_inv_m = <T as Scalar>::from_f64(0.25) * self.inv_m;
		let mut cost = T::zero();
		for j in 0..p {
			for i in 0..m {
				let aix = ws.ax.get(i, j);
				let residual = aix * aix - self.intensities.get(i);
				cost += residual * residual;
			}
		}
		quarter_inv_m * cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		self.compute_egrad(point, ws);
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
		let m = self.measurements.nrows();
		let p = point.ncols();
		ws.ax.gemm(
			T::one(),
			self.measurements.as_view(),
			point.as_view(),
			T::zero(),
		);
		let quarter_inv_m = <T as Scalar>::from_f64(0.25) * self.inv_m;
		let mut cost = T::zero();
		for j in 0..p {
			for i in 0..m {
				let aix = ws.ax.get(i, j);
				let intensity = aix * aix;
				let residual = intensity - self.intensities.get(i);
				cost += residual * residual;
				*ws.ax.get_mut(i, j) = residual * aix;
			}
		}
		ws.egrad.gemm_at(
			self.inv_m,
			self.measurements.as_view(),
			ws.ax.as_view(),
			T::zero(),
		);
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		quarter_inv_m * cost
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> PhaseRetrieval<T, B> {
	fn compute_egrad(&self, point: &B::Matrix, ws: &mut PhaseRetrievalWorkspace<T, B>) {
		let m = self.measurements.nrows();
		let p = point.ncols();
		ws.ax.gemm(
			T::one(),
			self.measurements.as_view(),
			point.as_view(),
			T::zero(),
		);
		for j in 0..p {
			for i in 0..m {
				let aix = ws.ax.get(i, j);
				let residual = aix * aix - self.intensities.get(i);
				*ws.ax.get_mut(i, j) = residual * aix;
			}
		}
		ws.egrad.gemm_at(
			self.inv_m,
			self.measurements.as_view(),
			ws.ax.as_view(),
			T::zero(),
		);
	}
}
