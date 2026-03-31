//! Optimization problems on the PSD cone S⁺(n).
//!
//! # Problems
//!
//! - [`NearestCorrelation`] — Nearest correlation matrix
//! - [`MaxCutSDP`] — Max-Cut semidefinite relaxation (Burer-Monteiro)

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Nearest Correlation Matrix
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct NearestCorrelation<T: Scalar, B: LinAlgBackend<T>> {
	pub target: B::Matrix,
	target_vec: B::Vector,
	pub mu: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> NearestCorrelation<T, B> {
	pub fn new(target: B::Matrix, mu: T) -> Self {
		debug_assert_eq!(
			MatrixView::nrows(&target),
			MatrixView::ncols(&target),
			"Target must be square"
		);
		let target_vec = B::Vector::from_slice(target.as_slice());
		Self {
			target,
			target_vec,
			mu,
			_phantom: PhantomData,
		}
	}
}

pub struct NearestCorrelationWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for NearestCorrelationWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for NearestCorrelationWorkspace<T, B> where
	B::Vector: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for NearestCorrelationWorkspace<T, B> where
	B::Vector: Sync
{
}

impl<T, B, M> Problem<T, M> for NearestCorrelation<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = NearestCorrelationWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let len = VectorView::len(proto_point);
		NearestCorrelationWorkspace {
			egrad: B::Vector::zeros(len),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — uses ws.egrad as diff buffer.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let half = <T as Scalar>::from_f64(0.5);
		// diff = x − c → ws.egrad
		ws.egrad.copy_from(point);
		ws.egrad.sub_assign(&self.target_vec);
		half * ws.egrad.norm_squared()
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		ws.egrad.copy_from(point);
		ws.egrad.sub_assign(&self.target_vec);
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Max-Cut SDP (Burer-Monteiro)
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct MaxCutSDP<T: Scalar, B: LinAlgBackend<T>> {
	pub laplacian: B::Matrix,
	laplacian_vec: B::Vector,
	pub mu: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> MaxCutSDP<T, B> {
	pub fn new(laplacian: B::Matrix, mu: T) -> Self {
		let laplacian_vec = B::Vector::from_slice(laplacian.as_slice());
		Self {
			laplacian,
			laplacian_vec,
			mu,
			_phantom: PhantomData,
		}
	}

	pub fn from_adjacency(w: B::Matrix, mu: T) -> Self {
		let n = MatrixView::nrows(&w);
		let ones = <B::Vector as VectorOps<T>>::from_fn(n, |_| T::one());
		let degree = w.mat_vec(&ones);
		let mut laplacian = B::Matrix::from_diagonal(&degree);
		laplacian.sub_assign(&w);
		Self::new(laplacian, mu)
	}
}

pub struct MaxCutSDPWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Vector,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for MaxCutSDPWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Vector::zeros(0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for MaxCutSDPWorkspace<T, B> where B::Vector: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for MaxCutSDPWorkspace<T, B> where B::Vector: Sync {}

impl<T, B, M> Problem<T, M> for MaxCutSDP<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Vector, TangentVector = B::Vector>,
{
	type Workspace = MaxCutSDPWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let len = VectorView::len(proto_point);
		MaxCutSDPWorkspace {
			egrad: B::Vector::zeros(len),
			_phantom: PhantomData,
		}
	}

	/// Pure dot product — no allocation needed.
	fn cost(
		&self,
		point: &M::Point,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let quarter = <T as Scalar>::from_f64(0.25);
		-quarter * self.laplacian_vec.dot(point)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let quarter = <T as Scalar>::from_f64(0.25);
		ws.egrad.copy_from(&self.laplacian_vec);
		ws.egrad.scale_mut(-quarter);
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, result, manifold_ws);
	}
}
