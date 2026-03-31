//! Optimization problems on the fixed-rank manifold M_k(m,n).
//!
//! # Problems
//!
//! - [`MatrixCompletion`] — Low-rank matrix completion from partial observations
//! - [`MatrixSensing`] — Recovery from linear measurements

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::{
		fixed_rank::{FixedRankPoint, FixedRankTangent},
		Manifold,
	},
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Matrix Completion
// ════════════════════════════════════════════════════════════════════════════

/// Low-rank matrix completion from partial observations on M_k(m,n).
#[derive(Debug, Clone)]
pub struct MatrixCompletion<T: Scalar> {
	pub rows: Vec<usize>,
	pub cols: Vec<usize>,
	pub values: Vec<T>,
	pub m: usize,
	pub n: usize,
}

impl<T: Scalar> MatrixCompletion<T> {
	pub fn new(m: usize, n: usize, rows: Vec<usize>, cols: Vec<usize>, values: Vec<T>) -> Self {
		debug_assert_eq!(rows.len(), cols.len());
		debug_assert_eq!(rows.len(), values.len());
		Self {
			rows,
			cols,
			values,
			m,
			n,
		}
	}

	#[inline]
	fn reconstruct_entry<B: LinAlgBackend<T>>(
		&self,
		point: &FixedRankPoint<T, B>,
		i: usize,
		j: usize,
	) -> T {
		let rank = VectorView::len(&point.s);
		let mut val = T::zero();
		for k in 0..rank {
			val = val + point.u.get(i, k) * point.s.get(k) * point.v.get(j, k);
		}
		val
	}
}

/// Workspace for [`MatrixCompletion`].
pub struct MatrixCompletionWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	r_omega: B::Matrix,
	rv: B::Matrix,
	ut_rv: B::Matrix,
	u_proj: B::Matrix,
	ut_r: B::Matrix,
	vt_proj: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for MatrixCompletionWorkspace<T, B> {
	fn default() -> Self {
		Self {
			r_omega: B::Matrix::zeros(0, 0),
			rv: B::Matrix::zeros(0, 0),
			ut_rv: B::Matrix::zeros(0, 0),
			u_proj: B::Matrix::zeros(0, 0),
			ut_r: B::Matrix::zeros(0, 0),
			vt_proj: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for MatrixCompletionWorkspace<T, B> where
	B::Matrix: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for MatrixCompletionWorkspace<T, B> where
	B::Matrix: Sync
{
}

impl<T, B, M> Problem<T, M> for MatrixCompletion<T>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = FixedRankPoint<T, B>, TangentVector = FixedRankTangent<T, B>>,
{
	type Workspace = MatrixCompletionWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let m = self.m;
		let n = self.n;
		let rank = VectorView::len(&proto_point.s);
		MatrixCompletionWorkspace {
			r_omega: B::Matrix::zeros(m, n),
			rv: B::Matrix::zeros(m, rank),
			ut_rv: B::Matrix::zeros(rank, rank),
			u_proj: B::Matrix::zeros(m, rank),
			ut_r: B::Matrix::zeros(rank, n),
			vt_proj: B::Matrix::zeros(rank, n),
			_phantom: PhantomData,
		}
	}

	fn cost(
		&self,
		point: &M::Point,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let half = <T as Scalar>::from_f64(0.5);
		let mut cost = T::zero();
		for idx in 0..self.rows.len() {
			let residual = self.reconstruct_entry::<B>(point, self.rows[idx], self.cols[idx])
				- self.values[idx];
			cost = cost + residual * residual;
		}
		half * cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let rank = VectorView::len(&point.s);

		result.s_dot.fill(T::zero());
		for kk in 0..rank {
			let mut val = T::zero();
			for idx in 0..self.rows.len() {
				let i = self.rows[idx];
				let j = self.cols[idx];
				let residual = self.reconstruct_entry::<B>(point, i, j) - self.values[idx];
				val = val + point.u.get(i, kk) * residual * point.v.get(j, kk);
			}
			*result.s_dot.get_mut(kk) = val;
		}

		ws.r_omega.fill(T::zero());
		for idx in 0..self.rows.len() {
			let i = self.rows[idx];
			let j = self.cols[idx];
			let residual = self.reconstruct_entry::<B>(point, i, j) - self.values[idx];
			*ws.r_omega.get_mut(i, j) = residual;
		}

		ws.rv
			.gemm(T::one(), ws.r_omega.as_view(), point.v.as_view(), T::zero());
		ws.ut_rv
			.gemm_at(T::one(), point.u.as_view(), ws.rv.as_view(), T::zero());
		ws.u_proj
			.gemm(T::one(), point.u.as_view(), ws.ut_rv.as_view(), T::zero());
		ws.rv.sub_assign(&ws.u_proj);
		result.u_perp_m.copy_from(&ws.rv);

		ws.ut_r
			.gemm_at(T::one(), point.u.as_view(), ws.r_omega.as_view(), T::zero());
		ws.vt_proj
			.gemm_bt(T::one(), ws.ut_rv.as_view(), point.v.as_view(), T::zero());
		ws.ut_r.sub_assign(&ws.vt_proj);
		result.v_perp_n.copy_from(&ws.ut_r);

		let _ = manifold;
		let _ = manifold_ws;
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Matrix Sensing
// ════════════════════════════════════════════════════════════════════════════

/// Matrix sensing: recover a rank-k matrix from linear measurements.
#[derive(Debug, Clone)]
pub struct MatrixSensing<T: Scalar, B: LinAlgBackend<T>> {
	pub measurements: Vec<B::Matrix>,
	pub observations: Vec<T>,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> MatrixSensing<T, B> {
	pub fn new(measurements: Vec<B::Matrix>, observations: Vec<T>) -> Self {
		debug_assert_eq!(measurements.len(), observations.len());
		Self {
			measurements,
			observations,
			_phantom: PhantomData,
		}
	}

	fn measure(&self, a: &B::Matrix, point: &FixedRankPoint<T, B>) -> T {
		let rank = VectorView::len(&point.s);
		let mut result = T::zero();
		for k in 0..rank {
			let mut dot = T::zero();
			for i in 0..a.nrows() {
				for j in 0..a.ncols() {
					dot = dot + a.get(i, j) * point.u.get(i, k) * point.v.get(j, k);
				}
			}
			result = result + point.s.get(k) * dot;
		}
		result
	}
}

/// Workspace for [`MatrixSensing`].
pub struct MatrixSensingWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Matrix,
	rv: B::Matrix,
	ut_rv: B::Matrix,
	u_proj: B::Matrix,
	ut_e: B::Matrix,
	vt_proj: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for MatrixSensingWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			rv: B::Matrix::zeros(0, 0),
			ut_rv: B::Matrix::zeros(0, 0),
			u_proj: B::Matrix::zeros(0, 0),
			ut_e: B::Matrix::zeros(0, 0),
			vt_proj: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for MatrixSensingWorkspace<T, B> where
	B::Matrix: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for MatrixSensingWorkspace<T, B> where
	B::Matrix: Sync
{
}

impl<T, B, M> Problem<T, M> for MatrixSensing<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = FixedRankPoint<T, B>, TangentVector = FixedRankTangent<T, B>>,
{
	type Workspace = MatrixSensingWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let m = proto_point.u.nrows();
		let n = proto_point.v.nrows();
		let rank = VectorView::len(&proto_point.s);
		MatrixSensingWorkspace {
			egrad: B::Matrix::zeros(m, n),
			rv: B::Matrix::zeros(m, rank),
			ut_rv: B::Matrix::zeros(rank, rank),
			u_proj: B::Matrix::zeros(m, rank),
			ut_e: B::Matrix::zeros(rank, n),
			vt_proj: B::Matrix::zeros(rank, n),
			_phantom: PhantomData,
		}
	}

	fn cost(
		&self,
		point: &M::Point,
		_ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let half = <T as Scalar>::from_f64(0.5);
		let mut cost = T::zero();
		for (a, &b) in self.measurements.iter().zip(&self.observations) {
			let r = self.measure(a, point) - b;
			cost = cost + r * r;
		}
		half * cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		let rank = VectorView::len(&point.s);

		ws.egrad.fill(T::zero());
		for (a, &b) in self.measurements.iter().zip(&self.observations) {
			let r = self.measure(a, point) - b;
			ws.egrad.mat_axpy(r, a, T::one());
		}

		ws.rv
			.gemm(T::one(), ws.egrad.as_view(), point.v.as_view(), T::zero());
		ws.ut_rv
			.gemm_at(T::one(), point.u.as_view(), ws.rv.as_view(), T::zero());
		for k in 0..rank {
			*result.s_dot.get_mut(k) = ws.ut_rv.get(k, k);
		}

		ws.u_proj
			.gemm(T::one(), point.u.as_view(), ws.ut_rv.as_view(), T::zero());
		ws.rv.sub_assign(&ws.u_proj);
		result.u_perp_m.copy_from(&ws.rv);

		ws.ut_e
			.gemm_at(T::one(), point.u.as_view(), ws.egrad.as_view(), T::zero());
		ws.vt_proj
			.gemm_bt(T::one(), ws.ut_rv.as_view(), point.v.as_view(), T::zero());
		ws.ut_e.sub_assign(&ws.vt_proj);
		result.v_perp_n.copy_from(&ws.ut_e);

		let _ = manifold;
		let _ = manifold_ws;
	}
}
