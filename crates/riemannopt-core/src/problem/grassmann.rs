//! Optimization problems on the Grassmann manifold Gr(n,p).
//!
//! # Problems
//!
//! - [`BrockettCost`] — PCA / Eigenspace computation
//! - [`RobustPCA`] — Principal Component Pursuit (low-rank + sparse)

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Brockett Cost (PCA / Eigenspace)
// ════════════════════════════════════════════════════════════════════════════

/// Brockett cost function for eigenspace computation on Gr(n,p).
///
/// f(Y) = −tr(YᵀAY)
#[derive(Debug, Clone)]
pub struct BrockettCost<T: Scalar, B: LinAlgBackend<T>> {
	pub a: B::Matrix,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> BrockettCost<T, B> {
	pub fn new(a: B::Matrix) -> Self {
		debug_assert_eq!(a.nrows(), a.ncols(), "A must be square");
		Self {
			a,
			_phantom: PhantomData,
		}
	}
}

pub struct BrockettWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	ay: B::Matrix,
	egrad: B::Matrix,
	ehvp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for BrockettWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ay: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for BrockettWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for BrockettWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M> Problem<T, M> for BrockettCost<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = BrockettWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		BrockettWorkspace {
			ay: B::Matrix::zeros(n, p),
			egrad: B::Matrix::zeros(n, p),
			ehvp: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — uses ws.ay for A·Y.
	#[inline]
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		ws.ay
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		-point.frobenius_dot(&ws.ay)
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		ws.ay
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		ws.egrad.copy_from(&ws.ay);
		ws.egrad.scale_mut(<T as Scalar>::from_f64(-2.0));
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
		ws.ay
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		let cost = -point.frobenius_dot(&ws.ay);
		ws.egrad.copy_from(&ws.ay);
		ws.egrad.scale_mut(<T as Scalar>::from_f64(-2.0));
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
		let two = <T as Scalar>::from_f64(2.0);
		ws.ay
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		ws.egrad.copy_from(&ws.ay);
		ws.egrad.scale_mut(-two);
		ws.ehvp
			.gemm(T::one(), self.a.as_view(), vector.as_view(), T::zero());
		ws.ehvp.scale_mut(-two);
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
//  Robust PCA
// ════════════════════════════════════════════════════════════════════════════

/// Robust PCA / Principal Component Pursuit on Gr(n,p).
///
/// f(Y) = ½ ‖D − YYᵀD‖² + μ Σ ψ_δ(R_{ij})
#[derive(Debug, Clone)]
pub struct RobustPCA<T: Scalar, B: LinAlgBackend<T>> {
	pub data: B::Matrix,
	pub mu: T,
	pub huber_delta: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> RobustPCA<T, B> {
	pub fn new(data: B::Matrix, mu: T, huber_delta: T) -> Self {
		debug_assert!(mu >= T::zero());
		debug_assert!(huber_delta > T::zero());
		Self {
			data,
			mu,
			huber_delta,
			_phantom: PhantomData,
		}
	}

	pub fn pca(data: B::Matrix) -> Self {
		Self::new(data, T::zero(), T::one())
	}
}

pub struct RobustPCAWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	ytd: B::Matrix,
	yytd: B::Matrix,
	residual: B::Matrix,
	egrad: B::Matrix,
	tmp_np: B::Matrix,
	wty: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for RobustPCAWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ytd: B::Matrix::zeros(0, 0),
			yytd: B::Matrix::zeros(0, 0),
			residual: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			tmp_np: B::Matrix::zeros(0, 0),
			wty: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for RobustPCAWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for RobustPCAWorkspace<T, B> where B::Matrix: Sync {}

#[inline]
fn huber<T: Scalar>(t: T, delta: T) -> T {
	let abs_t = t.abs();
	if abs_t <= delta {
		t * t / (delta + delta)
	} else {
		abs_t - delta * <T as Scalar>::from_f64(0.5)
	}
}

#[inline]
fn huber_deriv<T: Scalar>(t: T, delta: T) -> T {
	let abs_t = t.abs();
	if abs_t <= delta {
		t / delta
	} else if t > T::zero() {
		T::one()
	} else {
		-T::one()
	}
}

impl<T, B, M> Problem<T, M> for RobustPCA<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = RobustPCAWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		let m = self.data.ncols();
		RobustPCAWorkspace {
			ytd: B::Matrix::zeros(p, m),
			yytd: B::Matrix::zeros(n, m),
			residual: B::Matrix::zeros(n, m),
			egrad: B::Matrix::zeros(n, p),
			tmp_np: B::Matrix::zeros(n, p),
			wty: B::Matrix::zeros(m, p),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — uses ws.ytd, ws.yytd, ws.residual.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		self.compute_residual(point, ws);

		let n = self.data.nrows();
		let m = self.data.ncols();
		let half = <T as Scalar>::from_f64(0.5);
		let mut frobenius_sq = T::zero();
		let mut l1_cost = T::zero();

		let r_slice = ws.residual.as_slice();
		for k in 0..n * m {
			let rk = r_slice[k];
			frobenius_sq += rk * rk;
			if self.mu > T::zero() {
				l1_cost += huber(rk, self.huber_delta);
			}
		}

		half * frobenius_sq + self.mu * l1_cost
	}

	fn riemannian_gradient(
		&self,
		manifold: &M,
		point: &M::Point,
		result: &mut M::TangentVector,
		ws: &mut Self::Workspace,
		manifold_ws: &mut M::Workspace,
	) {
		self.compute_residual(point, ws);
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
		self.compute_residual(point, ws);

		let n = self.data.nrows();
		let m = self.data.ncols();
		let half = <T as Scalar>::from_f64(0.5);
		let mut frobenius_sq = T::zero();
		let mut l1_cost = T::zero();
		let r_slice = ws.residual.as_slice();
		for k in 0..n * m {
			let rk = r_slice[k];
			frobenius_sq += rk * rk;
			if self.mu > T::zero() {
				l1_cost += huber(rk, self.huber_delta);
			}
		}
		let cost = half * frobenius_sq + self.mu * l1_cost;

		self.compute_egrad(point, ws);
		manifold.euclidean_to_riemannian_gradient(point, &ws.egrad, gradient, manifold_ws);
		cost
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> RobustPCA<T, B> {
	fn compute_residual(&self, point: &B::Matrix, ws: &mut RobustPCAWorkspace<T, B>) {
		ws.ytd
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());
		ws.yytd
			.gemm(T::one(), point.as_view(), ws.ytd.as_view(), T::zero());
		ws.residual.copy_from(&self.data);
		ws.residual.sub_assign(&ws.yytd);
	}

	fn compute_egrad(&self, point: &B::Matrix, ws: &mut RobustPCAWorkspace<T, B>) {
		ws.egrad
			.gemm_bt(-T::one(), self.data.as_view(), ws.ytd.as_view(), T::zero());

		if self.mu > T::zero() {
			let n = self.data.nrows();
			let m = self.data.ncols();

			let r_slice = ws.residual.as_mut_slice();
			for k in 0..n * m {
				r_slice[k] = huber_deriv(r_slice[k], self.huber_delta);
			}

			ws.tmp_np
				.gemm_bt(-self.mu, ws.residual.as_view(), ws.ytd.as_view(), T::zero());
			ws.egrad.add_assign(&ws.tmp_np);

			ws.wty
				.gemm_at(T::one(), ws.residual.as_view(), point.as_view(), T::zero());
			ws.tmp_np
				.gemm(-self.mu, self.data.as_view(), ws.wty.as_view(), T::zero());
			ws.egrad.add_assign(&ws.tmp_np);
		}
	}
}
