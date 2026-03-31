//! Optimization problems on the Stiefel manifold St(n,p).
//!
//! # Problems
//!
//! - [`OrthogonalProcrustes`] — Optimal orthogonal alignment
//! - [`OrthogonalICA`] — Independent Component Analysis with orthogonality
//! - [`OrderedBrockett`] — Brockett cost with eigenvalue ordering

use std::marker::PhantomData;

use crate::{
	linalg::{LinAlgBackend, MatrixOps, MatrixView, RealScalar, VectorOps, VectorView},
	manifold::Manifold,
	problem::Problem,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Orthogonal Procrustes
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct OrthogonalProcrustes<T: Scalar, B: LinAlgBackend<T>> {
	pub a: B::Matrix,
	pub b: B::Matrix,
	ata: B::Matrix,
	atb: B::Matrix,
}

impl<T: Scalar, B: LinAlgBackend<T>> OrthogonalProcrustes<T, B> {
	pub fn new(a: B::Matrix, b: B::Matrix) -> Self {
		debug_assert_eq!(
			a.nrows(),
			b.nrows(),
			"A and B must have same number of rows"
		);
		let mut ata = B::Matrix::zeros(a.ncols(), a.ncols());
		ata.gemm_at(T::one(), a.as_view(), a.as_view(), T::zero());
		let mut atb = B::Matrix::zeros(a.ncols(), b.ncols());
		atb.gemm_at(T::one(), a.as_view(), b.as_view(), T::zero());
		Self { a, b, ata, atb }
	}
}

pub struct ProcrustesWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	egrad: B::Matrix,
	ehvp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for ProcrustesWorkspace<T, B> {
	fn default() -> Self {
		Self {
			egrad: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for ProcrustesWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for ProcrustesWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M> Problem<T, M> for OrthogonalProcrustes<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = ProcrustesWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		ProcrustesWorkspace {
			egrad: B::Matrix::zeros(n, p),
			ehvp: B::Matrix::zeros(n, p),
			_phantom: PhantomData,
		}
	}

	/// **Zero allocation** — uses ws.egrad for AᵀAX.
	fn cost(
		&self,
		point: &M::Point,
		ws: &mut Self::Workspace,
		_manifold_ws: &mut M::Workspace,
	) -> T {
		let half = <T as Scalar>::from_f64(0.5);
		ws.egrad
			.gemm(T::one(), self.ata.as_view(), point.as_view(), T::zero());
		let quad = half * point.frobenius_dot(&ws.egrad);
		let lin = point.frobenius_dot(&self.atb);
		let b_norm_sq = half * self.b.frobenius_dot(&self.b);
		quad - lin + b_norm_sq
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
			.gemm(T::one(), self.ata.as_view(), point.as_view(), T::zero());
		ws.egrad.sub_assign(&self.atb);
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
			.gemm(T::one(), self.ata.as_view(), point.as_view(), T::zero());
		let quad = half * point.frobenius_dot(&ws.egrad);
		let lin = point.frobenius_dot(&self.atb);
		let b_norm_sq = half * self.b.frobenius_dot(&self.b);
		let cost = quad - lin + b_norm_sq;
		ws.egrad.sub_assign(&self.atb);
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
			.gemm(T::one(), self.ata.as_view(), vector.as_view(), T::zero());
		ws.egrad
			.gemm(T::one(), self.ata.as_view(), point.as_view(), T::zero());
		ws.egrad.sub_assign(&self.atb);
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
//  Orthogonal ICA
// ════════════════════════════════════════════════════════════════════════════

/// Contrast function for ICA.
#[derive(Debug, Clone, Copy)]
pub enum ICAContrast {
	LogCosh,
	Exp,
	Kurtosis,
}

impl ICAContrast {
	#[inline]
	pub fn g<T: Scalar>(self, u: T) -> T {
		match self {
			Self::LogCosh => u.cosh().ln(),
			Self::Exp => -(u * u * <T as Scalar>::from_f64(0.5)).exp(),
			Self::Kurtosis => u * u * u * u * <T as Scalar>::from_f64(0.25),
		}
	}

	#[inline]
	pub fn g_prime<T: Scalar>(self, u: T) -> T {
		match self {
			Self::LogCosh => u.tanh(),
			Self::Exp => u * (-(u * u) * <T as Scalar>::from_f64(0.5)).exp(),
			Self::Kurtosis => u * u * u,
		}
	}
}

#[derive(Debug, Clone)]
pub struct OrthogonalICA<T: Scalar, B: LinAlgBackend<T>> {
	pub data: B::Matrix,
	pub contrast: ICAContrast,
	inv_m: T,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> OrthogonalICA<T, B> {
	pub fn new(data: B::Matrix, contrast: ICAContrast) -> Self {
		let m = data.ncols();
		Self {
			data,
			contrast,
			inv_m: T::one() / <T as RealScalar>::from_usize(m),
			_phantom: PhantomData,
		}
	}
}

pub struct ICAWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	wtx: B::Matrix,
	egrad: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for ICAWorkspace<T, B> {
	fn default() -> Self {
		Self {
			wtx: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for ICAWorkspace<T, B> where B::Matrix: Send {}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for ICAWorkspace<T, B> where B::Matrix: Sync {}

impl<T, B, M> Problem<T, M> for OrthogonalICA<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = ICAWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let p = proto_point.ncols();
		let m = self.data.ncols();
		let n = proto_point.nrows();
		ICAWorkspace {
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
				total = total + self.contrast.g(ws.wtx.get(k, j));
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
		let p = point.ncols();
		let m = self.data.ncols();
		ws.wtx
			.gemm_at(T::one(), point.as_view(), self.data.as_view(), T::zero());
		let mut total = T::zero();
		for k in 0..p {
			for j in 0..m {
				let s = ws.wtx.get(k, j);
				total = total + self.contrast.g(s);
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

impl<T: Scalar, B: LinAlgBackend<T>> OrthogonalICA<T, B> {
	fn compute_egrad(&self, point: &B::Matrix, ws: &mut ICAWorkspace<T, B>) {
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
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Ordered Brockett Cost
// ════════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone)]
pub struct OrderedBrockett<T: Scalar, B: LinAlgBackend<T>> {
	pub a: B::Matrix,
	pub weights: B::Vector,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> OrderedBrockett<T, B> {
	pub fn new(a: B::Matrix, weights: B::Vector) -> Self {
		debug_assert_eq!(a.nrows(), a.ncols(), "A must be square");
		Self {
			a,
			weights,
			_phantom: PhantomData,
		}
	}

	pub fn with_default_weights(a: B::Matrix, p: usize) -> Self {
		let weights = B::Vector::from_fn(p, |i| <T as RealScalar>::from_usize(p - i));
		Self::new(a, weights)
	}
}

pub struct OrderedBrockettWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	ax: B::Matrix,
	egrad: B::Matrix,
	axi: B::Matrix,
	ehvp: B::Matrix,
	_phantom: PhantomData<T>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for OrderedBrockettWorkspace<T, B> {
	fn default() -> Self {
		Self {
			ax: B::Matrix::zeros(0, 0),
			egrad: B::Matrix::zeros(0, 0),
			axi: B::Matrix::zeros(0, 0),
			ehvp: B::Matrix::zeros(0, 0),
			_phantom: PhantomData,
		}
	}
}

unsafe impl<T: Scalar, B: LinAlgBackend<T>> Send for OrderedBrockettWorkspace<T, B> where
	B::Matrix: Send
{
}
unsafe impl<T: Scalar, B: LinAlgBackend<T>> Sync for OrderedBrockettWorkspace<T, B> where
	B::Matrix: Sync
{
}

impl<T, B, M> Problem<T, M> for OrderedBrockett<T, B>
where
	T: Scalar,
	B: LinAlgBackend<T>,
	M: Manifold<T, Point = B::Matrix, TangentVector = B::Matrix>,
{
	type Workspace = OrderedBrockettWorkspace<T, B>;

	fn create_workspace(&self, _manifold: &M, proto_point: &M::Point) -> Self::Workspace {
		let n = proto_point.nrows();
		let p = proto_point.ncols();
		OrderedBrockettWorkspace {
			ax: B::Matrix::zeros(n, p),
			egrad: B::Matrix::zeros(n, p),
			axi: B::Matrix::zeros(n, p),
			ehvp: B::Matrix::zeros(n, p),
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
		let p = point.ncols();
		ws.ax
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		let mut cost = T::zero();
		for k in 0..p {
			cost = cost - self.weights.get(k) * point.column_dot(k, &ws.ax, k);
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
		ws.ax
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		ws.egrad.scale_columns(&ws.ax, &self.weights);
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
		let p = point.ncols();
		ws.ax
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		let mut cost = T::zero();
		for k in 0..p {
			cost = cost - self.weights.get(k) * point.column_dot(k, &ws.ax, k);
		}
		ws.egrad.scale_columns(&ws.ax, &self.weights);
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
		ws.ax
			.gemm(T::one(), self.a.as_view(), point.as_view(), T::zero());
		ws.egrad.scale_columns(&ws.ax, &self.weights);
		ws.egrad.scale_mut(-two);
		ws.axi
			.gemm(T::one(), self.a.as_view(), vector.as_view(), T::zero());
		ws.ehvp.scale_columns(&ws.axi, &self.weights);
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
