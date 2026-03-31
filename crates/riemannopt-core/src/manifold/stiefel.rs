//! # Stiefel Manifold St(n,p)
//!
//! The Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : XᵀX = Iₚ} is the set of all
//! n×p matrices with orthonormal columns.

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use crate::{
	linalg::{DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::Manifold,
	types::Scalar,
};

/// The Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : XᵀX = Iₚ}.
#[derive(Clone)]
pub struct Stiefel<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	n: usize,
	p: usize,
	_phantom: PhantomData<(T, B)>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for Stiefel<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Stiefel St({}, {})", self.n, self.p)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Stiefel<T, B> {
	#[inline]
	pub fn new(n: usize, p: usize) -> Self {
		assert!(p > 0, "Stiefel manifold requires p ≥ 1");
		assert!(n >= p, "Stiefel manifold St(n,p) requires n ≥ p");
		Self {
			n,
			p,
			_phantom: PhantomData,
		}
	}

	#[inline]
	pub fn rows(&self) -> usize {
		self.n
	}

	#[inline]
	pub fn cols(&self) -> usize {
		self.p
	}

	#[inline]
	fn symmetrize_in_place(mat: &mut B::Matrix, size: usize) {
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..size {
			for j in i + 1..size {
				let avg = half * (mat.get(i, j) + mat.get(j, i));
				*mat.get_mut(i, j) = avg;
				*mat.get_mut(j, i) = avg;
			}
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Fix QR sign ambiguity: flip column j of Q if R_{jj} < 0.
#[inline]
fn fix_qr_signs<T: Scalar + Float, B: LinAlgBackend<T>>(
	q: &mut B::Matrix,
	r: &B::Matrix,
	n: usize,
	p: usize,
) {
	for j in 0..p.min(r.ncols()) {
		if r.get(j, j) < T::zero() {
			for i in 0..n {
				*q.get_mut(i, j) = T::zero() - q.get(i, j);
			}
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Workspace
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated buffers for Stiefel operations.
pub struct StiefelWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// p×p buffer (receives R from QR, also XᵀZ products)
	pub pp_mat: B::Matrix,
	/// p×p secondary buffer
	pub pp_mat2: B::Matrix,
	/// n×p buffer (QR input — destroyed by retract)
	pub np_mat: B::Matrix,
	/// Householder factor buffer (blocksize × p)
	pub qr_h: B::Matrix,
	/// Aligned scratch bytes (shared across decompositions)
	pub decomp_scratch: <B::Matrix as DecompositionOps<T>>::ScratchBuffer,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for StiefelWorkspace<T, B> {
	fn default() -> Self {
		Self {
			pp_mat: B::Matrix::zeros(0, 0),
			pp_mat2: B::Matrix::zeros(0, 0),
			np_mat: B::Matrix::zeros(0, 0),
			qr_h: B::Matrix::zeros(0, 0),
			decomp_scratch: B::Matrix::create_qr_scratch(0, 0),
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold impl
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Manifold<T> for Stiefel<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Matrix;
	type TangentVector = B::Matrix;
	type Workspace = StiefelWorkspace<T, B>;

	#[inline]
	fn create_workspace(&self, _proto_point: &Self::Point) -> Self::Workspace {
		let (bs, _) = B::Matrix::qr_h_factor_shape(self.n, self.p);
		Self::Workspace {
			pp_mat: B::Matrix::zeros(self.p, self.p),
			pp_mat2: B::Matrix::zeros(self.p, self.p),
			np_mat: B::Matrix::zeros(self.n, self.p),
			qr_h: B::Matrix::zeros(bs, self.p),
			decomp_scratch: B::Matrix::create_qr_scratch(self.n, self.p),
		}
	}

	#[inline]
	fn name(&self) -> &str {
		"Stiefel"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.n * self.p - self.p * (self.p + 1) / 2
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if point.nrows() != self.n || point.ncols() != self.p {
			return false;
		}
		let mut xtx = B::Matrix::zeros(self.p, self.p);
		xtx.gemm_at(T::one(), point.as_view(), point.as_view(), T::zero());
		let mut err_sq = T::zero();
		for i in 0..self.p {
			for j in 0..self.p {
				let diff = xtx.get(i, j) - if i == j { T::one() } else { T::zero() };
				err_sq = err_sq + diff * diff;
			}
		}
		<T as Float>::sqrt(err_sq) <= tol
	}

	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		if vector.nrows() != self.n || vector.ncols() != self.p {
			return false;
		}
		let mut xtz = B::Matrix::zeros(self.p, self.p);
		xtz.gemm_at(T::one(), point.as_view(), vector.as_view(), T::zero());
		let mut skew_sq = T::zero();
		for i in 0..self.p {
			for j in 0..self.p {
				let s = xtz.get(i, j) + xtz.get(j, i);
				skew_sq = skew_sq + s * s;
			}
		}
		<T as Float>::sqrt(skew_sq) <= tol
	}

	/// Closest-point projection via SVD: X = UΣVᵀ → UVᵀ.
	///
	/// Cold path — allocates temporary SVD buffers.
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let k = self.p;
		let mut u = B::Matrix::zeros(self.n, k);
		let mut s = B::Vector::zeros(k);
		let mut vt = B::Matrix::zeros(k, k);

		point.svd(&mut u, &mut s, &mut vt);

		// result = U Vᵀ  (polar factor, ignores Σ)
		result.gemm(T::one(), u.as_view(), vt.as_view(), T::zero());
	}

	#[inline]
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// P_X(Z) = Z − X sym(XᵀZ)
		ws.pp_mat
			.gemm_at(T::one(), point.as_view(), vector.as_view(), T::zero());
		Self::symmetrize_in_place(&mut ws.pp_mat, self.p);
		result.copy_from(vector);
		result.gemm(-T::one(), point.as_view(), ws.pp_mat.as_view(), T::one());
	}

	#[inline]
	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> T {
		u.frobenius_dot(v)
	}

	#[inline]
	fn norm(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> T {
		<T as Float>::sqrt(self.inner_product(point, vector, vector, ws))
	}

	/// QR retraction: R_X(Z) = qf(X + Z).  **Zero allocation.**
	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		// 1. np_mat = X + Z
		ws.np_mat.copy_from(point);
		ws.np_mat.add_assign(tangent);

		// 2. QR: np_mat destroyed → result = Q (n×p), pp_mat = R (p×p)
		ws.np_mat
			.qr(result, &mut ws.pp_mat, &mut ws.qr_h, &mut ws.decomp_scratch);

		// 3. Fix sign ambiguity
		fix_qr_signs::<T, B>(result, &ws.pp_mat, self.n, self.p);
	}

	#[inline]
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// log_X(Y) ≈ P_X(Y − X)
		result.copy_from(other);
		result.sub_assign(point);

		ws.pp_mat
			.gemm_at(T::one(), point.as_view(), result.as_view(), T::zero());
		Self::symmetrize_in_place(&mut ws.pp_mat, self.p);
		result.gemm(-T::one(), point.as_view(), ws.pp_mat.as_view(), T::one());
	}

	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.project_tangent(point, euclidean_grad, result, ws);
	}

	#[inline]
	fn euclidean_to_riemannian_hessian(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		euclidean_hvp: &Self::TangentVector,
		tangent_vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Hess f(X)[ξ] = P_X(∇²f[ξ]) − ξ sym(Xᵀ∇f)
		self.project_tangent(point, euclidean_hvp, result, ws);
		ws.pp_mat.gemm_at(
			T::one(),
			point.as_view(),
			euclidean_grad.as_view(),
			T::zero(),
		);
		Self::symmetrize_in_place(&mut ws.pp_mat, self.p);
		result.gemm(
			-T::one(),
			tangent_vector.as_view(),
			ws.pp_mat.as_view(),
			T::one(),
		);
	}

	#[inline]
	fn parallel_transport(
		&self,
		_from: &Self::Point,
		to: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.project_tangent(to, vector, result, ws);
	}

	/// Cold path — allocates temporary QR buffers.
	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		let mut a: B::Matrix = MatrixOps::zeros(self.n, self.p);
		for i in 0..self.n {
			for j in 0..self.p {
				*a.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		let (bs, k) = B::Matrix::qr_h_factor_shape(self.n, self.p);
		let mut r = B::Matrix::zeros(self.p, self.p);
		let mut h = B::Matrix::zeros(bs, k);
		let mut scratch = B::Matrix::create_qr_scratch(self.n, self.p);

		a.qr(result, &mut r, &mut h, &mut scratch);
		fix_qr_signs::<T, B>(result, &r, self.n, self.p);
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for i in 0..self.n {
			for j in 0..self.p {
				*result.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		let z = result.clone();
		let mut ws = self.create_workspace(point);
		self.project_tangent(point, &z, result, &mut ws);

		let norm = self.norm(point, result, &mut ws);
		if norm > T::EPSILON {
			result.scale_mut(T::one() / norm);
		}
	}

	#[inline]
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let mut diff_sq = T::zero();
		for i in 0..self.n {
			for j in 0..self.p {
				let diff = x.get(i, j) - y.get(i, j);
				diff_sq = diff_sq + diff * diff;
			}
		}
		<T as Float>::sqrt(diff_sq)
	}

	#[inline]
	fn has_exact_exp_log(&self) -> bool {
		false
	}

	#[inline]
	fn is_flat(&self) -> bool {
		false
	}

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
		y.mat_axpy(alpha, x, T::one());
	}

	#[inline]
	fn allocate_point(&self) -> Self::Point {
		B::Matrix::zeros(self.n, self.p)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		B::Matrix::zeros(self.n, self.p)
	}
}
