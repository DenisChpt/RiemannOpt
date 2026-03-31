//! # Grassmann Manifold Gr(n,p)
//!
//! The Grassmann manifold Gr(n,p) is the space of all p-dimensional linear
//! subspaces of ℝⁿ.
//!
//! ## Retraction
//!
//! Uses QR retraction (zero allocation on the hot path).
//! The polar retraction (SVD) from the previous version is replaced by QR
//! which is cheaper and equally valid for Grassmann — both are first-order.

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use crate::{
	linalg::{DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::Manifold,
	types::Scalar,
};

/// The Grassmann manifold Gr(n,p) of p-dimensional subspaces in ℝⁿ.
#[derive(Clone)]
pub struct Grassmann<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	n: usize,
	p: usize,
	_phantom: PhantomData<(T, B)>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for Grassmann<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "Grassmann Gr({}, {})", self.n, self.p)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> Grassmann<T, B> {
	pub fn new(n: usize, p: usize) -> Self {
		assert!(p > 0, "Grassmann manifold requires p > 0");
		assert!(p < n, "Grassmann manifold Gr(n,p) requires p < n");
		Self {
			n,
			p,
			_phantom: PhantomData,
		}
	}

	#[inline]
	pub fn ambient_dim(&self) -> usize {
		self.n
	}

	#[inline]
	pub fn subspace_dim(&self) -> usize {
		self.p
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Workspace
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated buffers for Grassmann operations.
pub struct GrassmannWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// p×p buffer (receives R from QR, also used for YᵀZ products)
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

impl<T: Scalar, B: LinAlgBackend<T>> Default for GrassmannWorkspace<T, B> {
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
//  Helpers
// ════════════════════════════════════════════════════════════════════════════

/// Fix sign ambiguity of QR: flip column j of Q if R_{jj} < 0.
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
//  Manifold impl
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Manifold<T> for Grassmann<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Matrix;
	type TangentVector = B::Matrix;
	type Workspace = GrassmannWorkspace<T, B>;

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
		"Grassmann"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.p * (self.n - self.p)
	}

	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if point.nrows() != self.n || point.ncols() != self.p {
			return false;
		}
		// Cold path — local allocation OK.
		let mut yty = B::Matrix::zeros(self.p, self.p);
		yty.gemm_at(T::one(), point.as_view(), point.as_view(), T::zero());
		let mut err_sq = T::zero();
		for i in 0..self.p {
			for j in 0..self.p {
				let diff = yty.get(i, j) - if i == j { T::one() } else { T::zero() };
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
		let mut ytz = B::Matrix::zeros(self.p, self.p);
		ytz.gemm_at(T::one(), point.as_view(), vector.as_view(), T::zero());
		ytz.norm() <= tol
	}

	/// Cold path — allocates temporary QR buffers.
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let (bs, k) = B::Matrix::qr_h_factor_shape(self.n, self.p);
		let mut tmp = point.clone();
		let mut r = B::Matrix::zeros(self.p, self.p);
		let mut h = B::Matrix::zeros(bs, k);
		let mut scratch = B::Matrix::create_qr_scratch(self.n, self.p);

		tmp.qr(result, &mut r, &mut h, &mut scratch);
		fix_qr_signs::<T, B>(result, &r, self.n, self.p);
	}

	#[inline]
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Horizontal projection: result = Z - Y(YᵀZ)
		ws.pp_mat
			.gemm_at(T::one(), point.as_view(), vector.as_view(), T::zero());
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

	/// QR retraction: R_Y(Z) = qf(Y + Z).  **Zero allocation.**
	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		// 1. np_mat = Y + Z
		ws.np_mat.copy_from(point);
		ws.np_mat.add_assign(tangent);

		// 2. QR: np_mat destroyed → result = Q, pp_mat = R
		ws.np_mat
			.qr(result, &mut ws.pp_mat, &mut ws.qr_h, &mut ws.decomp_scratch);

		// 3. Fix sign ambiguity for continuity
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
		// log_Y(Ỹ) ≈ Πₕ(Ỹ − Y)
		result.copy_from(other);
		result.sub_assign(point);
		ws.pp_mat
			.gemm_at(T::one(), point.as_view(), result.as_view(), T::zero());
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
		// rhess = project(ehess) − ξ (Yᵀ egrad)
		self.project_tangent(point, euclidean_hvp, result, ws);
		ws.pp_mat.gemm_at(
			T::one(),
			point.as_view(),
			euclidean_grad.as_view(),
			T::zero(),
		);
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

		// Cold path — local workspace + clone OK.
		let z = result.clone();
		let mut ws = self.create_workspace(point);
		self.project_tangent(point, &z, result, &mut ws);

		let norm = self.norm(point, result, &mut ws);
		if norm > T::EPSILON {
			result.scale_mut(T::one() / norm);
		}
	}

	/// Geodesic distance via principal angles.
	///
	/// Not on the hot path — allocates SVD temporaries for the p×p product.
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let p = self.p;

		// YₜᵀY₂ (p×p)
		let mut y1ty2 = B::Matrix::zeros(p, p);
		y1ty2.gemm_at(T::one(), x.as_view(), y.as_view(), T::zero());

		// SVD of p×p matrix — cold path, allocate temps.
		let mut u = B::Matrix::zeros(p, p);
		let mut s = B::Vector::zeros(p);
		let mut vt = B::Matrix::zeros(p, p);
		y1ty2.svd(&mut u, &mut s, &mut vt);

		// d² = Σ arccos²(σᵢ)
		let mut dist_sq = T::zero();
		for i in 0..p {
			let sigma = <T as Float>::max(<T as Float>::min(s.get(i), T::one()), -T::one());
			let theta = <T as Float>::acos(sigma);
			dist_sq = dist_sq + theta * theta;
		}
		<T as Float>::sqrt(dist_sq)
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
