//! # Stiefel Manifold St(n,p)
//!
//! The Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = I_p} is the set of all
//! n×p matrices with orthonormal columns. It generalizes both the sphere (p=1)
//! and the orthogonal group (n=p), making it fundamental for problems involving
//! orthogonality constraints.
//!
//! ## Mathematical Definition
//!
//! ```text
//! St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = I_p}
//! ```

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use crate::{
	linalg::{DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, VectorView},
	manifold::Manifold,
	types::Scalar,
};

/// The Stiefel manifold St(n,p) = {X ∈ ℝⁿˣᵖ : X^T X = I_p}.
///
/// Generic over scalar type `T` and linear algebra backend `B`.
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
	/// Creates a new Stiefel manifold St(n,p).
	///
	/// # Panics
	///
	/// Panics if `p == 0` or `n < p`.
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

	/// Helper function to symmetrize a square matrix in-place: A = (A + A^T) / 2
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
//  Workspace Definition (Zero-Allocation Strategy)
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated memory buffers for Stiefel operations.
pub struct StiefelWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// A small p×p buffer for X^T Z, skew-symmetric factors, etc.
	pub pp_mat: B::Matrix,
	/// A secondary p×p buffer for additional matrix ops
	pub pp_mat2: B::Matrix,
	/// A buffer for X + Z (size n×p)
	pub np_mat: B::Matrix,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for StiefelWorkspace<T, B> {
	fn default() -> Self {
		Self {
			pp_mat: B::Matrix::zeros(0, 0),
			pp_mat2: B::Matrix::zeros(0, 0),
			np_mat: B::Matrix::zeros(0, 0),
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold trait implementation
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
		Self::Workspace {
			pp_mat: B::Matrix::zeros(self.p, self.p),
			pp_mat2: B::Matrix::zeros(self.p, self.p),
			np_mat: B::Matrix::zeros(self.n, self.p),
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

	#[inline]
	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if point.nrows() != self.n || point.ncols() != self.p {
			return false;
		}

		// Check X^T X = I_p
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

	#[inline]
	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		if vector.nrows() != self.n || vector.ncols() != self.p {
			return false;
		}

		// Skew-symmetry constraint: X^T Z + Z^T X = 0
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

	#[inline]
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Use SVD for exact closest-point projection: X = UΣV^T → UV^T
		let svd = point.svd();
		if let (Some(u), Some(vt)) = (svd.u, svd.vt) {
			let u_trunc = if u.ncols() > self.p {
				u.columns(0, self.p)
			} else {
				u.as_view()
			};
			let vt_trunc = if vt.nrows() > self.p {
				vt.rows(0, self.p)
			} else {
				vt.as_view()
			};
			result.gemm(T::one(), u_trunc, vt_trunc, T::zero());
		} else {
			// Fallback to QR
			let qr = point.qr();
			let q = qr.q();
			if q.ncols() > self.p {
				let q_trunc = q.columns_to_owned(0, self.p);
				result.copy_from(&q_trunc);
			} else {
				result.copy_from(&q);
			}
		}
	}

	#[inline]
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Stiefel projection: P_X(Z) = Z - X · sym(X^T Z)

		// 1. ws.pp_mat = X^T Z
		ws.pp_mat
			.gemm_at(T::one(), point.as_view(), vector.as_view(), T::zero());

		// 2. sym(X^T Z) in-place
		Self::symmetrize_in_place(&mut ws.pp_mat, self.p);

		// 3. result = Z - X * ws.pp_mat
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
		// Canonical metric: tr(U^T V) = ⟨U, V⟩_F
		let mut inner = T::zero();
		for j in 0..self.p {
			inner = inner + MatrixView::column(u, j).dot(&MatrixView::column(v, j));
		}
		inner
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

	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		// QR retraction: R_X(Z) = qf(X + Z)
		ws.np_mat.copy_from(point);
		ws.np_mat.add_assign(tangent);

		let qr = ws.np_mat.qr();
		let q = qr.q();

		if q.ncols() > self.p {
			let q_trunc = q.columns_to_owned(0, self.p);
			result.copy_from(&q_trunc);
		} else {
			result.copy_from(&q);
		}

		// Fix signs to ensure continuity (R has positive diagonal)
		let r = qr.r();
		for j in 0..self.p.min(r.ncols()) {
			if r.get(j, j) < T::zero() {
				for i in 0..self.n {
					*result.get_mut(i, j) = T::zero() - result.get(i, j);
				}
			}
		}
	}

	#[inline]
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// Approximate inverse retraction: log_X(Y) ≈ P_X(Y - X)
		result.copy_from(other);
		result.sub_assign(point);

		ws.pp_mat
			.gemm_at(T::one(), point.as_view(), result.as_view(), T::zero());
		Self::symmetrize_in_place(&mut ws.pp_mat, self.p);
	}

	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		euclidean_grad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		// grad f = P_X(∇f) = ∇f - X sym(X^T ∇f)
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
		// Weingarten Map for Stiefel:
		// Hess f(X)[ξ] = P_X(∇²f[ξ]) - ξ · sym(X^T ∇f)

		// 1. result = P_X(∇²f[ξ])
		self.project_tangent(point, euclidean_hvp, result, ws);

		// 2. ws.pp_mat = X^T ∇f
		ws.pp_mat.gemm_at(
			T::one(),
			point.as_view(),
			euclidean_grad.as_view(),
			T::zero(),
		);

		// 3. Symmetrize in place: ws.pp_mat = sym(X^T ∇f)
		Self::symmetrize_in_place(&mut ws.pp_mat, self.p);

		// 4. result -= ξ * ws.pp_mat
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
		// Projection-based vector transport: τ_{X→Y}(V) = P_Y(V)
		self.project_tangent(to, vector, result, ws);
	}

	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		let mut a: B::Matrix = MatrixOps::zeros(self.n, self.p);
		for i in 0..self.n {
			for j in 0..self.p {
				*a.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		let qr = a.qr();
		let q = qr.q();

		if q.ncols() > self.p {
			let q_trunc = q.columns_to_owned(0, self.p);
			result.copy_from(&q_trunc);
		} else {
			result.copy_from(&q);
		}
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for i in 0..self.n {
			for j in 0..self.p {
				*result.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		let mut ws = self.create_workspace(point);
		let z = result.clone();
		self.project_tangent(point, &z, result, &mut ws);

		let norm = self.norm(point, result, &mut ws);
		if norm > T::EPSILON {
			result.scale_mut(T::one() / norm);
		}
	}

	#[inline]
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		// Approximation using Frobenius distance: ||X - Y||_F
		// (Exact Stiefel distance requires complex matrix logarithms)
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

	// ════════════════════════════════════════════════════════════════════════
	// Matrix ops
	// ════════════════════════════════════════════════════════════════════════

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
