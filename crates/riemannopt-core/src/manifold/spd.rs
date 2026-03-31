//! # Symmetric Positive Definite Manifold S⁺⁺(n)
//!
//! The manifold S⁺⁺(n) of n×n symmetric positive definite (SPD) matrices.

use num_traits::Float;
use rand_distr::{Distribution, StandardNormal};
use std::fmt::{self, Debug};
use std::marker::PhantomData;

use crate::{
	linalg::{DecompositionOps, LinAlgBackend, MatrixOps, MatrixView, VectorOps, VectorView},
	manifold::Manifold,
	types::Scalar,
};

// ════════════════════════════════════════════════════════════════════════════
//  Structs & Configuration
// ════════════════════════════════════════════════════════════════════════════

/// Available Riemannian metrics on the SPD manifold.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SPDMetric {
	/// Affine-invariant metric (default)
	AffineInvariant,
	/// Log-Euclidean metric
	LogEuclidean,
	/// Bures-Wasserstein metric
	BuresWasserstein,
}

/// The manifold S⁺⁺(n) of symmetric positive definite matrices.
#[derive(Clone)]
pub struct SPD<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	n: usize,
	min_eigenvalue: T,
	metric: SPDMetric,
	_phantom: PhantomData<(T, B)>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for SPD<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "SPD S⁺⁺({}) with {:?} metric", self.n, self.metric)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> SPD<T, B> {
	pub fn new(n: usize) -> Self {
		assert!(n > 0, "SPD manifold requires n ≥ 1");
		Self {
			n,
			min_eigenvalue: <T as Scalar>::from_f64(1e-12),
			metric: SPDMetric::AffineInvariant,
			_phantom: PhantomData,
		}
	}

	pub fn with_metric(n: usize, metric: SPDMetric) -> Self {
		assert!(n > 0, "SPD manifold requires n ≥ 1");
		Self {
			n,
			min_eigenvalue: <T as Scalar>::from_f64(1e-12),
			metric,
			_phantom: PhantomData,
		}
	}

	#[inline]
	pub fn matrix_dim(&self) -> usize {
		self.n
	}

	#[inline]
	pub fn metric_type(&self) -> SPDMetric {
		self.metric
	}

	/// Symmetrize a matrix in-place: A ← (A + Aᵀ)/2.
	#[inline]
	fn symmetrize(mat: &mut B::Matrix, n: usize) {
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..n {
			for j in i + 1..n {
				let avg = half * (mat.get(i, j) + mat.get(j, i));
				*mat.get_mut(i, j) = avg;
				*mat.get_mut(j, i) = avg;
			}
		}
	}

	/// Clamp eigenvalues to `min_eigenvalue` then reconstruct Q diag(λ) Qᵀ.
	///
	/// - `eigenvalues`: mutated in-place (clamped)
	/// - `eigenvectors`: read-only basis Q
	/// - `buf`: scratch for Q·diag(λ)
	/// - `out`: receives Q diag(λ) Qᵀ (symmetrized)
	#[inline]
	fn clamp_and_reconstruct(
		&self,
		eigenvalues: &mut B::Vector,
		eigenvectors: &B::Matrix,
		buf: &mut B::Matrix,
		out: &mut B::Matrix,
	) {
		let threshold = self.min_eigenvalue + <T as Scalar>::from_f64(1e-8);
		for i in 0..self.n {
			if eigenvalues.get(i) <= self.min_eigenvalue {
				*eigenvalues.get_mut(i) = threshold;
			}
		}
		buf.scale_columns(eigenvectors, eigenvalues);
		out.gemm_bt(T::one(), buf.as_view(), eigenvectors.as_view(), T::zero());
		Self::symmetrize(out, self.n);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Workspace
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated workspace for SPD manifold operations.
///
/// Buffer roles:
/// - `buf_a`: P⁻¹U (inner_product), Q·diag(λ) (retract eigen)
/// - `buf_b`: P⁻¹V (inner_product), eigenvectors (retract eigen)
/// - `buf_c`: (P⁻¹U)(P⁻¹V) (inner_product), inverse fallback
/// - `eigenvalues`: eigenvalues output
pub struct SPDWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	pub buf_a: B::Matrix,
	pub buf_b: B::Matrix,
	pub buf_c: B::Matrix,
	pub eigenvalues: B::Vector,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for SPDWorkspace<T, B> {
	fn default() -> Self {
		Self {
			buf_a: B::Matrix::zeros(0, 0),
			buf_b: B::Matrix::zeros(0, 0),
			buf_c: B::Matrix::zeros(0, 0),
			eigenvalues: B::Vector::zeros(0),
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold impl
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Manifold<T> for SPD<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Matrix;
	type TangentVector = B::Matrix;
	type Workspace = SPDWorkspace<T, B>;

	#[inline]
	fn create_workspace(&self, _proto_point: &Self::Point) -> Self::Workspace {
		Self::Workspace {
			buf_a: B::Matrix::zeros(self.n, self.n),
			buf_b: B::Matrix::zeros(self.n, self.n),
			buf_c: B::Matrix::zeros(self.n, self.n),
			eigenvalues: B::Vector::zeros(self.n),
		}
	}

	#[inline]
	fn name(&self) -> &str {
		"SPD"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.n * (self.n + 1) / 2
	}

	/// Cold path — allocates temporary eigen buffers.
	fn is_point_on_manifold(&self, point: &Self::Point, tol: T) -> bool {
		if point.nrows() != self.n || point.ncols() != self.n {
			return false;
		}

		let mut sym_err = T::zero();
		for i in 0..self.n {
			for j in 0..self.n {
				let diff = point.get(i, j) - point.get(j, i);
				sym_err += diff * diff;
			}
		}
		if sym_err.sqrt() > tol {
			return false;
		}

		let mut eigenvalues = B::Vector::zeros(self.n);
		let mut eigenvectors = B::Matrix::zeros(self.n, self.n);
		point.symmetric_eigen(&mut eigenvalues, &mut eigenvectors);

		let mut min_eval = <T as Scalar>::from_f64(f64::INFINITY);
		for i in 0..self.n {
			let ev = eigenvalues.get(i);
			if ev < min_eval {
				min_eval = ev;
			}
		}
		min_eval > self.min_eigenvalue
	}

	fn is_vector_in_tangent_space(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		if vector.nrows() != self.n || vector.ncols() != self.n {
			return false;
		}
		let mut sym_err = T::zero();
		for i in 0..self.n {
			for j in 0..self.n {
				let diff = vector.get(i, j) - vector.get(j, i);
				sym_err += diff * diff;
			}
		}
		sym_err.sqrt() <= tol
	}

	/// Cold path — allocates temporary buffers.
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Symmetrize
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..self.n {
			for j in i..self.n {
				let avg = half * (point.get(i, j) + point.get(j, i));
				*result.get_mut(i, j) = avg;
				*result.get_mut(j, i) = avg;
			}
		}

		// Eigen of the symmetrized result
		let mut eigenvalues = B::Vector::zeros(self.n);
		let mut eigenvectors = B::Matrix::zeros(self.n, self.n);
		result.symmetric_eigen(&mut eigenvalues, &mut eigenvectors);

		// Clamp & reconstruct
		let mut buf = B::Matrix::zeros(self.n, self.n);
		self.clamp_and_reconstruct(&mut eigenvalues, &eigenvectors, &mut buf, result);
	}

	#[inline]
	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut Self::Workspace,
	) {
		let half = <T as Scalar>::from_f64(0.5);
		for i in 0..self.n {
			for j in i..self.n {
				let avg = half * (vector.get(i, j) + vector.get(j, i));
				*result.get_mut(i, j) = avg;
				*result.get_mut(j, i) = avg;
			}
		}
	}

	#[inline]
	fn inner_product(
		&self,
		point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> T {
		match self.metric {
			SPDMetric::AffineInvariant => {
				// ⟨U,V⟩_P = tr(P⁻¹U P⁻¹V)
				if !point.cholesky_solve(u, &mut ws.buf_a) {
					point.inverse(&mut ws.buf_c);
					ws.buf_a
						.gemm(T::one(), ws.buf_c.as_view(), u.as_view(), T::zero());
				}
				if !point.cholesky_solve(v, &mut ws.buf_b) {
					point.inverse(&mut ws.buf_c);
					ws.buf_b
						.gemm(T::one(), ws.buf_c.as_view(), v.as_view(), T::zero());
				}
				ws.buf_c
					.gemm(T::one(), ws.buf_a.as_view(), ws.buf_b.as_view(), T::zero());
				ws.buf_c.trace()
			}
			SPDMetric::LogEuclidean => u.frobenius_dot(v),
			SPDMetric::BuresWasserstein => u.frobenius_dot(v),
		}
	}

	#[inline]
	fn norm(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		ws: &mut Self::Workspace,
	) -> T {
		self.inner_product(point, vector, vector, ws).sqrt()
	}

	/// **Zero allocation** for all metric variants.
	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		match self.metric {
			SPDMetric::AffineInvariant => {
				// R_P(V) = P + V + ½ V P⁻¹ V
				if !point.cholesky_solve(tangent, &mut ws.buf_a) {
					point.inverse(&mut ws.buf_c);
					ws.buf_a
						.gemm(T::one(), ws.buf_c.as_view(), tangent.as_view(), T::zero());
				}

				result.copy_from(point);
				result.add_assign(tangent);
				result.gemm(
					<T as Scalar>::from_f64(0.5),
					tangent.as_view(),
					ws.buf_a.as_view(),
					T::one(),
				);
				Self::symmetrize(result, self.n);
			}
			_ => {
				// Euclidean step + eigenvalue clamping
				result.copy_from(point);
				result.add_assign(tangent);
				Self::symmetrize(result, self.n);

				// Eigen: result read as &self → eigenvalues in ws, eigenvectors in buf_b
				result.symmetric_eigen(&mut ws.eigenvalues, &mut ws.buf_b);

				// Clamp & reconstruct: buf_a = Q·diag(λ), result = buf_a·Qᵀ
				self.clamp_and_reconstruct(&mut ws.eigenvalues, &ws.buf_b, &mut ws.buf_a, result);
			}
		}
	}

	#[inline]
	fn inverse_retract(
		&self,
		point: &Self::Point,
		other: &Self::Point,
		result: &mut Self::TangentVector,
		_ws: &mut Self::Workspace,
	) {
		result.copy_from(other);
		result.sub_assign(point);
		Self::symmetrize(result, self.n);
	}

	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		egrad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		match self.metric {
			SPDMetric::AffineInvariant => {
				// grad = P ∇f P
				ws.buf_a
					.gemm(T::one(), point.as_view(), egrad.as_view(), T::zero());
				result.gemm(T::one(), ws.buf_a.as_view(), point.as_view(), T::zero());
			}
			SPDMetric::LogEuclidean => {
				// grad = P ∇f + ∇f P
				result.gemm(T::one(), point.as_view(), egrad.as_view(), T::zero());
				result.gemm(T::one(), egrad.as_view(), point.as_view(), T::one());
			}
			SPDMetric::BuresWasserstein => {
				result.copy_from(egrad);
			}
		}
		Self::symmetrize(result, self.n);
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

	fn random_point(&self, result: &mut Self::Point) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		let mut a = B::Matrix::zeros(self.n, self.n);
		for i in 0..self.n {
			for j in 0..self.n {
				*a.get_mut(i, j) = <T as Scalar>::from_f64(normal.sample(&mut rng));
			}
		}

		result.gemm_at(T::one(), a.as_view(), a.as_view(), T::zero());
		for i in 0..self.n {
			*result.get_mut(i, i) = result.get(i, i) + self.min_eigenvalue;
		}
	}

	fn random_tangent(&self, _point: &Self::Point, result: &mut Self::TangentVector) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for i in 0..self.n {
			for j in i..self.n {
				let val = <T as Scalar>::from_f64(normal.sample(&mut rng));
				*result.get_mut(i, j) = val;
				if i != j {
					*result.get_mut(j, i) = val;
				}
			}
		}

		let norm = result.frobenius_dot(result).sqrt();
		if norm > T::EPSILON {
			result.scale_mut(T::one() / norm);
		}
	}

	#[inline]
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let mut diff_sq = T::zero();
		for i in 0..self.n {
			for j in 0..self.n {
				let diff = x.get(i, j) - y.get(i, j);
				diff_sq += diff * diff;
			}
		}
		diff_sq.sqrt()
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
		B::Matrix::zeros(self.n, self.n)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		B::Matrix::zeros(self.n, self.n)
	}
}
