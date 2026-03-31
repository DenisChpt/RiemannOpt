//! # Positive Semi-Definite Cone S⁺(n)
//!
//! The cone S⁺(n) of n×n symmetric positive semi-definite (PSD) matrices.
//! Points are stored as packed vectors of size n(n+1)/2. Off-diagonal elements
//! are scaled by √2 to preserve the Frobenius inner product.

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
//  Workspace
// ════════════════════════════════════════════════════════════════════════════

/// Pre-allocated workspace for PSD Cone operations.
pub struct PSDConeWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// n×n buffer — holds unpacked symmetric matrix, then receives Q Λ Qᵀ
	pub mat_a: B::Matrix,
	/// n×n buffer — intermediate Q diag(λ)
	pub mat_b: B::Matrix,
	/// n×n buffer — eigenvectors from symmetric_eigen
	pub eigenvectors: B::Matrix,
	/// n-vector — eigenvalues from symmetric_eigen
	pub eigenvalues: B::Vector,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for PSDConeWorkspace<T, B> {
	fn default() -> Self {
		Self {
			mat_a: B::Matrix::zeros(0, 0),
			mat_b: B::Matrix::zeros(0, 0),
			eigenvectors: B::Matrix::zeros(0, 0),
			eigenvalues: B::Vector::zeros(0),
		}
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  PSDCone struct
// ════════════════════════════════════════════════════════════════════════════

/// The positive semi-definite cone S⁺(n).
#[derive(Clone)]
pub struct PSDCone<T: Scalar = f64, B: LinAlgBackend<T> = crate::linalg::DefaultBackend> {
	n: usize,
	dim: usize,
	tolerance: T,
	strict: bool,
	_phantom: PhantomData<B>,
}

impl<T: Scalar, B: LinAlgBackend<T>> Debug for PSDCone<T, B> {
	fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
		write!(f, "PSDCone S⁺({})", self.n)
	}
}

impl<T: Scalar, B: LinAlgBackend<T>> PSDCone<T, B> {
	pub fn new(n: usize) -> Self {
		assert!(n > 0, "PSD cone requires n ≥ 1");
		Self {
			n,
			dim: n * (n + 1) / 2,
			tolerance: <T as Scalar>::from_f64(1e-10),
			strict: false,
			_phantom: PhantomData,
		}
	}

	pub fn with_parameters(n: usize, tolerance: T, strict: bool) -> Self {
		assert!(n > 0, "PSD cone requires n ≥ 1");
		assert!(
			tolerance > T::zero() && tolerance < T::one(),
			"Tolerance must be in (0, 1)"
		);
		Self {
			n,
			dim: n * (n + 1) / 2,
			tolerance,
			strict,
			_phantom: PhantomData,
		}
	}

	#[inline]
	pub fn matrix_size(&self) -> usize {
		self.n
	}

	#[inline]
	fn vec_to_mat(&self, vec: &B::Vector, mat: &mut B::Matrix) {
		let inv_sqrt2 = T::one() / <T as Scalar>::from_f64(std::f64::consts::SQRT_2);
		let mut idx = 0;
		for i in 0..self.n {
			for j in i..self.n {
				let val = vec.get(idx);
				if i == j {
					*mat.get_mut(i, i) = val;
				} else {
					let scaled = val * inv_sqrt2;
					*mat.get_mut(i, j) = scaled;
					*mat.get_mut(j, i) = scaled;
				}
				idx += 1;
			}
		}
	}

	#[inline]
	fn mat_to_vec(&self, mat: &B::Matrix, vec: &mut B::Vector) {
		let sqrt2 = <T as Scalar>::from_f64(std::f64::consts::SQRT_2);
		let mut idx = 0;
		for i in 0..self.n {
			for j in i..self.n {
				if i == j {
					*vec.get_mut(idx) = mat.get(i, i);
				} else {
					*vec.get_mut(idx) = mat.get(i, j) * sqrt2;
				}
				idx += 1;
			}
		}
	}

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

	/// Clamp eigenvalues and reconstruct Q max(0,Λ) Qᵀ into `mat_out`.
	///
	/// Uses `mat_tmp` as scratch for Q·diag(λ).
	#[inline]
	fn clamp_and_reconstruct(
		&self,
		eigenvalues: &mut B::Vector,
		eigenvectors: &B::Matrix,
		mat_tmp: &mut B::Matrix,
		mat_out: &mut B::Matrix,
	) {
		let threshold = if self.strict { T::EPSILON } else { T::zero() };
		for i in 0..self.n {
			if eigenvalues.get(i) < threshold {
				*eigenvalues.get_mut(i) = threshold;
			}
		}
		// mat_tmp = Q · diag(λ)
		mat_tmp.scale_columns(eigenvectors, eigenvalues);
		// mat_out = mat_tmp · Qᵀ
		mat_out.gemm_bt(
			T::one(),
			mat_tmp.as_view(),
			eigenvectors.as_view(),
			T::zero(),
		);
		Self::symmetrize(mat_out, self.n);
	}
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold impl
// ════════════════════════════════════════════════════════════════════════════

impl<T, B> Manifold<T> for PSDCone<T, B>
where
	T: Scalar + Float,
	B: LinAlgBackend<T>,
{
	type Point = B::Vector;
	type TangentVector = B::Vector;
	type Workspace = PSDConeWorkspace<T, B>;

	#[inline]
	fn create_workspace(&self, _proto: &Self::Point) -> Self::Workspace {
		PSDConeWorkspace {
			mat_a: B::Matrix::zeros(self.n, self.n),
			mat_b: B::Matrix::zeros(self.n, self.n),
			eigenvectors: B::Matrix::zeros(self.n, self.n),
			eigenvalues: B::Vector::zeros(self.n),
		}
	}

	#[inline]
	fn name(&self) -> &str {
		"PSDCone"
	}

	#[inline]
	fn dimension(&self) -> usize {
		self.dim
	}

	/// Cold path — allocates temporary eigen buffers.
	fn is_point_on_manifold(&self, point: &Self::Point, _tol: T) -> bool {
		if point.len() != self.dim {
			return false;
		}
		let mut mat = B::Matrix::zeros(self.n, self.n);
		self.vec_to_mat(point, &mut mat);

		let mut eigenvalues = B::Vector::zeros(self.n);
		let mut eigenvectors = B::Matrix::zeros(self.n, self.n);
		mat.symmetric_eigen(&mut eigenvalues, &mut eigenvectors);

		let threshold = if self.strict {
			self.tolerance
		} else {
			-self.tolerance
		};
		(0..self.n).all(|i| eigenvalues.get(i) >= threshold)
	}

	/// Cold path — allocates temporary eigen buffers.
	fn is_vector_in_tangent_space(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		tol: T,
	) -> bool {
		if vector.len() != self.dim || point.len() != self.dim {
			return false;
		}
		if self.strict {
			return true;
		}

		let mut mat_x = B::Matrix::zeros(self.n, self.n);
		self.vec_to_mat(point, &mut mat_x);

		let mut eigenvalues = B::Vector::zeros(self.n);
		let mut eigenvectors = B::Matrix::zeros(self.n, self.n);
		mat_x.symmetric_eigen(&mut eigenvalues, &mut eigenvectors);

		let mut mat_v = B::Matrix::zeros(self.n, self.n);
		self.vec_to_mat(vector, &mut mat_v);

		for i in 0..self.n {
			if eigenvalues.get(i).abs() < self.tolerance {
				// Check vᵀ V v ≥ 0 for kernel eigenvector v
				let mut v_t_v_v = T::zero();
				for r in 0..self.n {
					let mut row_sum = T::zero();
					for c in 0..self.n {
						row_sum += mat_v.get(r, c) * eigenvectors.get(c, i);
					}
					v_t_v_v += eigenvectors.get(r, i) * row_sum;
				}
				if v_t_v_v < -tol {
					return false;
				}
			}
		}
		true
	}

	/// Cold path — allocates temporary eigen buffers.
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		let mut mat = B::Matrix::zeros(self.n, self.n);
		self.vec_to_mat(point, &mut mat);

		let mut eigenvalues = B::Vector::zeros(self.n);
		let mut eigenvectors = B::Matrix::zeros(self.n, self.n);
		mat.symmetric_eigen(&mut eigenvalues, &mut eigenvectors);

		let mut temp = B::Matrix::zeros(self.n, self.n);
		self.clamp_and_reconstruct(&mut eigenvalues, &eigenvectors, &mut temp, &mut mat);
		self.mat_to_vec(&mat, result);
	}

	#[inline]
	fn project_tangent(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		_ws: &mut Self::Workspace,
	) {
		// Interior: tangent space is all symmetric matrices → identity projection.
		// Boundary projection is an SDP — out of scope here.
		result.copy_from(vector);
	}

	#[inline]
	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> T {
		u.dot(v)
	}

	#[inline]
	fn norm(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> T {
		vector.norm()
	}

	/// Metric projection retraction.  **Zero allocation.**
	///
	/// X + V → unpack → eigen → clamp λ ≥ 0 → Q max(0,Λ) Qᵀ → repack.
	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		// 1. result = point + tangent (vector addition)
		result.copy_from(point);
		result.add_assign(tangent);

		// 2. Unpack into symmetric matrix
		self.vec_to_mat(result, &mut ws.mat_a);

		// 3. Eigendecomposition (mat_a is &self, not destroyed)
		ws.mat_a
			.symmetric_eigen(&mut ws.eigenvalues, &mut ws.eigenvectors);

		// 4. Clamp & reconstruct: mat_b = Q·diag(λ), mat_a = mat_b·Qᵀ
		self.clamp_and_reconstruct(
			&mut ws.eigenvalues,
			&ws.eigenvectors,
			&mut ws.mat_b,
			&mut ws.mat_a,
		);

		// 5. Repack to vector
		self.mat_to_vec(&ws.mat_a, result);
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
	}

	#[inline]
	fn euclidean_to_riemannian_gradient(
		&self,
		point: &Self::Point,
		egrad: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		self.project_tangent(point, egrad, result, ws);
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

		let mut psd = B::Matrix::zeros(self.n, self.n);
		psd.gemm_at(T::one(), a.as_view(), a.as_view(), T::zero());

		if self.strict {
			for i in 0..self.n {
				*psd.get_mut(i, i) = psd.get(i, i) + <T as Scalar>::from_f64(1e-3);
			}
		}

		self.mat_to_vec(&psd, result);
	}

	fn random_tangent(&self, point: &Self::Point, result: &mut Self::TangentVector) {
		let mut rng = rand::rng();
		let normal = StandardNormal;

		for i in 0..self.dim {
			*result.get_mut(i) = <T as Scalar>::from_f64(normal.sample(&mut rng));
		}

		let mut ws = self.create_workspace(point);
		let temp = result.clone();
		self.project_tangent(point, &temp, result, &mut ws);

		let norm = result.norm();
		if norm > T::EPSILON {
			result.scale_mut(T::one() / norm);
		}
	}

	#[inline]
	fn distance(&self, x: &Self::Point, y: &Self::Point) -> T {
		let mut diff = y.clone();
		diff.sub_assign(x);
		diff.norm()
	}

	#[inline]
	fn has_exact_exp_log(&self) -> bool {
		false
	}

	#[inline]
	fn is_flat(&self) -> bool {
		true
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
		y.axpy(alpha, x, T::one());
	}

	#[inline]
	fn allocate_point(&self) -> Self::Point {
		B::Vector::zeros(self.dim)
	}

	#[inline]
	fn allocate_tangent(&self) -> Self::TangentVector {
		B::Vector::zeros(self.dim)
	}
}
