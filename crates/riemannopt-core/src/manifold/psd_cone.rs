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

/// Pre-allocated workspace for PSD Cone operations.
pub struct PSDConeWorkspace<T: Scalar, B: LinAlgBackend<T>> {
	/// n×n matrix buffer for eigendecomposition
	pub mat_a: B::Matrix,
	/// Second n×n matrix buffer
	pub mat_b: B::Matrix,
}

impl<T: Scalar, B: LinAlgBackend<T>> Default for PSDConeWorkspace<T, B> {
	fn default() -> Self {
		Self {
			mat_a: B::Matrix::zeros(0, 0),
			mat_b: B::Matrix::zeros(0, 0),
		}
	}
}

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
	/// Creates a new PSD cone manifold S⁺(n).
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

	/// Creates a PSD cone with custom parameters.
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

	// ── Vector/Matrix Conversions ────────────────────────────────────────

	/// Unpack vector of size n(n+1)/2 to n×n symmetric matrix.
	/// Reverses the √2 scaling on off-diagonals.
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

	/// Pack n×n symmetric matrix to vector of size n(n+1)/2.
	/// Applies √2 scaling on off-diagonals to preserve inner product.
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

	/// Symmetrize matrix in-place
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
}

// ════════════════════════════════════════════════════════════════════════════
//  Manifold trait implementation
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

	#[inline]
	fn is_point_on_manifold(&self, point: &Self::Point, _tol: T) -> bool {
		if point.len() != self.dim {
			return false;
		}

		// Convert to matrix (allocates locally since we don't have workspace here)
		let mut mat = B::Matrix::zeros(self.n, self.n);
		self.vec_to_mat(point, &mut mat);

		let eigen = mat.symmetric_eigen();
		let threshold = if self.strict {
			self.tolerance
		} else {
			-self.tolerance
		};

		let is_psd = eigen.eigenvalues.iter().all(|lambda| lambda >= threshold);
		is_psd
	}

	#[inline]
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
		} // Interior: tangent space is all symmetric matrices

		// For boundary points, we need v^T V v >= 0 for v in ker(X).
		// Convert point to matrix to find kernel
		let mut mat_x = B::Matrix::zeros(self.n, self.n);
		self.vec_to_mat(point, &mut mat_x);
		let eigen = mat_x.symmetric_eigen();

		let mut mat_v = B::Matrix::zeros(self.n, self.n);
		self.vec_to_mat(vector, &mut mat_v);

		// Find eigenvalues near zero (kernel)
		for i in 0..self.n {
			if eigen.eigenvalues.get(i).abs() < self.tolerance {
				let mut v_t_v_v = T::zero();
				for r in 0..self.n {
					let mut row_sum = T::zero();
					for c in 0..self.n {
						row_sum = row_sum + mat_v.get(r, c) * eigen.eigenvectors.get(c, i);
					}
					v_t_v_v = v_t_v_v + eigen.eigenvectors.get(r, i) * row_sum;
				}
				if v_t_v_v < -tol {
					return false;
				}
			}
		}
		true
	}

	#[inline]
	fn project_point(&self, point: &Self::Point, result: &mut Self::Point) {
		// Allocate temporary workspace if none provided
		let mut mat = B::Matrix::zeros(self.n, self.n);
		self.vec_to_mat(point, &mut mat);

		let mut eigen = mat.symmetric_eigen();

		// Threshold eigenvalues: max(0, λ)
		let threshold = if self.strict { T::EPSILON } else { T::zero() };
		for i in 0..self.n {
			if eigen.eigenvalues.get(i) < threshold {
				*eigen.eigenvalues.get_mut(i) = threshold;
			}
		}

		// Reconstruct: Q max(0, Λ) Q^T
		let mut temp = B::Matrix::zeros(self.n, self.n);
		temp.scale_columns(&eigen.eigenvectors, &eigen.eigenvalues);
		mat.gemm_bt(
			T::one(),
			temp.as_view(),
			eigen.eigenvectors.as_view(),
			T::zero(),
		);

		Self::symmetrize(&mut mat, self.n);
		self.mat_to_vec(&mat, result);
	}

	#[inline]
	fn project_tangent(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
		result: &mut Self::TangentVector,
		ws: &mut Self::Workspace,
	) {
		if self.strict {
			result.copy_from(vector);
			return;
		}

		// Boundary projection logic (simplified version, exact projection is an SDP)
		// Here we just copy the vector. A true rigorous projection onto the tangent cone
		// at the boundary is highly non-trivial and often solved iteratively.
		// We assume the caller handles active set methods or interior point methods.
		result.copy_from(vector);

		// To avoid unused parameter warnings while acknowledging boundary logic
		let _ = point;
		let _ = ws;
	}

	#[inline]
	fn inner_product(
		&self,
		_point: &Self::Point,
		u: &Self::TangentVector,
		v: &Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> T {
		// Because of the sqrt(2) scaling during vec packing, the standard Euclidean
		// dot product exactly matches the Frobenius inner product of the matrices.
		u.dot(v)
	}

	#[inline]
	fn norm(
		&self,
		_point: &Self::Point,
		vector: &Self::TangentVector,
		_ws: &mut Self::Workspace,
	) -> T {
		vector.norm() // Equal to Frobenius norm of the matrix
	}

	#[inline]
	fn retract(
		&self,
		point: &Self::Point,
		tangent: &Self::TangentVector,
		result: &mut Self::Point,
		ws: &mut Self::Workspace,
	) {
		// Simple metric projection retraction
		// X + V is symmetric. Just need to clamp eigenvalues to 0.

		// 1. Vector addition (Euclidean space)
		result.copy_from(point);
		result.add_assign(tangent);

		// 2. Unpack to matrix in workspace
		self.vec_to_mat(result, &mut ws.mat_a);

		// 3. Eigendecomposition
		let mut eigen = ws.mat_a.symmetric_eigen();

		// 4. Threshold eigenvalues
		let threshold = if self.strict { T::EPSILON } else { T::zero() };
		for i in 0..self.n {
			if eigen.eigenvalues.get(i) < threshold {
				*eigen.eigenvalues.get_mut(i) = threshold;
			}
		}

		// 5. Reconstruct: Q max(0, Λ) Q^T
		ws.mat_b
			.scale_columns(&eigen.eigenvectors, &eigen.eigenvalues);
		ws.mat_a.gemm_bt(
			T::one(),
			ws.mat_b.as_view(),
			eigen.eigenvectors.as_view(),
			T::zero(),
		);

		Self::symmetrize(&mut ws.mat_a, self.n);

		// 6. Repack to vector
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
		// In the Euclidean metric, inverse retraction is just subtraction
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

		// Ensure strictly inside cone if requested
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
		// Since we scale off-diagonals by sqrt(2), Euclidean distance of packed vectors
		// equals Frobenius distance of the corresponding matrices.
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

	// ════════════════════════════════════════════════════════════════════════
	// Vector ops
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
