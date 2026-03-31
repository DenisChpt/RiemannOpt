//! Decomposition result types — backend-agnostic.

use super::traits::{MatrixOps, RealScalar};

/// Result of a Singular Value Decomposition: A = U Σ Vᵀ.
#[derive(Debug, Clone)]
pub struct SvdResult<T: RealScalar, M: MatrixOps<T>> {
	/// Left singular vectors (m×k), where k = min(m,n). `None` if not requested.
	pub u: Option<M>,
	/// Singular values σ₁ ≥ σ₂ ≥ … ≥ σ_k ≥ 0.
	pub singular_values: M::Col,
	/// Right singular vectors transposed (k×n). `None` if not requested.
	pub vt: Option<M>,
}

/// Result of a QR decomposition: A = Q R.
#[derive(Debug, Clone)]
pub struct QrResult<T: RealScalar, M: MatrixOps<T>> {
	/// Orthogonal/unitary factor Q (m×m or thin m×k).
	q: M,
	/// Upper triangular factor R (m×n or thin k×n).
	r: M,
	_phantom: std::marker::PhantomData<T>,
}

impl<T: RealScalar, M: MatrixOps<T>> QrResult<T, M> {
	/// Create a new QR result.
	pub fn new(q: M, r: M) -> Self {
		Self {
			q,
			r,
			_phantom: std::marker::PhantomData,
		}
	}

	/// The Q factor.
	pub fn q(&self) -> &M {
		&self.q
	}

	/// The R factor.
	pub fn r(&self) -> &M {
		&self.r
	}
}

/// Result of a symmetric eigendecomposition: A = V diag(λ) Vᵀ.
#[derive(Debug, Clone)]
pub struct EigenResult<T: RealScalar, M: MatrixOps<T>> {
	/// Eigenvalues in ascending order: λ₁ ≤ λ₂ ≤ … ≤ λ_n.
	pub eigenvalues: M::Col,
	/// Eigenvectors as columns of V (orthonormal).
	pub eigenvectors: M,
}

/// Result of a Cholesky decomposition: A = L Lᵀ.
#[derive(Debug, Clone)]
pub struct CholeskyResult<T: RealScalar, M: MatrixOps<T>> {
	/// Lower-triangular factor L.
	l: M,
	_phantom: std::marker::PhantomData<T>,
}

impl<T: RealScalar, M: MatrixOps<T>> CholeskyResult<T, M> {
	/// Create a new Cholesky result.
	pub fn new(l: M) -> Self {
		Self {
			l,
			_phantom: std::marker::PhantomData,
		}
	}

	/// The lower-triangular factor L.
	pub fn l(&self) -> &M {
		&self.l
	}
}
