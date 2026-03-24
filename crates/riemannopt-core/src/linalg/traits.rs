//! Backend-agnostic linear algebra traits.
//!
//! These traits define the operations that any linear algebra backend must support.
//! Implementations exist for nalgebra (default) and optionally faer.
//!
//! # Zero-Cost Abstraction
//!
//! All traits are monomorphized at compile time — there is no dynamic dispatch
//! overhead. `Sphere<f64, NalgebraBackend>` compiles to the same code as if
//! nalgebra types were used directly.

use num_traits::Float;
use std::fmt::Debug;

use crate::linalg::types::{CholeskyResult, EigenResult, QrResult, SvdResult};

/// A scalar type suitable for linear algebra operations.
///
/// This is deliberately minimal — backend-specific bounds (e.g. nalgebra's
/// `RealField`) are added only in the backend's `impl` blocks.
pub trait RealScalar:
	Float
	+ num_traits::FromPrimitive
	+ std::fmt::Display
	+ Debug
	+ Default
	+ Copy
	+ Send
	+ Sync
	+ 'static
	+ std::ops::AddAssign
	+ std::ops::SubAssign
	+ std::ops::MulAssign
	+ std::ops::DivAssign
{
	fn from_f64_const(v: f64) -> Self;
	fn to_f64_lossy(self) -> f64;
	fn from_usize(v: usize) -> Self;
}

impl RealScalar for f32 {
	#[inline]
	fn from_f64_const(v: f64) -> Self {
		v as f32
	}
	#[inline]
	fn to_f64_lossy(self) -> f64 {
		self as f64
	}
	#[inline]
	fn from_usize(v: usize) -> Self {
		v as f32
	}
}

impl RealScalar for f64 {
	#[inline]
	fn from_f64_const(v: f64) -> Self {
		v
	}
	#[inline]
	fn to_f64_lossy(self) -> f64 {
		self
	}
	#[inline]
	fn from_usize(v: usize) -> Self {
		v as f64
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Vector operations
// ═══════════════════════════════════════════════════════════════════════════

/// Dense column vector operations.
///
/// Every method has a default `#[inline]` hint — backends should override
/// with optimised implementations where possible.
pub trait VectorOps<T: RealScalar>: Sized + Clone + Debug + Send + Sync {
	// ── Construction ──────────────────────────────────────────────────

	/// Create a zero vector of length `len`.
	fn zeros(len: usize) -> Self;

	/// Create a vector from a closure `f(index) -> value`.
	fn from_fn(len: usize, f: impl FnMut(usize) -> T) -> Self;

	/// Create a vector from a slice.
	fn from_slice(data: &[T]) -> Self;

	// ── Queries ───────────────────────────────────────────────────────

	/// Number of elements.
	fn len(&self) -> usize;

	/// Whether the vector is empty.
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Inner product ⟨self, other⟩.
	fn dot(&self, other: &Self) -> T;

	/// Euclidean norm ‖self‖₂.
	fn norm(&self) -> T {
		Float::sqrt(self.norm_squared())
	}

	/// Squared Euclidean norm ‖self‖₂².
	fn norm_squared(&self) -> T {
		self.dot(self)
	}

	// ── Element access ────────────────────────────────────────────────

	/// Get element at index `i`.
	fn get(&self, i: usize) -> T;

	/// Get mutable reference to element at index `i`.
	fn get_mut(&mut self, i: usize) -> &mut T;

	/// View as a contiguous slice.
	fn as_slice(&self) -> &[T];

	/// View as a mutable contiguous slice.
	fn as_mut_slice(&mut self) -> &mut [T];

	// ── In-place mutations ────────────────────────────────────────────

	/// Copy all elements from `other`.
	fn copy_from(&mut self, other: &Self);

	/// Fill all elements with `value`.
	fn fill(&mut self, value: T);

	/// self = alpha * x + beta * self  (BLAS axpy generalisation).
	fn axpy(&mut self, alpha: T, x: &Self, beta: T);

	/// self *= alpha.
	fn scale_mut(&mut self, alpha: T);

	/// self /= alpha.
	fn div_scalar_mut(&mut self, alpha: T) {
		if alpha != T::zero() {
			self.scale_mut(T::one() / alpha);
		}
	}

	// ── Arithmetic (new allocation) ──────────────────────────────────

	/// self + other.
	fn add(&self, other: &Self) -> Self {
		let mut result = self.clone();
		result.add_assign(other);
		result
	}

	/// self - other.
	fn sub(&self, other: &Self) -> Self {
		let mut result = self.clone();
		result.sub_assign(other);
		result
	}

	/// -self.
	fn neg(&self) -> Self {
		let mut result = self.clone();
		result.scale_mut(T::zero() - T::one());
		result
	}

	/// Element-wise multiplication (Hadamard product).
	fn component_mul(&self, other: &Self) -> Self {
		Self::from_fn(self.len(), |i| self.get(i) * other.get(i))
	}

	// ── In-place mutations (cont.) ───────────────────────────────────

	/// self += other.
	fn add_assign(&mut self, other: &Self) {
		self.axpy(T::one(), other, T::one());
	}

	/// self -= other.
	fn sub_assign(&mut self, other: &Self) {
		self.axpy(T::zero() - T::one(), other, T::one());
	}

	// ── Functional ────────────────────────────────────────────────────

	/// Apply a function element-wise, returning a new vector.
	fn map(&self, f: impl FnMut(T) -> T) -> Self;

	/// Iterate over elements (by value).
	fn iter(&self) -> impl Iterator<Item = T> + '_;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Matrix operations
// ═══════════════════════════════════════════════════════════════════════════

/// Dense matrix operations (column-major).
pub trait MatrixOps<T: RealScalar>: Sized + Clone + Debug + Send + Sync {
	/// The associated vector type (e.g. DVector for DMatrix).
	type Col: VectorOps<T>;

	// ── Construction ──────────────────────────────────────────────────

	/// Create an m×n zero matrix.
	fn zeros(nrows: usize, ncols: usize) -> Self;

	/// Create an n×n identity matrix.
	fn identity(n: usize) -> Self;

	/// Create a matrix from a closure `f(row, col) -> value`.
	fn from_fn(nrows: usize, ncols: usize, f: impl FnMut(usize, usize) -> T) -> Self;

	/// Create a diagonal matrix from a vector.
	fn from_diagonal(diag: &Self::Col) -> Self;

	/// Create a matrix from column-major slice data.
	fn from_column_slice(nrows: usize, ncols: usize, data: &[T]) -> Self {
		Self::from_fn(nrows, ncols, |i, j| data[j * nrows + i])
	}

	// ── Queries ───────────────────────────────────────────────────────

	fn nrows(&self) -> usize;
	fn ncols(&self) -> usize;

	/// Frobenius norm ‖self‖_F.
	fn norm(&self) -> T;

	/// Trace tr(self).
	fn trace(&self) -> T;

	// ── Element access ────────────────────────────────────────────────

	fn get(&self, i: usize, j: usize) -> T;
	fn get_mut(&mut self, i: usize, j: usize) -> &mut T;

	/// Extract column `j` as an owned vector.
	fn column(&self, j: usize) -> Self::Col;

	/// Extract a contiguous block of columns [start..start+count).
	fn columns(&self, start: usize, count: usize) -> Self;

	/// Extract a contiguous block of rows [start..start+count).
	fn rows(&self, start: usize, count: usize) -> Self;

	/// Set column `j` from a vector.
	fn set_column(&mut self, j: usize, col: &Self::Col);

	/// Set a block of rows [start..start+count) from another matrix.
	fn set_rows(&mut self, start: usize, src: &Self);

	/// View the full matrix data as a contiguous column-major slice.
	fn as_slice(&self) -> &[T];

	/// View the full matrix data as a mutable contiguous column-major slice.
	fn as_mut_slice(&mut self) -> &mut [T];

	/// Get a mutable slice of column `j` data (column-major layout guarantees contiguity).
	fn column_as_mut_slice(&mut self, j: usize) -> &mut [T] {
		let nrows = self.nrows();
		let data = self.as_mut_slice();
		&mut data[j * nrows..(j + 1) * nrows]
	}

	/// Dot product between column `j` of self and column `k` of other.
	fn column_dot(&self, j: usize, other: &Self, k: usize) -> T {
		let n = self.nrows();
		let mut sum = T::zero();
		for i in 0..n {
			sum = sum + self.get(i, j) * other.get(i, k);
		}
		sum
	}

	// ── In-place mutations ────────────────────────────────────────────

	fn copy_from(&mut self, other: &Self);
	fn fill(&mut self, value: T);
	fn scale_mut(&mut self, alpha: T);

	// ── Arithmetic (new allocation) ───────────────────────────────────

	/// self^T (transpose, allocates).
	fn transpose(&self) -> Self;

	/// Matrix-matrix product self * other.
	fn mat_mul(&self, other: &Self) -> Self;

	/// Matrix-vector product self * v.
	fn mat_vec(&self, v: &Self::Col) -> Self::Col;

	/// self + other.
	fn add(&self, other: &Self) -> Self;

	/// self - other.
	fn sub(&self, other: &Self) -> Self;

	/// alpha * self.
	fn scale_by(&self, alpha: T) -> Self;

	// ── In-place BLAS-like ────────────────────────────────────────────

	/// self += other.
	fn add_assign(&mut self, other: &Self);

	/// self -= other.
	fn sub_assign(&mut self, other: &Self);

	/// C = alpha * A * B + beta * C  (GEMM).
	fn gemm(&mut self, alpha: T, a: &Self, b: &Self, beta: T);

	/// C = alpha * A^T * B + beta * C  (GEMM with left operand transposed).
	///
	/// Avoids allocating a transposed copy of A. Backends with native transpose
	/// views (faer, nalgebra) override this for zero-alloc performance.
	fn gemm_at(&mut self, alpha: T, a: &Self, b: &Self, beta: T) {
		let at = a.transpose();
		self.gemm(alpha, &at, b, beta);
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Decompositions (separate trait — only required by manifolds that use them)
// ═══════════════════════════════════════════════════════════════════════════

/// Matrix decomposition operations.
pub trait DecompositionOps<T: RealScalar>: MatrixOps<T> {
	/// Singular Value Decomposition: self = U Σ Vᵀ.
	fn svd(&self) -> SvdResult<T, Self>;

	/// QR decomposition: self = Q R.
	fn qr(&self) -> QrResult<T, Self>;

	/// Symmetric eigendecomposition: self = V diag(λ) Vᵀ.
	/// Eigenvalues are sorted in ascending order.
	fn symmetric_eigen(&self) -> EigenResult<T, Self>;

	/// Cholesky decomposition: self = L Lᵀ.
	/// Returns `None` if the matrix is not positive definite.
	fn cholesky(&self) -> Option<CholeskyResult<T, Self>>;

	/// Matrix inverse. Returns `None` if singular.
	fn try_inverse(&self) -> Option<Self>;

	/// Solve the system A x = rhs via Cholesky decomposition (A = L Lᵀ).
	/// Returns `None` if A is not positive definite.
	fn cholesky_solve(&self, rhs: &Self) -> Option<Self>;

	/// Solve the system A x = rhs via Cholesky, where rhs is a column vector.
	/// Returns `None` if A is not positive definite.
	fn cholesky_solve_vec(&self, rhs: &Self::Col) -> Option<Self::Col>
	where
		Self: MatrixOps<T>,
	{
		// Default: convert vector to 1-column matrix, solve, extract column
		let n = rhs.len();
		let rhs_mat = Self::from_fn(n, 1, |i, _| rhs.get(i));
		self.cholesky_solve(&rhs_mat)
			.map(|sol| MatrixOps::column(&sol, 0))
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Backend trait (type-level factory)
// ═══════════════════════════════════════════════════════════════════════════

/// Linear algebra backend — a type-level factory for vector and matrix types.
///
/// The backend is parameterized by the scalar type `T` to allow backends
/// that require additional trait bounds on `T` (e.g. nalgebra requires
/// `NalgebraScalar + RealField`).
///
/// # Example
///
/// ```rust,ignore
/// use riemannopt_core::linalg::{LinAlgBackend, NalgebraBackend, VectorOps};
///
/// fn create_zero_vec<B: LinAlgBackend<f64>>() -> B::Vector {
///     B::Vector::zeros(10)
/// }
///
/// let v = create_zero_vec::<NalgebraBackend>();
/// assert_eq!(VectorOps::len(&v), 10);
/// ```
pub trait LinAlgBackend<T: RealScalar>: Debug + Clone + Send + Sync + 'static {
	type Vector: VectorOps<T>;
	type Matrix: MatrixOps<T, Col = Self::Vector> + DecompositionOps<T>;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Convenience type aliases
// ═══════════════════════════════════════════════════════════════════════════

/// Shorthand for the vector type of a backend.
pub type VecOf<B, T> = <B as LinAlgBackend<T>>::Vector;

/// Shorthand for the matrix type of a backend.
pub type MatOf<B, T> = <B as LinAlgBackend<T>>::Matrix;
