//! Backend-agnostic linear algebra traits.
//!
//! These traits define the operations that any linear algebra backend must support.
//! Implementations exist for faer (default) and nalgebra.
//!
//! # Trait hierarchy
//!
//! ```text
//! VectorView  (read-only: len, get, dot, norm, iter)
//!   └─ VectorOps  (+ construction, mutation, allocation)
//!
//! MatrixView  (read-only: nrows, ncols, get, column, norm, trace)
//!   └─ MatrixOps  (+ construction, mutation, GEMM, views)
//!       └─ DecompositionOps  (SVD, QR, Cholesky, Eigen)
//! ```
//!
//! # Zero-Cost Abstraction
//!
//! All traits are monomorphized at compile time — there is no dynamic dispatch
//! overhead. `Sphere<f64, FaerBackend>` compiles to the same code as if
//! faer types were used directly.
//!
//! View types (`ColView`, `View`) are backend-specific borrowed references
//! (e.g. `faer::ColRef`, `faer::MatRef`) that avoid heap allocation.

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
//  Read-only vector operations
// ═══════════════════════════════════════════════════════════════════════════

/// Read-only vector operations.
///
/// Implemented by both owned vectors (`Col<T>`, `DVector<T>`) and borrowed
/// views (`ColRef<'_, T>`, `DVectorView<'_, T>`).  All methods are
/// non-mutating and zero-allocation.
pub trait VectorView<T: RealScalar>: Sized + Clone + Debug {
	/// Number of elements.
	fn len(&self) -> usize;

	/// Whether the vector is empty.
	fn is_empty(&self) -> bool {
		self.len() == 0
	}

	/// Get element at index `i`.
	fn get(&self, i: usize) -> T;

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

	/// Iterate over elements (by value).
	fn iter(&self) -> impl Iterator<Item = T> + '_;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Full vector operations (owned)
// ═══════════════════════════════════════════════════════════════════════════

/// Dense column vector operations (construction + mutation).
///
/// Extends [`VectorView`] with methods that require ownership or mutation.
pub trait VectorOps<T: RealScalar>: VectorView<T> + Send + Sync {
	// ── Construction ──────────────────────────────────────────────────

	/// Create a zero vector of length `len`.
	fn zeros(len: usize) -> Self;

	/// Create a vector from a closure `f(index) -> value`.
	fn from_fn(len: usize, f: impl FnMut(usize) -> T) -> Self;

	/// Create a vector from a slice.
	fn from_slice(data: &[T]) -> Self;

	// ── Element access (mutable) ─────────────────────────────────────

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

	// ── In-place element-wise ─────────────────────────────────────────

	/// Element-wise multiply in-place (Hadamard): `self[i] *= other[i]`.
	fn component_mul_assign(&mut self, other: &Self) {
		let dst = self.as_mut_slice();
		let src = other.as_slice();
		for (d, &s) in dst.iter_mut().zip(src) {
			*d = *d * s;
		}
	}

	/// Element-wise divide in-place: `self[i] /= other[i]`.
	fn component_div_assign(&mut self, other: &Self) {
		let dst = self.as_mut_slice();
		let src = other.as_slice();
		for (d, &s) in dst.iter_mut().zip(src) {
			*d = *d / s;
		}
	}

	/// Apply a function to each element in-place: `self[i] = f(self[i])`.
	fn map_mut(&mut self, mut f: impl FnMut(T) -> T) {
		let s = self.as_mut_slice();
		for v in s.iter_mut() {
			*v = f(*v);
		}
	}

	// ── Functional ────────────────────────────────────────────────────

	/// Apply a function element-wise, returning a new vector.
	fn map(&self, f: impl FnMut(T) -> T) -> Self;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Read-only matrix operations
// ═══════════════════════════════════════════════════════════════════════════

/// Read-only matrix operations.
///
/// Implemented by both owned matrices (`Mat<T>`, `DMatrix<T>`) and borrowed
/// views (`MatRef<'_, T>`, `DMatrixView<'_, T>`).  All methods are
/// non-mutating and zero-allocation.
pub trait MatrixView<T: RealScalar>: Sized + Clone + Debug {
	/// The column view type returned by [`column`](Self::column).
	type ColView<'a>: VectorView<T>
	where
		Self: 'a;

	fn nrows(&self) -> usize;
	fn ncols(&self) -> usize;

	fn get(&self, i: usize, j: usize) -> T;

	/// Extract column `j` as a borrowed view (zero-allocation).
	fn column(&self, j: usize) -> Self::ColView<'_>;

	/// Frobenius norm ‖self‖_F.
	fn norm(&self) -> T;

	/// Trace tr(self).
	fn trace(&self) -> T;

	/// Dot product between column `j` of self and column `k` of other.
	fn column_dot(&self, j: usize, other: &Self, k: usize) -> T {
		let n = self.nrows();
		let mut sum = T::zero();
		for i in 0..n {
			sum = sum + self.get(i, j) * other.get(i, k);
		}
		sum
	}
}

// ═══════════════════════════════════════════════════════════════════════════
//  Full matrix operations (owned)
// ═══════════════════════════════════════════════════════════════════════════

/// Dense matrix operations (column-major, construction + mutation).
///
/// Extends [`MatrixView`] with methods that require ownership or mutation.
/// Introduces a `View<'a>` GAT for zero-allocation sub-matrix and column
/// access that can be passed directly to GEMM operations.
pub trait MatrixOps<T: RealScalar>: MatrixView<T> + Send + Sync {
	/// The associated owned vector type.
	type Col: VectorOps<T>;

	/// The associated matrix view type (e.g. `faer::MatRef<'a, T>`).
	///
	/// Must implement [`MatrixView`] so it can be passed to GEMM and other
	/// read-only operations without allocation.
	type View<'a>: MatrixView<T>
	where
		Self: 'a;

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

	// ── View accessors (zero-allocation) ─────────────────────────────

	/// Borrow the entire matrix as a view.
	fn as_view(&self) -> Self::View<'_>;

	/// Extract a contiguous block of columns [start..start+count) as a view.
	fn columns(&self, start: usize, count: usize) -> Self::View<'_>;

	/// Extract a contiguous block of rows [start..start+count) as a view.
	fn rows(&self, start: usize, count: usize) -> Self::View<'_>;

	// ── Owned conversions (explicit allocation) ──────────────────────

	/// Extract column `j` as an owned vector (allocates).
	fn column_to_owned(&self, j: usize) -> Self::Col {
		let v = self.column(j);
		Self::Col::from_fn(VectorView::len(&v), |i| VectorView::get(&v, i))
	}

	/// Transpose into a new owned matrix (allocates).
	fn transpose_to_owned(&self) -> Self;

	/// Extract columns [start..start+count) as an owned matrix (allocates).
	fn columns_to_owned(&self, start: usize, count: usize) -> Self {
		let view = self.columns(start, count);
		Self::from_fn(
			MatrixView::nrows(&view),
			MatrixView::ncols(&view),
			|i, j| MatrixView::get(&view, i, j),
		)
	}

	/// Extract rows [start..start+count) as an owned matrix (allocates).
	fn rows_to_owned(&self, start: usize, count: usize) -> Self {
		let view = self.rows(start, count);
		Self::from_fn(
			MatrixView::nrows(&view),
			MatrixView::ncols(&view),
			|i, j| MatrixView::get(&view, i, j),
		)
	}

	// ── Element access ────────────────────────────────────────────────

	fn get_mut(&mut self, i: usize, j: usize) -> &mut T;

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

	// ── In-place mutations ────────────────────────────────────────────

	fn copy_from(&mut self, other: &Self);
	fn fill(&mut self, value: T);
	fn scale_mut(&mut self, alpha: T);

	/// Column-scaling: self[i,j] = source[i,j] * diag[j].
	///
	/// Computes `self = source * diag(diag)` without allocating a diagonal matrix.
	/// Each column j of the result is column j of source scaled by diag[j].
	fn scale_columns(&mut self, source: &Self, diag: &Self::Col) {
		let n = source.nrows();
		let p = source.ncols();
		for j in 0..p {
			let s = diag.get(j);
			for i in 0..n {
				*self.get_mut(i, j) = source.get(i, j) * s;
			}
		}
	}

	// ── Arithmetic (new allocation) ───────────────────────────────────

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

	// ── In-place element-wise ─────────────────────────────────────────

	/// self = alpha * x + beta * self  (element-wise AXPY for matrices).
	fn mat_axpy(&mut self, alpha: T, x: &Self, beta: T) {
		let dst = self.as_mut_slice();
		let src = x.as_slice();
		if beta == T::zero() {
			for (d, &s) in dst.iter_mut().zip(src) {
				*d = alpha * s;
			}
		} else if beta == T::one() {
			for (d, &s) in dst.iter_mut().zip(src) {
				*d = *d + alpha * s;
			}
		} else {
			for (d, &s) in dst.iter_mut().zip(src) {
				*d = beta * *d + alpha * s;
			}
		}
	}

	/// Element-wise multiply in-place (Hadamard): `self[i,j] *= other[i,j]`.
	fn mat_component_mul_assign(&mut self, other: &Self) {
		let dst = self.as_mut_slice();
		let src = other.as_slice();
		for (d, &s) in dst.iter_mut().zip(src) {
			*d = *d * s;
		}
	}

	// ── In-place BLAS-like ────────────────────────────────────────────

	/// self += other.
	fn add_assign(&mut self, other: &Self);

	/// self -= other.
	fn sub_assign(&mut self, other: &Self);

	/// C = alpha * A * B + beta * C  (GEMM).
	///
	/// Operands `a` and `b` are views — pass `m.as_view()` for full matrices,
	/// or `m.columns(start, count)` / `m.rows(start, count)` for sub-matrices.
	fn gemm(&mut self, alpha: T, a: Self::View<'_>, b: Self::View<'_>, beta: T);

	/// C = alpha * Aᵀ * B + beta * C  (GEMM with left operand transposed).
	///
	/// Zero-allocation: the backend transposes the view in-place (stride swap).
	fn gemm_at(&mut self, alpha: T, a: Self::View<'_>, b: Self::View<'_>, beta: T);

	/// C = alpha * A * Bᵀ + beta * C  (GEMM with right operand transposed).
	///
	/// Zero-allocation: the backend transposes the view in-place (stride swap).
	fn gemm_bt(&mut self, alpha: T, a: Self::View<'_>, b: Self::View<'_>, beta: T);
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

	/// Compute matrix inverse into pre-allocated `result`.
	///
	/// Returns `false` if singular (result is unchanged).
	fn inverse(&self, result: &mut Self) -> bool;

	/// Solve the system A x = rhs via Cholesky decomposition (A = L Lᵀ),
	/// writing into pre-allocated `result`.
	///
	/// Copies `rhs` into `result`, then solves in-place.
	/// Returns `false` if A is not positive definite (result is unchanged).
	fn cholesky_solve(&self, rhs: &Self, result: &mut Self) -> bool;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Backend trait (type-level factory)
// ═══════════════════════════════════════════════════════════════════════════

/// Linear algebra backend — a type-level factory for vector and matrix types.
///
/// The backend is parameterized by the scalar type `T` to allow backends
/// that require additional trait bounds on `T` (e.g. nalgebra requires
/// `NalgebraScalar + RealField`).
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
