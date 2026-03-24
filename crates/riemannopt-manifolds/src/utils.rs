//! # Utility Functions for Matrix/Vector Operations
//!
//! This module provides essential utilities for manipulation of matrices
//! and vectors. These functions are crucial for the performance of manifold
//! operations, especially when dealing with matrix manifolds where points
//! and tangent vectors may need to be viewed in different formats.
//!
//! ## Key Features
//!
//! 1. **Format Conversion**: Convert between vector and matrix representations
//! 2. **In-Place Operations**: Perform matrix multiplications directly into existing
//!    storage
//! 3. **Memory Safety**: All operations maintain Rust's memory safety guarantees
//!
//! ## Design Principles
//!
//! - **Performance First**: Every function is designed to minimize allocations
//! - **Type Safety**: Strong typing prevents dimension mismatches at compile time
//! - **Backend Agnostic**: Uses `linalg` abstraction for portability
//!
//! ## Common Use Cases
//!
//! ### Matrix Manifolds
//! Many manifolds (Stiefel, Grassmann, SPD) internally represent points as matrices
//! but may need to expose them as vectors for generic optimization algorithms:
//!
//! ```rust,no_run
//! use riemannopt_manifolds::utils::vector_to_matrix;
//! use riemannopt_core::linalg::{self, VectorOps};
//!
//! // Optimization algorithm provides a vector
//! let point_vec = linalg::Vec::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//!
//! // Convert to a 3×2 matrix for manifold operations
//! let point_mat = vector_to_matrix::<f64>(&point_vec, 3, 2);
//! // Now we can perform matrix operations
//! ```
//!
//! ### Efficient Updates
//! In-place operations are essential for performance in iterative algorithms:
//!
//! ```rust,no_run
//! use riemannopt_manifolds::utils::gemm_inplace;
//! use riemannopt_core::linalg::{self, MatrixOps};
//!
//! let a = linalg::Mat::<f64>::from_column_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
//! let b = linalg::Mat::<f64>::from_column_slice(2, 2, &[5.0, 6.0, 7.0, 8.0]);
//! let mut c = linalg::Mat::<f64>::zeros(2, 2);
//!
//! // Compute C = 1.0 * A * B + 0.0 * C without allocating temporary matrices
//! gemm_inplace(1.0, &a, &b, 0.0, &mut c);
//! ```

use riemannopt_core::{
	linalg::{self, DefaultBackend, LinAlgBackend, MatrixOps, VectorOps},
	types::Scalar,
};

/// Creates an owned matrix from a vector's data.
///
/// This function interprets a vector as a matrix in column-major order
/// and returns an owned matrix copy, which is essential for manifolds
/// that internally use matrix representations.
///
/// # Arguments
///
/// * `vector` - The source vector to interpret as a matrix
/// * `nrows` - Number of rows in the resulting matrix
/// * `ncols` - Number of columns in the resulting matrix
///
/// # Returns
///
/// An owned matrix containing the vector's data in column-major layout.
///
/// # Panics
///
/// Panics in debug mode if `vector.len() != nrows * ncols`.
///
/// # Memory Layout
///
/// The vector is interpreted in column-major order (Fortran order).
/// For a 2×2 matrix, the vector [a, b, c, d] represents the matrix:
/// ```text
/// [ a  c ]
/// [ b  d ]
/// ```
///
/// # Example
///
/// ```rust
/// use riemannopt_manifolds::utils::vector_to_matrix;
/// use riemannopt_core::linalg::{self, VectorOps, MatrixOps};
///
/// let vec = linalg::Vec::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
/// let mat = vector_to_matrix::<f64>(&vec, 2, 2);
///
/// assert_eq!(MatrixOps::get(&mat, 0, 0), 1.0);
/// assert_eq!(MatrixOps::get(&mat, 1, 0), 2.0);
/// assert_eq!(MatrixOps::get(&mat, 0, 1), 3.0);
/// assert_eq!(MatrixOps::get(&mat, 1, 1), 4.0);
/// ```
#[inline]
pub fn vector_to_matrix<T: Scalar>(
	vector: &linalg::Vec<T>,
	nrows: usize,
	ncols: usize,
) -> linalg::Mat<T>
where
	DefaultBackend: LinAlgBackend<T>,
{
	debug_assert_eq!(
		vector.len(),
		nrows * ncols,
		"Vector length must equal nrows * ncols"
	);

	// Create a matrix from the vector's slice data in column-major order
	linalg::Mat::<T>::from_column_slice(nrows, ncols, vector.as_slice())
}

/// Creates an owned vector from a matrix's data.
///
/// This function flattens a matrix into a vector in column-major order,
/// useful when optimization algorithms expect vector representations.
///
/// # Arguments
///
/// * `matrix` - The source matrix to flatten as a vector
///
/// # Returns
///
/// An owned vector containing the matrix's data in column-major order.
///
/// # Memory Layout
///
/// The matrix is flattened in column-major order. For a 2×2 matrix:
/// ```text
/// [ a  c ]  →  [a, b, c, d]
/// [ b  d ]
/// ```
///
/// # Example
///
/// ```rust
/// use riemannopt_manifolds::utils::matrix_to_vector;
/// use riemannopt_core::linalg::{self, VectorOps, MatrixOps};
///
/// let mat = linalg::Mat::<f64>::from_column_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
/// let vec = matrix_to_vector::<f64>(&mat);
///
/// assert_eq!(vec.len(), 4);
/// assert_eq!(VectorOps::get(&vec, 0), 1.0);
/// assert_eq!(VectorOps::get(&vec, 3), 4.0);
/// ```
#[inline]
pub fn matrix_to_vector<T: Scalar>(matrix: &linalg::Mat<T>) -> linalg::Vec<T>
where
	DefaultBackend: LinAlgBackend<T>,
{
	// Create a vector from the matrix's data in column-major order
	let nrows = matrix.nrows();
	let ncols = matrix.ncols();
	VectorOps::from_fn(nrows * ncols, |idx| {
		let j = idx / nrows;
		let i = idx % nrows;
		matrix.get(i, j)
	})
}

/// Reshapes a vector's data into a matrix, writing the result into an existing matrix.
///
/// This function copies the vector's data into the provided mutable matrix
/// in column-major order, useful for in-place manifold operations.
///
/// # Arguments
///
/// * `vector` - The source vector to interpret as a matrix
/// * `result` - The target matrix to write into (must have correct dimensions)
///
/// # Panics
///
/// Panics in debug mode if `vector.len() != result.nrows() * result.ncols()`.
///
/// # Example
///
/// ```rust
/// use riemannopt_manifolds::utils::vector_to_matrix_inplace;
/// use riemannopt_core::linalg::{self, VectorOps, MatrixOps};
///
/// let vec = linalg::Vec::<f64>::from_slice(&[1.0, 2.0, 3.0, 4.0]);
/// let mut mat = linalg::Mat::<f64>::zeros(2, 2);
/// vector_to_matrix_inplace::<f64>(&vec, &mut mat);
/// assert_eq!(MatrixOps::get(&mat, 0, 0), 1.0);
/// assert_eq!(MatrixOps::get(&mat, 1, 1), 4.0);
/// ```
#[inline]
pub fn vector_to_matrix_inplace<T: Scalar>(vector: &linalg::Vec<T>, result: &mut linalg::Mat<T>)
where
	DefaultBackend: LinAlgBackend<T>,
{
	debug_assert_eq!(
		vector.len(),
		result.nrows() * result.ncols(),
		"Vector length must equal nrows * ncols"
	);

	let nrows = result.nrows();
	for idx in 0..vector.len() {
		let j = idx / nrows;
		let i = idx % nrows;
		*result.get_mut(i, j) = vector.get(idx);
	}
}

/// In-place general matrix multiplication (GEMM): C = alpha * A * B + beta * C.
///
/// This function performs matrix multiplication without allocating temporary storage,
/// making it ideal for iterative algorithms where the same operation is performed
/// repeatedly.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for the product A * B
/// * `a` - First matrix (m × k)
/// * `b` - Second matrix (k × n)
/// * `beta` - Scalar multiplier for the initial value of C
/// * `c` - Result matrix (m × n), modified in-place
///
/// # Mathematical Operation
///
/// Computes: C ← αAB + βC
///
/// Common use cases:
/// - `beta = 0`: Pure multiplication C = alpha * A * B
/// - `beta = 1`: Accumulation C += alpha * A * B
/// - `alpha = 1, beta = 0`: Simple multiplication C = A * B
///
/// # Performance
///
/// When available, this delegates to optimized BLAS implementations (OpenBLAS,
/// Intel MKL, etc.) for maximum performance.
///
/// # Example
///
/// ```rust
/// use riemannopt_manifolds::utils::gemm_inplace;
/// use riemannopt_core::linalg::{self, MatrixOps};
///
/// let a = linalg::Mat::<f64>::from_column_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = linalg::Mat::<f64>::from_column_slice(3, 2, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
/// let mut c = linalg::Mat::<f64>::zeros(2, 2);
///
/// // Compute C = A * B
/// gemm_inplace(1.0, &a, &b, 0.0, &mut c);
/// ```
#[inline]
pub fn gemm_inplace<T: Scalar>(
	alpha: T,
	a: &linalg::Mat<T>,
	b: &linalg::Mat<T>,
	beta: T,
	c: &mut linalg::Mat<T>,
) where
	DefaultBackend: LinAlgBackend<T>,
{
	c.gemm(alpha, a, b, beta);
}

/// In-place general matrix-vector multiplication (GEMV): y = alpha * A * x + beta * y.
///
/// This function performs matrix-vector multiplication without allocating temporary
/// storage, crucial for efficient gradient computations and vector transport operations.
///
/// # Arguments
///
/// * `alpha` - Scalar multiplier for the product A * x
/// * `a` - Matrix (m × n)
/// * `x` - Vector (n × 1)
/// * `beta` - Scalar multiplier for the initial value of y
/// * `y` - Result vector (m × 1), modified in-place
///
/// # Mathematical Operation
///
/// Computes: y ← αAx + βy
///
/// Common use cases:
/// - `beta = 0`: Pure multiplication y = alpha * A * x
/// - `beta = 1`: Accumulation y += alpha * A * x
/// - `alpha = 1, beta = 0`: Simple multiplication y = A * x
///
/// # Performance
///
/// When available, this delegates to optimized BLAS implementations for
/// maximum performance.
///
/// # Example
///
/// ```rust
/// use riemannopt_manifolds::utils::gemv_inplace;
/// use riemannopt_core::linalg::{self, MatrixOps, VectorOps};
///
/// let a = linalg::Mat::<f64>::from_column_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let x = linalg::Vec::<f64>::from_slice(&[7.0, 8.0, 9.0]);
/// let mut y = linalg::Vec::<f64>::zeros(2);
///
/// // Compute y = A * x
/// gemv_inplace(1.0, &a, &x, 0.0, &mut y);
/// ```
#[inline]
pub fn gemv_inplace<T: Scalar>(
	alpha: T,
	a: &linalg::Mat<T>,
	x: &linalg::Vec<T>,
	beta: T,
	y: &mut linalg::Vec<T>,
) where
	DefaultBackend: LinAlgBackend<T>,
{
	// Compute y = alpha * A * x + beta * y
	// First scale y by beta
	y.scale_mut(beta);
	// Then add alpha * A * x
	let ax = a.mat_vec(x);
	y.axpy(alpha, &ax, T::one());
}
