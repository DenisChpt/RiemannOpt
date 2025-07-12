//! # Utility Functions for Zero-Copy Matrix/Vector Operations
//!
//! This module provides essential utilities for efficient manipulation of matrices
//! and vectors without unnecessary memory allocations. These functions are crucial
//! for the performance of manifold operations, especially when dealing with matrix
//! manifolds where points and tangent vectors may need to be viewed in different
//! formats.
//!
//! ## Key Features
//!
//! 1. **Zero-Copy Views**: Convert between vector and matrix representations without
//!    allocating new memory
//! 2. **In-Place Operations**: Perform matrix multiplications directly into existing
//!    storage
//! 3. **Memory Safety**: All operations maintain Rust's memory safety guarantees
//!
//! ## Design Principles
//!
//! - **Performance First**: Every function is designed to minimize allocations
//! - **Type Safety**: Strong typing prevents dimension mismatches at compile time
//! - **BLAS Integration**: Operations are compatible with BLAS when available
//!
//! ## Common Use Cases
//!
//! ### Matrix Manifolds
//! Many manifolds (Stiefel, Grassmann, SPD) internally represent points as matrices
//! but may need to expose them as vectors for generic optimization algorithms:
//!
//! ```rust,no_run
//! use riemannopt_manifolds::utils::vector_to_matrix_view;
//! use nalgebra::DVector;
//! 
//! // Optimization algorithm provides a vector
//! let point_vec = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
//! 
//! // View it as a 3×2 matrix for manifold operations
//! let point_mat = vector_to_matrix_view(&point_vec, 3, 2);
//! // Now we can perform matrix operations without copying data
//! ```
//!
//! ### Efficient Updates
//! In-place operations are essential for performance in iterative algorithms:
//!
//! ```rust,no_run
//! use riemannopt_manifolds::utils::gemm_inplace;
//! use nalgebra::DMatrix;
//! 
//! let a = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
//! let b = DMatrix::from_vec(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
//! let mut c = DMatrix::zeros(2, 2);
//! 
//! // Compute C = 1.0 * A * B + 0.0 * C without allocating temporary matrices
//! gemm_inplace(1.0, &a, &b, 0.0, &mut c);
//! ```

use nalgebra::{DMatrix, DVector, DVectorView, DMatrixView, Dyn};
use riemannopt_core::types::Scalar;

/// Creates a matrix view from a vector without cloning.
/// 
/// This function provides a zero-copy way to interpret a vector as a matrix,
/// which is essential for manifolds that internally use matrix representations.
/// 
/// # Arguments
/// 
/// * `vector` - The source vector to view as a matrix
/// * `nrows` - Number of rows in the resulting matrix view
/// * `ncols` - Number of columns in the resulting matrix view
/// 
/// # Returns
/// 
/// A read-only matrix view of the vector data.
/// 
/// # Panics
/// 
/// Panics in debug mode if `vector.len() != nrows * ncols`.
/// 
/// # Memory Layout
/// 
/// The vector is interpreted in column-major order (Fortran order), which is
/// the default for nalgebra and BLAS libraries. For a 2×2 matrix, the vector
/// [a, b, c, d] represents the matrix:
/// ```text
/// [ a  c ]
/// [ b  d ]
/// ```
/// 
/// # Example
/// 
/// ```rust
/// use riemannopt_manifolds::utils::vector_to_matrix_view;
/// use nalgebra::DVector;
/// 
/// let vec = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// let mat = vector_to_matrix_view(&vec, 2, 2);
/// 
/// assert_eq!(mat[(0, 0)], 1.0);
/// assert_eq!(mat[(1, 0)], 2.0);
/// assert_eq!(mat[(0, 1)], 3.0);
/// assert_eq!(mat[(1, 1)], 4.0);
/// ```
#[inline]
pub fn vector_to_matrix_view<'a, T: Scalar>(
    vector: &'a DVector<T>,
    nrows: usize,
    ncols: usize,
) -> DMatrixView<'a, T> {
    debug_assert_eq!(vector.len(), nrows * ncols, 
                     "Vector length must equal nrows * ncols");
    
    // Create a matrix view from the vector's data
    // nalgebra stores matrices in column-major order
    DMatrixView::from_slice_generic(vector.as_slice(), Dyn(nrows), Dyn(ncols))
}

/// Creates a vector view from a matrix without cloning.
/// 
/// This function provides a zero-copy way to interpret a matrix as a vector,
/// useful when optimization algorithms expect vector representations.
/// 
/// # Arguments
/// 
/// * `matrix` - The source matrix to view as a vector
/// 
/// # Returns
/// 
/// A read-only vector view of the matrix data in column-major order.
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
/// use riemannopt_manifolds::utils::matrix_to_vector_view;
/// use nalgebra::DMatrix;
/// 
/// let mat = DMatrix::from_vec(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
/// let vec = matrix_to_vector_view(&mat);
/// 
/// assert_eq!(vec.len(), 4);
/// assert_eq!(vec[0], 1.0);
/// assert_eq!(vec[3], 4.0);
/// ```
#[inline]
pub fn matrix_to_vector_view<'a, T: Scalar>(
    matrix: &'a DMatrix<T>,
) -> DVectorView<'a, T> {
    let len = matrix.nrows() * matrix.ncols();
    
    // Create a vector view from the matrix's data
    DVectorView::from_slice_generic(matrix.as_slice(), Dyn(len), nalgebra::U1)
}

/// Reshapes a mutable vector as a matrix view.
/// 
/// This function provides a zero-copy way to interpret and modify a vector
/// as a matrix, essential for in-place manifold operations.
/// 
/// # Arguments
/// 
/// * `vector` - The source vector to view as a mutable matrix
/// * `nrows` - Number of rows in the resulting matrix view
/// * `ncols` - Number of columns in the resulting matrix view
/// 
/// # Returns
/// 
/// A mutable matrix view of the vector data.
/// 
/// # Panics
/// 
/// Panics in debug mode if `vector.len() != nrows * ncols`.
/// 
/// # Example
/// 
/// ```rust
/// use riemannopt_manifolds::utils::vector_to_matrix_view_mut;
/// use nalgebra::DVector;
/// 
/// let mut vec = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
/// {
///     let mut mat = vector_to_matrix_view_mut(&mut vec, 2, 2);
///     mat[(0, 0)] *= 2.0;  // Modify through matrix view
/// }
/// assert_eq!(vec[0], 2.0);  // Change reflected in vector
/// ```
#[inline]
pub fn vector_to_matrix_view_mut<'a, T: Scalar>(
    vector: &'a mut DVector<T>,
    nrows: usize,
    ncols: usize,
) -> nalgebra::DMatrixViewMut<'a, T> {
    debug_assert_eq!(vector.len(), nrows * ncols,
                     "Vector length must equal nrows * ncols");
    
    // Create a mutable matrix view from the vector's data
    nalgebra::DMatrixViewMut::from_slice_generic(vector.as_mut_slice(), Dyn(nrows), Dyn(ncols))
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
/// use nalgebra::DMatrix;
/// 
/// let a = DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let b = DMatrix::from_vec(3, 2, vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
/// let mut c = DMatrix::zeros(2, 2);
/// 
/// // Compute C = A * B
/// gemm_inplace(1.0, &a, &b, 0.0, &mut c);
/// ```
#[inline]
pub fn gemm_inplace<T: Scalar>(
    alpha: T,
    a: &DMatrix<T>,
    b: &DMatrix<T>,
    beta: T,
    c: &mut DMatrix<T>,
) {
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
/// use nalgebra::{DMatrix, DVector};
/// 
/// let a = DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let x = DVector::from_vec(vec![7.0, 8.0, 9.0]);
/// let mut y = DVector::zeros(2);
/// 
/// // Compute y = A * x
/// gemv_inplace(1.0, &a, &x, 0.0, &mut y);
/// ```
#[inline]
pub fn gemv_inplace<T: Scalar>(
    alpha: T,
    a: &DMatrix<T>,
    x: &DVector<T>,
    beta: T,
    y: &mut DVector<T>,
) {
    y.gemv(alpha, a, x, beta);
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    
    #[test]
    fn test_vector_matrix_views() {
        let vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let matrix_view = vector_to_matrix_view(&vector, 3, 2);
        
        assert_eq!(matrix_view.nrows(), 3);
        assert_eq!(matrix_view.ncols(), 2);
        assert_eq!(matrix_view[(0, 0)], 1.0);
        assert_eq!(matrix_view[(2, 1)], 6.0);
        
        // Verify column-major order
        let matrix = DMatrix::from_vec(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(matrix_view, matrix);
    }
    
    #[test]
    fn test_matrix_vector_view() {
        let matrix = DMatrix::from_vec(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let vector_view = matrix_to_vector_view(&matrix);
        
        assert_eq!(vector_view.len(), 6);
        assert_eq!(vector_view[0], 1.0);
        assert_eq!(vector_view[5], 6.0);
    }
    
    #[test]
    fn test_mutable_view() {
        let mut vector = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        {
            let mut matrix_view = vector_to_matrix_view_mut(&mut vector, 2, 2);
            matrix_view[(0, 0)] = 10.0;
            matrix_view[(1, 1)] = 40.0;
        }
        
        assert_eq!(vector[0], 10.0);
        assert_eq!(vector[3], 40.0);
    }
}