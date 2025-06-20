//! Utility functions for efficient matrix/vector operations without unnecessary allocations

use nalgebra::{DMatrix, DVector, DVectorView, DMatrixView, Dyn};
use riemannopt_core::types::Scalar;

/// Creates a matrix view from a vector without cloning
/// 
/// # Safety
/// The caller must ensure that the vector has exactly nrows * ncols elements
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

/// Creates a vector view from a matrix without cloning
#[inline]
pub fn matrix_to_vector_view<'a, T: Scalar>(
    matrix: &'a DMatrix<T>,
) -> DVectorView<'a, T> {
    let len = matrix.nrows() * matrix.ncols();
    
    // Create a vector view from the matrix's data
    DVectorView::from_slice_generic(matrix.as_slice(), Dyn(len), nalgebra::U1)
}

/// Reshapes a mutable vector as a matrix view
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

/// In-place matrix multiplication: C = alpha * A * B + beta * C
/// 
/// This avoids allocating a new matrix for the result
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

/// In-place matrix-vector multiplication: y = alpha * A * x + beta * y
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