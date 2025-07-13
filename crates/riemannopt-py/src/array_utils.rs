//! Utilities for efficient conversion between NumPy arrays and nalgebra structures.
//!
//! This module provides zero-copy (when possible) conversions between Python's NumPy
//! arrays and Rust's nalgebra vectors/matrices, ensuring maximum performance.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, PyArrayMethods};
use nalgebra::{DVector, DMatrix};

/// Error type for array conversion operations.
#[derive(Debug)]
pub enum ArrayConversionError {
    /// Shape mismatch between source and target
    ShapeMismatch { expected: Vec<usize>, got: Vec<usize> },
    /// Invalid data layout (e.g., non-contiguous array)
    InvalidLayout(String),
    /// Dimension error
    DimensionError(String),
}

impl std::fmt::Display for ArrayConversionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArrayConversionError::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            ArrayConversionError::InvalidLayout(msg) => write!(f, "Invalid array layout: {}", msg),
            ArrayConversionError::DimensionError(msg) => write!(f, "Dimension error: {}", msg),
        }
    }
}

impl std::error::Error for ArrayConversionError {}

impl From<ArrayConversionError> for PyErr {
    fn from(err: ArrayConversionError) -> PyErr {
        pyo3::exceptions::PyValueError::new_err(err.to_string())
    }
}

/// Convert a 1D NumPy array to a nalgebra DVector.
///
/// This function attempts to create a DVector with minimal copying.
/// If the array is C-contiguous, it will use the data directly.
pub fn numpy_to_dvector(array: PyReadonlyArray1<'_, f64>) -> PyResult<DVector<f64>> {
    let len = array.len();
    
    // Check if array is contiguous for potential zero-copy
    if array.is_c_contiguous() {
        // Safe to use as_slice for contiguous arrays
        let slice = array.as_slice()?;
        Ok(DVector::from_row_slice(slice))
    } else {
        // Need to copy non-contiguous data
        let mut vec = Vec::with_capacity(array.len());
        for i in 0..array.len() {
            vec.push(*array.get([i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"))?);
        }
        Ok(DVector::from_vec(vec))
    }
}

/// Convert a 2D NumPy array to a nalgebra DMatrix.
///
/// This function handles both row-major (C-order) and column-major (F-order) arrays.
pub fn numpy_to_dmatrix(array: PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let shape = array.shape();
    let nrows = shape[0];
    let ncols = shape[1];
    
    if array.is_c_contiguous() {
        // Row-major order - can use slice directly
        let slice = array.as_slice()?;
        Ok(DMatrix::from_row_slice(nrows, ncols, slice))
    } else if array.is_fortran_contiguous() {
        // Column-major order - can use slice directly with column_slice
        let slice = array.as_slice()?;
        Ok(DMatrix::from_column_slice(nrows, ncols, slice))
    } else {
        // Non-contiguous - need to copy
        let mut data = Vec::with_capacity(nrows * ncols);
        for i in 0..nrows {
            for j in 0..ncols {
                data.push(*array.get([i, j]).unwrap_or(&0.0));
            }
        }
        Ok(DMatrix::from_row_slice(nrows, ncols, &data))
    }
}

/// Convert a nalgebra DVector to a 1D NumPy array.
///
/// This creates a new NumPy array with the vector data.
pub fn dvector_to_numpy<'py>(
    py: Python<'py>,
    vector: &DVector<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    Ok(PyArray1::from_slice_bound(py, vector.as_slice()))
}

/// Convert a nalgebra DMatrix to a 2D NumPy array.
///
/// This creates a new NumPy array with the matrix data in row-major order.
pub fn dmatrix_to_numpy<'py>(
    py: Python<'py>,
    matrix: &DMatrix<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    
    // Create a new array and copy data
    let array = PyArray2::zeros_bound(py, [nrows, ncols], false);
    unsafe {
        let ptr = unsafe { array.as_array_mut().as_mut_ptr() as *mut f64 };
        for (i, row) in matrix.row_iter().enumerate() {
            for (j, &value) in row.iter().enumerate() {
                *ptr.add(i * ncols + j) = value;
            }
        }
    }
    
    Ok(array)
}

/// Validate that a 1D array has the expected length.
pub fn validate_array_1d(array: &PyReadonlyArray1<'_, f64>, expected_len: usize) -> PyResult<()> {
    let len = array.len();
    if len != expected_len {
        return Err(ArrayConversionError::ShapeMismatch {
            expected: vec![expected_len],
            got: vec![len],
        }.into());
    }
    Ok(())
}

/// Validate that a 2D array has the expected shape.
pub fn validate_array_2d(
    array: &PyReadonlyArray2<'_, f64>,
    expected_rows: usize,
    expected_cols: usize,
) -> PyResult<()> {
    let shape = array.shape();
    if shape[0] != expected_rows || shape[1] != expected_cols {
        return Err(ArrayConversionError::ShapeMismatch {
            expected: vec![expected_rows, expected_cols],
            got: vec![shape[0], shape[1]],
        }.into());
    }
    Ok(())
}

/// Create a view of a DVector as a slice, useful for FFI.
pub fn dvector_as_slice(vector: &DVector<f64>) -> &[f64] {
    vector.as_slice()
}

/// Create a mutable view of a DVector as a slice, useful for FFI.
pub fn dvector_as_mut_slice(vector: &mut DVector<f64>) -> &mut [f64] {
    vector.as_mut_slice()
}

/// Efficiently copy data from one DVector to another.
pub fn copy_dvector(src: &DVector<f64>, dst: &mut DVector<f64>) -> PyResult<()> {
    if src.len() != dst.len() {
        return Err(ArrayConversionError::ShapeMismatch {
            expected: vec![dst.len()],
            got: vec![src.len()],
        }.into());
    }
    dst.copy_from(src);
    Ok(())
}

/// Efficiently copy data from one DMatrix to another.
pub fn copy_dmatrix(src: &DMatrix<f64>, dst: &mut DMatrix<f64>) -> PyResult<()> {
    if src.shape() != dst.shape() {
        return Err(ArrayConversionError::ShapeMismatch {
            expected: vec![dst.nrows(), dst.ncols()],
            got: vec![src.nrows(), src.ncols()],
        }.into());
    }
    dst.copy_from(src);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{dvector, dmatrix};

    #[test]
    fn test_dvector_slice_conversion() {
        let vec = dvector![1.0, 2.0, 3.0, 4.0];
        let slice = dvector_as_slice(&vec);
        assert_eq!(slice, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_copy_dvector() {
        let src = dvector![1.0, 2.0, 3.0];
        let mut dst = dvector![0.0, 0.0, 0.0];
        copy_dvector(&src, &mut dst).unwrap();
        assert_eq!(dst, src);
    }

    #[test]
    fn test_copy_dmatrix() {
        let src = dmatrix![1.0, 2.0; 3.0, 4.0];
        let mut dst = DMatrix::zeros(2, 2);
        copy_dmatrix(&src, &mut dst).unwrap();
        assert_eq!(dst, src);
    }
}