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
/// 
/// # Performance
/// - C-contiguous arrays: O(1) - direct slice reference
/// - Non-contiguous arrays: O(n) - requires copy
pub fn numpy_to_dvector(array: PyReadonlyArray1<'_, f64>) -> PyResult<DVector<f64>> {
    let len = array.len();
    
    // Check if array is contiguous for potential zero-copy
    if array.is_c_contiguous() {
        // Safe to use as_slice for contiguous arrays - zero copy!
        let slice = array.as_slice()?;
        Ok(DVector::from_row_slice(slice))
    } else {
        // Need to copy non-contiguous data
        // Pre-allocate to avoid reallocation
        let mut vec = Vec::with_capacity(len);

        for i in 0..len {
            vec.push(*array.get([i]).ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds"))?);
        }
        Ok(DVector::from_vec(vec))
    }
}

/// Convert a 2D NumPy array to a nalgebra DMatrix.
///
/// This function handles both row-major (C-order) and column-major (F-order) arrays.
/// 
/// # Performance
/// - C-contiguous arrays: O(1) - direct slice reference
/// - F-contiguous arrays: O(1) - direct slice reference
/// - Non-contiguous arrays: O(m*n) - requires copy
pub fn numpy_to_dmatrix(array: PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let shape = array.shape();
    let nrows = shape[0];
    let ncols = shape[1];
    
    if array.is_c_contiguous() {
        // Row-major order - can use slice directly - zero copy!
        let slice = array.as_slice()?;
        Ok(DMatrix::from_row_slice(nrows, ncols, slice))
    } else if array.is_fortran_contiguous() {
        // Column-major order - can use slice directly with column_slice - zero copy!
        let slice = array.as_slice()?;
        Ok(DMatrix::from_column_slice(nrows, ncols, slice))
    } else {
        // Non-contiguous - need to copy
        // Pre-allocate exact size to avoid reallocation
        let mut data = Vec::with_capacity(nrows * ncols);
        
        // Use unsafe block for faster iteration
        unsafe {
            data.set_len(nrows * ncols);
            let ptr = data.as_mut_ptr() as *mut f64;
            
            for i in 0..nrows {
                for j in 0..ncols {
                    *ptr.add(i * ncols + j) = *array.get([i, j])
                        .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>(
                            format!("Index out of bounds: [{}, {}]", i, j)
                        ))?;
                }
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
/// 
/// # Performance
/// - Always O(m*n) as we need to copy data
/// - Uses unsafe code for optimal performance
pub fn dmatrix_to_numpy<'py>(
    py: Python<'py>,
    matrix: &DMatrix<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let nrows = matrix.nrows();
    let ncols = matrix.ncols();
    
    // nalgebra matrices are always stored in column-major order
    // Create C-order array (row-major) for Python compatibility
    let array = PyArray2::zeros_bound(py, [nrows, ncols], false);
    unsafe {
        let ptr = array.as_array_mut().as_mut_ptr() as *mut f64;
        
        // Copy row by row for C-order output
        for (i, row) in matrix.row_iter().enumerate() {
            let row_ptr = ptr.add(i * ncols);
            for (j, &value) in row.iter().enumerate() {
                *row_ptr.add(j) = value;
            }
        }
    }
    Ok(array)
}