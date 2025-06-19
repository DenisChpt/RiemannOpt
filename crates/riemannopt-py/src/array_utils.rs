//! Utilities for efficient NumPy array conversions without unnecessary copies.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use nalgebra::{DVector, DMatrix};
use pyo3::exceptions::PyValueError;

/// Convert a nalgebra DVector to a NumPy array efficiently.
/// This avoids the intermediate to_vec() conversion.
pub fn dvector_to_pyarray<'py>(py: Python<'py>, vec: &DVector<f64>) -> Bound<'py, PyArray1<f64>> {
    // Create array from iterator to avoid intermediate Vec allocation
    vec.iter().copied().collect::<Vec<_>>().into_pyarray_bound(py)
}

/// Convert a nalgebra DMatrix to a NumPy array efficiently in row-major order.
pub fn dmatrix_to_pyarray<'py>(py: Python<'py>, mat: &DMatrix<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (nrows, ncols) = mat.shape();
    let mut data = Vec::with_capacity(nrows * ncols);
    
    // Copy in row-major order for NumPy
    for i in 0..nrows {
        for j in 0..ncols {
            data.push(mat[(i, j)]);
        }
    }
    
    let arr = PyArray1::from_vec_bound(py, data);
    arr.reshape([nrows, ncols])
}

/// Convert a NumPy 2D array to nalgebra DMatrix efficiently.
pub fn pyarray_to_dmatrix(arr: &PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let shape = arr.shape();
    let slice = arr.as_slice()?;
    
    // Create matrix and copy data (handling row-major to column-major conversion)
    let mut mat = DMatrix::zeros(shape[0], shape[1]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            mat[(i, j)] = slice[i * shape[1] + j];
        }
    }
    
    Ok(mat)
}

/// Convert DMatrix to DVector (column-major flattening) without copy.
pub fn dmatrix_to_dvector(mat: &DMatrix<f64>) -> DVector<f64> {
    // Create a view without copying
    DVector::from_column_slice(mat.as_slice())
}

/// Convert DVector to DMatrix with given shape without copy.
pub fn dvector_to_dmatrix(vec: &DVector<f64>, nrows: usize, ncols: usize) -> DMatrix<f64> {
    // Create from slice without intermediate Vec
    DMatrix::from_column_slice(nrows, ncols, vec.as_slice())
}

/// Optimized matrix operation for Stiefel/Grassmann manifolds.
/// Applies a function that operates on vectors to matrix data.
pub fn apply_matrix_op_via_vec<'py, F>(
    py: Python<'py>,
    matrix: &PyReadonlyArray2<'_, f64>,
    op: F,
) -> PyResult<Bound<'py, PyArray2<f64>>>
where
    F: FnOnce(&DVector<f64>) -> Result<DVector<f64>, Box<dyn std::error::Error>>,
{
    let shape = matrix.shape();
    
    // Convert numpy to DMatrix efficiently
    let mat = pyarray_to_dmatrix(matrix)?;
    
    // Convert to vector for operation
    let vec = dmatrix_to_dvector(&mat);
    
    // Apply operation
    let result_vec = op(&vec).map_err(|e| PyValueError::new_err(e.to_string()))?;
    
    // Convert back to matrix
    let result_mat = dvector_to_dmatrix(&result_vec, shape[0], shape[1]);
    
    // Convert to numpy
    dmatrix_to_pyarray(py, &result_mat)
}