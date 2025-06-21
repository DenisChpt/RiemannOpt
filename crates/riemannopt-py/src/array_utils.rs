//! Utilities for efficient NumPy array conversions without unnecessary copies.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use nalgebra::{DVector, DMatrix};

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