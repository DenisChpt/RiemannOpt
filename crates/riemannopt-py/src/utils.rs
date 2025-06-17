//! Utility functions for Python bindings.
//!
//! This module provides helper functions for array conversion,
//! type checking, and other utilities.

#![allow(dead_code)]

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix};

/// Convert a Python list or numpy array to a DVector.
pub fn to_dvector(obj: &Bound<'_, PyAny>) -> PyResult<DVector<f64>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray1<'_, f64>>() {
        Ok(DVector::from_column_slice(array.as_slice()?))
    } else if let Ok(list) = obj.extract::<Vec<f64>>() {
        Ok(DVector::from_vec(list))
    } else {
        Err(PyValueError::new_err("Expected numpy array or list of floats"))
    }
}

/// Convert a Python 2D array to a DMatrix.
pub fn to_dmatrix(obj: &Bound<'_, PyAny>) -> PyResult<DMatrix<f64>> {
    if let Ok(array) = obj.extract::<PyReadonlyArray2<'_, f64>>() {
        let shape = array.shape();
        Ok(DMatrix::from_row_slice(shape[0], shape[1], array.as_slice()?))
    } else if let Ok(nested_list) = obj.extract::<Vec<Vec<f64>>>() {
        if nested_list.is_empty() {
            return Err(PyValueError::new_err("Empty matrix"));
        }
        let nrows = nested_list.len();
        let ncols = nested_list[0].len();
        
        // Check rectangular
        for row in &nested_list {
            if row.len() != ncols {
                return Err(PyValueError::new_err("Matrix must be rectangular"));
            }
        }
        
        let mut data = Vec::with_capacity(nrows * ncols);
        for row in nested_list {
            data.extend(row);
        }
        
        Ok(DMatrix::from_row_slice(nrows, ncols, &data))
    } else {
        Err(PyValueError::new_err("Expected 2D numpy array or nested list"))
    }
}

/// Convert a DVector to a numpy array.
pub fn dvector_to_numpy<'py>(py: Python<'py>, vec: &DVector<f64>) -> Bound<'py, PyArray1<f64>> {
    numpy::PyArray1::from_slice_bound(py, vec.as_slice())
}

/// Convert a DMatrix to a numpy array.
pub fn dmatrix_to_numpy<'py>(py: Python<'py>, mat: &DMatrix<f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (nrows, ncols) = mat.shape();
    let data: Vec<f64> = mat.as_slice().to_vec();
    let arr = numpy::PyArray1::from_slice_bound(py, &data);
    Ok(arr.reshape([nrows, ncols])?)
}

/// Check if an object is a valid manifold.
pub fn is_manifold(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("project").unwrap_or(false) &&
    obj.hasattr("retract").unwrap_or(false) &&
    obj.hasattr("tangent_projection").unwrap_or(false) &&
    obj.hasattr("dim").unwrap_or(false)
}

/// Check if an object is a valid optimizer.
pub fn is_optimizer(obj: &Bound<'_, PyAny>) -> bool {
    obj.hasattr("optimize").unwrap_or(false) &&
    obj.hasattr("step").unwrap_or(false) &&
    obj.hasattr("config").unwrap_or(false)
}

/// Create a progress callback for Python.
pub struct PyProgressCallback {
    callback: Option<PyObject>,
}

impl PyProgressCallback {
    pub fn new(callback: Option<PyObject>) -> Self {
        Self { callback }
    }
    
    pub fn call(&self, iteration: usize, value: f64, gradient_norm: f64) -> PyResult<()> {
        if let Some(ref cb) = self.callback {
            Python::with_gil(|py| {
                cb.call1(py, (iteration, value, gradient_norm))?;
                Ok(())
            })
        } else {
            Ok(())
        }
    }
}

/// Format optimization result for display.
#[pyfunction]
pub fn format_result(result: &Bound<'_, pyo3::types::PyDict>) -> PyResult<String> {
    let iterations = result.get_item("iterations")?.unwrap().extract::<usize>()?;
    let value = result.get_item("value")?.unwrap().extract::<f64>()?;
    let converged = result.get_item("converged")?.unwrap().extract::<bool>()?;
    
    Ok(format!(
        "Optimization Result:\n  Iterations: {}\n  Final value: {:.6e}\n  Converged: {}",
        iterations, value, converged
    ))
}

/// Validate manifold-point compatibility.
#[pyfunction]
pub fn validate_point_shape(
    manifold: &Bound<'_, PyAny>,
    point: &Bound<'_, PyAny>,
) -> PyResult<()> {
    // Check sphere
    if let Ok(sphere) = manifold.extract::<PyRef<crate::manifolds::PySphere>>() {
        if let Ok(array) = point.extract::<PyReadonlyArray1<'_, f64>>() {
            let arr_len = array.len();
            if arr_len != sphere.ambient_dim() {
                return Err(PyValueError::new_err(format!(
                    "Point dimension {} doesn't match sphere ambient dimension {}",
                    arr_len, sphere.ambient_dim()
                )));
            }
            return Ok(());
        }
    }
    
    // Check Stiefel
    if let Ok(stiefel) = manifold.extract::<PyRef<crate::manifolds::PyStiefel>>() {
        if let Ok(array) = point.extract::<PyReadonlyArray2<'_, f64>>() {
            let shape = array.shape();
            if shape[0] != stiefel.n() || shape[1] != stiefel.p() {
                return Err(PyValueError::new_err(format!(
                    "Point shape {:?} doesn't match Stiefel({}, {})",
                    shape, stiefel.n(), stiefel.p()
                )));
            }
            return Ok(());
        }
    }
    
    Err(PyValueError::new_err("Unsupported manifold or point type"))
}

/// Create a simple line search callback.
#[pyfunction]
pub fn default_line_search() -> PyResult<PyObject> {
    Python::with_gil(|py| {
        let code = r#"
def line_search(f, x, direction, initial_step=1.0, c1=1e-4, max_iter=20):
    '''Simple backtracking line search.'''
    step = initial_step
    f0 = f(x)
    
    for _ in range(max_iter):
        x_new = x + step * direction
        if f(x_new) < f0 - c1 * step * np.dot(direction, direction):
            return step
        step *= 0.5
    
    return step
"#;
        
        let globals = pyo3::types::PyDict::new_bound(py);
        let numpy = py.import_bound("numpy")?;
        globals.set_item("np", numpy)?;
        
        py.run_bound(code, Some(&globals), None)?;
        
        Ok(globals.get_item("line_search")?.unwrap().to_object(py))
    })
}