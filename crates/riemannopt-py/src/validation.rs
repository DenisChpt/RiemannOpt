//! Validation utilities for Python bindings
//!
//! This module provides helper functions for validating inputs from Python
//! with clear, informative error messages.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix};

/// Validate that an array has the expected dimension
pub fn validate_array_dim_1d(
    array: &PyReadonlyArray1<'_, f64>,
    expected: usize,
    name: &str,
    context: &str,
) -> PyResult<()> {
    let shape = array.shape();
    if shape[0] != expected {
        return Err(PyValueError::new_err(format!(
            "{} dimension mismatch. Expected {}, got {}. {}",
            name, expected, shape[0], context
        )));
    }
    Ok(())
}

/// Validate that a matrix has the expected shape
pub fn validate_array_shape_2d(
    array: &PyReadonlyArray2<'_, f64>,
    expected_rows: usize,
    expected_cols: usize,
    name: &str,
    context: &str,
) -> PyResult<()> {
    let shape = array.shape();
    if shape[0] != expected_rows || shape[1] != expected_cols {
        return Err(PyValueError::new_err(format!(
            "{} shape mismatch. Expected ({}, {}), got ({}, {}). {}",
            name, expected_rows, expected_cols, shape[0], shape[1], context
        )));
    }
    Ok(())
}

/// Validate that a vector has non-zero norm
pub fn validate_nonzero_vector(vec: &DVector<f64>, name: &str) -> PyResult<()> {
    let norm = vec.norm();
    if norm < 1e-10 {
        return Err(PyValueError::new_err(format!(
            "{} must have non-zero norm. Got norm = {:.2e}",
            name, norm
        )));
    }
    Ok(())
}

/// Validate that a point is on the unit sphere
pub fn validate_on_sphere(point: &DVector<f64>, tolerance: f64) -> PyResult<()> {
    let norm = point.norm();
    if (norm - 1.0).abs() > tolerance {
        return Err(PyValueError::new_err(format!(
            "Point is not on the unit sphere. Norm = {:.6}, expected 1.0 (tolerance = {:.1e}). \
             Consider using project() to map the point onto the sphere.",
            norm, tolerance
        )));
    }
    Ok(())
}

/// Validate that a matrix has orthonormal columns (is on Stiefel manifold)
pub fn validate_on_stiefel(matrix: &DMatrix<f64>, tolerance: f64) -> PyResult<()> {
    let p = matrix.ncols();
    let gram = matrix.transpose() * matrix;
    let identity = DMatrix::<f64>::identity(p, p);
    let error = (&gram - &identity).norm();
    
    if error > tolerance {
        return Err(PyValueError::new_err(format!(
            "Matrix does not have orthonormal columns. ||X^T X - I|| = {:.6e}, tolerance = {:.1e}. \
             Consider using project() to orthonormalize the columns.",
            error, tolerance
        )));
    }
    Ok(())
}

/// Validate that a matrix is symmetric positive definite
pub fn validate_spd(matrix: &DMatrix<f64>, tolerance: f64) -> PyResult<()> {
    // Check symmetry
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(PyValueError::new_err(format!(
            "SPD matrix must be square. Got shape ({}, {})",
            matrix.nrows(), matrix.ncols()
        )));
    }
    
    let symmetry_error = (matrix - matrix.transpose()).norm();
    if symmetry_error > tolerance {
        return Err(PyValueError::new_err(format!(
            "Matrix is not symmetric. ||X - X^T|| = {:.6e}, tolerance = {:.1e}",
            symmetry_error, tolerance
        )));
    }
    
    // Check positive definiteness via Cholesky
    if matrix.clone().cholesky().is_none() {
        return Err(PyValueError::new_err(
            "Matrix is not positive definite. Cholesky decomposition failed. \
             All eigenvalues must be positive."
        ));
    }
    
    Ok(())
}

/// Validate that a tangent vector is in the tangent space
pub fn validate_tangent_sphere(
    point: &DVector<f64>,
    tangent: &DVector<f64>,
    tolerance: f64,
) -> PyResult<()> {
    let inner_prod = point.dot(tangent);
    if inner_prod.abs() > tolerance {
        return Err(PyValueError::new_err(format!(
            "Vector is not in the tangent space. <point, tangent> = {:.6e}, expected 0 (tolerance = {:.1e}). \
             Tangent vectors at a point on the sphere must be orthogonal to that point.",
            inner_prod, tolerance
        )));
    }
    Ok(())
}

/// Validate that step size is positive
pub fn validate_positive_parameter(value: f64, name: &str) -> PyResult<()> {
    if value <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "{} must be positive. Got {}",
            name, value
        )));
    }
    Ok(())
}

/// Format a dimension error message with helpful context
pub fn dimension_error_message(
    expected: usize,
    got: usize,
    manifold_name: &str,
    parameter_name: &str,
) -> String {
    format!(
        "{} dimension mismatch for {}. Expected dimension {}, got {}. \
         The {} manifold requires {}-dimensional {}.",
        parameter_name, manifold_name, expected, got,
        manifold_name, expected, parameter_name
    )
}

/// Format a shape error message for matrices
pub fn shape_error_message(
    expected: (usize, usize),
    got: (usize, usize),
    manifold_name: &str,
    parameter_name: &str,
) -> String {
    format!(
        "{} shape mismatch for {}. Expected ({}, {}), got ({}, {}). \
         The {} manifold requires matrices of shape ({}, {}).",
        parameter_name, manifold_name, expected.0, expected.1, got.0, got.1,
        manifold_name, expected.0, expected.1
    )
}