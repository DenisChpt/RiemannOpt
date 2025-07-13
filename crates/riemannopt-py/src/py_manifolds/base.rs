//! Base traits and utilities for Python manifold wrappers.
//!
//! This module provides the common interface and shared functionality
//! for all manifold wrappers, ensuring consistency and code reuse.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};

use crate::types::PyPoint;

/// Common trait for all Python manifold wrappers.
///
/// This trait defines the standard interface that all manifold wrappers
/// must implement to ensure consistency across the API.
pub trait PyManifoldBase {
    /// Get the manifold name for error messages.
    fn manifold_name(&self) -> &'static str;
    
    /// Get the ambient dimension of the manifold.
    fn ambient_dim(&self) -> usize;
    
    /// Get the intrinsic dimension of the manifold.
    fn intrinsic_dim(&self) -> usize;
    
    /// Validate that a point has the correct shape for this manifold.
    fn validate_point_shape(&self, point: &PyPoint) -> PyResult<()> {
        match (self.point_type(), point) {
            (PointType::Vector(expected_dim), PyPoint::Vector(vec)) => {
                if vec.len() != expected_dim {
                    return Err(crate::error::dimension_mismatch(
                        &[expected_dim],
                        &[vec.len()],
                    ));
                }
            }
            (PointType::Matrix(rows, cols), PyPoint::Matrix(mat)) => {
                if mat.nrows() != rows || mat.ncols() != cols {
                    return Err(crate::error::dimension_mismatch(
                        &[rows, cols],
                        &[mat.nrows(), mat.ncols()],
                    ));
                }
            }
            _ => {
                return Err(crate::error::type_error(
                    &format!("{} point", self.manifold_name()),
                    &format!("{:?} point", point),
                ));
            }
        }
        Ok(())
    }
    
    /// Get the expected point type for this manifold.
    fn point_type(&self) -> PointType;
}

/// Enum describing the expected point type for a manifold.
#[derive(Debug, Clone, Copy)]
pub enum PointType {
    /// Vector manifold with given dimension
    Vector(usize),
    /// Matrix manifold with given shape
    Matrix(usize, usize),
}

/// Helper macro to implement common Python methods for manifolds.
///
/// This macro reduces boilerplate by implementing standard methods
/// like `__repr__`, `dim`, `ambient_dim`, etc.
/// 
/// NOTE: This macro generates methods to be included in an existing #[pymethods] block.
/// Use within the #[pymethods] impl block, not as a standalone macro.
#[macro_export]
macro_rules! impl_py_manifold_methods {
    ($manifold_type:ty) => {
            /// String representation of the manifold.
            fn __repr__(&self) -> String {
                format!(
                    "{}(ambient_dim={}, intrinsic_dim={})",
                    self.manifold_name(),
                    self.ambient_dim(),
                    self.intrinsic_dim()
                )
            }
            
            /// Get the intrinsic dimension of the manifold.
            #[getter]
            fn dim(&self) -> usize {
                self.intrinsic_dim()
            }
            
            /// Get the ambient dimension of the manifold.
            #[getter]
            fn ambient_dim(&self) -> usize {
                PyManifoldBase::ambient_dim(self)
            }
            
            /// Check if a point lies on the manifold.
            ///
            /// Args:
            ///     point: Point to check
            ///     atol: Absolute tolerance (default: 1e-10)
            ///
            /// Returns:
            ///     bool: True if point is on manifold
            #[pyo3(signature = (point, atol=1e-10))]
            fn contains(&self, py: Python<'_>, point: PyObject, atol: f64) -> PyResult<bool> {
                let point = self.parse_point(py, point)?;
                self.validate_point_shape(&point)?;
                
                match &point {
                    PyPoint::Vector(vec) => {
                        self.contains_vector(vec, atol)
                    }
                    PyPoint::Matrix(mat) => {
                        self.contains_matrix(mat, atol)
                    }
                }
            }
            
            /// Check if a vector is in the tangent space.
            ///
            /// Args:
            ///     point: Base point on the manifold
            ///     vector: Vector to check
            ///     atol: Absolute tolerance (default: 1e-10)
            ///
            /// Returns:
            ///     bool: True if vector is in tangent space
            #[pyo3(signature = (point, vector, atol=1e-10))]
            fn is_tangent(&self, py: Python<'_>, point: PyObject, vector: PyObject, atol: f64) -> PyResult<bool> {
                let point = self.parse_point(py, point)?;
                let vector = self.parse_point(py, vector)?;
                self.validate_point_shape(&point)?;
                self.validate_point_shape(&vector)?;
                
                match (&point, &vector) {
                    (PyPoint::Vector(p), PyPoint::Vector(v)) => {
                        self.is_tangent_vector(p, v, atol)
                    }
                    (PyPoint::Matrix(p), PyPoint::Matrix(v)) => {
                        self.is_tangent_matrix(p, v, atol)
                    }
                    _ => Err(crate::error::type_error(
                        "matching point and vector types",
                        "mismatched types",
                    )),
                }
            }
    };
}


/// Utility function to convert Python array to PyPoint.
pub fn array_to_point(py: Python<'_>, array: PyObject) -> PyResult<PyPoint> {
    if let Ok(array_1d) = array.downcast_bound::<PyArray1<f64>>(py) {
        PyPoint::from_numpy_1d(array_1d.readonly())
    } else if let Ok(array_2d) = array.downcast_bound::<PyArray2<f64>>(py) {
        PyPoint::from_numpy_2d(array_2d.readonly())
    } else {
        Err(crate::error::type_error(
            "numpy array (1D or 2D)",
            "non-array object",
        ))
    }
}