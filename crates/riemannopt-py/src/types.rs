//! Core type definitions for Python interoperability.
//!
//! This module defines the central types used throughout the Python bindings
//! to ensure consistent and efficient data handling between Python and Rust.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use nalgebra::{DVector, DMatrix};
use std::sync::Arc;

/// Represents a point on a manifold.
///
/// This enum unifies the representation of points across different manifold types,
/// handling both vector and matrix manifolds transparently.
#[derive(Clone, Debug)]
pub enum PyPoint {
    /// Point represented as a vector (for manifolds like Sphere, Hyperbolic)
    Vector(DVector<f64>),
    /// Point represented as a matrix (for manifolds like Stiefel, SPD)
    Matrix(DMatrix<f64>),
}

impl PyPoint {
    /// Create a PyPoint from a 1D NumPy array.
    pub fn from_numpy_1d(array: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
        let vec = DVector::from_row_slice(array.as_slice()?);
        Ok(PyPoint::Vector(vec))
    }

    /// Create a PyPoint from a 2D NumPy array.
    pub fn from_numpy_2d(array: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
        let shape = array.shape();
        let data = array.as_slice()?;
        let mat = DMatrix::from_row_slice(shape[0], shape[1], data);
        Ok(PyPoint::Matrix(mat))
    }

    /// Convert to a 1D NumPy array (for vector points).
    pub fn to_numpy_1d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        match self {
            PyPoint::Vector(vec) => Ok(PyArray1::from_slice_bound(py, vec.as_slice())),
            PyPoint::Matrix(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Cannot convert matrix point to 1D array"
            )),
        }
    }

    /// Convert to a 2D NumPy array (for matrix points).
    pub fn to_numpy_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        match self {
            PyPoint::Matrix(mat) => {
                let shape = (mat.nrows(), mat.ncols());
                let mut vec2d = Vec::with_capacity(shape.0);
                for i in 0..shape.0 {
                    let mut row = Vec::with_capacity(shape.1);
                    for j in 0..shape.1 {
                        row.push(mat[(i, j)]);
                    }
                    vec2d.push(row);
                }
                Ok(PyArray2::from_vec2_bound(py, &vec2d)?)
            },
            PyPoint::Vector(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Cannot convert vector point to 2D array"
            )),
        }
    }

    /// Get dimensions of the point.
    pub fn shape(&self) -> Vec<usize> {
        match self {
            PyPoint::Vector(vec) => vec![vec.len()],
            PyPoint::Matrix(mat) => vec![mat.nrows(), mat.ncols()],
        }
    }

    /// Check if this is a vector point.
    pub fn is_vector(&self) -> bool {
        matches!(self, PyPoint::Vector(_))
    }

    /// Check if this is a matrix point.
    pub fn is_matrix(&self) -> bool {
        matches!(self, PyPoint::Matrix(_))
    }

    /// Get as vector reference (returns None if matrix).
    pub fn as_vector(&self) -> Option<&DVector<f64>> {
        match self {
            PyPoint::Vector(v) => Some(v),
            _ => None,
        }
    }

    /// Get as matrix reference (returns None if vector).
    pub fn as_matrix(&self) -> Option<&DMatrix<f64>> {
        match self {
            PyPoint::Matrix(m) => Some(m),
            _ => None,
        }
    }
}

/// Represents a tangent vector on a manifold.
///
/// This type is essentially the same as PyPoint but with semantic distinction
/// to make the API clearer and type-safe.
pub type PyTangentVector = PyPoint;

/// Shared workspace for computations.
///
/// This struct wraps the workspace in an Arc to allow safe sharing between
/// Python objects while maintaining Rust's memory safety guarantees.
pub struct PyWorkspace {
    /// The actual workspace wrapped in Arc for thread-safe sharing
    inner: Arc<parking_lot::RwLock<riemannopt_core::memory::workspace::Workspace<f64>>>,
}

impl PyWorkspace {
    /// Create a new workspace.
    pub fn new() -> Self {
        PyWorkspace {
            inner: Arc::new(parking_lot::RwLock::new(
                riemannopt_core::memory::workspace::Workspace::<f64>::new()
            )),
        }
    }

    /// Get a reference to the inner workspace.
    pub fn with<F, R>(&self, f: F) -> R
    where
        F: FnOnce(&mut riemannopt_core::memory::workspace::Workspace<f64>) -> R,
    {
        let mut guard = self.inner.write();
        f(&mut *guard)
    }
}

impl Default for PyWorkspace {
    fn default() -> Self {
        Self::new()
    }
}

/// Type alias for manifold dimension information.
pub type ManifoldDimension = usize;

/// Configuration for numerical tolerances.
#[derive(Clone, Debug)]
pub struct NumericalConfig {
    /// Tolerance for convergence checks
    pub tolerance: f64,
    /// Maximum number of iterations
    pub max_iterations: usize,
    /// Tolerance for line search
    pub line_search_tolerance: f64,
    /// Whether to use parallel computation
    pub use_parallel: bool,
}

impl Default for NumericalConfig {
    fn default() -> Self {
        NumericalConfig {
            tolerance: 1e-6,
            max_iterations: 1000,
            line_search_tolerance: 1e-4,
            use_parallel: true,
        }
    }
}