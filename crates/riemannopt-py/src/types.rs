//! Core type definitions for Python interoperability.
//!
//! This module defines the central types used throughout the Python bindings
//! to ensure consistent and efficient data handling between Python and Rust.

use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use riemannopt_core::linalg::VectorOps;

use crate::array_utils::{Mat64, Vec64};

/// Represents a point on a manifold.
///
/// This enum unifies the representation of points across different manifold types,
/// handling both vector and matrix manifolds transparently.
#[derive(Clone, Debug)]
pub enum PyPoint {
	/// Point represented as a vector (for manifolds like Sphere, Hyperbolic)
	Vector(Vec64),
	/// Point represented as a matrix (for manifolds like Stiefel, SPD)
	Matrix(Mat64),
}

impl PyPoint {
	/// Create a PyPoint from a 1D NumPy array.
	pub fn from_numpy_1d(array: PyReadonlyArray1<'_, f64>) -> PyResult<Self> {
		let vec = crate::array_utils::numpy_to_vec(array)?;
		Ok(PyPoint::Vector(vec))
	}

	/// Create a PyPoint from a 2D NumPy array.
	///
	/// Handles both C-contiguous (row-major) and F-contiguous (column-major)
	/// arrays correctly by delegating to [`numpy_to_mat`].
	pub fn from_numpy_2d(array: PyReadonlyArray2<'_, f64>) -> PyResult<Self> {
		let mat = crate::array_utils::numpy_to_mat(array)?;
		Ok(PyPoint::Matrix(mat))
	}

	/// Convert to a 1D NumPy array (for vector points).
	pub fn to_numpy_1d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
		match self {
			PyPoint::Vector(vec) => Ok(PyArray1::from_slice(py, vec.as_slice())),
			PyPoint::Matrix(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
				"Cannot convert matrix point to 1D array",
			)),
		}
	}

	/// Convert to a 2D NumPy array (for matrix points).
	pub fn to_numpy_2d<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		match self {
			PyPoint::Matrix(mat) => crate::array_utils::mat_to_numpy(py, mat),
			PyPoint::Vector(_) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
				"Cannot convert vector point to 2D array",
			)),
		}
	}

	/// Get dimensions of the point.
	pub fn shape(&self) -> std::vec::Vec<usize> {
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
	pub fn as_vector(&self) -> Option<&Vec64> {
		match self {
			PyPoint::Vector(v) => Some(v),
			_ => None,
		}
	}

	/// Get as matrix reference (returns None if vector).
	pub fn as_matrix(&self) -> Option<&Mat64> {
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
