//! Utilities for efficient conversion between NumPy arrays and linalg types.
//!
//! This module provides conversions between Python's NumPy arrays and
//! the backend-agnostic `linalg::Vec<f64>` / `linalg::Mat<f64>` types.

use numpy::{
	PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use riemannopt_core::linalg::{MatrixOps, MatrixView, VectorOps};

/// The vector type for the active backend (e.g. `faer::Col<f64>` or `DVector<f64>`).
pub type Vec64 = riemannopt_core::linalg::Vec<f64>;
/// The matrix type for the active backend (e.g. `faer::Mat<f64>` or `DMatrix<f64>`).
pub type Mat64 = riemannopt_core::linalg::Mat<f64>;

/// Error type for array conversion operations.
#[derive(Debug)]
pub enum ArrayConversionError {
	/// Shape mismatch between source and target
	ShapeMismatch {
		expected: std::vec::Vec<usize>,
		got: std::vec::Vec<usize>,
	},
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

/// Convert a 1D NumPy array to a `linalg::Vec<f64>`.
///
/// # Performance
/// - C-contiguous arrays: uses `VectorOps::from_slice` on contiguous data
/// - Non-contiguous arrays: O(n) element-by-element copy
pub fn numpy_to_vec(array: PyReadonlyArray1<'_, f64>) -> PyResult<Vec64> {
	let len = array.len();

	if array.is_c_contiguous() {
		let slice = array.as_slice()?;
		Ok(VectorOps::from_slice(slice))
	} else {
		let mut data = std::vec::Vec::with_capacity(len);
		for i in 0..len {
			data.push(*array.get([i]).ok_or_else(|| {
				PyErr::new::<pyo3::exceptions::PyIndexError, _>("Index out of bounds")
			})?);
		}
		Ok(VectorOps::from_slice(&data))
	}
}

/// Convert a `linalg::Vec<f64>` to a 1D NumPy array.
///
/// Uses `VectorOps::as_slice()` which is contiguous for all backends.
pub fn vec_to_numpy<'py>(py: Python<'py>, vector: &Vec64) -> PyResult<Bound<'py, PyArray1<f64>>> {
	Ok(PyArray1::from_slice(py, vector.as_slice()))
}

/// Convert a 2D NumPy array to a `linalg::Mat<f64>`.
///
/// Uses element-by-element access via `MatrixOps::from_fn` to handle
/// any NumPy memory layout and any backend column stride.
///
/// # Performance
/// - O(m*n) element-wise copy regardless of layout
pub fn numpy_to_mat(array: PyReadonlyArray2<'_, f64>) -> PyResult<Mat64> {
	let shape = array.shape();
	let nrows = shape[0];
	let ncols = shape[1];

	Ok(MatrixOps::from_fn(nrows, ncols, |i, j| {
		*array.get([i, j]).unwrap_or(&0.0)
	}))
}

/// Convert a `linalg::Mat<f64>` to a 2D NumPy array.
///
/// Iterates element-by-element using `MatrixView::get(i, j)` to handle
/// backends where column stride may not equal nrows (e.g. faer).
///
/// The returned array is C-contiguous (row-major).
pub fn mat_to_numpy<'py>(py: Python<'py>, matrix: &Mat64) -> PyResult<Bound<'py, PyArray2<f64>>> {
	let nrows = matrix.nrows();
	let ncols = matrix.ncols();

	// Create a C-order (row-major) array
	let array = PyArray2::zeros(py, [nrows, ncols], false);
	unsafe {
		let ptr: *mut f64 = array.as_array_mut().as_mut_ptr();
		for i in 0..nrows {
			for j in 0..ncols {
				*ptr.add(i * ncols + j) = MatrixView::get(matrix, i, j);
			}
		}
	}
	Ok(array)
}
