//! `NumPy` <-> faer zero-copy bridge.
//!
//! Two copies per `solve()` call: input (numpy -> faer) and output (faer -> numpy).
//! The hot loop runs entirely in Rust with no Python objects.

use numpy::{
	PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2,
	PyUntypedArrayMethods,
};
use pyo3::prelude::*;
use riemannopt_core::linalg::{MatrixOps, MatrixView, VectorOps};

pub type T = f64;
pub type B = riemannopt_core::linalg::DefaultBackend;
pub type Vec64 = <B as riemannopt_core::linalg::LinAlgBackend<T>>::Vector;
pub type Mat64 = <B as riemannopt_core::linalg::LinAlgBackend<T>>::Matrix;

/// Copy a 1D `NumPy` array into a faer Col<f64>.
pub fn numpy_1d_to_col(arr: PyReadonlyArray1<'_, f64>) -> Vec64 {
	let slice = arr
		.as_slice()
		.expect("NumPy array must be contiguous (C or F order)");
	Vec64::from_slice(slice)
}

/// Copy a faer Col<f64> into a new 1D `NumPy` array.
pub fn col_to_numpy_1d<'py>(py: Python<'py>, col: &Vec64) -> Bound<'py, PyArray1<f64>> {
	PyArray1::from_slice(py, col.as_slice())
}

/// Copy a 2D `NumPy` array into a faer Mat<f64>.
///
/// Handles both C-contiguous (row-major) and F-contiguous (column-major) layouts.
/// Falls back to element-wise copy for strided arrays.
pub fn numpy_2d_to_mat(arr: PyReadonlyArray2<'_, f64>) -> Mat64 {
	let shape = arr.shape();
	let nrows = shape[0];
	let ncols = shape[1];
	let arr_ref = arr.as_array();

	// Try to get a contiguous column-major slice
	if let Some(slice) = arr_ref.as_slice() {
		// Check if it's actually column-major (Fortran order)
		let strides = arr.strides();
		if strides[0] == 1 || (nrows == 1) {
			// Column-major or single row — direct copy
			return Mat64::from_column_slice(nrows, ncols, slice);
		}
	}

	// General case: element-wise copy (handles C-contiguous and strided)
	Mat64::from_fn(nrows, ncols, |i, j| arr_ref[[i, j]])
}

/// Copy a faer Mat<f64> into a new 2D `NumPy` array (row-major / C order).
pub fn mat_to_numpy_2d<'py>(py: Python<'py>, mat: &Mat64) -> Bound<'py, PyArray2<f64>> {
	let nrows = mat.nrows();
	let ncols = mat.ncols();

	// Collect in row-major order (C order) — matches NumPy default
	let mut data = std::vec::Vec::with_capacity(nrows * ncols);
	for i in 0..nrows {
		for j in 0..ncols {
			data.push(MatrixView::get(mat, i, j));
		}
	}

	// Build 1D then reshape to 2D (C-order is the default)
	let flat = PyArray1::from_vec(py, data);
	flat.reshape([nrows, ncols])
		.expect("reshape from flat to 2D should succeed")
}
