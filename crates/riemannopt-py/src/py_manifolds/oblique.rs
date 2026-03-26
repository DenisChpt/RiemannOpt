//! Python bindings for the Oblique manifold.
//!
//! The oblique manifold is the product of unit spheres, where each column
//! of a matrix is constrained to have unit norm.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use riemannopt_core::linalg::MatrixOps;
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::oblique::Oblique;

use crate::{
	array_utils::{mat_to_numpy, numpy_to_mat, Mat64},
	error::to_py_err,
};

use super::base::{PointType, PyManifoldBase};

/// Python wrapper for the Oblique manifold.
///
/// The oblique manifold OB(n,p) consists of n×p matrices where each column
/// has unit norm: OB(n,p) = {X ∈ ℝ^{n×p} : diag(X^T X) = I_p}.
///
/// Parameters
/// ----------
/// n : int
///     Number of rows (ambient dimension per column)
/// p : int
///     Number of columns (number of unit spheres)
#[pyclass(name = "Oblique", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyOblique {
	/// The underlying Rust Oblique manifold
	pub(crate) inner: Oblique,
	/// Number of rows
	pub(crate) n: usize,
	/// Number of columns
	pub(crate) p: usize,
}

impl PyManifoldBase for PyOblique {
	fn manifold_name(&self) -> &'static str {
		"Oblique"
	}

	fn ambient_dim(&self) -> usize {
		self.n * self.p
	}

	fn intrinsic_dim(&self) -> usize {
		self.p * (self.n - 1)
	}

	fn point_type(&self) -> PointType {
		PointType::Matrix(self.n, self.p)
	}
}

#[pymethods]
impl PyOblique {
	/// Create a new Oblique manifold.
	///
	/// Args:
	///     n: Number of rows (must be >= 2)
	///     p: Number of columns (must be >= 1)
	///
	/// Raises:
	///     ValueError: If n < 2 or p < 1
	#[new]
	pub fn new(n: usize, p: usize) -> PyResult<Self> {
		if n < 2 {
			return Err(crate::error::value_error(
				"Oblique manifold requires n >= 2",
			));
		}
		if p < 1 {
			return Err(crate::error::value_error(
				"Oblique manifold requires p >= 1",
			));
		}

		Ok(PyOblique {
			inner: Oblique::new(n, p).map_err(crate::error::to_py_err)?,
			n,
			p,
		})
	}

	/// String representation of the manifold.
	fn __repr__(&self) -> String {
		format!("Oblique(n={}, p={})", self.n, self.p)
	}

	/// Get the number of rows.
	#[getter]
	pub fn n(&self) -> usize {
		self.n
	}

	/// Get the number of columns.
	#[getter]
	pub fn p(&self) -> usize {
		self.p
	}

	/// Check if a point is on the manifold.
	///
	/// Args:
	///     point: Matrix to check
	///     atol: Absolute tolerance (default: 1e-10)
	///
	/// Returns:
	///     True if the point is on the manifold (each column has unit norm)
	#[pyo3(signature = (point, atol=1e-10))]
	pub fn contains(&self, point: PyReadonlyArray2<'_, f64>, atol: f64) -> PyResult<bool> {
		let point_mat = numpy_to_mat(point)?;

		// Validate dimensions
		if point_mat.nrows() != self.n || point_mat.ncols() != self.p {
			return Ok(false);
		}

		Ok(<Oblique as Manifold<f64>>::is_point_on_manifold(
			&self.inner,
			&point_mat,
			atol,
		))
	}

	/// Project a point onto the manifold.
	///
	/// Normalizes each column to have unit norm.
	///
	/// Args:
	///     point: Matrix to project
	///
	/// Returns:
	///     Projected matrix on the Oblique manifold
	pub fn project<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;

		// Validate dimensions
		if point_mat.nrows() != self.n || point_mat.ncols() != self.p {
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p],
				&[point_mat.nrows(), point_mat.ncols()],
			));
		}

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		Manifold::<f64>::project_point(&self.inner, &point_mat, &mut result);

		mat_to_numpy(py, &result)
	}

	/// Project a tangent vector at a point.
	///
	/// Args:
	///     point: Base point on the manifold
	///     tangent: Tangent vector to project
	///
	/// Returns:
	///     Projected tangent vector
	pub fn project_tangent<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;
		let tangent_mat = numpy_to_mat(tangent)?;

		// Validate dimensions
		if (point_mat.nrows() != self.n || point_mat.ncols() != self.p)
			|| (tangent_mat.nrows() != self.n || tangent_mat.ncols() != self.p)
		{
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p],
				&[tangent_mat.nrows(), tangent_mat.ncols()],
			));
		}

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		Manifold::<f64>::project_tangent(
			&self.inner,
			&point_mat,
			&tangent_mat,
			&mut result,
			&mut (),
		)
		.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Compute the Riemannian exponential map.
	///
	/// Args:
	///     point: Base point on the manifold
	///     tangent: Tangent vector at the base point
	///
	/// Returns:
	///     Point on the manifold
	pub fn exp<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;
		let tangent_mat = numpy_to_mat(tangent)?;

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		<Oblique as Manifold<f64>>::retract(
			&self.inner,
			&point_mat,
			&tangent_mat,
			&mut result,
			&mut (),
		)
		.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Compute the Riemannian logarithm map.
	///
	/// Args:
	///     x: First point on the manifold
	///     y: Second point on the manifold
	///
	/// Returns:
	///     Tangent vector at x pointing towards y
	pub fn log<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray2<'_, f64>,
		y: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let x_mat = numpy_to_mat(x)?;
		let y_mat = numpy_to_mat(y)?;

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		<Oblique as Manifold<f64>>::inverse_retract(
			&self.inner,
			&x_mat,
			&y_mat,
			&mut result,
			&mut (),
		)
		.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Compute the retraction.
	///
	/// Args:
	///     point: Base point on the manifold
	///     tangent: Tangent vector at the base point
	///
	/// Returns:
	///     Point on the manifold
	pub fn retract<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;
		let tangent_mat = numpy_to_mat(tangent)?;

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		<Oblique as Manifold<f64>>::retract(
			&self.inner,
			&point_mat,
			&tangent_mat,
			&mut result,
			&mut (),
		)
		.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Compute the inverse retraction.
	///
	/// Args:
	///     x: First point on the manifold
	///     y: Second point on the manifold
	///
	/// Returns:
	///     Tangent vector at x
	pub fn inverse_retract<'py>(
		&self,
		py: Python<'py>,
		x: PyReadonlyArray2<'_, f64>,
		y: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let x_mat = numpy_to_mat(x)?;
		let y_mat = numpy_to_mat(y)?;

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		<Oblique as Manifold<f64>>::inverse_retract(
			&self.inner,
			&x_mat,
			&y_mat,
			&mut result,
			&mut (),
		)
		.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Compute the Riemannian inner product.
	///
	/// Args:
	///     point: Base point on the manifold
	///     u: First tangent vector
	///     v: Second tangent vector
	///
	/// Returns:
	///     Inner product value
	pub fn inner(
		&self,
		point: PyReadonlyArray2<'_, f64>,
		u: PyReadonlyArray2<'_, f64>,
		v: PyReadonlyArray2<'_, f64>,
	) -> PyResult<f64> {
		let point_mat = numpy_to_mat(point)?;
		let u_mat = numpy_to_mat(u)?;
		let v_mat = numpy_to_mat(v)?;

		Ok(self
			.inner
			.inner_product(&point_mat, &u_mat, &v_mat, &mut ())
			.map_err(to_py_err)?)
	}

	/// Compute the norm of a tangent vector.
	///
	/// Args:
	///     point: Base point on the manifold
	///     tangent: Tangent vector
	///
	/// Returns:
	///     Norm of the tangent vector
	pub fn norm(
		&self,
		point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<f64> {
		let point_mat = numpy_to_mat(point)?;
		let tangent_mat = numpy_to_mat(tangent)?;

		Ok(self
			.inner
			.norm(&point_mat, &tangent_mat, &mut ())
			.map_err(to_py_err)?)
	}

	/// Compute the Riemannian distance between two points.
	///
	/// Args:
	///     x: First point on the manifold
	///     y: Second point on the manifold
	///
	/// Returns:
	///     Distance between x and y
	pub fn distance(
		&self,
		x: PyReadonlyArray2<'_, f64>,
		y: PyReadonlyArray2<'_, f64>,
	) -> PyResult<f64> {
		let x_mat = numpy_to_mat(x)?;
		let y_mat = numpy_to_mat(y)?;

		Ok(<Oblique as Manifold<f64>>::distance(&self.inner, &x_mat, &y_mat).map_err(to_py_err)?)
	}

	/// Generate a random point on the manifold.
	///
	/// Returns:
	///     Random matrix with normalized columns
	pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		<Oblique as Manifold<f64>>::random_point(&self.inner, &mut result).map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Generate a random tangent vector at a point.
	///
	/// Args:
	///     point: Base point on the manifold
	///     scale: Standard deviation of the tangent vector (default: 1.0)
	///
	/// Returns:
	///     Random tangent vector at the point
	#[pyo3(signature = (point, scale=1.0))]
	pub fn random_tangent<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		scale: f64,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		<Oblique as Manifold<f64>>::random_tangent(&self.inner, &point_mat, &mut result)
			.map_err(to_py_err)?;

		// Scale the result if needed
		if scale != 1.0 {
			result.scale_mut(scale);
		}

		mat_to_numpy(py, &result)
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

	/// Parallel transport a tangent vector along a geodesic.
	///
	/// Args:
	///     from_point: Starting point
	///     to_point: Ending point
	///     tangent: Tangent vector at from_point
	///
	/// Returns:
	///     Transported tangent vector at to_point
	pub fn parallel_transport<'py>(
		&self,
		py: Python<'py>,
		from_point: PyReadonlyArray2<'_, f64>,
		to_point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let from_mat = numpy_to_mat(from_point)?;
		let to_mat = numpy_to_mat(to_point)?;
		let tangent_mat = numpy_to_mat(tangent)?;

		// Validate dimensions
		if (from_mat.nrows() != self.n || from_mat.ncols() != self.p)
			|| (to_mat.nrows() != self.n || to_mat.ncols() != self.p)
			|| (tangent_mat.nrows() != self.n || tangent_mat.ncols() != self.p)
		{
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p, self.n, self.p, self.n, self.p],
				&[
					from_mat.nrows(),
					from_mat.ncols(),
					to_mat.nrows(),
					to_mat.ncols(),
					tangent_mat.nrows(),
					tangent_mat.ncols(),
				],
			));
		}

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);
		<Oblique as Manifold<f64>>::parallel_transport(
			&self.inner,
			&from_mat,
			&to_mat,
			&tangent_mat,
			&mut result,
			&mut (),
		)
		.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}
}
