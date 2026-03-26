//! Python wrapper for the Stiefel manifold.
//!
//! This module provides a Python-friendly interface to the Stiefel manifold,
//! representing matrices with orthonormal columns.

use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use riemannopt_core::linalg::MatrixOps;
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::Stiefel;

use super::base::{array_to_point, PointType, PyManifoldBase};
use crate::array_utils::{mat_to_numpy, numpy_to_mat, Mat64, Vec64};
use crate::error::to_py_err;
use crate::types::PyPoint;

/// Python wrapper for the Stiefel manifold St(n, p).
///
/// The Stiefel manifold represents all n×p matrices with orthonormal columns:
/// St(n, p) = {X ∈ R^{n×p} : X^T X = I_p}
///
/// This is commonly used for:
/// - Orthogonality constraints in neural networks
/// - Eigenvalue problems
/// - Dimensionality reduction with orthogonal projections
#[pyclass(name = "Stiefel", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyStiefel {
	/// The underlying Rust Stiefel manifold
	pub(crate) inner: Stiefel,
	/// Number of rows
	pub(crate) n: usize,
	/// Number of columns
	pub(crate) p: usize,
}

impl PyManifoldBase for PyStiefel {
	fn manifold_name(&self) -> &'static str {
		"Stiefel"
	}

	fn ambient_dim(&self) -> usize {
		self.n * self.p
	}

	fn intrinsic_dim(&self) -> usize {
		self.n * self.p - self.p * (self.p + 1) / 2
	}

	fn point_type(&self) -> PointType {
		PointType::Matrix(self.n, self.p)
	}
}

#[pymethods]
impl PyStiefel {
	/// Create a new Stiefel manifold.
	///
	/// Args:
	///     n: Number of rows (must be >= p)
	///     p: Number of columns (must be >= 1)
	///
	/// Raises:
	///     ValueError: If n < p or p < 1
	#[new]
	pub fn new(n: usize, p: usize) -> PyResult<Self> {
		if n < p {
			return Err(crate::error::value_error(format!(
				"Stiefel manifold requires n >= p, got n={}, p={}",
				n, p
			)));
		}
		if p < 1 {
			return Err(crate::error::value_error(
				"Stiefel manifold requires p >= 1",
			));
		}

		Ok(PyStiefel {
			inner: Stiefel::new(n, p).map_err(to_py_err)?,
			n,
			p,
		})
	}

	/// Get the shape of points on this manifold.
	#[getter]
	fn shape(&self) -> (usize, usize) {
		(self.n, self.p)
	}

	/// Project a matrix onto the manifold.
	///
	/// This performs QR decomposition to get a matrix with orthonormal columns.
	///
	/// Args:
	///     point: Matrix in ambient space
	///
	/// Returns:
	///     Projected matrix on the Stiefel manifold
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

		self.inner.project_point(&point_mat, &mut result);

		mat_to_numpy(py, &result)
	}

	/// Compute the Riemannian exponential map.
	///
	/// Args:
	///     point: Base point on the Stiefel manifold
	///     tangent: Tangent vector at the base point
	///
	/// Returns:
	///     Point on the Stiefel manifold
	pub fn exp<'py>(
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
				&[self.n, self.p, self.n, self.p],
				&[
					point_mat.nrows(),
					point_mat.ncols(),
					tangent_mat.nrows(),
					tangent_mat.ncols(),
				],
			));
		}

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);
		self.inner
			.retract(&point_mat, &tangent_mat, &mut result, &mut ())
			.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Compute the Riemannian logarithm map.
	///
	/// Args:
	///     point: Base point on the Stiefel manifold
	///     other: Target point on the Stiefel manifold
	///
	/// Returns:
	///     Tangent vector at the base point
	pub fn log<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		other: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;
		let other_mat = numpy_to_mat(other)?;

		// Validate dimensions
		if (point_mat.nrows() != self.n || point_mat.ncols() != self.p)
			|| (other_mat.nrows() != self.n || other_mat.ncols() != self.p)
		{
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p, self.n, self.p],
				&[
					point_mat.nrows(),
					point_mat.ncols(),
					other_mat.nrows(),
					other_mat.ncols(),
				],
			));
		}

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);
		self.inner
			.inverse_retract(&point_mat, &other_mat, &mut result, &mut ())
			.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Retract a tangent vector to the manifold using QR decomposition.
	///
	/// Args:
	///     point: Base point on the Stiefel manifold
	///     tangent: Tangent vector at the base point
	///
	/// Returns:
	///     Point on the Stiefel manifold
	pub fn retract<'py>(
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
				&[self.n, self.p, self.n, self.p],
				&[
					point_mat.nrows(),
					point_mat.ncols(),
					tangent_mat.nrows(),
					tangent_mat.ncols(),
				],
			));
		}

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		self.inner
			.retract(&point_mat, &tangent_mat, &mut result, &mut ())
			.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Project a matrix onto the tangent space.
	///
	/// Args:
	///     point: Base point on the Stiefel manifold
	///     vector: Matrix in ambient space
	///
	/// Returns:
	///     Projected matrix in tangent space
	pub fn project_tangent<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		vector: PyReadonlyArray2<'_, f64>,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;
		let vector_mat = numpy_to_mat(vector)?;

		// Validate dimensions
		if (point_mat.nrows() != self.n || point_mat.ncols() != self.p)
			|| (vector_mat.nrows() != self.n || vector_mat.ncols() != self.p)
		{
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p, self.n, self.p],
				&[
					point_mat.nrows(),
					point_mat.ncols(),
					vector_mat.nrows(),
					vector_mat.ncols(),
				],
			));
		}

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		self.inner
			.project_tangent(&point_mat, &vector_mat, &mut result, &mut ())
			.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Compute the Riemannian inner product (canonical metric).
	///
	/// Args:
	///     point: Base point on the Stiefel manifold
	///     u: First tangent vector
	///     v: Second tangent vector
	///
	/// Returns:
	///     Inner product value
	pub fn inner(
		&self,
		_py: Python<'_>,
		point: PyReadonlyArray2<'_, f64>,
		u: PyReadonlyArray2<'_, f64>,
		v: PyReadonlyArray2<'_, f64>,
	) -> PyResult<f64> {
		let point_mat = numpy_to_mat(point)?;
		let u_mat = numpy_to_mat(u)?;
		let v_mat = numpy_to_mat(v)?;

		// Validate dimensions
		if (point_mat.nrows() != self.n || point_mat.ncols() != self.p)
			|| (u_mat.nrows() != self.n || u_mat.ncols() != self.p)
			|| (v_mat.nrows() != self.n || v_mat.ncols() != self.p)
		{
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p, self.n, self.p, self.n, self.p],
				&[
					point_mat.nrows(),
					point_mat.ncols(),
					u_mat.nrows(),
					u_mat.ncols(),
					v_mat.nrows(),
					v_mat.ncols(),
				],
			));
		}

		self.inner
			.inner_product(&point_mat, &u_mat, &v_mat, &mut ())
			.map_err(to_py_err)
	}

	/// Compute the norm of a tangent vector.
	///
	/// Args:
	///     point: Base point on the Stiefel manifold
	///     tangent: Tangent vector
	///
	/// Returns:
	///     Norm of the tangent vector
	pub fn norm(
		&self,
		_py: Python<'_>,
		point: PyReadonlyArray2<'_, f64>,
		tangent: PyReadonlyArray2<'_, f64>,
	) -> PyResult<f64> {
		let point_mat = numpy_to_mat(point)?;
		let tangent_mat = numpy_to_mat(tangent)?;

		// Validate dimensions
		if (point_mat.nrows() != self.n || point_mat.ncols() != self.p)
			|| (tangent_mat.nrows() != self.n || tangent_mat.ncols() != self.p)
		{
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p, self.n, self.p],
				&[
					point_mat.nrows(),
					point_mat.ncols(),
					tangent_mat.nrows(),
					tangent_mat.ncols(),
				],
			));
		}

		Ok(self
			.inner
			.norm(&point_mat, &tangent_mat, &mut ())
			.map_err(to_py_err)?)
	}

	/// Compute the geodesic distance between two points.
	///
	/// Args:
	///     x: First point on the Stiefel manifold
	///     y: Second point on the Stiefel manifold
	///
	/// Returns:
	///     Geodesic distance
	pub fn distance(
		&self,
		_py: Python<'_>,
		x: PyReadonlyArray2<'_, f64>,
		y: PyReadonlyArray2<'_, f64>,
	) -> PyResult<f64> {
		let x_mat = numpy_to_mat(x)?;
		let y_mat = numpy_to_mat(y)?;

		// Validate dimensions
		if (x_mat.nrows() != self.n || x_mat.ncols() != self.p)
			|| (y_mat.nrows() != self.n || y_mat.ncols() != self.p)
		{
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p, self.n, self.p],
				&[x_mat.nrows(), x_mat.ncols(), y_mat.nrows(), y_mat.ncols()],
			));
		}

		Ok(self.inner.distance(&x_mat, &y_mat).map_err(to_py_err)?)
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

	/// Generate a random point on the Stiefel manifold.
	///
	/// Returns:
	///     Random matrix with orthonormal columns
	pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		self.inner.random_point(&mut result).map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	/// Generate a random tangent vector at a point.
	///
	/// Args:
	///     point: Base point on the Stiefel manifold
	///     scale: Standard deviation of the tangent vector (default: 1.0)
	///
	/// Returns:
	///     Random tangent vector
	#[pyo3(signature = (point, scale=1.0))]
	pub fn random_tangent<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray2<'_, f64>,
		scale: f64,
	) -> PyResult<Bound<'py, PyArray2<f64>>> {
		let point_mat = numpy_to_mat(point)?;

		// Validate dimension
		if point_mat.nrows() != self.n || point_mat.ncols() != self.p {
			return Err(crate::error::dimension_mismatch(
				&[self.n, self.p],
				&[point_mat.nrows(), point_mat.ncols()],
			));
		}

		let mut result: Mat64 = MatrixOps::zeros(self.n, self.p);

		self.inner
			.random_tangent(&point_mat, &mut result)
			.map_err(to_py_err)?;

		// Scale the result if needed
		if scale != 1.0 {
			result.scale_mut(scale);
		}

		mat_to_numpy(py, &result)
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
		self.inner
			.parallel_transport(&from_mat, &to_mat, &tangent_mat, &mut result, &mut ())
			.map_err(to_py_err)?;

		mat_to_numpy(py, &result)
	}

	// Include common methods from macro
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
			PyPoint::Vector(vec) => self.contains_vector(vec, atol),
			PyPoint::Matrix(mat) => self.contains_matrix(mat, atol),
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
	fn is_tangent(
		&self,
		py: Python<'_>,
		point: PyObject,
		vector: PyObject,
		atol: f64,
	) -> PyResult<bool> {
		let point = self.parse_point(py, point)?;
		let vector = self.parse_point(py, vector)?;
		self.validate_point_shape(&point)?;
		self.validate_point_shape(&vector)?;

		match (&point, &vector) {
			(PyPoint::Vector(p), PyPoint::Vector(v)) => self.is_tangent_vector(p, v, atol),
			(PyPoint::Matrix(p), PyPoint::Matrix(v)) => self.is_tangent_matrix(p, v, atol),
			_ => Err(crate::error::type_error(
				"matching point and vector types",
				"mismatched types",
			)),
		}
	}
}

// Additional helper methods specific to PyStiefel
impl PyStiefel {
	/// Parse a Python object as a point.
	fn parse_point(&self, py: Python<'_>, obj: PyObject) -> PyResult<PyPoint> {
		array_to_point(py, obj)
	}

	/// Check if a matrix point is on the manifold.
	fn contains_matrix(&self, point: &Mat64, atol: f64) -> PyResult<bool> {
		Ok(self.inner.is_point_on_manifold(point, atol))
	}

	/// Check if a vector point is on the manifold (not applicable for Stiefel).
	fn contains_vector(&self, _vec: &Vec64, _atol: f64) -> PyResult<bool> {
		Err(crate::error::type_error("matrix point", "vector point"))
	}

	/// Check if a matrix is in the tangent space.
	fn is_tangent_matrix(&self, point: &Mat64, vector: &Mat64, atol: f64) -> PyResult<bool> {
		Ok(self.inner.is_vector_in_tangent_space(point, vector, atol))
	}

	/// Check if a vector is in the tangent space (not applicable for Stiefel).
	fn is_tangent_vector(&self, _point: &Vec64, _vector: &Vec64, _atol: f64) -> PyResult<bool> {
		Err(crate::error::type_error("matrix tangent", "vector tangent"))
	}
}
