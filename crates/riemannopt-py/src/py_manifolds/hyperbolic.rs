//! Python wrapper for the Hyperbolic manifold.
//!
//! The hyperbolic manifold H^n is the n-dimensional hyperbolic space with
//! constant negative curvature. It is represented using the hyperboloid model
//! embedded in Minkowski space.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use riemannopt_core::linalg::{VectorOps, VectorView};
use riemannopt_core::manifold::Manifold;
use riemannopt_manifolds::hyperbolic::Hyperbolic;

use super::base::{PointType, PyManifoldBase};
use crate::{
	array_utils::{numpy_to_vec, vec_to_numpy, Mat64, Vec64},
	error::{dimension_mismatch, to_py_err},
	types::PyPoint,
};

/// The hyperbolic manifold H^n.
///
/// The hyperbolic manifold is the n-dimensional space with constant negative
/// curvature. This implementation uses the hyperboloid model, where points
/// are represented as (n+1)-dimensional vectors in Minkowski space satisfying
/// the constraint <x, x>_L = -1, where <., .>_L is the Minkowski inner product.
///
/// The Minkowski inner product is defined as:
/// <x, y>_L = -x[0]*y[0] + sum(x[i]*y[i] for i in 1..n+1)
///
/// Parameters
/// ----------
/// n : int
///     Dimension of the hyperbolic space
/// curvature : float, default=-1.0
///     Constant negative curvature (must be < 0)
#[pyclass(name = "Hyperbolic", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyHyperbolic {
	pub(crate) inner: Hyperbolic<f64>,
	n: usize,
	curvature: f64,
}

impl PyManifoldBase for PyHyperbolic {
	fn manifold_name(&self) -> &'static str {
		"Hyperbolic"
	}

	fn ambient_dim(&self) -> usize {
		// Poincare ball model: points live in R^n (not R^{n+1})
		self.n
	}

	fn intrinsic_dim(&self) -> usize {
		self.n
	}

	fn point_type(&self) -> PointType {
		PointType::Vector(self.n)
	}
}

#[pymethods]
impl PyHyperbolic {
	/// Create a new hyperbolic manifold.
	#[new]
	#[pyo3(signature = (n, curvature=-1.0))]
	pub fn new(n: usize, curvature: f64) -> PyResult<Self> {
		if n == 0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"n must be positive",
			));
		}
		if curvature >= 0.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"curvature must be negative for hyperbolic space",
			));
		}

		let inner = if (curvature - (-1.0)).abs() < 1e-10 {
			Hyperbolic::<f64>::new(n)
				.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
		} else {
			Hyperbolic::<f64>::with_parameters(n, 1e-6, curvature)
				.map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
		};
		Ok(PyHyperbolic {
			inner,
			n,
			curvature,
		})
	}

	fn __repr__(&self) -> String {
		format!(
			"{}(ambient_dim={}, intrinsic_dim={})",
			self.manifold_name(),
			self.ambient_dim(),
			self.intrinsic_dim()
		)
	}

	#[getter]
	fn dim(&self) -> usize {
		self.intrinsic_dim()
	}

	#[getter]
	fn ambient_dim(&self) -> usize {
		PyManifoldBase::ambient_dim(self)
	}

	#[pyo3(signature = (point, atol=1e-10))]
	fn contains(&self, py: Python<'_>, point: Py<PyAny>, atol: f64) -> PyResult<bool> {
		let point = self.parse_point(py, point)?;
		self.validate_point_shape(&point)?;

		match &point {
			PyPoint::Vector(vec) => self.contains_vector(vec, atol),
			PyPoint::Matrix(mat) => self.contains_matrix(mat, atol),
		}
	}

	#[pyo3(signature = (point, vector, atol=1e-10))]
	fn is_tangent(
		&self,
		py: Python<'_>,
		point: Py<PyAny>,
		vector: Py<PyAny>,
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

	#[getter]
	fn n(&self) -> usize {
		self.n
	}

	#[getter]
	fn curvature(&self) -> f64 {
		self.curvature
	}

	pub fn project<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray1<'_, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let point_vec = numpy_to_vec(point)?;

		if point_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[point_vec.len()]));
		}

		let mut result: Vec64 = VectorOps::zeros(self.n);

		self.inner.project_point(&point_vec, &mut result);
		vec_to_numpy(py, &result)
	}

	pub fn exp<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray1<'_, f64>,
		tangent: PyReadonlyArray1<'_, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let point_vec = numpy_to_vec(point)?;
		let tangent_vec = numpy_to_vec(tangent)?;

		if point_vec.len() != self.n || tangent_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[point_vec.len()]));
		}

		let mut result: Vec64 = VectorOps::zeros(self.n);

		self.inner
			.retract(&point_vec, &tangent_vec, &mut result, &mut ())
			.map_err(to_py_err)?;

		vec_to_numpy(py, &result)
	}

	pub fn log<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray1<'_, f64>,
		other: PyReadonlyArray1<'_, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let point_vec = numpy_to_vec(point)?;
		let other_vec = numpy_to_vec(other)?;

		if point_vec.len() != self.n || other_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[point_vec.len()]));
		}

		let mut result: Vec64 = VectorOps::zeros(self.n);

		self.inner
			.inverse_retract(&point_vec, &other_vec, &mut result, &mut ())
			.map_err(to_py_err)?;

		vec_to_numpy(py, &result)
	}

	pub fn retract<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray1<'_, f64>,
		tangent: PyReadonlyArray1<'_, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let point_vec = numpy_to_vec(point)?;
		let tangent_vec = numpy_to_vec(tangent)?;

		if point_vec.len() != self.n || tangent_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[point_vec.len()]));
		}

		let mut result: Vec64 = VectorOps::zeros(self.n);

		self.inner
			.retract(&point_vec, &tangent_vec, &mut result, &mut ())
			.map_err(to_py_err)?;

		vec_to_numpy(py, &result)
	}

	pub fn project_tangent<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray1<'_, f64>,
		vector: PyReadonlyArray1<'_, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let point_vec = numpy_to_vec(point)?;
		let vector_vec = numpy_to_vec(vector)?;

		if point_vec.len() != self.n || vector_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[point_vec.len()]));
		}

		let mut result: Vec64 = VectorOps::zeros(self.n);

		self.inner
			.project_tangent(&point_vec, &vector_vec, &mut result, &mut ())
			.map_err(to_py_err)?;

		vec_to_numpy(py, &result)
	}

	/// Minkowski inner product.
	pub fn minkowski_inner(
		&self,
		_py: Python<'_>,
		x: PyReadonlyArray1<'_, f64>,
		y: PyReadonlyArray1<'_, f64>,
	) -> PyResult<f64> {
		let x_vec = numpy_to_vec(x)?;
		let y_vec = numpy_to_vec(y)?;

		if x_vec.len() != self.n || y_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[x_vec.len()]));
		}

		// Compute Minkowski inner product
		let mut result = -x_vec.get(0) * y_vec.get(0);
		for i in 1..self.n {
			result += x_vec.get(i) * y_vec.get(i);
		}

		Ok(result / (-self.curvature))
	}

	pub fn inner(
		&self,
		_py: Python<'_>,
		point: PyReadonlyArray1<'_, f64>,
		u: PyReadonlyArray1<'_, f64>,
		v: PyReadonlyArray1<'_, f64>,
	) -> PyResult<f64> {
		let point_vec = numpy_to_vec(point)?;
		let u_vec = numpy_to_vec(u)?;
		let v_vec = numpy_to_vec(v)?;

		if point_vec.len() != self.n || u_vec.len() != self.n || v_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[point_vec.len()]));
		}

		self.inner
			.inner_product(&point_vec, &u_vec, &v_vec, &mut ())
			.map_err(to_py_err)
	}

	pub fn norm(
		&self,
		_py: Python<'_>,
		point: PyReadonlyArray1<'_, f64>,
		tangent: PyReadonlyArray1<'_, f64>,
	) -> PyResult<f64> {
		let point_vec = numpy_to_vec(point)?;
		let tangent_vec = numpy_to_vec(tangent)?;

		if point_vec.len() != self.n || tangent_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[point_vec.len()]));
		}

		self.inner
			.norm(&point_vec, &tangent_vec, &mut ())
			.map_err(to_py_err)
	}

	pub fn distance(
		&self,
		_py: Python<'_>,
		x: PyReadonlyArray1<'_, f64>,
		y: PyReadonlyArray1<'_, f64>,
	) -> PyResult<f64> {
		let x_vec = numpy_to_vec(x)?;
		let y_vec = numpy_to_vec(y)?;

		if x_vec.len() != self.n || y_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[x_vec.len()]));
		}

		self.inner.distance(&x_vec, &y_vec).map_err(to_py_err)
	}

	pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let mut result: Vec64 = VectorOps::zeros(self.n);

		self.inner.random_point(&mut result).map_err(to_py_err)?;

		vec_to_numpy(py, &result)
	}

	#[pyo3(signature = (point, scale=1.0))]
	pub fn random_tangent<'py>(
		&self,
		py: Python<'py>,
		point: PyReadonlyArray1<'_, f64>,
		scale: f64,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let point_vec = numpy_to_vec(point)?;

		if point_vec.len() != self.n {
			return Err(dimension_mismatch(&[self.n], &[point_vec.len()]));
		}

		let mut result: Vec64 = VectorOps::zeros(self.n);

		self.inner
			.random_tangent(&point_vec, &mut result)
			.map_err(to_py_err)?;

		if scale != 1.0 {
			result.scale_mut(scale);
		}

		vec_to_numpy(py, &result)
	}

	pub fn parallel_transport<'py>(
		&self,
		py: Python<'py>,
		from_point: PyReadonlyArray1<'_, f64>,
		to_point: PyReadonlyArray1<'_, f64>,
		tangent: PyReadonlyArray1<'_, f64>,
	) -> PyResult<Bound<'py, PyArray1<f64>>> {
		let from_vec = numpy_to_vec(from_point)?;
		let to_vec = numpy_to_vec(to_point)?;
		let tangent_vec = numpy_to_vec(tangent)?;

		if from_vec.len() != self.n
			|| to_vec.len() != self.n
			|| tangent_vec.len() != self.n
		{
			return Err(dimension_mismatch(&[self.n], &[from_vec.len()]));
		}

		let result = self
			.inner
			.parallel_transport(&from_vec, &to_vec, &tangent_vec, &mut ())
			.map_err(to_py_err)?;

		vec_to_numpy(py, &result)
	}
}

impl PyHyperbolic {
	fn parse_point(&self, py: Python<'_>, obj: Py<PyAny>) -> PyResult<PyPoint> {
		super::base::array_to_point(py, obj)
	}

	fn contains_vector(&self, vec: &Vec64, atol: f64) -> PyResult<bool> {
		Ok(self.inner.is_point_on_manifold(vec, atol))
	}

	fn is_tangent_vector(&self, point: &Vec64, vector: &Vec64, atol: f64) -> PyResult<bool> {
		Ok(self.inner.is_vector_in_tangent_space(point, vector, atol))
	}

	fn contains_matrix(&self, _mat: &Mat64, _atol: f64) -> PyResult<bool> {
		Err(crate::error::type_error("vector point", "matrix point"))
	}

	fn is_tangent_matrix(&self, _point: &Mat64, _vector: &Mat64, _atol: f64) -> PyResult<bool> {
		Err(crate::error::type_error("vector tangent", "matrix tangent"))
	}
}
