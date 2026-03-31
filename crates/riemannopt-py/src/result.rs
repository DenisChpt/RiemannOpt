use pyo3::prelude::*;
use riemannopt_core::solver::SolverResult;

use crate::convert::{col_to_numpy_1d, mat_to_numpy_2d, Mat64, Vec64};

/// Internal storage for the solution point.
pub(crate) enum PointData {
	Vector(Vec64),
	Matrix(Mat64),
}

#[pyclass(name = "SolverResult")]
pub struct PySolverResult {
	#[pyo3(get)]
	pub value: f64,
	#[pyo3(get)]
	pub gradient_norm: Option<f64>,
	#[pyo3(get)]
	pub iterations: usize,
	#[pyo3(get)]
	pub function_evaluations: usize,
	#[pyo3(get)]
	pub gradient_evaluations: usize,
	#[pyo3(get)]
	pub converged: bool,
	#[pyo3(get)]
	pub duration_secs: f64,
	#[pyo3(get)]
	pub termination_reason: String,
	point: PointData,
}

impl PySolverResult {
	pub fn from_vec_result(r: SolverResult<f64, Vec64>) -> Self {
		Self {
			value: r.value,
			gradient_norm: r.gradient_norm,
			iterations: r.iterations,
			function_evaluations: r.function_evaluations,
			gradient_evaluations: r.gradient_evaluations,
			converged: r.converged,
			duration_secs: r.duration.as_secs_f64(),
			termination_reason: format!("{:?}", r.termination_reason),
			point: PointData::Vector(r.point),
		}
	}

	pub fn from_mat_result(r: SolverResult<f64, Mat64>) -> Self {
		Self {
			value: r.value,
			gradient_norm: r.gradient_norm,
			iterations: r.iterations,
			function_evaluations: r.function_evaluations,
			gradient_evaluations: r.gradient_evaluations,
			converged: r.converged,
			duration_secs: r.duration.as_secs_f64(),
			termination_reason: format!("{:?}", r.termination_reason),
			point: PointData::Matrix(r.point),
		}
	}
}

#[pymethods]
impl PySolverResult {
	#[getter]
	fn point(&self, py: Python<'_>) -> Py<PyAny> {
		match &self.point {
			PointData::Vector(v) => col_to_numpy_1d(py, v).into_any().unbind(),
			PointData::Matrix(m) => mat_to_numpy_2d(py, m).into_any().unbind(),
		}
	}

	fn __repr__(&self) -> String {
		format!(
			"SolverResult(converged={}, value={:.6e}, iterations={}, duration={:.3}s, reason={})",
			self.converged, self.value, self.iterations, self.duration_secs, self.termination_reason,
		)
	}
}
