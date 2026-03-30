use pyo3::prelude::*;
use riemannopt_core::solver::StoppingCriterion;
use std::time::Duration;

#[pyclass(name = "StoppingCriterion", from_py_object)]
#[derive(Clone)]
pub struct PyStoppingCriterion {
	pub(crate) inner: StoppingCriterion<f64>,
}

#[pymethods]
impl PyStoppingCriterion {
	#[new]
	#[pyo3(signature = (
		max_iterations = 1000,
		gradient_tolerance = 1e-6,
		function_tolerance = None,
		max_time_secs = None,
		target_value = None,
	))]
	fn new(
		max_iterations: usize,
		gradient_tolerance: f64,
		function_tolerance: Option<f64>,
		max_time_secs: Option<f64>,
		target_value: Option<f64>,
	) -> Self {
		let mut sc = StoppingCriterion::new()
			.with_max_iterations(max_iterations)
			.with_gradient_tolerance(gradient_tolerance);
		if let Some(ft) = function_tolerance {
			sc = sc.with_function_tolerance(ft);
		}
		if let Some(secs) = max_time_secs {
			sc = sc.with_max_time(Duration::from_secs_f64(secs));
		}
		if let Some(tv) = target_value {
			sc = sc.with_target_value(tv);
		}
		Self { inner: sc }
	}

	fn __repr__(&self) -> String {
		format!(
			"StoppingCriterion(max_iterations={}, gradient_tolerance={:e})",
			self.inner
				.max_iterations
				.map_or("None".to_string(), |v| v.to_string()),
			self.inner.gradient_tolerance.unwrap_or(1e-6),
		)
	}
}
