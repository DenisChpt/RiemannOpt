//! Base types and utilities for Python optimizer wrappers.
//!
//! This module provides common functionality shared by all optimizer implementations,
//! including result types, callback adapters, and base traits.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use riemannopt_core::optimization::optimizer::{
	OptimizationResult as RustOptimizationResult, TerminationReason,
};

/// Python-friendly optimization result.
///
/// This provides a Pythonic interface to optimization results,
/// converting Rust types to appropriate Python representations.
///
/// Attributes
/// ----------
/// point : numpy.ndarray
///     Final point found by the optimizer
/// value : float
///     Final objective function value
/// gradient_norm : float or None
///     Norm of the gradient at the final point
/// converged : bool
///     Whether the optimization converged successfully
/// iterations : int
///     Number of iterations performed
/// function_evals : int
///     Number of function evaluations
/// gradient_evals : int
///     Number of gradient evaluations
/// time_seconds : float
///     Total optimization time in seconds
/// termination_reason : str
///     Reason for termination (e.g., 'Converged', 'MaxIterations')
/// history : dict or None
///     Optional optimization history (if callbacks were used)
#[pyclass(name = "OptimizationResult", module = "riemannopt.optimizers")]
pub struct PyOptimizationResult {
	/// Final point as numpy array
	pub point: Py<PyAny>,
	/// Final objective value
	pub value: f64,
	/// Final gradient norm
	pub gradient_norm: Option<f64>,
	/// Whether optimization converged
	pub converged: bool,
	/// Number of iterations performed
	pub iterations: usize,
	/// Number of function evaluations
	pub function_evals: usize,
	/// Number of gradient evaluations
	pub gradient_evals: usize,
	/// Total optimization time in seconds
	pub time_seconds: f64,
	/// Termination reason as string
	pub termination_reason: String,
	/// Optional optimization history
	pub history: Option<Py<PyAny>>,
}

#[pymethods]
impl PyOptimizationResult {
	/// String representation of the result.
	fn __repr__(&self) -> String {
		format!(
            "OptimizationResult(value={:.6e}, gradient_norm={}, converged={}, iterations={}, reason='{}')",
            self.value,
            self.gradient_norm.map(|g| format!("{:.6e}", g)).unwrap_or("None".to_string()),
            self.converged,
            self.iterations,
            self.termination_reason
        )
	}

	/// Get all result data as a dictionary.
	#[getter]
	fn as_dict(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
		let dict = PyDict::new(py);
		dict.set_item("point", &self.point)?;
		dict.set_item("value", self.value)?;
		dict.set_item("gradient_norm", self.gradient_norm)?;
		dict.set_item("converged", self.converged)?;
		dict.set_item("iterations", self.iterations)?;
		dict.set_item("function_evals", self.function_evals)?;
		dict.set_item("gradient_evals", self.gradient_evals)?;
		dict.set_item("time_seconds", self.time_seconds)?;
		dict.set_item("termination_reason", &self.termination_reason)?;
		if let Some(ref history) = self.history {
			dict.set_item("history", history)?;
		}
		Ok(dict.into())
	}

	/// Get a summary of the optimization result.
	fn summary(&self) -> String {
		let status = if self.converged {
			"✓ Converged"
		} else {
			"✗ Not converged"
		};
		format!(
			"Optimization Result:\n\
            {} after {} iterations\n\
            Final value: {:.6e}\n\
            Gradient norm: {}\n\
            Time: {:.3}s\n\
            Termination: {}",
			status,
			self.iterations,
			self.value,
			self.gradient_norm
				.map(|g| format!("{:.6e}", g))
				.unwrap_or("N/A".to_string()),
			self.time_seconds,
			self.termination_reason
		)
	}

	/// Check if the optimization was successful.
	#[getter]
	fn success(&self) -> bool {
		self.converged
	}

	/// Get the final cost value (alias for value).
	#[getter]
	fn cost(&self) -> f64 {
		self.value
	}

	/// Get the final point (alias for point).
	#[getter]
	fn x(&self) -> Py<PyAny> {
		Python::attach(|py| self.point.clone_ref(py))
	}

	/// Get the final value.
	#[getter]
	fn value(&self) -> f64 {
		self.value
	}

	/// Get the final gradient norm.
	#[getter]
	fn gradient_norm(&self) -> Option<f64> {
		self.gradient_norm
	}

	/// Get whether optimization converged.
	#[getter]
	fn converged(&self) -> bool {
		self.converged
	}

	/// Get the number of iterations.
	#[getter]
	fn iterations(&self) -> usize {
		self.iterations
	}

	/// Get the number of function evaluations.
	#[getter]
	fn function_evals(&self) -> usize {
		self.function_evals
	}

	/// Get the number of gradient evaluations.
	#[getter]
	fn gradient_evals(&self) -> usize {
		self.gradient_evals
	}

	/// Get the optimization time in seconds.
	#[getter]
	fn time_seconds(&self) -> f64 {
		self.time_seconds
	}

	/// Get the termination reason.
	#[getter]
	fn termination_reason(&self) -> &str {
		&self.termination_reason
	}

	/// Get the final point.
	#[getter]
	fn point(&self) -> Py<PyAny> {
		Python::attach(|py| self.point.clone_ref(py))
	}
}

impl PyOptimizationResult {
	/// Convert a Rust optimization result to Python format.
	pub fn from_rust_result<P>(
		_py: Python<'_>,
		result: RustOptimizationResult<f64, P>,
		point_converter: impl FnOnce(&P) -> PyResult<Py<PyAny>>,
	) -> PyResult<Self> {
		let point = point_converter(&result.point)?;

		let termination_reason = match result.termination_reason {
			TerminationReason::Converged => "Converged",
			TerminationReason::MaxIterations => "MaxIterations",
			TerminationReason::MaxTime => "MaxTime",
			TerminationReason::TargetReached => "TargetReached",
			TerminationReason::LineSearchFailed => "LineSearchFailed",
			TerminationReason::MaxFunctionEvaluations => "MaxFunctionEvaluations",
			TerminationReason::NumericalError => "NumericalError",
			TerminationReason::UserTerminated => "UserTerminated",
			_ => "Unknown",
		}
		.to_string();

		Ok(PyOptimizationResult {
			point,
			value: result.value,
			gradient_norm: result.gradient_norm,
			converged: result.converged,
			iterations: result.iterations,
			function_evals: result.function_evaluations,
			gradient_evals: result.gradient_evaluations,
			time_seconds: result.duration.as_secs_f64(),
			termination_reason,
			history: None, // Can be populated by callbacks
		})
	}
}

/// Base trait for Python optimizer implementations.
///
/// This trait provides common functionality that all optimizers should implement.
#[allow(dead_code)]
pub trait PyOptimizerBase {
	/// Get the name of the optimizer.
	fn name(&self) -> &'static str;

	/// Validate that the configuration is valid.
	fn validate_config(&self) -> PyResult<()> {
		Ok(())
	}
}
