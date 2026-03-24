//! Simplified Python wrapper for SGD optimizer.
//!
//! This is a minimal implementation to get started.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use riemannopt_core::{
	optimizer::{Optimizer, StoppingCriterion},
	step_size::StepSizeSchedule,
};
use riemannopt_optim::{MomentumMethod, SGDConfig, SGD};

use super::base::{PyOptimizationResult, PyOptimizerBase};
use crate::py_optimizers::generic::PyOptimizerGeneric;
use crate::{
	array_utils::{mat_to_numpy, numpy_to_mat, numpy_to_vec, vec_to_numpy},
	error::to_py_err,
	impl_optimizer_generic_default,
	py_cost::PyCostFunction,
	py_manifolds::{
		grassmann::PyGrassmann,
		hyperbolic::PyHyperbolic,
		oblique::PyOblique,
		// fixed_rank::PyFixedRank,  // TODO: Fix FixedRankPoint representation mismatch
		psd_cone::PyPSDCone,
		spd::PySPD,
		sphere::PySphere,
		stiefel::PyStiefel,
	},
};

/// Simple SGD optimizer for testing.
#[pyclass(name = "SGD", module = "riemannopt.optimizers")]
#[derive(Clone)]
pub struct PySGD {
	pub learning_rate: f64,
	pub momentum: f64,
}

#[pymethods]
impl PySGD {
	#[new]
	#[pyo3(signature = (learning_rate=0.01, momentum=0.0))]
	pub fn new(learning_rate: f64, momentum: f64) -> PyResult<Self> {
		if learning_rate <= 0.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"learning_rate must be positive",
			));
		}
		if momentum < 0.0 || momentum >= 1.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"momentum must be in [0, 1)",
			));
		}
		Ok(PySGD {
			learning_rate,
			momentum,
		})
	}

	fn __repr__(&self) -> String {
		format!(
			"SGD(learning_rate={}, momentum={})",
			self.learning_rate, self.momentum
		)
	}

	/// Unified optimize method that accepts any supported manifold.
	///
	/// Parameters
	/// ----------
	/// cost_function : CostFunction
	///     The cost function to minimize
	/// manifold : Manifold
	///     The Riemannian manifold to optimize on (Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, or PSDCone)
	/// initial_point : array_like
	///     Starting point for optimization (shape must match manifold requirements)
	/// max_iterations : int
	///     Maximum number of iterations
	/// gradient_tolerance : float, optional
	///     Gradient norm tolerance for convergence
	/// callback : callable, optional
	///     Function called after each iteration
	/// target_value : float, optional
	///     Target cost value to stop optimization early
	/// max_time : float, optional
	///     Maximum optimization time in seconds
	///
	/// Returns
	/// -------
	/// OptimizationResult
	///     Object containing the optimized point, final cost, and optimization statistics
	#[pyo3(signature = (cost_function, manifold, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None, callback=None, target_value=None, max_time=None))]
	pub fn optimize(
		&mut self,
		py: Python<'_>,
		cost_function: PyObject,
		manifold: PyObject,
		initial_point: PyObject,
		max_iterations: usize,
		gradient_tolerance: Option<f64>,
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
		callback: Option<PyObject>,
		target_value: Option<f64>,
		max_time: Option<f64>,
	) -> PyResult<PyObject> {
		// Try native cost functions first (pure Rust, no GIL overhead)
		if let Some(result) = self.try_native_optimize(
			py,
			&cost_function,
			&manifold,
			&initial_point,
			max_iterations,
			gradient_tolerance,
		)? {
			return Ok(result);
		}

		// Fall back to Python callback cost function
		use super::generic::optimize_dispatcher;
		let py_cf = cost_function.extract::<PyRef<'_, PyCostFunction>>(py)?;
		optimize_dispatcher(
			self,
			py,
			py_cf,
			manifold,
			initial_point,
			max_iterations,
			gradient_tolerance,
			function_tolerance,
			point_tolerance,
			callback,
			target_value,
			max_time,
		)
	}
}

// Implement the base trait
impl PyOptimizerBase for PySGD {
	fn name(&self) -> &'static str {
		"SGD"
	}

	fn validate_config(&self) -> PyResult<()> {
		if self.learning_rate <= 0.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"learning_rate must be positive",
			));
		}
		if self.momentum < 0.0 || self.momentum >= 1.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"momentum must be in [0, 1)",
			));
		}
		Ok(())
	}
}

// Implement generic optimizer interface
impl_optimizer_generic_default!(PySGD, SGD<f64>, SGDConfig<f64>, |opt: &PySGD| {
	let momentum_method = if opt.momentum > 0.0 {
		MomentumMethod::Classical {
			coefficient: opt.momentum,
		}
	} else {
		MomentumMethod::None
	};

	SGDConfig {
		step_size: StepSizeSchedule::Constant(opt.learning_rate),
		momentum: momentum_method,
		gradient_clip: None,
		line_search: None,
	}
});
