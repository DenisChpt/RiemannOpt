//! Python bindings for Natural Gradient optimizer.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use riemannopt_core::optimizer::{Optimizer, StoppingCriterion};
use riemannopt_optim::{FisherApproximation, NaturalGradient, NaturalGradientConfig};

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

use super::base::{PyOptimizationResult, PyOptimizerBase};

/// Python wrapper for Natural Gradient optimizer.
///
/// The natural gradient method uses the Fisher information matrix to
/// precondition the gradient, leading to faster convergence in many cases.
///
/// Parameters
/// ----------
/// learning_rate : float, default=0.01
///     Learning rate (step size)
/// fisher_damping : float, default=1e-6
///     Damping factor for Fisher matrix regularization
/// fisher_subsample : int or None, default=None
///     Number of samples to use for Fisher estimation. If None, uses all samples.
/// momentum : float, default=0.0
///     Momentum coefficient (0 = no momentum)
#[pyclass(name = "NaturalGradient", module = "riemannopt.optimizers")]
#[derive(Clone)]
pub struct PyNaturalGradient {
	/// Learning rate
	pub learning_rate: f64,
	/// Fisher matrix damping
	pub fisher_damping: f64,
	/// Fisher subsampling size
	pub fisher_subsample: Option<usize>,
	/// Momentum coefficient
	pub momentum: f64,
}

impl PyOptimizerBase for PyNaturalGradient {
	fn name(&self) -> &'static str {
		"NaturalGradient"
	}

	fn validate_config(&self) -> PyResult<()> {
		if self.learning_rate <= 0.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"learning_rate must be positive",
			));
		}
		if self.fisher_damping < 0.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"fisher_damping must be non-negative",
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

#[pymethods]
impl PyNaturalGradient {
	/// Create a new Natural Gradient optimizer.
	#[new]
	#[pyo3(signature = (learning_rate=0.01, fisher_damping=1e-6, fisher_subsample=None, momentum=0.0))]
	fn new(
		learning_rate: f64,
		fisher_damping: f64,
		fisher_subsample: Option<usize>,
		momentum: f64,
	) -> PyResult<Self> {
		let opt = PyNaturalGradient {
			learning_rate,
			fisher_damping,
			fisher_subsample,
			momentum,
		};
		opt.validate_config()?;
		Ok(opt)
	}

	/// String representation of the optimizer.
	fn __repr__(&self) -> String {
		format!(
            "NaturalGradient(learning_rate={}, fisher_damping={}, fisher_subsample={:?}, momentum={})",
            self.learning_rate, self.fisher_damping, self.fisher_subsample, self.momentum
        )
	}

	/// Get optimizer configuration as a dictionary.
	#[getter]
	fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
		let dict = PyDict::new(py);
		dict.set_item("learning_rate", self.learning_rate)?;
		dict.set_item("fisher_damping", self.fisher_damping)?;
		dict.set_item("fisher_subsample", self.fisher_subsample)?;
		dict.set_item("momentum", self.momentum)?;
		Ok(dict.into())
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

// Implement generic optimizer interface
impl_optimizer_generic_default!(
	PyNaturalGradient,
	NaturalGradient<f64>,
	NaturalGradientConfig<f64>,
	|opt: &PyNaturalGradient| {
		let fisher_approximation = if let Some(_subsample) = opt.fisher_subsample {
			FisherApproximation::Empirical
		} else {
			FisherApproximation::Full
		};

		NaturalGradientConfig {
			learning_rate: opt.learning_rate,
			damping: opt.fisher_damping,
			fisher_approximation,
			fisher_update_freq: 1,
			fisher_num_samples: opt.fisher_subsample.unwrap_or(100),
		}
	}
);
