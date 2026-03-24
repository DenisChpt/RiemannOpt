//! Python wrapper for the Conjugate Gradient optimizer.
//!
//! Conjugate Gradient methods are among the most efficient for large-scale
//! optimization, requiring only first-order information.

use crate::py_optimizers::generic::PyOptimizerGeneric;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use riemannopt_core::{
	line_search::LineSearchParams,
	optimizer::{Optimizer, StoppingCriterion},
};
use riemannopt_optim::{CGConfig, CGLineSearchType, ConjugateGradient, ConjugateGradientMethod};

use super::base::{PyOptimizationResult, PyOptimizerBase};
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

/// Conjugate Gradient optimizer for Riemannian manifolds.
///
/// The Conjugate Gradient method generates search directions that are
/// conjugate with respect to the Hessian, leading to faster convergence
/// than steepest descent.
///
/// Parameters
/// ----------
/// method : str, default="FletcherReeves"
///     The CG update formula. Options: "FletcherReeves", "PolakRibiere",
///     "HestenesStiefel", "DaiYuan", "HagerZhang", "LiuStorey".
/// reset_every : int, default=None
///     Reset to steepest descent every N iterations. If None, uses n
///     (dimension) as the reset frequency.
/// max_line_search_iterations : int, default=20
///     Maximum iterations for line search.
/// c1 : float, default=1e-4
///     Wolfe condition parameter for sufficient decrease.
/// c2 : float, default=0.1
///     Wolfe condition parameter for curvature.
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> sphere = ro.manifolds.Sphere(100)
/// >>> optimizer = ro.optimizers.ConjugateGradient(method="PolakRibiere")
/// >>>
/// >>> # Define quadratic cost
/// >>> Q = np.random.randn(100, 100)
/// >>> Q = Q.T @ Q  # Positive definite
/// >>>
/// >>> def cost(x):
/// ...     return x.T @ Q @ x
/// >>>
/// >>> x0 = sphere.random_point()
/// >>> result = optimizer.optimize(
/// ...     cost_function=cost,
/// ...     manifold=sphere,
/// ...     initial_point=x0,
/// ...     max_iterations=50
/// ... )
#[pyclass(name = "ConjugateGradient", module = "riemannopt.optimizers")]
#[derive(Clone)]
pub struct PyConjugateGradient {
	pub method: String,
	pub reset_every: Option<usize>,
	pub max_line_search_iterations: usize,
	pub c1: f64,
	pub c2: f64,
	pub line_search: String,
}

#[pymethods]
impl PyConjugateGradient {
	#[new]
	#[pyo3(signature = (method="FletcherReeves", reset_every=None, max_line_search_iterations=20, c1=1e-4, c2=0.1, line_search="adaptive"))]
	fn new(
		method: &str,
		reset_every: Option<usize>,
		max_line_search_iterations: usize,
		c1: f64,
		c2: f64,
		line_search: &str,
	) -> PyResult<Self> {
		// Validate method
		let valid_methods = [
			"FletcherReeves",
			"PolakRibiere",
			"HestenesStiefel",
			"DaiYuan",
			"HagerZhang",
			"LiuStorey",
		];
		if !valid_methods.contains(&method) {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"Invalid method '{}'. Choose from: {:?}",
				method, valid_methods
			)));
		}

		// Validate line search type
		let valid_ls = ["adaptive", "strong_wolfe"];
		if !valid_ls.contains(&line_search) {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"Invalid line_search '{}'. Choose from: {:?}",
				line_search, valid_ls
			)));
		}

		// Validate parameters
		if max_line_search_iterations == 0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"max_line_search_iterations must be positive",
			));
		}
		if c1 <= 0.0 || c1 >= 1.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"c1 must be in (0, 1)",
			));
		}
		// c2 validation only matters for Strong Wolfe
		if line_search == "strong_wolfe" && (c2 <= c1 || c2 >= 1.0) {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"c2 must be in (c1, 1) for strong_wolfe line search",
			));
		}

		Ok(PyConjugateGradient {
			method: method.to_string(),
			reset_every,
			max_line_search_iterations,
			c1,
			c2,
			line_search: line_search.to_string(),
		})
	}

	fn __repr__(&self) -> String {
		format!(
            "ConjugateGradient(method='{}', reset_every={:?}, max_line_search_iterations={}, c1={}, c2={}, line_search='{}')",
            self.method, self.reset_every, self.max_line_search_iterations, self.c1, self.c2, self.line_search
        )
	}

	/// Get optimizer configuration as a dictionary.
	#[getter]
	fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
		let dict = PyDict::new(py);
		dict.set_item("method", &self.method)?;
		dict.set_item("reset_every", self.reset_every)?;
		dict.set_item(
			"max_line_search_iterations",
			self.max_line_search_iterations,
		)?;
		dict.set_item("c1", self.c1)?;
		dict.set_item("c2", self.c2)?;
		dict.set_item("line_search", &self.line_search)?;
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

// Implement the base trait
impl PyOptimizerBase for PyConjugateGradient {
	fn name(&self) -> &'static str {
		"ConjugateGradient"
	}

	fn validate_config(&self) -> PyResult<()> {
		let valid_methods = [
			"FletcherReeves",
			"PolakRibiere",
			"HestenesStiefel",
			"DaiYuan",
			"HagerZhang",
			"LiuStorey",
		];
		if !valid_methods.contains(&self.method.as_str()) {
			return Err(pyo3::exceptions::PyValueError::new_err(format!(
				"Invalid method '{}'. Choose from: {:?}",
				self.method, valid_methods
			)));
		}
		if self.max_line_search_iterations == 0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"max_line_search_iterations must be positive",
			));
		}
		if self.c1 <= 0.0 || self.c1 >= 1.0 {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"c1 must be in (0, 1)",
			));
		}
		if self.line_search == "strong_wolfe" && (self.c2 <= self.c1 || self.c2 >= 1.0) {
			return Err(pyo3::exceptions::PyValueError::new_err(
				"c2 must be in (c1, 1) for strong_wolfe line search",
			));
		}
		Ok(())
	}
}

// Implement generic optimizer interface
impl_optimizer_generic_default!(
	PyConjugateGradient,
	ConjugateGradient<f64>,
	CGConfig<f64>,
	|opt: &PyConjugateGradient| {
		let cg_method = match opt.method.as_str() {
			"FletcherReeves" => ConjugateGradientMethod::FletcherReeves,
			"PolakRibiere" => ConjugateGradientMethod::PolakRibiere,
			"HestenesStiefel" => ConjugateGradientMethod::HestenesStiefel,
			"DaiYuan" => ConjugateGradientMethod::DaiYuan,
			"HagerZhang" => ConjugateGradientMethod::HagerZhang,
			"LiuStorey" => ConjugateGradientMethod::LiuStorey,
			_ => ConjugateGradientMethod::FletcherReeves,
		};

		let line_search_params = LineSearchParams {
			initial_step_size: 1.0,
			max_step_size: 100.0,
			min_step_size: 1e-10,
			max_iterations: opt.max_line_search_iterations,
			c1: opt.c1,
			c2: opt.c2,
			rho: 0.5,
		};

		let ls_type = match opt.line_search.as_str() {
			"strong_wolfe" => CGLineSearchType::StrongWolfe,
			_ => CGLineSearchType::Adaptive,
		};

		CGConfig {
			method: cg_method,
			restart_period: opt.reset_every.unwrap_or(100), // Default restart period
			use_pr_plus: false,
			min_beta: None,
			max_beta: None,
			line_search_params,
			line_search_type: ls_type,
		}
	}
);
