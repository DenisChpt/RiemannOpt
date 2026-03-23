//! Python wrappers for optimization algorithms.
//!
//! This module provides Python-friendly interfaces to the Rust optimization
//! algorithms, managing the interaction between Python cost functions and
//! Rust optimizers.
//!
//! # Design Philosophy
//!
//! The Python optimizer wrappers are designed to:
//! - Hide Rust complexity (workspaces, lifetimes, etc.) from Python users
//! - Provide a Pythonic API with keyword arguments and sensible defaults
//! - Minimize GIL overhead during optimization
//! - Support both synchronous and asynchronous optimization
//! - Provide detailed progress tracking via callbacks
//!
//! # Architecture
//!
//! Each optimizer wrapper follows a consistent pattern:
//! 1. Configuration via `__init__` with Python-friendly parameters
//! 2. Type dispatch based on manifold point types (vector vs matrix)
//! 3. Efficient conversion between numpy arrays and nalgebra structures
//! 4. Progress tracking via optional callbacks
//! 5. Rich result objects with detailed optimization metadata

use pyo3::prelude::*;
use pyo3::types::PyDict;

mod adam;
mod base;
mod conjugate_gradient;
pub mod generic;
mod lbfgs;
mod natural_gradient;
mod newton;
mod sgd_simple;
mod trust_region;

pub use adam::PyAdam;
pub use base::PyOptimizationResult;
pub use conjugate_gradient::PyConjugateGradient;
pub use lbfgs::PyLBFGS;
pub use natural_gradient::PyNaturalGradient;
pub use newton::PyNewton;
pub use sgd_simple::PySGD;
pub use trust_region::PyTrustRegion;

/// Create an optimizer instance with default parameters.
///
/// This function creates an optimizer instance with sensible default parameters.
/// It's designed to be used both from Rust (internal) and Python (via PyO3).
///
/// Parameters
/// ----------
/// optimizer_name : str
///     Name of the optimizer (case-insensitive).
///     Options: "SGD", "Adam", "LBFGS", "ConjugateGradient", "TrustRegion", "Newton", "NaturalGradient"
/// kwargs : dict, optional
///     Optional parameters to override defaults
///
/// Returns
/// -------
/// optimizer : PyObject
///     The initialized optimizer instance
#[pyfunction]
#[pyo3(signature = (optimizer_name, **kwargs))]
pub fn create_optimizer(
	py: Python<'_>,
	optimizer_name: &str,
	kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<PyObject> {
	// Normalize optimizer name (case-insensitive)
	let normalized = match optimizer_name.to_lowercase().as_str() {
        "sgd" => "SGD",
        "adam" => "Adam",
        "lbfgs" | "l-bfgs" => "LBFGS",
        "conjugategradient" | "cg" => "ConjugateGradient",
        "trustregion" | "trust-region" | "tr" => "TrustRegion",
        "newton" | "riemanniannewton" => "Newton",
        "naturalgradient" | "natural-gradient" | "ng" => "NaturalGradient",
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown optimizer: {}. Available: SGD, Adam, LBFGS, ConjugateGradient, TrustRegion, Newton, NaturalGradient", optimizer_name)
        )),
    };

	// Create optimizer with default parameters, then update with kwargs if provided
	// Since the `new` methods are Python constructors, we need to create them directly with struct literals
	match normalized {
		"SGD" => {
			let mut opt = PySGD {
				learning_rate: 0.01,
				momentum: 0.0,
			};
			if let Some(kw) = kwargs {
				if let Ok(Some(lr)) = kw.get_item("learning_rate") {
					opt.learning_rate = lr.extract()?;
				}
				if let Ok(Some(m)) = kw.get_item("momentum") {
					opt.momentum = m.extract()?;
				}
			}
			Ok(opt.into_py(py))
		}
		"Adam" => {
			let mut opt = PyAdam {
				learning_rate: 0.001,
				beta1: 0.9,
				beta2: 0.999,
				epsilon: 1e-8,
				amsgrad: false,
			};
			if let Some(kw) = kwargs {
				if let Ok(Some(lr)) = kw.get_item("learning_rate") {
					opt.learning_rate = lr.extract()?;
				}
				if let Ok(Some(b1)) = kw.get_item("beta1") {
					opt.beta1 = b1.extract()?;
				}
				if let Ok(Some(b2)) = kw.get_item("beta2") {
					opt.beta2 = b2.extract()?;
				}
				if let Ok(Some(eps)) = kw.get_item("epsilon") {
					opt.epsilon = eps.extract()?;
				}
				if let Ok(Some(ams)) = kw.get_item("amsgrad") {
					opt.amsgrad = ams.extract()?;
				}
			}
			// Validate parameters
			if opt.learning_rate <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"learning_rate must be positive",
				));
			}
			if opt.beta1 < 0.0 || opt.beta1 >= 1.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"beta1 must be in [0, 1)",
				));
			}
			if opt.beta2 < 0.0 || opt.beta2 >= 1.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"beta2 must be in [0, 1)",
				));
			}
			if opt.epsilon <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"epsilon must be positive",
				));
			}
			Ok(opt.into_py(py))
		}
		"LBFGS" => {
			let mut opt = PyLBFGS {
				memory_size: 10,
				max_line_search_iterations: 20,
				c1: 1e-4,
				c2: 0.9,
				initial_step_size: 1.0,
			};
			if let Some(kw) = kwargs {
				if let Ok(Some(ms)) = kw.get_item("memory_size") {
					opt.memory_size = ms.extract()?;
				}
				if let Ok(Some(mls)) = kw.get_item("max_line_search_iterations") {
					opt.max_line_search_iterations = mls.extract()?;
				}
				if let Ok(Some(c1)) = kw.get_item("c1") {
					opt.c1 = c1.extract()?;
				}
				if let Ok(Some(c2)) = kw.get_item("c2") {
					opt.c2 = c2.extract()?;
				}
				if let Ok(Some(iss)) = kw.get_item("initial_step_size") {
					opt.initial_step_size = iss.extract()?;
				}
			}
			// Validate parameters
			if opt.memory_size == 0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"memory_size must be positive",
				));
			}
			if opt.max_line_search_iterations == 0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"max_line_search_iterations must be positive",
				));
			}
			if opt.c1 <= 0.0 || opt.c1 >= 1.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"c1 must be in (0, 1)",
				));
			}
			if opt.c2 <= opt.c1 || opt.c2 >= 1.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"c2 must be in (c1, 1)",
				));
			}
			if opt.initial_step_size <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"initial_step_size must be positive",
				));
			}
			Ok(opt.into_py(py))
		}
		"ConjugateGradient" => {
			let mut opt = PyConjugateGradient {
				method: "FletcherReeves".to_string(),
				reset_every: None,
				max_line_search_iterations: 20,
				c1: 1e-4,
				c2: 0.1,
			};
			if let Some(kw) = kwargs {
				if let Ok(Some(m)) = kw.get_item("method") {
					opt.method = m.extract()?;
				}
				if let Ok(Some(re)) = kw.get_item("reset_every") {
					opt.reset_every = re.extract()?;
				}
				if let Ok(Some(mls)) = kw.get_item("max_line_search_iterations") {
					opt.max_line_search_iterations = mls.extract()?;
				}
				if let Ok(Some(c1)) = kw.get_item("c1") {
					opt.c1 = c1.extract()?;
				}
				if let Ok(Some(c2)) = kw.get_item("c2") {
					opt.c2 = c2.extract()?;
				}
			}
			// Validate method
			if ![
				"FletcherReeves",
				"PolakRibiere",
				"HestenesStiefel",
				"DaiYuan",
			]
			.contains(&opt.method.as_str())
			{
				return Err(pyo3::exceptions::PyValueError::new_err(
                    format!("Unknown CG method: {}. Available: FletcherReeves, PolakRibiere, HestenesStiefel, DaiYuan", opt.method)
                ));
			}
			Ok(opt.into_py(py))
		}
		"TrustRegion" => {
			let mut opt = PyTrustRegion {
				initial_radius: 1.0,
				max_radius: 10.0,
				eta: 0.1,
				radius_decrease_factor: 0.25,
				radius_increase_factor: 2.0,
				subproblem_solver: "CG".to_string(),
				max_subproblem_iterations: None,
			};
			if let Some(kw) = kwargs {
				if let Ok(Some(ir)) = kw.get_item("initial_radius") {
					opt.initial_radius = ir.extract()?;
				}
				if let Ok(Some(mr)) = kw.get_item("max_radius") {
					opt.max_radius = mr.extract()?;
				}
				if let Ok(Some(eta)) = kw.get_item("eta") {
					opt.eta = eta.extract()?;
				}
				if let Ok(Some(rdf)) = kw.get_item("radius_decrease_factor") {
					opt.radius_decrease_factor = rdf.extract()?;
				}
				if let Ok(Some(rif)) = kw.get_item("radius_increase_factor") {
					opt.radius_increase_factor = rif.extract()?;
				}
				if let Ok(Some(ss)) = kw.get_item("subproblem_solver") {
					opt.subproblem_solver = ss.extract()?;
				}
				if let Ok(Some(msi)) = kw.get_item("max_subproblem_iterations") {
					opt.max_subproblem_iterations = msi.extract()?;
				}
			}
			// Validate parameters
			if opt.initial_radius <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"initial_radius must be positive",
				));
			}
			if opt.max_radius <= opt.initial_radius {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"max_radius must be greater than initial_radius",
				));
			}
			Ok(opt.into_py(py))
		}
		"Newton" => {
			let mut opt = PyNewton {
				max_cg_iterations: None,
				cg_tolerance: 1e-6,
				use_line_search: true,
				alpha: 0.5,
				beta: 0.5,
				max_line_search_iterations: 20,
				force_positive_definite: true,
			};
			if let Some(kw) = kwargs {
				if let Ok(Some(mci)) = kw.get_item("max_cg_iterations") {
					opt.max_cg_iterations = mci.extract()?;
				}
				if let Ok(Some(ct)) = kw.get_item("cg_tolerance") {
					opt.cg_tolerance = ct.extract()?;
				}
				if let Ok(Some(uls)) = kw.get_item("use_line_search") {
					opt.use_line_search = uls.extract()?;
				}
				if let Ok(Some(a)) = kw.get_item("alpha") {
					opt.alpha = a.extract()?;
				}
				if let Ok(Some(b)) = kw.get_item("beta") {
					opt.beta = b.extract()?;
				}
				if let Ok(Some(mlsi)) = kw.get_item("max_line_search_iterations") {
					opt.max_line_search_iterations = mlsi.extract()?;
				}
				if let Ok(Some(fpd)) = kw.get_item("force_positive_definite") {
					opt.force_positive_definite = fpd.extract()?;
				}
			}
			// Validate parameters
			if opt.cg_tolerance <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"cg_tolerance must be positive",
				));
			}
			if opt.alpha <= 0.0 || opt.alpha >= 1.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"alpha must be in (0, 1)",
				));
			}
			if opt.beta <= 0.0 || opt.beta >= 1.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"beta must be in (0, 1)",
				));
			}
			Ok(opt.into_py(py))
		}
		"NaturalGradient" => {
			let mut opt = PyNaturalGradient {
				learning_rate: 0.01,
				fisher_damping: 1e-6,
				fisher_subsample: None,
				momentum: 0.0,
			};
			if let Some(kw) = kwargs {
				if let Ok(Some(lr)) = kw.get_item("learning_rate") {
					opt.learning_rate = lr.extract()?;
				}
				if let Ok(Some(fd)) = kw.get_item("fisher_damping") {
					opt.fisher_damping = fd.extract()?;
				}
				if let Ok(Some(fs)) = kw.get_item("fisher_subsample") {
					opt.fisher_subsample = fs.extract()?;
				}
				if let Ok(Some(m)) = kw.get_item("momentum") {
					opt.momentum = m.extract()?;
				}
			}
			// Validate parameters
			if opt.learning_rate <= 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"learning_rate must be positive",
				));
			}
			if opt.fisher_damping < 0.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"fisher_damping must be non-negative",
				));
			}
			if opt.momentum < 0.0 || opt.momentum >= 1.0 {
				return Err(pyo3::exceptions::PyValueError::new_err(
					"momentum must be in [0, 1)",
				));
			}
			Ok(opt.into_py(py))
		}
		_ => unreachable!(),
	}
}

/// Simple high-level optimization function.
///
/// This function provides a convenient interface for optimization on Riemannian manifolds.
///
/// Parameters
/// ----------
/// cost_function : CostFunction
///     The cost function to minimize
/// manifold : Manifold
///     The Riemannian manifold to optimize on
/// initial_point : array_like
///     Starting point for optimization
/// optimizer : str, default="Adam"
///     Name of the optimizer to use. Options: "SGD", "Adam", "LBFGS", "ConjugateGradient", "TrustRegion", "Newton", "NaturalGradient"
/// max_iterations : int, default=1000
///     Maximum number of iterations
/// gradient_tolerance : float, default=1e-6
///     Gradient norm tolerance for convergence
///
/// Returns
/// -------
/// OptimizationResult
///     Object containing the optimized point, final cost, and optimization statistics
#[pyfunction]
#[pyo3(signature = (
    cost_function,
    manifold,
    initial_point,
    optimizer="Adam",
    max_iterations=1000,
    gradient_tolerance=1e-6
))]
pub fn optimize(
	py: Python<'_>,
	cost_function: PyRef<'_, crate::py_cost::PyCostFunction>,
	manifold: PyObject,
	initial_point: PyObject,
	optimizer: &str,
	max_iterations: usize,
	gradient_tolerance: f64,
) -> PyResult<PyObject> {
	// Create optimizer using the centralized function
	let opt_obj = create_optimizer(py, optimizer, None)?;

	// Use the generic dispatcher via the optimize method
	// All optimizers now have a unified optimize method thanks to our refactoring
	use crate::py_optimizers::generic::optimize_dispatcher;

	// We need to determine which optimizer type we have
	// This is a bit verbose but necessary due to Rust's type system
	if let Ok(mut opt) = opt_obj.extract::<PyRefMut<PySGD>>(py) {
		optimize_dispatcher(
			&mut *opt,
			py,
			cost_function,
			manifold,
			initial_point,
			max_iterations,
			Some(gradient_tolerance),
			None,
			None,
			None,
		)
	} else if let Ok(mut opt) = opt_obj.extract::<PyRefMut<PyAdam>>(py) {
		optimize_dispatcher(
			&mut *opt,
			py,
			cost_function,
			manifold,
			initial_point,
			max_iterations,
			Some(gradient_tolerance),
			None,
			None,
			None,
		)
	} else if let Ok(mut opt) = opt_obj.extract::<PyRefMut<PyLBFGS>>(py) {
		optimize_dispatcher(
			&mut *opt,
			py,
			cost_function,
			manifold,
			initial_point,
			max_iterations,
			Some(gradient_tolerance),
			None,
			None,
			None,
		)
	} else if let Ok(mut opt) = opt_obj.extract::<PyRefMut<PyConjugateGradient>>(py) {
		optimize_dispatcher(
			&mut *opt,
			py,
			cost_function,
			manifold,
			initial_point,
			max_iterations,
			Some(gradient_tolerance),
			None,
			None,
			None,
		)
	} else if let Ok(mut opt) = opt_obj.extract::<PyRefMut<PyTrustRegion>>(py) {
		optimize_dispatcher(
			&mut *opt,
			py,
			cost_function,
			manifold,
			initial_point,
			max_iterations,
			Some(gradient_tolerance),
			None,
			None,
			None,
		)
	} else if let Ok(mut opt) = opt_obj.extract::<PyRefMut<PyNewton>>(py) {
		optimize_dispatcher(
			&mut *opt,
			py,
			cost_function,
			manifold,
			initial_point,
			max_iterations,
			Some(gradient_tolerance),
			None,
			None,
			None,
		)
	} else if let Ok(mut opt) = opt_obj.extract::<PyRefMut<PyNaturalGradient>>(py) {
		optimize_dispatcher(
			&mut *opt,
			py,
			cost_function,
			manifold,
			initial_point,
			max_iterations,
			Some(gradient_tolerance),
			None,
			None,
			None,
		)
	} else {
		Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
			"Failed to extract optimizer type",
		))
	}
}

/// Register all optimizer classes with the Python module.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
	let m = PyModule::new(parent.py(), "optimizers")?;

	// Register optimizer classes
	m.add_class::<PySGD>()?;
	m.add_class::<PyAdam>()?;
	m.add_class::<PyLBFGS>()?;
	m.add_class::<PyConjugateGradient>()?;
	m.add_class::<PyTrustRegion>()?;
	m.add_class::<PyNewton>()?;
	m.add_class::<PyNaturalGradient>()?;

	// Register result type
	m.add_class::<PyOptimizationResult>()?;

	// Add high-level functions
	m.add_function(wrap_pyfunction!(optimize, &m)?)?;
	m.add_function(wrap_pyfunction!(create_optimizer, &m)?)?;

	parent.add_submodule(&m)?;
	Ok(())
}
