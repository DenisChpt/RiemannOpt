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

mod sgd_simple;
mod adam;
mod lbfgs;
mod conjugate_gradient;
mod trust_region;
mod newton;
mod natural_gradient;
mod base;
pub mod generic;

pub use sgd_simple::PySGD;
pub use adam::PyAdam;
pub use lbfgs::PyLBFGS;
pub use conjugate_gradient::PyConjugateGradient;
pub use trust_region::PyTrustRegion;
pub use newton::PyNewton;
pub use natural_gradient::PyNaturalGradient;
pub use base::PyOptimizationResult;

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
        
    // Create optimizer instance based on name with default parameters
    let mut opt: Box<dyn std::any::Any> = match optimizer {
        "SGD" => Box::new(PySGD {
            learning_rate: 0.01,
            momentum: 0.0,
        }),
        "Adam" => Box::new(PyAdam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            amsgrad: false,
        }),
        "LBFGS" => Box::new(PyLBFGS {
            memory_size: 10,
            max_line_search_iterations: 20,
            c1: 1e-4,
            c2: 0.9,
            initial_step_size: 1.0,
        }),
        "ConjugateGradient" => Box::new(PyConjugateGradient {
            method: "FletcherReeves".to_string(),
            reset_every: None,
            max_line_search_iterations: 20,
            c1: 1e-4,
            c2: 0.1,
        }),
        "TrustRegion" => Box::new(PyTrustRegion {
            initial_radius: 1.0,
            max_radius: 10.0,
            eta: 0.1,
            radius_decrease_factor: 0.25,
            radius_increase_factor: 2.0,
            subproblem_solver: "CG".to_string(),
            max_subproblem_iterations: None,
        }),
        "Newton" => Box::new(PyNewton {
            max_cg_iterations: None,
            cg_tolerance: 1e-6,
            use_line_search: true,
            alpha: 0.5,
            beta: 0.5,
            max_line_search_iterations: 20,
            force_positive_definite: true,
        }),
        "NaturalGradient" => Box::new(PyNaturalGradient {
            learning_rate: 0.01,
            fisher_damping: 1e-6,
            fisher_subsample: None,
            momentum: 0.0,
        }),
        _ => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            format!("Unknown optimizer: {}. Available: SGD, Adam, LBFGS, ConjugateGradient, TrustRegion, Newton, NaturalGradient", optimizer)
        )),
    };
    
    // Now use the unified optimize method
    match optimizer {
        "SGD" => opt.downcast_mut::<PySGD>().unwrap().optimize(
            py, cost_function, manifold, initial_point, max_iterations, 
            Some(gradient_tolerance), None, None, None
        ),
        "Adam" => opt.downcast_mut::<PyAdam>().unwrap().optimize(
            py, cost_function, manifold, initial_point, max_iterations, 
            Some(gradient_tolerance), None, None, None
        ),
        "LBFGS" => opt.downcast_mut::<PyLBFGS>().unwrap().optimize(
            py, cost_function, manifold, initial_point, max_iterations, 
            Some(gradient_tolerance), None, None, None
        ),
        "ConjugateGradient" => opt.downcast_mut::<PyConjugateGradient>().unwrap().optimize(
            py, cost_function, manifold, initial_point, max_iterations, 
            Some(gradient_tolerance), None, None, None
        ),
        "TrustRegion" => opt.downcast_mut::<PyTrustRegion>().unwrap().optimize(
            py, cost_function, manifold, initial_point, max_iterations, 
            Some(gradient_tolerance), None, None, None
        ),
        "Newton" => opt.downcast_mut::<PyNewton>().unwrap().optimize(
            py, cost_function, manifold, initial_point, max_iterations, 
            Some(gradient_tolerance), None, None, None
        ),
        "NaturalGradient" => opt.downcast_mut::<PyNaturalGradient>().unwrap().optimize(
            py, cost_function, manifold, initial_point, max_iterations, 
            Some(gradient_tolerance), None, None, None
        ),
        _ => unreachable!(),
    }
}

/// Register all optimizer classes with the Python module.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "optimizers")?;
    
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
    
    // Add high-level optimize function
    m.add_function(wrap_pyfunction!(optimize, &m)?)?;
    
    parent.add_submodule(&m)?;
    Ok(())
}

// TODO: Implement high-level optimize() function after individual optimizers are working