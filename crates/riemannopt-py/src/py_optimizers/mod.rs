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
use numpy::PyArrayMethods;

mod sgd_simple;
mod adam;
mod lbfgs;
mod conjugate_gradient;
mod trust_region;
mod newton;
mod base;

pub use sgd_simple::PySGD;
pub use adam::PyAdam;
pub use lbfgs::PyLBFGS;
pub use conjugate_gradient::PyConjugateGradient;
pub use trust_region::PyTrustRegion;
pub use newton::PyNewton;
pub use base::PyOptimizationResult;

/// Simple high-level optimization function.
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
    // Simple dispatch for sphere only for now
    if let Ok(sphere) = manifold.extract::<PyRef<crate::py_manifolds::sphere::PySphere>>(py) {
        match optimizer {
            "SGD" => {
                let mut sgd = PySGD::new(0.01, 0.0)?;
                if let Ok(initial_array) = initial_point.downcast_bound::<numpy::PyArray1<f64>>(py) {
                    sgd.optimize_sphere(py, cost_function, sphere, initial_array.readonly(), max_iterations)
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "initial_point must be a 1D numpy array for Sphere manifold"
                    ))
                }
            },
            "Adam" => {
                let mut adam = PyAdam::new(0.001, 0.9, 0.999, 1e-8, false)?;
                if let Ok(initial_array) = initial_point.downcast_bound::<numpy::PyArray1<f64>>(py) {
                    adam.optimize_sphere(py, cost_function, sphere, initial_array.readonly(), max_iterations, Some(gradient_tolerance))
                } else {
                    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "initial_point must be a 1D numpy array for Sphere manifold"
                    ))
                }
            },
            _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Unknown optimizer: {}. Available: SGD, Adam", optimizer)
            )),
        }
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Only Sphere manifold is supported in this simplified version"
        ))
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
    
    // Register result type
    m.add_class::<PyOptimizationResult>()?;
    
    // Add high-level optimize function
    m.add_function(wrap_pyfunction!(optimize, &m)?)?;
    
    parent.add_submodule(&m)?;
    Ok(())
}

// TODO: Implement high-level optimize() function after individual optimizers are working