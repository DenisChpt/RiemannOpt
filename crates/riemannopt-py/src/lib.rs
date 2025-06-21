//! Python bindings for RiemannOpt.
//!
//! This module provides PyO3 bindings to expose RiemannOpt's functionality
//! to Python users, with NumPy integration for efficient array handling.

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

mod manifolds;
mod manifolds_oblique;
mod manifolds_fixed_rank;
mod manifolds_psd_cone;
mod manifolds_optimized;
mod optimizers;
mod optimizers_newton;
mod cost_function;
mod utils;
mod validation;
mod array_utils;
mod callbacks;

use manifolds::{PySphere, PyEuclidean, PyHyperbolic, PyProductManifold, PyProductManifoldStatic, check_point_on_manifold, check_vector_in_tangent_space};
use manifolds_oblique::PyOblique;
use manifolds_fixed_rank::PyFixedRank;
use manifolds_psd_cone::PyPSDCone;
use manifolds_optimized::{PyStiefel, PyGrassmann, PySPD};
use optimizers::*;
use optimizers_newton::PyNewton;
use cost_function::{PyCostFunction, quadratic_cost, rosenbrock_cost};
use utils::{format_result, validate_point_shape, default_line_search};
use callbacks::{PyOptimizationCallback, PyCallbackInfo};

/// RiemannOpt: High-performance Riemannian optimization in Python.
///
/// This module provides tools for optimization on Riemannian manifolds,
/// including various manifold types (Sphere, Stiefel, Grassmann, etc.)
/// and optimization algorithms (SGD, Adam, L-BFGS, etc.).
#[pymodule]
fn _riemannopt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Add manifold classes
    m.add_class::<PySphere>()?;
    m.add_class::<PyStiefel>()?;
    m.add_class::<PyGrassmann>()?;
    m.add_class::<PyEuclidean>()?;
    m.add_class::<PySPD>()?;
    m.add_class::<PyHyperbolic>()?;
    m.add_class::<PyProductManifold>()?;
    m.add_class::<PyProductManifoldStatic>()?;
    m.add_class::<PyOblique>()?;
    m.add_class::<PyFixedRank>()?;
    m.add_class::<PyPSDCone>()?;
    
    // Add optimizer classes
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyLBFGS>()?;
    m.add_class::<PyConjugateGradient>()?;
    m.add_class::<PyTrustRegion>()?;
    m.add_class::<PyNewton>()?;
    
    // Add cost function class
    m.add_class::<PyCostFunction>()?;
    
    // Add callback classes
    m.add_class::<PyOptimizationCallback>()?;
    m.add_class::<PyCallbackInfo>()?;
    
    // Add utility functions
    m.add_function(wrap_pyfunction!(check_point_on_manifold, m)?)?;
    m.add_function(wrap_pyfunction!(check_vector_in_tangent_space, m)?)?;
    m.add_function(wrap_pyfunction!(quadratic_cost, m)?)?;
    m.add_function(wrap_pyfunction!(rosenbrock_cost, m)?)?;
    m.add_function(wrap_pyfunction!(format_result, m)?)?;
    m.add_function(wrap_pyfunction!(validate_point_shape, m)?)?;
    m.add_function(wrap_pyfunction!(default_line_search, m)?)?;
    
    // Add submodules
    let manifolds = PyModule::new_bound(m.py(), "manifolds")?;
    init_manifolds_module(&manifolds)?;
    m.add_submodule(&manifolds)?;
    
    let optimizers = PyModule::new_bound(m.py(), "optimizers")?; 
    init_optimizers_module(&optimizers)?;
    m.add_submodule(&optimizers)?;
    
    Ok(())
}

/// Initialize the manifolds submodule.
fn init_manifolds_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySphere>()?;
    m.add_class::<PyStiefel>()?;
    m.add_class::<PyGrassmann>()?;
    m.add_class::<PyEuclidean>()?;
    m.add_class::<PySPD>()?;
    m.add_class::<PyHyperbolic>()?;
    m.add_class::<PyProductManifold>()?;
    m.add_class::<PyProductManifoldStatic>()?;
    m.add_class::<PyOblique>()?;
    m.add_class::<PyFixedRank>()?;
    m.add_class::<PyPSDCone>()?;
    Ok(())
}

/// Initialize the optimizers submodule.
fn init_optimizers_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySGD>()?;
    m.add_class::<PyAdam>()?;
    m.add_class::<PyLBFGS>()?;
    m.add_class::<PyConjugateGradient>()?;
    m.add_class::<PyTrustRegion>()?;
    m.add_class::<PyNewton>()?;
    Ok(())
}