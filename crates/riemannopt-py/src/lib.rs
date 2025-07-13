//! Python bindings for RiemannOpt.
//!
//! This module provides PyO3 bindings to expose RiemannOpt's functionality
//! to Python users, with NumPy integration for efficient array handling.
//!
//! Architecture:
//! - `types`: Core type definitions for Python interop (PyPoint, PyTangentVector)
//! - `py_manifolds`: Python wrappers for manifold implementations
//! - `py_optimizers`: Python wrappers for optimization algorithms
//! - `py_cost`: Python cost function interface
//! - `error`: Error conversion utilities
//! - `array_utils`: NumPy/nalgebra conversion utilities

use pyo3::prelude::*;

// Core modules
mod types;
mod error;
mod array_utils;

// Feature modules
#[macro_use]
mod py_manifolds;
mod py_optimizers;
mod py_cost;

// Re-exports for easier access
// use error::RiemannOptError;

/// RiemannOpt: High-performance Riemannian optimization in Python.
///
/// This module provides tools for optimization on Riemannian manifolds,
/// including various manifold types (Sphere, Stiefel, Grassmann, etc.)
/// and optimization algorithms (SGD, Adam, L-BFGS, etc.).
#[pymodule]
fn _riemannopt(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add version information
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    
    // Initialize submodules
    py_manifolds::register_module(m)?;
    py_optimizers::register_module(m)?;
    py_cost::register_module(m)?;
    
    // Register all error types
    error::register_exceptions(m)?;
    
    Ok(())
}