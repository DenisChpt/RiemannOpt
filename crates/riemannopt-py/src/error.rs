//! Error handling and conversion utilities for Python bindings.
//!
//! This module provides comprehensive error conversion between Rust errors
//! from riemannopt-core and Python exceptions, ensuring users get meaningful
//! error messages with proper Python exception types.

use pyo3::prelude::*;
use pyo3::exceptions::{PyValueError, PyTypeError, PyRuntimeError, PyIndexError, PyException};
use pyo3::create_exception;
use riemannopt_core::error::ManifoldError;
use std::fmt;

// Create custom Python exceptions for RiemannOpt
create_exception!(_riemannopt, RiemannOptError, PyException, "Base exception for RiemannOpt errors");
create_exception!(_riemannopt, ManifoldValidationError, RiemannOptError, "Error in manifold validation");
create_exception!(_riemannopt, OptimizationFailedError, RiemannOptError, "Optimization algorithm failed");
create_exception!(_riemannopt, ConvergenceError, OptimizationFailedError, "Algorithm failed to converge");
create_exception!(_riemannopt, LineSearchError, OptimizationFailedError, "Line search failed");
create_exception!(_riemannopt, DimensionMismatchError, RiemannOptError, "Dimension mismatch in operations");
create_exception!(_riemannopt, NumericalError, RiemannOptError, "Numerical computation error");
create_exception!(_riemannopt, NotImplementedMethodError, RiemannOptError, "Method not implemented");

/// Unified error type for the Python bindings.
#[derive(Debug)]
pub enum PyRiemannOptError {
    /// Manifold-related errors
    Manifold(ManifoldError),
    /// Optimization-related errors
    Optimization(String),
    /// Array conversion errors
    ArrayConversion(crate::array_utils::ArrayConversionError),
    /// Type errors
    TypeError(String),
    /// Value errors
    ValueError(String),
    /// Index errors
    IndexError(String),
    /// Generic runtime errors
    RuntimeError(String),
}

impl fmt::Display for PyRiemannOptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyRiemannOptError::Manifold(e) => write!(f, "Manifold error: {}", e),
            PyRiemannOptError::Optimization(e) => write!(f, "Optimization error: {}", e),
            PyRiemannOptError::ArrayConversion(e) => write!(f, "Array conversion error: {}", e),
            PyRiemannOptError::TypeError(msg) => write!(f, "Type error: {}", msg),
            PyRiemannOptError::ValueError(msg) => write!(f, "Value error: {}", msg),
            PyRiemannOptError::IndexError(msg) => write!(f, "Index error: {}", msg),
            PyRiemannOptError::RuntimeError(msg) => write!(f, "Runtime error: {}", msg),
        }
    }
}

impl std::error::Error for PyRiemannOptError {}

impl From<ManifoldError> for PyRiemannOptError {
    fn from(err: ManifoldError) -> Self {
        PyRiemannOptError::Manifold(err)
    }
}


impl From<crate::array_utils::ArrayConversionError> for PyRiemannOptError {
    fn from(err: crate::array_utils::ArrayConversionError) -> Self {
        PyRiemannOptError::ArrayConversion(err)
    }
}

impl From<PyRiemannOptError> for PyErr {
    fn from(err: PyRiemannOptError) -> PyErr {
        match err {
            PyRiemannOptError::Manifold(e) => match &e {
                ManifoldError::InvalidPoint { .. } => {
                    ManifoldValidationError::new_err(e.to_string())
                }
                ManifoldError::InvalidTangent { .. } => {
                    ManifoldValidationError::new_err(e.to_string())
                }
                ManifoldError::DimensionMismatch { .. } => {
                    DimensionMismatchError::new_err(e.to_string())
                }
                ManifoldError::NumericalError { .. } => {
                    NumericalError::new_err(e.to_string())
                }
                ManifoldError::NotImplemented { .. } => {
                    NotImplementedMethodError::new_err(e.to_string())
                }
            },
            PyRiemannOptError::Optimization(e) => {
                OptimizationFailedError::new_err(e)
            },
            PyRiemannOptError::ArrayConversion(e) => {
                PyValueError::new_err(e.to_string())
            }
            PyRiemannOptError::TypeError(msg) => PyTypeError::new_err(msg),
            PyRiemannOptError::ValueError(msg) => PyValueError::new_err(msg),
            PyRiemannOptError::IndexError(msg) => PyIndexError::new_err(msg),
            PyRiemannOptError::RuntimeError(msg) => PyRuntimeError::new_err(msg),
        }
    }
}

/// Convert a ManifoldError to a Python exception.
pub fn to_py_err(err: ManifoldError) -> PyErr {
    PyRiemannOptError::from(err).into()
}


/// Convert an OptimizerError to a Python exception.
pub fn optimization_to_py_err(err: String) -> PyErr {
    PyRiemannOptError::Optimization(err).into()
}

/// Helper function to create a dimension mismatch error.
pub fn dimension_mismatch(expected: &[usize], got: &[usize]) -> PyErr {
    DimensionMismatchError::new_err(format!(
        "Expected dimensions {:?}, but got {:?}",
        expected, got
    ))
}

/// Helper function to create a type error.
pub fn type_error(expected: &str, got: &str) -> PyErr {
    PyTypeError::new_err(format!("Expected {}, but got {}", expected, got))
}

/// Helper function to create a value error.
pub fn value_error(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

/// Register all custom exceptions with the Python module.
pub fn register_exceptions(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add("RiemannOptError", module.py().get_type_bound::<RiemannOptError>())?;
    module.add("ManifoldValidationError", module.py().get_type_bound::<ManifoldValidationError>())?;
    module.add("OptimizationFailedError", module.py().get_type_bound::<OptimizationFailedError>())?;
    module.add("ConvergenceError", module.py().get_type_bound::<ConvergenceError>())?;
    module.add("LineSearchError", module.py().get_type_bound::<LineSearchError>())?;
    module.add("DimensionMismatchError", module.py().get_type_bound::<DimensionMismatchError>())?;
    module.add("NumericalError", module.py().get_type_bound::<NumericalError>())?;
    module.add("NotImplementedMethodError", module.py().get_type_bound::<NotImplementedMethodError>())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_conversion() {
        let manifold_err = ManifoldError::InvalidPoint {
            reason: "test point".to_string(),
        };
        
        let py_err: PyErr = PyRiemannOptError::from(manifold_err).into();
        // In actual Python context, this would raise the appropriate exception
        assert!(format!("{}", py_err).contains("test point"));
    }
}