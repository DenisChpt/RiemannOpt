//! Base types and utilities for Python optimizer wrappers.
//!
//! This module provides common functionality shared by all optimizer implementations,
//! including result types, callback adapters, and base traits.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use riemannopt_core::{
    optimization::{
        optimizer::{OptimizationResult as RustOptimizationResult, TerminationReason},
        callback::{OptimizationCallback, CallbackInfo},
    },
    types::Scalar,
};
use std::sync::Arc;
use parking_lot::RwLock;

use crate::{
    array_utils::{dvector_to_numpy, dmatrix_to_numpy},
    types::PyPoint,
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
    pub point: PyObject,
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
    pub history: Option<PyObject>,
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
    fn as_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
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
        let status = if self.converged { "✓ Converged" } else { "✗ Not converged" };
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
            self.gradient_norm.map(|g| format!("{:.6e}", g)).unwrap_or("N/A".to_string()),
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
    fn x(&self) -> PyObject {
        Python::with_gil(|py| self.point.clone_ref(py))
    }
}

impl PyOptimizationResult {
    /// Convert a Rust optimization result to Python format.
    pub fn from_rust_result<P>(
        py: Python<'_>,
        result: RustOptimizationResult<f64, P>,
        point_converter: impl FnOnce(&P) -> PyResult<PyObject>,
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
        }.to_string();
        
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
            history: None,  // Can be populated by callbacks
        })
    }
}

/// Adapter that allows Python callbacks to be used with Rust optimizers.
///
/// This handles GIL acquisition and data conversion for Python callbacks.
pub struct PyCallbackAdapter {
    callback: PyObject,
    /// Cache for converted data to minimize allocations
    cache: Arc<RwLock<CallbackCache>>,
}

struct CallbackCache {
    last_iteration: Option<usize>,
    last_point: Option<PyObject>,
}

impl PyCallbackAdapter {
    pub fn new(callback: PyObject) -> Self {
        PyCallbackAdapter {
            callback,
            cache: Arc::new(RwLock::new(CallbackCache {
                last_iteration: None,
                last_point: None,
            })),
        }
    }
}

impl<T, P, TV> OptimizationCallback<T, P, TV> for PyCallbackAdapter
where
    T: Scalar + numpy::Element,
    P: Clone + std::fmt::Debug + 'static,
    TV: Clone + std::fmt::Debug + 'static,
{
    fn on_optimization_start(&mut self) -> riemannopt_core::error::Result<()> {
        Python::with_gil(|py| {
            // Call Python callback
            let _ = self.callback.call_method0(py, "on_optimization_start");
        });
        Ok(())
    }
    
    fn on_iteration_end(&mut self, info: &CallbackInfo<T, P, TV>) -> riemannopt_core::error::Result<bool> {
        let result = Python::with_gil(|py| {
            // Check if we need to convert the point (optimization: cache if same iteration)
            let mut cache = self.cache.write();
            let py_point = if cache.last_iteration != Some(info.iteration) {
                match convert_point_to_python(py, &info.point) {
                    Ok(p) => {
                        cache.last_iteration = Some(info.iteration);
                        cache.last_point = Some(p.clone_ref(py));
                        p
                    },
                    Err(_) => return Ok(true), // Continue on error
                }
            } else {
                cache.last_point.as_ref().unwrap().clone_ref(py)
            };
            
            // Create info dict
            let info_dict = PyDict::new_bound(py);
            let _ = info_dict.set_item("iteration", info.iteration);
            let _ = info_dict.set_item("point", py_point);
            let _ = info_dict.set_item("value", info.value.to_f64());
            let _ = info_dict.set_item("gradient_norm", info.gradient_norm.map(|g| g.to_f64()));
            let _ = info_dict.set_item("elapsed_seconds", info.elapsed.as_secs_f64());
            let _ = info_dict.set_item("converged", info.converged);
            
            // Call Python callback, default to continuing if error
            match self.callback.call_method1(py, "on_iteration_end", (info_dict,)) {
                Ok(result) => {
                    // If callback returns False, stop optimization
                    Ok(result.extract::<bool>(py).unwrap_or(true))
                },
                Err(_) => Ok(true), // Continue on error
            }
        })?;
        Ok(result)
    }
    
    fn on_optimization_end(&mut self, info: &CallbackInfo<T, P, TV>) -> riemannopt_core::error::Result<()> {
        Python::with_gil(|py| {
            // Convert point to Python format
            let py_point = match convert_point_to_python(py, &info.point) {
                Ok(p) => p,
                Err(_) => return,
            };
            
            // Create info dict
            let info_dict = PyDict::new_bound(py);
            let _ = info_dict.set_item("point", py_point);
            let _ = info_dict.set_item("value", info.value.to_f64());
            let _ = info_dict.set_item("gradient_norm", info.gradient_norm.map(|g| g.to_f64()));
            let _ = info_dict.set_item("converged", info.converged);
            let _ = info_dict.set_item("iterations", info.iteration);
            let _ = info_dict.set_item("elapsed_seconds", info.elapsed.as_secs_f64());
            
            // Call Python callback
            let _ = self.callback.call_method1(py, "on_optimization_end", (info_dict,));
        });
        Ok(())
    }
}

/// Helper function to convert manifold points to Python format.
///
/// This is used by callbacks and result conversion.
fn convert_point_to_python<'a, P: 'static>(py: Python<'a>, point: &P) -> PyResult<PyObject> {
    // This is a placeholder - actual implementation depends on the point type
    // In practice, this would dispatch based on the actual type of P
    if let Some(py_point) = (point as &dyn std::any::Any).downcast_ref::<PyPoint>() {
        match py_point {
            PyPoint::Vector(vec) => Ok(dvector_to_numpy(py, vec)?.into()),
            PyPoint::Matrix(mat) => Ok(dmatrix_to_numpy(py, mat)?.into()),
        }
    } else if let Some(vec) = (point as &dyn std::any::Any).downcast_ref::<DVector<f64>>() {
        Ok(dvector_to_numpy(py, vec)?.into())
    } else if let Some(mat) = (point as &dyn std::any::Any).downcast_ref::<DMatrix<f64>>() {
        Ok(dmatrix_to_numpy(py, mat)?.into())
    } else {
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
            "Unsupported point type for Python conversion"
        ))
    }
}

/// Base trait for Python optimizer implementations.
///
/// This trait provides common functionality that all optimizers should implement.
pub trait PyOptimizerBase {
    /// Get the name of the optimizer.
    fn name(&self) -> &'static str;
    
    /// Validate that the configuration is valid.
    fn validate_config(&self) -> PyResult<()> {
        Ok(())
    }
}