//! Python callback support for optimization algorithms.
//!
//! This module provides a trait and implementation for calling Python callbacks
//! during optimization, allowing users to monitor and control the optimization process.

use pyo3::prelude::*;
use nalgebra::{Dim, DefaultAllocator};
use nalgebra::allocator::Allocator;
use riemannopt_core::optimizer::OptimizerStateLegacy as OptimizerState;
use riemannopt_core::error::{Result as CoreResult, ManifoldError};
use std::time::Duration;

/// Information passed to Python callbacks during optimization.
#[pyclass(name = "CallbackInfo")]
#[derive(Clone)]
pub struct PyCallbackInfo {
    /// Current iteration number
    #[pyo3(get)]
    pub iteration: usize,
    
    /// Current cost value
    #[pyo3(get)]
    pub cost: f64,
    
    /// Current gradient norm (if available)
    #[pyo3(get)]
    pub gradient_norm: Option<f64>,
    
    /// Elapsed time since optimization start
    #[pyo3(get)]
    pub elapsed_seconds: f64,
    
    /// Whether convergence has been achieved
    #[pyo3(get)]
    pub converged: bool,
}

#[pymethods]
impl PyCallbackInfo {
    fn __repr__(&self) -> String {
        format!(
            "CallbackInfo(iteration={}, cost={:.6e}, gradient_norm={:?}, elapsed={:.2}s, converged={})",
            self.iteration, self.cost, self.gradient_norm, self.elapsed_seconds, self.converged
        )
    }
}

/// Trait for optimization callbacks that can be called from Rust.
pub trait OptimizationCallback {
    /// Called at the start of optimization.
    fn on_optimization_start(&mut self) -> Result<(), PyErr>;
    
    /// Called at the end of each iteration.
    /// Returns true to continue, false to stop early.
    fn on_iteration_end(&mut self, info: &PyCallbackInfo) -> Result<bool, PyErr>;
    
    /// Called at the end of optimization.
    fn on_optimization_end(&mut self, info: &PyCallbackInfo) -> Result<(), PyErr>;
}

/// Wrapper for Python callbacks that implements the Rust callback trait.
pub struct PythonCallback {
    callback: PyObject,
}

impl PythonCallback {
    /// Create a new Python callback wrapper.
    pub fn new(callback: PyObject) -> Self {
        Self { callback }
    }
}

/// Rust callback adapter that wraps a Python callback.
pub struct RustCallbackAdapter<T, D>
where
    T: riemannopt_core::types::Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    python_callback: PythonCallback,
    _phantom: std::marker::PhantomData<(T, D)>,
}

impl<T, D> RustCallbackAdapter<T, D>
where
    T: riemannopt_core::types::Scalar + Into<f64> + Clone,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    pub fn new(callback: PyObject) -> Self {
        Self {
            python_callback: PythonCallback::new(callback),
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T, D> riemannopt_core::optimization::OptimizationCallback<T, D> for RustCallbackAdapter<T, D>
where
    T: riemannopt_core::types::Scalar + Into<f64> + Clone,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    fn on_optimization_start(&mut self) -> CoreResult<()> {
        self.python_callback.on_optimization_start()
            .map_err(|e| ManifoldError::numerical_error(format!("Callback error: {}", e)))
    }
    
    fn on_iteration_end(&mut self, info: &riemannopt_core::optimization::CallbackInfo<T, D>) -> CoreResult<bool> {
        let py_info = state_to_callback_info(&info.state, info.elapsed, info.converged);
        self.python_callback.on_iteration_end(&py_info)
            .map_err(|e| ManifoldError::numerical_error(format!("Callback error: {}", e)))
    }
    
    fn on_optimization_end(&mut self, info: &riemannopt_core::optimization::CallbackInfo<T, D>) -> CoreResult<()> {
        let py_info = state_to_callback_info(&info.state, info.elapsed, info.converged);
        self.python_callback.on_optimization_end(&py_info)
            .map_err(|e| ManifoldError::numerical_error(format!("Callback error: {}", e)))
    }
}

impl OptimizationCallback for PythonCallback {
    fn on_optimization_start(&mut self) -> Result<(), PyErr> {
        Python::with_gil(|py| {
            // Check if the callback has the method
            if self.callback.getattr(py, "on_optimization_start").is_ok() {
                self.callback.call_method0(py, "on_optimization_start")?;
            }
            Ok(())
        })
    }
    
    fn on_iteration_end(&mut self, info: &PyCallbackInfo) -> Result<bool, PyErr> {
        Python::with_gil(|py| {
            // Check if the callback has the method
            if let Ok(method) = self.callback.getattr(py, "on_iteration_end") {
                let result = method.call1(py, (info.clone(),))?;
                // If the method returns None or True, continue; if False, stop
                if let Ok(should_continue) = result.extract::<bool>(py) {
                    Ok(should_continue)
                } else {
                    // If not a bool, assume we should continue
                    Ok(true)
                }
            } else {
                // If method doesn't exist, continue
                Ok(true)
            }
        })
    }
    
    fn on_optimization_end(&mut self, info: &PyCallbackInfo) -> Result<(), PyErr> {
        Python::with_gil(|py| {
            // Check if the callback has the method
            if self.callback.getattr(py, "on_optimization_end").is_ok() {
                self.callback.call_method1(py, "on_optimization_end", (info.clone(),))?;
            }
            Ok(())
        })
    }
}

/// Convert optimizer state to callback info.
pub fn state_to_callback_info<T, D: Dim>(
    state: &OptimizerState<T, D>,
    elapsed: Duration,
    converged: bool,
) -> PyCallbackInfo 
where
    T: Into<f64> + Clone + riemannopt_core::types::Scalar,
    DefaultAllocator: Allocator<D>,
{
    PyCallbackInfo {
        iteration: state.iteration,
        cost: state.value.clone().into(),
        gradient_norm: state.gradient_norm.clone().map(|g| g.into()),
        elapsed_seconds: elapsed.as_secs_f64(),
        converged,
    }
}

/// Base class for Python callbacks.
///
/// Users can inherit from this class to implement custom callbacks:
///
/// ```python
/// class MyCallback(OptimizationCallback):
///     def on_optimization_start(self):
///         print("Starting optimization...")
///     
///     def on_iteration_end(self, info):
///         print(f"Iteration {info.iteration}: cost = {info.cost:.6f}")
///         # Return False to stop early
///         return info.cost < 1e-6
///     
///     def on_optimization_end(self, info):
///         print(f"Optimization complete: {info}")
/// ```
#[pyclass(name = "OptimizationCallback", subclass)]
pub struct PyOptimizationCallback;

#[pymethods]
impl PyOptimizationCallback {
    #[new]
    fn new() -> Self {
        Self
    }
    
    /// Called at the start of optimization.
    fn on_optimization_start(&self) {
        // Default implementation does nothing
    }
    
    /// Called at the end of each iteration.
    /// 
    /// Args:
    ///     info: Information about the current state
    ///     
    /// Returns:
    ///     bool: True to continue optimization, False to stop early
    fn on_iteration_end(&self, _info: &PyCallbackInfo) -> bool {
        // Default implementation continues
        true
    }
    
    /// Called at the end of optimization.
    fn on_optimization_end(&self, _info: &PyCallbackInfo) {
        // Default implementation does nothing
    }
}