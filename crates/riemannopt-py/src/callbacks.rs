//! Python callback support for optimization algorithms.
//!
//! This module provides a trait and implementation for calling Python callbacks
//! during optimization, allowing users to monitor and control the optimization process.

use pyo3::prelude::*;

/// Information passed to Python callbacks during optimization.
#[pyclass(name = "CallbackInfo")]
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
    
    /// Step size used in the current iteration (if available)
    #[pyo3(get)]
    pub step_size: Option<f64>,
    
    /// Line search iterations (if applicable)
    #[pyo3(get)]
    pub line_search_iterations: Option<usize>,
    
    /// Trust region radius (if applicable)
    #[pyo3(get)]
    pub trust_region_radius: Option<f64>,
    
    /// Whether the step was accepted (for trust region methods)
    #[pyo3(get)]
    pub step_accepted: Option<bool>,
    
    /// Relative improvement in cost
    #[pyo3(get)]
    pub relative_improvement: Option<f64>,
    
    /// Optimizer-specific information as a dictionary
    #[pyo3(get)]
    pub extra_info: Option<PyObject>,
}

#[pymethods]
impl PyCallbackInfo {
    fn __repr__(&self) -> String {
        let mut parts = vec![
            format!("iteration={}", self.iteration),
            format!("cost={:.6e}", self.cost),
        ];
        
        if let Some(grad_norm) = self.gradient_norm {
            parts.push(format!("gradient_norm={:.6e}", grad_norm));
        }
        
        parts.push(format!("elapsed={:.2}s", self.elapsed_seconds));
        parts.push(format!("converged={}", self.converged));
        
        if let Some(step_size) = self.step_size {
            parts.push(format!("step_size={:.6e}", step_size));
        }
        
        format!("CallbackInfo({})", parts.join(", "))
    }
    
    /// Get the current point (if available)
    /// Returns None if point tracking is disabled for memory efficiency
    #[getter]
    fn point(&self, _py: Python<'_>) -> Option<PyObject> {
        // This will be populated by the optimizer if point tracking is enabled
        None
    }

    /// Get the current gradient (if available)
    /// Returns None if gradient tracking is disabled
    #[getter]
    fn gradient(&self, _py: Python<'_>) -> Option<PyObject> {
        // This will be populated by the optimizer if gradient tracking is enabled
        None
    }

    /// Get the search direction (if available)
    #[getter]
    fn search_direction(&self, _py: Python<'_>) -> Option<PyObject> {
        // This will be populated by the optimizer if available
        None
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