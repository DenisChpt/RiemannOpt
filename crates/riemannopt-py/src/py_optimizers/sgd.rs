//! Python wrapper for the Riemannian SGD optimizer.
//!
//! This module provides a high-performance Python interface to the Rust SGD optimizer,
//! handling all necessary type conversions and workspace management internally.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use riemannopt_optim::{SGD, SGDConfig, MomentumMethod};
use riemannopt_core::{
    optimization::{
        optimizer::{Optimizer, StoppingCriterion},
        step_size::StepSizeSchedule,
    },
};
use std::time::Duration;

use crate::{
    py_manifolds::{
        sphere::PySphere,
        stiefel::PyStiefel,
    },
    py_cost::PyCostFunction,
    array_utils::{numpy_to_dvector, numpy_to_dmatrix, dvector_to_numpy, dmatrix_to_numpy},
    error::to_py_err,
};
use super::base::PyOptimizationResult;

/// Stochastic Gradient Descent optimizer for Riemannian manifolds.
///
/// SGD is the fundamental optimization algorithm, here extended to handle
/// the non-Euclidean geometry of manifolds through retraction operations
/// and proper handling of the Riemannian metric.
///
/// Parameters
/// ----------
/// learning_rate : float, default=0.01
///     The step size for gradient descent. Can be a constant or use a schedule.
/// momentum : float, default=0.0
///     Momentum coefficient in [0, 1). Higher values give more momentum.
///     If 0, no momentum is used (pure gradient descent).
/// nesterov : bool, default=False
///     Whether to use Nesterov accelerated gradient instead of classical momentum.
/// gradient_clip : float, optional
///     If provided, gradients are clipped to this norm to prevent exploding gradients.
/// line_search : bool, default=False
///     Whether to use backtracking line search for adaptive step size.
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> # Define a simple optimization problem on the sphere
/// >>> sphere = ro.manifolds.Sphere(10)
/// >>> 
/// >>> def cost(x):
/// ...     return -x[0]  # Maximize first component
/// >>> 
/// >>> # Create optimizer with momentum
/// >>> optimizer = ro.optimizers.SGD(learning_rate=0.1, momentum=0.9)
/// >>> 
/// >>> # Run optimization
/// >>> x0 = sphere.random_point()
/// >>> result = optimizer.optimize(
/// ...     cost_function=cost,
/// ...     manifold=sphere,
/// ...     initial_point=x0,
/// ...     max_iterations=100
/// ... )
/// >>> print(f"Final value: {result.value:.4f}")
#[pyclass(name = "SGD", module = "riemannopt.optimizers")]
pub struct PySGD {
    /// Step size (learning rate)
    learning_rate: f64,
    /// Momentum coefficient
    momentum: f64,
    /// Whether to use Nesterov acceleration
    nesterov: bool,
    /// Gradient clipping threshold
    gradient_clip: Option<f64>,
    /// Whether to use line search
    line_search: bool,
}

impl PySGD {
    fn validate_config(&self) -> PyResult<()> {
        if self.learning_rate <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "learning_rate must be positive"
            ));
        }
        if self.momentum < 0.0 || self.momentum >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "momentum must be in [0, 1)"
            ));
        }
        if let Some(clip) = self.gradient_clip {
            if clip <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "gradient_clip must be positive"
                ));
            }
        }
        Ok(())
    }
}

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer.
    #[new]
    #[pyo3(signature = (learning_rate=0.01, momentum=0.0, nesterov=false, gradient_clip=None, line_search=false))]
    fn new(
        learning_rate: f64,
        momentum: f64,
        nesterov: bool,
        gradient_clip: Option<f64>,
        line_search: bool,
    ) -> PyResult<Self> {
        let sgd = PySGD {
            learning_rate,
            momentum,
            nesterov,
            gradient_clip,
            line_search,
        };
        sgd.validate_config()?;
        Ok(sgd)
    }
    
    /// String representation of the optimizer.
    fn __repr__(&self) -> String {
        format!(
            "SGD(learning_rate={}, momentum={}, nesterov={}, gradient_clip={:?}, line_search={})",
            self.learning_rate, self.momentum, self.nesterov, self.gradient_clip, self.line_search
        )
    }
    
    /// Get optimizer configuration as a dictionary.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("learning_rate", self.learning_rate)?;
        dict.set_item("momentum", self.momentum)?;
        dict.set_item("nesterov", self.nesterov)?;
        dict.set_item("gradient_clip", self.gradient_clip)?;
        dict.set_item("line_search", self.line_search)?;
        Ok(dict.into())
    }
    
    /// Run optimization.
    ///
    /// Parameters
    /// ----------
    /// cost_function : CostFunction
    ///     The objective function to minimize.
    /// manifold : Manifold
    ///     The manifold to optimize on.
    /// initial_point : array_like
    ///     Starting point on the manifold.
    /// max_iterations : int, default=1000
    ///     Maximum number of iterations.
    /// gradient_tolerance : float, default=1e-6
    ///     Stop when gradient norm is below this threshold.
    /// function_tolerance : float, default=1e-8
    ///     Stop when function value change is below this threshold.
    /// point_tolerance : float, default=1e-8
    ///     Stop when point change is below this threshold.
    /// max_time : float, optional
    ///     Maximum time in seconds.
    /// target_value : float, optional
    ///     Stop when function value is below this target.
    /// callback : callable, optional
    ///     Function called after each iteration.
    ///
    /// Returns
    /// -------
    /// OptimizationResult
    ///     Result containing the optimal point and optimization metadata.
    #[pyo3(signature = (
        cost_function,
        manifold,
        initial_point,
        max_iterations=1000,
        gradient_tolerance=1e-6,
        function_tolerance=1e-8,
        point_tolerance=1e-8,
        max_time=None,
        target_value=None,
        callback=None
    ))]
    pub fn optimize(
        &mut self,
        py: Python<'_>,
        cost_function: PyObject,
        manifold: PyObject,
        initial_point: PyObject,
        max_iterations: usize,
        gradient_tolerance: f64,
        function_tolerance: f64,
        point_tolerance: f64,
        max_time: Option<f64>,
        target_value: Option<f64>,
        callback: Option<PyObject>,
    ) -> PyResult<PyOptimizationResult> {
        // Validate configuration
        self.validate_config()?;
        
        // Extract cost function
        let cost_fn = cost_function.extract::<PyRef<PyCostFunction>>(py)?;
        
        // Create stopping criterion
        let mut criterion = StoppingCriterion::new()
            .with_max_iterations(max_iterations)
            .with_gradient_tolerance(gradient_tolerance)
            .with_function_tolerance(function_tolerance)
            .with_point_tolerance(point_tolerance);
        
        if let Some(max_t) = max_time {
            criterion = criterion.with_max_time(Duration::from_secs_f64(max_t));
        }
        
        if let Some(target) = target_value {
            criterion = criterion.with_target_value(target);
        }
        
        // Create SGD configuration
        let momentum_method = if self.momentum > 0.0 {
            if self.nesterov {
                MomentumMethod::Nesterov { coefficient: self.momentum }
            } else {
                MomentumMethod::Classical { coefficient: self.momentum }
            }
        } else {
            MomentumMethod::None
        };
        
        let sgd_config = SGDConfig {
            step_size: StepSizeSchedule::Constant(self.learning_rate),
            momentum: momentum_method,
            gradient_clip: self.gradient_clip,
            line_search: if self.line_search {
                Some(riemannopt_core::optimization::line_search::BacktrackingLineSearch::default())
            } else {
                None
            },
        };
        
        // Create optimizer
        let mut sgd = SGD::new(sgd_config);
        
        // TODO: Add callback support when available in Rust API
        let _callback = callback;  // Unused for now
        
        // Dispatch based on manifold type
        if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>(py) {
            self.optimize_on_sphere(py, &mut sgd, cost_fn, &sphere, initial_point, &criterion)
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>(py) {
            self.optimize_on_stiefel(py, &mut sgd, cost_fn, &stiefel, initial_point, &criterion)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported manifold type"
            ))
        }
    }
    
    /// Set the learning rate.
    #[setter]
    fn set_learning_rate(&mut self, value: f64) -> PyResult<()> {
        if value <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "learning_rate must be positive"
            ));
        }
        self.learning_rate = value;
        Ok(())
    }
    
    /// Set the momentum coefficient.
    #[setter]
    fn set_momentum(&mut self, value: f64) -> PyResult<()> {
        if value < 0.0 || value >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "momentum must be in [0, 1)"
            ));
        }
        self.momentum = value;
        Ok(())
    }
}

// Private implementation methods
impl PySGD {
    /// Optimize on sphere manifold.
    fn optimize_on_sphere(
        &self,
        py: Python<'_>,
        optimizer: &mut SGD<f64>,
        cost_fn: PyRef<'_, PyCostFunction>,
        sphere: &PySphere,
        initial_point: PyObject,
        criterion: &StoppingCriterion<f64>,
    ) -> PyResult<PyOptimizationResult> {
        // Convert initial point
        let x0_array = initial_point.downcast_bound::<PyArray1<f64>>(py)?;
        let x0 = numpy_to_dvector(x0_array.readonly())?;
        
        // Get the inner manifold
        let manifold = &sphere.inner;
        
        // Create adapter for the cost function
        let cost_fn_adapter = crate::py_cost::PyCostFunctionSphere::new(&*cost_fn);
        
        // Run optimization
        let result = optimizer.optimize(&cost_fn_adapter, manifold, &x0, criterion)
            .map_err(to_py_err)?;
        
        // Convert result
        PyOptimizationResult::from_rust_result(py, result, |point| {
            Ok(dvector_to_numpy(py, point)?.into())
        })
    }
    
    /// Optimize on Stiefel manifold.
    fn optimize_on_stiefel(
        &self,
        py: Python<'_>,
        optimizer: &mut SGD<f64>,
        cost_fn: PyRef<'_, PyCostFunction>,
        stiefel: &PyStiefel,
        initial_point: PyObject,
        criterion: &StoppingCriterion<f64>,
    ) -> PyResult<PyOptimizationResult> {
        // Convert initial point
        let x0_array = initial_point.downcast_bound::<PyArray2<f64>>(py)?;
        let x0 = numpy_to_dmatrix(x0_array.readonly())?;
        
        // Get the inner manifold
        let manifold = &stiefel.inner;
        
        // Create adapter for the cost function
        let cost_fn_adapter = crate::py_cost::PyCostFunctionStiefel::new(&*cost_fn);
        
        // Run optimization
        let result = optimizer.optimize(&cost_fn_adapter, manifold, &x0, criterion)
            .map_err(to_py_err)?;
        
        // Convert result
        PyOptimizationResult::from_rust_result(py, result, |point| {
            Ok(dmatrix_to_numpy(py, point)?.into())
        })
    }
}