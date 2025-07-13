//! Python wrapper for the Adam optimizer.
//!
//! Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization
//! algorithm that combines ideas from momentum and RMSprop.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use riemannopt_optim::{Adam, AdamConfig};
use riemannopt_core::{
    optimization::{
        optimizer::{Optimizer, StoppingCriterion},
        step_size::StepSizeSchedule,
    },
};
use std::time::Duration;

use crate::{
    py_manifolds::{sphere::PySphere, stiefel::PyStiefel},
    py_cost::{PyCostFunction, PyCostFunctionSphere, PyCostFunctionStiefel},
    array_utils::{numpy_to_dvector, numpy_to_dmatrix, dvector_to_numpy, dmatrix_to_numpy},
    error::to_py_err,
};
use super::base::PyOptimizationResult;

/// Adam optimizer for Riemannian manifolds.
///
/// Adam adapts the learning rate for each parameter based on estimates of
/// first and second moments of the gradients. This Riemannian version properly
/// handles the manifold geometry.
///
/// Parameters
/// ----------
/// learning_rate : float, default=0.001
///     The initial step size.
/// beta1 : float, default=0.9
///     The exponential decay rate for the first moment estimates.
/// beta2 : float, default=0.999
///     The exponential decay rate for the second moment estimates.
/// epsilon : float, default=1e-8
///     A small constant for numerical stability.
/// amsgrad : bool, default=False
///     Whether to use the AMSGrad variant of Adam.
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> sphere = ro.manifolds.Sphere(10)
/// >>> optimizer = ro.optimizers.Adam(learning_rate=0.001)
/// >>> 
/// >>> # Define cost function
/// >>> def cost(x):
/// ...     return np.sum(x**2)  # Simple quadratic
/// >>> 
/// >>> x0 = sphere.random_point()
/// >>> result = optimizer.optimize(
/// ...     cost_function=cost,
/// ...     manifold=sphere,
/// ...     initial_point=x0,
/// ...     max_iterations=1000
/// ... )
#[pyclass(name = "Adam", module = "riemannopt.optimizers")]
pub struct PyAdam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    amsgrad: bool,
}

#[pymethods]
impl PyAdam {
    #[new]
    #[pyo3(signature = (learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, amsgrad=false))]
    pub fn new(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        amsgrad: bool,
    ) -> PyResult<Self> {
        // Validate parameters
        if learning_rate <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "learning_rate must be positive"
            ));
        }
        if beta1 < 0.0 || beta1 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beta1 must be in [0, 1)"
            ));
        }
        if beta2 < 0.0 || beta2 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beta2 must be in [0, 1)"
            ));
        }
        if epsilon <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "epsilon must be positive"
            ));
        }
        
        Ok(PyAdam {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            amsgrad,
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "Adam(learning_rate={}, beta1={}, beta2={}, epsilon={}, amsgrad={})",
            self.learning_rate, self.beta1, self.beta2, self.epsilon, self.amsgrad
        )
    }
    
    /// Get optimizer configuration as a dictionary.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("learning_rate", self.learning_rate)?;
        dict.set_item("beta1", self.beta1)?;
        dict.set_item("beta2", self.beta2)?;
        dict.set_item("epsilon", self.epsilon)?;
        dict.set_item("amsgrad", self.amsgrad)?;
        Ok(dict.into())
    }
    
    /// Run optimization on Sphere manifold.
    pub fn optimize_sphere(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        sphere: PyRef<'_, PySphere>,
        initial_point: PyReadonlyArray1<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        // Convert initial point
        let x0 = numpy_to_dvector(initial_point)?;
        
        // Create stopping criterion
        let mut criterion = StoppingCriterion::new()
            .with_max_iterations(max_iterations);
        
        if let Some(tol) = gradient_tolerance {
            criterion = criterion.with_gradient_tolerance(tol);
        }
        
        // Create Adam configuration
        let adam_config = AdamConfig {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            use_amsgrad: self.amsgrad,
            weight_decay: None,  // No weight decay for now
            gradient_clip: None,  // No gradient clipping for now
        };
        
        // Create optimizer
        let mut adam = Adam::new(adam_config);
        
        // Get the inner manifold
        let manifold = &sphere.inner;
        
        // Create adapter for the cost function
        let cost_fn_sphere = PyCostFunctionSphere::new(&*cost_function);
        
        // Run optimization
        let result = adam.optimize(&cost_fn_sphere, manifold, &x0, &criterion)
            .map_err(to_py_err)?;
        
        // Convert result
        PyOptimizationResult::from_rust_result(py, result, |point| {
            Ok(dvector_to_numpy(py, point)?.into())
        }).map(|r| r.into_py(py))
    }
    
    /// Run optimization on Stiefel manifold.
    pub fn optimize_stiefel(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        stiefel: PyRef<'_, PyStiefel>,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        // Convert initial point
        let x0 = numpy_to_dmatrix(initial_point)?;
        
        // Create stopping criterion
        let mut criterion = StoppingCriterion::new()
            .with_max_iterations(max_iterations);
        
        if let Some(tol) = gradient_tolerance {
            criterion = criterion.with_gradient_tolerance(tol);
        }
        
        // Create Adam configuration
        let adam_config = AdamConfig {
            learning_rate: self.learning_rate,
            beta1: self.beta1,
            beta2: self.beta2,
            epsilon: self.epsilon,
            use_amsgrad: self.amsgrad,
            weight_decay: None,  // No weight decay for now
            gradient_clip: None,  // No gradient clipping for now
        };
        
        // Create optimizer
        let mut adam = Adam::new(adam_config);
        
        // Get the inner manifold
        let manifold = &stiefel.inner;
        
        // Create adapter for the cost function
        let cost_fn_stiefel = PyCostFunctionStiefel::new(&*cost_function);
        
        // Run optimization
        let result = adam.optimize(&cost_fn_stiefel, manifold, &x0, &criterion)
            .map_err(to_py_err)?;
        
        // Convert result
        PyOptimizationResult::from_rust_result(py, result, |point| {
            Ok(dmatrix_to_numpy(py, point)?.into())
        }).map(|r| r.into_py(py))
    }
}