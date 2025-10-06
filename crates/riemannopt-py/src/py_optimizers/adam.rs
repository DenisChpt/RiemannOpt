//! Python wrapper for the Adam optimizer.
//!
//! Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization
//! algorithm that combines ideas from momentum and RMSprop.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use riemannopt_optim::{Adam, AdamConfig};
use riemannopt_core::optimizer::{Optimizer, StoppingCriterion};

use crate::{
    py_manifolds::{
        sphere::PySphere,
        stiefel::PyStiefel,
        grassmann::PyGrassmann,
        spd::PySPD,
        hyperbolic::PyHyperbolic,
        oblique::PyOblique,
        // fixed_rank::PyFixedRank,  // TODO: Fix FixedRankPoint representation mismatch
        psd_cone::PyPSDCone,
    },
    py_cost::PyCostFunction,
    array_utils::{numpy_to_dvector, numpy_to_dmatrix, dvector_to_numpy, dmatrix_to_numpy},
    error::to_py_err,
    impl_optimizer_generic_default,
};
use super::base::{PyOptimizationResult, PyOptimizerBase};

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
#[derive(Clone)]
pub struct PyAdam {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub amsgrad: bool,
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

    /// Unified optimize method that accepts any supported manifold.
    ///
    /// Parameters
    /// ----------
    /// cost_function : CostFunction
    ///     The cost function to minimize
    /// manifold : Manifold
    ///     The Riemannian manifold to optimize on (Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, or PSDCone)
    /// initial_point : array_like
    ///     Starting point for optimization (shape must match manifold requirements)
    /// max_iterations : int
    ///     Maximum number of iterations
    /// gradient_tolerance : float, optional
    ///     Gradient norm tolerance for convergence
    /// callback : callable, optional
    ///     Function called after each iteration
    /// target_value : float, optional
    ///     Target cost value to stop optimization early
    /// max_time : float, optional
    ///     Maximum optimization time in seconds
    ///
    /// Returns
    /// -------
    /// OptimizationResult
    ///     Object containing the optimized point, final cost, and optimization statistics
    #[pyo3(signature = (cost_function, manifold, initial_point, max_iterations, gradient_tolerance=None, callback=None, target_value=None, max_time=None))]
    pub fn optimize(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        manifold: PyObject,
        initial_point: PyObject,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
        callback: Option<PyObject>,
        target_value: Option<f64>,
        max_time: Option<f64>,
    ) -> PyResult<PyObject> {
        // Use the generic dispatcher to route to the correct manifold implementation
        use super::generic::optimize_dispatcher;
        optimize_dispatcher(
            self,
            py,
            cost_function,
            manifold,
            initial_point,
            max_iterations,
            gradient_tolerance,
            callback,
            target_value,
            max_time,
        )
    }
}

// Implement the base trait
impl PyOptimizerBase for PyAdam {
    fn name(&self) -> &'static str {
        "Adam"
    }
    
    fn validate_config(&self) -> PyResult<()> {
        if self.learning_rate <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "learning_rate must be positive"
            ));
        }
        if self.beta1 < 0.0 || self.beta1 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beta1 must be in [0, 1)"
            ));
        }
        if self.beta2 < 0.0 || self.beta2 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beta2 must be in [0, 1)"
            ));
        }
        if self.epsilon <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "epsilon must be positive"
            ));
        }
        Ok(())
    }
}

// Implement generic optimizer interface
impl_optimizer_generic_default!(PyAdam, Adam<f64>, AdamConfig<f64>, |opt: &PyAdam| {
    AdamConfig {
        learning_rate: opt.learning_rate,
        beta1: opt.beta1,
        beta2: opt.beta2,
        epsilon: opt.epsilon,
        use_amsgrad: opt.amsgrad,
        weight_decay: None,
        gradient_clip: None,
    }
});
