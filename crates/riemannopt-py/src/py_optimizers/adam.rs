//! Python wrapper for the Adam optimizer.
//!
//! Adam (Adaptive Moment Estimation) is an adaptive learning rate optimization
//! algorithm that combines ideas from momentum and RMSprop.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use riemannopt_optim::{Adam, AdamConfig};
use riemannopt_core::optimizer::{Optimizer, StoppingCriterion};
use std::time::Duration;

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
    py_cost::{PyCostFunction, PyCostFunctionSphere, PyCostFunctionStiefel},
    array_utils::{numpy_to_dvector, numpy_to_dmatrix, dvector_to_numpy, dmatrix_to_numpy},
    error::to_py_err,
    impl_optimizer_methods, impl_optimizer_generic_default,
};
use super::base::{PyOptimizationResult, PyOptimizerBase};
use super::generic::PyOptimizerGeneric;

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

    /// Optimize on a Sphere manifold
    #[pyo3(signature = (cost_function, sphere, initial_point, max_iterations, gradient_tolerance=None))]
    pub fn optimize_sphere(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        sphere: PyRef<'_, PySphere>,
        initial_point: PyReadonlyArray1<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        self.optimize_sphere_impl(
            py, &*cost_function, &*sphere, initial_point, 
            max_iterations, gradient_tolerance
        ).map(|r| r.into_py(py))
    }
    
    /// Optimize on a Stiefel manifold
    #[pyo3(signature = (cost_function, stiefel, initial_point, max_iterations, gradient_tolerance=None))]
    pub fn optimize_stiefel(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        stiefel: PyRef<'_, PyStiefel>,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        self.optimize_stiefel_impl(
            py, &*cost_function, &*stiefel, initial_point, 
            max_iterations, gradient_tolerance
        ).map(|r| r.into_py(py))
    }
    
    /// Optimize on a Grassmann manifold
    #[pyo3(signature = (cost_function, grassmann, initial_point, max_iterations, gradient_tolerance=None))]
    pub fn optimize_grassmann(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        grassmann: PyRef<'_, PyGrassmann>,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        self.optimize_grassmann_impl(
            py, &*cost_function, &*grassmann, initial_point, 
            max_iterations, gradient_tolerance
        ).map(|r| r.into_py(py))
    }
    
    /// Optimize on a SPD manifold
    #[pyo3(signature = (cost_function, spd, initial_point, max_iterations, gradient_tolerance=None))]
    pub fn optimize_spd(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        spd: PyRef<'_, PySPD>,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        self.optimize_spd_impl(
            py, &*cost_function, &*spd, initial_point, 
            max_iterations, gradient_tolerance
        ).map(|r| r.into_py(py))
    }
    
    /// Optimize on a Hyperbolic manifold
    #[pyo3(signature = (cost_function, hyperbolic, initial_point, max_iterations, gradient_tolerance=None))]
    pub fn optimize_hyperbolic(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        hyperbolic: PyRef<'_, PyHyperbolic>,
        initial_point: PyReadonlyArray1<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        self.optimize_hyperbolic_impl(
            py, &*cost_function, &*hyperbolic, initial_point, 
            max_iterations, gradient_tolerance
        ).map(|r| r.into_py(py))
    }
    
    /// Optimize on an Oblique manifold
    #[pyo3(signature = (cost_function, oblique, initial_point, max_iterations, gradient_tolerance=None))]
    pub fn optimize_oblique(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        oblique: PyRef<'_, PyOblique>,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        self.optimize_oblique_impl(
            py, &*cost_function, &*oblique, initial_point, 
            max_iterations, gradient_tolerance
        ).map(|r| r.into_py(py))
    }
    
    // /// Optimize on a Fixed-Rank manifold
    // #[pyo3(signature = (cost_function, fixed_rank, initial_point, max_iterations, gradient_tolerance=None))]
    // pub fn optimize_fixed_rank(
    //     &mut self,
    //     py: Python<'_>,
    //     cost_function: PyRef<'_, PyCostFunction>,
    //     fixed_rank: PyRef<'_, PyFixedRank>,
    //     initial_point: PyReadonlyArray2<'_, f64>,
    //     max_iterations: usize,
    //     gradient_tolerance: Option<f64>,
    // ) -> PyResult<PyObject> {
    //     self.optimize_fixed_rank_impl(
    //         py, &*cost_function, &*fixed_rank, initial_point, 
    //         max_iterations, gradient_tolerance
    //     ).map(|r| r.into_py(py))
    // }
    
    /// Optimize on a PSD Cone manifold
    #[pyo3(signature = (cost_function, psd_cone, initial_point, max_iterations, gradient_tolerance=None))]
    pub fn optimize_psd_cone(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        psd_cone: PyRef<'_, PyPSDCone>,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyObject> {
        self.optimize_psd_cone_impl(
            py, &*cost_function, &*psd_cone, initial_point, 
            max_iterations, gradient_tolerance
        ).map(|r| r.into_py(py))
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
