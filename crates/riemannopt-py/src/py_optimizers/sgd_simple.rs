//! Simplified Python wrapper for SGD optimizer.
//!
//! This is a minimal implementation to get started.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use riemannopt_optim::{SGD, SGDConfig, MomentumMethod};
use riemannopt_core::{
    optimizer::{Optimizer, StoppingCriterion},
    step_size::StepSizeSchedule,
};
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
    impl_optimizer_generic_default,
};
use super::base::{PyOptimizationResult, PyOptimizerBase};
use super::generic::PyOptimizerGeneric;

/// Simple SGD optimizer for testing.
#[pyclass(name = "SGD", module = "riemannopt.optimizers")]
#[derive(Clone)]
pub struct PySGD {
    pub learning_rate: f64,
    pub momentum: f64,
}

#[pymethods]
impl PySGD {
    #[new]
    #[pyo3(signature = (learning_rate=0.01, momentum=0.0))]
    pub fn new(learning_rate: f64, momentum: f64) -> PyResult<Self> {
        if learning_rate <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "learning_rate must be positive"
            ));
        }
        if momentum < 0.0 || momentum >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "momentum must be in [0, 1)"
            ));
        }
        Ok(PySGD { learning_rate, momentum })
    }
    
    fn __repr__(&self) -> String {
        format!("SGD(learning_rate={}, momentum={})", self.learning_rate, self.momentum)
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
        // Dispatch based on manifold type
        if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>(py) {
            if let Ok(initial_array) = initial_point.downcast_bound::<PyArray1<f64>>(py) {
                self.optimize_sphere_impl(
                    py, &*cost_function, &*sphere, initial_array.readonly(), 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "initial_point must be a 1D numpy array for Sphere manifold"
                ))
            }
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>(py) {
            if let Ok(initial_array) = initial_point.downcast_bound::<PyArray2<f64>>(py) {
                self.optimize_stiefel_impl(
                    py, &*cost_function, &*stiefel, initial_array.readonly(), 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "initial_point must be a 2D numpy array for Stiefel manifold"
                ))
            }
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>(py) {
            if let Ok(initial_array) = initial_point.downcast_bound::<PyArray2<f64>>(py) {
                self.optimize_grassmann_impl(
                    py, &*cost_function, &*grassmann, initial_array.readonly(), 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "initial_point must be a 2D numpy array for Grassmann manifold"
                ))
            }
        } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>(py) {
            if let Ok(initial_array) = initial_point.downcast_bound::<PyArray2<f64>>(py) {
                self.optimize_spd_impl(
                    py, &*cost_function, &*spd, initial_array.readonly(), 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "initial_point must be a 2D numpy array for SPD manifold"
                ))
            }
        } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>(py) {
            if let Ok(initial_array) = initial_point.downcast_bound::<PyArray1<f64>>(py) {
                self.optimize_hyperbolic_impl(
                    py, &*cost_function, &*hyperbolic, initial_array.readonly(), 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "initial_point must be a 1D numpy array for Hyperbolic manifold"
                ))
            }
        } else if let Ok(oblique) = manifold.extract::<PyRef<PyOblique>>(py) {
            if let Ok(initial_array) = initial_point.downcast_bound::<PyArray2<f64>>(py) {
                self.optimize_oblique_impl(
                    py, &*cost_function, &*oblique, initial_array.readonly(), 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "initial_point must be a 2D numpy array for Oblique manifold"
                ))
            }
        } else if let Ok(psd_cone) = manifold.extract::<PyRef<PyPSDCone>>(py) {
            if let Ok(initial_array) = initial_point.downcast_bound::<PyArray2<f64>>(py) {
                self.optimize_psd_cone_impl(
                    py, &*cost_function, &*psd_cone, initial_array.readonly(), 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "initial_point must be a 2D numpy array for PSDCone manifold"
                ))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Unsupported manifold type. Supported manifolds: Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, PSDCone"
            ))
        }
    }
}

// Implement the base trait
impl PyOptimizerBase for PySGD {
    fn name(&self) -> &'static str {
        "SGD"
    }
    
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
        Ok(())
    }
}

// Implement generic optimizer interface
impl_optimizer_generic_default!(PySGD, SGD<f64>, SGDConfig<f64>, |opt: &PySGD| {
    let momentum_method = if opt.momentum > 0.0 {
        MomentumMethod::Classical { coefficient: opt.momentum }
    } else {
        MomentumMethod::None
    };
    
    SGDConfig {
        step_size: StepSizeSchedule::Constant(opt.learning_rate),
        momentum: momentum_method,
        gradient_clip: None,
        line_search: None,
    }
});