//! Python wrapper for the Conjugate Gradient optimizer.
//!
//! Conjugate Gradient methods are among the most efficient for large-scale
//! optimization, requiring only first-order information.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use riemannopt_optim::{ConjugateGradient, CGConfig, ConjugateGradientMethod};
use riemannopt_core::{
    optimizer::{Optimizer, StoppingCriterion},
    line_search::LineSearchParams,
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

/// Conjugate Gradient optimizer for Riemannian manifolds.
///
/// The Conjugate Gradient method generates search directions that are
/// conjugate with respect to the Hessian, leading to faster convergence
/// than steepest descent.
///
/// Parameters
/// ----------
/// method : str, default="FletcherReeves"
///     The CG update formula. Options: "FletcherReeves", "PolakRibiere",
///     "HestenesStiefel", "DaiYuan".
/// reset_every : int, default=None
///     Reset to steepest descent every N iterations. If None, uses n
///     (dimension) as the reset frequency.
/// max_line_search_iterations : int, default=20
///     Maximum iterations for line search.
/// c1 : float, default=1e-4
///     Wolfe condition parameter for sufficient decrease.
/// c2 : float, default=0.1
///     Wolfe condition parameter for curvature.
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> sphere = ro.manifolds.Sphere(100)
/// >>> optimizer = ro.optimizers.ConjugateGradient(method="PolakRibiere")
/// >>> 
/// >>> # Define quadratic cost
/// >>> Q = np.random.randn(100, 100)
/// >>> Q = Q.T @ Q  # Positive definite
/// >>> 
/// >>> def cost(x):
/// ...     return x.T @ Q @ x
/// >>> 
/// >>> x0 = sphere.random_point()
/// >>> result = optimizer.optimize(
/// ...     cost_function=cost,
/// ...     manifold=sphere,
/// ...     initial_point=x0,
/// ...     max_iterations=50
/// ... )
#[pyclass(name = "ConjugateGradient", module = "riemannopt.optimizers")]
#[derive(Clone)]
pub struct PyConjugateGradient {
    pub method: String,
    pub reset_every: Option<usize>,
    pub max_line_search_iterations: usize,
    pub c1: f64,
    pub c2: f64,
}

#[pymethods]
impl PyConjugateGradient {
    #[new]
    #[pyo3(signature = (method="FletcherReeves", reset_every=None, max_line_search_iterations=20, c1=1e-4, c2=0.1))]
    fn new(
        method: &str,
        reset_every: Option<usize>,
        max_line_search_iterations: usize,
        c1: f64,
        c2: f64,
    ) -> PyResult<Self> {
        // Validate method
        let valid_methods = ["FletcherReeves", "PolakRibiere", "HestenesStiefel", "DaiYuan"];
        if !valid_methods.contains(&method) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid method '{}'. Choose from: {:?}", method, valid_methods)
            ));
        }
        
        // Validate parameters
        if max_line_search_iterations == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_line_search_iterations must be positive"
            ));
        }
        if c1 <= 0.0 || c1 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "c1 must be in (0, 1)"
            ));
        }
        if c2 <= c1 || c2 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "c2 must be in (c1, 1)"
            ));
        }
        
        Ok(PyConjugateGradient {
            method: method.to_string(),
            reset_every,
            max_line_search_iterations,
            c1,
            c2,
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "ConjugateGradient(method='{}', reset_every={:?}, max_line_search_iterations={}, c1={}, c2={})",
            self.method, self.reset_every, self.max_line_search_iterations, self.c1, self.c2
        )
    }
    
    /// Get optimizer configuration as a dictionary.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("method", &self.method)?;
        dict.set_item("reset_every", self.reset_every)?;
        dict.set_item("max_line_search_iterations", self.max_line_search_iterations)?;
        dict.set_item("c1", self.c1)?;
        dict.set_item("c2", self.c2)?;
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
impl PyOptimizerBase for PyConjugateGradient {
    fn name(&self) -> &'static str {
        "ConjugateGradient"
    }
    
    fn validate_config(&self) -> PyResult<()> {
        let valid_methods = ["FletcherReeves", "PolakRibiere", "HestenesStiefel", "DaiYuan"];
        if !valid_methods.contains(&self.method.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid method '{}'. Choose from: {:?}", self.method, valid_methods)
            ));
        }
        if self.max_line_search_iterations == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_line_search_iterations must be positive"
            ));
        }
        if self.c1 <= 0.0 || self.c1 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "c1 must be in (0, 1)"
            ));
        }
        if self.c2 <= self.c1 || self.c2 >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "c2 must be in (c1, 1)"
            ));
        }
        Ok(())
    }
}

// Implement generic optimizer interface
impl_optimizer_generic_default!(PyConjugateGradient, ConjugateGradient<f64>, CGConfig<f64>, |opt: &PyConjugateGradient| {
    let cg_method = match opt.method.as_str() {
        "FletcherReeves" => ConjugateGradientMethod::FletcherReeves,
        "PolakRibiere" => ConjugateGradientMethod::PolakRibiere,
        "HestenesStiefel" => ConjugateGradientMethod::HestenesStiefel,
        "DaiYuan" => ConjugateGradientMethod::DaiYuan,
        _ => ConjugateGradientMethod::FletcherReeves,
    };
    
    let line_search_params = LineSearchParams {
        initial_step_size: 1.0,
        max_step_size: 100.0,
        min_step_size: 1e-10,
        max_iterations: opt.max_line_search_iterations,
        c1: opt.c1,
        c2: opt.c2,
        rho: 0.5,
    };
    
    CGConfig {
        method: cg_method,
        restart_period: opt.reset_every.unwrap_or(100), // Default restart period
        use_pr_plus: false,
        min_beta: None,
        max_beta: None,
        line_search_params,
    }
});

