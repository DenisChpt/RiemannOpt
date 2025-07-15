//! Python wrapper for the Newton optimizer.
//!
//! Newton's method uses second-order information (Hessian) to achieve
//! quadratic convergence near the optimum.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use riemannopt_optim::{Newton, NewtonConfig};
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

/// Newton optimizer for Riemannian manifolds.
///
/// Newton's method computes search directions by solving the Newton equation
/// Hess[f](η) = -grad f, where Hess[f] is the Riemannian Hessian. This
/// provides quadratic convergence but requires Hessian computations.
///
/// Parameters
/// ----------
/// max_cg_iterations : int, default=None
///     Maximum iterations for CG solver. If None, uses dimension.
/// cg_tolerance : float, default=1e-6
///     Tolerance for CG solver.
/// use_line_search : bool, default=True
///     Whether to use line search for step size.
/// alpha : float, default=0.5
///     Backtracking line search parameter.
/// beta : float, default=0.5
///     Backtracking line search parameter.
/// max_line_search_iterations : int, default=20
///     Maximum iterations for line search.
/// force_positive_definite : bool, default=True
///     Whether to modify Hessian to ensure positive definiteness.
///
/// Notes
/// -----
/// Newton's method requires the cost function to provide Hessian-vector
/// products. If not available, consider using L-BFGS or Trust Region methods.
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> sphere = ro.manifolds.Sphere(50)
/// >>> optimizer = ro.optimizers.Newton(max_cg_iterations=10)
/// >>> 
/// >>> # Need a cost function with Hessian
/// >>> def cost(x):
/// ...     return 0.5 * x.T @ A @ x  # Quadratic
/// >>> 
/// >>> def hess_vec_prod(x, v):
/// ...     return A @ v  # For quadratic, Hessian is constant
/// >>> 
/// >>> x0 = sphere.random_point()
/// >>> result = optimizer.optimize(
/// ...     cost_function=cost,
/// ...     manifold=sphere,
/// ...     initial_point=x0,
/// ...     max_iterations=20
/// ... )
#[pyclass(name = "Newton", module = "riemannopt.optimizers")]
#[derive(Clone)]
pub struct PyNewton {
    pub max_cg_iterations: Option<usize>,
    pub cg_tolerance: f64,
    pub use_line_search: bool,
    pub alpha: f64,
    pub beta: f64,
    pub max_line_search_iterations: usize,
    pub force_positive_definite: bool,
}

#[pymethods]
impl PyNewton {
    #[new]
    #[pyo3(signature = (
        max_cg_iterations=None,
        cg_tolerance=1e-6,
        use_line_search=true,
        alpha=0.5,
        beta=0.5,
        max_line_search_iterations=20,
        force_positive_definite=true
    ))]
    fn new(
        max_cg_iterations: Option<usize>,
        cg_tolerance: f64,
        use_line_search: bool,
        alpha: f64,
        beta: f64,
        max_line_search_iterations: usize,
        force_positive_definite: bool,
    ) -> PyResult<Self> {
        // Validate parameters
        if cg_tolerance <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cg_tolerance must be positive"
            ));
        }
        if alpha <= 0.0 || alpha >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "alpha must be in (0, 1)"
            ));
        }
        if beta <= 0.0 || beta >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beta must be in (0, 1)"
            ));
        }
        if max_line_search_iterations == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_line_search_iterations must be positive"
            ));
        }
        
        Ok(PyNewton {
            max_cg_iterations,
            cg_tolerance,
            use_line_search,
            alpha,
            beta,
            max_line_search_iterations,
            force_positive_definite,
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "Newton(max_cg_iterations={:?}, cg_tolerance={}, use_line_search={}, alpha={}, beta={}, max_line_search_iterations={}, force_positive_definite={})",
            self.max_cg_iterations, self.cg_tolerance, self.use_line_search,
            self.alpha, self.beta, self.max_line_search_iterations, self.force_positive_definite
        )
    }
    
    /// Get optimizer configuration as a dictionary.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("max_cg_iterations", self.max_cg_iterations)?;
        dict.set_item("cg_tolerance", self.cg_tolerance)?;
        dict.set_item("use_line_search", self.use_line_search)?;
        dict.set_item("alpha", self.alpha)?;
        dict.set_item("beta", self.beta)?;
        dict.set_item("max_line_search_iterations", self.max_line_search_iterations)?;
        dict.set_item("force_positive_definite", self.force_positive_definite)?;
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
impl PyOptimizerBase for PyNewton {
    fn name(&self) -> &'static str {
        "Newton"
    }
    
    fn validate_config(&self) -> PyResult<()> {
        if self.cg_tolerance <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "cg_tolerance must be positive"
            ));
        }
        if self.alpha <= 0.0 || self.alpha >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "alpha must be in (0, 1)"
            ));
        }
        if self.beta <= 0.0 || self.beta >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "beta must be in (0, 1)"
            ));
        }
        if self.max_line_search_iterations == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_line_search_iterations must be positive"
            ));
        }
        Ok(())
    }
}

// Implement generic optimizer interface
impl_optimizer_generic_default!(PyNewton, Newton<f64>, NewtonConfig<f64>, |opt: &PyNewton| {
    let line_search_params = if opt.use_line_search {
        LineSearchParams {
            initial_step_size: 1.0,
            max_step_size: 100.0,
            min_step_size: 1e-10,
            max_iterations: opt.max_line_search_iterations,
            c1: 1e-4,
            c2: 0.9,
            rho: 0.5,
        }
    } else {
        LineSearchParams {
            initial_step_size: 1.0,
            max_step_size: 100.0,
            min_step_size: 1e-10,
            max_iterations: 1,
            c1: 0.0,
            c2: 0.0,
            rho: 0.5,
        }
    };
    
    NewtonConfig {
        line_search_params,
        hessian_regularization: if opt.force_positive_definite { 1e-8 } else { 0.0 },
        use_gauss_newton: false,
        max_cg_iterations: opt.max_cg_iterations.unwrap_or(100), // Default CG iterations
        cg_tolerance: opt.cg_tolerance,
    }
});

