//! Python wrapper for the L-BFGS optimizer.
//!
//! L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton
//! method that approximates the Hessian using a limited history of gradients.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use riemannopt_optim::{LBFGS, LBFGSConfig};
use riemannopt_core::{
    optimization::{
        optimizer::{Optimizer, StoppingCriterion},
        line_search::LineSearchParams,
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

/// L-BFGS optimizer for Riemannian manifolds.
///
/// L-BFGS is a quasi-Newton method that builds an approximation of the
/// inverse Hessian using a limited history of gradient differences.
/// This makes it suitable for large-scale problems where storing the
/// full Hessian is impractical.
///
/// Parameters
/// ----------
/// memory_size : int, default=10
///     Number of vector pairs to store for Hessian approximation.
/// max_line_search_iterations : int, default=20
///     Maximum iterations for line search.
/// c1 : float, default=1e-4
///     Wolfe condition parameter for sufficient decrease.
/// c2 : float, default=0.9
///     Wolfe condition parameter for curvature.
/// initial_step_size : float, default=1.0
///     Initial step size for line search.
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> stiefel = ro.manifolds.Stiefel(10, 3)
/// >>> optimizer = ro.optimizers.LBFGS(memory_size=20)
/// >>> 
/// >>> # Define cost function
/// >>> def cost(X):
/// ...     return -np.trace(X.T @ A @ X)  # Maximize trace
/// >>> 
/// >>> X0 = stiefel.random_point()
/// >>> result = optimizer.optimize(
/// ...     cost_function=cost,
/// ...     manifold=stiefel,
/// ...     initial_point=X0,
/// ...     max_iterations=100
/// ... )
#[pyclass(name = "LBFGS", module = "riemannopt.optimizers")]
pub struct PyLBFGS {
    memory_size: usize,
    max_line_search_iterations: usize,
    c1: f64,
    c2: f64,
    initial_step_size: f64,
}

#[pymethods]
impl PyLBFGS {
    #[new]
    #[pyo3(signature = (memory_size=10, max_line_search_iterations=20, c1=1e-4, c2=0.9, initial_step_size=1.0))]
    fn new(
        memory_size: usize,
        max_line_search_iterations: usize,
        c1: f64,
        c2: f64,
        initial_step_size: f64,
    ) -> PyResult<Self> {
        // Validate parameters
        if memory_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "memory_size must be positive"
            ));
        }
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
        if initial_step_size <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_step_size must be positive"
            ));
        }
        
        Ok(PyLBFGS {
            memory_size,
            max_line_search_iterations,
            c1,
            c2,
            initial_step_size,
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "LBFGS(memory_size={}, max_line_search_iterations={}, c1={}, c2={}, initial_step_size={})",
            self.memory_size, self.max_line_search_iterations, self.c1, self.c2, self.initial_step_size
        )
    }
    
    /// Get optimizer configuration as a dictionary.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("memory_size", self.memory_size)?;
        dict.set_item("max_line_search_iterations", self.max_line_search_iterations)?;
        dict.set_item("c1", self.c1)?;
        dict.set_item("c2", self.c2)?;
        dict.set_item("initial_step_size", self.initial_step_size)?;
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
        
        // Create line search parameters
        let line_search_params = LineSearchParams {
            initial_step_size: self.initial_step_size,
            max_step_size: 100.0,  // Default max step size
            min_step_size: 1e-10,  // Default min step size
            max_iterations: self.max_line_search_iterations,
            c1: self.c1,
            c2: self.c2,
            rho: 0.5,  // Default backtracking factor
        };
        
        // Create LBFGS configuration
        let lbfgs_config = LBFGSConfig {
            memory_size: self.memory_size,
            initial_step_size: self.initial_step_size,
            use_cautious_updates: false,
            line_search_params,
        };
        
        // Create optimizer
        let mut lbfgs = LBFGS::new(lbfgs_config);
        
        // Get the inner manifold
        let manifold = &sphere.inner;
        
        // Create adapter for the cost function
        let cost_fn_sphere = PyCostFunctionSphere::new(&*cost_function);
        
        // Run optimization
        let result = lbfgs.optimize(&cost_fn_sphere, manifold, &x0, &criterion)
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
        
        // Create line search parameters
        let line_search_params = LineSearchParams {
            initial_step_size: self.initial_step_size,
            max_step_size: 100.0,  // Default max step size
            min_step_size: 1e-10,  // Default min step size
            max_iterations: self.max_line_search_iterations,
            c1: self.c1,
            c2: self.c2,
            rho: 0.5,  // Default backtracking factor
        };
        
        // Create LBFGS configuration
        let lbfgs_config = LBFGSConfig {
            memory_size: self.memory_size,
            initial_step_size: self.initial_step_size,
            use_cautious_updates: false,
            line_search_params,
        };
        
        // Create optimizer
        let mut lbfgs = LBFGS::new(lbfgs_config);
        
        // Get the inner manifold
        let manifold = &stiefel.inner;
        
        // Create adapter for the cost function
        let cost_fn_stiefel = PyCostFunctionStiefel::new(&*cost_function);
        
        // Run optimization
        let result = lbfgs.optimize(&cost_fn_stiefel, manifold, &x0, &criterion)
            .map_err(to_py_err)?;
        
        // Convert result
        PyOptimizationResult::from_rust_result(py, result, |point| {
            Ok(dmatrix_to_numpy(py, point)?.into())
        }).map(|r| r.into_py(py))
    }
}