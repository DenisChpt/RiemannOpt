//! Python wrapper for the Conjugate Gradient optimizer.
//!
//! Conjugate Gradient methods are among the most efficient for large-scale
//! optimization, requiring only first-order information.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use riemannopt_optim::{ConjugateGradient, CGConfig, ConjugateGradientMethod};
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
pub struct PyConjugateGradient {
    method: String,
    reset_every: Option<usize>,
    max_line_search_iterations: usize,
    c1: f64,
    c2: f64,
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
        
        // Parse CG method
        let cg_method = match self.method.as_str() {
            "FletcherReeves" => ConjugateGradientMethod::FletcherReeves,
            "PolakRibiere" => ConjugateGradientMethod::PolakRibiere,
            "HestenesStiefel" => ConjugateGradientMethod::HestenesStiefel,
            "DaiYuan" => ConjugateGradientMethod::DaiYuan,
            _ => unreachable!("Invalid method should have been caught in new()"),
        };
        
        // Create line search parameters
        let line_search_params = LineSearchParams {
            initial_step_size: 1.0,
            max_step_size: 100.0,  // Default max step size
            min_step_size: 1e-10,  // Default min step size
            max_iterations: self.max_line_search_iterations,
            c1: self.c1,
            c2: self.c2,
            rho: 0.5,  // Default backtracking factor
        };
        
        // Create CG configuration
        let cg_config = CGConfig {
            method: cg_method,
            restart_period: self.reset_every.unwrap_or(sphere.dimension),
            use_pr_plus: false,
            min_beta: None,
            max_beta: None,
            line_search_params,
        };
        
        // Create optimizer
        let mut cg = ConjugateGradient::new(cg_config);
        
        // Get the inner manifold
        let manifold = &sphere.inner;
        
        // Create adapter for the cost function
        let cost_fn_sphere = PyCostFunctionSphere::new(&*cost_function);
        
        // Run optimization
        let result = cg.optimize(&cost_fn_sphere, manifold, &x0, &criterion)
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
        
        // Parse CG method
        let cg_method = match self.method.as_str() {
            "FletcherReeves" => ConjugateGradientMethod::FletcherReeves,
            "PolakRibiere" => ConjugateGradientMethod::PolakRibiere,
            "HestenesStiefel" => ConjugateGradientMethod::HestenesStiefel,
            "DaiYuan" => ConjugateGradientMethod::DaiYuan,
            _ => unreachable!("Invalid method should have been caught in new()"),
        };
        
        // Create line search parameters
        let line_search_params = LineSearchParams {
            initial_step_size: 1.0,
            max_step_size: 100.0,  // Default max step size
            min_step_size: 1e-10,  // Default min step size
            max_iterations: self.max_line_search_iterations,
            c1: self.c1,
            c2: self.c2,
            rho: 0.5,  // Default backtracking factor
        };
        
        // Create CG configuration
        let cg_config = CGConfig {
            method: cg_method,
            restart_period: self.reset_every.unwrap_or(stiefel.n * stiefel.p),
            use_pr_plus: false,
            min_beta: None,
            max_beta: None,
            line_search_params,
        };
        
        // Create optimizer
        let mut cg = ConjugateGradient::new(cg_config);
        
        // Get the inner manifold
        let manifold = &stiefel.inner;
        
        // Create adapter for the cost function
        let cost_fn_stiefel = PyCostFunctionStiefel::new(&*cost_function);
        
        // Run optimization
        let result = cg.optimize(&cost_fn_stiefel, manifold, &x0, &criterion)
            .map_err(to_py_err)?;
        
        // Convert result
        PyOptimizationResult::from_rust_result(py, result, |point| {
            Ok(dmatrix_to_numpy(py, point)?.into())
        }).map(|r| r.into_py(py))
    }
}