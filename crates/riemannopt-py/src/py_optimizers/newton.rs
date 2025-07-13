//! Python wrapper for the Newton optimizer.
//!
//! Newton's method uses second-order information (Hessian) to achieve
//! quadratic convergence near the optimum.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use riemannopt_optim::{Newton, NewtonConfig};
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
pub struct PyNewton {
    max_cg_iterations: Option<usize>,
    cg_tolerance: f64,
    use_line_search: bool,
    alpha: f64,
    beta: f64,
    max_line_search_iterations: usize,
    force_positive_definite: bool,
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
        let line_search_params = if self.use_line_search {
            LineSearchParams {
                initial_step_size: 1.0,
                max_step_size: 100.0,  // Default max step size
                min_step_size: 1e-10,  // Default min step size
                max_iterations: self.max_line_search_iterations,
                c1: 1e-4,  // Default Armijo condition
                c2: 0.9,   // Default curvature condition
                rho: 0.5,  // Default backtracking factor
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
        
        // Create Newton configuration
        let newton_config = NewtonConfig {
            line_search_params,
            hessian_regularization: if self.force_positive_definite { 1e-8 } else { 0.0 },
            use_gauss_newton: false,
            max_cg_iterations: self.max_cg_iterations.unwrap_or(sphere.dimension),
            cg_tolerance: self.cg_tolerance,
        };
        
        // Create optimizer
        let mut newton = Newton::new(newton_config);
        
        // Get the inner manifold
        let manifold = &sphere.inner;
        
        // Create adapter for the cost function
        let cost_fn_sphere = PyCostFunctionSphere::new(&*cost_function);
        
        // Run optimization
        let result = newton.optimize(&cost_fn_sphere, manifold, &x0, &criterion)
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
        let line_search_params = if self.use_line_search {
            LineSearchParams {
                initial_step_size: 1.0,
                max_step_size: 100.0,  // Default max step size
                min_step_size: 1e-10,  // Default min step size
                max_iterations: self.max_line_search_iterations,
                c1: 1e-4,  // Default Armijo condition
                c2: 0.9,   // Default curvature condition
                rho: 0.5,  // Default backtracking factor
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
        
        // Create Newton configuration
        let newton_config = NewtonConfig {
            line_search_params,
            hessian_regularization: if self.force_positive_definite { 1e-8 } else { 0.0 },
            use_gauss_newton: false,
            max_cg_iterations: self.max_cg_iterations.unwrap_or(stiefel.n * stiefel.p),
            cg_tolerance: self.cg_tolerance,
        };
        
        // Create optimizer
        let mut newton = Newton::new(newton_config);
        
        // Get the inner manifold
        let manifold = &stiefel.inner;
        
        // Create adapter for the cost function
        let cost_fn_stiefel = PyCostFunctionStiefel::new(&*cost_function);
        
        // Run optimization
        let result = newton.optimize(&cost_fn_stiefel, manifold, &x0, &criterion)
            .map_err(to_py_err)?;
        
        // Convert result
        PyOptimizationResult::from_rust_result(py, result, |point| {
            Ok(dmatrix_to_numpy(py, point)?.into())
        }).map(|r| r.into_py(py))
    }
}