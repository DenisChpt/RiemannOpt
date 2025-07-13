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
    impl_optimizer_methods, impl_optimizer_generic_default,
};
use super::base::{PyOptimizationResult, PyOptimizerBase};
use super::generic::PyOptimizerGeneric;

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
#[derive(Clone)]
pub struct PyLBFGS {
    pub memory_size: usize,
    pub max_line_search_iterations: usize,
    pub c1: f64,
    pub c2: f64,
    pub initial_step_size: f64,
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
impl PyOptimizerBase for PyLBFGS {
    fn name(&self) -> &'static str {
        "LBFGS"
    }
    
    fn validate_config(&self) -> PyResult<()> {
        if self.memory_size == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "memory_size must be positive"
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
        if self.initial_step_size <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_step_size must be positive"
            ));
        }
        Ok(())
    }
}

// Implement generic optimizer interface
impl_optimizer_generic_default!(PyLBFGS, LBFGS<f64>, LBFGSConfig<f64>, |opt: &PyLBFGS| {
    let line_search_params = LineSearchParams {
        initial_step_size: opt.initial_step_size,
        max_step_size: 100.0,
        min_step_size: 1e-10,
        max_iterations: opt.max_line_search_iterations,
        c1: opt.c1,
        c2: opt.c2,
        rho: 0.5,
    };
    
    LBFGSConfig {
        memory_size: opt.memory_size,
        initial_step_size: opt.initial_step_size,
        use_cautious_updates: false,
        line_search_params,
    }
});

