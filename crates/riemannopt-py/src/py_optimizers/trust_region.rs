//! Python wrapper for the Trust Region optimizer.
//!
//! Trust region methods are robust second-order optimization algorithms that
//! maintain a region where a quadratic model is trusted.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods};
use riemannopt_optim::{TrustRegion, TrustRegionConfig};
use riemannopt_core::{
    optimizer::{Optimizer, StoppingCriterion},
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

/// Trust Region optimizer for Riemannian manifolds.
///
/// Trust region methods solve a sequence of subproblems where the objective
/// is approximated by a quadratic model within a trust region. The size of
/// the trust region is adaptively adjusted based on the agreement between
/// the model and the actual function.
///
/// Parameters
/// ----------
/// initial_radius : float, default=1.0
///     Initial trust region radius.
/// max_radius : float, default=10.0
///     Maximum allowed trust region radius.
/// eta : float, default=0.1
///     Threshold for accepting a step (ratio of actual to predicted decrease).
/// radius_decrease_factor : float, default=0.25
///     Factor for decreasing radius when step is rejected.
/// radius_increase_factor : float, default=2.0
///     Factor for increasing radius when step is very successful.
/// subproblem_solver : str, default="CG"
///     Method for solving trust region subproblem: "CG" or "Lanczos".
/// max_subproblem_iterations : int, default=None
///     Maximum iterations for subproblem solver. If None, uses dimension.
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> grassmann = ro.manifolds.Grassmann(10, 3)
/// >>> optimizer = ro.optimizers.TrustRegion(initial_radius=0.5)
/// >>> 
/// >>> # Define a smooth cost function
/// >>> def cost(X):
/// ...     return np.linalg.norm(X - target)**2
/// >>> 
/// >>> X0 = grassmann.random_point()
/// >>> result = optimizer.optimize(
/// ...     cost_function=cost,
/// ...     manifold=grassmann,
/// ...     initial_point=X0,
/// ...     max_iterations=100
/// ... )
#[pyclass(name = "TrustRegion", module = "riemannopt.optimizers")]
#[derive(Clone)]
pub struct PyTrustRegion {
    pub initial_radius: f64,
    pub max_radius: f64,
    pub eta: f64,
    pub radius_decrease_factor: f64,
    pub radius_increase_factor: f64,
    pub subproblem_solver: String,
    pub max_subproblem_iterations: Option<usize>,
}

#[pymethods]
impl PyTrustRegion {
    #[new]
    #[pyo3(signature = (
        initial_radius=1.0,
        max_radius=10.0,
        eta=0.1,
        radius_decrease_factor=0.25,
        radius_increase_factor=2.0,
        subproblem_solver="CG",
        max_subproblem_iterations=None
    ))]
    fn new(
        initial_radius: f64,
        max_radius: f64,
        eta: f64,
        radius_decrease_factor: f64,
        radius_increase_factor: f64,
        subproblem_solver: &str,
        max_subproblem_iterations: Option<usize>,
    ) -> PyResult<Self> {
        // Validate parameters
        if initial_radius <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_radius must be positive"
            ));
        }
        if max_radius <= initial_radius {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_radius must be greater than initial_radius"
            ));
        }
        if eta <= 0.0 || eta >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eta must be in (0, 1)"
            ));
        }
        if radius_decrease_factor <= 0.0 || radius_decrease_factor >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radius_decrease_factor must be in (0, 1)"
            ));
        }
        if radius_increase_factor <= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radius_increase_factor must be greater than 1"
            ));
        }
        
        let valid_solvers = ["CG", "Lanczos"];
        if !valid_solvers.contains(&subproblem_solver) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid subproblem_solver '{}'. Choose from: {:?}", subproblem_solver, valid_solvers)
            ));
        }
        
        Ok(PyTrustRegion {
            initial_radius,
            max_radius,
            eta,
            radius_decrease_factor,
            radius_increase_factor,
            subproblem_solver: subproblem_solver.to_string(),
            max_subproblem_iterations,
        })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "TrustRegion(initial_radius={}, max_radius={}, eta={}, radius_decrease_factor={}, radius_increase_factor={}, subproblem_solver='{}', max_subproblem_iterations={:?})",
            self.initial_radius, self.max_radius, self.eta, 
            self.radius_decrease_factor, self.radius_increase_factor,
            self.subproblem_solver, self.max_subproblem_iterations
        )
    }
    
    /// Get optimizer configuration as a dictionary.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("initial_radius", self.initial_radius)?;
        dict.set_item("max_radius", self.max_radius)?;
        dict.set_item("eta", self.eta)?;
        dict.set_item("radius_decrease_factor", self.radius_decrease_factor)?;
        dict.set_item("radius_increase_factor", self.radius_increase_factor)?;
        dict.set_item("subproblem_solver", &self.subproblem_solver)?;
        dict.set_item("max_subproblem_iterations", self.max_subproblem_iterations)?;
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
impl PyOptimizerBase for PyTrustRegion {
    fn name(&self) -> &'static str {
        "TrustRegion"
    }
    
    fn validate_config(&self) -> PyResult<()> {
        if self.initial_radius <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "initial_radius must be positive"
            ));
        }
        if self.max_radius <= self.initial_radius {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "max_radius must be greater than initial_radius"
            ));
        }
        if self.eta <= 0.0 || self.eta >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "eta must be in (0, 1)"
            ));
        }
        if self.radius_decrease_factor <= 0.0 || self.radius_decrease_factor >= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radius_decrease_factor must be in (0, 1)"
            ));
        }
        if self.radius_increase_factor <= 1.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "radius_increase_factor must be greater than 1"
            ));
        }
        let valid_solvers = ["CG", "Lanczos"];
        if !valid_solvers.contains(&self.subproblem_solver.as_str()) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("Invalid subproblem_solver '{}'. Choose from: {:?}", self.subproblem_solver, valid_solvers)
            ));
        }
        Ok(())
    }
}

// Implement generic optimizer interface
impl_optimizer_generic_default!(PyTrustRegion, TrustRegion<f64>, TrustRegionConfig<f64>, |opt: &PyTrustRegion| {
    TrustRegionConfig {
        initial_radius: opt.initial_radius,
        max_radius: opt.max_radius,
        min_radius: 1e-10,
        acceptance_ratio: opt.eta,
        increase_threshold: 0.75,
        decrease_threshold: 0.25,
        increase_factor: opt.radius_increase_factor,
        decrease_factor: opt.radius_decrease_factor,
        max_cg_iterations: opt.max_subproblem_iterations,
        cg_tolerance: 1e-6,
        use_exact_hessian: true,
    }
});

