//! Simplified Python wrapper for SGD optimizer.
//!
//! This is a minimal implementation to get started.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::{DVector, DMatrix};
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
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
    impl_optimizer_methods, impl_optimizer_generic_default,
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