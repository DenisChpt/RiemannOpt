//! Python bindings for Natural Gradient optimizer.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use riemannopt_core::{manifold::Manifold, optimizer::{StoppingCriterion, Optimizer}};
use riemannopt_optim::{NaturalGradient, NaturalGradientConfig, FisherApproximation};

use crate::{
    array_utils::{numpy_to_dvector, numpy_to_dmatrix, dvector_to_numpy, dmatrix_to_numpy},
    error::to_py_err,
    py_cost::{PyCostFunction, PyCostFunctionSphere, PyCostFunctionStiefel},
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
    impl_optimizer_methods, impl_optimizer_generic_default,
};

use super::base::{PyOptimizationResult, PyOptimizerBase};
use super::generic::PyOptimizerGeneric;

/// Python wrapper for Natural Gradient optimizer.
///
/// The natural gradient method uses the Fisher information matrix to
/// precondition the gradient, leading to faster convergence in many cases.
///
/// Parameters
/// ----------
/// learning_rate : float, default=0.01
///     Learning rate (step size)
/// fisher_damping : float, default=1e-6
///     Damping factor for Fisher matrix regularization
/// fisher_subsample : int or None, default=None
///     Number of samples to use for Fisher estimation. If None, uses all samples.
/// momentum : float, default=0.0
///     Momentum coefficient (0 = no momentum)
#[pyclass(name = "NaturalGradient", module = "riemannopt.optimizers")]
#[derive(Clone)]
pub struct PyNaturalGradient {
    /// Learning rate
    pub learning_rate: f64,
    /// Fisher matrix damping
    pub fisher_damping: f64,
    /// Fisher subsampling size
    pub fisher_subsample: Option<usize>,
    /// Momentum coefficient
    pub momentum: f64,
}

impl PyOptimizerBase for PyNaturalGradient {
    fn name(&self) -> &'static str {
        "NaturalGradient"
    }
    
    fn validate_config(&self) -> PyResult<()> {
        if self.learning_rate <= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "learning_rate must be positive"
            ));
        }
        if self.fisher_damping < 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "fisher_damping must be non-negative"
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

#[pymethods]
impl PyNaturalGradient {
    /// Create a new Natural Gradient optimizer.
    #[new]
    #[pyo3(signature = (learning_rate=0.01, fisher_damping=1e-6, fisher_subsample=None, momentum=0.0))]
    fn new(
        learning_rate: f64,
        fisher_damping: f64,
        fisher_subsample: Option<usize>,
        momentum: f64,
    ) -> PyResult<Self> {
        let opt = PyNaturalGradient {
            learning_rate,
            fisher_damping,
            fisher_subsample,
            momentum,
        };
        opt.validate_config()?;
        Ok(opt)
    }
    
    /// String representation of the optimizer.
    fn __repr__(&self) -> String {
        format!(
            "NaturalGradient(learning_rate={}, fisher_damping={}, fisher_subsample={:?}, momentum={})",
            self.learning_rate, self.fisher_damping, self.fisher_subsample, self.momentum
        )
    }
    
    /// Get optimizer configuration as a dictionary.
    #[getter]
    fn config(&self, py: Python<'_>) -> PyResult<PyObject> {
        let dict = PyDict::new_bound(py);
        dict.set_item("learning_rate", self.learning_rate)?;
        dict.set_item("fisher_damping", self.fisher_damping)?;
        dict.set_item("fisher_subsample", self.fisher_subsample)?;
        dict.set_item("momentum", self.momentum)?;
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

// Implement generic optimizer interface
impl_optimizer_generic_default!(PyNaturalGradient, NaturalGradient<f64>, NaturalGradientConfig<f64>, |opt: &PyNaturalGradient| {
    let fisher_approximation = if let Some(_subsample) = opt.fisher_subsample {
        FisherApproximation::Empirical
    } else {
        FisherApproximation::Full
    };
    
    NaturalGradientConfig {
        learning_rate: opt.learning_rate,
        damping: opt.fisher_damping,
        fisher_approximation,
        fisher_update_freq: 1,
        fisher_num_samples: opt.fisher_subsample.unwrap_or(100),
    }
});

