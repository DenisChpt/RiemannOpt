//! Simplified Python wrapper for SGD optimizer.
//!
//! This is a minimal implementation to get started.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use nalgebra::DVector;
use numpy::{PyArray1, PyReadonlyArray1};
use riemannopt_optim::{SGD, SGDConfig, MomentumMethod};
use riemannopt_core::{
    optimization::{
        optimizer::{Optimizer, StoppingCriterion},
        step_size::StepSizeSchedule,
    },
};
use std::time::Duration;

use crate::{
    py_manifolds::sphere::PySphere,
    py_cost::{PyCostFunction, PyCostFunctionSphere},
    array_utils::{numpy_to_dvector, dvector_to_numpy},
    error::to_py_err,
};
use super::base::PyOptimizationResult;

/// Simple SGD optimizer for testing.
#[pyclass(name = "SGD", module = "riemannopt.optimizers")]
pub struct PySGD {
    learning_rate: f64,
    momentum: f64,
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
    
    /// Run optimization on Sphere manifold only (for testing).
    pub fn optimize_sphere(
        &mut self,
        py: Python<'_>,
        cost_function: PyRef<'_, PyCostFunction>,
        sphere: PyRef<'_, PySphere>,
        initial_point: PyReadonlyArray1<'_, f64>,
        max_iterations: usize,
    ) -> PyResult<PyObject> {
        // Convert initial point
        let x0 = numpy_to_dvector(initial_point)?;
        
        // Create stopping criterion
        let criterion = StoppingCriterion::new()
            .with_max_iterations(max_iterations)
            .with_gradient_tolerance(1e-6);
        
        // Create SGD configuration
        let momentum_method = if self.momentum > 0.0 {
            MomentumMethod::Classical { coefficient: self.momentum }
        } else {
            MomentumMethod::None
        };
        
        let sgd_config = SGDConfig {
            step_size: StepSizeSchedule::Constant(self.learning_rate),
            momentum: momentum_method,
            gradient_clip: None,
            line_search: None,
        };
        
        // Create optimizer
        let mut sgd = SGD::new(sgd_config);
        
        // Get the inner manifold
        let manifold = &sphere.inner;
        
        // Create adapter for the cost function
        let cost_fn_sphere = PyCostFunctionSphere::new(&*cost_function);
        
        // Run optimization
        let result = sgd.optimize(&cost_fn_sphere, manifold, &x0, &criterion)
            .map_err(to_py_err)?;
        
        // Convert result to Python dict
        let dict = PyDict::new_bound(py);
        dict.set_item("point", dvector_to_numpy(py, &result.point)?)?;
        dict.set_item("value", result.value)?;
        dict.set_item("gradient_norm", result.gradient_norm)?;
        dict.set_item("converged", result.converged)?;
        dict.set_item("iterations", result.iterations)?;
        
        Ok(dict.into())
    }
}