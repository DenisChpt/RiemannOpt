//! Python bindings for Riemannian optimizers.
//!
//! This module provides Python-friendly wrappers around the Rust optimizer implementations.

use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::DVector;
use std::sync::{Arc, Mutex};

use riemannopt_core::{
    manifold::Manifold,
    optimizer_state::ConjugateGradientMethod,
};
use riemannopt_optim::{
    SGDConfig, MomentumMethod,
    AdamConfig,
    LBFGSConfig,
    CGConfig,
};
use riemannopt_core::step_size::StepSizeSchedule;

use crate::manifolds::*;
use crate::cost_function::PyCostFunction;

// Note: PyOptimizer trait removed as it's not needed for the current implementation

/// Riemannian Stochastic Gradient Descent optimizer.
#[pyclass(name = "SGD")]
pub struct PySGD {
    manifold: PyObject,
    config: SGDConfig<f64>,
    max_iterations: usize,
    tolerance: f64,
    _state: Arc<Mutex<Option<DVector<f64>>>>,
}

#[pymethods]
impl PySGD {
    /// Create a new SGD optimizer.
    ///
    /// Args:
    ///     step_size: Learning rate (default: 0.01)
    ///     momentum: Momentum coefficient (default: 0.0)
    ///     max_iterations: Maximum iterations (default: 1000)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    #[new]
    #[pyo3(signature = (step_size=0.01, momentum=0.0, max_iterations=1000, tolerance=1e-6))]
    pub fn new(
        py: Python<'_>,
        step_size: f64,
        momentum: f64,
        max_iterations: i32,  // Changed to i32 to match Python expectations
        tolerance: f64,
    ) -> PyResult<Self> {
        let config = SGDConfig {
            step_size: StepSizeSchedule::Constant(step_size),
            momentum: if momentum > 0.0 {
                MomentumMethod::Classical { coefficient: momentum }
            } else {
                MomentumMethod::None
            },
            gradient_clip: None,
            use_line_search: false,
            max_line_search_iterations: 20,
        };
        
        Ok(Self {
            manifold: PyObject::from(py.None()),  // Will be set later via setter
            config,
            max_iterations: max_iterations as usize,
            tolerance,
            _state: Arc::new(Mutex::new(None)),
        })
    }

    /// Perform one optimization step.
    ///
    /// Args:
    ///     manifold: The manifold to optimize on
    ///     point: Current point on the manifold
    ///     gradient: Gradient at the current point
    ///
    /// Returns:
    ///     new_point: Updated point on the manifold
    pub fn step<'py>(
        &self,
        py: Python<'py>,
        manifold: &Bound<'_, PyAny>,
        point: PyReadonlyArray1<'_, f64>,
        gradient: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let gradient_vec = DVector::from_column_slice(gradient.as_slice()?);
        
        // Apply SGD update with manifold projection
        let learning_rate = match &self.config.step_size {
            StepSizeSchedule::Constant(lr) => *lr,
            _ => 0.01,  // Default fallback
        };
        
        // Compute update in tangent space
        let _update = -learning_rate * &gradient_vec;
        
        // Retract to manifold
        let new_point = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            // First project gradient to tangent space
            let riem_grad = sphere.get_inner().project_tangent(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            sphere.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            // For Stiefel, we need to handle matrix form
            let riem_grad = stiefel.get_inner().project_tangent(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            stiefel.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            // For Grassmann
            let riem_grad = grassmann.get_inner().project_tangent(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            grassmann.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err("Unsupported manifold type"));
        };
        
        Ok(numpy::PyArray1::from_slice_bound(py, new_point.as_slice()))
    }

    /// Optimize a cost function.
    ///
    /// Args:
    ///     cost_function: The cost function to minimize
    ///     initial_point: Starting point on the manifold
    ///
    /// Returns:
    ///     dict: Optimization result with 'point', 'value', 'iterations', 'converged'
    pub fn optimize<'py>(
        &self,
        py: Python<'py>,
        cost_function: &PyCostFunction,
        initial_point: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let mut point = DVector::from_column_slice(initial_point.as_slice()?);
        let mut value = f64::INFINITY;
        let mut iterations = 0;
        let mut converged = false;
        
        for i in 0..self.max_iterations {
            let (new_value, gradient) = cost_function.value_and_gradient(py, &point)?;
            
            // Check convergence
            if gradient.norm() < self.tolerance {
                converged = true;
                value = new_value;
                iterations = i;
                break;
            }
            
            // Update
            let learning_rate = match &self.config.step_size {
                StepSizeSchedule::Constant(lr) => *lr,
                _ => 0.01,  // Default fallback
            };
            let update = -learning_rate * &gradient;
            
            // Retract to manifold
            point = Python::with_gil(|py| -> PyResult<DVector<f64>> {
                let manifold_ref = self.manifold.bind(py);
                
                if let Ok(sphere) = manifold_ref.extract::<PyRef<PySphere>>() {
                    sphere.get_inner().retract(&point, &update)
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                } else if let Ok(stiefel) = manifold_ref.extract::<PyRef<PyStiefel>>() {
                    stiefel.get_inner().retract(&point, &update)
                        .map_err(|e| PyValueError::new_err(e.to_string()))
                } else {
                    Err(PyValueError::new_err("Unsupported manifold type"))
                }
            })?;
            
            value = new_value;
            iterations = i + 1;
        }
        
        // Create result dictionary
        let result = pyo3::types::PyDict::new_bound(py);
        result.set_item("point", numpy::PyArray1::from_slice_bound(py, point.as_slice()))?;
        result.set_item("value", value)?;
        result.set_item("iterations", iterations)?;
        result.set_item("converged", converged)?;
        
        Ok(result)
    }

    /// Get the current configuration.
    #[getter]
    pub fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let config_dict = pyo3::types::PyDict::new_bound(py);
        let lr = match &self.config.step_size {
            StepSizeSchedule::Constant(lr) => *lr,
            _ => 0.0,
        };
        let momentum_coeff = match &self.config.momentum {
            MomentumMethod::Classical { coefficient } => *coefficient,
            MomentumMethod::Nesterov { coefficient } => *coefficient,
            MomentumMethod::None => 0.0,
        };
        config_dict.set_item("learning_rate", lr)?;
        config_dict.set_item("momentum", momentum_coeff)?;
        config_dict.set_item("max_iterations", self.max_iterations)?;
        config_dict.set_item("tolerance", self.tolerance)?;
        Ok(config_dict)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        let lr = match &self.config.step_size {
            StepSizeSchedule::Constant(lr) => *lr,
            _ => 0.0,
        };
        let momentum_coeff = match &self.config.momentum {
            MomentumMethod::Classical { coefficient } => *coefficient,
            MomentumMethod::Nesterov { coefficient } => *coefficient,
            MomentumMethod::None => 0.0,
        };
        format!(
            "SGD(learning_rate={}, momentum={}, max_iterations={}, tolerance={})",
            lr, momentum_coeff, self.max_iterations, self.tolerance
        )
    }
}

/// Riemannian Adam optimizer.
#[pyclass(name = "Adam")]
pub struct PyAdam {
    _manifold: PyObject,
    config: AdamConfig<f64>,
}

#[pymethods]
impl PyAdam {
    /// Create a new Adam optimizer.
    ///
    /// Args:
    ///     learning_rate: Learning rate (default: 0.001)
    ///     beta1: First moment decay (default: 0.9)
    ///     beta2: Second moment decay (default: 0.999)
    ///     epsilon: Numerical stability constant (default: 1e-8)
    #[new]
    #[pyo3(signature = (learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8))]
    pub fn new(
        py: Python<'_>,
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> PyResult<Self> {
        let config = AdamConfig::default()
            .with_learning_rate(learning_rate)
            .with_beta1(beta1)
            .with_beta2(beta2)
            .with_epsilon(epsilon);
        
        Ok(Self { 
            _manifold: PyObject::from(py.None()),
            config 
        })
    }

    /// Perform one optimization step.
    ///
    /// Args:
    ///     manifold: The manifold to optimize on
    ///     point: Current point on the manifold
    ///     gradient: Gradient at the current point
    ///
    /// Returns:
    ///     new_point: Updated point on the manifold
    pub fn step<'py>(
        &self,
        py: Python<'py>,
        manifold: &Bound<'_, PyAny>,
        point: PyReadonlyArray1<'_, f64>,
        gradient: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // For now, implement a simple gradient descent step
        // TODO: Implement proper Adam algorithm with moment estimates
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let gradient_vec = DVector::from_column_slice(gradient.as_slice()?);
        
        let learning_rate = self.config.learning_rate;
        
        // Retract to manifold
        let new_point = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            let riem_grad = sphere.get_inner().project_tangent(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            sphere.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            let riem_grad = stiefel.get_inner().project_tangent(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            stiefel.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err("Unsupported manifold type"));
        };
        
        Ok(numpy::PyArray1::from_slice_bound(py, new_point.as_slice()))
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)"
        )
    }
}

/// Riemannian L-BFGS optimizer.
#[pyclass(name = "LBFGS")]
pub struct PyLBFGS {
    _manifold: PyObject,
    _config: LBFGSConfig<f64>,
}

#[pymethods]
impl PyLBFGS {
    /// Create a new L-BFGS optimizer.
    ///
    /// Args:
    ///     manifold: The manifold to optimize on
    ///     memory_size: Number of past iterations to store (default: 10)
    ///     max_iterations: Maximum iterations (default: 1000)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    #[new]
    #[pyo3(signature = (manifold, memory_size=10, _max_iterations=1000, _tolerance=1e-6))]
    pub fn new(
        manifold: PyObject,
        memory_size: usize,
        _max_iterations: usize,
        _tolerance: f64,
    ) -> PyResult<Self> {
        let config = LBFGSConfig::default()
            .with_memory_size(memory_size);
        
        Ok(Self { _manifold: manifold, _config: config })
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "LBFGS(memory_size=10)"
        )
    }
}

/// Riemannian Conjugate Gradient optimizer.
#[pyclass(name = "ConjugateGradient")]
pub struct PyConjugateGradient {
    _manifold: PyObject,
    config: CGConfig<f64>,
}

#[pymethods]
impl PyConjugateGradient {
    /// Create a new Conjugate Gradient optimizer.
    ///
    /// Args:
    ///     manifold: The manifold to optimize on
    ///     variant: CG variant ('FletcherReeves' or 'PolakRibiere', default: 'FletcherReeves')
    ///     max_iterations: Maximum iterations (default: 1000)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    #[new]
    #[pyo3(signature = (manifold, variant="FletcherReeves", _max_iterations=1000, _tolerance=1e-6))]
    pub fn new(
        manifold: PyObject,
        variant: &str,
        _max_iterations: usize,
        _tolerance: f64,
    ) -> PyResult<Self> {
        let cg_type = match variant {
            "FletcherReeves" => ConjugateGradientMethod::FletcherReeves,
            "PolakRibiere" => ConjugateGradientMethod::PolakRibiere,
            _ => return Err(PyValueError::new_err("Invalid CG variant. Use 'FletcherReeves' or 'PolakRibiere'")),
        };
        
        let config = CGConfig {
            method: cg_type,
            ..Default::default()
        };
        
        Ok(Self { _manifold: manifold, config })
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        let variant = match self.config.method {
            ConjugateGradientMethod::FletcherReeves => "FletcherReeves",
            ConjugateGradientMethod::PolakRibiere => "PolakRibiere",
            _ => "Other",
        };
        format!(
            "ConjugateGradient(variant='{}', restart_period={})",
            variant,
            self.config.restart_period
        )
    }
}