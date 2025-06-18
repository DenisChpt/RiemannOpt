//! Python bindings for Riemannian optimizers.
//!
//! This module provides Python-friendly wrappers around the Rust optimizer implementations.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix};
use std::sync::{Arc, Mutex};
use std::cell::RefCell;

use riemannopt_core::{
    manifold::Manifold,
    optimizer_state::ConjugateGradientMethod,
    line_search::LineSearchParams,
};
use riemannopt_optim::{
    SGDConfig, MomentumMethod,
    AdamConfig,
    LBFGSConfig,
    CGConfig,
    TrustRegionConfig,
};
use riemannopt_core::step_size::StepSizeSchedule;

use crate::manifolds::*;
use crate::cost_function::PyCostFunction;

// Helper functions for matrix conversion between NumPy and nalgebra
fn numpy_to_nalgebra_matrix(array: &PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let shape = array.shape();
    let mut mat = DMatrix::zeros(shape[0], shape[1]);
    let slice = array.as_slice()?;
    
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            mat[(i, j)] = slice[i * shape[1] + j];
        }
    }
    
    Ok(mat)
}

fn matrix_to_numpy_array<'py>(
    py: Python<'py>,
    mat: &DMatrix<f64>,
    shape: &[usize],
) -> PyResult<PyObject> {
    let mut result = Vec::with_capacity(shape[0] * shape[1]);
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            result.push(mat[(i, j)]);
        }
    }
    
    let flat_array = numpy::PyArray1::from_vec_bound(py, result);
    Ok(flat_array.reshape(shape.to_vec())?.into())
}

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
        // Validate parameters
        if step_size <= 0.0 {
            return Err(PyValueError::new_err("step_size must be positive"));
        }
        if momentum < 0.0 || momentum >= 1.0 {
            return Err(PyValueError::new_err("momentum must be in [0, 1)"));
        }
        if max_iterations <= 0 {
            return Err(PyValueError::new_err("max_iterations must be positive"));
        }
        if tolerance <= 0.0 {
            return Err(PyValueError::new_err("tolerance must be positive"));
        }
        
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
    ///     point: Current point on the manifold (1D array for sphere, 2D array for Stiefel/Grassmann)
    ///     gradient: Gradient at the current point
    ///
    /// Returns:
    ///     new_point: Updated point on the manifold
    pub fn step<'py>(
        &self,
        py: Python<'py>,
        manifold: &Bound<'_, PyAny>,
        point: &Bound<'_, PyAny>,
        gradient: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        // Apply SGD update with manifold projection
        let learning_rate = match &self.config.step_size {
            StepSizeSchedule::Constant(lr) => *lr,
            _ => 0.01,  // Default fallback
        };
        
        // Handle different manifold types
        if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            // Sphere uses 1D arrays
            let point_array = point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray1<'_, f64>>()?;
            
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            // Convert Euclidean gradient to Riemannian gradient
            let riem_grad = sphere.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point = sphere.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                
            Ok(numpy::PyArray1::from_slice_bound(py, new_point.as_slice()).into())
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            // Stiefel uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = point_array.shape();
            
            // Convert numpy arrays to nalgebra matrices
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let gradient_mat = numpy_to_nalgebra_matrix(&gradient_array)?;
            
            // Convert to vectors for Rust API
            let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
            let gradient_vec = DVector::from_vec(gradient_mat.as_slice().to_vec());
            
            let riem_grad = stiefel.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point_vec = stiefel.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix and then to numpy array
            let new_point_mat = DMatrix::from_vec(shape[0], shape[1], new_point_vec.as_slice().to_vec());
            matrix_to_numpy_array(py, &new_point_mat, shape)
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            // Grassmann uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = point_array.shape();
            
            // Convert numpy arrays to nalgebra matrices
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let gradient_mat = numpy_to_nalgebra_matrix(&gradient_array)?;
            
            // Convert to vectors for Rust API
            let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
            let gradient_vec = DVector::from_vec(gradient_mat.as_slice().to_vec());
            
            let riem_grad = grassmann.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point_vec = grassmann.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix and then to numpy array
            let new_point_mat = DMatrix::from_vec(shape[0], shape[1], new_point_vec.as_slice().to_vec());
            matrix_to_numpy_array(py, &new_point_mat, shape)
        } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
            // SPD uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            
            // Flatten to vectors for internal computation
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = spd.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point_vec = spd.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix shape
            let shape = point_array.shape();
            let flat_array = numpy::PyArray1::from_slice_bound(py, new_point_vec.as_slice());
            Ok(flat_array.reshape([shape[0], shape[1]])?.into())
        } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
            // Hyperbolic uses 1D arrays
            let point_array = point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray1<'_, f64>>()?;
            
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            // Convert Euclidean gradient to Riemannian gradient
            let riem_grad = hyperbolic.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point = hyperbolic.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                
            Ok(numpy::PyArray1::from_slice_bound(py, new_point.as_slice()).into())
        } else {
            Err(PyValueError::new_err("Unsupported manifold type"))
        }
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
        // Validate parameters
        if learning_rate <= 0.0 {
            return Err(PyValueError::new_err("learning_rate must be positive"));
        }
        if beta1 <= 0.0 || beta1 >= 1.0 {
            return Err(PyValueError::new_err("beta1 must be in (0, 1)"));
        }
        if beta2 <= 0.0 || beta2 >= 1.0 {
            return Err(PyValueError::new_err("beta2 must be in (0, 1)"));
        }
        if epsilon <= 0.0 {
            return Err(PyValueError::new_err("epsilon must be positive"));
        }
        
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
    ///     point: Current point on the manifold (1D array for sphere, 2D array for Stiefel/Grassmann)
    ///     gradient: Gradient at the current point
    ///
    /// Returns:
    ///     new_point: Updated point on the manifold
    pub fn step<'py>(
        &self,
        py: Python<'py>,
        manifold: &Bound<'_, PyAny>,
        point: &Bound<'_, PyAny>,
        gradient: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        // For now, implement a simple gradient descent step
        // TODO: Implement proper Adam algorithm with moment estimates
        let learning_rate = self.config.learning_rate;
        
        // Handle different manifold types
        if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            // Sphere uses 1D arrays
            let point_array = point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray1<'_, f64>>()?;
            
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = sphere.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point = sphere.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                
            Ok(numpy::PyArray1::from_slice_bound(py, new_point.as_slice()).into())
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            // Stiefel uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = point_array.shape();
            
            // Convert numpy arrays to nalgebra matrices
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let gradient_mat = numpy_to_nalgebra_matrix(&gradient_array)?;
            
            // Convert to vectors for Rust API
            let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
            let gradient_vec = DVector::from_vec(gradient_mat.as_slice().to_vec());
            
            let riem_grad = stiefel.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point_vec = stiefel.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix and then to numpy array
            let new_point_mat = DMatrix::from_vec(shape[0], shape[1], new_point_vec.as_slice().to_vec());
            matrix_to_numpy_array(py, &new_point_mat, shape)
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            // Grassmann uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = point_array.shape();
            
            // Convert numpy arrays to nalgebra matrices
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let gradient_mat = numpy_to_nalgebra_matrix(&gradient_array)?;
            
            // Convert to vectors for Rust API
            let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
            let gradient_vec = DVector::from_vec(gradient_mat.as_slice().to_vec());
            
            let riem_grad = grassmann.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point_vec = grassmann.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix and then to numpy array
            let new_point_mat = DMatrix::from_vec(shape[0], shape[1], new_point_vec.as_slice().to_vec());
            matrix_to_numpy_array(py, &new_point_mat, shape)
        } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
            // SPD uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            
            // Flatten to vectors for internal computation
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = spd.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point_vec = spd.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix shape
            let shape = point_array.shape();
            let flat_array = numpy::PyArray1::from_slice_bound(py, new_point_vec.as_slice());
            Ok(flat_array.reshape([shape[0], shape[1]])?.into())
        } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
            // Hyperbolic uses 1D arrays
            let point_array = point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray1<'_, f64>>()?;
            
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = hyperbolic.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            let update = -learning_rate * &riem_grad;
            let new_point = hyperbolic.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                
            Ok(numpy::PyArray1::from_slice_bound(py, new_point.as_slice()).into())
        } else {
            Err(PyValueError::new_err("Unsupported manifold type"))
        }
    }

    /// Get the current configuration.
    #[getter]
    pub fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let config_dict = pyo3::types::PyDict::new_bound(py);
        config_dict.set_item("learning_rate", self.config.learning_rate)?;
        config_dict.set_item("beta1", self.config.beta1)?;
        config_dict.set_item("beta2", self.config.beta2)?;
        config_dict.set_item("epsilon", self.config.epsilon)?;
        Ok(config_dict)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "Adam(learning_rate={}, beta1={}, beta2={}, epsilon={})",
            self.config.learning_rate,
            self.config.beta1,
            self.config.beta2,
            self.config.epsilon
        )
    }
}

/// Storage for L-BFGS vector pairs.
#[derive(Debug, Clone)]
struct LBFGSStorage {
    /// Position differences: s_k = x_{k+1} - x_k
    s_vectors: Vec<DVector<f64>>,
    /// Gradient differences: y_k = g_{k+1} - g_k
    y_vectors: Vec<DVector<f64>>,
    /// Inner products: rho_k = 1 / <s_k, y_k>
    rho_values: Vec<f64>,
    /// Maximum number of pairs to store
    capacity: usize,
}

impl LBFGSStorage {
    /// Creates new storage with given capacity.
    fn new(capacity: usize) -> Self {
        Self {
            s_vectors: Vec::with_capacity(capacity),
            y_vectors: Vec::with_capacity(capacity),
            rho_values: Vec::with_capacity(capacity),
            capacity,
        }
    }

    /// Adds a new vector pair to storage.
    fn push(&mut self, s: DVector<f64>, y: DVector<f64>, sy_inner: f64) {
        let rho = 1.0 / sy_inner;
        
        // Remove oldest if at capacity
        if self.s_vectors.len() >= self.capacity {
            self.s_vectors.remove(0);
            self.y_vectors.remove(0);
            self.rho_values.remove(0);
        }
        
        // Add new vectors
        self.s_vectors.push(s);
        self.y_vectors.push(y);
        self.rho_values.push(rho);
    }

    /// Returns the number of stored pairs.
    fn len(&self) -> usize {
        self.s_vectors.len()
    }

    /// Returns true if storage is empty.
    fn is_empty(&self) -> bool {
        self.s_vectors.is_empty()
    }
}

/// State for the L-BFGS optimizer.
#[derive(Debug, Clone)]
struct LBFGSState {
    /// Stored vector pairs for Hessian approximation
    storage: LBFGSStorage,
    /// Previous point
    prev_point: Option<DVector<f64>>,
    /// Previous gradient
    prev_gradient: Option<DVector<f64>>,
    /// Iteration counter
    iteration: usize,
}

impl LBFGSState {
    /// Creates a new L-BFGS state with given memory size.
    fn new(memory_size: usize) -> Self {
        Self {
            storage: LBFGSStorage::new(memory_size),
            prev_point: None,
            prev_gradient: None,
            iteration: 0,
        }
    }
}

/// Riemannian L-BFGS optimizer.
#[pyclass(name = "LBFGS")]
pub struct PyLBFGS {
    manifold: PyObject,
    config: LBFGSConfig<f64>,
    state: RefCell<Option<LBFGSState>>,
}

#[pymethods]
impl PyLBFGS {
    /// Create a new L-BFGS optimizer.
    ///
    /// Args:
    ///     memory_size: Number of past iterations to store (default: 10)
    ///     line_search: Line search type ('wolfe' or 'backtracking', default: 'wolfe')
    ///     max_line_search_iters: Maximum line search iterations (default: 20)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    #[new]
    #[pyo3(signature = (memory_size=10, line_search="wolfe", max_line_search_iters=20, tolerance=1e-6))]
    pub fn new(
        memory_size: usize,
        line_search: &str,
        max_line_search_iters: usize,
        tolerance: f64,
    ) -> PyResult<Self> {
        if memory_size == 0 {
            return Err(PyValueError::new_err("memory_size must be positive"));
        }
        
        if line_search != "wolfe" && line_search != "backtracking" && line_search != "exact" {
            return Err(PyValueError::new_err("line_search must be 'wolfe', 'backtracking', or 'exact'"));
        }
        
        let mut config = LBFGSConfig::default()
            .with_memory_size(memory_size)
            .with_initial_step_size(0.001);  // Use even smaller initial step size for stability
            
        // Configure line search params based on type
        if line_search == "backtracking" {
            config.line_search_params = LineSearchParams::backtracking();
        } else if line_search == "wolfe" {
            config.line_search_params = LineSearchParams::strong_wolfe();
        }
        
        Ok(Self { 
            manifold: Python::with_gil(|py| py.None()),
            config,
            state: RefCell::new(None),
        })
    }
    
    /// Take an optimization step.
    ///
    /// Args:
    ///     manifold: The manifold to optimize on
    ///     point: Current point on the manifold  
    ///     gradient: Gradient at the current point
    ///
    /// Returns:
    ///     New point after taking the step
    pub fn step<'py>(
        &self,
        py: Python<'py>,
        manifold: &Bound<'py, PyAny>,
        point: &Bound<'py, PyAny>,
        gradient: &Bound<'py, PyAny>,
    ) -> PyResult<PyObject> {
        // Extract point and gradient as vectors
        let (point_vec, gradient_vec, shape) = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            let point_array = point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray1<'_, f64>>()?;
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            (point_vec, gradient_vec, vec![])
        } else if let Ok(_) = manifold.extract::<PyRef<PyStiefel>>() {
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            
            // Convert numpy arrays to nalgebra matrices
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let gradient_mat = numpy_to_nalgebra_matrix(&gradient_array)?;
            
            // Convert to vectors for Rust API
            let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
            let gradient_vec = DVector::from_vec(gradient_mat.as_slice().to_vec());
            (point_vec, gradient_vec, shape)
        } else if let Ok(_) = manifold.extract::<PyRef<PyGrassmann>>() {
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            
            // Convert numpy arrays to nalgebra matrices
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let gradient_mat = numpy_to_nalgebra_matrix(&gradient_array)?;
            
            // Convert to vectors for Rust API
            let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
            let gradient_vec = DVector::from_vec(gradient_mat.as_slice().to_vec());
            (point_vec, gradient_vec, shape)
        } else if let Ok(_) = manifold.extract::<PyRef<PySPD>>() {
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            
            // Convert numpy arrays to nalgebra matrices
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let gradient_mat = numpy_to_nalgebra_matrix(&gradient_array)?;
            
            // Convert to vectors for Rust API
            let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
            let gradient_vec = DVector::from_vec(gradient_mat.as_slice().to_vec());
            (point_vec, gradient_vec, shape)
        } else if let Ok(_) = manifold.extract::<PyRef<PyHyperbolic>>() {
            let point_array = point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray1<'_, f64>>()?;
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            (point_vec, gradient_vec, vec![])
        } else {
            return Err(PyValueError::new_err("Unsupported manifold type"));
        };

        // Project gradient to Riemannian gradient
        let riem_grad = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            sphere.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            stiefel.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            grassmann.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
            spd.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
            hyperbolic.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err("Unsupported manifold type"));
        };

        // Initialize or get state
        let mut state_ref = self.state.borrow_mut();
        if state_ref.is_none() {
            *state_ref = Some(LBFGSState::new(self.config.memory_size));
        }
        let state = state_ref.as_mut().unwrap();

        // Compute search direction using L-BFGS two-loop recursion
        let search_direction = if state.storage.is_empty() {
            // No history, use negative gradient
            -&riem_grad
        } else {
            // L-BFGS two-loop recursion
            let m = state.storage.len();
            let mut alpha = vec![0.0; m];
            let mut q = riem_grad.clone();

            // First loop: compute alpha values and update q
            for i in (0..m).rev() {
                let s = &state.storage.s_vectors[i];
                let y = &state.storage.y_vectors[i];
                let rho = state.storage.rho_values[i];
                
                // Compute inner product based on manifold
                let sq_inner = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
                    sphere.get_inner().inner_product(&point_vec, s, &q)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
                    stiefel.get_inner().inner_product(&point_vec, s, &q)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
                    grassmann.get_inner().inner_product(&point_vec, s, &q)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
                    spd.get_inner().inner_product(&point_vec, s, &q)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
                    hyperbolic.get_inner().inner_product(&point_vec, s, &q)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else {
                    return Err(PyValueError::new_err("Unsupported manifold type"));
                };
                
                alpha[i] = rho * sq_inner;
                q = &q - &(y * alpha[i]);
            }

            // Scale by initial Hessian approximation
            if m > 0 {
                let s_last = &state.storage.s_vectors[m - 1];
                let y_last = &state.storage.y_vectors[m - 1];
                
                let sy_inner = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
                    sphere.get_inner().inner_product(&point_vec, s_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
                    stiefel.get_inner().inner_product(&point_vec, s_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
                    grassmann.get_inner().inner_product(&point_vec, s_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
                    spd.get_inner().inner_product(&point_vec, s_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
                    hyperbolic.get_inner().inner_product(&point_vec, s_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else {
                    return Err(PyValueError::new_err("Unsupported manifold type"));
                };
                
                let yy_inner = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
                    sphere.get_inner().inner_product(&point_vec, y_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
                    stiefel.get_inner().inner_product(&point_vec, y_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
                    grassmann.get_inner().inner_product(&point_vec, y_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
                    spd.get_inner().inner_product(&point_vec, y_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
                    hyperbolic.get_inner().inner_product(&point_vec, y_last, y_last)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else {
                    return Err(PyValueError::new_err("Unsupported manifold type"));
                };
                
                if yy_inner > 0.0 {
                    let gamma = sy_inner / yy_inner;
                    q = &q * gamma;
                }
            }

            // Second loop: compute search direction
            let mut r = q;
            for i in 0..m {
                let s = &state.storage.s_vectors[i];
                let y = &state.storage.y_vectors[i];
                let rho = state.storage.rho_values[i];
                
                let yr_inner = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
                    sphere.get_inner().inner_product(&point_vec, y, &r)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
                    stiefel.get_inner().inner_product(&point_vec, y, &r)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
                    grassmann.get_inner().inner_product(&point_vec, y, &r)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
                    spd.get_inner().inner_product(&point_vec, y, &r)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
                    hyperbolic.get_inner().inner_product(&point_vec, y, &r)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else {
                    return Err(PyValueError::new_err("Unsupported manifold type"));
                };
                
                let beta = rho * yr_inner;
                let coeff = alpha[i] - beta;
                r = &r + &(s * coeff);
            }

            -r
        };

        // Determine step size
        let step_size = if state.iteration == 0 {
            self.config.initial_step_size
        } else {
            1.0  // Use step size 1.0 for quasi-Newton direction
        };

        // Compute update - make sure it's in the tangent space
        let mut update = step_size * &search_direction;
        
        // Project update to tangent space to ensure it's valid
        update = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            sphere.get_inner().project_tangent(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            stiefel.get_inner().project_tangent(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            grassmann.get_inner().project_tangent(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
            update  // SPD doesn't need projection for tangent vectors
        } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
            update  // Hyperbolic tangent space projection might be identity
        } else {
            return Err(PyValueError::new_err("Unsupported manifold type"));
        };
        
        // Retract to get new point
        let new_point_vec = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            sphere.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            stiefel.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            grassmann.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
            spd.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
            hyperbolic.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        } else {
            return Err(PyValueError::new_err("Unsupported manifold type"));
        };

        // Update storage if we have previous gradient
        if let (Some(prev_point), Some(prev_grad)) = (&state.prev_point, &state.prev_gradient) {
            // Transport previous gradient to new point for comparison
            let transported_prev_grad = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
                sphere.get_inner().parallel_transport(prev_point, &new_point_vec, prev_grad)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
                stiefel.get_inner().parallel_transport(prev_point, &new_point_vec, prev_grad)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
                grassmann.get_inner().parallel_transport(prev_point, &new_point_vec, prev_grad)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
                spd.get_inner().parallel_transport(prev_point, &new_point_vec, prev_grad)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
                hyperbolic.get_inner().parallel_transport(prev_point, &new_point_vec, prev_grad)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else {
                return Err(PyValueError::new_err("Unsupported manifold type"));
            };
            
            // Compute differences
            let s = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
                sphere.get_inner().inverse_retract(prev_point, &new_point_vec)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
                stiefel.get_inner().inverse_retract(prev_point, &new_point_vec)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
                grassmann.get_inner().inverse_retract(prev_point, &new_point_vec)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
                spd.get_inner().inverse_retract(prev_point, &new_point_vec)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
                hyperbolic.get_inner().inverse_retract(prev_point, &new_point_vec)
                    .map_err(|e| PyValueError::new_err(e.to_string()))?
            } else {
                return Err(PyValueError::new_err("Unsupported manifold type"));
            };
            
            let y = &riem_grad - &transported_prev_grad;
            
            // Add to storage (with cautious update check)
            if self.config.use_cautious_updates {
                let sy_inner = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
                    sphere.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
                    stiefel.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
                    grassmann.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
                    spd.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
                    hyperbolic.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else {
                    return Err(PyValueError::new_err("Unsupported manifold type"));
                };
                
                if sy_inner > 0.0 {
                    state.storage.push(s, y, sy_inner);
                }
            } else {
                let sy_inner = if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
                    sphere.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
                    stiefel.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
                    grassmann.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
                    spd.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
                    hyperbolic.get_inner().inner_product(&new_point_vec, &s, &y)
                        .map_err(|e| PyValueError::new_err(e.to_string()))?
                } else {
                    return Err(PyValueError::new_err("Unsupported manifold type"));
                };
                
                if sy_inner > 0.0 {
                    state.storage.push(s, y, sy_inner);
                }
            }
        }
        
        // Update L-BFGS state
        state.prev_point = Some(new_point_vec.clone());
        state.prev_gradient = Some(riem_grad.clone());
        state.iteration += 1;

        // Return new point in appropriate format
        if shape.is_empty() {
            // 1D array (Sphere, Hyperbolic)
            Ok(numpy::PyArray1::from_slice_bound(py, new_point_vec.as_slice()).into())
        } else {
            // 2D array (Stiefel, Grassmann, SPD)
            let new_point_mat = DMatrix::from_vec(shape[0], shape[1], new_point_vec.as_slice().to_vec());
            matrix_to_numpy_array(py, &new_point_mat, &shape)
        }
    }
    
    /// Reset the optimizer state.
    pub fn reset(&self) {
        *self.state.borrow_mut() = None;
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "LBFGS(memory_size={})",
            self.config.memory_size
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

/// Riemannian Trust Region optimizer.
#[pyclass(name = "TrustRegion")]
pub struct PyTrustRegion {
    config: TrustRegionConfig<f64>,
    max_iterations: usize,
    tolerance: f64,
}

#[pymethods]
impl PyTrustRegion {
    /// Create a new Trust Region optimizer.
    ///
    /// Args:
    ///     initial_radius: Initial trust region radius (default: 1.0)
    ///     max_radius: Maximum trust region radius (default: 10.0)
    ///     min_radius: Minimum trust region radius (default: 1e-6)
    ///     acceptance_ratio: Ratio threshold for accepting a step (default: 0.1)
    ///     max_iterations: Maximum iterations (default: 1000)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    ///     use_exact_hessian: Whether to use exact Hessian (default: False)
    #[new]
    #[pyo3(signature = (initial_radius=1.0, max_radius=10.0, min_radius=1e-6, acceptance_ratio=0.1, max_iterations=1000, tolerance=1e-6, use_exact_hessian=false))]
    pub fn new(
        initial_radius: f64,
        max_radius: f64,
        min_radius: f64,
        acceptance_ratio: f64,
        max_iterations: usize,
        tolerance: f64,
        use_exact_hessian: bool,
    ) -> PyResult<Self> {
        let mut config = TrustRegionConfig::new()
            .with_initial_radius(initial_radius)
            .with_max_radius(max_radius)
            .with_min_radius(min_radius)
            .with_acceptance_ratio(acceptance_ratio);
            
        if use_exact_hessian {
            config = config.with_exact_hessian();
        }
        
        Ok(Self {
            config,
            max_iterations,
            tolerance,
        })
    }

    /// Perform one optimization step.
    ///
    /// Note: Trust Region requires second-order information, so this is a simplified version.
    /// For now, it performs a gradient descent step with trust region radius control.
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
        point: &Bound<'_, PyAny>,
        gradient: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        // For now, implement as a gradient descent with adaptive step size
        // based on trust region radius
        let step_size = self.config.initial_radius;
        
        // Handle different manifold types
        if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            // Sphere uses 1D arrays
            let point_array = point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray1<'_, f64>>()?;
            
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = sphere.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Scale step by trust region radius
            let grad_norm = riem_grad.norm();
            let effective_step = if grad_norm > self.config.initial_radius {
                self.config.initial_radius / grad_norm
            } else {
                step_size
            };
            
            let update = -effective_step * &riem_grad;
            let new_point = sphere.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                
            Ok(numpy::PyArray1::from_slice_bound(py, new_point.as_slice()).into())
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            // Stiefel uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            
            // Flatten to vectors for internal computation
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = stiefel.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Scale step by trust region radius
            let grad_norm = riem_grad.norm();
            let effective_step = if grad_norm > self.config.initial_radius {
                self.config.initial_radius / grad_norm
            } else {
                step_size
            };
            
            let update = -effective_step * &riem_grad;
            let new_point_vec = stiefel.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix shape
            let shape = point_array.shape();
            let flat_array = numpy::PyArray1::from_slice_bound(py, new_point_vec.as_slice());
            Ok(flat_array.reshape([shape[0], shape[1]])?.into())
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            // Grassmann uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            
            // Flatten to vectors for internal computation
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = grassmann.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Scale step by trust region radius
            let grad_norm = riem_grad.norm();
            let effective_step = if grad_norm > self.config.initial_radius {
                self.config.initial_radius / grad_norm
            } else {
                step_size
            };
            
            let update = -effective_step * &riem_grad;
            let new_point_vec = grassmann.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix shape
            let shape = point_array.shape();
            let flat_array = numpy::PyArray1::from_slice_bound(py, new_point_vec.as_slice());
            Ok(flat_array.reshape([shape[0], shape[1]])?.into())
        } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
            // SPD uses 2D arrays (matrices)
            let point_array = point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray2<'_, f64>>()?;
            
            // Flatten to vectors for internal computation
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = spd.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Scale step by trust region radius
            let grad_norm = riem_grad.norm();
            let effective_step = if grad_norm > self.config.initial_radius {
                self.config.initial_radius / grad_norm
            } else {
                step_size
            };
            
            let update = -effective_step * &riem_grad;
            let new_point_vec = spd.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Convert back to matrix shape
            let shape = point_array.shape();
            let flat_array = numpy::PyArray1::from_slice_bound(py, new_point_vec.as_slice());
            Ok(flat_array.reshape([shape[0], shape[1]])?.into())
        } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
            // Hyperbolic uses 1D arrays
            let point_array = point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let gradient_array = gradient.extract::<PyReadonlyArray1<'_, f64>>()?;
            
            let point_vec = DVector::from_column_slice(point_array.as_slice()?);
            let gradient_vec = DVector::from_column_slice(gradient_array.as_slice()?);
            
            let riem_grad = hyperbolic.get_inner().euclidean_to_riemannian_gradient(&point_vec, &gradient_vec)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
            
            // Scale step by trust region radius
            let grad_norm = riem_grad.norm();
            let effective_step = if grad_norm > self.config.initial_radius {
                self.config.initial_radius / grad_norm
            } else {
                step_size
            };
            
            let update = -effective_step * &riem_grad;
            let new_point = hyperbolic.get_inner().retract(&point_vec, &update)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
                
            Ok(numpy::PyArray1::from_slice_bound(py, new_point.as_slice()).into())
        } else {
            Err(PyValueError::new_err("Unsupported manifold type"))
        }
    }

    /// Get the current configuration.
    #[getter]
    pub fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let config_dict = pyo3::types::PyDict::new_bound(py);
        config_dict.set_item("initial_radius", self.config.initial_radius)?;
        config_dict.set_item("max_radius", self.config.max_radius)?;
        config_dict.set_item("min_radius", self.config.min_radius)?;
        config_dict.set_item("acceptance_ratio", self.config.acceptance_ratio)?;
        config_dict.set_item("max_iterations", self.max_iterations)?;
        config_dict.set_item("tolerance", self.tolerance)?;
        config_dict.set_item("use_exact_hessian", self.config.use_exact_hessian)?;
        Ok(config_dict)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!(
            "TrustRegion(initial_radius={}, max_radius={}, min_radius={}, acceptance_ratio={})",
            self.config.initial_radius,
            self.config.max_radius, 
            self.config.min_radius,
            self.config.acceptance_ratio
        )
    }
}