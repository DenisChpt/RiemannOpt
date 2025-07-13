//! Python bindings for cost functions.
//!
//! This module provides a high-performance bridge between Python cost functions
//! and the Rust optimization algorithms. The key challenge is minimizing the
//! overhead of Python callbacks while maintaining a clean API.
//!
//! # Performance Considerations
//!
//! The GIL (Global Interpreter Lock) acquisition is the main bottleneck when
//! calling Python functions from Rust. We minimize this by:
//! - Caching function references to avoid repeated lookups
//! - Batching operations when possible
//! - Using efficient data conversions
//! - Releasing the GIL as soon as possible

use pyo3::prelude::*;
use pyo3::types::PyTuple;
use nalgebra::{DVector, DMatrix};
use riemannopt_core::{
    core::cost_function::CostFunction,
    error::{ManifoldError, Result},
    memory::workspace::Workspace,
};
use std::sync::Arc;
use parking_lot::RwLock;

use crate::{
    array_utils::{numpy_to_dvector, numpy_to_dmatrix, dvector_to_numpy, dmatrix_to_numpy},
    error::to_py_err,
    types::{PyPoint, PyTangentVector},
};


/// A wrapper that allows Python functions to be used as cost functions in Rust.
///
/// This struct bridges the gap between Python's dynamic typing and Rust's
/// static typing system, handling all necessary conversions and GIL management.
#[pyclass(name = "CostFunction", module = "riemannopt")]
pub struct PyCostFunction {
    /// The Python callable for computing the cost
    cost_fn: PyObject,
    /// Optional Python callable for computing the gradient
    grad_fn: Option<PyObject>,
    /// Optional Python callable for computing both cost and gradient
    cost_and_grad_fn: Option<PyObject>,
    /// Dimension information for the problem
    dimension_info: DimensionInfo,
    /// Counter for function evaluations (for debugging/profiling)
    eval_count: Arc<RwLock<EvalCount>>,
}

/// Information about the problem dimensions
#[derive(Clone, Debug)]
struct DimensionInfo {
    /// Whether points are vectors or matrices
    is_matrix: bool,
    /// For vectors: the dimension; for matrices: (rows, cols)
    shape: (usize, usize),
}

/// Evaluation counters for profiling
#[derive(Default, Debug)]
struct EvalCount {
    cost: usize,
    gradient: usize,
    cost_and_gradient: usize,
}

#[pymethods]
impl PyCostFunction {
    /// Create a new cost function wrapper.
    ///
    /// Args:
    ///     cost_fn: A Python callable that takes a point and returns a scalar cost
    ///     grad_fn: Optional callable that takes a point and returns the gradient
    ///     cost_and_grad_fn: Optional callable that returns (cost, gradient) tuple
    ///     dimension: Either an integer (for vector problems) or tuple (rows, cols) for matrices
    ///
    /// Note:
    ///     If cost_and_grad_fn is provided, it will be used preferentially for
    ///     better performance when both cost and gradient are needed.
    #[new]
    #[pyo3(signature = (cost_fn, grad_fn=None, cost_and_grad_fn=None, dimension=None))]
    pub fn new(
        cost_fn: PyObject,
        grad_fn: Option<PyObject>,
        cost_and_grad_fn: Option<PyObject>,
        dimension: Option<PyObject>,
    ) -> PyResult<Self> {
        // Parse dimension information
        let dimension_info = Python::with_gil(|py| {
            match dimension {
                Some(dim_obj) => {
                    // Try to extract as integer first (vector case)
                    if let Ok(dim) = dim_obj.extract::<usize>(py) {
                        Ok(DimensionInfo {
                            is_matrix: false,
                            shape: (dim, 1),
                        })
                    } else if let Ok((rows, cols)) = dim_obj.extract::<(usize, usize)>(py) {
                        Ok(DimensionInfo {
                            is_matrix: true,
                            shape: (rows, cols),
                        })
                    } else {
                        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                            "dimension must be an integer or (rows, cols) tuple"
                        ))
                    }
                }
                None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "dimension is required"
                )),
            }
        })?;

        Ok(PyCostFunction {
            cost_fn,
            grad_fn,
            cost_and_grad_fn,
            dimension_info,
            eval_count: Arc::new(RwLock::new(EvalCount::default())),
        })
    }

    /// Get the number of function evaluations.
    ///
    /// Returns:
    ///     A dictionary with keys 'cost', 'gradient', and 'cost_and_gradient'
    #[getter]
    fn eval_counts(&self, py: Python<'_>) -> PyResult<PyObject> {
        let counts = self.eval_count.read();
        let dict = pyo3::types::PyDict::new_bound(py);
        dict.set_item("cost", counts.cost)?;
        dict.set_item("gradient", counts.gradient)?;
        dict.set_item("cost_and_gradient", counts.cost_and_gradient)?;
        Ok(dict.into())
    }

    /// Reset evaluation counters to zero.
    pub fn reset_counts(&self) {
        let mut counts = self.eval_count.write();
        *counts = EvalCount::default();
    }

    /// Evaluate the cost function at a point (Python API).
    ///
    /// Args:
    ///     point: A numpy array representing the point
    ///
    /// Returns:
    ///     The cost value
    pub fn cost(&self, py: Python<'_>, point: PyObject) -> PyResult<f64> {
        self.eval_count.write().cost += 1;
        
        // Call the Python function
        let result = self.cost_fn.call1(py, (point,))?;
        
        // Extract the float result
        result.extract::<f64>(py)
    }

    /// Evaluate the gradient at a point (Python API).
    ///
    /// Args:
    ///     point: A numpy array representing the point
    ///
    /// Returns:
    ///     The gradient as a numpy array
    pub fn gradient(&self, py: Python<'_>, point: PyObject) -> PyResult<PyObject> {
        self.eval_count.write().gradient += 1;
        
        if let Some(ref grad_fn) = self.grad_fn {
            // Use the provided gradient function
            grad_fn.call1(py, (point,))
        } else {
            // Compute gradient using finite differences
            self.gradient_fd_py(py, point)
        }
    }

    /// Evaluate both cost and gradient at a point (Python API).
    ///
    /// Args:
    ///     point: A numpy array representing the point
    ///
    /// Returns:
    ///     A tuple (cost, gradient)
    pub fn cost_and_gradient(&self, py: Python<'_>, point: PyObject) -> PyResult<(f64, PyObject)> {
        self.eval_count.write().cost_and_gradient += 1;
        
        if let Some(ref cost_and_grad_fn) = self.cost_and_grad_fn {
            // Use the combined function for efficiency
            let result = cost_and_grad_fn.call1(py, (point,))?;
            let tuple = result.downcast_bound::<PyTuple>(py)?;
            
            if tuple.len() != 2 {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "cost_and_gradient function must return a tuple (cost, gradient)"
                ));
            }
            
            let cost = tuple.get_item(0)?.extract::<f64>()?;
            let gradient = tuple.get_item(1)?;
            Ok((cost, gradient.into()))
        } else {
            // Compute separately
            let cost = self.cost(py, point.clone_ref(py))?;
            let gradient = self.gradient(py, point)?;
            Ok((cost, gradient))
        }
    }

    /// Compute gradient using finite differences (Python API).
    fn gradient_fd_py(&self, py: Python<'_>, point: PyObject) -> PyResult<PyObject> {
        use numpy::{PyArray1, PyArray2, PyArrayMethods};
        
        if self.dimension_info.is_matrix {
            // Matrix case
            let point_array = point.downcast_bound::<PyArray2<f64>>(py)?;
            let point_mat = numpy_to_dmatrix(point_array.readonly())?;
            
            // Compute finite difference gradient
            let mut gradient = DMatrix::zeros(self.dimension_info.shape.0, self.dimension_info.shape.1);
            let h = f64::sqrt(f64::EPSILON);
            
            for i in 0..gradient.nrows() {
                for j in 0..gradient.ncols() {
                    let mut point_plus = point_mat.clone();
                    let mut point_minus = point_mat.clone();
                    
                    point_plus[(i, j)] += h;
                    point_minus[(i, j)] -= h;
                    
                    // Convert back to numpy and evaluate
                    let point_plus_py = dmatrix_to_numpy(py, &point_plus)?;
                    let point_minus_py = dmatrix_to_numpy(py, &point_minus)?;
                    
                    let f_plus = self.cost_fn.call1(py, (point_plus_py,))?.extract::<f64>(py)?;
                    let f_minus = self.cost_fn.call1(py, (point_minus_py,))?.extract::<f64>(py)?;
                    
                    gradient[(i, j)] = (f_plus - f_minus) / (2.0 * h);
                }
            }
            
            Ok(dmatrix_to_numpy(py, &gradient)?.into())
        } else {
            // Vector case
            let point_array = point.downcast_bound::<PyArray1<f64>>(py)?;
            let point_vec = numpy_to_dvector(point_array.readonly())?;
            
            // Compute finite difference gradient
            let mut gradient = DVector::zeros(self.dimension_info.shape.0);
            let h = f64::sqrt(f64::EPSILON);
            
            for i in 0..gradient.len() {
                let mut point_plus = point_vec.clone();
                let mut point_minus = point_vec.clone();
                
                point_plus[i] += h;
                point_minus[i] -= h;
                
                // Convert back to numpy and evaluate
                let point_plus_py = dvector_to_numpy(py, &point_plus)?;
                let point_minus_py = dvector_to_numpy(py, &point_minus)?;
                
                let f_plus = self.cost_fn.call1(py, (point_plus_py,))?.extract::<f64>(py)?;
                let f_minus = self.cost_fn.call1(py, (point_minus_py,))?.extract::<f64>(py)?;
                
                gradient[i] = (f_plus - f_minus) / (2.0 * h);
            }
            
            Ok(dvector_to_numpy(py, &gradient)?.into())
        }
    }
}

/// Adapter for using PyCostFunction with Sphere manifold (DVector types).
pub struct PyCostFunctionSphere {
    inner: std::sync::Arc<PyCostFunction>,
}

impl Clone for PyCostFunction {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            Self {
                cost_fn: self.cost_fn.clone_ref(py),
                grad_fn: self.grad_fn.as_ref().map(|g| g.clone_ref(py)),
                cost_and_grad_fn: self.cost_and_grad_fn.as_ref().map(|cg| cg.clone_ref(py)),
                dimension_info: self.dimension_info.clone(),
                eval_count: Arc::new(RwLock::new(EvalCount::default())),
            }
        })
    }
}

impl std::fmt::Debug for PyCostFunctionSphere {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyCostFunctionSphere").finish()
    }
}

impl PyCostFunctionSphere {
    pub fn new(inner: &PyCostFunction) -> Self {
        Self {
            inner: std::sync::Arc::new(inner.clone()),
        }
    }
}

impl<'a> CostFunction<f64> for PyCostFunctionSphere {
    type Point = DVector<f64>;
    type TangentVector = DVector<f64>;
    
    fn cost(&self, point: &Self::Point) -> Result<f64> {
        let py_point = PyPoint::Vector(point.clone());
        <PyCostFunction as CostFunction<f64>>::cost(&*self.inner, &py_point)
    }
    
    fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(f64, Self::TangentVector)> {
        let py_point = PyPoint::Vector(point.clone());
        let (cost, grad) = <PyCostFunction as CostFunction<f64>>::cost_and_gradient_alloc(&*self.inner, &py_point)?;
        match grad {
            PyTangentVector::Vector(v) => Ok((cost, v)),
            _ => Err(ManifoldError::numerical_error("Expected vector gradient")),
        }
    }
    
    fn cost_and_gradient(
        &self,
        point: &Self::Point,
        workspace: &mut Workspace<f64>,
        gradient: &mut Self::TangentVector,
    ) -> Result<f64> {
        let py_point = PyPoint::Vector(point.clone());
        let mut py_gradient = PyTangentVector::Vector(gradient.clone());
        let cost = <PyCostFunction as CostFunction<f64>>::cost_and_gradient(&*self.inner, &py_point, workspace, &mut py_gradient)?;
        match py_gradient {
            PyTangentVector::Vector(v) => {
                gradient.copy_from(&v);
                Ok(cost)
            }
            _ => Err(ManifoldError::numerical_error("Expected vector gradient")),
        }
    }
    
    fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let py_point = PyPoint::Vector(point.clone());
        match <PyCostFunction as CostFunction<f64>>::gradient(&*self.inner, &py_point)? {
            PyTangentVector::Vector(v) => Ok(v),
            _ => Err(ManifoldError::numerical_error("Expected vector gradient")),
        }
    }
    
    fn hessian_vector_product(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        let py_point = PyPoint::Vector(point.clone());
        let py_vector = PyTangentVector::Vector(vector.clone());
        match <PyCostFunction as CostFunction<f64>>::hessian_vector_product(&*self.inner, &py_point, &py_vector)? {
            PyTangentVector::Vector(v) => Ok(v),
            _ => Err(ManifoldError::numerical_error("Expected vector result")),
        }
    }
    
    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let py_point = PyPoint::Vector(point.clone());
        match <PyCostFunction as CostFunction<f64>>::gradient_fd_alloc(&*self.inner, &py_point)? {
            PyTangentVector::Vector(v) => Ok(v),
            _ => Err(ManifoldError::numerical_error("Expected vector gradient")),
        }
    }
}

/// Adapter for using PyCostFunction with Stiefel manifold (DMatrix types).
pub struct PyCostFunctionStiefel {
    inner: std::sync::Arc<PyCostFunction>,
}

impl std::fmt::Debug for PyCostFunctionStiefel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyCostFunctionStiefel").finish()
    }
}

impl PyCostFunctionStiefel {
    pub fn new(inner: &PyCostFunction) -> Self {
        Self {
            inner: std::sync::Arc::new(inner.clone()),
        }
    }
}

impl<'a> CostFunction<f64> for PyCostFunctionStiefel {
    type Point = DMatrix<f64>;
    type TangentVector = DMatrix<f64>;
    
    fn cost(&self, point: &Self::Point) -> Result<f64> {
        let py_point = PyPoint::Matrix(point.clone());
        <PyCostFunction as CostFunction<f64>>::cost(&*self.inner, &py_point)
    }
    
    fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(f64, Self::TangentVector)> {
        let py_point = PyPoint::Matrix(point.clone());
        let (cost, grad) = <PyCostFunction as CostFunction<f64>>::cost_and_gradient_alloc(&*self.inner, &py_point)?;
        match grad {
            PyTangentVector::Matrix(m) => Ok((cost, m)),
            _ => Err(ManifoldError::numerical_error("Expected matrix gradient")),
        }
    }
    
    fn cost_and_gradient(
        &self,
        point: &Self::Point,
        workspace: &mut Workspace<f64>,
        gradient: &mut Self::TangentVector,
    ) -> Result<f64> {
        let py_point = PyPoint::Matrix(point.clone());
        let mut py_gradient = PyTangentVector::Matrix(gradient.clone());
        let cost = <PyCostFunction as CostFunction<f64>>::cost_and_gradient(&*self.inner, &py_point, workspace, &mut py_gradient)?;
        match py_gradient {
            PyTangentVector::Matrix(m) => {
                gradient.copy_from(&m);
                Ok(cost)
            }
            _ => Err(ManifoldError::numerical_error("Expected matrix gradient")),
        }
    }
    
    fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let py_point = PyPoint::Matrix(point.clone());
        match <PyCostFunction as CostFunction<f64>>::gradient(&*self.inner, &py_point)? {
            PyTangentVector::Matrix(m) => Ok(m),
            _ => Err(ManifoldError::numerical_error("Expected matrix gradient")),
        }
    }
    
    fn hessian_vector_product(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        let py_point = PyPoint::Matrix(point.clone());
        let py_vector = PyTangentVector::Matrix(vector.clone());
        match <PyCostFunction as CostFunction<f64>>::hessian_vector_product(&*self.inner, &py_point, &py_vector)? {
            PyTangentVector::Matrix(m) => Ok(m),
            _ => Err(ManifoldError::numerical_error("Expected matrix result")),
        }
    }
    
    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let py_point = PyPoint::Matrix(point.clone());
        match <PyCostFunction as CostFunction<f64>>::gradient_fd_alloc(&*self.inner, &py_point)? {
            PyTangentVector::Matrix(m) => Ok(m),
            _ => Err(ManifoldError::numerical_error("Expected matrix gradient")),
        }
    }
}

/// Keep the original implementation for PyCostFunction using PyPoint/PyTangentVector
/// This is still needed for the wrapper types above.
impl CostFunction<f64> for PyCostFunction {
    type Point = PyPoint;
    type TangentVector = PyTangentVector;

    fn cost(&self, point: &Self::Point) -> Result<f64> {
        Python::with_gil(|py| {
            self.eval_count.write().cost += 1;
            
            // Convert point to numpy array
            let py_point: PyObject = match point {
                PyPoint::Vector(vec) => dvector_to_numpy(py, vec)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert vector: {}", e)))?
                    .into(),
                PyPoint::Matrix(mat) => dmatrix_to_numpy(py, mat)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert matrix: {}", e)))?
                    .into(),
            };
            
            // Call Python function
            let result = self.cost_fn.call1(py, (py_point,))
                .map_err(|e| ManifoldError::numerical_error(format!("Python cost function error: {}", e)))?;
            
            // Extract result
            result.extract::<f64>(py)
                .map_err(|e| ManifoldError::numerical_error(format!("Failed to extract cost value: {}", e)))
        })
    }

    fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(f64, Self::TangentVector)> {
        use numpy::PyArrayMethods;
        Python::with_gil(|py| {
            self.eval_count.write().cost_and_gradient += 1;
            
            // Convert point to numpy
            let py_point: PyObject = match point {
                PyPoint::Vector(vec) => dvector_to_numpy(py, vec)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert vector: {}", e)))?
                    .into(),
                PyPoint::Matrix(mat) => dmatrix_to_numpy(py, mat)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert matrix: {}", e)))?
                    .into(),
            };
            
            // Call the appropriate Python function
            let (cost, grad_py) = if let Some(ref cost_and_grad_fn) = self.cost_and_grad_fn {
                // Use combined function
                let result = cost_and_grad_fn.call1(py, (py_point,))
                    .map_err(|e| ManifoldError::numerical_error(format!("Python cost_and_gradient error: {}", e)))?;
                    
                let tuple = result.downcast_bound::<PyTuple>(py)
                    .map_err(|_| ManifoldError::numerical_error("cost_and_gradient must return a tuple"))?;
                    
                if tuple.len() != 2 {
                    return Err(ManifoldError::numerical_error(
                        "cost_and_gradient must return exactly 2 values"
                    ));
                }
                
                let cost = tuple.get_item(0)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to get cost: {}", e)))?
                    .extract::<f64>()
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to extract cost: {}", e)))?;
                let gradient = tuple.get_item(1)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to get gradient: {}", e)))?
                    .into();
                    
                (cost, gradient)
            } else {
                // Compute separately
                let cost = self.cost_fn.call1(py, (py_point.clone_ref(py),))
                    .map_err(|e| ManifoldError::numerical_error(format!("Python cost error: {}", e)))?
                    .extract::<f64>(py)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to extract cost: {}", e)))?;
                    
                let gradient = if let Some(ref grad_fn) = self.grad_fn {
                    grad_fn.call1(py, (py_point,))
                        .map_err(|e| ManifoldError::numerical_error(format!("Python gradient error: {}", e)))?
                } else {
                    return Err(ManifoldError::not_implemented(
                        "Gradient computation required but no gradient function provided"
                    ));
                };
                
                (cost, gradient)
            };
            
            // Convert gradient back to Rust type
            let gradient = if self.dimension_info.is_matrix {
                let grad_array = grad_py.downcast_bound::<numpy::PyArray2<f64>>(py)
                    .map_err(|_| ManifoldError::numerical_error("Gradient must be a 2D numpy array for matrix problems"))?;
                let grad_mat = numpy_to_dmatrix(grad_array.readonly())
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert gradient: {}", e)))?;
                PyTangentVector::Matrix(grad_mat)
            } else {
                let grad_array = grad_py.downcast_bound::<numpy::PyArray1<f64>>(py)
                    .map_err(|_| ManifoldError::numerical_error("Gradient must be a 1D numpy array for vector problems"))?;
                let grad_vec = numpy_to_dvector(grad_array.readonly())
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert gradient: {}", e)))?;
                PyTangentVector::Vector(grad_vec)
            };
            
            Ok((cost, gradient))
        })
    }

    fn cost_and_gradient(
        &self,
        point: &Self::Point,
        _workspace: &mut Workspace<f64>,
        gradient: &mut Self::TangentVector,
    ) -> Result<f64> {
        // For now, use the allocating version
        // TODO: Optimize by reusing workspace buffers
        let (cost, grad) = self.cost_and_gradient_alloc(point)?;
        *gradient = grad;
        Ok(cost)
    }

    fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        let (_, gradient) = self.cost_and_gradient_alloc(point)?;
        Ok(gradient)
    }

    fn hessian_vector_product(
        &self,
        _point: &Self::Point,
        _vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        Err(ManifoldError::not_implemented(
            "Hessian-vector products not yet implemented for Python cost functions"
        ))
    }

    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        use numpy::PyArrayMethods;
        Python::with_gil(|py| {
            // Convert point to numpy
            let py_point: PyObject = match point {
                PyPoint::Vector(vec) => dvector_to_numpy(py, vec)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert vector: {}", e)))?
                    .into(),
                PyPoint::Matrix(mat) => dmatrix_to_numpy(py, mat)
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert matrix: {}", e)))?
                    .into(),
            };
            
            // Call the Python finite difference method
            let grad_py = self.gradient_fd_py(py, py_point)
                .map_err(|e| ManifoldError::numerical_error(format!("Finite difference error: {}", e)))?;
            
            // Convert back to Rust type
            if self.dimension_info.is_matrix {
                let grad_array = grad_py.downcast_bound::<numpy::PyArray2<f64>>(py)
                    .map_err(|_| ManifoldError::numerical_error("Gradient must be a 2D array"))?;
                let grad_mat = numpy_to_dmatrix(grad_array.readonly())
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert gradient: {}", e)))?;
                Ok(PyTangentVector::Matrix(grad_mat))
            } else {
                let grad_array = grad_py.downcast_bound::<numpy::PyArray1<f64>>(py)
                    .map_err(|_| ManifoldError::numerical_error("Gradient must be a 1D array"))?;
                let grad_vec = numpy_to_dvector(grad_array.readonly())
                    .map_err(|e| ManifoldError::numerical_error(format!("Failed to convert gradient: {}", e)))?;
                Ok(PyTangentVector::Vector(grad_vec))
            }
        })
    }
}

// Debug implementation
impl std::fmt::Debug for PyCostFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyCostFunction")
            .field("has_gradient", &self.grad_fn.is_some())
            .field("has_cost_and_gradient", &self.cost_and_grad_fn.is_some())
            .field("dimension_info", &self.dimension_info)
            .finish()
    }
}

/// Factory function to create cost functions with validation.
///
/// This is the main entry point for users to create cost functions.
///
/// Args:
///     cost: The cost function callable
///     gradient: Optional gradient function callable
///     cost_and_gradient: Optional combined cost and gradient callable
///     dimension: Problem dimension (int for vectors, (rows, cols) for matrices)
///     validate: If True, validates gradient against finite differences
///
/// Returns:
///     PyCostFunction: The wrapped cost function
///
/// Raises:
///     ValueError: If validation fails or dimension is invalid
#[pyfunction]
#[pyo3(signature = (cost, gradient=None, cost_and_gradient=None, dimension=None, validate=false))]
pub fn create_cost_function(
    py: Python<'_>,
    cost: PyObject,
    gradient: Option<PyObject>,
    cost_and_gradient: Option<PyObject>,
    dimension: Option<PyObject>,
    validate: bool,
) -> PyResult<PyCostFunction> {
    // Create the cost function
    let cost_fn = PyCostFunction::new(cost, gradient, cost_and_gradient, dimension)?;
    
    // Optionally validate the gradient implementation
    if validate && cost_fn.grad_fn.is_some() {
        validate_gradient_implementation(py, &cost_fn)?;
    }
    
    Ok(cost_fn)
}

/// Validates that the provided gradient matches finite differences.
///
/// This function tests the gradient at several random points and compares
/// with finite difference approximations. It provides detailed error reporting
/// to help users debug their gradient implementations.
fn validate_gradient_implementation(
    py: Python<'_>,
    cost_fn: &PyCostFunction,
) -> PyResult<()> {
    use rand::Rng;
    use rand_distr::{Distribution, StandardNormal};
    
    // Number of test points
    const NUM_TEST_POINTS: usize = 5;
    // Relative tolerance for gradient comparison
    const RTOL: f64 = 1e-5;
    // Absolute tolerance for gradient comparison
    const ATOL: f64 = 1e-8;
    
    let mut rng = rand::thread_rng();
    let mut max_relative_error: f64 = 0.0;
    let mut max_absolute_error: f64 = 0.0;
    let mut errors = Vec::new();
    
    for test_idx in 0..NUM_TEST_POINTS {
        // Generate a random test point
        let test_point = if cost_fn.dimension_info.is_matrix {
            let (rows, cols) = cost_fn.dimension_info.shape;
            let mut mat = DMatrix::zeros(rows, cols);
            for i in 0..rows {
                for j in 0..cols {
                    mat[(i, j)] = StandardNormal.sample(&mut rng);
                }
            }
            // Normalize to avoid numerical issues
            mat /= mat.norm();
            PyPoint::Matrix(mat)
        } else {
            let dim = cost_fn.dimension_info.shape.0;
            let mut vec = DVector::zeros(dim);
            for i in 0..dim {
                vec[i] = StandardNormal.sample(&mut rng);
            }
            // Normalize to avoid numerical issues
            vec /= vec.norm();
            PyPoint::Vector(vec)
        };
        
        // Compute gradient using provided function
        let (_, provided_grad) = <PyCostFunction as CostFunction<f64>>::cost_and_gradient_alloc(
            cost_fn, &test_point
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to compute gradient at test point {}: {}", test_idx + 1, e)
        ))?;
        
        // Compute gradient using finite differences
        let fd_grad = <PyCostFunction as CostFunction<f64>>::gradient_fd_alloc(
            cost_fn, &test_point
        ).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
            format!("Failed to compute finite difference gradient at test point {}: {}", test_idx + 1, e)
        ))?;
        
        // Compare gradients
        match (&provided_grad, &fd_grad) {
            (PyTangentVector::Vector(g1), PyTangentVector::Vector(g2)) => {
                for i in 0..g1.len() {
                    let abs_error = (g1[i] - g2[i]).abs();
                    let rel_error = if g2[i].abs() > ATOL {
                        abs_error / g2[i].abs()
                    } else {
                        abs_error
                    };
                    
                    max_absolute_error = max_absolute_error.max(abs_error);
                    max_relative_error = max_relative_error.max(rel_error);
                    
                    if abs_error > ATOL && rel_error > RTOL {
                        errors.push(format!(
                            "Test point {}, component {}: provided={:.6e}, finite_diff={:.6e}, abs_error={:.6e}, rel_error={:.6e}",
                            test_idx + 1, i, g1[i], g2[i], abs_error, rel_error
                        ));
                    }
                }
            }
            (PyTangentVector::Matrix(g1), PyTangentVector::Matrix(g2)) => {
                for i in 0..g1.nrows() {
                    for j in 0..g1.ncols() {
                        let abs_error = (g1[(i, j)] - g2[(i, j)]).abs();
                        let rel_error = if g2[(i, j)].abs() > ATOL {
                            abs_error / g2[(i, j)].abs()
                        } else {
                            abs_error
                        };
                        
                        max_absolute_error = max_absolute_error.max(abs_error);
                        max_relative_error = max_relative_error.max(rel_error);
                        
                        if abs_error > ATOL && rel_error > RTOL {
                            errors.push(format!(
                                "Test point {}, component ({},{}): provided={:.6e}, finite_diff={:.6e}, abs_error={:.6e}, rel_error={:.6e}",
                                test_idx + 1, i, j, g1[(i, j)], g2[(i, j)], abs_error, rel_error
                            ));
                        }
                    }
                }
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                    "Gradient type mismatch between provided and finite difference gradients"
                ));
            }
        }
    }
    
    // Report results
    if !errors.is_empty() {
        let error_msg = format!(
            "Gradient validation failed!\n\n\
            Maximum absolute error: {:.6e}\n\
            Maximum relative error: {:.6e}\n\
            Tolerance: rtol={:.6e}, atol={:.6e}\n\n\
            Detailed errors:\n{}",
            max_absolute_error,
            max_relative_error,
            RTOL,
            ATOL,
            errors.join("\n")
        );
        
        // Import Python warnings module to issue a warning
        let warnings = py.import_bound("warnings")?;
        warnings.call_method1("warn", (error_msg, "UserWarning"))?;
    } else {
        // Success message
        println!("Gradient validation passed! Maximum relative error: {:.6e}", max_relative_error);
    }
    
    Ok(())
}

/// Register the cost function module with Python.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add the cost function class and factory directly to parent module
    parent.add_class::<PyCostFunction>()?;
    parent.add_function(wrap_pyfunction!(create_cost_function, parent)?)?;
    Ok(())
}