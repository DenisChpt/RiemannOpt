//! Python cost function interface.
//!
//! This module provides a bridge between Python callable objects and
//! Rust's CostFunction trait.

use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, Dyn};

use riemannopt_core::{
    cost_function::CostFunction,
    error::Result,
    manifold::{Point, TangentVector},
};

/// Python cost function wrapper.
///
/// This struct wraps a Python callable to implement the Rust CostFunction trait.
#[pyclass(name = "CostFunction")]
pub struct PyCostFunction {
    function: PyObject,
    gradient: Option<PyObject>,
}

#[pymethods]
impl PyCostFunction {
    /// Create a new cost function.
    ///
    /// Args:
    ///     function: Callable that takes a numpy array and returns a scalar
    ///     gradient: Optional callable that returns the gradient (numpy array)
    #[new]
    #[pyo3(signature = (function, gradient=None))]
    pub fn new(function: PyObject, gradient: Option<PyObject>) -> Self {
        Self { function, gradient }
    }

    /// Evaluate the cost function at a point.
    ///
    /// Args:
    ///     point: Point to evaluate at (numpy array)
    ///
    /// Returns:
    ///     Cost function value
    pub fn __call__(&self, py: Python<'_>, point: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let point_obj = point.to_object(py);
        let result = self.function.call1(py, (point_obj,))?;
        result.extract::<f64>(py)
    }

    /// Compute the gradient at a point.
    ///
    /// Args:
    ///     point: Point to compute gradient at (numpy array)
    ///
    /// Returns:
    ///     Gradient vector (numpy array)
    pub fn gradient<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if let Some(ref grad_fn) = self.gradient {
            let point_obj = point.to_object(py);
            let result = grad_fn.call1(py, (point_obj,))?;
            result.extract::<Bound<'py, PyArray1<f64>>>(py)
        } else {
            Err(PyValueError::new_err("No gradient function provided"))
        }
    }
}

impl PyCostFunction {
    /// Evaluate both value and gradient at once.
    pub fn value_and_gradient(
        &self,
        py: Python<'_>,
        point: &DVector<f64>,
    ) -> PyResult<(f64, DVector<f64>)> {
        // Convert to numpy array
        let np_point = numpy::PyArray1::from_slice_bound(py, point.as_slice());
        
        // Compute value
        let value = self.function.call1(py, (np_point.to_object(py),))?.extract::<f64>(py)?;
        
        // Compute gradient
        let gradient = if let Some(ref grad_fn) = self.gradient {
            let grad_result = grad_fn.call1(py, (np_point.to_object(py),))?;
            let grad_array = grad_result.extract::<Bound<'_, PyArray1<f64>>>(py)?;
            let grad_readonly = grad_array.readonly();
            let grad_slice = grad_readonly.as_slice()?;
            DVector::from_column_slice(grad_slice)
        } else {
            // Use finite differences if no gradient provided
            self.finite_difference_gradient(py, point)?
        };
        
        Ok((value, gradient))
    }

    /// Compute gradient using finite differences.
    fn finite_difference_gradient(
        &self,
        py: Python<'_>,
        point: &DVector<f64>,
    ) -> PyResult<DVector<f64>> {
        let epsilon = 1e-8;
        let n = point.len();
        let mut gradient = DVector::zeros(n);
        
        for i in 0..n {
            let mut point_plus = point.clone();
            let mut point_minus = point.clone();
            
            point_plus[i] += epsilon;
            point_minus[i] -= epsilon;
            
            let np_plus = numpy::PyArray1::from_slice_bound(py, point_plus.as_slice());
            let np_minus = numpy::PyArray1::from_slice_bound(py, point_minus.as_slice());
            
            let f_plus = self.function.call1(py, (np_plus.to_object(py),))?.extract::<f64>(py)?;
            let f_minus = self.function.call1(py, (np_minus.to_object(py),))?.extract::<f64>(py)?;
            
            gradient[i] = (f_plus - f_minus) / (2.0 * epsilon);
        }
        
        Ok(gradient)
    }
}

/// Create a quadratic cost function for testing.
///
/// Returns a cost function f(x) = 0.5 * ||x||^2 with gradient g(x) = x.
#[pyfunction]
pub fn quadratic_cost(py: Python<'_>) -> PyResult<PyCostFunction> {
    // Define the function
    let code = r#"
def f(x):
    return 0.5 * np.dot(x, x)

def g(x):
    return x.copy()
"#;
    
    // Execute the code to define the functions
    let globals = pyo3::types::PyDict::new_bound(py);
    let numpy = py.import_bound("numpy")?;
    globals.set_item("np", numpy)?;
    
    py.run_bound(code, Some(&globals), None)?;
    
    let f = globals.get_item("f")?.unwrap().to_object(py);
    let g = globals.get_item("g")?.unwrap().to_object(py);
    
    Ok(PyCostFunction::new(f, Some(g)))
}

// Implement Debug trait for PyCostFunction
impl std::fmt::Debug for PyCostFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PyCostFunction")
            .field("has_gradient", &self.gradient.is_some())
            .finish()
    }
}

// Implement CostFunction trait to bridge Python functions with Rust optimization
impl CostFunction<f64, Dyn> for PyCostFunction {
    fn cost(&self, point: &Point<f64, Dyn>) -> Result<f64> {
        Python::with_gil(|py| {
            let np_point = numpy::PyArray1::from_slice_bound(py, point.as_slice());
            let value = self.function.call1(py, (np_point.to_object(py),))
                .map_err(|e| riemannopt_core::error::ManifoldError::numerical_error(
                    format!("Python cost function error: {}", e)
                ))?
                .extract::<f64>(py)
                .map_err(|e| riemannopt_core::error::ManifoldError::numerical_error(
                    format!("Cost function must return a float: {}", e)
                ))?;
            Ok(value)
        })
    }
    
    fn cost_and_gradient(&self, point: &Point<f64, Dyn>) -> Result<(f64, TangentVector<f64, Dyn>)> {
        Python::with_gil(|py| {
            self.value_and_gradient(py, point)
                .map_err(|e| riemannopt_core::error::ManifoldError::numerical_error(
                    format!("Python gradient computation error: {}", e)
                ))
        })
    }
    
    fn gradient(&self, point: &Point<f64, Dyn>) -> Result<TangentVector<f64, Dyn>> {
        Python::with_gil(|py| {
            let np_point = numpy::PyArray1::from_slice_bound(py, point.as_slice());
            
            let gradient = if let Some(ref grad_fn) = self.gradient {
                let grad_result = grad_fn.call1(py, (np_point.to_object(py),))
                    .map_err(|e| riemannopt_core::error::ManifoldError::numerical_error(
                        format!("Python gradient function error: {}", e)
                    ))?;
                let grad_array = grad_result.extract::<Bound<'_, PyArray1<f64>>>(py)
                    .map_err(|e| riemannopt_core::error::ManifoldError::numerical_error(
                        format!("Gradient must return a numpy array: {}", e)
                    ))?;
                let grad_readonly = grad_array.readonly();
                let grad_slice = grad_readonly.as_slice()
                    .map_err(|e| riemannopt_core::error::ManifoldError::numerical_error(
                        format!("Failed to read gradient array: {}", e)
                    ))?;
                DVector::from_column_slice(grad_slice)
            } else {
                self.finite_difference_gradient(py, point)
                    .map_err(|e| riemannopt_core::error::ManifoldError::numerical_error(
                        format!("Finite difference gradient error: {}", e)
                    ))?
            };
            
            Ok(gradient)
        })
    }
}

/// Create a Rosenbrock cost function for testing.
///
/// The Rosenbrock function is a classic non-convex optimization test function.
#[pyfunction]
pub fn rosenbrock_cost(py: Python<'_>) -> PyResult<PyCostFunction> {
    let code = r#"
def f(x):
    n = len(x)
    if n % 2 != 0:
        raise ValueError("Dimension must be even")
    
    value = 0.0
    for i in range(0, n, 2):
        value += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
    return value

def g(x):
    n = len(x)
    grad = np.zeros_like(x)
    
    for i in range(0, n, 2):
        grad[i] = -400 * x[i] * (x[i+1] - x[i]**2) - 2 * (1 - x[i])
        grad[i+1] = 200 * (x[i+1] - x[i]**2)
    
    return grad
"#;
    
    let globals = pyo3::types::PyDict::new_bound(py);
    let numpy = py.import_bound("numpy")?;
    globals.set_item("np", numpy)?;
    
    py.run_bound(code, Some(&globals), None)?;
    
    let f = globals.get_item("f")?.unwrap().to_object(py);
    let g = globals.get_item("g")?.unwrap().to_object(py);
    
    Ok(PyCostFunction::new(f, Some(g)))
}
