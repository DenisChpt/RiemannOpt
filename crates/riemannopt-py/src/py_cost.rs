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

use numpy::PyArrayMethods;
use parking_lot::RwLock;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use riemannopt_core::linalg::{MatrixOps, VectorOps};
use riemannopt_core::{
	cost_function::CostFunction,
	error::{ManifoldError, Result},
};
use std::sync::Arc;

use crate::{
	array_utils::{mat_to_numpy, numpy_to_mat, numpy_to_vec, vec_to_numpy, Mat64, Vec64},
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
	/// Whether to use finite differences for gradient (when no gradient function is provided)
	use_finite_differences: bool,
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
	///     cost_fn: A Python callable that takes a point and returns a scalar cost,
	///              or a tuple (cost, gradient) if grad_fn is None
	///     grad_fn: Optional callable that takes a point and returns the gradient
	///     cost_and_grad_fn: Optional callable that returns (cost, gradient) tuple
	///     dimension: Either an integer (for vector problems) or tuple (rows, cols) for matrices
	///
	/// Note:
	///     - If cost_and_grad_fn is provided, it will be used preferentially for
	///       better performance when both cost and gradient are needed.
	///     - If no gradient function is provided, finite differences will be used automatically.
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
							"dimension must be an integer or (rows, cols) tuple",
						))
					}
				}
				None => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
					"dimension is required",
				)),
			}
		})?;

		// Determine if we should use finite differences
		let use_finite_differences = grad_fn.is_none() && cost_and_grad_fn.is_none();

		Ok(PyCostFunction {
			cost_fn,
			grad_fn,
			cost_and_grad_fn,
			dimension_info,
			eval_count: Arc::new(RwLock::new(EvalCount::default())),
			use_finite_differences,
		})
	}

	/// Get the number of function evaluations.
	///
	/// Returns:
	///     A dictionary with keys 'cost', 'gradient', and 'cost_and_gradient'
	#[getter]
	fn eval_counts(&self, py: Python<'_>) -> PyResult<PyObject> {
		let counts = self.eval_count.read();
		let dict = pyo3::types::PyDict::new(py);
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
					"cost_and_gradient function must return a tuple (cost, gradient)",
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
			let point_mat = numpy_to_mat(point_array.readonly())?;

			// Compute finite difference gradient
			let mut gradient: Mat64 =
				MatrixOps::zeros(self.dimension_info.shape.0, self.dimension_info.shape.1);
			let h = f64::sqrt(f64::EPSILON);

			for i in 0..gradient.nrows() {
				for j in 0..gradient.ncols() {
					let mut point_plus = point_mat.clone();
					let mut point_minus = point_mat.clone();

					*point_plus.get_mut(i, j) += h;
					*point_minus.get_mut(i, j) -= h;

					// Convert back to numpy and evaluate
					let point_plus_py = mat_to_numpy(py, &point_plus)?;
					let point_minus_py = mat_to_numpy(py, &point_minus)?;

					let f_plus = self
						.cost_fn
						.call1(py, (point_plus_py,))?
						.extract::<f64>(py)?;
					let f_minus = self
						.cost_fn
						.call1(py, (point_minus_py,))?
						.extract::<f64>(py)?;

					*gradient.get_mut(i, j) = (f_plus - f_minus) / (2.0 * h);
				}
			}

			Ok(mat_to_numpy(py, &gradient)?.into())
		} else {
			// Vector case
			let point_array = point.downcast_bound::<PyArray1<f64>>(py)?;
			let point_vec = numpy_to_vec(point_array.readonly())?;

			// Compute finite difference gradient
			let mut gradient: Vec64 = VectorOps::zeros(self.dimension_info.shape.0);
			let h = f64::sqrt(f64::EPSILON);

			for i in 0..gradient.len() {
				let mut point_plus = point_vec.clone();
				let mut point_minus = point_vec.clone();

				*point_plus.get_mut(i) += h;
				*point_minus.get_mut(i) -= h;

				// Convert back to numpy and evaluate
				let point_plus_py = vec_to_numpy(py, &point_plus)?;
				let point_minus_py = vec_to_numpy(py, &point_minus)?;

				let f_plus = self
					.cost_fn
					.call1(py, (point_plus_py,))?
					.extract::<f64>(py)?;
				let f_minus = self
					.cost_fn
					.call1(py, (point_minus_py,))?
					.extract::<f64>(py)?;

				*gradient.get_mut(i) = (f_plus - f_minus) / (2.0 * h);
			}

			Ok(vec_to_numpy(py, &gradient)?.into())
		}
	}
}

/// Adapter for using PyCostFunction with Sphere manifold (DVector types).
pub struct PyCostFunctionSphere {
	inner: std::sync::Arc<PyCostFunction>,
}

impl Clone for PyCostFunction {
	fn clone(&self) -> Self {
		Python::with_gil(|py| Self {
			cost_fn: self.cost_fn.clone_ref(py),
			grad_fn: self.grad_fn.as_ref().map(|g| g.clone_ref(py)),
			cost_and_grad_fn: self.cost_and_grad_fn.as_ref().map(|cg| cg.clone_ref(py)),
			dimension_info: self.dimension_info.clone(),
			eval_count: Arc::new(RwLock::new(EvalCount::default())),
			use_finite_differences: self.use_finite_differences,
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
	type Point = Vec64;
	type TangentVector = Vec64;

	fn cost(&self, point: &Self::Point) -> Result<f64> {
		let py_point = PyPoint::Vector(point.clone());
		<PyCostFunction as CostFunction<f64>>::cost(&*self.inner, &py_point)
	}

	fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		let py_point = PyPoint::Vector(point.clone());
		let (cost, grad) = <PyCostFunction as CostFunction<f64>>::cost_and_gradient_alloc(
			&*self.inner,
			&py_point,
		)?;
		match grad {
			PyTangentVector::Vector(v) => Ok((cost, v)),
			_ => Err(ManifoldError::numerical_error("Expected vector gradient")),
		}
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<f64> {
		let py_point = PyPoint::Vector(point.clone());
		let mut py_gradient = PyTangentVector::Vector(gradient.clone());
		let cost = <PyCostFunction as CostFunction<f64>>::cost_and_gradient(
			&*self.inner,
			&py_point,
			&mut py_gradient,
		)?;
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
		match <PyCostFunction as CostFunction<f64>>::hessian_vector_product(
			&*self.inner,
			&py_point,
			&py_vector,
		)? {
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

	fn gradient_fd(&self, point: &Self::Point, gradient: &mut Self::TangentVector) -> Result<()> {
		let py_point = PyPoint::Vector(point.clone());
		let mut py_gradient = PyTangentVector::Vector(gradient.clone());
		<PyCostFunction as CostFunction<f64>>::gradient_fd(
			&*self.inner,
			&py_point,
			&mut py_gradient,
		)?;
		match py_gradient {
			PyTangentVector::Vector(v) => {
				gradient.copy_from(&v);
				Ok(())
			}
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
	type Point = Mat64;
	type TangentVector = Mat64;

	fn cost(&self, point: &Self::Point) -> Result<f64> {
		let py_point = PyPoint::Matrix(point.clone());
		<PyCostFunction as CostFunction<f64>>::cost(&*self.inner, &py_point)
	}

	fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		let py_point = PyPoint::Matrix(point.clone());
		let (cost, grad) = <PyCostFunction as CostFunction<f64>>::cost_and_gradient_alloc(
			&*self.inner,
			&py_point,
		)?;
		match grad {
			PyTangentVector::Matrix(m) => Ok((cost, m)),
			_ => Err(ManifoldError::numerical_error("Expected matrix gradient")),
		}
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<f64> {
		let py_point = PyPoint::Matrix(point.clone());
		let mut py_gradient = PyTangentVector::Matrix(gradient.clone());
		let cost = <PyCostFunction as CostFunction<f64>>::cost_and_gradient(
			&*self.inner,
			&py_point,
			&mut py_gradient,
		)?;
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
		match <PyCostFunction as CostFunction<f64>>::hessian_vector_product(
			&*self.inner,
			&py_point,
			&py_vector,
		)? {
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

	fn gradient_fd(&self, point: &Self::Point, gradient: &mut Self::TangentVector) -> Result<()> {
		let py_point = PyPoint::Matrix(point.clone());
		let mut py_gradient = PyTangentVector::Matrix(gradient.clone());
		<PyCostFunction as CostFunction<f64>>::gradient_fd(
			&*self.inner,
			&py_point,
			&mut py_gradient,
		)?;
		match py_gradient {
			PyTangentVector::Matrix(m) => {
				gradient.copy_from(&m);
				Ok(())
			}
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
		// Release GIL as soon as possible after getting the result
		Python::with_gil(|py| {
			self.eval_count.write().cost += 1;

			// Convert point to numpy array
			let py_point: PyObject = match point {
				PyPoint::Vector(vec) => vec_to_numpy(py, vec)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to convert vector: {}", e))
					})?
					.into(),
				PyPoint::Matrix(mat) => mat_to_numpy(py, mat)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to convert matrix: {}", e))
					})?
					.into(),
			};

			// Call Python function
			let result = self.cost_fn.call1(py, (py_point,)).map_err(|e| {
				ManifoldError::numerical_error(format!("Python cost function error: {}", e))
			})?;

			// Extract result
			result.extract::<f64>(py).map_err(|e| {
				ManifoldError::numerical_error(format!("Failed to extract cost value: {}", e))
			})
		})
	}

	fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		use numpy::PyArrayMethods;
		Python::with_gil(|py| {
			self.eval_count.write().cost_and_gradient += 1;

			// Convert point to numpy
			let py_point: PyObject = match point {
				PyPoint::Vector(vec) => vec_to_numpy(py, vec)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to convert vector: {}", e))
					})?
					.into(),
				PyPoint::Matrix(mat) => mat_to_numpy(py, mat)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to convert matrix: {}", e))
					})?
					.into(),
			};

			// Call the appropriate Python function
			let (cost, grad_py) = if let Some(ref cost_and_grad_fn) = self.cost_and_grad_fn {
				// Use combined function
				let result = cost_and_grad_fn.call1(py, (py_point,)).map_err(|e| {
					ManifoldError::numerical_error(format!("Python cost_and_gradient error: {}", e))
				})?;

				let tuple = result.downcast_bound::<PyTuple>(py).map_err(|_| {
					ManifoldError::numerical_error("cost_and_gradient must return a tuple")
				})?;

				if tuple.len() != 2 {
					return Err(ManifoldError::numerical_error(
						"cost_and_gradient must return exactly 2 values",
					));
				}

				let cost = tuple
					.get_item(0)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to get cost: {}", e))
					})?
					.extract::<f64>()
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to extract cost: {}", e))
					})?;
				let gradient = tuple
					.get_item(1)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to get gradient: {}", e))
					})?
					.into();

				(cost, gradient)
			} else {
				// Compute separately
				let cost = self
					.cost_fn
					.call1(py, (py_point.clone_ref(py),))
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Python cost error: {}", e))
					})?
					.extract::<f64>(py)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to extract cost: {}", e))
					})?;

				let gradient = if let Some(ref grad_fn) = self.grad_fn {
					grad_fn.call1(py, (py_point,)).map_err(|e| {
						ManifoldError::numerical_error(format!("Python gradient error: {}", e))
					})?
				} else {
					// Automatically compute gradient using finite differences
					let grad = self.gradient_fd_alloc(point)?;
					// Convert back to PyObject for consistency with the rest of the code
					match &grad {
						PyTangentVector::Vector(vec) => vec_to_numpy(py, vec)
							.map_err(|e| {
								ManifoldError::numerical_error(format!(
									"Failed to convert gradient: {}",
									e
								))
							})?
							.into(),
						PyTangentVector::Matrix(mat) => mat_to_numpy(py, mat)
							.map_err(|e| {
								ManifoldError::numerical_error(format!(
									"Failed to convert gradient: {}",
									e
								))
							})?
							.into(),
					}
				};

				(cost, gradient)
			};

			// Convert gradient back to Rust type
			let gradient = if self.dimension_info.is_matrix {
				let grad_array =
					grad_py
						.downcast_bound::<numpy::PyArray2<f64>>(py)
						.map_err(|_| {
							ManifoldError::numerical_error(
								"Gradient must be a 2D numpy array for matrix problems",
							)
						})?;
				let grad_mat = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					ManifoldError::numerical_error(format!("Failed to convert gradient: {}", e))
				})?;
				PyTangentVector::Matrix(grad_mat)
			} else {
				let grad_array =
					grad_py
						.downcast_bound::<numpy::PyArray1<f64>>(py)
						.map_err(|_| {
							ManifoldError::numerical_error(
								"Gradient must be a 1D numpy array for vector problems",
							)
						})?;
				let grad_vec = numpy_to_vec(grad_array.readonly()).map_err(|e| {
					ManifoldError::numerical_error(format!("Failed to convert gradient: {}", e))
				})?;
				PyTangentVector::Vector(grad_vec)
			};

			Ok((cost, gradient))
		})
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
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
		point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		// Compute Hessian-vector product using finite differences:
		// H*v ≈ (∇f(x + ε*v) - ∇f(x)) / ε
		// where ε = sqrt(machine epsilon) ≈ 1.49e-8
		let epsilon = 1.4901161193847656e-8_f64; // sqrt(f64::EPSILON)

		// Compute point + ε*v
		let point_perturbed = match (point, vector) {
			(PyPoint::Vector(p), PyTangentVector::Vector(v)) => {
				let mut scaled_v = v.clone();
				scaled_v.scale_mut(epsilon);
				PyPoint::Vector(VectorOps::add(p, &scaled_v))
			}
			(PyPoint::Matrix(p), PyTangentVector::Matrix(v)) => {
				let mut scaled_v = v.clone();
				scaled_v.scale_mut(epsilon);
				PyPoint::Matrix(MatrixOps::add(p, &scaled_v))
			}
			_ => {
				return Err(ManifoldError::numerical_error(
					"Point and vector types must match for Hessian-vector product",
				));
			}
		};

		// Compute ∇f(x + ε*v) using cost_and_gradient_alloc
		let (_, grad_perturbed) = self.cost_and_gradient_alloc(&point_perturbed)?;

		// Compute ∇f(x) using cost_and_gradient_alloc
		let (_, grad_original) = self.cost_and_gradient_alloc(point)?;

		// Compute (∇f(x + ε*v) - ∇f(x)) / ε
		let hv_product = match (grad_perturbed, grad_original) {
			(PyTangentVector::Vector(gp), PyTangentVector::Vector(go)) => {
				let mut diff = VectorOps::sub(&gp, &go);
				diff.scale_mut(epsilon.recip());
				PyTangentVector::Vector(diff)
			}
			(PyTangentVector::Matrix(gp), PyTangentVector::Matrix(go)) => {
				let mut diff = MatrixOps::sub(&gp, &go);
				diff.scale_mut(epsilon.recip());
				PyTangentVector::Matrix(diff)
			}
			_ => {
				return Err(ManifoldError::numerical_error(
					"Gradient type mismatch in Hessian-vector product",
				));
			}
		};

		Ok(hv_product)
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		use numpy::PyArrayMethods;
		Python::with_gil(|py| {
			// Convert point to numpy
			let py_point: PyObject = match point {
				PyPoint::Vector(vec) => vec_to_numpy(py, vec)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to convert vector: {}", e))
					})?
					.into(),
				PyPoint::Matrix(mat) => mat_to_numpy(py, mat)
					.map_err(|e| {
						ManifoldError::numerical_error(format!("Failed to convert matrix: {}", e))
					})?
					.into(),
			};

			// Call the Python finite difference method
			let grad_py = self.gradient_fd_py(py, py_point).map_err(|e| {
				ManifoldError::numerical_error(format!("Finite difference error: {}", e))
			})?;

			// Convert back to Rust type
			if self.dimension_info.is_matrix {
				let grad_array = grad_py
					.downcast_bound::<numpy::PyArray2<f64>>(py)
					.map_err(|_| ManifoldError::numerical_error("Gradient must be a 2D array"))?;
				let grad_mat = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					ManifoldError::numerical_error(format!("Failed to convert gradient: {}", e))
				})?;
				Ok(PyTangentVector::Matrix(grad_mat))
			} else {
				let grad_array = grad_py
					.downcast_bound::<numpy::PyArray1<f64>>(py)
					.map_err(|_| ManifoldError::numerical_error("Gradient must be a 1D array"))?;
				let grad_vec = numpy_to_vec(grad_array.readonly()).map_err(|e| {
					ManifoldError::numerical_error(format!("Failed to convert gradient: {}", e))
				})?;
				Ok(PyTangentVector::Vector(grad_vec))
			}
		})
	}

	fn gradient_fd(&self, point: &Self::Point, gradient: &mut Self::TangentVector) -> Result<()> {
		// For now, use the allocating version
		// TODO: Optimize by using workspace buffers for finite differences
		let grad = self.gradient_fd_alloc(point)?;
		*gradient = grad;
		Ok(())
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
///     cost: The cost function callable. Can be:
///           - A function that returns a scalar cost
///           - A function that returns (cost, gradient) tuple
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
///
/// Note:
///     If no gradient is provided, finite differences will be used automatically.
///     This may be slower but ensures the optimization can proceed.
#[pyfunction]
#[pyo3(signature = (cost, gradient=None, cost_and_gradient=None, dimension=None, validate=false, auto_detect=true))]
pub fn create_cost_function(
	py: Python<'_>,
	cost: PyObject,
	gradient: Option<PyObject>,
	cost_and_gradient: Option<PyObject>,
	dimension: Option<PyObject>,
	validate: bool,
	auto_detect: bool,
) -> PyResult<PyCostFunction> {
	// Try to detect if cost function returns (cost, gradient)
	let (actual_cost, actual_gradient, actual_cost_and_gradient, detected_tuple) = if auto_detect
		&& gradient.is_none()
		&& cost_and_gradient.is_none()
		&& dimension.is_some()
	{
		// Try calling the cost function to see what it returns
		Python::with_gil(|py| {
			// Create a test point based on dimension
			let test_point = match &dimension {
				Some(dim_obj) => {
					if let Ok(dim) = dim_obj.extract::<usize>(py) {
						// Vector case
						vec_to_numpy(py, &VectorOps::zeros(dim)).map(PyObject::from)?
					} else if let Ok((rows, cols)) = dim_obj.extract::<(usize, usize)>(py) {
						// Matrix case
						mat_to_numpy(py, &MatrixOps::zeros(rows, cols)).map(PyObject::from)?
					} else {
						// Can't determine dimension, skip auto-detection
						return Ok::<_, PyErr>((cost, gradient, cost_and_gradient, false));
					}
				}
				None => {
					// No dimension info, skip auto-detection
					return Ok::<_, PyErr>((cost, gradient, cost_and_gradient, false));
				}
			};

			// Try calling the function
			match cost.call1(py, (test_point,)) {
				Ok(result) => {
					// Check if it's a tuple
					if let Ok(tuple) = result.downcast_bound::<pyo3::types::PyTuple>(py) {
						if tuple.len() == 2 {
							// Check if the second element looks like a gradient (array-like)
							if let Ok(_) = tuple
								.get_item(1)
								.unwrap()
								.downcast::<numpy::PyArray1<f64>>()
							{
								// It returns (cost, gradient), use it as cost_and_gradient
								Ok((cost.clone_ref(py), None, Some(cost), true))
							} else if let Ok(_) = tuple
								.get_item(1)
								.unwrap()
								.downcast::<numpy::PyArray2<f64>>()
							{
								// It returns (cost, gradient), use it as cost_and_gradient
								Ok((cost.clone_ref(py), None, Some(cost), true))
							} else {
								Ok((cost, gradient, cost_and_gradient, false))
							}
						} else {
							Ok((cost, gradient, cost_and_gradient, false))
						}
					} else {
						// Returns scalar, keep original
						Ok((cost, gradient, cost_and_gradient, false))
					}
				}
				Err(_) => {
					// Call failed, keep original
					Ok((cost, gradient, cost_and_gradient, false))
				}
			}
		})?
	} else {
		(cost, gradient, cost_and_gradient, false)
	};

	// Create the cost function
	let cost_fn = PyCostFunction::new(
		actual_cost,
		actual_gradient,
		actual_cost_and_gradient,
		dimension,
	)?;

	// Print info message if using finite differences
	if cost_fn.use_finite_differences && !detected_tuple {
		py.run(
            c"print('[RiemannOpt] Note: No gradient function provided. Using finite differences approximation.\\n              For better performance, consider providing a gradient function or returning (cost, gradient) tuple.')",
            None,
            None
        ).ok();
	} else if detected_tuple {
		py.run(
            c"print('[RiemannOpt] Detected that cost function returns (cost, gradient) tuple. Using it for optimization.')",
            None,
            None
        ).ok();
	}

	// Optionally validate the gradient implementation
	if validate && (cost_fn.grad_fn.is_some() || cost_fn.cost_and_grad_fn.is_some()) {
		validate_gradient_implementation(py, &cost_fn)?;
	}

	Ok(cost_fn)
}

/// Validates that the provided gradient matches finite differences.
///
/// This function tests the gradient at several random points and compares
/// with finite difference approximations. It provides detailed error reporting
/// to help users debug their gradient implementations.
fn validate_gradient_implementation(py: Python<'_>, cost_fn: &PyCostFunction) -> PyResult<()> {
	use rand_distr::{Distribution, StandardNormal};

	// Number of test points
	const NUM_TEST_POINTS: usize = 5;
	// Relative tolerance for gradient comparison
	const RTOL: f64 = 1e-5;
	// Absolute tolerance for gradient comparison
	const ATOL: f64 = 1e-8;

	let mut rng = rand::rng();
	let mut max_relative_error: f64 = 0.0;
	let mut max_absolute_error: f64 = 0.0;
	let mut errors = Vec::new();

	for test_idx in 0..NUM_TEST_POINTS {
		// Generate a random test point
		let test_point = if cost_fn.dimension_info.is_matrix {
			let (rows, cols) = cost_fn.dimension_info.shape;
			let mut mat: Mat64 = MatrixOps::zeros(rows, cols);
			for i in 0..rows {
				for j in 0..cols {
					*mat.get_mut(i, j) = StandardNormal.sample(&mut rng);
				}
			}
			// Normalize to avoid numerical issues
			mat /= mat.norm();
			PyPoint::Matrix(mat)
		} else {
			let dim = cost_fn.dimension_info.shape.0;
			let mut vec: Vec64 = VectorOps::zeros(dim);
			for i in 0..dim {
				vec[i] = StandardNormal.sample(&mut rng);
			}
			// Normalize to avoid numerical issues
			vec /= vec.norm();
			PyPoint::Vector(vec)
		};

		// Compute gradient using provided function
		let (_, provided_grad) =
			<PyCostFunction as CostFunction<f64>>::cost_and_gradient_alloc(cost_fn, &test_point)
				.map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to compute gradient at test point {}: {}",
						test_idx + 1,
						e
					))
				})?;

		// Compute gradient using finite differences
		let fd_grad =
			<PyCostFunction as CostFunction<f64>>::gradient_fd_alloc(cost_fn, &test_point)
				.map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to compute finite difference gradient at test point {}: {}",
						test_idx + 1,
						e
					))
				})?;

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
					"Gradient type mismatch between provided and finite difference gradients",
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
		let warnings = py.import("warnings")?;
		warnings.call_method1("warn", (error_msg, "UserWarning"))?;
	} else {
		// Success message
		println!(
			"Gradient validation passed! Maximum relative error: {:.6e}",
			max_relative_error
		);
	}

	Ok(())
}

/// Adapter for generic vector manifolds
pub struct PyCostFunctionVector {
	inner: std::sync::Arc<PyCostFunction>,
}

impl PyCostFunctionVector {
	pub fn new(inner: &PyCostFunction) -> Self {
		Self {
			inner: std::sync::Arc::new(inner.clone()),
		}
	}
}

impl std::fmt::Debug for PyCostFunctionVector {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("PyCostFunctionVector").finish()
	}
}

impl riemannopt_core::cost_function::CostFunction<f64> for PyCostFunctionVector {
	type Point = Vec64;
	type TangentVector = Vec64;

	fn cost(&self, x: &Self::Point) -> Result<f64> {
		Python::with_gil(|py| {
			let inner = &self.inner;
			let x_py = vec_to_numpy(py, x).map(PyObject::from)?;

			if let Some(ref cost_fn) = inner.cost_and_grad_fn {
				let result = cost_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let cost: f64 = tuple.get_item(0)?.extract()?;
				Ok(cost)
			} else {
				let result = inner.cost_fn.call1(py, (x_py,))?;
				let cost: f64 = result.extract(py)?;
				Ok(cost)
			}
		})
		.map_err(|e: PyErr| ManifoldError::numerical_error(format!("Python error in cost: {}", e)))
	}

	fn gradient(&self, x: &Self::Point) -> Result<Self::TangentVector> {
		Python::with_gil(|py| -> PyResult<Vec64> {
			let inner = &self.inner;
			let x_py = vec_to_numpy(py, x).map(PyObject::from)?;

			if let Some(ref grad_fn) = inner.grad_fn {
				let result = grad_fn.call1(py, (x_py,))?;
				let grad_array = result.downcast_bound::<numpy::PyArray1<f64>>(py)?;
				let grad_vec = numpy_to_vec(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok(grad_vec)
			} else if let Some(ref cost_and_grad_fn) = inner.cost_and_grad_fn {
				let result = cost_and_grad_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let grad_obj = tuple.get_item(1)?;
				let grad_array = grad_obj.downcast::<numpy::PyArray1<f64>>()?;
				let grad_vec = numpy_to_vec(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok(grad_vec)
			} else {
				// For finite differences, we need to handle ManifoldError -> PyErr conversion
				self.gradient_fd_alloc(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Finite difference gradient failed: {}",
						e
					))
				})
			}
		})
		.map_err(|e: PyErr| {
			ManifoldError::numerical_error(format!("Python error in gradient: {}", e))
		})
	}

	fn cost_and_gradient_alloc(&self, x: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		Python::with_gil(|py| -> PyResult<(f64, Vec64)> {
			let inner = &self.inner;
			let x_py = vec_to_numpy(py, x).map(PyObject::from)?;

			if let Some(ref cost_and_grad_fn) = inner.cost_and_grad_fn {
				let result = cost_and_grad_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let cost: f64 = tuple.get_item(0)?.extract()?;
				let grad_obj = tuple.get_item(1)?;
				let grad_array = grad_obj.downcast::<numpy::PyArray1<f64>>()?;
				let grad_vec = numpy_to_vec(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok((cost, grad_vec))
			} else {
				// Need to handle errors when calling self.cost and self.gradient
				let cost = self.cost(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Cost computation failed: {}",
						e
					))
				})?;
				let grad = self.gradient(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Gradient computation failed: {}",
						e
					))
				})?;
				Ok((cost, grad))
			}
		})
		.map_err(|e: PyErr| {
			ManifoldError::numerical_error(format!("Python error in cost_and_gradient: {}", e))
		})
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(cost)
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		let n = point.len();
		let mut gradient: Vec64 = VectorOps::zeros(n);
		let h = f64::sqrt(f64::EPSILON);

		for i in 0..n {
			let mut point_plus = point.clone();
			let mut point_minus = point.clone();

			point_plus[i] += h;
			point_minus[i] -= h;

			let f_plus = self.cost(&point_plus)?;
			let f_minus = self.cost(&point_minus)?;

			gradient[i] = (f_plus - f_minus) / (2.0 * h);
		}

		Ok(gradient)
	}

	fn gradient_fd(&self, point: &Self::Point, gradient: &mut Self::TangentVector) -> Result<()> {
		let grad = self.gradient_fd_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(())
	}

	fn hessian_vector_product(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		// Compute Hessian-vector product using finite differences:
		// H*v ≈ (∇f(x + ε*v) - ∇f(x)) / ε
		let epsilon = 1.4901161193847656e-8_f64; // sqrt(f64::EPSILON)

		// Compute point + ε*v
		let point_perturbed = point + vector * epsilon;

		// Compute ∇f(x + ε*v) using cost_and_gradient_alloc
		let (_, grad_perturbed) = self.cost_and_gradient_alloc(&point_perturbed)?;

		// Compute ∇f(x) using cost_and_gradient_alloc
		let (_, grad_original) = self.cost_and_gradient_alloc(point)?;

		// Compute (∇f(x + ε*v) - ∇f(x)) / ε
		let hv_product = (grad_perturbed - grad_original) / epsilon;

		Ok(hv_product)
	}
}

/// Adapter for generic matrix manifolds
pub struct PyCostFunctionMatrix {
	inner: std::sync::Arc<PyCostFunction>,
}

impl PyCostFunctionMatrix {
	pub fn new(inner: &PyCostFunction) -> Self {
		Self {
			inner: std::sync::Arc::new(inner.clone()),
		}
	}
}

impl std::fmt::Debug for PyCostFunctionMatrix {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("PyCostFunctionMatrix").finish()
	}
}

impl riemannopt_core::cost_function::CostFunction<f64> for PyCostFunctionMatrix {
	type Point = Mat64;
	type TangentVector = Mat64;

	fn cost(&self, x: &Self::Point) -> Result<f64> {
		Python::with_gil(|py| {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, x).map(PyObject::from)?;

			if let Some(ref cost_fn) = inner.cost_and_grad_fn {
				let result = cost_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let cost: f64 = tuple.get_item(0)?.extract()?;
				Ok(cost)
			} else {
				let result = inner.cost_fn.call1(py, (x_py,))?;
				let cost: f64 = result.extract(py)?;
				Ok(cost)
			}
		})
		.map_err(|e: PyErr| ManifoldError::numerical_error(format!("Python error in cost: {}", e)))
	}

	fn gradient(&self, x: &Self::Point) -> Result<Self::TangentVector> {
		Python::with_gil(|py| -> PyResult<Mat64> {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, x).map(PyObject::from)?;

			if let Some(ref grad_fn) = inner.grad_fn {
				let result = grad_fn.call1(py, (x_py,))?;
				let grad_array = result.downcast_bound::<numpy::PyArray2<f64>>(py)?;
				let grad_mat = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok(grad_mat)
			} else if let Some(ref cost_and_grad_fn) = inner.cost_and_grad_fn {
				let result = cost_and_grad_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let grad_obj = tuple.get_item(1)?;
				let grad_array = grad_obj.downcast::<numpy::PyArray2<f64>>()?;
				let grad_mat = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok(grad_mat)
			} else {
				self.gradient_fd_alloc(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Finite difference gradient failed: {}",
						e
					))
				})
			}
		})
		.map_err(|e: PyErr| {
			ManifoldError::numerical_error(format!("Python error in gradient: {}", e))
		})
	}

	fn cost_and_gradient_alloc(&self, x: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		Python::with_gil(|py| -> PyResult<(f64, Mat64)> {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, x).map(PyObject::from)?;

			if let Some(ref cost_and_grad_fn) = inner.cost_and_grad_fn {
				let result = cost_and_grad_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let cost: f64 = tuple.get_item(0)?.extract()?;
				let grad_obj = tuple.get_item(1)?;
				let grad_array = grad_obj.downcast::<numpy::PyArray2<f64>>()?;
				let grad_mat = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok((cost, grad_mat))
			} else {
				let cost = self.cost(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Cost computation failed: {}",
						e
					))
				})?;
				let grad = self.gradient(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Gradient computation failed: {}",
						e
					))
				})?;
				Ok((cost, grad))
			}
		})
		.map_err(|e: PyErr| {
			ManifoldError::numerical_error(format!("Python error in cost_and_gradient: {}", e))
		})
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(cost)
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		let (m, n) = (point.nrows(), point.ncols());
		let mut gradient: Mat64 = MatrixOps::zeros(m, n);
		let h = f64::sqrt(f64::EPSILON);

		for i in 0..m {
			for j in 0..n {
				let mut point_plus = point.clone();
				let mut point_minus = point.clone();

				point_plus[(i, j)] += h;
				point_minus[(i, j)] -= h;

				let f_plus = self.cost(&point_plus)?;
				let f_minus = self.cost(&point_minus)?;

				gradient[(i, j)] = (f_plus - f_minus) / (2.0 * h);
			}
		}

		Ok(gradient)
	}

	fn gradient_fd(&self, point: &Self::Point, gradient: &mut Self::TangentVector) -> Result<()> {
		let grad = self.gradient_fd_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(())
	}

	fn hessian_vector_product(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		// Compute Hessian-vector product using finite differences:
		// H*v ≈ (∇f(x + ε*v) - ∇f(x)) / ε
		let epsilon = 1.4901161193847656e-8_f64; // sqrt(f64::EPSILON)

		// Compute point + ε*v
		let point_perturbed = point + vector * epsilon;

		// Compute ∇f(x + ε*v) using cost_and_gradient_alloc
		let (_, grad_perturbed) = self.cost_and_gradient_alloc(&point_perturbed)?;

		// Compute ∇f(x) using cost_and_gradient_alloc
		let (_, grad_original) = self.cost_and_gradient_alloc(point)?;

		// Compute (∇f(x + ε*v) - ∇f(x)) / ε
		let hv_product = (grad_perturbed - grad_original) / epsilon;

		Ok(hv_product)
	}
}

/// Cost function wrapper for PSD Cone manifold.
///
/// This wrapper handles the conversion between the matrix representation
/// used in Python and the vectorized representation used internally by
/// the PSDCone manifold.
pub struct PyCostFunctionPSDCone {
	inner: std::sync::Arc<PyCostFunction>,
	n: usize,
}

impl PyCostFunctionPSDCone {
	pub fn new(inner: &PyCostFunction, n: usize) -> Self {
		Self {
			inner: std::sync::Arc::new(inner.clone()),
			n,
		}
	}

	/// Convert a vectorized symmetric matrix to a full matrix
	fn vector_to_matrix(&self, vec: &Vec64) -> Mat64 {
		let mut mat: Mat64 = MatrixOps::zeros(self.n, self.n);
		let mut idx = 0;

		// Fill upper triangular part
		for i in 0..self.n {
			for j in i..self.n {
				if i == j {
					*mat.get_mut(i, j) = VectorOps::get(vec, idx);
				} else {
					let val = VectorOps::get(vec, idx) / f64::sqrt(2.0);
					*mat.get_mut(i, j) = val;
					*mat.get_mut(j, i) = val;
				}
				idx += 1;
			}
		}

		mat
	}

	/// Convert a full matrix to vectorized symmetric matrix representation
	fn matrix_to_vector(&self, mat: &Mat64) -> Vec64 {
		let vec_dim = self.n * (self.n + 1) / 2;
		let mut vec: Vec64 = VectorOps::zeros(vec_dim);
		let mut idx = 0;

		// Extract upper triangular part
		for i in 0..self.n {
			for j in i..self.n {
				if i == j {
					*vec.get_mut(idx) = MatrixOps::get(mat, i, j);
				} else {
					*vec.get_mut(idx) = MatrixOps::get(mat, i, j) * f64::sqrt(2.0);
				}
				idx += 1;
			}
		}

		vec
	}
}

impl std::fmt::Debug for PyCostFunctionPSDCone {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("PyCostFunctionPSDCone").finish()
	}
}

impl riemannopt_core::cost_function::CostFunction<f64> for PyCostFunctionPSDCone {
	type Point = Vec64;
	type TangentVector = Vec64;

	fn cost(&self, x: &Self::Point) -> Result<f64> {
		// Convert vector to matrix for Python
		let x_mat = self.vector_to_matrix(x);

		Python::with_gil(|py| {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, &x_mat).map(PyObject::from)?;
			inner.cost_fn.call1(py, (x_py,))?.extract(py)
		})
		.map_err(|e: PyErr| ManifoldError::numerical_error(format!("Python error in cost: {}", e)))
	}

	fn gradient(&self, x: &Self::Point) -> Result<Self::TangentVector> {
		// Convert vector to matrix for Python
		let x_mat = self.vector_to_matrix(x);

		Python::with_gil(|py| -> PyResult<Vec64> {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, &x_mat).map(PyObject::from)?;

			if let Some(ref grad_fn) = inner.grad_fn {
				let grad_obj = grad_fn.call1(py, (x_py,))?;
				let grad_array = grad_obj.downcast_bound::<numpy::PyArray2<f64>>(py)?;
				let grad_mat = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok(self.matrix_to_vector(&grad_mat))
			} else if let Some(ref cost_and_grad_fn) = inner.cost_and_grad_fn {
				let result = cost_and_grad_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let grad_obj = tuple.get_item(1)?;
				let grad_array = grad_obj.downcast::<numpy::PyArray2<f64>>()?;
				let grad_mat = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok(self.matrix_to_vector(&grad_mat))
			} else {
				self.gradient_fd_alloc(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Finite difference gradient failed: {}",
						e
					))
				})
			}
		})
		.map_err(|e: PyErr| {
			ManifoldError::numerical_error(format!("Python error in gradient: {}", e))
		})
	}

	fn cost_and_gradient_alloc(&self, x: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		// Convert vector to matrix for Python
		let x_mat = self.vector_to_matrix(x);

		Python::with_gil(|py| -> PyResult<(f64, Vec64)> {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, &x_mat).map(PyObject::from)?;

			if let Some(ref cost_and_grad_fn) = inner.cost_and_grad_fn {
				let result = cost_and_grad_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let cost: f64 = tuple.get_item(0)?.extract()?;
				let grad_obj = tuple.get_item(1)?;
				let grad_array = grad_obj.downcast::<numpy::PyArray2<f64>>()?;
				let grad_mat = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {}",
						e
					))
				})?;
				Ok((cost, self.matrix_to_vector(&grad_mat)))
			} else {
				let cost = self.cost(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Cost computation failed: {}",
						e
					))
				})?;
				let grad = self.gradient(x).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Gradient computation failed: {}",
						e
					))
				})?;
				Ok((cost, grad))
			}
		})
		.map_err(|e: PyErr| {
			ManifoldError::numerical_error(format!("Python error in cost_and_gradient: {}", e))
		})
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(cost)
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		let vec_dim = self.n * (self.n + 1) / 2;
		let mut gradient: Vec64 = VectorOps::zeros(vec_dim);
		let h = f64::sqrt(f64::EPSILON);

		for i in 0..vec_dim {
			let mut point_plus = point.clone();
			let mut point_minus = point.clone();

			point_plus[i] += h;
			point_minus[i] -= h;

			let f_plus = self.cost(&point_plus)?;
			let f_minus = self.cost(&point_minus)?;

			gradient[i] = (f_plus - f_minus) / (2.0 * h);
		}

		Ok(gradient)
	}

	fn gradient_fd(&self, point: &Self::Point, gradient: &mut Self::TangentVector) -> Result<()> {
		let grad = self.gradient_fd_alloc(point)?;
		gradient.copy_from(&grad);
		Ok(())
	}

	fn hessian_vector_product(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		// Compute Hessian-vector product using finite differences:
		// H*v ≈ (∇f(x + ε*v) - ∇f(x)) / ε
		let epsilon = 1.4901161193847656e-8_f64; // sqrt(f64::EPSILON)

		// Compute point + ε*v
		let point_perturbed = point + vector * epsilon;

		// Compute ∇f(x + ε*v) using cost_and_gradient_alloc
		let (_, grad_perturbed) = self.cost_and_gradient_alloc(&point_perturbed)?;

		// Compute ∇f(x) using cost_and_gradient_alloc
		let (_, grad_original) = self.cost_and_gradient_alloc(point)?;

		// Compute (∇f(x + ε*v) - ∇f(x)) / ε
		let hv_product = (grad_perturbed - grad_original) / epsilon;

		Ok(hv_product)
	}
}

/// Cost function adapter for the FixedRank manifold.
///
/// Python callbacks receive and return m×n matrices. Internally, this adapter
/// converts between `FixedRankPoint<f64>` / `FixedRankTangent<f64>` and full
/// matrices via SVD factorization and `FixedRankTangent::from_ambient`.
pub struct PyCostFunctionFixedRank {
	inner: std::sync::Arc<PyCostFunction>,
}

impl PyCostFunctionFixedRank {
	pub fn new(inner: &PyCostFunction) -> Self {
		Self {
			inner: std::sync::Arc::new(inner.clone()),
		}
	}

	/// Convert a FixedRankPoint to a full m×n matrix for Python
	fn point_to_matrix(point: &riemannopt_manifolds::fixed_rank::FixedRankPoint<f64>) -> Mat64 {
		point.to_matrix()
	}

	/// Convert an ambient (euclidean) gradient matrix to a FixedRankTangent
	fn gradient_to_tangent(
		point: &riemannopt_manifolds::fixed_rank::FixedRankPoint<f64>,
		grad_mat: &Mat64,
	) -> riemannopt_manifolds::fixed_rank::FixedRankTangent<f64> {
		riemannopt_manifolds::fixed_rank::FixedRankTangent::from_ambient(point, grad_mat)
	}
}

impl std::fmt::Debug for PyCostFunctionFixedRank {
	fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
		f.debug_struct("PyCostFunctionFixedRank").finish()
	}
}

impl riemannopt_core::cost_function::CostFunction<f64> for PyCostFunctionFixedRank {
	type Point = riemannopt_manifolds::fixed_rank::FixedRankPoint<f64>;
	type TangentVector = riemannopt_manifolds::fixed_rank::FixedRankTangent<f64>;

	fn cost(&self, point: &Self::Point) -> Result<f64> {
		let mat = Self::point_to_matrix(point);
		Python::with_gil(|py| {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, &mat).map(PyObject::from)?;

			if let Some(ref cost_fn) = inner.cost_and_grad_fn {
				let result = cost_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let cost: f64 = tuple.get_item(0)?.extract()?;
				Ok(cost)
			} else {
				let result = inner.cost_fn.call1(py, (x_py,))?;
				let cost: f64 = result.extract(py)?;
				Ok(cost)
			}
		})
		.map_err(|e: PyErr| ManifoldError::numerical_error(format!("Python error in cost: {}", e)))
	}

	fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		let mat = Self::point_to_matrix(point);
		let grad_mat: Mat64 = Python::with_gil(|py| -> PyResult<Mat64> {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, &mat).map(PyObject::from)?;

			if let Some(ref grad_fn) = inner.grad_fn {
				let result = grad_fn.call1(py, (x_py,))?;
				let grad_array = result.downcast_bound::<numpy::PyArray2<f64>>(py)?;
				numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {e}"
					))
				})
			} else if let Some(ref cost_and_grad_fn) = inner.cost_and_grad_fn {
				let result = cost_and_grad_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let grad_obj = tuple.get_item(1)?;
				let grad_array = grad_obj.downcast::<numpy::PyArray2<f64>>()?;
				numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {e}"
					))
				})
			} else {
				Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
					"FixedRank requires an explicit gradient function",
				))
			}
		})
		.map_err(|e: PyErr| {
			ManifoldError::numerical_error(format!("Python error in gradient: {e}"))
		})?;

		Ok(Self::gradient_to_tangent(point, &grad_mat))
	}

	fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(f64, Self::TangentVector)> {
		let mat = Self::point_to_matrix(point);
		let (cost, grad_mat): (f64, Mat64) = Python::with_gil(|py| -> PyResult<(f64, Mat64)> {
			let inner = &self.inner;
			let x_py = mat_to_numpy(py, &mat).map(PyObject::from)?;

			if let Some(ref cost_and_grad_fn) = inner.cost_and_grad_fn {
				let result = cost_and_grad_fn.call1(py, (x_py,))?;
				let tuple = result.downcast_bound::<pyo3::types::PyTuple>(py)?;
				let c: f64 = tuple.get_item(0)?.extract()?;
				let grad_obj = tuple.get_item(1)?;
				let grad_array = grad_obj.downcast::<numpy::PyArray2<f64>>()?;
				let g = numpy_to_mat(grad_array.readonly()).map_err(|e| {
					PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
						"Failed to convert gradient: {e}"
					))
				})?;
				Ok((c, g))
			} else {
				let cost_result = inner.cost_fn.call1(py, (x_py.clone_ref(py),))?;
				let c: f64 = cost_result.extract(py)?;

				if let Some(ref grad_fn) = inner.grad_fn {
					let grad_result = grad_fn.call1(py, (x_py,))?;
					let grad_array = grad_result.downcast_bound::<numpy::PyArray2<f64>>(py)?;
					let g = numpy_to_mat(grad_array.readonly()).map_err(|e| {
						PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(format!(
							"Failed to convert gradient: {e}"
						))
					})?;
					Ok((c, g))
				} else {
					Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
						"FixedRank requires an explicit gradient function",
					))
				}
			}
		})
		.map_err(|e: PyErr| {
			ManifoldError::numerical_error(format!("Python error in cost_and_gradient: {e}"))
		})?;

		Ok((cost, Self::gradient_to_tangent(point, &grad_mat)))
	}

	fn cost_and_gradient(
		&self,
		point: &Self::Point,
		gradient: &mut Self::TangentVector,
	) -> Result<f64> {
		let (cost, grad) = self.cost_and_gradient_alloc(point)?;
		gradient.u_perp_m = grad.u_perp_m;
		gradient.s_dot = grad.s_dot;
		gradient.v_perp_n = grad.v_perp_n;
		Ok(cost)
	}

	fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
		// Finite differences in ambient space, projected back to tangent space
		let mat = Self::point_to_matrix(point);
		let (m, n) = (MatrixOps::nrows(&mat), MatrixOps::ncols(&mat));
		let h = f64::sqrt(f64::EPSILON);
		let mut grad_mat: Mat64 = MatrixOps::zeros(m, n);

		for i in 0..m {
			for j in 0..n {
				let mut plus = mat.clone();
				let mut minus = mat.clone();
				*MatrixOps::get_mut(&mut plus, i, j) += h;
				*MatrixOps::get_mut(&mut minus, i, j) -= h;

				// Project perturbed matrices back to rank-k and evaluate cost
				let k = VectorOps::len(&point.s);
				let p_plus =
					riemannopt_manifolds::fixed_rank::FixedRankPoint::from_matrix(&plus, k)?;
				let p_minus =
					riemannopt_manifolds::fixed_rank::FixedRankPoint::from_matrix(&minus, k)?;

				let f_plus = self.cost(&p_plus)?;
				let f_minus = self.cost(&p_minus)?;

				*MatrixOps::get_mut(&mut grad_mat, i, j) = (f_plus - f_minus) / (2.0 * h);
			}
		}

		Ok(Self::gradient_to_tangent(point, &grad_mat))
	}

	fn hessian_vector_product(
		&self,
		point: &Self::Point,
		vector: &Self::TangentVector,
	) -> Result<Self::TangentVector> {
		// Finite difference: H*v ≈ (∇f(x + εξ) - ∇f(x)) / ε
		// where x + εξ is computed via retraction-like perturbation in ambient space
		let epsilon = 1.4901161193847656e-8_f64;

		let mat = Self::point_to_matrix(point);
		let tangent_mat = vector.to_matrix(point);
		let perturbed = MatrixOps::add(&mat, &(tangent_mat * epsilon));

		// Truncate perturbed matrix back to rank k via SVD
		let k = VectorOps::len(&point.s);
		let perturbed_point =
			riemannopt_manifolds::fixed_rank::FixedRankPoint::from_matrix(&perturbed, k)?;

		let grad_perturbed = self.gradient(&perturbed_point)?;
		let grad_original = self.gradient(point)?;

		// Compute (grad_perturbed - grad_original) / epsilon in tangent components
		let u_perp_m = (grad_perturbed.u_perp_m - &grad_original.u_perp_m) / epsilon;
		let s_dot = <Vec64 as VectorOps<f64>>::from_fn(VectorOps::len(&grad_original.s_dot), |i| {
			(VectorOps::get(&grad_perturbed.s_dot, i) - VectorOps::get(&grad_original.s_dot, i))
				/ epsilon
		});
		let v_perp_n = (grad_perturbed.v_perp_n - &grad_original.v_perp_n) / epsilon;

		Ok(riemannopt_manifolds::fixed_rank::FixedRankTangent::new(
			u_perp_m, s_dot, v_perp_n,
		))
	}
}

/// Register the cost function module with Python.
pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
	use pyo3::wrap_pyfunction;

	// Add the cost function class and factory directly to parent module
	parent.add_class::<PyCostFunction>()?;
	parent.add_function(wrap_pyfunction!(create_cost_function, parent)?)?;
	Ok(())
}
