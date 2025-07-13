//! Python wrapper for the Euclidean manifold.
//!
//! The Euclidean manifold R^n is the standard n-dimensional Euclidean space
//! with the usual inner product and flat metric. This is the simplest manifold
//! and serves as a baseline for optimization algorithms.

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};
use nalgebra::{DVector, DMatrix};
use riemannopt_manifolds::euclidean::Euclidean;
use riemannopt_core::manifold::Manifold;

use crate::{
    array_utils::{numpy_to_dvector, dvector_to_numpy},
    error::{to_py_err, dimension_mismatch},
    types::PyPoint,
};
use super::base::{PyManifoldBase, PointType};

/// The Euclidean manifold R^n.
///
/// The Euclidean manifold is the standard n-dimensional vector space with
/// the usual Euclidean metric. All operations are trivial:
/// - Projection is the identity
/// - Exponential map is addition
/// - Logarithmic map is subtraction
/// - Parallel transport is the identity
///
/// This manifold is useful as a baseline and for unconstrained optimization
/// problems that can benefit from the Riemannian optimization framework.
///
/// Parameters
/// ----------
/// n : int
///     Dimension of the space
///
/// Attributes
/// ----------
/// n : int
///     Dimension of the space
/// dim : int
///     Intrinsic dimension (equals n)
/// ambient_dim : int
///     Dimension of the ambient space (equals n)
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> # Create 10-dimensional Euclidean space
/// >>> euclidean = ro.manifolds.Euclidean(10)
/// >>> print(f"Dimension: {euclidean.dim}")
/// Dimension: 10
/// >>>
/// >>> # All manifold operations are trivial
/// >>> x = euclidean.random_point()
/// >>> v = euclidean.random_tangent(x)
/// >>> 
/// >>> # Exponential map is just addition
/// >>> y = euclidean.exp(x, v)
/// >>> assert np.allclose(y, x + v)
#[pyclass(name = "Euclidean", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyEuclidean {
    pub(crate) inner: Euclidean<f64>,
    n: usize,
}

impl PyManifoldBase for PyEuclidean {
    fn manifold_name(&self) -> &'static str {
        "Euclidean"
    }
    
    fn ambient_dim(&self) -> usize {
        self.n
    }
    
    fn intrinsic_dim(&self) -> usize {
        self.n
    }
    
    fn point_type(&self) -> PointType {
        PointType::Vector(self.n)
    }
}

#[pymethods]
impl PyEuclidean {
    /// Create a new Euclidean manifold.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Dimension of the space (must be > 0)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If n <= 0
    #[new]
    pub fn new(n: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n must be positive"
            ));
        }
        
        let inner = Euclidean::<f64>::new(n)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyEuclidean { inner, n })
    }
    
    /// String representation of the manifold.
    fn __repr__(&self) -> String {
        format!(
            "{}(ambient_dim={}, intrinsic_dim={})",
            self.manifold_name(),
            self.ambient_dim(),
            self.intrinsic_dim()
        )
    }
    
    /// Get the intrinsic dimension of the manifold.
    #[getter]
    fn dim(&self) -> usize {
        self.intrinsic_dim()
    }
    
    /// Get the ambient dimension of the manifold.
    #[getter]
    fn ambient_dim(&self) -> usize {
        PyManifoldBase::ambient_dim(self)
    }
    
    /// Get the dimension n.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    
    /// Check if a point lies on the manifold.
    ///
    /// For Euclidean space, all finite vectors are valid points.
    ///
    /// Args:
    ///     point: Point to check
    ///     atol: Absolute tolerance (default: 1e-10)
    ///
    /// Returns:
    ///     bool: True if point is valid (always true for finite vectors)
    #[pyo3(signature = (point, atol=1e-10))]
    fn contains(&self, py: Python<'_>, point: PyObject, atol: f64) -> PyResult<bool> {
        let point = self.parse_point(py, point)?;
        self.validate_point_shape(&point)?;
        
        match &point {
            PyPoint::Vector(vec) => Ok(vec.iter().all(|x| x.is_finite())),
            _ => Err(crate::error::type_error(
                "vector point",
                "non-vector point",
            )),
        }
    }
    
    /// Check if a vector is in the tangent space.
    ///
    /// For Euclidean space, the tangent space at any point is the whole space.
    ///
    /// Args:
    ///     point: Base point on the manifold
    ///     vector: Vector to check
    ///     atol: Absolute tolerance (default: 1e-10)
    ///
    /// Returns:
    ///     bool: True if vector is valid (always true for finite vectors)
    #[pyo3(signature = (point, vector, atol=1e-10))]
    fn is_tangent(&self, py: Python<'_>, point: PyObject, vector: PyObject, atol: f64) -> PyResult<bool> {
        let point = self.parse_point(py, point)?;
        let vector = self.parse_point(py, vector)?;
        self.validate_point_shape(&point)?;
        self.validate_point_shape(&vector)?;
        
        match (&point, &vector) {
            (PyPoint::Vector(_), PyPoint::Vector(v)) => Ok(v.iter().all(|x| x.is_finite())),
            _ => Err(crate::error::type_error(
                "matching point and vector types",
                "mismatched types",
            )),
        }
    }
    
    /// Project a point onto the manifold (identity operation).
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n,)
    ///     Point to project
    ///
    /// Returns
    /// -------
    /// array_like, shape (n,)
    ///     Same point (identity operation)
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        
        // Validate dimension
        if point_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[point_vec.len()],
            ));
        }
        
        // For Euclidean space, projection is identity
        dvector_to_numpy(py, &point_vec)
    }
    
    /// Exponential map (addition in Euclidean space).
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n,)
    ///     Point on the manifold
    /// tangent : array_like, shape (n,)
    ///     Tangent vector at the point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n,)
    ///     Result of exp_point(tangent) = point + tangent
    pub fn exp<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if point_vec.len() != self.n || tangent_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[point_vec.len()],
            ));
        }
        
        // Exponential map is just addition
        let result = &point_vec + &tangent_vec;
        dvector_to_numpy(py, &result)
    }
    
    /// Logarithmic map (subtraction in Euclidean space).
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n,)
    ///     Starting point
    /// other : array_like, shape (n,)
    ///     Target point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n,)
    ///     Tangent vector log_point(other) = other - point
    pub fn log<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, other: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let other_vec = numpy_to_dvector(other)?;
        
        // Validate dimensions
        if point_vec.len() != self.n || other_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[point_vec.len()],
            ));
        }
        
        // Logarithmic map is just subtraction
        let result = &other_vec - &point_vec;
        dvector_to_numpy(py, &result)
    }
    
    /// Retraction (same as exponential map for Euclidean space).
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n,)
    ///     Point on the manifold
    /// tangent : array_like, shape (n,)
    ///     Tangent vector at the point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n,)
    ///     Retracted point = point + tangent
    pub fn retract<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // For Euclidean space, retraction is the same as exponential map
        self.exp(py, point, tangent)
    }
    
    /// Project a vector onto the tangent space (identity operation).
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n,)
    ///     Point on the manifold
    /// vector : array_like, shape (n,)
    ///     Vector to project
    ///
    /// Returns
    /// -------
    /// array_like, shape (n,)
    ///     Same vector (identity operation)
    pub fn project_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, vector: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let vector_vec = numpy_to_dvector(vector)?;
        
        // Validate dimensions
        if point_vec.len() != self.n || vector_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[point_vec.len()],
            ));
        }
        
        // For Euclidean space, tangent projection is identity
        dvector_to_numpy(py, &vector_vec)
    }
    
    /// Riemannian inner product (standard dot product).
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n,)
    ///     Point on the manifold (unused for Euclidean)
    /// u : array_like, shape (n,)
    ///     First tangent vector
    /// v : array_like, shape (n,)
    ///     Second tangent vector
    ///
    /// Returns
    /// -------
    /// float
    ///     Inner product <u, v> = u.T @ v
    pub fn inner(&self, _py: Python<'_>, point: PyReadonlyArray1<'_, f64>, u: PyReadonlyArray1<'_, f64>, v: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let point_vec = numpy_to_dvector(point)?;
        let u_vec = numpy_to_dvector(u)?;
        let v_vec = numpy_to_dvector(v)?;
        
        // Validate dimensions
        if point_vec.len() != self.n || u_vec.len() != self.n || v_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[point_vec.len()],
            ));
        }
        
        // Standard dot product
        Ok(u_vec.dot(&v_vec))
    }
    
    /// Riemannian norm (standard Euclidean norm).
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n,)
    ///     Point on the manifold (unused for Euclidean)
    /// tangent : array_like, shape (n,)
    ///     Tangent vector
    ///
    /// Returns
    /// -------
    /// float
    ///     Norm ||tangent|| = sqrt(tangent.T @ tangent)
    pub fn norm(&self, _py: Python<'_>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let point_vec = numpy_to_dvector(point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if point_vec.len() != self.n || tangent_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[point_vec.len()],
            ));
        }
        
        // Standard Euclidean norm
        Ok(tangent_vec.norm())
    }
    
    /// Geodesic distance (Euclidean distance).
    ///
    /// Parameters
    /// ----------
    /// x : array_like, shape (n,)
    ///     First point
    /// y : array_like, shape (n,)
    ///     Second point
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance ||x - y||
    pub fn distance(&self, _py: Python<'_>, x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let x_vec = numpy_to_dvector(x)?;
        let y_vec = numpy_to_dvector(y)?;
        
        // Validate dimensions
        if x_vec.len() != self.n || y_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[x_vec.len()],
            ));
        }
        
        // Euclidean distance
        Ok((&x_vec - &y_vec).norm())
    }
    
    /// Generate a random point.
    ///
    /// Returns
    /// -------
    /// array_like, shape (n,)
    ///     Random point sampled from standard normal distribution
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut result = DVector::zeros(self.n);
        
        self.inner.random_point(&mut result)
            ;
        
        dvector_to_numpy(py, &result)
    }
    
    /// Generate a random tangent vector.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n,)
    ///     Point on the manifold (unused for Euclidean)
    /// scale : float, default=1.0
    ///     Scaling factor for the random vector
    ///
    /// Returns
    /// -------
    /// array_like, shape (n,)
    ///     Random tangent vector
    #[pyo3(signature = (point, scale=1.0))]
    pub fn random_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, scale: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        
        // Validate dimension
        if point_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[point_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.n);
        
        self.inner.random_tangent(&point_vec, &mut result)
            ;
        
        // Scale the result if needed
        if scale != 1.0 {
            result *= scale;
        }
        
        dvector_to_numpy(py, &result)
    }
    
    /// Parallel transport (identity operation in Euclidean space).
    ///
    /// Parameters
    /// ----------
    /// from_point : array_like, shape (n,)
    ///     Starting point
    /// to_point : array_like, shape (n,)
    ///     End point
    /// tangent : array_like, shape (n,)
    ///     Tangent vector at from_point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n,)
    ///     Same tangent vector (parallel transport is identity)
    pub fn parallel_transport<'py>(&self, py: Python<'py>, from_point: PyReadonlyArray1<'_, f64>, to_point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let from_vec = numpy_to_dvector(from_point)?;
        let to_vec = numpy_to_dvector(to_point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if from_vec.len() != self.n || to_vec.len() != self.n || tangent_vec.len() != self.n {
            return Err(dimension_mismatch(
                &[self.n],
                &[from_vec.len()],
            ));
        }
        
        // Parallel transport is identity in Euclidean space
        dvector_to_numpy(py, &tangent_vec)
    }
}

// Internal methods for trait implementation
impl PyEuclidean {
    fn parse_point(&self, py: Python<'_>, obj: PyObject) -> PyResult<PyPoint> {
        super::base::array_to_point(py, obj)
    }
    
    fn contains_vector(&self, vec: &DVector<f64>, atol: f64) -> PyResult<bool> {
        Ok(self.inner.is_point_on_manifold(vec, atol))
    }
    
    fn is_tangent_vector(&self, point: &DVector<f64>, vector: &DVector<f64>, atol: f64) -> PyResult<bool> {
        Ok(self.inner.is_vector_in_tangent_space(point, vector, atol))
    }
    
    fn contains_matrix(&self, _mat: &DMatrix<f64>, _atol: f64) -> PyResult<bool> {
        Err(crate::error::type_error(
            "vector point",
            "matrix point",
        ))
    }
    
    fn is_tangent_matrix(&self, _point: &DMatrix<f64>, _vector: &DMatrix<f64>, _atol: f64) -> PyResult<bool> {
        Err(crate::error::type_error(
            "vector tangent",
            "matrix tangent",
        ))
    }
}