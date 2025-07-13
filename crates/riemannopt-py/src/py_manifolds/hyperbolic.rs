//! Python wrapper for the Hyperbolic manifold.
//!
//! The hyperbolic manifold H^n is the n-dimensional hyperbolic space with
//! constant negative curvature. It is represented using the hyperboloid model
//! embedded in Minkowski space.

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1, PyArrayMethods};
use nalgebra::{DVector, DMatrix};
use rand::Rng;
use rand_distr::{Distribution, StandardNormal};
use riemannopt_manifolds::hyperbolic::Hyperbolic;
use riemannopt_core::manifold::Manifold;

use crate::{
    array_utils::{numpy_to_dvector, dvector_to_numpy},
    error::{to_py_err, dimension_mismatch},
    types::PyPoint,
};
use super::base::{PyManifoldBase, PointType};

/// The hyperbolic manifold H^n.
///
/// The hyperbolic manifold is the n-dimensional space with constant negative
/// curvature. This implementation uses the hyperboloid model, where points
/// are represented as (n+1)-dimensional vectors in Minkowski space satisfying
/// the constraint <x, x>_L = -1, where <·, ·>_L is the Minkowski inner product.
///
/// The Minkowski inner product is defined as:
/// <x, y>_L = -x[0]*y[0] + sum(x[i]*y[i] for i in 1..n+1)
///
/// Parameters
/// ----------
/// n : int
///     Dimension of the hyperbolic space
/// curvature : float, default=-1.0
///     Constant negative curvature (must be < 0)
///
/// Attributes
/// ----------
/// n : int
///     Dimension of the hyperbolic space
/// curvature : float
///     The constant negative curvature
/// dim : int
///     Intrinsic dimension of the manifold (equals n)
/// ambient_dim : int
///     Dimension of the ambient space (n + 1)
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> # Create a 2D hyperbolic manifold
/// >>> hyp = ro.manifolds.Hyperbolic(2)
/// >>> print(f"Manifold dimension: {hyp.dim}")
/// Manifold dimension: 2
/// >>>
/// >>> # Generate a random point
/// >>> x = hyp.random_point()
/// >>> print(x.shape)
/// (3,)
/// >>>
/// >>> # Verify constraint: <x, x>_L = -1
/// >>> minkowski_norm = -x[0]**2 + np.sum(x[1:]**2)
/// >>> print(f"Minkowski norm: {minkowski_norm:.6f}")
/// Minkowski norm: -1.000000
#[pyclass(name = "Hyperbolic", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyHyperbolic {
    pub(crate) inner: Hyperbolic<f64>,
    n: usize,
    curvature: f64,
}

impl PyManifoldBase for PyHyperbolic {
    fn manifold_name(&self) -> &'static str {
        "Hyperbolic"
    }
    
    fn ambient_dim(&self) -> usize {
        self.n + 1
    }
    
    fn intrinsic_dim(&self) -> usize {
        self.n
    }
    
    fn point_type(&self) -> PointType {
        PointType::Vector(self.n + 1)
    }
}

#[pymethods]
impl PyHyperbolic {
    /// Create a new hyperbolic manifold.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Dimension of the hyperbolic space (must be > 0)
    /// curvature : float, default=-1.0
    ///     Constant negative curvature (must be < 0)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If n <= 0 or curvature >= 0
    #[new]
    #[pyo3(signature = (n, curvature=-1.0))]
    pub fn new(n: usize, curvature: f64) -> PyResult<Self> {
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n must be positive"
            ));
        }
        if curvature >= 0.0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "curvature must be negative for hyperbolic space"
            ));
        }
        
        let inner = if (curvature - (-1.0)).abs() < 1e-10 {
            Hyperbolic::<f64>::new(n)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        } else {
            Hyperbolic::<f64>::with_parameters(n, 1e-6, curvature)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
        };
        Ok(PyHyperbolic { inner, n, curvature })
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
    
    /// Check if a point lies on the manifold.
    ///
    /// Args:
    ///     point: Point to check
    ///     atol: Absolute tolerance (default: 1e-10)
    ///
    /// Returns:
    ///     bool: True if point is on manifold
    #[pyo3(signature = (point, atol=1e-10))]
    fn contains(&self, py: Python<'_>, point: PyObject, atol: f64) -> PyResult<bool> {
        let point = self.parse_point(py, point)?;
        self.validate_point_shape(&point)?;
        
        match &point {
            PyPoint::Vector(vec) => {
                self.contains_vector(vec, atol)
            }
            PyPoint::Matrix(mat) => {
                self.contains_matrix(mat, atol)
            }
        }
    }
    
    /// Check if a vector is in the tangent space.
    ///
    /// Args:
    ///     point: Base point on the manifold
    ///     vector: Vector to check
    ///     atol: Absolute tolerance (default: 1e-10)
    ///
    /// Returns:
    ///     bool: True if vector is in tangent space
    #[pyo3(signature = (point, vector, atol=1e-10))]
    fn is_tangent(&self, py: Python<'_>, point: PyObject, vector: PyObject, atol: f64) -> PyResult<bool> {
        let point = self.parse_point(py, point)?;
        let vector = self.parse_point(py, vector)?;
        self.validate_point_shape(&point)?;
        self.validate_point_shape(&vector)?;
        
        match (&point, &vector) {
            (PyPoint::Vector(p), PyPoint::Vector(v)) => {
                self.is_tangent_vector(p, v, atol)
            }
            (PyPoint::Matrix(p), PyPoint::Matrix(v)) => {
                self.is_tangent_matrix(p, v, atol)
            }
            _ => Err(crate::error::type_error(
                "matching point and vector types",
                "mismatched types",
            )),
        }
    }
    
    /// Get the dimension n of the hyperbolic space.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    
    /// Get the curvature.
    #[getter]
    fn curvature(&self) -> f64 {
        self.curvature
    }
    
    /// Project a vector onto the hyperboloid.
    ///
    /// Given a vector in ambient space, projects it onto the hyperboloid
    /// to satisfy the constraint <x, x>_L = -1.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n+1,)
    ///     Vector to project
    ///
    /// Returns
    /// -------
    /// array_like, shape (n+1,)
    ///     Projected point on the hyperboloid
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        
        // Validate dimension
        if point_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[point_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.n + 1);
        
        self.inner.project_point(&point_vec, &mut result);
        
        dvector_to_numpy(py, &result)
    }
    
    /// Exponential map from a point in the direction of a tangent vector.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n+1,)
    ///     Point on the hyperboloid
    /// tangent : array_like, shape (n+1,)
    ///     Tangent vector at the point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n+1,)
    ///     Resulting point on the hyperboloid
    pub fn exp<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if point_vec.len() != self.n + 1 || tangent_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[point_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.n + 1);
        
        self.inner.retract(&point_vec, &tangent_vec, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }
    
    /// Logarithmic map from one point to another.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n+1,)
    ///     Starting point on the hyperboloid
    /// other : array_like, shape (n+1,)
    ///     Target point on the hyperboloid
    ///
    /// Returns
    /// -------
    /// array_like, shape (n+1,)
    ///     Tangent vector at 'point' that points to 'other'
    pub fn log<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, other: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let other_vec = numpy_to_dvector(other)?;
        
        // Validate dimensions
        if point_vec.len() != self.n + 1 || other_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[point_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.n + 1);
        
        self.inner.inverse_retract(&point_vec, &other_vec, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }
    
    /// Retraction mapping.
    ///
    /// This is a first-order approximation of the exponential map.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n+1,)
    ///     Point on the hyperboloid
    /// tangent : array_like, shape (n+1,)
    ///     Tangent vector at the point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n+1,)
    ///     Retracted point on the hyperboloid
    pub fn retract<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if point_vec.len() != self.n + 1 || tangent_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[point_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.n + 1);
        
        self.inner.retract(&point_vec, &tangent_vec, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }
    
    /// Project a vector onto the tangent space at a point.
    ///
    /// The tangent space at x consists of all vectors v such that
    /// <x, v>_L = 0.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n+1,)
    ///     Point on the hyperboloid
    /// vector : array_like, shape (n+1,)
    ///     Ambient space vector to project
    ///
    /// Returns
    /// -------
    /// array_like, shape (n+1,)
    ///     Projected tangent vector
    pub fn project_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, vector: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let vector_vec = numpy_to_dvector(vector)?;
        
        // Validate dimensions
        if point_vec.len() != self.n + 1 || vector_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[point_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.n + 1);
        
        self.inner.project_tangent(&point_vec, &vector_vec, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }
    
    /// Minkowski inner product.
    ///
    /// Computes <x, y>_L = -x[0]*y[0] + sum(x[i]*y[i] for i in 1..n+1)
    ///
    /// Parameters
    /// ----------
    /// x : array_like, shape (n+1,)
    ///     First vector
    /// y : array_like, shape (n+1,)
    ///     Second vector
    ///
    /// Returns
    /// -------
    /// float
    ///     Minkowski inner product
    pub fn minkowski_inner(&self, _py: Python<'_>, x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let x_vec = numpy_to_dvector(x)?;
        let y_vec = numpy_to_dvector(y)?;
        
        // Validate dimensions
        if x_vec.len() != self.n + 1 || y_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[x_vec.len()],
            ));
        }
        
        // Compute Minkowski inner product
        let mut result = -x_vec[0] * y_vec[0];
        for i in 1..self.n + 1 {
            result += x_vec[i] * y_vec[i];
        }
        
        Ok(result / (-self.curvature))
    }
    
    /// Riemannian inner product between two tangent vectors.
    ///
    /// This is the restriction of the Minkowski inner product to the
    /// tangent space.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n+1,)
    ///     Point on the hyperboloid
    /// u : array_like, shape (n+1,)
    ///     First tangent vector
    /// v : array_like, shape (n+1,)
    ///     Second tangent vector
    ///
    /// Returns
    /// -------
    /// float
    ///     Inner product <u, v>_point
    pub fn inner(&self, py: Python<'_>, point: PyReadonlyArray1<'_, f64>, u: PyReadonlyArray1<'_, f64>, v: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let point_vec = numpy_to_dvector(point)?;
        let u_vec = numpy_to_dvector(u)?;
        let v_vec = numpy_to_dvector(v)?;
        
        // Validate dimensions
        if point_vec.len() != self.n + 1 || u_vec.len() != self.n + 1 || v_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[point_vec.len()],
            ));
        }
        
        self.inner.inner_product(&point_vec, &u_vec, &v_vec)
            .map_err(to_py_err)
    }
    
    /// Riemannian norm of a tangent vector.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n+1,)
    ///     Point on the hyperboloid
    /// tangent : array_like, shape (n+1,)
    ///     Tangent vector
    ///
    /// Returns
    /// -------
    /// float
    ///     Norm ||tangent||_point
    pub fn norm(&self, _py: Python<'_>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let point_vec = numpy_to_dvector(point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if point_vec.len() != self.n + 1 || tangent_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[point_vec.len()],
            ));
        }
        
        self.inner.norm(&point_vec, &tangent_vec)
            .map_err(to_py_err)
    }
    
    /// Geodesic distance between two points.
    ///
    /// Parameters
    /// ----------
    /// x : array_like, shape (n+1,)
    ///     First point
    /// y : array_like, shape (n+1,)
    ///     Second point
    ///
    /// Returns
    /// -------
    /// float
    ///     Geodesic distance
    pub fn distance(&self, _py: Python<'_>, x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let x_vec = numpy_to_dvector(x)?;
        let y_vec = numpy_to_dvector(y)?;
        
        // Validate dimensions
        if x_vec.len() != self.n + 1 || y_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[x_vec.len()],
            ));
        }
        
        self.inner.distance(&x_vec, &y_vec)
            .map_err(to_py_err)
    }
    
    /// Generate a random point on the hyperboloid.
    ///
    /// Returns
    /// -------
    /// array_like, shape (n+1,)
    ///     Random point satisfying <x, x>_L = -1
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut rng = rand::thread_rng();
        let mut point = DVector::zeros(self.n + 1);
        
        // Generate random values for spatial components
        for i in 1..=self.n {
            point[i] = rng.sample::<f64, _>(StandardNormal);
        }
        
        // Compute time component to satisfy constraint
        let spatial_norm_sq: f64 = point.rows(1, self.n).norm_squared();
        point[0] = (1.0 / (-self.curvature) + spatial_norm_sq).sqrt();
        
        dvector_to_numpy(py, &point)
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n+1,)
    ///     Point on the hyperboloid
    /// scale : float, default=1.0
    ///     Scaling factor for the random vector
    ///
    /// Returns
    /// -------
    /// array_like, shape (n+1,)
    ///     Random tangent vector
    #[pyo3(signature = (point, scale=1.0))]
    pub fn random_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, scale: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        
        // Validate dimension
        if point_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[point_vec.len()],
            ));
        }
        
        let mut rng = rand::thread_rng();
        let mut vector = DVector::zeros(self.n + 1);
        
        // Generate random vector
        for i in 0..=self.n {
            vector[i] = rng.sample::<f64, _>(StandardNormal) * scale;
        }
        
        // Project to tangent space
        let mut result = DVector::zeros(self.n + 1);
        
        self.inner.project_tangent(&point_vec, &vector, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }
    
    /// Parallel transport a tangent vector along a geodesic.
    ///
    /// Parameters
    /// ----------
    /// from_point : array_like, shape (n+1,)
    ///     Starting point
    /// to_point : array_like, shape (n+1,)
    ///     End point
    /// tangent : array_like, shape (n+1,)
    ///     Tangent vector at from_point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n+1,)
    ///     Transported tangent vector at to_point
    pub fn parallel_transport<'py>(&self, py: Python<'py>, from_point: PyReadonlyArray1<'_, f64>, to_point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let from_vec = numpy_to_dvector(from_point)?;
        let to_vec = numpy_to_dvector(to_point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if from_vec.len() != self.n + 1 || to_vec.len() != self.n + 1 || tangent_vec.len() != self.n + 1 {
            return Err(dimension_mismatch(
                &[self.n + 1],
                &[from_vec.len()],
            ));
        }
        
        let result = self.inner.parallel_transport(&from_vec, &to_vec, &tangent_vec)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }
}

// Internal methods for trait implementation
impl PyHyperbolic {
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