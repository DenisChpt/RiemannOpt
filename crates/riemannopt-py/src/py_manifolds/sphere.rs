//! Python wrapper for the Sphere manifold.
//!
//! This module provides a Python-friendly interface to the Sphere manifold,
//! handling all workspace management and type conversions internally.

use pyo3::prelude::*;
use numpy::{PyArray1, PyReadonlyArray1};
use nalgebra::{DVector, DMatrix};
use riemannopt_manifolds::Sphere;
use riemannopt_core::core::manifold::Manifold;
use rand::thread_rng;
use rand_distr::{Distribution, StandardNormal};

use crate::array_utils::{numpy_to_dvector, dvector_to_numpy};
use crate::error::to_py_err;
use crate::types::PyPoint;
use super::base::{PyManifoldBase, PointType, array_to_point};

/// Python wrapper for the unit sphere S^{n-1} in R^n.
///
/// The sphere manifold represents all points with unit norm in n-dimensional space:
/// S^{n-1} = {x ∈ R^n : ||x||_2 = 1}
///
/// This is commonly used for:
/// - Normalized embeddings in machine learning
/// - Directional statistics
/// - Optimization with norm constraints
#[pyclass(name = "Sphere", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PySphere {
    /// The underlying Rust sphere manifold
    pub(crate) inner: Sphere,
    /// Dimension of the ambient space
    pub(crate) dimension: usize,
}

impl PyManifoldBase for PySphere {
    fn manifold_name(&self) -> &'static str {
        "Sphere"
    }
    
    fn ambient_dim(&self) -> usize {
        self.dimension
    }
    
    fn intrinsic_dim(&self) -> usize {
        self.dimension - 1
    }
    
    fn point_type(&self) -> PointType {
        PointType::Vector(self.dimension)
    }
}

#[pymethods]
impl PySphere {
    /// Create a new Sphere manifold.
    ///
    /// Args:
    ///     dimension: Dimension of the ambient space (must be >= 2)
    ///
    /// Raises:
    ///     ValueError: If dimension < 2
    #[new]
    pub fn new(dimension: usize) -> PyResult<Self> {
        if dimension < 2 {
            return Err(crate::error::value_error(
                "Sphere dimension must be at least 2"
            ));
        }
        
        Ok(PySphere {
            inner: Sphere::new(dimension).map_err(to_py_err)?,
            dimension,
        })
    }

    /// Project a point onto the manifold.
    ///
    /// This normalizes the point to have unit norm.
    ///
    /// Args:
    ///     point: Point in ambient space
    ///
    /// Returns:
    ///     Projected point on the sphere
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        
        // Validate dimension
        if point_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension],
                &[point_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.dimension);
        
        self.inner.project_point(&point_vec, &mut result);
        
        dvector_to_numpy(py, &result)
    }

    /// Compute the Riemannian exponential map.
    ///
    /// Maps a tangent vector at a point to a point on the manifold.
    ///
    /// Args:
    ///     point: Base point on the sphere
    ///     tangent: Tangent vector at the base point
    ///
    /// Returns:
    ///     Point on the sphere
    pub fn exp<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if point_vec.len() != self.dimension || tangent_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension, self.dimension],
                &[point_vec.len(), tangent_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.dimension);
        self.inner.retract(&point_vec, &tangent_vec, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }

    /// Compute the Riemannian logarithm map.
    ///
    /// Maps a point to a tangent vector at the base point.
    ///
    /// Args:
    ///     point: Base point on the sphere
    ///     other: Target point on the sphere
    ///
    /// Returns:
    ///     Tangent vector at the base point
    pub fn log<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, other: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let other_vec = numpy_to_dvector(other)?;
        
        // Validate dimensions
        if point_vec.len() != self.dimension || other_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension, self.dimension],
                &[point_vec.len(), other_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.dimension);
        self.inner.inverse_retract(&point_vec, &other_vec, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }

    /// Retract a tangent vector to the manifold.
    ///
    /// This is a first-order approximation of the exponential map.
    ///
    /// Args:
    ///     point: Base point on the sphere
    ///     tangent: Tangent vector at the base point
    ///
    /// Returns:
    ///     Point on the sphere
    pub fn retract<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if point_vec.len() != self.dimension || tangent_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension, self.dimension],
                &[point_vec.len(), tangent_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.dimension);
        
        self.inner.retract(&point_vec, &tangent_vec, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }

    /// Project a vector onto the tangent space.
    ///
    /// Args:
    ///     point: Base point on the sphere
    ///     vector: Vector in ambient space
    ///
    /// Returns:
    ///     Projected vector in tangent space
    pub fn project_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, vector: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        let vector_vec = numpy_to_dvector(vector)?;
        
        // Validate dimensions
        if point_vec.len() != self.dimension || vector_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension, self.dimension],
                &[point_vec.len(), vector_vec.len()],
            ));
        }
        
        let mut result = DVector::zeros(self.dimension);
        
        self.inner.project_tangent(&point_vec, &vector_vec, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }

    /// Compute the Riemannian inner product.
    ///
    /// Args:
    ///     point: Base point on the sphere
    ///     u: First tangent vector
    ///     v: Second tangent vector
    ///
    /// Returns:
    ///     Inner product value
    pub fn inner(&self, _py: Python<'_>, point: PyReadonlyArray1<'_, f64>, u: PyReadonlyArray1<'_, f64>, v: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let point_vec = numpy_to_dvector(point)?;
        let u_vec = numpy_to_dvector(u)?;
        let v_vec = numpy_to_dvector(v)?;
        
        // Validate dimensions
        if point_vec.len() != self.dimension || u_vec.len() != self.dimension || v_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension, self.dimension, self.dimension],
                &[point_vec.len(), u_vec.len(), v_vec.len()],
            ));
        }
        
        self.inner.inner_product(&point_vec, &u_vec, &v_vec)
            .map_err(to_py_err)
    }

    /// Compute the norm of a tangent vector.
    ///
    /// Args:
    ///     point: Base point on the sphere
    ///     tangent: Tangent vector
    ///
    /// Returns:
    ///     Norm of the tangent vector
    pub fn norm(&self, _py: Python<'_>, point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let point_vec = numpy_to_dvector(point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if point_vec.len() != self.dimension || tangent_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension, self.dimension],
                &[point_vec.len(), tangent_vec.len()],
            ));
        }
        
        Ok(self.inner.norm(&point_vec, &tangent_vec)
            .map_err(to_py_err)?)
    }

    /// Compute the geodesic distance between two points.
    ///
    /// Args:
    ///     x: First point on the sphere
    ///     y: Second point on the sphere
    ///
    /// Returns:
    ///     Geodesic distance
    pub fn distance(&self, _py: Python<'_>, x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let x_vec = numpy_to_dvector(x)?;
        let y_vec = numpy_to_dvector(y)?;
        
        // Validate dimensions
        if x_vec.len() != self.dimension || y_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension, self.dimension],
                &[x_vec.len(), y_vec.len()],
            ));
        }
        
        Ok(self.inner.distance(&x_vec, &y_vec)
            .map_err(to_py_err)?)
    }

    /// Generate a random point on the sphere.
    ///
    /// Returns:
    ///     Random point uniformly distributed on the sphere
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut rng = thread_rng();
        
        // Generate random Gaussian vector
        let mut point = DVector::zeros(self.dimension);
        for i in 0..self.dimension {
            point[i] = StandardNormal.sample(&mut rng);
        }
        
        // Project to sphere (normalize)
        let mut result = DVector::zeros(self.dimension);
        
        self.inner.project_point(&point, &mut result);
        
        dvector_to_numpy(py, &result)
    }

    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Base point on the sphere
    ///     scale: Standard deviation of the tangent vector (default: 1.0)
    ///
    /// Returns:
    ///     Random tangent vector
    #[pyo3(signature = (point, scale=1.0))]
    pub fn random_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>, scale: f64) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = numpy_to_dvector(point)?;
        
        // Validate dimension
        if point_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension],
                &[point_vec.len()],
            ));
        }
        
        let mut rng = thread_rng();
        
        // Generate random vector in ambient space
        let mut ambient = DVector::zeros(self.dimension);
        for i in 0..self.dimension {
            let val: f64 = StandardNormal.sample(&mut rng);
            ambient[i] = val * scale;
        }
        
        // Project to tangent space
        let mut result = DVector::zeros(self.dimension);
        
        self.inner.project_tangent(&point_vec, &ambient, &mut result)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }

    /// Parallel transport a tangent vector along a geodesic.
    ///
    /// Args:
    ///     from_point: Starting point
    ///     to_point: Ending point
    ///     tangent: Tangent vector at from_point
    ///
    /// Returns:
    ///     Transported tangent vector at to_point
    pub fn parallel_transport<'py>(&self, py: Python<'py>, from_point: PyReadonlyArray1<'_, f64>, to_point: PyReadonlyArray1<'_, f64>, tangent: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let from_vec = numpy_to_dvector(from_point)?;
        let to_vec = numpy_to_dvector(to_point)?;
        let tangent_vec = numpy_to_dvector(tangent)?;
        
        // Validate dimensions
        if from_vec.len() != self.dimension || to_vec.len() != self.dimension || tangent_vec.len() != self.dimension {
            return Err(crate::error::dimension_mismatch(
                &[self.dimension, self.dimension, self.dimension],
                &[from_vec.len(), to_vec.len(), tangent_vec.len()],
            ));
        }
        
        let result = self.inner.parallel_transport(&from_vec, &to_vec, &tangent_vec)
            .map_err(to_py_err)?;
        
        dvector_to_numpy(py, &result)
    }
    
    // Include common methods from macro
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
}

// Additional helper methods specific to PySphere
impl PySphere {
    /// Parse a Python object as a point.
    fn parse_point(&self, py: Python<'_>, obj: PyObject) -> PyResult<PyPoint> {
        array_to_point(py, obj)
    }
    
    /// Check if a vector point is on the manifold.
    fn contains_vector(&self, point: &DVector<f64>, atol: f64) -> PyResult<bool> {
        Ok(self.inner.is_point_on_manifold(point, atol))
    }
    
    /// Check if a matrix point is on the manifold (not applicable for sphere).
    fn contains_matrix(&self, _mat: &DMatrix<f64>, _atol: f64) -> PyResult<bool> {
        Err(crate::error::type_error("vector point", "matrix point"))
    }
    
    /// Check if a vector is in the tangent space.
    fn is_tangent_vector(&self, point: &DVector<f64>, vector: &DVector<f64>, atol: f64) -> PyResult<bool> {
        Ok(self.inner.is_vector_in_tangent_space(point, vector, atol))
    }
    
    /// Check if a matrix is in the tangent space (not applicable for sphere).
    fn is_tangent_matrix(&self, _point: &DMatrix<f64>, _vector: &DMatrix<f64>, _atol: f64) -> PyResult<bool> {
        Err(crate::error::type_error("vector tangent", "matrix tangent"))
    }
}