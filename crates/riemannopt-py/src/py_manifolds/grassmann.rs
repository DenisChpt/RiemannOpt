//! Python wrapper for the Grassmann manifold.
//!
//! The Grassmann manifold Gr(n, p) is the set of all p-dimensional
//! linear subspaces of R^n. Points are represented as n×p matrices
//! with orthonormal columns.

use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, PyArrayMethods};
use nalgebra::{DMatrix, DVector};
use riemannopt_manifolds::grassmann::Grassmann;
use riemannopt_core::manifold::Manifold;

use crate::{
    array_utils::{numpy_to_dmatrix, dmatrix_to_numpy},
    error::{to_py_err, dimension_mismatch},
    types::PyPoint,
};
use super::base::{PyManifoldBase, PointType};

/// The Grassmann manifold Gr(n, p).
///
/// The Grassmann manifold is the set of all p-dimensional linear subspaces
/// of n-dimensional Euclidean space. Points on this manifold are represented
/// as n×p matrices with orthonormal columns.
///
/// Parameters
/// ----------
/// n : int
///     Dimension of the ambient space
/// p : int
///     Dimension of the subspaces
///
/// Attributes
/// ----------
/// n : int
///     Dimension of the ambient space
/// p : int
///     Dimension of the subspaces
/// dim : int
///     Intrinsic dimension of the manifold (p * (n - p))
/// ambient_dim : int
///     Dimension of the ambient space (n * p)
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> # Create a Grassmann manifold Gr(5, 2)
/// >>> grass = ro.manifolds.Grassmann(5, 2)
/// >>> print(f"Manifold dimension: {grass.dim}")
/// Manifold dimension: 6
/// >>>
/// >>> # Generate a random point
/// >>> X = grass.random_point()
/// >>> print(X.shape)
/// (5, 2)
/// >>>
/// >>> # Check orthonormality
/// >>> print(np.linalg.norm(X.T @ X - np.eye(2)))
/// 0.0
#[pyclass(name = "Grassmann", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyGrassmann {
    pub(crate) inner: Grassmann<f64>,
    n: usize,
    p: usize,
}

impl PyManifoldBase for PyGrassmann {
    fn manifold_name(&self) -> &'static str {
        "Grassmann"
    }
    
    fn ambient_dim(&self) -> usize {
        self.n * self.p
    }
    
    fn intrinsic_dim(&self) -> usize {
        self.p * (self.n - self.p)
    }
    
    fn point_type(&self) -> PointType {
        PointType::Matrix(self.n, self.p)
    }
}

#[pymethods]
impl PyGrassmann {
    /// Create a new Grassmann manifold.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Dimension of the ambient space (must be > 0)
    /// p : int
    ///     Dimension of the subspaces (must be > 0 and <= n)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If n <= 0, p <= 0, or p > n
    #[new]
    pub fn new(n: usize, p: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "n must be positive"
            ));
        }
        if p == 0 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "p must be positive"
            ));
        }
        if p > n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "p must be <= n"
            ));
        }
        
        let inner = Grassmann::<f64>::new(n, p)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyGrassmann { inner, n, p })
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
    
    /// Get the ambient space dimension n.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    
    /// Get the subspace dimension p.
    #[getter]
    fn p(&self) -> usize {
        self.p
    }
    
    /// Project a matrix onto the manifold.
    ///
    /// This performs a QR decomposition to orthonormalize the columns.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, p)
    ///     Matrix to project
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, p)
    ///     Projected point with orthonormal columns
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        // Validate shape
        if point_mat.nrows() != self.n || point_mat.ncols() != self.p {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.p);
        self.inner.project_point(&point_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Exponential map from a point in the direction of a tangent vector.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, p)
    ///     Point on the manifold (orthonormal columns)
    /// tangent : array_like, shape (n, p)
    ///     Tangent vector at the point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, p)
    ///     Resulting point on the manifold
    pub fn exp<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.p) || tangent_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.p);
        self.inner.retract(&point_mat, &tangent_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Logarithmic map from one point to another.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, p)
    ///     Starting point on the manifold
    /// other : array_like, shape (n, p)
    ///     Target point on the manifold
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, p)
    ///     Tangent vector at 'point' that points to 'other'
    pub fn log<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, other: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let other_mat = numpy_to_dmatrix(other)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.p) || other_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.p);
        self.inner.inverse_retract(&point_mat, &other_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Retraction mapping.
    ///
    /// This is a first-order approximation of the exponential map that is
    /// computationally cheaper.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, p)
    ///     Point on the manifold
    /// tangent : array_like, shape (n, p)
    ///     Tangent vector at the point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, p)
    ///     Retracted point on the manifold
    pub fn retract<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.p) || tangent_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.p);
        self.inner.retract(&point_mat, &tangent_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Project a vector onto the tangent space at a point.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, p)
    ///     Point on the manifold
    /// vector : array_like, shape (n, p)
    ///     Ambient space vector to project
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, p)
    ///     Projected tangent vector
    pub fn project_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, vector: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let vector_mat = numpy_to_dmatrix(vector)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.p) || vector_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.p);
        self.inner.project_tangent(&point_mat, &vector_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Riemannian inner product between two tangent vectors.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, p)
    ///     Point on the manifold
    /// u : array_like, shape (n, p)
    ///     First tangent vector
    /// v : array_like, shape (n, p)
    ///     Second tangent vector
    ///
    /// Returns
    /// -------
    /// float
    ///     Inner product <u, v>_point
    pub fn inner(&self, _py: Python<'_>, point: PyReadonlyArray2<'_, f64>, u: PyReadonlyArray2<'_, f64>, v: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let point_mat = numpy_to_dmatrix(point)?;
        let u_mat = numpy_to_dmatrix(u)?;
        let v_mat = numpy_to_dmatrix(v)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.p) || u_mat.shape() != (self.n, self.p) || v_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        self.inner.inner_product(&point_mat, &u_mat, &v_mat)
            .map_err(to_py_err)
    }
    
    /// Riemannian norm of a tangent vector.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, p)
    ///     Point on the manifold
    /// tangent : array_like, shape (n, p)
    ///     Tangent vector
    ///
    /// Returns
    /// -------
    /// float
    ///     Norm ||tangent||_point
    pub fn norm(&self, _py: Python<'_>, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.p) || tangent_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        self.inner.norm(&point_mat, &tangent_mat)
            .map_err(to_py_err)
    }
    
    /// Geodesic distance between two points.
    ///
    /// Parameters
    /// ----------
    /// x : array_like, shape (n, p)
    ///     First point
    /// y : array_like, shape (n, p)
    ///     Second point
    ///
    /// Returns
    /// -------
    /// float
    ///     Geodesic distance
    pub fn distance(&self, _py: Python<'_>, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let x_mat = numpy_to_dmatrix(x)?;
        let y_mat = numpy_to_dmatrix(y)?;
        
        // Validate shapes
        if x_mat.shape() != (self.n, self.p) || y_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[x_mat.nrows(), x_mat.ncols()],
            ));
        }
        
        self.inner.distance(&x_mat, &y_mat)
            .map_err(to_py_err)
    }
    
    /// Generate a random point on the manifold.
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, p)
    ///     Random point with orthonormal columns
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut result = DMatrix::zeros(self.n, self.p);
        
        self.inner.random_point(&mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, p)
    ///     Point on the manifold
    /// scale : float, default=1.0
    ///     Scaling factor for the random vector
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, p)
    ///     Random tangent vector
    #[pyo3(signature = (point, scale=1.0))]
    pub fn random_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, scale: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        // Validate shape
        if point_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.p);
        
        self.inner.random_tangent(&point_mat, &mut result)
            ;
        
        // Scale the result if needed
        if scale != 1.0 {
            result *= scale;
        }
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Parallel transport a tangent vector along a geodesic.
    ///
    /// Parameters
    /// ----------
    /// from_point : array_like, shape (n, p)
    ///     Starting point
    /// to_point : array_like, shape (n, p)
    ///     End point
    /// tangent : array_like, shape (n, p)
    ///     Tangent vector at from_point
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, p)
    ///     Transported tangent vector at to_point
    pub fn parallel_transport<'py>(&self, py: Python<'py>, from_point: PyReadonlyArray2<'_, f64>, to_point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let from_mat = numpy_to_dmatrix(from_point)?;
        let to_mat = numpy_to_dmatrix(to_point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate shapes
        if from_mat.shape() != (self.n, self.p) || to_mat.shape() != (self.n, self.p) || tangent_mat.shape() != (self.n, self.p) {
            return Err(dimension_mismatch(
                &[self.n, self.p],
                &[from_mat.nrows(), from_mat.ncols()],
            ));
        }
        
        let result = self.inner.parallel_transport(&from_mat, &to_mat, &tangent_mat)
            .map_err(to_py_err)?;
        
        dmatrix_to_numpy(py, &result)
    }
}

// Internal methods for trait implementation
impl PyGrassmann {
    fn parse_point(&self, py: Python<'_>, obj: PyObject) -> PyResult<PyPoint> {
        super::base::array_to_point(py, obj)
    }
    
    fn contains_matrix(&self, mat: &DMatrix<f64>, atol: f64) -> PyResult<bool> {
        Ok(self.inner.is_point_on_manifold(mat, atol))
    }
    
    fn is_tangent_matrix(&self, point: &DMatrix<f64>, vector: &DMatrix<f64>, atol: f64) -> PyResult<bool> {
        Ok(self.inner.is_vector_in_tangent_space(point, vector, atol))
    }
    
    fn contains_vector(&self, _vec: &DVector<f64>, _atol: f64) -> PyResult<bool> {
        Err(crate::error::type_error(
            "matrix point",
            "vector point",
        ))
    }
    
    fn is_tangent_vector(&self, _point: &DVector<f64>, _vector: &DVector<f64>, _atol: f64) -> PyResult<bool> {
        Err(crate::error::type_error(
            "matrix tangent",
            "vector tangent",
        ))
    }
}