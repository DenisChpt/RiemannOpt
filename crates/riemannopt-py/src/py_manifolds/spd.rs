//! Python wrapper for the Symmetric Positive Definite (SPD) manifold.
//!
//! The SPD manifold is the set of all symmetric positive definite matrices.
//! This manifold has applications in covariance matrix estimation, metric learning,
//! and diffusion tensor imaging.

use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2, PyArrayMethods};
use nalgebra::{DMatrix, DVector};
use riemannopt_manifolds::spd::SPD;
use riemannopt_core::manifold::Manifold;

use crate::{
    array_utils::{numpy_to_dmatrix, dmatrix_to_numpy},
    error::{to_py_err, dimension_mismatch},
    types::PyPoint,
};
use super::base::{PyManifoldBase, PointType};

/// The manifold of symmetric positive definite matrices.
///
/// The SPD manifold SPD(n) is the set of all n×n symmetric positive definite
/// matrices. This manifold has dimension n(n+1)/2.
///
/// The Riemannian metric used is the affine-invariant metric (also known as
/// the natural metric), which is invariant under congruence transformations.
///
/// Parameters
/// ----------
/// n : int
///     Size of the square matrices
///
/// Attributes
/// ----------
/// n : int
///     Size of the matrices
/// dim : int
///     Intrinsic dimension of the manifold (n * (n + 1) / 2)
/// ambient_dim : int
///     Dimension of the ambient space (n * n)
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> # Create SPD manifold for 3x3 matrices
/// >>> spd = ro.manifolds.SPD(3)
/// >>> print(f"Manifold dimension: {spd.dim}")
/// Manifold dimension: 6
/// >>>
/// >>> # Generate a random SPD matrix
/// >>> X = spd.random_point()
/// >>> print(f"Is symmetric: {np.allclose(X, X.T)}")
/// Is symmetric: True
/// >>> print(f"Eigenvalues > 0: {np.all(np.linalg.eigvals(X) > 0)}")
/// Eigenvalues > 0: True
#[pyclass(name = "SPD", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PySPD {
    pub(crate) inner: SPD<f64>,
    n: usize,
}

impl PyManifoldBase for PySPD {
    fn manifold_name(&self) -> &'static str {
        "SPD"
    }
    
    fn ambient_dim(&self) -> usize {
        self.n * self.n
    }
    
    fn intrinsic_dim(&self) -> usize {
        self.n * (self.n + 1) / 2
    }
    
    fn point_type(&self) -> PointType {
        PointType::Matrix(self.n, self.n)
    }
}

#[pymethods]
impl PySPD {
    /// Create a new SPD manifold.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Size of the square matrices (must be > 0)
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
        
        let inner = SPD::<f64>::new(n)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PySPD { inner, n })
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
    
    /// Get the matrix size n.
    #[getter]
    fn n(&self) -> usize {
        self.n
    }
    
    /// Project a matrix onto the manifold.
    ///
    /// This ensures the matrix is symmetric positive definite by:
    /// 1. Symmetrizing: X = (X + X^T) / 2
    /// 2. Making positive definite via eigendecomposition
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, n)
    ///     Matrix to project
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, n)
    ///     Projected SPD matrix
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        // Validate shape
        if point_mat.nrows() != self.n || point_mat.ncols() != self.n {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.n);
        
        self.inner.project_point(&point_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Exponential map from a point in the direction of a tangent vector.
    ///
    /// Uses the matrix exponential: exp_X(V) = X^{1/2} exp(X^{-1/2} V X^{-1/2}) X^{1/2}
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, n)
    ///     Point on the manifold (SPD matrix)
    /// tangent : array_like, shape (n, n)
    ///     Tangent vector at the point (symmetric matrix)
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, n)
    ///     Resulting SPD matrix
    pub fn exp<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.n) || tangent_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.n);
        
        self.inner.retract(&point_mat, &tangent_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Logarithmic map from one point to another.
    ///
    /// Computes the tangent vector at 'point' that points to 'other'.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, n)
    ///     Starting SPD matrix
    /// other : array_like, shape (n, n)
    ///     Target SPD matrix
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, n)
    ///     Symmetric tangent vector at 'point'
    pub fn log<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, other: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let other_mat = numpy_to_dmatrix(other)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.n) || other_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.n);
        
        self.inner.inverse_retract(&point_mat, &other_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Retraction mapping.
    ///
    /// Uses the Cholesky-based retraction for efficiency:
    /// R_X(V) = X + V + (1/2) V X^{-1} V
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, n)
    ///     Point on the manifold (SPD matrix)
    /// tangent : array_like, shape (n, n)
    ///     Tangent vector at the point (symmetric matrix)
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, n)
    ///     Retracted SPD matrix
    pub fn retract<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.n) || tangent_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.n);
        
        self.inner.retract(&point_mat, &tangent_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Project a matrix onto the tangent space at a point.
    ///
    /// The tangent space at X consists of all symmetric matrices.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, n)
    ///     Point on the manifold (SPD matrix)
    /// vector : array_like, shape (n, n)
    ///     Ambient space matrix to project
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, n)
    ///     Projected symmetric tangent matrix
    pub fn project_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, vector: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let vector_mat = numpy_to_dmatrix(vector)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.n) || vector_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.n);
        
        self.inner.project_tangent(&point_mat, &vector_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Riemannian inner product between two tangent vectors.
    ///
    /// Uses the affine-invariant metric: <U, V>_X = tr(X^{-1} U X^{-1} V)
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, n)
    ///     Point on the manifold (SPD matrix)
    /// u : array_like, shape (n, n)
    ///     First tangent vector (symmetric matrix)
    /// v : array_like, shape (n, n)
    ///     Second tangent vector (symmetric matrix)
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
        if point_mat.shape() != (self.n, self.n) || u_mat.shape() != (self.n, self.n) || v_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
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
    /// point : array_like, shape (n, n)
    ///     Point on the manifold (SPD matrix)
    /// tangent : array_like, shape (n, n)
    ///     Tangent vector (symmetric matrix)
    ///
    /// Returns
    /// -------
    /// float
    ///     Norm ||tangent||_point
    pub fn norm(&self, _py: Python<'_>, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate shapes
        if point_mat.shape() != (self.n, self.n) || tangent_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        self.inner.norm(&point_mat, &tangent_mat)
            .map_err(to_py_err)
    }
    
    /// Geodesic distance between two SPD matrices.
    ///
    /// Uses the affine-invariant distance:
    /// d(X, Y) = ||log(X^{-1/2} Y X^{-1/2})||_F
    ///
    /// Parameters
    /// ----------
    /// x : array_like, shape (n, n)
    ///     First SPD matrix
    /// y : array_like, shape (n, n)
    ///     Second SPD matrix
    ///
    /// Returns
    /// -------
    /// float
    ///     Geodesic distance
    pub fn distance(&self, _py: Python<'_>, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let x_mat = numpy_to_dmatrix(x)?;
        let y_mat = numpy_to_dmatrix(y)?;
        
        // Validate shapes
        if x_mat.shape() != (self.n, self.n) || y_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[x_mat.nrows(), x_mat.ncols()],
            ));
        }
        
        self.inner.distance(&x_mat, &y_mat)
            .map_err(to_py_err)
    }
    
    /// Generate a random SPD matrix.
    ///
    /// Creates a random SPD matrix by generating a random matrix A
    /// and computing A @ A.T + epsilon * I.
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, n)
    ///     Random SPD matrix
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut result = DMatrix::zeros(self.n, self.n);
        
        self.inner.random_point(&mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Creates a random symmetric matrix scaled appropriately.
    ///
    /// Parameters
    /// ----------
    /// point : array_like, shape (n, n)
    ///     Point on the manifold (SPD matrix)
    /// scale : float, default=1.0
    ///     Scaling factor for the random vector
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, n)
    ///     Random symmetric tangent matrix
    #[pyo3(signature = (point, scale=1.0))]
    pub fn random_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, scale: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        // Validate shape
        if point_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.n, self.n);
        
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
    /// from_point : array_like, shape (n, n)
    ///     Starting SPD matrix
    /// to_point : array_like, shape (n, n)
    ///     End SPD matrix
    /// tangent : array_like, shape (n, n)
    ///     Tangent vector at from_point (symmetric matrix)
    ///
    /// Returns
    /// -------
    /// array_like, shape (n, n)
    ///     Transported tangent vector at to_point
    pub fn parallel_transport<'py>(&self, py: Python<'py>, from_point: PyReadonlyArray2<'_, f64>, to_point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let from_mat = numpy_to_dmatrix(from_point)?;
        let to_mat = numpy_to_dmatrix(to_point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate shapes
        if from_mat.shape() != (self.n, self.n) || to_mat.shape() != (self.n, self.n) || tangent_mat.shape() != (self.n, self.n) {
            return Err(dimension_mismatch(
                &[self.n, self.n],
                &[from_mat.nrows(), from_mat.ncols()],
            ));
        }
        
        let result = self.inner.parallel_transport(&from_mat, &to_mat, &tangent_mat)
            .map_err(to_py_err)?;
        
        dmatrix_to_numpy(py, &result)
    }
}

// Internal methods for trait implementation
impl PySPD {
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