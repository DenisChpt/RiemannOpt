//! Python bindings for the Fixed-Rank manifold.
//!
//! The fixed-rank manifold consists of matrices with a fixed rank constraint.

use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use nalgebra::DMatrix;
use riemannopt_manifolds::fixed_rank::FixedRank;
use riemannopt_core::manifold::Manifold;

use crate::{
    array_utils::{numpy_to_dmatrix, dmatrix_to_numpy},
    error::to_py_err,
};

use super::base::{PyManifoldBase, PointType};

/// Python wrapper for the Fixed-Rank manifold.
///
/// The fixed-rank manifold consists of m×n matrices of rank k:
/// {X ∈ ℝ^{m×n} : rank(X) = k}.
///
/// Parameters
/// ----------
/// m : int
///     Number of rows
/// n : int
///     Number of columns
/// k : int
///     Rank constraint
#[pyclass(name = "FixedRank", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyFixedRank {
    /// The underlying Rust FixedRank manifold
    pub(crate) inner: FixedRank,
    /// Number of rows
    pub(crate) m: usize,
    /// Number of columns
    pub(crate) n: usize,
    /// Rank
    pub(crate) k: usize,
}

impl PyManifoldBase for PyFixedRank {
    fn manifold_name(&self) -> &'static str {
        "FixedRank"
    }
    
    fn ambient_dim(&self) -> usize {
        self.m * self.n
    }
    
    fn intrinsic_dim(&self) -> usize {
        self.k * (self.m + self.n - self.k)
    }
    
    fn point_type(&self) -> PointType {
        PointType::Matrix(self.m, self.n)
    }
}

#[pymethods]
impl PyFixedRank {
    /// Create a new Fixed-Rank manifold.
    ///
    /// Args:
    ///     m: Number of rows (must be >= 1)
    ///     n: Number of columns (must be >= 1)
    ///     k: Rank (must be <= min(m, n))
    ///
    /// Raises:
    ///     ValueError: If m < 1, n < 1, or k > min(m, n)
    #[new]
    pub fn new(m: usize, n: usize, k: usize) -> PyResult<Self> {
        if m < 1 || n < 1 {
            return Err(crate::error::value_error(
                "FixedRank manifold requires m >= 1 and n >= 1"
            ));
        }
        if k > m.min(n) {
            return Err(crate::error::value_error(
                format!("FixedRank manifold requires k <= min(m, n), got k={}, min(m, n)={}", k, m.min(n))
            ));
        }
        
        Ok(PyFixedRank {
            inner: FixedRank::new(m, n, k).map_err(crate::error::to_py_err)?,
            m,
            n,
            k,
        })
    }
    
    /// String representation of the manifold.
    fn __repr__(&self) -> String {
        format!("FixedRank(m={}, n={}, k={})", self.m, self.n, self.k)
    }
    
    /// Get the number of rows.
    #[getter]
    pub fn m(&self) -> usize {
        self.m
    }
    
    /// Get the number of columns.
    #[getter]
    pub fn n(&self) -> usize {
        self.n
    }
    
    /// Get the rank.
    #[getter]
    pub fn k(&self) -> usize {
        self.k
    }
    
    /// Check if a point is on the manifold.
    ///
    /// Args:
    ///     point: Matrix to check
    ///     atol: Absolute tolerance (default: 1e-10)
    ///
    /// Returns:
    ///     True if the point has the correct rank
    #[pyo3(signature = (point, atol=1e-10))]
    pub fn contains(&self, point: PyReadonlyArray2<'_, f64>, atol: f64) -> PyResult<bool> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        // Validate dimensions
        if point_mat.nrows() != self.m || point_mat.ncols() != self.n {
            return Ok(false);
        }
        
        Ok(self.inner.is_point_on_manifold(&point_mat, atol))
    }
    
    /// Project a point onto the manifold.
    ///
    /// Performs truncated SVD to project to the nearest rank-k matrix.
    ///
    /// Args:
    ///     point: Matrix to project
    ///
    /// Returns:
    ///     Projected matrix of rank k
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        // Validate dimensions
        if point_mat.nrows() != self.m || point_mat.ncols() != self.n {
            return Err(crate::error::dimension_mismatch(
                &[self.m, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.m, self.n);
        
        self.inner.project_point(&point_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Project a tangent vector at a point.
    ///
    /// Args:
    ///     point: Base point on the manifold
    ///     tangent: Tangent vector to project
    ///
    /// Returns:
    ///     Projected tangent vector
    pub fn project_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate dimensions
        if point_mat.shape() != (self.m, self.n) || tangent_mat.shape() != (self.m, self.n) {
            return Err(crate::error::dimension_mismatch(
                &[self.m, self.n],
                &[tangent_mat.nrows(), tangent_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.m, self.n);
        
        self.inner.project_tangent(&point_mat, &tangent_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Compute the retraction.
    ///
    /// Args:
    ///     point: Base point on the manifold
    ///     tangent: Tangent vector at the base point
    ///
    /// Returns:
    ///     Point on the manifold
    pub fn retract<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        let mut result = DMatrix::zeros(self.m, self.n);
        
        self.inner.retract(&point_mat, &tangent_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Compute the inverse retraction.
    ///
    /// Args:
    ///     x: First point on the manifold
    ///     y: Second point on the manifold
    ///
    /// Returns:
    ///     Tangent vector at x
    pub fn inverse_retract<'py>(&self, py: Python<'py>, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let x_mat = numpy_to_dmatrix(x)?;
        let y_mat = numpy_to_dmatrix(y)?;
        
        let mut result = DMatrix::zeros(self.m, self.n);
        
        self.inner.inverse_retract(&x_mat, &y_mat, &mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Compute the Riemannian inner product.
    ///
    /// Args:
    ///     point: Base point on the manifold
    ///     u: First tangent vector
    ///     v: Second tangent vector
    ///
    /// Returns:
    ///     Inner product value
    pub fn inner(&self, point: PyReadonlyArray2<'_, f64>, u: PyReadonlyArray2<'_, f64>, v: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let point_mat = numpy_to_dmatrix(point)?;
        let u_mat = numpy_to_dmatrix(u)?;
        let v_mat = numpy_to_dmatrix(v)?;
        
        Ok(self.inner.inner_product(&point_mat, &u_mat, &v_mat)
            .map_err(to_py_err)?)
    }
    
    /// Compute the norm of a tangent vector.
    ///
    /// Args:
    ///     point: Base point on the manifold
    ///     tangent: Tangent vector
    ///
    /// Returns:
    ///     Norm of the tangent vector
    pub fn norm(&self, point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let point_mat = numpy_to_dmatrix(point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        Ok(self.inner.norm(&point_mat, &tangent_mat)
            .map_err(to_py_err)?)
    }
    
    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random matrix of rank k
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut result = DMatrix::zeros(self.m, self.n);
        
        self.inner.random_point(&mut result)
            ;
        
        dmatrix_to_numpy(py, &result)
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Base point on the manifold
    ///     scale: Standard deviation of the tangent vector (default: 1.0)
    ///
    /// Returns:
    ///     Random tangent vector at the point
    #[pyo3(signature = (point, scale=1.0))]
    pub fn random_tangent<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>, scale: f64) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        let mut result = DMatrix::zeros(self.m, self.n);
        
        self.inner.random_tangent(&point_mat, &mut result)
            ;
        
        // Scale the result if needed
        if scale != 1.0 {
            result *= scale;
        }
        
        dmatrix_to_numpy(py, &result)
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
    
    /// Parallel transport a tangent vector along a geodesic.
    ///
    /// Args:
    ///     from_point: Starting point
    ///     to_point: Ending point
    ///     tangent: Tangent vector at from_point
    ///
    /// Returns:
    ///     Transported tangent vector at to_point
    pub fn parallel_transport<'py>(&self, py: Python<'py>, from_point: PyReadonlyArray2<'_, f64>, to_point: PyReadonlyArray2<'_, f64>, tangent: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let from_mat = numpy_to_dmatrix(from_point)?;
        let to_mat = numpy_to_dmatrix(to_point)?;
        let tangent_mat = numpy_to_dmatrix(tangent)?;
        
        // Validate dimensions
        if from_mat.shape() != (self.m, self.n) || to_mat.shape() != (self.m, self.n) || tangent_mat.shape() != (self.m, self.n) {
            return Err(crate::error::dimension_mismatch(
                &[self.m, self.n, self.m, self.n, self.m, self.n],
                &[from_mat.nrows(), from_mat.ncols(), to_mat.nrows(), to_mat.ncols(), tangent_mat.nrows(), tangent_mat.ncols()],
            ));
        }
        
        let mut result = DMatrix::zeros(self.m, self.n);
        self.inner.parallel_transport(&from_mat, &to_mat, &tangent_mat, &mut result)
            .map_err(to_py_err)?;
        
        dmatrix_to_numpy(py, &result)
    }
}