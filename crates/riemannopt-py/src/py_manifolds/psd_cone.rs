//! Python bindings for the PSD Cone manifold.
//!
//! The positive semi-definite cone consists of symmetric positive semi-definite matrices.

use pyo3::prelude::*;
use numpy::{PyArray2, PyReadonlyArray2};
use nalgebra::{DMatrix, DVector};
use riemannopt_manifolds::psd_cone::PSDCone;
use riemannopt_core::manifold::Manifold;

use crate::{
    array_utils::{numpy_to_dmatrix, dmatrix_to_numpy},
    error::to_py_err,
};

use super::base::{PyManifoldBase, PointType};

/// Python wrapper for the PSD Cone manifold.
///
/// The positive semi-definite cone S_+^n consists of n×n symmetric positive
/// semi-definite matrices: S_+^n = {X ∈ ℝ^{n×n} : X = X^T, X ⪰ 0}.
///
/// Parameters
/// ----------
/// n : int
///     Matrix dimension
#[pyclass(name = "PSDCone", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyPSDCone {
    /// The underlying Rust PSDCone manifold
    pub(crate) inner: PSDCone,
    /// Matrix dimension
    pub(crate) n: usize,
}

impl PyManifoldBase for PyPSDCone {
    fn manifold_name(&self) -> &'static str {
        "PSDCone"
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
impl PyPSDCone {
    /// Create a new PSD Cone manifold.
    ///
    /// Args:
    ///     n: Matrix dimension (must be >= 1)
    ///
    /// Raises:
    ///     ValueError: If n < 1
    #[new]
    pub fn new(n: usize) -> PyResult<Self> {
        if n < 1 {
            return Err(crate::error::value_error(
                "PSDCone manifold requires n >= 1"
            ));
        }
        
        Ok(PyPSDCone {
            inner: PSDCone::new(n).map_err(crate::error::to_py_err)?,
            n,
        })
    }
    
    /// String representation of the manifold.
    fn __repr__(&self) -> String {
        format!("PSDCone(n={})", self.n)
    }
    
    /// Get the matrix dimension.
    #[getter]
    pub fn n(&self) -> usize {
        self.n
    }
    
    /// Check if a point is on the manifold.
    ///
    /// Args:
    ///     point: Matrix to check
    ///     atol: Absolute tolerance (default: 1e-10)
    ///
    /// Returns:
    ///     True if the matrix is symmetric positive semi-definite
    #[pyo3(signature = (point, atol=1e-10))]
    pub fn contains(&self, point: PyReadonlyArray2<'_, f64>, atol: f64) -> PyResult<bool> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        // Validate dimensions
        if point_mat.nrows() != self.n || point_mat.ncols() != self.n {
            return Ok(false);
        }
        
        // Convert matrix to vector representation
        let point_vec = self.inner.matrix_to_vector(&point_mat);
        Ok(self.inner.is_point_on_manifold(&point_vec, atol))
    }
    
    /// Project a point onto the manifold.
    ///
    /// Projects to the nearest symmetric positive semi-definite matrix.
    ///
    /// Args:
    ///     point: Matrix to project
    ///
    /// Returns:
    ///     Projected symmetric positive semi-definite matrix
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_dmatrix(point)?;
        
        // Validate dimensions
        if point_mat.nrows() != self.n || point_mat.ncols() != self.n {
            return Err(crate::error::dimension_mismatch(
                &[self.n, self.n],
                &[point_mat.nrows(), point_mat.ncols()],
            ));
        }
        
        // Convert matrix to vector representation
        let point_vec = self.inner.matrix_to_vector(&point_mat);
        let mut result_vec = DVector::zeros(self.n * (self.n + 1) / 2);
        
        self.inner.project_point(&point_vec, &mut result_vec);
        
        // Convert back to matrix
        let result = self.inner.vector_to_matrix(&result_vec);
        
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
        if point_mat.shape() != (self.n, self.n) || tangent_mat.shape() != (self.n, self.n) {
            return Err(crate::error::dimension_mismatch(
                &[self.n, self.n],
                &[tangent_mat.nrows(), tangent_mat.ncols()],
            ));
        }
        
        // Convert matrices to vector representation
        let point_vec = self.inner.matrix_to_vector(&point_mat);
        let tangent_vec = self.inner.matrix_to_vector(&tangent_mat);
        let mut result_vec = DVector::zeros(self.n * (self.n + 1) / 2);
        
        self.inner.project_tangent(&point_vec, &tangent_vec, &mut result_vec);
        
        // Convert back to matrix
        let result = self.inner.vector_to_matrix(&result_vec);
        
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
        
        // Convert matrices to vector representation
        let point_vec = self.inner.matrix_to_vector(&point_mat);
        let tangent_vec = self.inner.matrix_to_vector(&tangent_mat);
        let mut result_vec = DVector::zeros(self.n * (self.n + 1) / 2);
        
        self.inner.retract(&point_vec, &tangent_vec, &mut result_vec);
        
        // Convert back to matrix
        let result = self.inner.vector_to_matrix(&result_vec);
        
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
        
        // Convert matrices to vector representation
        let x_vec = self.inner.matrix_to_vector(&x_mat);
        let y_vec = self.inner.matrix_to_vector(&y_mat);
        let mut result_vec = DVector::zeros(self.n * (self.n + 1) / 2);
        
        self.inner.inverse_retract(&x_vec, &y_vec, &mut result_vec);
        
        // Convert back to matrix
        let result = self.inner.vector_to_matrix(&result_vec);
        
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
        
        // Convert matrices to vector representation
        let point_vec = self.inner.matrix_to_vector(&point_mat);
        let u_vec = self.inner.matrix_to_vector(&u_mat);
        let v_vec = self.inner.matrix_to_vector(&v_mat);
        
        Ok(self.inner.inner_product(&point_vec, &u_vec, &v_vec)
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
        
        // Convert matrices to vector representation
        let point_vec = self.inner.matrix_to_vector(&point_mat);
        let tangent_vec = self.inner.matrix_to_vector(&tangent_mat);
        
        Ok(self.inner.norm(&point_vec, &tangent_vec)
            .map_err(to_py_err)?)
    }
    
    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random symmetric positive semi-definite matrix
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mut result_vec = DVector::zeros(self.n * (self.n + 1) / 2);
        
        self.inner.random_point(&mut result_vec);
        
        // Convert back to matrix
        let result = self.inner.vector_to_matrix(&result_vec);
        
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
        
        // Convert matrix to vector representation
        let point_vec = self.inner.matrix_to_vector(&point_mat);
        let mut result_vec = DVector::zeros(self.n * (self.n + 1) / 2);
        
        self.inner.random_tangent(&point_vec, &mut result_vec);
        
        // Scale the result if needed
        if scale != 1.0 {
            result_vec *= scale;
        }
        
        // Convert back to matrix
        let result = self.inner.vector_to_matrix(&result_vec);
        
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
        if from_mat.shape() != (self.n, self.n) || to_mat.shape() != (self.n, self.n) || tangent_mat.shape() != (self.n, self.n) {
            return Err(crate::error::dimension_mismatch(
                &[self.n, self.n, self.n, self.n, self.n, self.n],
                &[from_mat.nrows(), from_mat.ncols(), to_mat.nrows(), to_mat.ncols(), tangent_mat.nrows(), tangent_mat.ncols()],
            ));
        }
        
        // Convert matrices to vector representation
        let from_vec = self.inner.matrix_to_vector(&from_mat);
        let to_vec = self.inner.matrix_to_vector(&to_mat);
        let tangent_vec = self.inner.matrix_to_vector(&tangent_mat);
        let mut result_vec = DVector::zeros(self.n * (self.n + 1) / 2);
        
        self.inner.parallel_transport(&from_vec, &to_vec, &tangent_vec, &mut result_vec)
            .map_err(to_py_err)?;
        
        // Convert back to matrix
        let result = self.inner.vector_to_matrix(&result_vec);
        
        dmatrix_to_numpy(py, &result)
    }
}