//! Python bindings for the Fixed-Rank manifold.
//!
//! The fixed-rank manifold consists of matrices with a fixed rank constraint.

use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use nalgebra::{DMatrix, DVector};
use riemannopt_manifolds::fixed_rank::{FixedRank, FixedRankPoint, FixedRankTangent};
use riemannopt_core::manifold::Manifold;

use crate::{
    array_utils::{numpy_to_dmatrix, dmatrix_to_numpy, numpy_to_dvector, dvector_to_numpy},
    error::to_py_err,
};

use super::base::{PyManifoldBase, PointType};

/// Python wrapper for a point on the Fixed-Rank manifold.
///
/// A point on the fixed-rank manifold is represented using its SVD factorization:
/// X = U * S * V^T
///
/// where:
/// - U: m × k matrix with orthonormal columns (left singular vectors)
/// - S: k-dimensional vector of singular values
/// - V: n × k matrix with orthonormal columns (right singular vectors)
///
/// Attributes
/// ----------
/// u : ndarray
///     Left singular vectors (m × k matrix)
/// s : ndarray
///     Singular values (k-dimensional vector)
/// v : ndarray
///     Right singular vectors (n × k matrix)
#[pyclass(name = "FixedRankPoint", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyFixedRankPoint {
    /// Internal Rust representation
    pub(crate) inner: FixedRankPoint<f64>,
}

#[pymethods]
impl PyFixedRankPoint {
    /// Create a new FixedRankPoint from SVD components.
    ///
    /// Args:
    ///     u: Left singular vectors (m × k matrix with orthonormal columns)
    ///     s: Singular values (k-dimensional positive vector)
    ///     v: Right singular vectors (n × k matrix with orthonormal columns)
    ///
    /// Returns:
    ///     FixedRankPoint: The point on the fixed-rank manifold
    #[new]
    pub fn new(
        u: PyReadonlyArray2<'_, f64>,
        s: PyReadonlyArray1<'_, f64>,
        v: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let u_mat = numpy_to_dmatrix(u)?;
        let s_vec = numpy_to_dvector(s)?;
        let v_mat = numpy_to_dmatrix(v)?;
        
        // Validate dimensions
        let k = s_vec.len();
        if u_mat.ncols() != k {
            return Err(crate::error::dimension_mismatch(
                &[u_mat.nrows(), k],
                &[u_mat.nrows(), u_mat.ncols()],
            ));
        }
        if v_mat.ncols() != k {
            return Err(crate::error::dimension_mismatch(
                &[v_mat.nrows(), k],
                &[v_mat.nrows(), v_mat.ncols()],
            ));
        }
        
        Ok(PyFixedRankPoint {
            inner: FixedRankPoint::new(u_mat, s_vec, v_mat),
        })
    }
    
    /// Get the left singular vectors U.
    #[getter]
    pub fn u<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        dmatrix_to_numpy(py, &self.inner.u)
    }
    
    /// Get the singular values S.
    #[getter]
    pub fn s<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        dvector_to_numpy(py, &self.inner.s)
    }
    
    /// Get the right singular vectors V.
    #[getter]
    pub fn v<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        dmatrix_to_numpy(py, &self.inner.v)
    }
    
    /// Convert to full matrix representation X = U * S * V^T.
    ///
    /// Returns:
    ///     ndarray: The full m × n matrix
    pub fn to_matrix<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat = self.inner.to_matrix();
        dmatrix_to_numpy(py, &mat)
    }
    
    /// Create from a full matrix using truncated SVD.
    ///
    /// Args:
    ///     matrix: The m × n matrix to decompose
    ///     k: The target rank
    ///
    /// Returns:
    ///     FixedRankPoint: The rank-k approximation
    #[staticmethod]
    pub fn from_matrix(matrix: PyReadonlyArray2<'_, f64>, k: usize) -> PyResult<Self> {
        let mat = numpy_to_dmatrix(matrix)?;
        let point = FixedRankPoint::from_matrix(&mat, k).map_err(to_py_err)?;
        Ok(PyFixedRankPoint { inner: point })
    }
    
    fn __repr__(&self) -> String {
        format!(
            "FixedRankPoint(m={}, n={}, k={})",
            self.inner.u.nrows(),
            self.inner.v.nrows(),
            self.inner.s.len()
        )
    }
}

/// Python wrapper for a tangent vector on the Fixed-Rank manifold.
///
/// A tangent vector at a point X = U*S*V^T is represented as:
/// ξ = U_perp*M*V^T + U*S_dot*V^T + U*N*V_perp^T
///
/// where:
/// - U_perp*M*V^T: Component in the direction of increasing left singular space
/// - U*S_dot*V^T: Component changing the singular values
/// - U*N*V_perp^T: Component in the direction of increasing right singular space
///
/// Attributes
/// ----------
/// u_perp_m : ndarray
///     Left orthogonal component (m-k) × k matrix
/// s_dot : ndarray
///     Singular value derivatives (k-dimensional vector)
/// v_perp_n : ndarray
///     Right orthogonal component (n-k) × k matrix
#[pyclass(name = "FixedRankTangent", module = "riemannopt.manifolds")]
#[derive(Clone)]
pub struct PyFixedRankTangent {
    /// Internal Rust representation
    pub(crate) inner: FixedRankTangent<f64>,
}

#[pymethods]
impl PyFixedRankTangent {
    /// Create a new FixedRankTangent from components.
    ///
    /// Args:
    ///     u_perp_m: Left orthogonal component ((m-k) × k matrix)
    ///     s_dot: Singular value derivatives (k-dimensional vector)
    ///     v_perp_n: Right orthogonal component ((n-k) × k matrix)
    ///
    /// Returns:
    ///     FixedRankTangent: The tangent vector
    #[new]
    pub fn new(
        u_perp_m: PyReadonlyArray2<'_, f64>,
        s_dot: PyReadonlyArray1<'_, f64>,
        v_perp_n: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Self> {
        let u_perp_m_mat = numpy_to_dmatrix(u_perp_m)?;
        let s_dot_vec = numpy_to_dvector(s_dot)?;
        let v_perp_n_mat = numpy_to_dmatrix(v_perp_n)?;
        
        Ok(PyFixedRankTangent {
            inner: FixedRankTangent::new(u_perp_m_mat, s_dot_vec, v_perp_n_mat),
        })
    }
    
    /// Get the left orthogonal component.
    #[getter]
    pub fn u_perp_m<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        dmatrix_to_numpy(py, &self.inner.u_perp_m)
    }
    
    /// Get the singular value derivatives.
    #[getter]
    pub fn s_dot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        dvector_to_numpy(py, &self.inner.s_dot)
    }
    
    /// Get the right orthogonal component.
    #[getter]
    pub fn v_perp_n<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        dmatrix_to_numpy(py, &self.inner.v_perp_n)
    }
    
    /// Convert to full matrix representation given a base point.
    ///
    /// Args:
    ///     point: The base point (FixedRankPoint)
    ///
    /// Returns:
    ///     ndarray: The full m × n tangent matrix
    pub fn to_matrix<'py>(
        &self, 
        py: Python<'py>, 
        point: &PyFixedRankPoint
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let mat = self.inner.to_matrix(&point.inner);
        dmatrix_to_numpy(py, &mat)
    }
    
    fn __repr__(&self) -> String {
        let (m_k, k1) = (self.inner.u_perp_m.nrows(), self.inner.u_perp_m.ncols());
        let k2 = self.inner.s_dot.len();
        let (n_k, k3) = (self.inner.v_perp_n.nrows(), self.inner.v_perp_n.ncols());
        format!(
            "FixedRankTangent(u_perp_m={}×{}, s_dot={}, v_perp_n={}×{})",
            m_k, k1, k2, n_k, k3
        )
    }
}

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
        
        self.inner.project_point(&point_mat, &mut result);
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
            .map_err(to_py_err)?;

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
            .map_err(to_py_err)?;

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
            .map_err(to_py_err)?;

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
            .map_err(to_py_err)?;

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