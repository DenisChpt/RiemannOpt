//! Python bindings for the Fixed-rank manifold.

use numpy::{PyArray2, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix};

use riemannopt_manifolds::{FixedRank, FixedRankPoint};
use riemannopt_core::manifold::Manifold;

/// Helper function to convert NumPy array to nalgebra matrix.
fn numpy_to_nalgebra_matrix(array: &PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let shape = array.shape();
    let mut mat = DMatrix::zeros(shape[0], shape[1]);
    let slice = array.as_slice()?;
    
    // Copy element by element to handle row-major to column-major conversion
    for i in 0..shape[0] {
        for j in 0..shape[1] {
            mat[(i, j)] = slice[i * shape[1] + j];
        }
    }
    
    Ok(mat)
}

/// Helper function to convert nalgebra matrix to NumPy array.
fn nalgebra_to_numpy_matrix<'py>(
    py: Python<'py>,
    mat: &DMatrix<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (nrows, ncols) = mat.shape();
    let mut data = Vec::with_capacity(nrows * ncols);
    
    // Convert column-major to row-major
    for i in 0..nrows {
        for j in 0..ncols {
            data.push(mat[(i, j)]);
        }
    }
    
    let flat_array = numpy::PyArray1::from_vec_bound(py, data);
    Ok(flat_array.reshape([nrows, ncols])?.into())
}

/// Fixed-rank manifold M_k(m,n) in Python.
///
/// The manifold of m×n matrices with fixed rank k.
#[pyclass(name = "FixedRank")]
#[derive(Clone)]
pub struct PyFixedRank {
    inner: FixedRank,
}

#[pymethods]
impl PyFixedRank {
    /// Create a new fixed-rank manifold.
    ///
    /// Args:
    ///     m: Number of rows
    ///     n: Number of columns  
    ///     k: Rank (must satisfy k ≤ min(m, n))
    #[new]
    pub fn new(m: usize, n: usize, k: usize) -> PyResult<Self> {
        if m == 0 || n == 0 || k == 0 {
            return Err(PyValueError::new_err(
                "Fixed-rank manifold dimensions must be positive. m, n, and k must be >= 1."
            ));
        }
        if k > m.min(n) {
            return Err(PyValueError::new_err(
                format!("Rank k={} cannot exceed min(m={}, n={})", k, m, n)
            ));
        }
        Ok(Self {
            inner: FixedRank::new(m, n, k).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        <FixedRank as Manifold<f64, nalgebra::Dyn>>::dimension(&self.inner)
    }

    /// Get the number of rows.
    #[getter]
    pub fn m(&self) -> usize {
        self.inner.m()
    }

    /// Get the number of columns.
    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n()
    }

    /// Get the rank.
    #[getter]
    pub fn k(&self) -> usize {
        self.inner.k()
    }

    /// Project a matrix onto the manifold.
    ///
    /// This performs a truncated SVD to get the best rank-k approximation.
    ///
    /// Args:
    ///     point: Matrix to project (numpy array of shape (m, n))
    ///
    /// Returns:
    ///     Best rank-k approximation
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        if shape[0] != self.inner.m() || shape[1] != self.inner.n() {
            return Err(PyValueError::new_err(
                format!("Point shape mismatch. Expected ({}, {}), got ({}, {})",
                        self.inner.m(), self.inner.n(), shape[0], shape[1])
            ));
        }
        
        // Convert to matrix and create FixedRankPoint
        let mat = numpy_to_nalgebra_matrix(&point)?;
        let fr_point = FixedRankPoint::from_matrix(&mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Convert to vector, project, and convert back
        let vec = fr_point.to_vector();
        let projected_vec = self.inner.project_point(&vec);
        
        // Convert back to matrix form
        let projected_point = FixedRankPoint::<f64>::from_vector(&projected_vec, self.inner.m(), self.inner.n(), self.inner.k());
        let projected_mat = projected_point.to_matrix();
        
        nalgebra_to_numpy_matrix(py, &projected_mat)
    }

    /// Compute the retraction at a point.
    ///
    /// Args:
    ///     point: Point on the manifold (rank-k matrix)
    ///     tangent: Tangent vector
    ///
    /// Returns:
    ///     Retracted point on the manifold
    pub fn retract<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        tangent: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_shape = point.shape();
        let tangent_shape = tangent.shape();
        
        if point_shape != tangent_shape {
            return Err(PyValueError::new_err(
                format!("Shape mismatch: point has shape {:?}, tangent has shape {:?}",
                        point_shape, tangent_shape)
            ));
        }
        
        // Convert to FixedRankPoint representations
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let tangent_mat = numpy_to_nalgebra_matrix(&tangent)?;
        
        // Check if tangent is zero
        let is_zero_tangent = tangent_mat.iter().all(|&x| x.abs() < 1e-14);
        if is_zero_tangent {
            // If tangent is zero, return the point unchanged
            return nalgebra_to_numpy_matrix(py, &point_mat);
        }
        
        let point_fr = FixedRankPoint::from_matrix(&point_mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let tangent_fr = FixedRankPoint::from_matrix(&tangent_mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let point_vec = point_fr.to_vector();
        let tangent_vec = tangent_fr.to_vector();
        
        let retracted_vec = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let retracted_point = FixedRankPoint::<f64>::from_vector(&retracted_vec, self.inner.m(), self.inner.n(), self.inner.k());
        let retracted_mat = retracted_point.to_matrix();
        
        nalgebra_to_numpy_matrix(py, &retracted_mat)
    }

    /// Project a vector onto the tangent space at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     vector: Vector in ambient space
    ///
    /// Returns:
    ///     Projected tangent vector
    pub fn tangent_projection<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        vector: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Convert to FixedRankPoint representations
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let vector_mat = numpy_to_nalgebra_matrix(&vector)?;
        
        let point_fr = FixedRankPoint::from_matrix(&point_mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let vector_fr = FixedRankPoint::from_matrix(&vector_mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let point_vec = point_fr.to_vector();
        let vector_vec = vector_fr.to_vector();
        
        let projected_vec = self.inner.project_tangent(&point_vec, &vector_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let projected_point = FixedRankPoint::<f64>::from_vector(&projected_vec, self.inner.m(), self.inner.n(), self.inner.k());
        let projected_mat = projected_point.to_matrix();
        
        nalgebra_to_numpy_matrix(py, &projected_mat)
    }

    /// Inner product in the tangent space.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     u: First tangent vector
    ///     v: Second tangent vector
    ///
    /// Returns:
    ///     Inner product value
    pub fn inner_product(
        &self,
        point: PyReadonlyArray2<'_, f64>,
        u: PyReadonlyArray2<'_, f64>,
        v: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<f64> {
        // Convert to FixedRankPoint representations
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let u_mat = numpy_to_nalgebra_matrix(&u)?;
        let v_mat = numpy_to_nalgebra_matrix(&v)?;
        
        let point_fr = FixedRankPoint::from_matrix(&point_mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let u_fr = FixedRankPoint::from_matrix(&u_mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let v_fr = FixedRankPoint::from_matrix(&v_mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let point_vec = point_fr.to_vector();
        let u_vec = u_fr.to_vector();
        let v_vec = v_fr.to_vector();
        
        self.inner.inner_product(&point_vec, &u_vec, &v_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random rank-k matrix
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_vec = self.inner.random_point();
        let point = FixedRankPoint::<f64>::from_vector(&point_vec, self.inner.m(), self.inner.n(), self.inner.k());
        let mat = point.to_matrix();
        nalgebra_to_numpy_matrix(py, &mat)
    }

    /// Generate a random tangent vector.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///
    /// Returns:
    ///     Random tangent vector
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let point_fr = FixedRankPoint::from_matrix(&point_mat, self.inner.k())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let point_vec = point_fr.to_vector();
        
        let tangent_vec = self.inner.random_tangent(&point_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let tangent_point = FixedRankPoint::<f64>::from_vector(&tangent_vec, self.inner.m(), self.inner.n(), self.inner.k());
        let tangent_mat = tangent_point.to_matrix();
        
        nalgebra_to_numpy_matrix(py, &tangent_mat)
    }

    /// Check if a point is on the manifold.
    ///
    /// Args:
    ///     point: Point to check
    ///
    /// Returns:
    ///     True if the point has rank k
    pub fn check_point(&self, point: PyReadonlyArray2<'_, f64>) -> PyResult<bool> {
        let shape = point.shape();
        if shape[0] != self.inner.m() || shape[1] != self.inner.n() {
            return Err(PyValueError::new_err(
                format!("Point shape mismatch. Expected ({}, {}), got ({}, {})",
                        self.inner.m(), self.inner.n(), shape[0], shape[1])
            ));
        }
        
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        
        // Check rank using SVD
        let svd = nalgebra::SVD::new(point_mat, true, true);
        let singular_values = &svd.singular_values;
        
        // Count non-zero singular values
        let mut rank = 0;
        let tol = 1e-10;
        for &s in singular_values.iter() {
            if s > tol {
                rank += 1;
            }
        }
        
        Ok(rank == self.inner.k())
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("FixedRank(m={}, n={}, k={})", self.inner.m(), self.inner.n(), self.inner.k())
    }
}

impl PyFixedRank {
    /// Get reference to inner FixedRank
    pub fn get_inner(&self) -> &FixedRank {
        &self.inner
    }
}