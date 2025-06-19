//! Python bindings for the Oblique manifold.

use numpy::{PyArray2, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix};

use riemannopt_manifolds::Oblique;
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

/// Helper function to flatten matrix to vector in column-major order.
fn matrix_to_vector(mat: &DMatrix<f64>) -> DVector<f64> {
    DVector::from_column_slice(mat.as_slice())
}

/// Helper function to reshape vector to matrix in column-major order.
fn vector_to_matrix(vec: &DVector<f64>, nrows: usize, ncols: usize) -> DMatrix<f64> {
    DMatrix::from_column_slice(nrows, ncols, vec.as_slice())
}

/// Oblique manifold OB(n,p) in Python.
///
/// The oblique manifold consists of n×p matrices with unit-norm columns.
#[pyclass(name = "Oblique")]
#[derive(Clone)]
pub struct PyOblique {
    inner: Oblique,
}

#[pymethods]
impl PyOblique {
    /// Create a new oblique manifold.
    ///
    /// Args:
    ///     n: Number of rows (dimension of each column)
    ///     p: Number of columns
    #[new]
    pub fn new(n: usize, p: usize) -> PyResult<Self> {
        if n == 0 || p == 0 {
            return Err(PyValueError::new_err(
                "Oblique manifold dimensions must be positive. Both n and p must be >= 1."
            ));
        }
        Ok(Self {
            inner: Oblique::new(n, p).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        <Oblique as Manifold<f64, nalgebra::Dyn>>::dimension(&self.inner)
    }

    /// Get the number of rows.
    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n()
    }

    /// Get the number of columns.
    #[getter]
    pub fn p(&self) -> usize {
        self.inner.p()
    }

    /// Project a matrix onto the manifold.
    ///
    /// Args:
    ///     point: Matrix to project (numpy array of shape (n, p))
    ///
    /// Returns:
    ///     Matrix with unit-norm columns
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        if shape[0] != self.inner.n() || shape[1] != self.inner.p() {
            return Err(PyValueError::new_err(
                format!("Point shape mismatch. Expected ({}, {}), got ({}, {})",
                        self.inner.n(), self.inner.p(), shape[0], shape[1])
            ));
        }
        
        // Convert to matrix then to vector
        let mat = numpy_to_nalgebra_matrix(&point)?;
        let vec = matrix_to_vector(&mat);
        let projected_vec = self.inner.project_point(&vec);
        
        // Convert back to matrix
        let projected_mat = vector_to_matrix(&projected_vec, self.inner.n(), self.inner.p());
        nalgebra_to_numpy_matrix(py, &projected_mat)
    }

    /// Compute the retraction at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
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
        
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let tangent_mat = numpy_to_nalgebra_matrix(&tangent)?;
        let point_vec = matrix_to_vector(&point_mat);
        let tangent_vec = matrix_to_vector(&tangent_mat);
        
        let retracted_vec = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let retracted_mat = vector_to_matrix(&retracted_vec, self.inner.n(), self.inner.p());
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
        let shape = point.shape();
        
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let vector_mat = numpy_to_nalgebra_matrix(&vector)?;
        let point_vec = matrix_to_vector(&point_mat);
        let vector_vec = matrix_to_vector(&vector_mat);
        
        let projected_vec = self.inner.project_tangent(&point_vec, &vector_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let projected_mat = vector_to_matrix(&projected_vec, self.inner.n(), self.inner.p());
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
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let u_mat = numpy_to_nalgebra_matrix(&u)?;
        let v_mat = numpy_to_nalgebra_matrix(&v)?;
        let point_vec = matrix_to_vector(&point_mat);
        let u_vec = matrix_to_vector(&u_mat);
        let v_vec = matrix_to_vector(&v_mat);
        
        self.inner.inner_product(&point_vec, &u_vec, &v_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Compute the distance between two points.
    ///
    /// Args:
    ///     x: First point
    ///     y: Second point
    ///
    /// Returns:
    ///     Geodesic distance
    pub fn distance(
        &self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<f64> {
        let x_mat = numpy_to_nalgebra_matrix(&x)?;
        let y_mat = numpy_to_nalgebra_matrix(&y)?;
        let x_vec = matrix_to_vector(&x_mat);
        let y_vec = matrix_to_vector(&y_mat);
        
        self.inner.distance(&x_vec, &y_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random matrix with unit-norm columns
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_vec = self.inner.random_point();
        let point_mat = vector_to_matrix(&point_vec, self.inner.n(), self.inner.p());
        nalgebra_to_numpy_matrix(py, &point_mat)
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
        let point_vec = matrix_to_vector(&point_mat);
        let tangent_vec = self.inner.random_tangent(&point_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        let tangent_mat = vector_to_matrix(&tangent_vec, self.inner.n(), self.inner.p());
        nalgebra_to_numpy_matrix(py, &tangent_mat)
    }

    /// Check if a point is on the manifold.
    ///
    /// Args:
    ///     point: Point to check
    ///
    /// Returns:
    ///     True if the point has unit-norm columns
    pub fn check_point(&self, point: PyReadonlyArray2<'_, f64>) -> PyResult<bool> {
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let point_vec = matrix_to_vector(&point_mat);
        Ok(self.inner.is_point_on_manifold(&point_vec, 1e-6))
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("Oblique(n={}, p={})", self.inner.n(), self.inner.p())
    }
}

impl PyOblique {
    /// Get reference to inner Oblique
    pub fn get_inner(&self) -> &Oblique {
        &self.inner
    }
}