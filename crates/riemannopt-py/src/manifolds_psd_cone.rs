//! Python bindings for the PSD cone manifold.

use numpy::{PyArray2, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::DMatrix;

use riemannopt_manifolds::PSDCone;
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

/// Positive Semi-Definite cone S^n_+ in Python.
///
/// The manifold of n×n symmetric positive semi-definite matrices.
#[pyclass(name = "PSDCone")]
#[derive(Clone)]
pub struct PyPSDCone {
    inner: PSDCone,
}

#[pymethods]
impl PyPSDCone {
    /// Create a new PSD cone manifold.
    ///
    /// Args:
    ///     n: Size of the matrices (n×n)
    #[new]
    pub fn new(n: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err(
                "PSD cone dimension must be positive. n must be >= 1."
            ));
        }
        Ok(Self {
            inner: PSDCone::new(n).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        <PSDCone as Manifold<f64, nalgebra::Dyn>>::dimension(&self.inner)
    }

    /// Get the size n.
    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n()
    }

    /// Project a matrix onto the PSD cone.
    ///
    /// This projects to the nearest PSD matrix in Frobenius norm.
    ///
    /// Args:
    ///     point: Matrix to project (numpy array of shape (n, n))
    ///
    /// Returns:
    ///     Projected PSD matrix
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        if shape[0] != self.inner.n() || shape[1] != self.inner.n() {
            return Err(PyValueError::new_err(
                format!("Point shape mismatch. Expected ({}, {}), got ({}, {})",
                        self.inner.n(), self.inner.n(), shape[0], shape[1])
            ));
        }
        
        // Convert to matrix
        let mat = numpy_to_nalgebra_matrix(&point)?;
        
        // Convert to vector form for manifold operations
        let n = self.inner.n();
        let dim = n * (n + 1) / 2;
        let mut vec = nalgebra::DVector::zeros(dim);
        let mut idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    vec[idx] = mat[(i, j)];
                } else {
                    vec[idx] = mat[(i, j)] * std::f64::consts::SQRT_2;
                }
                idx += 1;
            }
        }
        
        let projected_vec = self.inner.project_point(&vec);
        
        // Convert back to matrix form
        let mut proj_mat = DMatrix::zeros(n, n);
        idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    proj_mat[(i, j)] = projected_vec[idx];
                } else {
                    let val = projected_vec[idx] / std::f64::consts::SQRT_2;
                    proj_mat[(i, j)] = val;
                    proj_mat[(j, i)] = val;
                }
                idx += 1;
            }
        }
        
        nalgebra_to_numpy_matrix(py, &proj_mat)
    }

    /// Compute the retraction at a point.
    ///
    /// Args:
    ///     point: Point on the manifold (PSD matrix)
    ///     tangent: Tangent vector (symmetric matrix)
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
        
        // Convert to matrices
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let tangent_mat = numpy_to_nalgebra_matrix(&tangent)?;
        
        // Convert to vector form
        let n = self.inner.n();
        let dim = n * (n + 1) / 2;
        let mut point_vec = nalgebra::DVector::zeros(dim);
        let mut tangent_vec = nalgebra::DVector::zeros(dim);
        let mut idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    point_vec[idx] = point_mat[(i, j)];
                    tangent_vec[idx] = tangent_mat[(i, j)];
                } else {
                    point_vec[idx] = point_mat[(i, j)] * std::f64::consts::SQRT_2;
                    tangent_vec[idx] = tangent_mat[(i, j)] * std::f64::consts::SQRT_2;
                }
                idx += 1;
            }
        }
        
        let retracted_vec = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Convert back to matrix
        let mut retracted_mat = DMatrix::zeros(n, n);
        idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    retracted_mat[(i, j)] = retracted_vec[idx];
                } else {
                    let val = retracted_vec[idx] / std::f64::consts::SQRT_2;
                    retracted_mat[(i, j)] = val;
                    retracted_mat[(j, i)] = val;
                }
                idx += 1;
            }
        }
        
        nalgebra_to_numpy_matrix(py, &retracted_mat)
    }

    /// Project a vector onto the tangent space at a point.
    ///
    /// For the PSD cone, this amounts to symmetrization.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     vector: Vector in ambient space
    ///
    /// Returns:
    ///     Projected tangent vector (symmetric matrix)
    pub fn tangent_projection<'py>(
        &self,
        py: Python<'py>,
        _point: PyReadonlyArray2<'_, f64>,
        vector: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let vector_mat = numpy_to_nalgebra_matrix(&vector)?;
        
        // Symmetrize
        let sym = (vector_mat.clone() + vector_mat.transpose()) / 2.0;
        
        nalgebra_to_numpy_matrix(py, &sym)
    }

    /// Inner product in the tangent space.
    ///
    /// Uses the Frobenius inner product.
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
        let u_mat = numpy_to_nalgebra_matrix(&u)?;
        let v_mat = numpy_to_nalgebra_matrix(&v)?;
        
        // Frobenius inner product: tr(U^T V)
        let inner = (u_mat.transpose() * v_mat).trace();
        
        Ok(inner)
    }

    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random PSD matrix
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_vec = self.inner.random_point();
        
        // Convert to matrix
        let n = self.inner.n();
        let mut mat = DMatrix::zeros(n, n);
        let mut idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    mat[(i, j)] = point_vec[idx];
                } else {
                    let val = point_vec[idx] / std::f64::consts::SQRT_2;
                    mat[(i, j)] = val;
                    mat[(j, i)] = val;
                }
                idx += 1;
            }
        }
        
        nalgebra_to_numpy_matrix(py, &mat)
    }

    /// Generate a random tangent vector.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///
    /// Returns:
    ///     Random tangent vector (symmetric matrix)
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Convert point to vector form
        let point_mat = numpy_to_nalgebra_matrix(&point)?;
        let n = self.inner.n();
        let dim = n * (n + 1) / 2;
        let mut point_vec = nalgebra::DVector::zeros(dim);
        let mut idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    point_vec[idx] = point_mat[(i, j)];
                } else {
                    point_vec[idx] = point_mat[(i, j)] * std::f64::consts::SQRT_2;
                }
                idx += 1;
            }
        }
        
        let tangent_vec = self.inner.random_tangent(&point_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        
        // Convert to matrix
        let mut tangent_mat = DMatrix::zeros(n, n);
        idx = 0;
        
        for i in 0..n {
            for j in i..n {
                if i == j {
                    tangent_mat[(i, j)] = tangent_vec[idx];
                } else {
                    let val = tangent_vec[idx] / std::f64::consts::SQRT_2;
                    tangent_mat[(i, j)] = val;
                    tangent_mat[(j, i)] = val;
                }
                idx += 1;
            }
        }
        
        nalgebra_to_numpy_matrix(py, &tangent_mat)
    }

    /// Check if a point is in the PSD cone.
    ///
    /// Args:
    ///     point: Point to check
    ///
    /// Returns:
    ///     True if the matrix is PSD
    pub fn check_point(&self, point: PyReadonlyArray2<'_, f64>) -> PyResult<bool> {
        let shape = point.shape();
        if shape[0] != self.inner.n() || shape[1] != self.inner.n() {
            return Err(PyValueError::new_err(
                format!("Point shape mismatch. Expected ({}, {}), got ({}, {})",
                        self.inner.n(), self.inner.n(), shape[0], shape[1])
            ));
        }
        
        let mat = numpy_to_nalgebra_matrix(&point)?;
        
        // Check symmetry
        for i in 0..self.inner.n() {
            for j in i+1..self.inner.n() {
                if (mat[(i, j)] - mat[(j, i)]).abs() > 1e-10 {
                    return Ok(false);
                }
            }
        }
        
        // Check positive semi-definiteness via eigenvalues
        let eigen = mat.symmetric_eigen();
        Ok(eigen.eigenvalues.iter().all(|&lambda| lambda >= -1e-10))
    }

    /// Compute the distance between two points.
    ///
    /// Uses the Frobenius distance.
    ///
    /// Args:
    ///     x: First point
    ///     y: Second point
    ///
    /// Returns:
    ///     Distance between x and y
    pub fn distance(
        &self,
        x: PyReadonlyArray2<'_, f64>,
        y: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<f64> {
        let x_mat = numpy_to_nalgebra_matrix(&x)?;
        let y_mat = numpy_to_nalgebra_matrix(&y)?;
        
        let diff = x_mat - y_mat;
        let dist = (diff.transpose() * diff).trace().sqrt();
        
        Ok(dist)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("PSDCone(n={})", self.inner.n())
    }
}

impl PyPSDCone {
    /// Get reference to inner PSDCone
    pub fn get_inner(&self) -> &PSDCone {
        &self.inner
    }
}