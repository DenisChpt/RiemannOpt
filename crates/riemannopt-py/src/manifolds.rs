//! Python bindings for Riemannian manifolds.
//!
//! This module provides Python-friendly wrappers around the Rust manifold implementations,
//! with NumPy array integration for seamless interoperability.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix, Dyn};

use riemannopt_manifolds::{
    Sphere, Stiefel, Grassmann, SPD, Hyperbolic,
};
use riemannopt_core::manifold::Manifold;
use crate::array_utils::{dvector_to_pyarray, pyarray_to_dmatrix, dmatrix_to_pyarray, dmatrix_to_dvector, dvector_to_dmatrix};

/// Sphere manifold S^{n-1} in Python.
///
/// The sphere manifold consists of unit vectors in R^n.
#[pyclass(name = "Sphere")]
#[derive(Clone)]
pub struct PySphere {
    inner: Sphere,
}

#[pymethods]
impl PySphere {
    /// Create a new sphere manifold.
    ///
    /// Args:
    ///     dimension: The ambient space dimension (n for S^{n-1})
    #[new]
    pub fn new(dimension: usize) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("Dimension must be positive"));
        }
        Ok(Self {
            inner: Sphere::new(dimension).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.ambient_dimension() - 1
    }
    
    /// Get the manifold dimension (alias).
    #[getter]
    pub fn manifold_dim(&self) -> usize {
        self.inner.ambient_dimension() - 1
    }

    /// Get the ambient space dimension.
    #[getter]
    pub fn ambient_dim(&self) -> usize {
        self.inner.ambient_dimension()
    }

    /// Project a point onto the manifold.
    ///
    /// Args:
    ///     point: Point in ambient space (numpy array)
    ///
    /// Returns:
    ///     Projected point on the sphere
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let projected = self.inner.project_point(&point_vec);
        Ok(dvector_to_pyarray(py, &projected))
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
        point: PyReadonlyArray1<'_, f64>,
        tangent: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let tangent_vec = DVector::from_column_slice(tangent.as_slice()?);
        let retracted = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_pyarray(py, &retracted))
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
        point: PyReadonlyArray1<'_, f64>,
        vector: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let vector_vec = DVector::from_column_slice(vector.as_slice()?);
        let projected = self.inner.project_tangent(&point_vec, &vector_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_pyarray(py, &projected))
    }

    /// Compute the exponential map.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     tangent: Tangent vector
    ///
    /// Returns:
    ///     Exponential map result
    pub fn exp<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
        tangent: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let tangent_vec = DVector::from_column_slice(tangent.as_slice()?);
        let result = self.inner.exp_map(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_pyarray(py, &result))
    }

    /// Compute the logarithmic map.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     other: Another point on the manifold
    ///
    /// Returns:
    ///     Logarithmic map result (tangent vector)
    pub fn log<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
        other: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let other_vec = DVector::from_column_slice(other.as_slice()?);
        match self.inner.log_map(&point_vec, &other_vec) {
            Ok(result) => Ok(dvector_to_pyarray(py, &result)),
            Err(e) => Err(PyValueError::new_err(format!("Logarithm failed: {}", e))),
        }
    }

    /// Compute the Riemannian distance between two points.
    ///
    /// Args:
    ///     x: First point on the manifold
    ///     y: Second point on the manifold
    ///
    /// Returns:
    ///     Distance between the points
    pub fn distance(&self, x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let x_vec = DVector::from_column_slice(x.as_slice()?);
        let y_vec = DVector::from_column_slice(y.as_slice()?);
        Ok(self.inner.distance(&x_vec, &y_vec).map_err(|e| PyValueError::new_err(e.to_string()))?)
    }

    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random point on the sphere
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point = self.inner.random_point();
        Ok(dvector_to_pyarray(py, &point))
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///
    /// Returns:
    ///     Random tangent vector at the point
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let tangent = self.inner.random_tangent(&point_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_pyarray(py, &tangent))
    }
    
    /// Compute the Riemannian inner product.
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
        point: PyReadonlyArray1<'_, f64>,
        u: PyReadonlyArray1<'_, f64>,
        v: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let u_vec = DVector::from_column_slice(u.as_slice()?);
        let v_vec = DVector::from_column_slice(v.as_slice()?);
        self.inner.inner_product(&point_vec, &u_vec, &v_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Alias for retract method (for compatibility).
    pub fn retraction<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
        tangent: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.retract(py, point, tangent)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("Sphere(dimension={})", self.inner.ambient_dimension())
    }
}

impl PySphere {
    /// Get reference to inner Sphere
    pub fn get_inner(&self) -> &Sphere {
        &self.inner
    }
}

/// Stiefel manifold St(n,p) in Python.
///
/// The Stiefel manifold consists of n×p matrices with orthonormal columns.
#[pyclass(name = "Stiefel")]
#[derive(Clone)]
pub struct PyStiefel {
    inner: Stiefel,
}

#[pymethods]
impl PyStiefel {
    /// Create a new Stiefel manifold.
    ///
    /// Args:
    ///     n: Number of rows
    ///     p: Number of columns (p <= n)
    #[new]
    pub fn new(n: usize, p: usize) -> PyResult<Self> {
        if p > n {
            return Err(PyValueError::new_err(
                format!("Stiefel manifold St(n,p) requires p <= n. Got n={}, p={}. \
                        The Stiefel manifold consists of n×p matrices with orthonormal columns, \
                        which is only possible when p <= n.", n, p)
            ));
        }
        if n == 0 || p == 0 {
            return Err(PyValueError::new_err(
                format!("Stiefel manifold dimensions must be positive. Got n={}, p={}. \
                        Both dimensions must be at least 1.", n, p)
            ));
        }
        Ok(Self {
            inner: Stiefel::new(n, p).map_err(|e| PyValueError::new_err(
                format!("Failed to create Stiefel manifold: {}", e)
            ))?,
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.n() * self.inner.p() - self.inner.p() * (self.inner.p() + 1) / 2
    }
    
    /// Get the manifold dimension (alias).
    #[getter]
    pub fn manifold_dim(&self) -> usize {
        self.inner.n() * self.inner.p() - self.inner.p() * (self.inner.p() + 1) / 2
    }

    /// Get the number of rows.
    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n()
    }
    
    /// Get the number of rows (alias).
    #[getter]
    pub fn ambient_rows(&self) -> usize {
        self.inner.n()
    }

    /// Get the number of columns.
    #[getter]
    pub fn p(&self) -> usize {
        self.inner.p()
    }
    
    /// Get the number of columns (alias).
    #[getter]
    pub fn ambient_cols(&self) -> usize {
        self.inner.p()
    }

    /// Project a matrix onto the manifold.
    ///
    /// Args:
    ///     matrix: Matrix in ambient space (numpy array)
    ///
    /// Returns:
    ///     Projected matrix on Stiefel manifold
    pub fn project<'py>(&self, py: Python<'py>, matrix: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = matrix.shape();
        if shape[0] != self.inner.n() || shape[1] != self.inner.p() {
            return Err(PyValueError::new_err(format!(
                "Expected {}x{} matrix, got {}x{}", 
                self.inner.n(), self.inner.p(), shape[0], shape[1]
            )));
        }
        
        // Convert numpy array to nalgebra matrix with proper handling of row-major to column-major
        let mut mat = DMatrix::zeros(shape[0], shape[1]);
        let slice = matrix.as_slice()?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                mat[(i, j)] = slice[i * shape[1] + j];
            }
        }
        
        // Convert to vector for Rust API
        let vec = DVector::from_vec(mat.as_slice().to_vec());
        let projected_vec = self.inner.project_point(&vec);
        
        // Convert back to matrix
        let projected = DMatrix::from_vec(shape[0], shape[1], projected_vec.as_slice().to_vec());
        
        // Convert to numpy array with row-major layout
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(projected[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
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
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut tangent_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let tangent_slice = tangent.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                tangent_mat[(i, j)] = tangent_slice[i * shape[1] + j];
            }
        }
        
        // Convert to vectors
        let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
        let tangent_vec = DVector::from_vec(tangent_mat.as_slice().to_vec());
        
        // Retract
        let retracted_vec = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let retracted = DMatrix::from_vec(shape[0], shape[1], retracted_vec.as_slice().to_vec());
        
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(retracted[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }

    /// Project a matrix onto the tangent space at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     matrix: Matrix in ambient space
    ///
    /// Returns:
    ///     Projected tangent matrix
    pub fn tangent_projection<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        matrix: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut matrix_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let matrix_slice = matrix.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                matrix_mat[(i, j)] = matrix_slice[i * shape[1] + j];
            }
        }
        
        // Convert to vectors
        let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
        let matrix_vec = DVector::from_vec(matrix_mat.as_slice().to_vec());
        
        // Project
        let projected_vec = self.inner.project_tangent(&point_vec, &matrix_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let projected = DMatrix::from_vec(shape[0], shape[1], projected_vec.as_slice().to_vec());
        
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(projected[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }

    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random point on Stiefel manifold
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point = self.inner.random_point();
        // Convert from column-major DVector to row-major numpy array
        // The DVector contains the matrix in column-major order
        let matrix = DMatrix::from_vec(self.inner.n(), self.inner.p(), point.as_slice().to_vec());
        
        // Create numpy array with proper row-major layout
        let mut result = Vec::with_capacity(self.inner.n() * self.inner.p());
        for i in 0..self.inner.n() {
            for j in 0..self.inner.p() {
                result.push(matrix[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([self.inner.n(), self.inner.p()])?)
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Point on the manifold (matrix)
    ///
    /// Returns:
    ///     Random tangent vector at the point (matrix)
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        
        // Convert numpy array to nalgebra matrix
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
            }
        }
        
        // Convert to vector
        let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
        
        // Generate random tangent
        let tangent_vec = self.inner.random_tangent(&point_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let tangent = DMatrix::from_vec(shape[0], shape[1], tangent_vec.as_slice().to_vec());
        
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(tangent[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }

    /// Alias for retract method (for compatibility).
    pub fn retraction<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        tangent: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.retract(py, point, tangent)
    }

    /// Convert Euclidean gradient to Riemannian gradient.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     grad: Euclidean gradient
    ///
    /// Returns:
    ///     Riemannian gradient (projected to tangent space)
    pub fn euclidean_to_riemannian_gradient<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        grad: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // For Stiefel, the Riemannian gradient is the projection of the Euclidean gradient
        // onto the tangent space
        self.tangent_projection(py, point, grad)
    }

    /// Parallel transport a tangent vector along a geodesic.
    ///
    /// Args:
    ///     from_point: Starting point on the manifold
    ///     to_point: Ending point on the manifold
    ///     tangent: Tangent vector to transport
    ///
    /// Returns:
    ///     Transported tangent vector at to_point
    pub fn parallel_transport<'py>(
        &self,
        py: Python<'py>,
        _from_point: PyReadonlyArray2<'_, f64>,
        to_point: PyReadonlyArray2<'_, f64>,
        tangent: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // For now, we use the simplest parallel transport: projection
        // This is not the true parallel transport but a reasonable approximation
        self.tangent_projection(py, to_point, tangent)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("Stiefel(n={}, p={})", self.inner.n(), self.inner.p())
    }
}

impl PyStiefel {
    /// Get reference to inner Stiefel
    pub fn get_inner(&self) -> &Stiefel {
        &self.inner
    }
}

/// Grassmann manifold Gr(n,p) in Python.
///
/// The Grassmann manifold consists of p-dimensional subspaces of R^n.
#[pyclass(name = "Grassmann")]
#[derive(Clone)]
pub struct PyGrassmann {
    inner: Grassmann,
}

#[pymethods]
impl PyGrassmann {
    /// Create a new Grassmann manifold.
    ///
    /// Args:
    ///     n: Ambient space dimension
    ///     p: Subspace dimension (p <= n)
    #[new]
    pub fn new(n: usize, p: usize) -> PyResult<Self> {
        if p > n {
            return Err(PyValueError::new_err("p must be less than or equal to n"));
        }
        if n == 0 || p == 0 {
            return Err(PyValueError::new_err("Dimensions must be positive"));
        }
        Ok(Self {
            inner: Grassmann::new(n, p).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.subspace_dimension() * (self.inner.ambient_dimension() - self.inner.subspace_dimension())
    }
    
    /// Get the manifold dimension (alias).
    #[getter]
    pub fn manifold_dim(&self) -> usize {
        self.inner.subspace_dimension() * (self.inner.ambient_dimension() - self.inner.subspace_dimension())
    }

    /// Get the ambient space dimension.
    #[getter]
    pub fn n(&self) -> usize {
        self.inner.ambient_dimension()
    }
    
    /// Get the ambient space dimension (alias).
    #[getter]
    pub fn ambient_dim(&self) -> usize {
        self.inner.ambient_dimension()
    }

    /// Get the subspace dimension.
    #[getter]
    pub fn p(&self) -> usize {
        self.inner.subspace_dimension()
    }
    
    /// Get the subspace dimension (alias).
    #[getter]
    pub fn subspace_dim(&self) -> usize {
        self.inner.subspace_dimension()
    }

    /// Project a matrix onto the manifold.
    pub fn project<'py>(&self, py: Python<'py>, matrix: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = matrix.shape();
        if shape[0] != self.inner.ambient_dimension() || shape[1] != self.inner.subspace_dimension() {
            return Err(PyValueError::new_err(format!(
                "Expected {}x{} matrix, got {}x{}", 
                self.inner.ambient_dimension(), self.inner.subspace_dimension(), shape[0], shape[1]
            )));
        }
        
        // Convert numpy array to nalgebra matrix with proper handling of row-major to column-major
        let mut mat = DMatrix::zeros(shape[0], shape[1]);
        let slice = matrix.as_slice()?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                mat[(i, j)] = slice[i * shape[1] + j];
            }
        }
        
        // Convert to vector for Rust API
        let vec = DVector::from_vec(mat.as_slice().to_vec());
        let projected_vec = self.inner.project_point(&vec);
        
        // Convert back to matrix
        let projected = DMatrix::from_vec(shape[0], shape[1], projected_vec.as_slice().to_vec());
        
        // Convert to numpy array with row-major layout
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(projected[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
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
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut tangent_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let tangent_slice = tangent.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                tangent_mat[(i, j)] = tangent_slice[i * shape[1] + j];
            }
        }
        
        // Convert to vectors
        let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
        let tangent_vec = DVector::from_vec(tangent_mat.as_slice().to_vec());
        
        // Retract
        let retracted_vec = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let retracted = DMatrix::from_vec(shape[0], shape[1], retracted_vec.as_slice().to_vec());
        
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(retracted[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }
    
    /// Project a matrix onto the tangent space at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     matrix: Matrix in ambient space
    ///
    /// Returns:
    ///     Projected tangent matrix
    pub fn tangent_projection<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        matrix: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut matrix_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let matrix_slice = matrix.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                matrix_mat[(i, j)] = matrix_slice[i * shape[1] + j];
            }
        }
        
        // Convert to vectors
        let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
        let matrix_vec = DVector::from_vec(matrix_mat.as_slice().to_vec());
        
        // Project
        let projected_vec = self.inner.project_tangent(&point_vec, &matrix_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let projected = DMatrix::from_vec(shape[0], shape[1], projected_vec.as_slice().to_vec());
        
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(projected[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }
    
    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random point on Grassmann manifold (orthonormal basis)
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point = self.inner.random_point();
        // Convert from column-major DVector to row-major numpy array
        let matrix = DMatrix::from_vec(self.inner.ambient_dimension(), self.inner.subspace_dimension(), point.as_slice().to_vec());
        
        // Create numpy array with proper row-major layout
        let mut result = Vec::with_capacity(self.inner.ambient_dimension() * self.inner.subspace_dimension());
        for i in 0..self.inner.ambient_dimension() {
            for j in 0..self.inner.subspace_dimension() {
                result.push(matrix[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([self.inner.ambient_dimension(), self.inner.subspace_dimension()])?)
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Point on the manifold (matrix)
    ///
    /// Returns:
    ///     Random tangent vector at the point (matrix)
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        
        // Convert numpy array to nalgebra matrix
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
            }
        }
        
        // Convert to vector
        let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
        
        // Generate random tangent
        let tangent_vec = self.inner.random_tangent(&point_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let tangent = DMatrix::from_vec(shape[0], shape[1], tangent_vec.as_slice().to_vec());
        
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(tangent[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }

    /// Convert Euclidean gradient to Riemannian gradient.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     grad: Euclidean gradient
    ///
    /// Returns:
    ///     Riemannian gradient (projected to horizontal tangent space)
    pub fn euclidean_to_riemannian_gradient<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        grad: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // This is the same as tangent projection for Grassmann
        self.tangent_projection(py, point, grad)
    }
    
    /// Compute the Riemannian distance between two points.
    ///
    /// Args:
    ///     x: First point on the manifold
    ///     y: Second point on the manifold
    ///
    /// Returns:
    ///     Distance between the points
    pub fn distance(&self, x: PyReadonlyArray2<'_, f64>, y: PyReadonlyArray2<'_, f64>) -> PyResult<f64> {
        let shape_x = x.shape();
        let shape_y = y.shape();
        
        if shape_x != shape_y {
            return Err(PyValueError::new_err("Points must have the same shape"));
        }
        
        // Convert numpy arrays to nalgebra matrices
        let mut x_mat = DMatrix::zeros(shape_x[0], shape_x[1]);
        let mut y_mat = DMatrix::zeros(shape_y[0], shape_y[1]);
        let x_slice = x.as_slice()?;
        let y_slice = y.as_slice()?;
        
        for i in 0..shape_x[0] {
            for j in 0..shape_x[1] {
                x_mat[(i, j)] = x_slice[i * shape_x[1] + j];
                y_mat[(i, j)] = y_slice[i * shape_y[1] + j];
            }
        }
        
        // Convert to vectors
        let x_vec = DVector::from_vec(x_mat.as_slice().to_vec());
        let y_vec = DVector::from_vec(y_mat.as_slice().to_vec());
        
        Ok(self.inner.distance(&x_vec, &y_vec).map_err(|e| PyValueError::new_err(e.to_string()))?)
    }

    /// Alias for retract method (for compatibility).
    pub fn retraction<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        tangent: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.retract(py, point, tangent)
    }

    /// Compute the Riemannian inner product.
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
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut u_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut v_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let u_slice = u.as_slice()?;
        let v_slice = v.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                u_mat[(i, j)] = u_slice[i * shape[1] + j];
                v_mat[(i, j)] = v_slice[i * shape[1] + j];
            }
        }
        
        // Convert to vectors
        let point_vec = DVector::from_vec(point_mat.as_slice().to_vec());
        let u_vec = DVector::from_vec(u_mat.as_slice().to_vec());
        let v_vec = DVector::from_vec(v_mat.as_slice().to_vec());
        
        self.inner.inner_product(&point_vec, &u_vec, &v_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("Grassmann(n={}, p={})", self.inner.ambient_dimension(), self.inner.subspace_dimension())
    }
}

impl PyGrassmann {
    /// Get reference to inner Grassmann
    pub fn get_inner(&self) -> &Grassmann {
        &self.inner
    }
}

/// Simple Euclidean manifold implementation.
#[derive(Clone, Debug)]
pub struct SimpleEuclidean {
    pub dimension: usize,
}

#[allow(dead_code)]
impl SimpleEuclidean {
    pub fn new(dimension: usize) -> Self {
        Self { dimension }
    }
    
    pub fn dim(&self) -> usize {
        self.dimension
    }
    
    pub fn project(&self, point: &DVector<f64>) -> DVector<f64> {
        point.clone() // Identity projection for Euclidean space
    }
    
    pub fn retract(&self, point: &DVector<f64>, tangent: &DVector<f64>) -> DVector<f64> {
        point + tangent // Simple addition for Euclidean space
    }
    
    pub fn tangent_projection(&self, _point: &DVector<f64>, vector: &DVector<f64>) -> DVector<f64> {
        vector.clone() // All vectors are tangent vectors in Euclidean space
    }
}

/// Euclidean manifold R^n in Python.
#[pyclass(name = "Euclidean")]
#[derive(Clone)]
pub struct PyEuclidean {
    inner: SimpleEuclidean,
}

#[pymethods]
impl PyEuclidean {
    /// Create a new Euclidean manifold.
    ///
    /// Args:
    ///     dimension: The space dimension
    #[new]
    pub fn new(dimension: usize) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("Dimension must be positive"));
        }
        Ok(Self {
            inner: SimpleEuclidean::new(dimension),
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dim()
    }

    /// Project a point (identity for Euclidean space).
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(point.to_owned_array().into_pyarray_bound(py))
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("Euclidean(dimension={})", self.inner.dimension)
    }
}

impl PyEuclidean {
    /// Get reference to inner Euclidean
    pub fn get_inner(&self) -> &SimpleEuclidean {
        &self.inner
    }
}

/// Symmetric Positive Definite manifold SPD(n) in Python.
///
/// The SPD manifold consists of n×n symmetric positive definite matrices.
#[pyclass(name = "SPD")]
#[derive(Clone)]
pub struct PySPD {
    inner: SPD,
}

#[pymethods]
impl PySPD {
    /// Create a new SPD manifold.
    ///
    /// Args:
    ///     size: Matrix size (n for n×n matrices)
    #[new]
    pub fn new(size: usize) -> PyResult<Self> {
        if size == 0 {
            return Err(PyValueError::new_err(
                "SPD manifold requires matrix dimension n > 0. \
                 The SPD(n) manifold represents n×n symmetric positive definite matrices."
            ));
        }
        Ok(Self {
            inner: SPD::new(size).map_err(|e| PyValueError::new_err(
                format!("Failed to create SPD({}) manifold: {}", size, e)
            ))?,
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        use riemannopt_core::manifold::Manifold;
        <SPD as Manifold<f64, Dyn>>::dimension(&self.inner)
    }
    
    /// Get the manifold dimension (alias).
    #[getter]
    pub fn manifold_dim(&self) -> usize {
        use riemannopt_core::manifold::Manifold;
        <SPD as Manifold<f64, Dyn>>::dimension(&self.inner)
    }

    /// Get the matrix size.
    #[getter]
    pub fn size(&self) -> usize {
        self.inner.matrix_dimension()
    }

    /// Project a matrix onto the manifold.
    ///
    /// Args:
    ///     matrix: Matrix in ambient space (numpy array)
    ///
    /// Returns:
    ///     Projected matrix on SPD manifold
    pub fn project<'py>(&self, py: Python<'py>, matrix: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = matrix.shape();
        if shape[0] != shape[1] {
            return Err(PyValueError::new_err(format!(
                "SPD manifold requires square matrices. Got {}×{} matrix, but matrices must be square (n×n).", 
                shape[0], shape[1]
            )));
        }
        if shape[0] != self.inner.matrix_dimension() {
            return Err(PyValueError::new_err(format!(
                "Matrix dimension mismatch. This is an SPD({}) manifold which expects {}×{} matrices, but got {}×{} matrix.", 
                self.inner.matrix_dimension(), self.inner.matrix_dimension(), self.inner.matrix_dimension(), shape[0], shape[1]
            )));
        }
        
        // Convert numpy array to nalgebra matrix
        let mut mat = DMatrix::zeros(shape[0], shape[1]);
        let slice = matrix.as_slice()?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                mat[(i, j)] = slice[i * shape[1] + j];
            }
        }
        
        // Quick check for NaN or Inf values
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                if !mat[(i, j)].is_finite() {
                    return Err(PyValueError::new_err(
                        "Matrix contains NaN or Inf values. SPD matrices must have finite entries."
                    ));
                }
            }
        }
        
        // Convert matrix to upper triangular vector for Rust API
        let vec = self.matrix_to_vector(&mat);
        let projected_vec = self.inner.project_point(&vec);
        
        // Convert back from upper triangular vector to full matrix
        let projected = self.vector_to_matrix(&projected_vec, shape[0]);
        
        // Convert to numpy array with row-major layout
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(projected[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
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
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut tangent_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let tangent_slice = tangent.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                tangent_mat[(i, j)] = tangent_slice[i * shape[1] + j];
            }
        }
        
        // Convert to upper triangular vectors using helper method
        let point_vec = self.matrix_to_vector(&point_mat);
        let tangent_vec = self.matrix_to_vector(&tangent_mat);
        
        // Retract
        let retracted_vec = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(
                format!("Retraction failed on SPD manifold: {}. \
                        Make sure the base point is symmetric positive definite and the tangent vector is symmetric.", e)
            ))?;
        
        // Convert back to full matrix using helper method
        let retracted = self.vector_to_matrix(&retracted_vec, shape[0]);
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(retracted[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }

    /// Project a matrix onto the tangent space at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     matrix: Matrix in ambient space
    ///
    /// Returns:
    ///     Projected tangent matrix
    pub fn tangent_projection<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        matrix: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut matrix_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let matrix_slice = matrix.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                matrix_mat[(i, j)] = matrix_slice[i * shape[1] + j];
            }
        }
        
        // Convert to upper triangular vectors using helper method
        let point_vec = self.matrix_to_vector(&point_mat);
        let matrix_vec = self.matrix_to_vector(&matrix_mat);
        
        // Project
        let projected_vec = self.inner.project_tangent(&point_vec, &matrix_vec)
            .map_err(|e| PyValueError::new_err(
                format!("Tangent projection failed: {}. \
                        The base point must be a valid SPD matrix (symmetric and positive definite).", e)
            ))?;
        
        // Convert back to full matrix using helper method
        let projected = self.vector_to_matrix(&projected_vec, shape[0]);
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(projected[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }

    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random point on SPD manifold
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let point_vec = self.inner.random_point();
        let n = self.inner.matrix_dimension();
        
        // Convert from upper triangular vector to full matrix
        let matrix = self.vector_to_matrix(&point_vec, n);
        
        // Create numpy array with proper row-major layout
        let mut result = Vec::with_capacity(n * n);
        for i in 0..n {
            for j in 0..n {
                result.push(matrix[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([n, n])?)
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Point on the manifold (symmetric positive definite matrix)
    ///
    /// Returns:
    ///     Random tangent vector at the point (symmetric matrix)
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        
        // Convert numpy array to nalgebra matrix
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
            }
        }
        
        // Convert to upper triangular vector using helper method
        let point_vec = self.matrix_to_vector(&point_mat);
        
        // Generate random tangent
        let tangent_vec = self.inner.random_tangent(&point_vec)
            .map_err(|e| PyValueError::new_err(
                format!("Failed to generate random tangent vector: {}. \
                        The base point must be a valid SPD matrix (symmetric and positive definite).", e)
            ))?;
        
        // Convert back to full matrix using helper method
        let tangent = self.vector_to_matrix(&tangent_vec, shape[0]);
        
        // Convert back to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(tangent[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }

    /// Check if a point is on the manifold.
    ///
    /// Args:
    ///     point: Point to check
    ///     tolerance: Tolerance for checking (default: 1e-10)
    ///
    /// Returns:
    ///     True if point is on the manifold
    #[pyo3(signature = (point, tolerance=None))]
    pub fn is_point_on_manifold(&self, point: PyReadonlyArray2<'_, f64>, tolerance: Option<f64>) -> PyResult<bool> {
        let tol = tolerance.unwrap_or(1e-10);
        let shape = point.shape();
        
        // Convert numpy array to nalgebra matrix
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
            }
        }
        
        // Convert to upper triangular vector
        let point_vec = self.matrix_to_vector(&point_mat);
        
        // Check if on manifold
        Ok(self.inner.is_point_on_manifold(&point_vec, tol))
    }

    /// Check if a vector is in the tangent space.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     vector: Vector to check
    ///     tolerance: Tolerance for checking (default: 1e-10)
    ///
    /// Returns:
    ///     True if vector is in tangent space
    #[pyo3(signature = (point, vector, tolerance=None))]
    pub fn is_vector_in_tangent_space(
        &self,
        point: PyReadonlyArray2<'_, f64>,
        vector: PyReadonlyArray2<'_, f64>,
        tolerance: Option<f64>,
    ) -> PyResult<bool> {
        let tol = tolerance.unwrap_or(1e-10);
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut vector_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let vector_slice = vector.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                vector_mat[(i, j)] = vector_slice[i * shape[1] + j];
            }
        }
        
        // Convert to upper triangular vectors
        let point_vec = self.matrix_to_vector(&point_mat);
        let vector_vec = self.matrix_to_vector(&vector_mat);
        
        // Check if in tangent space
        Ok(self.inner.is_vector_in_tangent_space(&point_vec, &vector_vec, tol))
    }

    /// Compute the Riemannian inner product.
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
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut u_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut v_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let u_slice = u.as_slice()?;
        let v_slice = v.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                u_mat[(i, j)] = u_slice[i * shape[1] + j];
                v_mat[(i, j)] = v_slice[i * shape[1] + j];
            }
        }
        
        // Convert to upper triangular vectors
        let point_vec = self.matrix_to_vector(&point_mat);
        let u_vec = self.matrix_to_vector(&u_mat);
        let v_vec = self.matrix_to_vector(&v_mat);
        
        // Compute inner product
        self.inner.inner_product(&point_vec, &u_vec, &v_vec)
            .map_err(|e| PyValueError::new_err(
                format!("Inner product computation failed: {}. \
                        Ensure the base point is SPD and both vectors are symmetric matrices in the tangent space.", e)
            ))
    }

    /// Compute the Riemannian distance between two points.
    ///
    /// Args:
    ///     point1: First point on the manifold
    ///     point2: Second point on the manifold
    ///
    /// Returns:
    ///     Riemannian distance
    pub fn distance(
        &self,
        point1: PyReadonlyArray2<'_, f64>,
        point2: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<f64> {
        let shape = point1.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point1_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut point2_mat = DMatrix::zeros(shape[0], shape[1]);
        let point1_slice = point1.as_slice()?;
        let point2_slice = point2.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point1_mat[(i, j)] = point1_slice[i * shape[1] + j];
                point2_mat[(i, j)] = point2_slice[i * shape[1] + j];
            }
        }
        
        // Convert to upper triangular vectors
        let point1_vec = self.matrix_to_vector(&point1_mat);
        let point2_vec = self.matrix_to_vector(&point2_mat);
        
        // Compute distance
        self.inner.distance(&point1_vec, &point2_vec)
            .map_err(|e| PyValueError::new_err(
                format!("Distance computation failed: {}. \
                        Both matrices must be symmetric positive definite.", e)
            ))
    }

    /// Compute the inverse retraction (logarithmic map).
    ///
    /// Args:
    ///     point: Base point on the manifold
    ///     other: Target point on the manifold
    ///
    /// Returns:
    ///     Tangent vector at point that maps to other
    pub fn inverse_retract<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        other: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = point.shape();
        
        // Convert numpy arrays to nalgebra matrices
        let mut point_mat = DMatrix::zeros(shape[0], shape[1]);
        let mut other_mat = DMatrix::zeros(shape[0], shape[1]);
        let point_slice = point.as_slice()?;
        let other_slice = other.as_slice()?;
        
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                point_mat[(i, j)] = point_slice[i * shape[1] + j];
                other_mat[(i, j)] = other_slice[i * shape[1] + j];
            }
        }
        
        // Convert to upper triangular vectors
        let point_vec = self.matrix_to_vector(&point_mat);
        let other_vec = self.matrix_to_vector(&other_mat);
        
        // Compute inverse retraction
        let tangent_vec = self.inner.inverse_retract(&point_vec, &other_vec)
            .map_err(|e| PyValueError::new_err(
                format!("Inverse retraction (logarithm) failed: {}. \
                        Both matrices must be symmetric positive definite and the computation may fail if they are too far apart.", e)
            ))?;
        
        // Convert back to full matrix
        let tangent_mat = self.vector_to_matrix(&tangent_vec, shape[0]);
        
        // Convert to numpy
        let mut result = Vec::with_capacity(shape[0] * shape[1]);
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                result.push(tangent_mat[(i, j)]);
            }
        }
        
        let arr = numpy::PyArray1::from_vec_bound(py, result);
        Ok(arr.reshape([shape[0], shape[1]])?)
    }

    /// Alias for project method (for compatibility).
    pub fn project_point<'py>(&self, py: Python<'py>, matrix: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.project(py, matrix)
    }

    /// Alias for retract method (for compatibility).
    pub fn retraction<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        tangent: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.retract(py, point, tangent)
    }

    /// Check if a matrix is symmetric positive definite and provide diagnostic information.
    ///
    /// Args:
    ///     matrix: Matrix to check
    ///
    /// Returns:
    ///     A tuple (is_spd, message) where is_spd is True if the matrix is SPD,
    ///     and message provides diagnostic information if it's not.
    pub fn check_spd_with_info(&self, matrix: PyReadonlyArray2<'_, f64>) -> PyResult<(bool, String)> {
        let shape = matrix.shape();
        
        // Check if square
        if shape[0] != shape[1] {
            return Ok((false, format!("Matrix is not square: {}×{}", shape[0], shape[1])));
        }
        
        // Check dimension
        if shape[0] != self.inner.matrix_dimension() {
            return Ok((false, format!(
                "Wrong dimension: expected {}×{}, got {}×{}", 
                self.inner.matrix_dimension(), self.inner.matrix_dimension(), shape[0], shape[1]
            )));
        }
        
        // Convert to nalgebra matrix
        let mut mat = DMatrix::zeros(shape[0], shape[1]);
        let slice = matrix.as_slice()?;
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                mat[(i, j)] = slice[i * shape[1] + j];
            }
        }
        
        // Check for NaN/Inf
        for i in 0..shape[0] {
            for j in 0..shape[1] {
                if !mat[(i, j)].is_finite() {
                    return Ok((false, format!("Matrix contains non-finite values at position ({}, {})", i, j)));
                }
            }
        }
        
        // Check symmetry
        let mut max_asymmetry = 0.0;
        for i in 0..shape[0] {
            for j in i+1..shape[1] {
                let diff = (mat[(i, j)] - mat[(j, i)]).abs();
                if diff > max_asymmetry {
                    max_asymmetry = diff;
                }
            }
        }
        
        if max_asymmetry > 1e-10 {
            return Ok((false, format!("Matrix is not symmetric. Maximum asymmetry: {:.2e}", max_asymmetry)));
        }
        
        // Check positive definiteness using is_point_on_manifold
        let point_vec = self.matrix_to_vector(&mat);
        if self.inner.is_point_on_manifold(&point_vec, 1e-10) {
            Ok((true, "Matrix is symmetric positive definite".to_string()))
        } else {
            // Try to compute eigenvalues for more info
            let eigen = mat.symmetric_eigen();
            let min_eigenvalue = eigen.eigenvalues.min();
            if min_eigenvalue <= 0.0 {
                Ok((false, format!("Matrix is not positive definite. Minimum eigenvalue: {:.2e}", min_eigenvalue)))
            } else {
                Ok((false, "Matrix failed SPD check (may have numerical issues)".to_string()))
            }
        }
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("SPD(size={})", self.inner.matrix_dimension())
    }
}

impl PySPD {
    /// Get reference to inner SPD
    pub fn get_inner(&self) -> &SPD {
        &self.inner
    }
    
    // Helper function to convert matrix to upper triangular vector
    fn matrix_to_vector(&self, matrix: &DMatrix<f64>) -> DVector<f64> {
        let n = matrix.nrows();
        let vec_size = n * (n + 1) / 2;
        let mut vec_data = Vec::with_capacity(vec_size);
        
        // Extract upper triangular part in column-major order
        for j in 0..n {
            for i in 0..=j {
                vec_data.push(matrix[(i, j)]);
            }
        }
        
        DVector::from_vec(vec_data)
    }
    
    // Helper function to convert upper triangular vector to matrix
    fn vector_to_matrix(&self, vec: &DVector<f64>, n: usize) -> DMatrix<f64> {
        let mut matrix = DMatrix::zeros(n, n);
        let mut idx = 0;
        
        for j in 0..n {
            for i in 0..=j {
                matrix[(i, j)] = vec[idx];
                if i != j {
                    matrix[(j, i)] = vec[idx]; // Symmetric
                }
                idx += 1;
            }
        }
        
        matrix
    }
}

/// Hyperbolic manifold H^n in Python.
///
/// The hyperbolic manifold using the Poincaré ball model.
#[pyclass(name = "Hyperbolic")]
#[derive(Clone)]
pub struct PyHyperbolic {
    inner: Hyperbolic,
}

#[pymethods]
impl PyHyperbolic {
    /// Create a new Hyperbolic manifold.
    ///
    /// Args:
    ///     dimension: The manifold dimension
    #[new]
    pub fn new(dimension: usize) -> PyResult<Self> {
        Ok(Self {
            inner: Hyperbolic::new(dimension).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dimension_space()
    }
    
    /// Get the manifold dimension (alias).
    #[getter]
    pub fn manifold_dim(&self) -> usize {
        self.inner.dimension_space()
    }

    /// Get the ambient space dimension.
    #[getter]
    pub fn ambient_dim(&self) -> usize {
        self.inner.dimension_space()  // Same as manifold dimension for Hyperbolic
    }

    /// Project a point onto the manifold.
    ///
    /// Args:
    ///     point: Point in ambient space (numpy array)
    ///
    /// Returns:
    ///     Projected point on the Poincaré ball
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let projected = self.inner.project_point(&point_vec);
        Ok(dvector_to_pyarray(py, &projected))
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
        point: PyReadonlyArray1<'_, f64>,
        tangent: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let tangent_vec = DVector::from_column_slice(tangent.as_slice()?);
        let retracted = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_pyarray(py, &retracted))
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
        point: PyReadonlyArray1<'_, f64>,
        vector: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let vector_vec = DVector::from_column_slice(vector.as_slice()?);
        let projected = self.inner.project_tangent(&point_vec, &vector_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_pyarray(py, &projected))
    }

    /// Compute the exponential map.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     tangent: Tangent vector
    ///
    /// Returns:
    ///     Exponential map result
    pub fn exp<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
        tangent: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let tangent_vec = DVector::from_column_slice(tangent.as_slice()?);
        // Use retraction as exponential map may not be implemented
        let result = self.inner.retract(&point_vec, &tangent_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_pyarray(py, &result))
    }

    /// Compute the logarithmic map.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     other: Another point on the manifold
    ///
    /// Returns:
    ///     Logarithmic map result (tangent vector)
    pub fn log<'py>(
        &self,
        _py: Python<'py>,
        _point: PyReadonlyArray1<'_, f64>,
        _other: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // Log map may not be implemented, return error for now
        Err(PyValueError::new_err("Logarithmic map not yet implemented for Hyperbolic manifold"))
    }

    /// Compute the Riemannian distance between two points.
    ///
    /// Args:
    ///     x: First point on the manifold
    ///     y: Second point on the manifold
    ///
    /// Returns:
    ///     Distance between the points
    pub fn distance(&self, x: PyReadonlyArray1<'_, f64>, y: PyReadonlyArray1<'_, f64>) -> PyResult<f64> {
        let x_vec = DVector::from_column_slice(x.as_slice()?);
        let y_vec = DVector::from_column_slice(y.as_slice()?);
        Ok(self.inner.distance(&x_vec, &y_vec).map_err(|e| PyValueError::new_err(e.to_string()))?)
    }

    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random point on the Poincaré ball
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point = self.inner.random_point();
        Ok(dvector_to_pyarray(py, &point))
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///
    /// Returns:
    ///     Random tangent vector at the point
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let tangent = self.inner.random_tangent(&point_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_pyarray(py, &tangent))
    }
    
    /// Compute the Riemannian inner product.
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
        point: PyReadonlyArray1<'_, f64>,
        u: PyReadonlyArray1<'_, f64>,
        v: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let u_vec = DVector::from_column_slice(u.as_slice()?);
        let v_vec = DVector::from_column_slice(v.as_slice()?);
        self.inner.inner_product(&point_vec, &u_vec, &v_vec)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Alias for retract method (for compatibility).
    pub fn retraction<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
        tangent: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.retract(py, point, tangent)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("Hyperbolic(dimension={})", self.inner.dimension_space())
    }
}

impl PyHyperbolic {
    /// Get reference to inner Hyperbolic
    pub fn get_inner(&self) -> &Hyperbolic {
        &self.inner
    }
}

/// Check if a point is on the manifold.
#[pyfunction]
#[pyo3(signature = (manifold, point, tolerance=None))]
pub fn check_point_on_manifold(
    manifold: &Bound<'_, PyAny>,
    point: PyReadonlyArray1<'_, f64>,
    tolerance: Option<f64>,
) -> PyResult<bool> {
    let tol = tolerance.unwrap_or(1e-10);
    
    if let Ok(_sphere) = manifold.extract::<PyRef<PySphere>>() {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let norm = point_vec.norm();
        Ok((norm - 1.0).abs() < tol)
    } else {
        Err(PyValueError::new_err("Unsupported manifold type"))
    }
}

/// Check if a vector is in the tangent space.
#[pyfunction]
#[pyo3(signature = (manifold, point, vector, tolerance=None))]
pub fn check_vector_in_tangent_space(
    manifold: &Bound<'_, PyAny>,
    point: PyReadonlyArray1<'_, f64>,
    vector: PyReadonlyArray1<'_, f64>,
    tolerance: Option<f64>,
) -> PyResult<bool> {
    let tol = tolerance.unwrap_or(1e-10);
    
    if let Ok(_sphere) = manifold.extract::<PyRef<PySphere>>() {
        let point_vec = DVector::from_column_slice(point.as_slice()?);
        let vector_vec = DVector::from_column_slice(vector.as_slice()?);
        let inner = point_vec.dot(&vector_vec);
        Ok(inner.abs() < tol)
    } else {
        Err(PyValueError::new_err("Unsupported manifold type"))
    }
}

/// Product manifold M1 × M2 in Python.
///
/// The product manifold allows combining two manifolds.
#[pyclass(name = "ProductManifold")]
pub struct PyProductManifold {
    /// First manifold (as Python object)
    manifold1: PyObject,
    /// Second manifold (as Python object)
    manifold2: PyObject,
    /// Manifold dimension of first manifold
    manifold_dim1: usize,
    /// Manifold dimension of second manifold
    manifold_dim2: usize,
    /// Ambient dimension of first manifold
    ambient_dim1: usize,
    /// Ambient dimension of second manifold
    ambient_dim2: usize,
}

#[pymethods]
impl PyProductManifold {
    /// Create a new product manifold.
    ///
    /// Args:
    ///     manifold1: First component manifold
    ///     manifold2: Second component manifold
    #[new]
    pub fn new(manifold1: PyObject, manifold2: PyObject) -> PyResult<Self> {
        Python::with_gil(|py| {
            // Get dimensions from manifolds
            let manifold_dim1 = manifold1.getattr(py, "manifold_dim")?.extract::<usize>(py)?;
            let manifold_dim2 = manifold2.getattr(py, "manifold_dim")?.extract::<usize>(py)?;
            
            // Get ambient dimensions (for actual array sizes)
            let ambient_dim1 = if let Ok(ambient) = manifold1.getattr(py, "ambient_dim") {
                ambient.extract::<usize>(py)?
            } else {
                // For manifolds where ambient_dim == manifold_dim (like Euclidean, Hyperbolic)
                manifold_dim1
            };
            
            let ambient_dim2 = if let Ok(ambient) = manifold2.getattr(py, "ambient_dim") {
                ambient.extract::<usize>(py)?
            } else {
                manifold_dim2
            };
            
            Ok(Self {
                manifold1,
                manifold2,
                manifold_dim1,
                manifold_dim2,
                ambient_dim1,
                ambient_dim2,
            })
        })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        self.ambient_dim1 + self.ambient_dim2
    }
    
    /// Get the manifold dimension (alias).
    #[getter]
    pub fn manifold_dim(&self) -> usize {
        self.ambient_dim1 + self.ambient_dim2
    }

    /// Get the first component manifold.
    #[getter]
    pub fn manifold1(&self, py: Python<'_>) -> PyObject {
        self.manifold1.clone_ref(py)
    }

    /// Get the second component manifold.
    #[getter]
    pub fn manifold2(&self, py: Python<'_>) -> PyObject {
        self.manifold2.clone_ref(py)
    }

    /// Project a point onto the manifold.
    ///
    /// Args:
    ///     point: Point in ambient space (numpy array)
    ///
    /// Returns:
    ///     Projected point on the product manifold
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = point.as_slice()?;
        if point_vec.len() != self.ambient_dim1 + self.ambient_dim2 {
            return Err(PyValueError::new_err(format!(
                "Expected point of dimension {}, got {}", 
                self.ambient_dim1 + self.ambient_dim2, point_vec.len()
            )));
        }
        
        // Split the point
        let point1 = &point_vec[..self.ambient_dim1];
        let point2 = &point_vec[self.ambient_dim1..];
        
        // Project each component
        let arr1 = numpy::PyArray1::from_slice_bound(py, point1);
        let arr2 = numpy::PyArray1::from_slice_bound(py, point2);
        
        let proj1 = self.manifold1.call_method1(py, "project", (arr1,))?;
        let proj2 = self.manifold2.call_method1(py, "project", (arr2,))?;
        
        // Concatenate results
        let mut result = Vec::with_capacity(self.ambient_dim1 + self.ambient_dim2);
        let proj1_arr: Bound<PyArray1<f64>> = proj1.extract(py)?;
        let proj2_arr: Bound<PyArray1<f64>> = proj2.extract(py)?;
        
        result.extend_from_slice(proj1_arr.readonly().as_slice()?);
        result.extend_from_slice(proj2_arr.readonly().as_slice()?);
        
        Ok(numpy::PyArray1::from_vec_bound(py, result))
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
        point: PyReadonlyArray1<'_, f64>,
        tangent: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = point.as_slice()?;
        let tangent_vec = tangent.as_slice()?;
        
        // Split points and tangents
        let point1 = &point_vec[..self.ambient_dim1];
        let point2 = &point_vec[self.ambient_dim1..];
        let tangent1 = &tangent_vec[..self.ambient_dim1];
        let tangent2 = &tangent_vec[self.ambient_dim1..];
        
        // Create numpy arrays
        let arr_p1 = numpy::PyArray1::from_slice_bound(py, point1);
        let arr_p2 = numpy::PyArray1::from_slice_bound(py, point2);
        let arr_t1 = numpy::PyArray1::from_slice_bound(py, tangent1);
        let arr_t2 = numpy::PyArray1::from_slice_bound(py, tangent2);
        
        // Retract each component
        let ret1 = self.manifold1.call_method1(py, "retract", (arr_p1, arr_t1))?;
        let ret2 = self.manifold2.call_method1(py, "retract", (arr_p2, arr_t2))?;
        
        // Concatenate results
        let mut result = Vec::with_capacity(self.ambient_dim1 + self.ambient_dim2);
        let ret1_arr: Bound<PyArray1<f64>> = ret1.extract(py)?;
        let ret2_arr: Bound<PyArray1<f64>> = ret2.extract(py)?;
        
        result.extend_from_slice(ret1_arr.readonly().as_slice()?);
        result.extend_from_slice(ret2_arr.readonly().as_slice()?);
        
        Ok(numpy::PyArray1::from_vec_bound(py, result))
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
        point: PyReadonlyArray1<'_, f64>,
        vector: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = point.as_slice()?;
        let vector_vec = vector.as_slice()?;
        
        // Split
        let point1 = &point_vec[..self.ambient_dim1];
        let point2 = &point_vec[self.ambient_dim1..];
        let vector1 = &vector_vec[..self.ambient_dim1];
        let vector2 = &vector_vec[self.ambient_dim1..];
        
        // Create numpy arrays
        let arr_p1 = numpy::PyArray1::from_slice_bound(py, point1);
        let arr_p2 = numpy::PyArray1::from_slice_bound(py, point2);
        let arr_v1 = numpy::PyArray1::from_slice_bound(py, vector1);
        let arr_v2 = numpy::PyArray1::from_slice_bound(py, vector2);
        
        // Project each component
        let proj1 = self.manifold1.call_method1(py, "tangent_projection", (arr_p1, arr_v1))?;
        let proj2 = self.manifold2.call_method1(py, "tangent_projection", (arr_p2, arr_v2))?;
        
        // Concatenate results
        let mut result = Vec::with_capacity(self.ambient_dim1 + self.ambient_dim2);
        let proj1_arr: Bound<PyArray1<f64>> = proj1.extract(py)?;
        let proj2_arr: Bound<PyArray1<f64>> = proj2.extract(py)?;
        
        result.extend_from_slice(proj1_arr.readonly().as_slice()?);
        result.extend_from_slice(proj2_arr.readonly().as_slice()?);
        
        Ok(numpy::PyArray1::from_vec_bound(py, result))
    }

    /// Generate a random point on the manifold.
    ///
    /// Returns:
    ///     Random point on the product manifold
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        // Get random points from each manifold
        let rand1 = self.manifold1.call_method0(py, "random_point")?;
        let rand2 = self.manifold2.call_method0(py, "random_point")?;
        
        // Concatenate results
        let mut result = Vec::with_capacity(self.ambient_dim1 + self.ambient_dim2);
        let rand1_arr: Bound<PyArray1<f64>> = rand1.extract(py)?;
        let rand2_arr: Bound<PyArray1<f64>> = rand2.extract(py)?;
        
        result.extend_from_slice(rand1_arr.readonly().as_slice()?);
        result.extend_from_slice(rand2_arr.readonly().as_slice()?);
        
        Ok(numpy::PyArray1::from_vec_bound(py, result))
    }
    
    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///
    /// Returns:
    ///     Random tangent vector at the point
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = point.as_slice()?;
        
        // Split the point
        let point1 = &point_vec[..self.ambient_dim1];
        let point2 = &point_vec[self.ambient_dim1..];
        
        // Create numpy arrays
        let arr_p1 = numpy::PyArray1::from_slice_bound(py, point1);
        let arr_p2 = numpy::PyArray1::from_slice_bound(py, point2);
        
        // Get random tangents from each manifold
        let rand_tan1 = self.manifold1.call_method1(py, "random_tangent", (arr_p1,))?;
        let rand_tan2 = self.manifold2.call_method1(py, "random_tangent", (arr_p2,))?;
        
        // Concatenate results
        let mut result = Vec::with_capacity(self.ambient_dim1 + self.ambient_dim2);
        let rand_tan1_arr: Bound<PyArray1<f64>> = rand_tan1.extract(py)?;
        let rand_tan2_arr: Bound<PyArray1<f64>> = rand_tan2.extract(py)?;
        
        result.extend_from_slice(rand_tan1_arr.readonly().as_slice()?);
        result.extend_from_slice(rand_tan2_arr.readonly().as_slice()?);
        
        Ok(numpy::PyArray1::from_vec_bound(py, result))
    }

    /// Alias for retract method (for compatibility).
    pub fn retraction<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
        tangent: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        self.retract(py, point, tangent)
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        Python::with_gil(|py| {
            let repr1 = self.manifold1.call_method0(py, "__repr__").ok()
                .and_then(|r| r.extract::<String>(py).ok())
                .unwrap_or_else(|| "?".to_string());
            let repr2 = self.manifold2.call_method0(py, "__repr__").ok()
                .and_then(|r| r.extract::<String>(py).ok())
                .unwrap_or_else(|| "?".to_string());
            format!("ProductManifold({} × {})", repr1, repr2)
        })
    }
}