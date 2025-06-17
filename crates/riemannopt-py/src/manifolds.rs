//! Python bindings for Riemannian manifolds.
//!
//! This module provides Python-friendly wrappers around the Rust manifold implementations,
//! with NumPy array integration for seamless interoperability.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix};

use riemannopt_manifolds::{
    Sphere, Stiefel, Grassmann,
};
use riemannopt_core::manifold::Manifold;

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
        Ok(projected.as_slice().to_vec().into_pyarray_bound(py))
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
        Ok(retracted.as_slice().to_vec().into_pyarray_bound(py))
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
        Ok(projected.as_slice().to_vec().into_pyarray_bound(py))
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
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
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
            Ok(result) => Ok(result.as_slice().to_vec().into_pyarray_bound(py)),
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
        Ok(point.as_slice().to_vec().into_pyarray_bound(py))
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
            return Err(PyValueError::new_err("p must be less than or equal to n"));
        }
        if n == 0 || p == 0 {
            return Err(PyValueError::new_err("Dimensions must be positive"));
        }
        Ok(Self {
            inner: Stiefel::new(n, p).map_err(|e| PyValueError::new_err(e.to_string()))?,
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
        
        let mat = DMatrix::from_row_slice(shape[0], shape[1], matrix.as_slice()?);
        let vec = DVector::from_vec(mat.as_slice().to_vec());
        let projected_vec = self.inner.project_point(&vec);
        let projected = DMatrix::from_vec(shape[0], shape[1], projected_vec.as_slice().to_vec());
        
        let result_slice: Vec<f64> = projected.as_slice().to_vec();
        let arr = numpy::PyArray1::from_slice_bound(py, &result_slice);
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

/// Symmetric Positive Definite manifold SPD(n) in Python.
/// NOTE: This is a placeholder - SPD manifold is not yet implemented in the core library.
#[pyclass(name = "SPD")]
#[derive(Clone)]
pub struct PySPD {
    size: usize,
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
            return Err(PyValueError::new_err("Size must be positive"));
        }
        Ok(Self { size })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        self.size * (self.size + 1) / 2
    }

    /// Get the matrix size.
    #[getter]
    pub fn size(&self) -> usize {
        self.size
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("SPD(size={})", self.size)
    }
}

/// Hyperbolic manifold H^n in Python.
/// NOTE: This is a placeholder - Hyperbolic manifold is not yet implemented in the core library.
#[pyclass(name = "Hyperbolic")]
#[derive(Clone)]
pub struct PyHyperbolic {
    dimension: usize,
}

#[pymethods]
impl PyHyperbolic {
    /// Create a new Hyperbolic manifold.
    ///
    /// Args:
    ///     dimension: The manifold dimension
    #[new]
    pub fn new(dimension: usize) -> PyResult<Self> {
        if dimension == 0 {
            return Err(PyValueError::new_err("Dimension must be positive"));
        }
        Ok(Self { dimension })
    }

    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        self.dimension
    }

    /// String representation.
    pub fn __repr__(&self) -> String {
        format!("Hyperbolic(dimension={})", self.dimension)
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