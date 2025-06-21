//! Optimized manifold implementations with better performance.

use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods, PyArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::DMatrix;

use riemannopt_manifolds::{Stiefel, Grassmann, SPD};
use crate::array_utils::{pyarray_to_dmatrix, dmatrix_to_pyarray};

/// Optimized Stiefel manifold implementation.
#[pyclass(name = "Stiefel")]
#[derive(Clone)]
pub struct PyStiefel {
    inner: Stiefel,
}

#[pymethods]
impl PyStiefel {
    #[new]
    pub fn new(n: usize, p: usize) -> PyResult<Self> {
        if n < p {
            return Err(PyValueError::new_err("n must be >= p"));
        }
        Ok(Self {
            inner: Stiefel::new(n, p).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    #[getter]
    pub fn n(&self) -> usize {
        self.inner.n()
    }

    #[getter]
    pub fn p(&self) -> usize {
        self.inner.p()
    }

    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.n() * self.inner.p() - self.inner.p() * (self.inner.p() + 1) / 2
    }

    /// Optimized projection using direct matrix operations.
    pub fn project<'py>(&self, py: Python<'py>, matrix: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = matrix.shape();
        if shape[0] != self.inner.n() || shape[1] != self.inner.p() {
            return Err(PyValueError::new_err(format!(
                "Expected {}x{} matrix, got {}x{}", 
                self.inner.n(), self.inner.p(), shape[0], shape[1]
            )));
        }
        
        // Direct conversion without intermediate allocations
        let mat = pyarray_to_dmatrix(&matrix)?;
        
        // Use QR decomposition directly on the matrix
        let qr = mat.qr();
        let (q, _r) = (qr.q(), qr.r());
        
        // Extract the first p columns of Q
        let result = q.columns(0, self.inner.p()).into_owned();
        
        // Convert back to numpy
        dmatrix_to_pyarray(py, &result)
    }

    /// Optimized retraction.
    pub fn retract<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        tangent: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let _shape = point.shape();
        
        // Direct matrix operations
        let x = pyarray_to_dmatrix(&point)?;
        let v = pyarray_to_dmatrix(&tangent)?;
        
        // QR retraction: qr(X + V)
        let sum = &x + &v;
        let qr = sum.qr();
        let result = qr.q().columns(0, self.inner.p()).into_owned();
        
        dmatrix_to_pyarray(py, &result)
    }

    /// Optimized tangent projection.
    pub fn tangent_projection<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        vector: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Direct matrix operations
        let x = pyarray_to_dmatrix(&point)?;
        let v = pyarray_to_dmatrix(&vector)?;
        
        // Tangent projection: V - X * (X^T * V + V^T * X) / 2
        let xtv = x.transpose() * &v;
        let sym = &xtv + xtv.transpose();
        let result = &v - &x * (sym * 0.5);
        
        dmatrix_to_pyarray(py, &result)
    }

    /// Generate random point efficiently.
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Generate random Gaussian matrix
        let mut mat = DMatrix::zeros(self.inner.n(), self.inner.p());
        for i in 0..self.inner.n() {
            for j in 0..self.inner.p() {
                mat[(i, j)] = rng.sample(rand_distr::StandardNormal);
            }
        }
        
        // QR decomposition for orthogonalization
        let qr = mat.qr();
        let result = qr.q().columns(0, self.inner.p()).into_owned();
        
        dmatrix_to_pyarray(py, &result)
    }

    pub fn __repr__(&self) -> String {
        format!("Stiefel(n={}, p={})", self.inner.n(), self.inner.p())
    }
}

// Non-Python methods for PyStiefel
impl PyStiefel {
    /// Get access to the inner Rust manifold (for compatibility with optimizer code).
    /// This method is not exposed to Python.
    pub fn get_inner(&self) -> &Stiefel {
        &self.inner
    }
}

/// Optimized Grassmann manifold implementation.
#[pyclass(name = "Grassmann")]
#[derive(Clone)]
pub struct PyGrassmann {
    inner: Grassmann,
}

#[pymethods]
impl PyGrassmann {
    #[new]
    pub fn new(n: usize, p: usize) -> PyResult<Self> {
        if n <= p || p == 0 {
            return Err(PyValueError::new_err("Grassmann manifold requires 0 < p < n"));
        }
        Ok(Self {
            inner: Grassmann::new(n, p).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    #[getter]
    pub fn n(&self) -> usize {
        self.inner.ambient_dimension()
    }

    #[getter]
    pub fn p(&self) -> usize {
        self.inner.subspace_dimension()
    }

    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.subspace_dimension() * (self.inner.ambient_dimension() - self.inner.subspace_dimension())
    }

    /// Optimized projection using direct matrix operations.
    pub fn project<'py>(&self, py: Python<'py>, matrix: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = matrix.shape();
        if shape[0] != self.inner.ambient_dimension() || shape[1] != self.inner.subspace_dimension() {
            return Err(PyValueError::new_err(format!(
                "Expected {}x{} matrix, got {}x{}", 
                self.inner.ambient_dimension(), self.inner.subspace_dimension(), shape[0], shape[1]
            )));
        }
        
        // Direct conversion without intermediate allocations
        let mat = pyarray_to_dmatrix(&matrix)?;
        
        // Use QR decomposition to get canonical representation
        let qr = mat.qr();
        let mut q = qr.q().columns(0, self.inner.subspace_dimension()).into_owned();
        let r = qr.r();
        
        // Ensure positive diagonal elements in R for canonical form
        for i in 0..self.inner.subspace_dimension().min(r.nrows()) {
            if r[(i, i)] < 0.0 {
                // Flip sign of column if diagonal element is negative
                for j in 0..self.inner.ambient_dimension() {
                    q[(j, i)] = -q[(j, i)];
                }
            }
        }
        
        // Convert back to numpy
        dmatrix_to_pyarray(py, &q)
    }

    /// Optimized retraction on Grassmann manifold.
    pub fn retract<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        tangent: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Direct matrix operations
        let x = pyarray_to_dmatrix(&point)?;
        let v = pyarray_to_dmatrix(&tangent)?;
        
        // QR retraction on Grassmann: qf([X V])
        let mut combined = DMatrix::zeros(self.inner.ambient_dimension(), 2 * self.inner.subspace_dimension());
        combined.columns_mut(0, self.inner.subspace_dimension()).copy_from(&x);
        combined.columns_mut(self.inner.subspace_dimension(), self.inner.subspace_dimension()).copy_from(&v);
        
        let qr = combined.qr();
        let q = qr.q();
        
        // Extract first p columns and canonicalize
        let mut result = q.columns(0, self.inner.subspace_dimension()).into_owned();
        
        // Canonical representation
        let qr_result = result.clone().qr();
        let r = qr_result.r();
        for i in 0..self.inner.subspace_dimension().min(r.nrows()) {
            if r[(i, i)] < 0.0 {
                for j in 0..self.inner.ambient_dimension() {
                    result[(j, i)] = -result[(j, i)];
                }
            }
        }
        
        dmatrix_to_pyarray(py, &result)
    }

    /// Optimized tangent projection (horizontal space).
    pub fn tangent_projection<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        vector: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Direct matrix operations
        let x = pyarray_to_dmatrix(&point)?;
        let v = pyarray_to_dmatrix(&vector)?;
        
        // Project to horizontal space: V - X(X^T V)
        let xtv = x.transpose() * &v;
        let result = &v - &x * xtv;
        
        dmatrix_to_pyarray(py, &result)
    }

    /// Generate random point efficiently.
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        // Generate random Gaussian matrix
        let mut mat = DMatrix::zeros(self.inner.ambient_dimension(), self.inner.subspace_dimension());
        for i in 0..self.inner.ambient_dimension() {
            for j in 0..self.inner.subspace_dimension() {
                mat[(i, j)] = rng.sample::<f64, _>(rand_distr::StandardNormal);
            }
        }
        
        // QR decomposition for orthogonalization
        let qr = mat.qr();
        let mut q = qr.q().columns(0, self.inner.subspace_dimension()).into_owned();
        let r = qr.r();
        
        // Canonical representation
        for i in 0..self.inner.subspace_dimension().min(r.nrows()) {
            if r[(i, i)] < 0.0 {
                for j in 0..self.inner.ambient_dimension() {
                    q[(j, i)] = -q[(j, i)];
                }
            }
        }
        
        dmatrix_to_pyarray(py, &q)
    }

    pub fn __repr__(&self) -> String {
        format!("Grassmann(n={}, p={})", self.inner.ambient_dimension(), self.inner.subspace_dimension())
    }
}

// Non-Python methods for PyGrassmann
impl PyGrassmann {
    /// Get access to the inner Rust manifold (for compatibility with optimizer code).
    /// This method is not exposed to Python.
    pub fn get_inner(&self) -> &Grassmann {
        &self.inner
    }
}

/// Optimized SPD manifold implementation.
#[pyclass(name = "SPD")]
#[derive(Clone)]
pub struct PySPD {
    inner: SPD,
}

#[pymethods]
impl PySPD {
    #[new]
    pub fn new(n: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("SPD manifold requires n > 0"));
        }
        Ok(Self {
            inner: SPD::new(n).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    #[getter]
    pub fn n(&self) -> usize {
        self.inner.matrix_dimension()
    }

    #[getter]
    pub fn dim(&self) -> usize {
        let n = self.inner.matrix_dimension();
        n * (n + 1) / 2
    }

    /// Optimized projection using direct matrix operations.
    pub fn project<'py>(&self, py: Python<'py>, matrix: PyReadonlyArray2<'_, f64>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let shape = matrix.shape();
        if shape[0] != self.inner.matrix_dimension() || shape[1] != self.inner.matrix_dimension() {
            return Err(PyValueError::new_err(format!(
                "Expected {}x{} matrix, got {}x{}", 
                self.inner.matrix_dimension(), self.inner.matrix_dimension(), shape[0], shape[1]
            )));
        }
        
        // Direct conversion without intermediate allocations
        let mat = pyarray_to_dmatrix(&matrix)?;
        
        // Symmetrize the matrix
        let symmetric = (&mat + mat.transpose()) * 0.5;
        
        // Eigendecomposition
        let eigen = symmetric.symmetric_eigen();
        let mut eigenvalues = eigen.eigenvalues.clone();
        
        // Clamp eigenvalues to be positive
        let min_eig = self.inner.min_eigenvalue();
        for i in 0..eigenvalues.len() {
            if eigenvalues[i] < min_eig {
                eigenvalues[i] = min_eig;
            }
        }
        
        // Reconstruct matrix
        let diag = DMatrix::from_diagonal(&eigenvalues);
        let result = &eigen.eigenvectors * diag * eigen.eigenvectors.transpose();
        
        // Convert back to numpy
        dmatrix_to_pyarray(py, &result)
    }

    /// Optimized retraction on SPD manifold.
    pub fn retract<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray2<'_, f64>,
        tangent: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Direct matrix operations
        let p = pyarray_to_dmatrix(&point)?;
        let v = pyarray_to_dmatrix(&tangent)?;
        
        // First-order retraction: R_P(V) = P + V + 0.5 * V * P^{-1} * V
        let chol = p.clone().cholesky()
            .ok_or_else(|| PyValueError::new_err("Matrix is not positive definite"))?;
        let p_inv_v = chol.solve(&v);
        let result = &p + &v + (&v * p_inv_v) * 0.5;
        
        // Ensure positive definiteness by projection
        // Need to convert result to numpy array and then project
        let result_array = dmatrix_to_pyarray(py, &result)?;
        self.project(py, result_array.readonly())
    }

    /// Optimized tangent projection (symmetrization).
    pub fn tangent_projection<'py>(
        &self,
        py: Python<'py>,
        _point: PyReadonlyArray2<'_, f64>,
        vector: PyReadonlyArray2<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        // Direct matrix operations
        let v = pyarray_to_dmatrix(&vector)?;
        
        // Tangent space consists of symmetric matrices
        let result = (&v + v.transpose()) * 0.5;
        
        dmatrix_to_pyarray(py, &result)
    }

    /// Generate random SPD matrix efficiently.
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let n = self.inner.matrix_dimension();
        
        // Generate random symmetric matrix
        let mut a = DMatrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                let value = rng.sample::<f64, _>(rand_distr::StandardNormal);
                a[(i, j)] = value;
                if i != j {
                    a[(j, i)] = value;
                }
            }
        }
        
        // Make it positive definite: P = A^T A + εI
        let result = &a.transpose() * &a + DMatrix::<f64>::identity(n, n) * self.inner.min_eigenvalue();
        
        dmatrix_to_pyarray(py, &result)
    }

    pub fn __repr__(&self) -> String {
        format!("SPD(n={})", self.inner.matrix_dimension())
    }
}

// Non-Python methods for PySPD
impl PySPD {
    /// Get access to the inner Rust manifold (for compatibility with optimizer code).
    /// This method is not exposed to Python.
    pub fn get_inner(&self) -> &SPD {
        &self.inner
    }
}

