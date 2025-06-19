//! Optimized manifold implementations with better performance.

use numpy::{PyArray2, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::DMatrix;

use riemannopt_manifolds::Stiefel;
use crate::array_utils::{pyarray_to_dmatrix, dmatrix_to_pyarray};

/// Optimized Stiefel manifold implementation.
#[pyclass(name = "StiefelOpt")]
#[derive(Clone)]
pub struct PyStiefelOpt {
    inner: Stiefel,
}

#[pymethods]
impl PyStiefelOpt {
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
        format!("StiefelOpt(n={}, p={})", self.inner.n(), self.inner.p())
    }
}