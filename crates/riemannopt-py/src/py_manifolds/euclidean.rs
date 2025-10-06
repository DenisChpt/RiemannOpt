use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray1, PyReadonlyArray1, IntoPyArray};
use nalgebra::DVector;

use riemannopt_manifolds::Euclidean;
use riemannopt_core::manifold::Manifold;

// Helper macro to check array dimensions
macro_rules! check_dim {
    ($arr:expr, $expected:expr, $name:expr) => {
        if $arr.len()? != $expected {
            return Err(PyValueError::new_err(format!(
                "Dimension mismatch for {}: expected {}, got {}",
                $name,
                $expected,
                $arr.len()?
            )));
        }
    };
}

/// Python wrapper for the Euclidean manifold ℝ^n.
///
/// The Euclidean manifold is the standard n-dimensional vector space equipped
/// with the usual Euclidean metric. All Riemannian operations are trivial:
/// - Projection is identity
/// - Retraction is addition  
/// - Inner product is dot product
/// - Parallel transport is identity
///
/// This manifold provides a baseline for optimization algorithms and enables
/// unconstrained optimization within the Riemannian framework.
///
/// Parameters
/// ----------
/// n : int
///     Dimension of the Euclidean space
///
/// Examples
/// --------
/// >>> import numpy as np
/// >>> from riemannopt import Euclidean
/// >>> 
/// >>> # Create 10-dimensional Euclidean space
/// >>> manifold = Euclidean(10)
/// >>> 
/// >>> # Random point
/// >>> x = manifold.random_point()
/// >>> 
/// >>> # Random tangent vector
/// >>> v = manifold.random_tangent(x)
/// >>> 
/// >>> # Retraction (addition)
/// >>> y = manifold.retract(x, v)
/// >>> assert np.allclose(y, x + v)
/// >>> 
/// >>> # Inner product (dot product)
/// >>> inner = manifold.inner(x, v, v)
/// >>> assert np.isclose(inner, np.dot(v, v))
#[pyclass(name = "Euclidean", module = "riemannopt.manifolds")]
pub struct PyEuclidean {
    pub(crate) inner: Euclidean<f64>,
}

#[pymethods]
impl PyEuclidean {
    /// Create a new Euclidean manifold of dimension n.
    ///
    /// Parameters
    /// ----------
    /// n : int
    ///     Dimension of the space (must be positive)
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If n is 0 or negative
    #[new]
    pub fn new(n: usize) -> PyResult<Self> {
        if n == 0 {
            return Err(PyValueError::new_err("Dimension must be positive"));
        }
        
        match Euclidean::new(n) {
            Ok(inner) => Ok(Self { inner }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create Euclidean manifold: {}", e)))
        }
    }

    /// Get the intrinsic dimension of the manifold.
    ///
    /// Returns
    /// -------
    /// int
    ///     The dimension n
    #[getter]
    pub fn dim(&self) -> usize {
        self.inner.dimension()
    }

    /// Get the dimension of the ambient space.
    ///
    /// For Euclidean space, this is the same as dim.
    ///
    /// Returns
    /// -------
    /// int
    ///     The dimension n
    #[getter]  
    pub fn ambient_dim(&self) -> usize {
        self.inner.dimension()
    }

    /// Get the name of the manifold.
    ///
    /// Returns
    /// -------
    /// str
    ///     "Euclidean"
    #[getter]
    pub fn name(&self) -> &str {
        self.inner.name()
    }

    /// Check if a point lies on the manifold.
    ///
    /// For Euclidean space, this only checks dimension.
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Point to check
    /// tol : float, optional
    ///     Tolerance for the check (default: 1e-10)
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if point has correct dimension
    #[pyo3(signature = (point, tol=None))]
    pub fn is_point_on_manifold(&self, point: PyReadonlyArray1<f64>, tol: Option<f64>) -> PyResult<bool> {
        if point.len()? != self.inner.dimension() {
            return Ok(false);
        }
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        Ok(self.inner.is_point_on_manifold(&point, tol.unwrap_or(1e-10)))
    }

    /// Project a point onto the manifold.
    ///
    /// For Euclidean space, this is the identity operation.
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Point to project
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     The same point (identity projection)
    pub fn project_point<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        check_dim!(point, self.inner.dimension(), "point");
        
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        let mut result = DVector::zeros(self.inner.dimension());
        
        self.inner.project_point(&point, &mut result);
        
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
    }

    /// Project a vector onto the tangent space at a point.
    ///
    /// For Euclidean space, this is the identity operation.
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Base point (not used for Euclidean)
    /// vector : array_like
    ///     Vector to project
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     The same vector (identity projection)
    pub fn project_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
        vector: PyReadonlyArray1<f64>
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "point", self.inner.dimension(), point.len()?))); }
        if vector.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "vector", self.inner.dimension(), vector.len()?))); }
        
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        let vector = DVector::from_column_slice(vector.as_slice().unwrap());
        let mut result = DVector::zeros(self.inner.dimension());
        
        self.inner.project_tangent(&point, &vector, &mut result)
            .map_err(|e| PyValueError::new_err(format!("Projection failed: {}", e)))?;
        
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
    }

    /// Compute the Riemannian inner product between two tangent vectors.
    ///
    /// For Euclidean space, this is the standard dot product.
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Base point (not used for Euclidean)
    /// u : array_like
    ///     First tangent vector
    /// v : array_like
    ///     Second tangent vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The dot product u·v
    pub fn inner(
        &self,
        point: PyReadonlyArray1<f64>,
        u: PyReadonlyArray1<f64>,
        v: PyReadonlyArray1<f64>
    ) -> PyResult<f64> {
        if point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "point", self.inner.dimension(), point.len()?))); }
        if u.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "u", self.inner.dimension(), u.len()?))); }
        if v.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "v", self.inner.dimension(), v.len()?))); }
        
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        let u = DVector::from_column_slice(u.as_slice().unwrap());
        let v = DVector::from_column_slice(v.as_slice().unwrap());
        
        self.inner.inner_product(&point, &u, &v)
            .map_err(|e| PyValueError::new_err(format!("Inner product failed: {}", e)))
    }

    /// Compute the norm of a tangent vector.
    ///
    /// For Euclidean space, this is the standard Euclidean norm.
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Base point (not used for Euclidean)
    /// vector : array_like
    ///     Tangent vector
    ///
    /// Returns
    /// -------
    /// float
    ///     The Euclidean norm ||v||
    pub fn norm(
        &self,
        point: PyReadonlyArray1<f64>,
        vector: PyReadonlyArray1<f64>
    ) -> PyResult<f64> {
        if point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "point", self.inner.dimension(), point.len()?))); }
        if vector.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "vector", self.inner.dimension(), vector.len()?))); }
        
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        let vector = DVector::from_column_slice(vector.as_slice().unwrap());
        
        self.inner.norm(&point, &vector)
            .map_err(|e| PyValueError::new_err(format!("Norm computation failed: {}", e)))
    }

    /// Perform a retraction from the tangent space to the manifold.
    ///
    /// For Euclidean space, retraction is simple addition: R_x(v) = x + v
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Base point
    /// tangent : array_like
    ///     Tangent vector
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     The retracted point x + v
    pub fn retract<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
        tangent: PyReadonlyArray1<f64>
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "point", self.inner.dimension(), point.len()?))); }
        if tangent.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "tangent", self.inner.dimension(), tangent.len()?))); }
        
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        let tangent = DVector::from_column_slice(tangent.as_slice().unwrap());
        let mut result = DVector::zeros(self.inner.dimension());
        
        self.inner.retract(&point, &tangent, &mut result)
            .map_err(|e| PyValueError::new_err(format!("Retraction failed: {}", e)))?;
        
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
    }

    /// Compute the inverse retraction (logarithmic map).
    ///
    /// For Euclidean space, this is subtraction: log_x(y) = y - x
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Base point
    /// other : array_like
    ///     Target point
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     The tangent vector y - x
    pub fn inverse_retract<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
        other: PyReadonlyArray1<f64>
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "point", self.inner.dimension(), point.len()?))); }
        if other.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "other", self.inner.dimension(), other.len()?))); }
        
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        let other = DVector::from_column_slice(other.as_slice().unwrap());
        let mut result = DVector::zeros(self.inner.dimension());
        
        self.inner.inverse_retract(&point, &other, &mut result)
            .map_err(|e| PyValueError::new_err(format!("Inverse retraction failed: {}", e)))?;
        
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
    }

    /// Convert Euclidean gradient to Riemannian gradient.
    ///
    /// For Euclidean space, these are identical.
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Base point (not used for Euclidean)
    /// euclidean_grad : array_like
    ///     Euclidean gradient
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     The same gradient (identity)
    pub fn euclidean_to_riemannian_gradient<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>,
        euclidean_grad: PyReadonlyArray1<f64>
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "point", self.inner.dimension(), point.len()?))); }
        if euclidean_grad.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "euclidean_grad", self.inner.dimension(), euclidean_grad.len()?))); }
        
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        let euclidean_grad = DVector::from_column_slice(euclidean_grad.as_slice().unwrap());
        let mut result = DVector::zeros(self.inner.dimension());
        
        self.inner.euclidean_to_riemannian_gradient(&point, &euclidean_grad, &mut result)
            .map_err(|e| PyValueError::new_err(format!("Gradient conversion failed: {}", e)))?;
        
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
    }

    /// Perform parallel transport of a vector.
    ///
    /// For Euclidean space (flat), parallel transport is identity.
    ///
    /// Parameters
    /// ----------
    /// from_point : array_like
    ///     Starting point
    /// to_point : array_like
    ///     Ending point
    /// vector : array_like
    ///     Vector to transport
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     The same vector (identity transport)
    pub fn parallel_transport<'py>(
        &self,
        py: Python<'py>,
        from_point: PyReadonlyArray1<f64>,
        to_point: PyReadonlyArray1<f64>,
        vector: PyReadonlyArray1<f64>
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if from_point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "from_point", self.inner.dimension(), from_point.len()?))); }
        if to_point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "to_point", self.inner.dimension(), to_point.len()?))); }
        if vector.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "vector", self.inner.dimension(), vector.len()?))); }
        
        let from_point = DVector::from_column_slice(from_point.as_slice().unwrap());
        let to_point = DVector::from_column_slice(to_point.as_slice().unwrap());
        let vector = DVector::from_column_slice(vector.as_slice().unwrap());
        let mut result = DVector::zeros(self.inner.dimension());
        
        self.inner.parallel_transport(&from_point, &to_point, &vector, &mut result)
            .map_err(|e| PyValueError::new_err(format!("Parallel transport failed: {}", e)))?;
        
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
    }

    /// Generate a random point on the manifold.
    ///
    /// Points are sampled from a standard normal distribution.
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Random point in ℝ^n
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let mut result = DVector::zeros(self.inner.dimension());
        
        self.inner.random_point(&mut result)
            .map_err(|e| PyValueError::new_err(format!("Random point generation failed: {}", e)))?;
        
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
    }

    /// Generate a random tangent vector at a point.
    ///
    /// Vectors are sampled from a standard normal distribution.
    ///
    /// Parameters
    /// ----------
    /// point : array_like
    ///     Base point (not used for Euclidean)
    ///
    /// Returns
    /// -------
    /// ndarray
    ///     Random tangent vector
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<f64>
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        if point.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "point", self.inner.dimension(), point.len()?))); }
        
        let point = DVector::from_column_slice(point.as_slice().unwrap());
        let mut result = DVector::zeros(self.inner.dimension());
        
        self.inner.random_tangent(&point, &mut result)
            .map_err(|e| PyValueError::new_err(format!("Random tangent generation failed: {}", e)))?;
        
        Ok(result.as_slice().to_vec().into_pyarray_bound(py))
    }

    /// Compute the geodesic distance between two points.
    ///
    /// For Euclidean space, this is the standard Euclidean distance.
    ///
    /// Parameters
    /// ----------
    /// x : array_like
    ///     First point
    /// y : array_like
    ///     Second point
    ///
    /// Returns
    /// -------
    /// float
    ///     The Euclidean distance ||y - x||
    pub fn distance(&self, x: PyReadonlyArray1<f64>, y: PyReadonlyArray1<f64>) -> PyResult<f64> {
        if x.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "x", self.inner.dimension(), x.len()?))); }
        if y.len()? != self.inner.dimension() { return Err(PyValueError::new_err(format!("Dimension mismatch for {}: expected {}, got {}", "y", self.inner.dimension(), y.len()?))); }
        
        let x = DVector::from_column_slice(x.as_slice().unwrap());
        let y = DVector::from_column_slice(y.as_slice().unwrap());
        
        self.inner.distance(&x, &y)
            .map_err(|e| PyValueError::new_err(format!("Distance computation failed: {}", e)))
    }

    /// String representation of the manifold.
    ///
    /// Returns
    /// -------
    /// str
    ///     "Euclidean(n=<dimension>)"
    pub fn __repr__(&self) -> String {
        format!("Euclidean(n={})", self.inner.dimension())
    }

    /// Check if the manifold is flat (zero curvature).
    ///
    /// Returns
    /// -------
    /// bool
    ///     True (Euclidean space is flat)
    pub fn is_flat(&self) -> bool {
        true
    }
}