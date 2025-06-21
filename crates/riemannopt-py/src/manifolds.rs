//! Python bindings for Riemannian manifolds.
//!
//! This module provides Python-friendly wrappers around the Rust manifold implementations,
//! with NumPy array integration for seamless interoperability.

use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix, Dyn};

use riemannopt_manifolds::{
    Sphere, Hyperbolic,
    ProductManifold, ProductManifoldStatic
};
use riemannopt_core::manifold::Manifold;
use crate::array_utils::dvector_to_pyarray;

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

/// Enum to represent different types of product manifolds statically
enum ProductManifoldInner {
    SphereSphere(riemannopt_manifolds::ProductManifoldStatic<f64, riemannopt_manifolds::Sphere, riemannopt_manifolds::Sphere>),
    SphereStiefel(riemannopt_manifolds::ProductManifoldStatic<f64, riemannopt_manifolds::Sphere, riemannopt_manifolds::Stiefel>),
    StiefelSphere(riemannopt_manifolds::ProductManifoldStatic<f64, riemannopt_manifolds::Stiefel, riemannopt_manifolds::Sphere>),
    StiefelStiefel(riemannopt_manifolds::ProductManifoldStatic<f64, riemannopt_manifolds::Stiefel, riemannopt_manifolds::Stiefel>),
    SphereGrassmann(riemannopt_manifolds::ProductManifoldStatic<f64, riemannopt_manifolds::Sphere, riemannopt_manifolds::Grassmann>),
    GrassmannSphere(riemannopt_manifolds::ProductManifoldStatic<f64, riemannopt_manifolds::Grassmann, riemannopt_manifolds::Sphere>),
    // Fallback to old implementation for unsupported combinations
    Dynamic(PyObject, PyObject, usize, usize, usize, usize),
}

/// Product manifold M1 × M2 in Python (new static implementation).
///
/// The product manifold allows combining two manifolds with better performance.
#[pyclass(name = "ProductManifoldStatic")]
pub struct PyProductManifoldStatic {
    /// Internal implementation using static dispatch when possible
    inner: ProductManifoldInner,
}

#[pymethods]
impl PyProductManifoldStatic {
    /// Create a new product manifold.
    ///
    /// Args:
    ///     manifold1: First component manifold
    ///     manifold2: Second component manifold
    #[new]
    pub fn new(manifold1: &Bound<'_, PyAny>, manifold2: &Bound<'_, PyAny>) -> PyResult<Self> {
        use riemannopt_manifolds::ProductManifoldStatic;
        
        // Try to create static implementations based on manifold types
        let inner = if let (Ok(sphere1), Ok(sphere2)) = (manifold1.extract::<PyRef<PySphere>>(), manifold2.extract::<PyRef<PySphere>>()) {
            ProductManifoldInner::SphereSphere(
                ProductManifoldStatic::new(sphere1.get_inner().clone(), sphere2.get_inner().clone())
            )
        } else if let (Ok(sphere), Ok(stiefel)) = (manifold1.extract::<PyRef<PySphere>>(), manifold2.extract::<PyRef<crate::manifolds_optimized::PyStiefel>>()) {
            ProductManifoldInner::SphereStiefel(
                ProductManifoldStatic::new(sphere.get_inner().clone(), stiefel.get_inner().clone())
            )
        } else if let (Ok(stiefel), Ok(sphere)) = (manifold1.extract::<PyRef<crate::manifolds_optimized::PyStiefel>>(), manifold2.extract::<PyRef<PySphere>>()) {
            ProductManifoldInner::StiefelSphere(
                ProductManifoldStatic::new(stiefel.get_inner().clone(), sphere.get_inner().clone())
            )
        } else if let (Ok(stiefel1), Ok(stiefel2)) = (manifold1.extract::<PyRef<crate::manifolds_optimized::PyStiefel>>(), manifold2.extract::<PyRef<crate::manifolds_optimized::PyStiefel>>()) {
            ProductManifoldInner::StiefelStiefel(
                ProductManifoldStatic::new(stiefel1.get_inner().clone(), stiefel2.get_inner().clone())
            )
        } else if let (Ok(sphere), Ok(grassmann)) = (manifold1.extract::<PyRef<PySphere>>(), manifold2.extract::<PyRef<crate::manifolds_optimized::PyGrassmann>>()) {
            ProductManifoldInner::SphereGrassmann(
                ProductManifoldStatic::new(sphere.get_inner().clone(), grassmann.get_inner().clone())
            )
        } else if let (Ok(grassmann), Ok(sphere)) = (manifold1.extract::<PyRef<crate::manifolds_optimized::PyGrassmann>>(), manifold2.extract::<PyRef<PySphere>>()) {
            ProductManifoldInner::GrassmannSphere(
                ProductManifoldStatic::new(grassmann.get_inner().clone(), sphere.get_inner().clone())
            )
        } else {
            // Fallback to dynamic implementation
            let py = manifold1.py();
            let manifold1_py = manifold1.to_object(py);
            let manifold2_py = manifold2.to_object(py);
            
            // Get dimensions from manifolds
            let manifold_dim1 = manifold1.getattr("manifold_dim")?.extract::<usize>()?;
            let manifold_dim2 = manifold2.getattr("manifold_dim")?.extract::<usize>()?;
            
            // Get ambient dimensions (for actual array sizes)
            let ambient_dim1 = if let Ok(ambient) = manifold1.getattr("ambient_dim") {
                ambient.extract::<usize>()?
            } else {
                manifold_dim1
            };
            
            let ambient_dim2 = if let Ok(ambient) = manifold2.getattr("ambient_dim") {
                ambient.extract::<usize>()?
            } else {
                manifold_dim2
            };
            
            ProductManifoldInner::Dynamic(manifold1_py, manifold2_py, manifold_dim1, manifold_dim2, ambient_dim1, ambient_dim2)
        };
        
        Ok(Self { inner })
    }
    
    /// Get the manifold dimension.
    #[getter]
    pub fn dim(&self) -> usize {
        use riemannopt_core::manifold::Manifold;
        use nalgebra::Dyn;
        
        match &self.inner {
            ProductManifoldInner::SphereSphere(p) => p.dimension(),
            ProductManifoldInner::SphereStiefel(p) => p.dimension(),
            ProductManifoldInner::StiefelSphere(p) => p.dimension(),
            ProductManifoldInner::StiefelStiefel(p) => p.dimension(),
            ProductManifoldInner::SphereGrassmann(p) => p.dimension(),
            ProductManifoldInner::GrassmannSphere(p) => p.dimension(),
            ProductManifoldInner::Dynamic(_, _, _, _, ambient_dim1, ambient_dim2) => ambient_dim1 + ambient_dim2,
        }
    }
    
    /// Get the manifold dimension (alias).
    #[getter]
    pub fn manifold_dim(&self) -> usize {
        self.dim()
    }
    
    /// Project a point onto the manifold.
    ///
    /// Args:
    ///     point: Point in ambient space (numpy array)
    ///
    /// Returns:
    ///     Projected point on the product manifold
    pub fn project<'py>(&self, py: Python<'py>, point: PyReadonlyArray1<'_, f64>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_vec(point.as_slice()?.to_vec());
        
        let result = match &self.inner {
            ProductManifoldInner::SphereSphere(p) => p.project_point(&point_vec),
            ProductManifoldInner::SphereStiefel(p) => p.project_point(&point_vec),
            ProductManifoldInner::StiefelSphere(p) => p.project_point(&point_vec),
            ProductManifoldInner::StiefelStiefel(p) => p.project_point(&point_vec),
            ProductManifoldInner::SphereGrassmann(p) => p.project_point(&point_vec),
            ProductManifoldInner::GrassmannSphere(p) => p.project_point(&point_vec),
            ProductManifoldInner::Dynamic(manifold1, manifold2, _, _, ambient_dim1, ambient_dim2) => {
                // Fallback to Python calls
                let point_slice = point.as_slice()?;
                if point_slice.len() != ambient_dim1 + ambient_dim2 {
                    return Err(PyValueError::new_err(format!(
                        "Expected point of dimension {}, got {}", 
                        ambient_dim1 + ambient_dim2, point_slice.len()
                    )));
                }
                
                // Split the point
                let point1 = &point_slice[..*ambient_dim1];
                let point2 = &point_slice[*ambient_dim1..];
                
                // Project each component
                let arr1 = numpy::PyArray1::from_slice_bound(py, point1);
                let arr2 = numpy::PyArray1::from_slice_bound(py, point2);
                
                let proj1 = manifold1.call_method1(py, "project", (arr1,))?;
                let proj2 = manifold2.call_method1(py, "project", (arr2,))?;
                
                // Concatenate results
                let mut result = Vec::with_capacity(ambient_dim1 + ambient_dim2);
                let proj1_arr: Bound<PyArray1<f64>> = proj1.extract(py)?;
                let proj2_arr: Bound<PyArray1<f64>> = proj2.extract(py)?;
                
                result.extend_from_slice(proj1_arr.readonly().as_slice()?);
                result.extend_from_slice(proj2_arr.readonly().as_slice()?);
                
                DVector::from_vec(result)
            }
        };
        
        Ok(numpy::PyArray1::from_vec_bound(py, result.as_slice().to_vec()))
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
        let point_vec = DVector::from_vec(point.as_slice()?.to_vec());
        let tangent_vec = DVector::from_vec(tangent.as_slice()?.to_vec());
        
        let result = match &self.inner {
            ProductManifoldInner::SphereSphere(p) => p.retract(&point_vec, &tangent_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::SphereStiefel(p) => p.retract(&point_vec, &tangent_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::StiefelSphere(p) => p.retract(&point_vec, &tangent_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::StiefelStiefel(p) => p.retract(&point_vec, &tangent_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::SphereGrassmann(p) => p.retract(&point_vec, &tangent_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::GrassmannSphere(p) => p.retract(&point_vec, &tangent_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::Dynamic(manifold1, manifold2, _, _, ambient_dim1, ambient_dim2) => {
                // Fallback to Python calls
                let point_slice = point.as_slice()?;
                let tangent_slice = tangent.as_slice()?;
                
                // Split points and tangents
                let point1 = &point_slice[..*ambient_dim1];
                let point2 = &point_slice[*ambient_dim1..];
                let tangent1 = &tangent_slice[..*ambient_dim1];
                let tangent2 = &tangent_slice[*ambient_dim1..];
                
                // Create numpy arrays
                let arr_p1 = numpy::PyArray1::from_slice_bound(py, point1);
                let arr_p2 = numpy::PyArray1::from_slice_bound(py, point2);
                let arr_t1 = numpy::PyArray1::from_slice_bound(py, tangent1);
                let arr_t2 = numpy::PyArray1::from_slice_bound(py, tangent2);
                
                // Retract each component
                let ret1 = manifold1.call_method1(py, "retract", (arr_p1, arr_t1))?;
                let ret2 = manifold2.call_method1(py, "retract", (arr_p2, arr_t2))?;
                
                // Concatenate results
                let mut result = Vec::with_capacity(ambient_dim1 + ambient_dim2);
                let ret1_arr: Bound<PyArray1<f64>> = ret1.extract(py)?;
                let ret2_arr: Bound<PyArray1<f64>> = ret2.extract(py)?;
                
                result.extend_from_slice(ret1_arr.readonly().as_slice()?);
                result.extend_from_slice(ret2_arr.readonly().as_slice()?);
                
                DVector::from_vec(result)
            }
        };
        
        Ok(numpy::PyArray1::from_vec_bound(py, result.as_slice().to_vec()))
    }
    
    /// Random point on the product manifold.
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let result = match &self.inner {
            ProductManifoldInner::SphereSphere(p) => p.random_point(),
            ProductManifoldInner::SphereStiefel(p) => p.random_point(),
            ProductManifoldInner::StiefelSphere(p) => p.random_point(),
            ProductManifoldInner::StiefelStiefel(p) => p.random_point(),
            ProductManifoldInner::SphereGrassmann(p) => p.random_point(),
            ProductManifoldInner::GrassmannSphere(p) => p.random_point(),
            ProductManifoldInner::Dynamic(manifold1, manifold2, _, _, ambient_dim1, ambient_dim2) => {
                // Fallback to Python calls
                let rand1 = manifold1.call_method0(py, "random_point")?;
                let rand2 = manifold2.call_method0(py, "random_point")?;
                
                // Concatenate results
                let mut result = Vec::with_capacity(ambient_dim1 + ambient_dim2);
                let rand1_arr: Bound<PyArray1<f64>> = rand1.extract(py)?;
                let rand2_arr: Bound<PyArray1<f64>> = rand2.extract(py)?;
                
                result.extend_from_slice(rand1_arr.readonly().as_slice()?);
                result.extend_from_slice(rand2_arr.readonly().as_slice()?);
                
                DVector::from_vec(result)
            }
        };
        
        Ok(numpy::PyArray1::from_vec_bound(py, result.as_slice().to_vec()))
    }
    
    /// Compute the inner product between two tangent vectors.
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
        py: Python<'_>,
        point: PyReadonlyArray1<'_, f64>,
        u: PyReadonlyArray1<'_, f64>,
        v: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<f64> {
        let point_vec = DVector::from_vec(point.as_slice()?.to_vec());
        let u_vec = DVector::from_vec(u.as_slice()?.to_vec());
        let v_vec = DVector::from_vec(v.as_slice()?.to_vec());
        
        match &self.inner {
            ProductManifoldInner::SphereSphere(p) => p.inner_product(&point_vec, &u_vec, &v_vec).map_err(|e| PyValueError::new_err(e.to_string())),
            ProductManifoldInner::SphereStiefel(p) => p.inner_product(&point_vec, &u_vec, &v_vec).map_err(|e| PyValueError::new_err(e.to_string())),
            ProductManifoldInner::StiefelSphere(p) => p.inner_product(&point_vec, &u_vec, &v_vec).map_err(|e| PyValueError::new_err(e.to_string())),
            ProductManifoldInner::StiefelStiefel(p) => p.inner_product(&point_vec, &u_vec, &v_vec).map_err(|e| PyValueError::new_err(e.to_string())),
            ProductManifoldInner::SphereGrassmann(p) => p.inner_product(&point_vec, &u_vec, &v_vec).map_err(|e| PyValueError::new_err(e.to_string())),
            ProductManifoldInner::GrassmannSphere(p) => p.inner_product(&point_vec, &u_vec, &v_vec).map_err(|e| PyValueError::new_err(e.to_string())),
            ProductManifoldInner::Dynamic(manifold1, manifold2, _, _, ambient_dim1, ambient_dim2) => {
                // Fallback to Python calls
                let point_slice = point.as_slice()?;
                let u_slice = u.as_slice()?;
                let v_slice = v.as_slice()?;
                
                // Split
                let point1 = &point_slice[..*ambient_dim1];
                let point2 = &point_slice[*ambient_dim1..];
                let u1 = &u_slice[..*ambient_dim1];
                let u2 = &u_slice[*ambient_dim1..];
                let v1 = &v_slice[..*ambient_dim1];
                let v2 = &v_slice[*ambient_dim1..];
                
                // Create numpy arrays
                let arr_p1 = numpy::PyArray1::from_slice_bound(py, point1);
                let arr_p2 = numpy::PyArray1::from_slice_bound(py, point2);
                let arr_u1 = numpy::PyArray1::from_slice_bound(py, u1);
                let arr_u2 = numpy::PyArray1::from_slice_bound(py, u2);
                let arr_v1 = numpy::PyArray1::from_slice_bound(py, v1);
                let arr_v2 = numpy::PyArray1::from_slice_bound(py, v2);
                
                // Compute inner products
                let inner1 = manifold1.call_method1(py, "inner_product", (arr_p1, arr_u1, arr_v1))?;
                let inner2 = manifold2.call_method1(py, "inner_product", (arr_p2, arr_u2, arr_v2))?;
                
                let inner1_val: f64 = inner1.extract(py)?;
                let inner2_val: f64 = inner2.extract(py)?;
                
                Ok(inner1_val + inner2_val)
            }
        }
    }
    
    
    /// Generate a random tangent vector at a point.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///
    /// Returns:
    ///     Random tangent vector
    pub fn random_tangent<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_vec(point.as_slice()?.to_vec());
        
        let result = match &self.inner {
            ProductManifoldInner::SphereSphere(p) => p.random_tangent(&point_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::SphereStiefel(p) => p.random_tangent(&point_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::StiefelSphere(p) => p.random_tangent(&point_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::StiefelStiefel(p) => p.random_tangent(&point_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::SphereGrassmann(p) => p.random_tangent(&point_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::GrassmannSphere(p) => p.random_tangent(&point_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::Dynamic(manifold1, manifold2, _, _, ambient_dim1, ambient_dim2) => {
                // Fallback to Python calls
                let point_slice = point.as_slice()?;
                
                // Split the point
                let point1 = &point_slice[..*ambient_dim1];
                let point2 = &point_slice[*ambient_dim1..];
                
                // Create numpy arrays
                let arr_p1 = numpy::PyArray1::from_slice_bound(py, point1);
                let arr_p2 = numpy::PyArray1::from_slice_bound(py, point2);
                
                // Generate tangent vectors
                let tan1 = manifold1.call_method1(py, "random_tangent", (arr_p1,))?;
                let tan2 = manifold2.call_method1(py, "random_tangent", (arr_p2,))?;
                
                // Concatenate results
                let mut result = Vec::with_capacity(ambient_dim1 + ambient_dim2);
                let tan1_arr: Bound<PyArray1<f64>> = tan1.extract(py)?;
                let tan2_arr: Bound<PyArray1<f64>> = tan2.extract(py)?;
                
                result.extend_from_slice(tan1_arr.readonly().as_slice()?);
                result.extend_from_slice(tan2_arr.readonly().as_slice()?);
                
                DVector::from_vec(result)
            }
        };
        
        Ok(numpy::PyArray1::from_vec_bound(py, result.as_slice().to_vec()))
    }
    
    /// Convert Euclidean gradient to Riemannian gradient.
    ///
    /// Args:
    ///     point: Point on the manifold
    ///     euclidean_grad: Euclidean gradient
    ///
    /// Returns:
    ///     Riemannian gradient
    pub fn euclidean_to_riemannian_gradient<'py>(
        &self,
        py: Python<'py>,
        point: PyReadonlyArray1<'_, f64>,
        euclidean_grad: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let point_vec = DVector::from_vec(point.as_slice()?.to_vec());
        let grad_vec = DVector::from_vec(euclidean_grad.as_slice()?.to_vec());
        
        let result = match &self.inner {
            ProductManifoldInner::SphereSphere(p) => p.euclidean_to_riemannian_gradient(&point_vec, &grad_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::SphereStiefel(p) => p.euclidean_to_riemannian_gradient(&point_vec, &grad_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::StiefelSphere(p) => p.euclidean_to_riemannian_gradient(&point_vec, &grad_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::StiefelStiefel(p) => p.euclidean_to_riemannian_gradient(&point_vec, &grad_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::SphereGrassmann(p) => p.euclidean_to_riemannian_gradient(&point_vec, &grad_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::GrassmannSphere(p) => p.euclidean_to_riemannian_gradient(&point_vec, &grad_vec).map_err(|e| PyValueError::new_err(e.to_string()))?,
            ProductManifoldInner::Dynamic(manifold1, manifold2, _, _, ambient_dim1, ambient_dim2) => {
                // Fallback to Python calls
                let point_slice = point.as_slice()?;
                let grad_slice = euclidean_grad.as_slice()?;
                
                // Split
                let point1 = &point_slice[..*ambient_dim1];
                let point2 = &point_slice[*ambient_dim1..];
                let grad1 = &grad_slice[..*ambient_dim1];
                let grad2 = &grad_slice[*ambient_dim1..];
                
                // Create numpy arrays
                let arr_p1 = numpy::PyArray1::from_slice_bound(py, point1);
                let arr_p2 = numpy::PyArray1::from_slice_bound(py, point2);
                let arr_g1 = numpy::PyArray1::from_slice_bound(py, grad1);
                let arr_g2 = numpy::PyArray1::from_slice_bound(py, grad2);
                
                // Convert each component
                let riem1 = manifold1.call_method1(py, "euclidean_to_riemannian_gradient", (arr_p1, arr_g1))?;
                let riem2 = manifold2.call_method1(py, "euclidean_to_riemannian_gradient", (arr_p2, arr_g2))?;
                
                // Concatenate results
                let mut result = Vec::with_capacity(ambient_dim1 + ambient_dim2);
                let riem1_arr: Bound<PyArray1<f64>> = riem1.extract(py)?;
                let riem2_arr: Bound<PyArray1<f64>> = riem2.extract(py)?;
                
                result.extend_from_slice(riem1_arr.readonly().as_slice()?);
                result.extend_from_slice(riem2_arr.readonly().as_slice()?);
                
                DVector::from_vec(result)
            }
        };
        
        Ok(numpy::PyArray1::from_vec_bound(py, result.as_slice().to_vec()))
    }
}

/// Legacy Product manifold M1 × M2 in Python.
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