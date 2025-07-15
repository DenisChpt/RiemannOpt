//! Python wrapper for the Product manifold.
//!
//! The product manifold is the Cartesian product of multiple manifolds,
//! allowing optimization over multiple coupled variables with different geometries.

use pyo3::prelude::*;
use pyo3::types::{PyList, PyTuple};
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use nalgebra::{DVector, DMatrix};

use crate::{
    array_utils::{numpy_to_dvector, dvector_to_numpy, numpy_to_dmatrix, dmatrix_to_numpy},
    error::dimension_mismatch,
    types::PyPoint,
};
use super::base::{PyManifoldBase, PointType};

/// The product of multiple manifolds.
///
/// A product manifold M = M_1 × M_2 × ... × M_k is the Cartesian product
/// of k manifolds. Points on the product manifold are tuples (x_1, ..., x_k)
/// where x_i is a point on M_i.
///
/// The Riemannian metric on the product is the sum of the metrics on each
/// component, making all operations decomposable.
///
/// Parameters
/// ----------
/// manifolds : list
///     List of component manifolds
///
/// Attributes
/// ----------
/// manifolds : list
///     The component manifolds
/// n_manifolds : int
///     Number of component manifolds
/// dim : int
///     Total intrinsic dimension (sum of component dimensions)
/// ambient_dim : int
///     Total ambient dimension (sum of component ambient dimensions)
///
/// Examples
/// --------
/// >>> import riemannopt as ro
/// >>> import numpy as np
/// >>>
/// >>> # Optimize over a sphere and a Stiefel manifold jointly
/// >>> sphere = ro.manifolds.Sphere(3)
/// >>> stiefel = ro.manifolds.Stiefel(5, 2)
/// >>> product = ro.manifolds.ProductManifold([sphere, stiefel])
/// >>>
/// >>> print(f"Total dimension: {product.dim}")
/// Total dimension: 8  # 2 (sphere) + 6 (stiefel)
/// >>>
/// >>> # Points are represented as tuples
/// >>> point = product.random_point()
/// >>> x_sphere, X_stiefel = point
/// >>> print(x_sphere.shape, X_stiefel.shape)
/// (3,) (5, 2)
///
/// Notes
/// -----
/// Currently, this implementation is a placeholder that demonstrates the API.
/// Full implementation would require dynamic type handling for heterogeneous
/// manifold collections.
#[pyclass(name = "ProductManifold", module = "riemannopt.manifolds")]
#[derive()]
pub struct PyProductManifold {
    /// Component manifolds (stored as Python objects)
    manifolds: Vec<PyObject>,
    /// Total dimension
    total_dim: usize,
    /// Total ambient dimension
    total_ambient_dim: usize,
}

// Manual Clone implementation for PyObject handling
impl Clone for PyProductManifold {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            PyProductManifold {
                manifolds: self.manifolds.iter().map(|obj| obj.clone_ref(py)).collect(),
                total_dim: self.total_dim,
                total_ambient_dim: self.total_ambient_dim,
            }
        })
    }
}

impl PyManifoldBase for PyProductManifold {
    fn manifold_name(&self) -> &'static str {
        "ProductManifold"
    }
    
    fn ambient_dim(&self) -> usize {
        self.total_ambient_dim
    }
    
    fn intrinsic_dim(&self) -> usize {
        self.total_dim
    }
    
    fn point_type(&self) -> PointType {
        // Product manifolds use composite points
        PointType::Vector(self.total_ambient_dim)
    }
}

#[pymethods]
impl PyProductManifold {
    /// Create a new product manifold.
    ///
    /// Parameters
    /// ----------
    /// manifolds : list
    ///     List of component manifolds
    ///
    /// Raises
    /// ------
    /// ValueError
    ///     If manifolds list is empty
    #[new]
    pub fn new(py: Python<'_>, manifolds: &Bound<'_, PyList>) -> PyResult<Self> {
        if manifolds.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "manifolds list cannot be empty"
            ));
        }
        
        let mut total_dim = 0;
        let mut total_ambient_dim = 0;
        let mut manifold_objects = Vec::new();
        
        // Collect manifolds and compute total dimensions
        for item in manifolds.iter() {
            let manifold = item.to_object(py);
            
            // Get dimension using Python attribute access
            let dim: usize = manifold.getattr(py, "dim")?.extract(py)?;
            let ambient_dim: usize = manifold.getattr(py, "ambient_dim")?.extract(py)?;
            
            total_dim += dim;
            total_ambient_dim += ambient_dim;
            manifold_objects.push(manifold);
        }
        
        Ok(PyProductManifold {
            manifolds: manifold_objects,
            total_dim,
            total_ambient_dim,
        })
    }
    
    /// String representation of the manifold.
    fn __repr__(&self) -> String {
        format!(
            "ProductManifold(n_manifolds={}, dim={}, ambient_dim={})",
            self.manifolds.len(),
            self.total_dim,
            self.total_ambient_dim
        )
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
    
    /// Get the component manifolds.
    #[getter]
    fn manifolds(&self, py: Python<'_>) -> PyResult<PyObject> {
        Ok(PyList::new_bound(py, &self.manifolds).into())
    }
    
    /// Get the number of component manifolds.
    #[getter]
    fn n_manifolds(&self) -> usize {
        self.manifolds.len()
    }
    
    /// Check if a point lies on the manifold.
    ///
    /// A point is on the product manifold if each component is on its
    /// respective manifold.
    ///
    /// Args:
    ///     point: Tuple of points, one for each component manifold
    ///     atol: Absolute tolerance (default: 1e-10)
    ///
    /// Returns:
    ///     bool: True if all components are on their manifolds
    #[pyo3(signature = (point, atol=1e-10))]
    fn contains(&self, py: Python<'_>, point: PyObject, atol: f64) -> PyResult<bool> {
        let tuple = point.downcast_bound::<PyTuple>(py)?;
        
        if tuple.len() != self.manifolds.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!(
                    "Expected {} components, got {}",
                    self.manifolds.len(),
                    tuple.len()
                )
            ));
        }
        
        // Check each component
        for (i, (manifold, component)) in self.manifolds.iter().zip(tuple.iter()).enumerate() {
            let contains: bool = manifold
                .call_method1(py, "contains", (component, atol))?
                .extract(py)?;
            
            if !contains {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
    
    /// Generate a random point on the product manifold.
    ///
    /// Returns
    /// -------
    /// tuple
    ///     Tuple of random points, one from each component manifold
    pub fn random_point<'py>(&self, py: Python<'py>) -> PyResult<PyObject> {
        let components = self.manifolds
            .iter()
            .map(|manifold| manifold.call_method0(py, "random_point"))
            .collect::<PyResult<Vec<_>>>()?;
        
        Ok(PyTuple::new_bound(py, components).into())
    }
    
    /// Project a point onto the manifold.
    ///
    /// Projects each component onto its respective manifold.
    ///
    /// Parameters
    /// ----------
    /// point : tuple
    ///     Tuple of points to project
    ///
    /// Returns
    /// -------
    /// tuple
    ///     Tuple of projected points
    pub fn project<'py>(&self, py: Python<'py>, point: PyObject) -> PyResult<PyObject> {
        let tuple = point.downcast_bound::<PyTuple>(py)?;
        
        if tuple.len() != self.manifolds.len() {
            return Err(dimension_mismatch(
                &[self.manifolds.len()],
                &[tuple.len()],
            ));
        }
        
        let projected = self.manifolds
            .iter()
            .zip(tuple.iter())
            .map(|(manifold, component)| {
                manifold.call_method1(py, "project", (component,))
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        Ok(PyTuple::new_bound(py, projected).into())
    }
    
    /// Exponential map on the product manifold.
    ///
    /// Applies the exponential map component-wise.
    ///
    /// Parameters
    /// ----------
    /// point : tuple
    ///     Base point (tuple of points)
    /// tangent : tuple
    ///     Tangent vector (tuple of tangent vectors)
    ///
    /// Returns
    /// -------
    /// tuple
    ///     Result of exponential map (tuple of points)
    pub fn exp<'py>(&self, py: Python<'py>, point: PyObject, tangent: PyObject) -> PyResult<PyObject> {
        let point_tuple = point.downcast_bound::<PyTuple>(py)?;
        let tangent_tuple = tangent.downcast_bound::<PyTuple>(py)?;
        
        if point_tuple.len() != self.manifolds.len() || tangent_tuple.len() != self.manifolds.len() {
            return Err(dimension_mismatch(
                &[self.manifolds.len()],
                &[point_tuple.len()],
            ));
        }
        
        let results = self.manifolds
            .iter()
            .zip(point_tuple.iter().zip(tangent_tuple.iter()))
            .map(|(manifold, (p, v))| {
                manifold.call_method1(py, "exp", (p, v))
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        Ok(PyTuple::new_bound(py, results).into())
    }
    
    
    /// Generate a random tangent vector.
    ///
    /// Parameters
    /// ----------
    /// point : tuple
    ///     Base point (tuple of points)
    ///
    /// Returns
    /// -------
    /// tuple
    ///     Random tangent vector (tuple of tangent vectors)
    pub fn random_tangent<'py>(&self, py: Python<'py>, point: PyObject) -> PyResult<PyObject> {
        let point_tuple = point.downcast_bound::<PyTuple>(py)?;
        
        if point_tuple.len() != self.manifolds.len() {
            return Err(dimension_mismatch(
                &[self.manifolds.len()],
                &[point_tuple.len()],
            ));
        }
        
        let tangents = self.manifolds
            .iter()
            .zip(point_tuple.iter())
            .map(|(manifold, pt)| {
                manifold.call_method1(py, "random_tangent", (pt,))
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        Ok(PyTuple::new_bound(py, tangents).into())
    }
    
    /// Project a tangent vector onto the tangent space.
    ///
    /// Projects each component onto its respective tangent space.
    ///
    /// Parameters
    /// ----------
    /// point : tuple
    ///     Base point (tuple of points)
    /// vector : tuple
    ///     Vector to project (tuple of vectors)
    ///
    /// Returns
    /// -------
    /// tuple
    ///     Projected tangent vector (tuple of tangent vectors)
    pub fn project_tangent<'py>(&self, py: Python<'py>, point: PyObject, vector: PyObject) -> PyResult<PyObject> {
        let point_tuple = point.downcast_bound::<PyTuple>(py)?;
        let vector_tuple = vector.downcast_bound::<PyTuple>(py)?;
        
        if point_tuple.len() != self.manifolds.len() || vector_tuple.len() != self.manifolds.len() {
            return Err(dimension_mismatch(
                &[self.manifolds.len()],
                &[point_tuple.len(), vector_tuple.len()],
            ));
        }
        
        let projected = self.manifolds
            .iter()
            .zip(point_tuple.iter().zip(vector_tuple.iter()))
            .map(|(manifold, (pt, vec))| {
                manifold.call_method1(py, "project_tangent", (pt, vec))
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        Ok(PyTuple::new_bound(py, projected).into())
    }
    
    /// Retract a tangent vector.
    ///
    /// Applies the retraction map component-wise.
    ///
    /// Parameters
    /// ----------
    /// point : tuple
    ///     Base point (tuple of points)
    /// tangent : tuple
    ///     Tangent vector (tuple of tangent vectors)
    ///
    /// Returns
    /// -------
    /// tuple
    ///     Result of retraction (tuple of points)
    pub fn retract<'py>(&self, py: Python<'py>, point: PyObject, tangent: PyObject) -> PyResult<PyObject> {
        let point_tuple = point.downcast_bound::<PyTuple>(py)?;
        let tangent_tuple = tangent.downcast_bound::<PyTuple>(py)?;
        
        if point_tuple.len() != self.manifolds.len() || tangent_tuple.len() != self.manifolds.len() {
            return Err(dimension_mismatch(
                &[self.manifolds.len()],
                &[point_tuple.len(), tangent_tuple.len()],
            ));
        }
        
        let retracted = self.manifolds
            .iter()
            .zip(point_tuple.iter().zip(tangent_tuple.iter()))
            .map(|(manifold, (pt, tan))| {
                manifold.call_method1(py, "retract", (pt, tan))
            })
            .collect::<PyResult<Vec<_>>>()?;
        
        Ok(PyTuple::new_bound(py, retracted).into())
    }
    
    /// Compute the inner product in the tangent space.
    ///
    /// The inner product is the sum of component inner products.
    ///
    /// Parameters
    /// ----------
    /// point : tuple
    ///     Base point (tuple of points)
    /// u : tuple
    ///     First tangent vector (tuple of tangent vectors)
    /// v : tuple
    ///     Second tangent vector (tuple of tangent vectors)
    ///
    /// Returns
    /// -------
    /// float
    ///     Inner product value
    pub fn inner<'py>(&self, py: Python<'py>, point: PyObject, u: PyObject, v: PyObject) -> PyResult<f64> {
        let point_tuple = point.downcast_bound::<PyTuple>(py)?;
        let u_tuple = u.downcast_bound::<PyTuple>(py)?;
        let v_tuple = v.downcast_bound::<PyTuple>(py)?;
        
        if point_tuple.len() != self.manifolds.len() || 
           u_tuple.len() != self.manifolds.len() || 
           v_tuple.len() != self.manifolds.len() {
            return Err(dimension_mismatch(
                &[self.manifolds.len()],
                &[point_tuple.len(), u_tuple.len(), v_tuple.len()],
            ));
        }
        
        let mut total_inner = 0.0;
        
        for (i, (manifold, ((pt, u_comp), v_comp))) in self.manifolds
            .iter()
            .zip(point_tuple.iter().zip(u_tuple.iter()).zip(v_tuple.iter()))
            .enumerate() 
        {
            let inner_value: f64 = manifold
                .call_method1(py, "inner", (pt, u_comp, v_comp))?
                .extract(py)?;
            total_inner += inner_value;
        }
        
        Ok(total_inner)
    }
    
    /// Compute the norm of a tangent vector.
    ///
    /// Parameters
    /// ----------
    /// point : tuple
    ///     Base point (tuple of points)
    /// tangent : tuple
    ///     Tangent vector (tuple of tangent vectors)
    ///
    /// Returns
    /// -------
    /// float
    ///     Norm of the tangent vector
    pub fn norm<'py>(&self, py: Python<'py>, point: PyObject, tangent: PyObject) -> PyResult<f64> {
        let inner_product = self.inner(py, point.clone_ref(py), tangent.clone_ref(py), tangent)?;
        Ok(inner_product.sqrt())
    }
    
    /// Compute the Riemannian distance between two points.
    ///
    /// The distance is the square root of the sum of squared component distances.
    ///
    /// Parameters
    /// ----------
    /// x : tuple
    ///     First point (tuple of points)
    /// y : tuple
    ///     Second point (tuple of points)
    ///
    /// Returns
    /// -------
    /// float
    ///     Distance between x and y
    pub fn dist<'py>(&self, py: Python<'py>, x: PyObject, y: PyObject) -> PyResult<f64> {
        let x_tuple = x.downcast_bound::<PyTuple>(py)?;
        let y_tuple = y.downcast_bound::<PyTuple>(py)?;
        
        if x_tuple.len() != self.manifolds.len() || y_tuple.len() != self.manifolds.len() {
            return Err(dimension_mismatch(
                &[self.manifolds.len()],
                &[x_tuple.len(), y_tuple.len()],
            ));
        }
        
        let mut squared_distance = 0.0;
        
        for (manifold, (x_comp, y_comp)) in self.manifolds
            .iter()
            .zip(x_tuple.iter().zip(y_tuple.iter())) 
        {
            let dist: f64 = manifold
                .call_method1(py, "dist", (x_comp, y_comp))?
                .extract(py)?;
            squared_distance += dist * dist;
        }
        
        Ok(squared_distance.sqrt())
    }
}

// Internal methods would handle the complexity of mixed point types
impl PyProductManifold {
    /// Convert a tuple of component points to a concatenated vector representation
    fn tuple_to_vector(&self, py: Python<'_>, tuple: &Bound<'_, PyTuple>) -> PyResult<DVector<f64>> {
        if tuple.len() != self.manifolds.len() {
            return Err(dimension_mismatch(
                &[self.manifolds.len()],
                &[tuple.len()],
            ));
        }
        
        let mut concatenated = DVector::zeros(self.total_ambient_dim);
        let mut offset = 0;
        
        for (manifold, component) in self.manifolds.iter().zip(tuple.iter()) {
            // Get the numpy array representation of this component
            let array = if let Ok(arr) = component.downcast::<numpy::PyArray1<f64>>() {
                // Vector case
                numpy_to_dvector(arr.readonly())?
            } else if let Ok(arr) = component.downcast::<numpy::PyArray2<f64>>() {
                // Matrix case - flatten to vector
                let mat = numpy_to_dmatrix(arr.readonly())?;
                DVector::from_iterator(mat.nrows() * mat.ncols(), mat.iter().cloned())
            } else {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "Product manifold components must be numpy arrays"
                ));
            };
            
            // Copy to the appropriate position in the concatenated vector
            let component_dim = array.len();
            concatenated.rows_mut(offset, component_dim).copy_from(&array);
            offset += component_dim;
        }
        
        Ok(concatenated)
    }
    
    /// Convert a concatenated vector to a tuple of component points
    fn vector_to_tuple(&self, py: Python<'_>, vector: &DVector<f64>) -> PyResult<PyObject> {
        if vector.len() != self.total_ambient_dim {
            return Err(dimension_mismatch(
                &[self.total_ambient_dim],
                &[vector.len()],
            ));
        }
        
        let mut components: Vec<PyObject> = Vec::new();
        let mut offset = 0;
        
        for manifold in &self.manifolds {
            // Get the ambient dimension of this component
            let ambient_dim: usize = manifold.getattr(py, "ambient_dim")?.extract(py)?;
            
            // Extract the component vector
            let component_vec = vector.rows(offset, ambient_dim).clone_owned();
            
            // Check if this manifold uses matrix representation
            let point_type: String = manifold.getattr(py, "__class__")?
                .getattr(py, "__name__")?.extract(py)?;
            
            let component_array = match point_type.as_str() {
                "Sphere" | "Hyperbolic" => {
                    // Vector manifolds
                    dvector_to_numpy(py, &component_vec)?.into()
                }
                "Stiefel" | "Grassmann" | "SPD" | "Oblique" | "PSDCone" => {
                    // Matrix manifolds - need to get shape
                    if let Ok(n) = manifold.getattr(py, "n") {
                        let n: usize = n.extract(py)?;
                        if let Ok(p) = manifold.getattr(py, "p") {
                            // Stiefel, Grassmann
                            let p: usize = p.extract(py)?;
                            let mat = nalgebra::DMatrix::from_iterator(
                                n, p, component_vec.iter().cloned()
                            );
                            dmatrix_to_numpy(py, &mat)?.into()
                        } else {
                            // SPD, PSDCone - square matrices
                            let mat = nalgebra::DMatrix::from_iterator(
                                n, n, component_vec.iter().cloned()
                            );
                            dmatrix_to_numpy(py, &mat)?.into()
                        }
                    } else if let Ok(shape) = manifold.getattr(py, "shape") {
                        // Oblique manifold
                        let shape: (usize, usize) = shape.extract(py)?;
                        let mat = nalgebra::DMatrix::from_iterator(
                            shape.0, shape.1, component_vec.iter().cloned()
                        );
                        crate::array_utils::dmatrix_to_numpy(py, &mat)?.into()
                    } else {
                        // Default to vector
                        dvector_to_numpy(py, &component_vec)?.into()
                    }
                }
                _ => {
                    // Unknown manifold type, default to vector
                    dvector_to_numpy(py, &component_vec)?.into()
                }
            };
            
            components.push(component_array);
            offset += ambient_dim;
        }
        
        Ok(PyTuple::new_bound(py, components).into())
    }
}