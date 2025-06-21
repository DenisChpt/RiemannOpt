//! Python bindings for Riemannian Newton optimizer.

use numpy::{PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use crate::manifolds_optimized::{PyStiefel, PyGrassmann, PySPD};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use nalgebra::{DVector, DMatrix};

use riemannopt_core::{
    manifold::Manifold,
    optimizer::{Optimizer, StoppingCriterion},
};
use riemannopt_optim::{Newton, NewtonConfig};

use crate::manifolds::*;
use crate::manifolds_oblique::PyOblique;
use crate::manifolds_fixed_rank::PyFixedRank;
use crate::manifolds_psd_cone::PyPSDCone;
use crate::cost_function::PyCostFunction;

// Helper function to convert numpy array to nalgebra matrix
fn numpy_to_nalgebra_matrix(array: &PyReadonlyArray2<'_, f64>) -> PyResult<DMatrix<f64>> {
    let shape = array.shape();
    let data = array.as_slice()?.to_vec();
    
    // Note: NumPy arrays are row-major while nalgebra expects column-major
    // So we need to transpose
    let mat = DMatrix::from_vec(shape[1], shape[0], data);
    Ok(mat.transpose())
}

/// Riemannian Newton method optimizer.
#[pyclass(name = "Newton")]
pub struct PyNewton {
    config: NewtonConfig<f64>,
    max_iterations: usize,
    tolerance: f64,
}

#[pymethods]
impl PyNewton {
    /// Create a new Newton optimizer.
    ///
    /// Args:
    ///     hessian_regularization: Regularization for Hessian (default: 1e-8)
    ///     use_gauss_newton: Use Gauss-Newton approximation (default: False)
    ///     max_cg_iterations: Maximum CG iterations (default: 100)
    ///     cg_tolerance: CG solver tolerance (default: 1e-6)
    ///     max_iterations: Maximum iterations (default: 100)
    ///     tolerance: Convergence tolerance (default: 1e-6)
    #[new]
    #[pyo3(signature = (hessian_regularization=1e-8, use_gauss_newton=false, max_cg_iterations=100, cg_tolerance=1e-6, max_iterations=100, tolerance=1e-6))]
    pub fn new(
        hessian_regularization: f64,
        use_gauss_newton: bool,
        max_cg_iterations: usize,
        cg_tolerance: f64,
        max_iterations: i32,
        tolerance: f64,
    ) -> PyResult<Self> {
        // Validate parameters
        if hessian_regularization < 0.0 {
            return Err(PyValueError::new_err(
                "Newton hessian_regularization must be non-negative"
            ));
        }
        
        if max_cg_iterations == 0 {
            return Err(PyValueError::new_err(
                "Newton max_cg_iterations must be positive"
            ));
        }
        
        if cg_tolerance <= 0.0 {
            return Err(PyValueError::new_err(
                "Newton cg_tolerance must be positive"
            ));
        }
        
        if max_iterations <= 0 {
            return Err(PyValueError::new_err(
                "max_iterations must be positive"
            ));
        }
        
        if tolerance <= 0.0 {
            return Err(PyValueError::new_err(
                "tolerance must be positive"
            ));
        }
        
        let mut config = NewtonConfig::new()
            .with_regularization(hessian_regularization)
            .with_cg_params(max_cg_iterations, cg_tolerance);
            
        if use_gauss_newton {
            config = config.with_gauss_newton();
        }
        
        Ok(Self {
            config,
            max_iterations: max_iterations as usize,
            tolerance,
        })
    }
    
    /// Perform one optimization step.
    ///
    /// Args:
    ///     manifold: The manifold to optimize on
    ///     point: Current point (numpy array)
    ///     gradient: Current gradient (numpy array)
    ///
    /// Returns:
    ///     New point after taking the step
    pub fn step<'py>(
        &mut self,
        _py: Python<'py>,
        _manifold: &Bound<'_, PyAny>,
        point: PyReadonlyArray1<'_, f64>,
        gradient: PyReadonlyArray1<'_, f64>,
    ) -> PyResult<PyObject> {
        let _point_vec = point.as_slice()?.to_vec();
        let _gradient_vec = gradient.as_slice()?.to_vec();
        
        // Note: This is a simplified version. A full implementation would need
        // access to the cost function to compute Hessian-vector products.
        // For now, we return an error indicating this limitation.
        
        Err(PyValueError::new_err(
            "Newton optimizer's step method requires full optimize() call with cost function. \
             Direct step() is not supported due to need for Hessian computation."
        ))
    }
    
    /// Optimize a cost function on a manifold.
    ///
    /// Args:
    ///     cost_fn: The cost function to minimize
    ///     manifold: The manifold to optimize on
    ///     initial_point: Starting point (numpy array)
    ///
    /// Returns:
    ///     Optimization result dictionary
    pub fn optimize<'py>(
        &mut self,
        py: Python<'py>,
        cost_fn: &PyCostFunction,
        manifold: &Bound<'_, PyAny>,
        initial_point: &Bound<'_, PyAny>,
    ) -> PyResult<PyObject> {
        
        // Extract initial point based on manifold type
        let (x0, shape) = if let Ok(_sphere) = manifold.extract::<PyRef<PySphere>>() {
            let point_array = initial_point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let x0 = DVector::from_vec(point_array.as_slice()?.to_vec());
            (x0, vec![])
        } else if let Ok(_stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            let point_array = initial_point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let x0 = DVector::from_vec(point_mat.as_slice().to_vec());
            (x0, shape)
        } else if let Ok(_grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            let point_array = initial_point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let x0 = DVector::from_vec(point_mat.as_slice().to_vec());
            (x0, shape)
        } else if let Ok(_spd) = manifold.extract::<PyRef<PySPD>>() {
            let point_array = initial_point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let x0 = DVector::from_vec(point_mat.as_slice().to_vec());
            (x0, shape)
        } else if let Ok(_hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
            let point_array = initial_point.extract::<PyReadonlyArray1<'_, f64>>()?;
            let x0 = DVector::from_vec(point_array.as_slice()?.to_vec());
            (x0, vec![])
        } else if let Ok(_oblique) = manifold.extract::<PyRef<PyOblique>>() {
            let point_array = initial_point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let x0 = DVector::from_vec(point_mat.as_slice().to_vec());
            (x0, shape)
        } else if let Ok(_fixed_rank) = manifold.extract::<PyRef<PyFixedRank>>() {
            let point_array = initial_point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let x0 = DVector::from_vec(point_mat.as_slice().to_vec());
            (x0, shape)
        } else if let Ok(_psd_cone) = manifold.extract::<PyRef<PyPSDCone>>() {
            let point_array = initial_point.extract::<PyReadonlyArray2<'_, f64>>()?;
            let shape = vec![point_array.shape()[0], point_array.shape()[1]];
            let point_mat = numpy_to_nalgebra_matrix(&point_array)?;
            let x0 = DVector::from_vec(point_mat.as_slice().to_vec());
            (x0, shape)
        } else {
            return Err(PyValueError::new_err(
                "Unsupported manifold type for Newton optimizer"
            ));
        };
        
        let criterion = StoppingCriterion::new()
            .with_max_iterations(self.max_iterations)
            .with_gradient_tolerance(self.tolerance);
        
        // Check manifold type and perform optimization
        if let Ok(sphere) = manifold.extract::<PyRef<PySphere>>() {
            self.optimize_on_manifold(py, cost_fn, sphere.get_inner(), x0, criterion, shape)
        } else if let Ok(stiefel) = manifold.extract::<PyRef<PyStiefel>>() {
            self.optimize_on_manifold(py, cost_fn, stiefel.get_inner(), x0, criterion, shape)
        } else if let Ok(grassmann) = manifold.extract::<PyRef<PyGrassmann>>() {
            self.optimize_on_manifold(py, cost_fn, grassmann.get_inner(), x0, criterion, shape)
        } else if let Ok(_euclidean) = manifold.extract::<PyRef<PyEuclidean>>() {
            // For now, Euclidean manifold is not supported directly in Newton
            Err(PyValueError::new_err(
                "Euclidean manifold not yet supported in Newton optimizer. \
                 Please use one of the other manifolds."
            ))
        } else if let Ok(spd) = manifold.extract::<PyRef<PySPD>>() {
            self.optimize_on_manifold(py, cost_fn, spd.get_inner(), x0, criterion, shape)
        } else if let Ok(hyperbolic) = manifold.extract::<PyRef<PyHyperbolic>>() {
            self.optimize_on_manifold(py, cost_fn, hyperbolic.get_inner(), x0, criterion, shape)
        } else if let Ok(oblique) = manifold.extract::<PyRef<PyOblique>>() {
            self.optimize_on_manifold(py, cost_fn, oblique.get_inner(), x0, criterion, shape)
        } else if let Ok(fixed_rank) = manifold.extract::<PyRef<PyFixedRank>>() {
            self.optimize_on_manifold(py, cost_fn, fixed_rank.get_inner(), x0, criterion, shape)
        } else if let Ok(psd_cone) = manifold.extract::<PyRef<PyPSDCone>>() {
            self.optimize_on_manifold(py, cost_fn, psd_cone.get_inner(), x0, criterion, shape)
        } else {
            Err(PyValueError::new_err(
                "Unsupported manifold type for Newton optimizer"
            ))
        }
    }
    
    /// String representation
    pub fn __repr__(&self) -> String {
        format!(
            "Newton(hessian_regularization={:.0e}, use_gauss_newton={}, max_cg_iterations={}, cg_tolerance={:.0e}, max_iterations={}, tolerance={:.0e})",
            self.config.hessian_regularization,
            self.config.use_gauss_newton,
            self.config.max_cg_iterations,
            self.config.cg_tolerance,
            self.max_iterations,
            self.tolerance
        )
    }
}

// Internal implementation methods
impl PyNewton {
    // Internal optimization method
    fn optimize_on_manifold<M: Manifold<f64, nalgebra::Dyn>>(
        &mut self,
        py: Python<'_>,
        cost_fn: &PyCostFunction,
        manifold: &M,
        x0: DVector<f64>,
        criterion: StoppingCriterion<f64>,
        shape: Vec<usize>,
    ) -> PyResult<PyObject> {
        let mut optimizer = Newton::new(self.config.clone());
        
        let result = optimizer.optimize(cost_fn, manifold, &x0, &criterion)
            .map_err(|e| PyValueError::new_err(format!("Optimization failed: {}", e)))?;
        
        // Convert result to Python dictionary
        let dict = pyo3::types::PyDict::new_bound(py);
        
        // Convert point to numpy array with appropriate shape
        if shape.is_empty() {
            // 1D array
            let point_array = numpy::PyArray1::from_vec_bound(py, result.point.data.as_vec().clone());
            dict.set_item("point", point_array)?;
        } else {
            // 2D array - reshape the result
            let data = result.point.data.as_vec().clone();
            let n_rows = shape[0];
            let n_cols = shape[1];
            
            // Convert to 2D array by creating a vector of vectors
            let mut rows = Vec::with_capacity(n_rows);
            for i in 0..n_rows {
                let mut row = Vec::with_capacity(n_cols);
                for j in 0..n_cols {
                    row.push(data[j * n_rows + i]); // column-major to row-major
                }
                rows.push(row);
            }
            
            let point_array = numpy::PyArray2::from_vec2_bound(py, &rows)?;
            dict.set_item("point", point_array)?;
        }
        
        dict.set_item("value", result.value)?;
        dict.set_item("iterations", result.iterations)?;
        dict.set_item("converged", result.converged)?;
        dict.set_item("gradient_norm", result.gradient_norm)?;
        dict.set_item("function_evaluations", result.function_evaluations)?;
        dict.set_item("gradient_evaluations", result.gradient_evaluations)?;
        dict.set_item("duration_seconds", result.duration.as_secs_f64())?;
        dict.set_item("termination_reason", format!("{:?}", result.termination_reason))?;
        
        Ok(dict.into())
    }
}