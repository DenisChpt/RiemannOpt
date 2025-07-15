//! Generic optimizer implementation for all manifolds.

use pyo3::prelude::*;
use nalgebra::{DVector, DMatrix};
use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use riemannopt_core::{
    manifold::Manifold,
    core::{MatrixManifold, CachedDynamicCostFunction},
    optimizer::StoppingCriterion,
};

use crate::{
    array_utils::{numpy_to_dvector, numpy_to_dmatrix, dvector_to_numpy, dmatrix_to_numpy},
    error::to_py_err,
    py_cost::{PyCostFunction, PyCostFunctionVector, PyCostFunctionMatrix},
    py_manifolds::{
        sphere::PySphere,
        stiefel::PyStiefel,
        grassmann::PyGrassmann,
        spd::PySPD,
        hyperbolic::PyHyperbolic,
        oblique::PyOblique,
        // fixed_rank::PyFixedRank,  // TODO: Fix FixedRankPoint representation mismatch
        psd_cone::PyPSDCone,
    },
};

use super::base::PyOptimizationResult;

/// Trait for Python optimizers that can optimize on any manifold
pub trait PyOptimizerGeneric {
    /// Optimize on a Sphere manifold
    fn optimize_sphere_impl(
        &mut self,
        py: Python<'_>,
        cost_function: &PyCostFunction,
        sphere: &PySphere,
        initial_point: PyReadonlyArray1<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyOptimizationResult>;
    
    /// Optimize on a Stiefel manifold
    fn optimize_stiefel_impl(
        &mut self,
        py: Python<'_>,
        cost_function: &PyCostFunction,
        stiefel: &PyStiefel,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyOptimizationResult>;
    
    /// Optimize on a Grassmann manifold
    fn optimize_grassmann_impl(
        &mut self,
        py: Python<'_>,
        cost_function: &PyCostFunction,
        grassmann: &PyGrassmann,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyOptimizationResult>;
    
    /// Optimize on an SPD manifold
    fn optimize_spd_impl(
        &mut self,
        py: Python<'_>,
        cost_function: &PyCostFunction,
        spd: &PySPD,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyOptimizationResult>;
    
    /// Optimize on a Hyperbolic manifold
    fn optimize_hyperbolic_impl(
        &mut self,
        py: Python<'_>,
        cost_function: &PyCostFunction,
        hyperbolic: &PyHyperbolic,
        initial_point: PyReadonlyArray1<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyOptimizationResult>;
    
    /// Optimize on an Oblique manifold
    fn optimize_oblique_impl(
        &mut self,
        py: Python<'_>,
        cost_function: &PyCostFunction,
        oblique: &PyOblique,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyOptimizationResult>;
    
    // /// Optimize on a FixedRank manifold
    // fn optimize_fixed_rank_impl(
    //     &mut self,
    //     py: Python<'_>,
    //     cost_function: &PyCostFunction,
    //     fixed_rank: &PyFixedRank,
    //     initial_point: PyReadonlyArray2<'_, f64>,
    //     max_iterations: usize,
    //     gradient_tolerance: Option<f64>,
    // ) -> PyResult<PyOptimizationResult>;
    
    /// Optimize on a PSDCone manifold
    fn optimize_psd_cone_impl(
        &mut self,
        py: Python<'_>,
        cost_function: &PyCostFunction,
        psd_cone: &PyPSDCone,
        initial_point: PyReadonlyArray2<'_, f64>,
        max_iterations: usize,
        gradient_tolerance: Option<f64>,
    ) -> PyResult<PyOptimizationResult>;
}

/// Macro to generate optimize methods for all manifolds
/// This macro should be invoked inside a #[pymethods] impl block
#[macro_export]
macro_rules! impl_optimizer_methods {
    () => {
            /// Optimize on a Sphere manifold
            #[pyo3(signature = (cost_function, sphere, initial_point, max_iterations, gradient_tolerance=None))]
            pub fn optimize_sphere(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                sphere: PyRef<'_, PySphere>,
                initial_point: PyReadonlyArray1<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyObject> {
                self.optimize_sphere_impl(
                    py, &*cost_function, &*sphere, initial_point, 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            }
            
            /// Optimize on a Stiefel manifold
            #[pyo3(signature = (cost_function, stiefel, initial_point, max_iterations, gradient_tolerance=None))]
            pub fn optimize_stiefel(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                stiefel: PyRef<'_, PyStiefel>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyObject> {
                self.optimize_stiefel_impl(
                    py, &*cost_function, &*stiefel, initial_point, 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            }
            /// Optimize on a Grassmann manifold
            #[pyo3(signature = (cost_function, grassmann, initial_point, max_iterations, gradient_tolerance=None))]
            pub fn optimize_grassmann(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                grassmann: PyRef<'_, PyGrassmann>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyObject> {
                self.optimize_grassmann_impl(
                    py, &*cost_function, &*grassmann, initial_point, 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            }
            
            /// Optimize on an SPD manifold
            #[pyo3(signature = (cost_function, spd, initial_point, max_iterations, gradient_tolerance=None))]
            pub fn optimize_spd(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                spd: PyRef<'_, PySPD>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyObject> {
                self.optimize_spd_impl(
                    py, &*cost_function, &*spd, initial_point, 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            }
            
            /// Optimize on a Hyperbolic manifold
            #[pyo3(signature = (cost_function, hyperbolic, initial_point, max_iterations, gradient_tolerance=None))]
            pub fn optimize_hyperbolic(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                hyperbolic: PyRef<'_, PyHyperbolic>,
                initial_point: PyReadonlyArray1<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyObject> {
                self.optimize_hyperbolic_impl(
                    py, &*cost_function, &*hyperbolic, initial_point, 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            }
            
            /// Optimize on an Oblique manifold
            #[pyo3(signature = (cost_function, oblique, initial_point, max_iterations, gradient_tolerance=None))]
            pub fn optimize_oblique(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                oblique: PyRef<'_, PyOblique>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyObject> {
                self.optimize_oblique_impl(
                    py, &*cost_function, &*oblique, initial_point, 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            }
            
            // /// Optimize on a FixedRank manifold
            // #[pyo3(signature = (cost_function, fixed_rank, initial_point, max_iterations, gradient_tolerance=None))]
            // pub fn optimize_fixed_rank(
            //     &mut self,
            //     py: Python<'_>,
            //     cost_function: PyRef<'_, PyCostFunction>,
            //     fixed_rank: PyRef<'_, PyFixedRank>,
            //     initial_point: PyReadonlyArray2<'_, f64>,
            //     max_iterations: usize,
            //     gradient_tolerance: Option<f64>,
            // ) -> PyResult<PyObject> {
            //     self.optimize_fixed_rank_impl(
            //         py, &*cost_function, &*fixed_rank, initial_point, 
            //         max_iterations, gradient_tolerance
            //     ).map(|r| r.into_py(py))
            // }
            
            /// Optimize on a PSDCone manifold
            #[pyo3(signature = (cost_function, psd_cone, initial_point, max_iterations, gradient_tolerance=None))]
            pub fn optimize_psd_cone(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                psd_cone: PyRef<'_, PyPSDCone>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyObject> {
                self.optimize_psd_cone_impl(
                    py, &*cost_function, &*psd_cone, initial_point, 
                    max_iterations, gradient_tolerance
                ).map(|r| r.into_py(py))
            }
    };
}

/// Create cost function adapters for vector manifolds
pub struct PyCostFunctionHyperbolic<'a> {
    inner: &'a PyCostFunction,
}

impl<'a> PyCostFunctionHyperbolic<'a> {
    pub fn new(inner: &'a PyCostFunction) -> Self {
        Self { inner }
    }
}

/// Macro to implement PyOptimizerGeneric trait with default implementations
#[macro_export]
macro_rules! impl_optimizer_generic_default {
    ($optimizer_type:ty, $rust_optimizer:ty, $config_type:ty, $create_config:expr) => {
        impl crate::py_optimizers::generic::PyOptimizerGeneric for $optimizer_type {
            fn optimize_sphere_impl(
                &mut self,
                py: Python<'_>,
                cost_function: &PyCostFunction,
                sphere: &PySphere,
                initial_point: PyReadonlyArray1<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyOptimizationResult> {
                use crate::py_cost::PyCostFunctionSphere;
                use riemannopt_core::core::CachedDynamicCostFunction;
                
                let x0 = numpy_to_dvector(initial_point)?;
                let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
                if let Some(tol) = gradient_tolerance {
                    criterion = criterion.with_gradient_tolerance(tol);
                }
                
                let config: $config_type = $create_config(self);
                let mut optimizer = <$rust_optimizer>::new(config);
                let cost_fn = PyCostFunctionSphere::new(cost_function);
                let cost_fn = CachedDynamicCostFunction::new(cost_fn);
                
                let result = py.allow_threads(|| {
                    optimizer.optimize(&cost_fn, &sphere.inner, &x0, &criterion)
                }).map_err(to_py_err)?;
                
                PyOptimizationResult::from_rust_result(py, result, |point| {
                    Ok(dvector_to_numpy(py, point)?.into())
                })
            }
            
            fn optimize_stiefel_impl(
                &mut self,
                py: Python<'_>,
                cost_function: &PyCostFunction,
                stiefel: &PyStiefel,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyOptimizationResult> {
                use crate::py_cost::PyCostFunctionStiefel;
                
                let x0 = numpy_to_dmatrix(initial_point)?;
                let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
                if let Some(tol) = gradient_tolerance {
                    criterion = criterion.with_gradient_tolerance(tol);
                }
                
                let config: $config_type = $create_config(self);
                let mut optimizer = <$rust_optimizer>::new(config);
                let cost_fn = PyCostFunctionStiefel::new(cost_function);
                // Note: CachedDynamicCostFunction only works with DVector types, not DMatrix
                
                let result = py.allow_threads(|| {
                    optimizer.optimize(&cost_fn, &stiefel.inner, &x0, &criterion)
                }).map_err(to_py_err)?;
                
                PyOptimizationResult::from_rust_result(py, result, |point| {
                    Ok(dmatrix_to_numpy(py, point)?.into())
                })
            }
            
            fn optimize_grassmann_impl(
                &mut self,
                py: Python<'_>,
                cost_function: &PyCostFunction,
                grassmann: &PyGrassmann,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyOptimizationResult> {
                use crate::py_cost::PyCostFunctionMatrix;
                
                let x0 = numpy_to_dmatrix(initial_point)?;
                let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
                if let Some(tol) = gradient_tolerance {
                    criterion = criterion.with_gradient_tolerance(tol);
                }
                
                let config: $config_type = $create_config(self);
                let mut optimizer = <$rust_optimizer>::new(config);
                let cost_fn = PyCostFunctionMatrix::new(cost_function);
                
                let result = py.allow_threads(|| {
                    optimizer.optimize(&cost_fn, &grassmann.inner, &x0, &criterion)
                }).map_err(to_py_err)?;
                
                PyOptimizationResult::from_rust_result(py, result, |point| {
                    Ok(dmatrix_to_numpy(py, point)?.into())
                })
            }
            
            fn optimize_spd_impl(
                &mut self,
                py: Python<'_>,
                cost_function: &PyCostFunction,
                spd: &PySPD,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyOptimizationResult> {
                use crate::py_cost::PyCostFunctionMatrix;
                
                let x0 = numpy_to_dmatrix(initial_point)?;
                let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
                if let Some(tol) = gradient_tolerance {
                    criterion = criterion.with_gradient_tolerance(tol);
                }
                
                let config: $config_type = $create_config(self);
                let mut optimizer = <$rust_optimizer>::new(config);
                let cost_fn = PyCostFunctionMatrix::new(cost_function);
                
                let result = py.allow_threads(|| {
                    optimizer.optimize(&cost_fn, &spd.inner, &x0, &criterion)
                }).map_err(to_py_err)?;
                
                PyOptimizationResult::from_rust_result(py, result, |point| {
                    Ok(dmatrix_to_numpy(py, point)?.into())
                })
            }
            
            fn optimize_hyperbolic_impl(
                &mut self,
                py: Python<'_>,
                cost_function: &PyCostFunction,
                hyperbolic: &PyHyperbolic,
                initial_point: PyReadonlyArray1<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyOptimizationResult> {
                use crate::py_cost::PyCostFunctionVector;
                use riemannopt_core::core::CachedDynamicCostFunction;
                
                let x0 = numpy_to_dvector(initial_point)?;
                let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
                if let Some(tol) = gradient_tolerance {
                    criterion = criterion.with_gradient_tolerance(tol);
                }
                
                let config: $config_type = $create_config(self);
                let mut optimizer = <$rust_optimizer>::new(config);
                let cost_fn = PyCostFunctionVector::new(cost_function);
                let cost_fn = CachedDynamicCostFunction::new(cost_fn);
                
                let result = py.allow_threads(|| {
                    optimizer.optimize(&cost_fn, &hyperbolic.inner, &x0, &criterion)
                }).map_err(to_py_err)?;
                
                PyOptimizationResult::from_rust_result(py, result, |point| {
                    Ok(dvector_to_numpy(py, point)?.into())
                })
            }
            
            fn optimize_oblique_impl(
                &mut self,
                py: Python<'_>,
                cost_function: &PyCostFunction,
                oblique: &PyOblique,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyOptimizationResult> {
                use crate::py_cost::PyCostFunctionMatrix;
                
                let x0 = numpy_to_dmatrix(initial_point)?;
                let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
                if let Some(tol) = gradient_tolerance {
                    criterion = criterion.with_gradient_tolerance(tol);
                }
                
                let config: $config_type = $create_config(self);
                let mut optimizer = <$rust_optimizer>::new(config);
                let cost_fn = PyCostFunctionMatrix::new(cost_function);
                
                let result = py.allow_threads(|| {
                    optimizer.optimize(&cost_fn, &oblique.inner, &x0, &criterion)
                }).map_err(to_py_err)?;
                
                PyOptimizationResult::from_rust_result(py, result, |point| {
                    Ok(dmatrix_to_numpy(py, point)?.into())
                })
            }
            
            // fn optimize_fixed_rank_impl(
            //     &mut self,
            //     py: Python<'_>,
            //     cost_function: &PyCostFunction,
            //     fixed_rank: &PyFixedRank,
            //     initial_point: PyReadonlyArray2<'_, f64>,
            //     max_iterations: usize,
            //     gradient_tolerance: Option<f64>,
            // ) -> PyResult<PyOptimizationResult> {
            //     use crate::py_cost::PyCostFunctionMatrix;
            //     
            //     let x0 = numpy_to_dmatrix(initial_point)?;
            //     let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
            //     if let Some(tol) = gradient_tolerance {
            //         criterion = criterion.with_gradient_tolerance(tol);
            //     }
            //     
            //     let config: $config_type = $create_config(self);
            //     let mut optimizer = <$rust_optimizer>::new(config);
            //     let cost_fn = PyCostFunctionMatrix::new(cost_function);
            //     
            //     let result = py.allow_threads(|| {
            //         optimizer.optimize(&cost_fn, &fixed_rank.inner, &x0, &criterion)
            //     }).map_err(to_py_err)?;
            //     
            //     PyOptimizationResult::from_rust_result(py, result, |point| {
            //         Ok(dmatrix_to_numpy(py, point)?.into())
            //     })
            // }
            
            fn optimize_psd_cone_impl(
                &mut self,
                py: Python<'_>,
                cost_function: &PyCostFunction,
                psd_cone: &PyPSDCone,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
            ) -> PyResult<PyOptimizationResult> {
                use crate::py_cost::PyCostFunctionPSDCone;
                
                let x0_mat = numpy_to_dmatrix(initial_point)?;
                // Convert matrix to vector representation
                let x0 = psd_cone.inner.matrix_to_vector(&x0_mat);
                
                let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
                if let Some(tol) = gradient_tolerance {
                    criterion = criterion.with_gradient_tolerance(tol);
                }
                
                let config: $config_type = $create_config(self);
                let mut optimizer = <$rust_optimizer>::new(config);
                let cost_fn = PyCostFunctionPSDCone::new(cost_function, psd_cone.n);
                
                let result = py.allow_threads(|| {
                    optimizer.optimize(&cost_fn, &psd_cone.inner, &x0, &criterion)
                }).map_err(to_py_err)?;
                
                PyOptimizationResult::from_rust_result(py, result, |point| {
                    // Convert vector back to matrix
                    let point_mat = psd_cone.inner.vector_to_matrix(point);
                    Ok(dmatrix_to_numpy(py, &point_mat)?.into())
                })
            }
        }
    };
}