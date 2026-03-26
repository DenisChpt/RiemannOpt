//! Generic optimizer implementation for all manifolds.
//!
//! Uses an enum-based dispatch to route optimization to the correct manifold
//! type, replacing fragile `if/else if` chains with exhaustive `match`.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

use crate::{
	py_cost::PyCostFunction,
	py_manifolds::{
		euclidean::PyEuclidean, fixed_rank::PyFixedRank, grassmann::PyGrassmann,
		hyperbolic::PyHyperbolic, oblique::PyOblique, psd_cone::PyPSDCone, spd::PySPD,
		sphere::PySphere, stiefel::PyStiefel,
	},
};

use super::base::PyOptimizationResult;

// ---------------------------------------------------------------------------
// Manifold / point enum dispatch
// ---------------------------------------------------------------------------

/// Extracted manifold reference, avoiding long `if/else if` chains.
pub enum ManifoldKind<'py> {
	Sphere(PyRef<'py, PySphere>),
	Stiefel(PyRef<'py, PyStiefel>),
	Grassmann(PyRef<'py, PyGrassmann>),
	SPD(PyRef<'py, PySPD>),
	Hyperbolic(PyRef<'py, PyHyperbolic>),
	Oblique(PyRef<'py, PyOblique>),
	PSDCone(PyRef<'py, PyPSDCone>),
	Euclidean(PyRef<'py, PyEuclidean>),
	FixedRank(PyRef<'py, PyFixedRank>),
}

impl<'py> ManifoldKind<'py> {
	/// Try to extract a manifold from a Python object.
	pub fn extract(py: Python<'py>, obj: &'py Py<PyAny>) -> PyResult<Self> {
		if let Ok(m) = obj.extract::<PyRef<PySphere>>(py) {
			return Ok(Self::Sphere(m));
		}
		if let Ok(m) = obj.extract::<PyRef<PyStiefel>>(py) {
			return Ok(Self::Stiefel(m));
		}
		if let Ok(m) = obj.extract::<PyRef<PyGrassmann>>(py) {
			return Ok(Self::Grassmann(m));
		}
		if let Ok(m) = obj.extract::<PyRef<PySPD>>(py) {
			return Ok(Self::SPD(m));
		}
		if let Ok(m) = obj.extract::<PyRef<PyHyperbolic>>(py) {
			return Ok(Self::Hyperbolic(m));
		}
		if let Ok(m) = obj.extract::<PyRef<PyOblique>>(py) {
			return Ok(Self::Oblique(m));
		}
		if let Ok(m) = obj.extract::<PyRef<PyPSDCone>>(py) {
			return Ok(Self::PSDCone(m));
		}
		if let Ok(m) = obj.extract::<PyRef<PyEuclidean>>(py) {
			return Ok(Self::Euclidean(m));
		}
		if let Ok(m) = obj.extract::<PyRef<PyFixedRank>>(py) {
			return Ok(Self::FixedRank(m));
		}
		Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
			"Unsupported manifold type. Supported: Sphere, Stiefel, Grassmann, SPD, Hyperbolic, Oblique, PSDCone, Euclidean, FixedRank",
		))
	}
}

/// Generic dispatcher function that routes optimization to the correct manifold implementation.
pub fn optimize_dispatcher<O: PyOptimizerGeneric>(
	optimizer: &mut O,
	py: Python<'_>,
	cost_function: PyRef<'_, PyCostFunction>,
	manifold: Py<PyAny>,
	initial_point: Py<PyAny>,
	max_iterations: usize,
	gradient_tolerance: Option<f64>,
	function_tolerance: Option<f64>,
	point_tolerance: Option<f64>,
	_callback: Option<Py<PyAny>>, // For future use
	_target_value: Option<f64>,  // For future use
	_max_time: Option<f64>,      // For future use
) -> PyResult<Py<PyAny>> {
	use numpy::{PyArray1, PyArray2, PyArrayMethods};

	let kind = ManifoldKind::extract(py, &manifold)?;

	match kind {
		ManifoldKind::Sphere(sphere) => {
			let arr = initial_point
				.downcast_bound::<PyArray1<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 1D numpy array for Sphere manifold",
					)
				})?;
			optimizer
				.optimize_sphere_impl(
					py,
					&*cost_function,
					&*sphere,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
		ManifoldKind::Stiefel(stiefel) => {
			let arr = initial_point
				.downcast_bound::<PyArray2<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 2D numpy array for Stiefel manifold",
					)
				})?;
			optimizer
				.optimize_stiefel_impl(
					py,
					&*cost_function,
					&*stiefel,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
		ManifoldKind::Grassmann(grassmann) => {
			let arr = initial_point
				.downcast_bound::<PyArray2<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 2D numpy array for Grassmann manifold",
					)
				})?;
			optimizer
				.optimize_grassmann_impl(
					py,
					&*cost_function,
					&*grassmann,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
		ManifoldKind::SPD(spd) => {
			let arr = initial_point
				.downcast_bound::<PyArray2<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 2D numpy array for SPD manifold",
					)
				})?;
			optimizer
				.optimize_spd_impl(
					py,
					&*cost_function,
					&*spd,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
		ManifoldKind::Hyperbolic(hyperbolic) => {
			let arr = initial_point
				.downcast_bound::<PyArray1<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 1D numpy array for Hyperbolic manifold",
					)
				})?;
			optimizer
				.optimize_hyperbolic_impl(
					py,
					&*cost_function,
					&*hyperbolic,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
		ManifoldKind::Oblique(oblique) => {
			let arr = initial_point
				.downcast_bound::<PyArray2<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 2D numpy array for Oblique manifold",
					)
				})?;
			optimizer
				.optimize_oblique_impl(
					py,
					&*cost_function,
					&*oblique,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
		ManifoldKind::PSDCone(psd_cone) => {
			let arr = initial_point
				.downcast_bound::<PyArray2<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 2D numpy array for PSDCone manifold",
					)
				})?;
			optimizer
				.optimize_psd_cone_impl(
					py,
					&*cost_function,
					&*psd_cone,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
		ManifoldKind::Euclidean(euclidean) => {
			let arr = initial_point
				.downcast_bound::<PyArray1<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 1D numpy array for Euclidean manifold",
					)
				})?;
			optimizer
				.optimize_euclidean_impl(
					py,
					&*cost_function,
					&*euclidean,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
		ManifoldKind::FixedRank(fixed_rank) => {
			let arr = initial_point
				.downcast_bound::<PyArray2<f64>>(py)
				.map_err(|_| {
					PyErr::new::<pyo3::exceptions::PyTypeError, _>(
						"initial_point must be a 2D numpy array for FixedRank manifold",
					)
				})?;
			optimizer
				.optimize_fixed_rank_impl(
					py,
					&*cost_function,
					&*fixed_rank,
					arr.readonly(),
					max_iterations,
					gradient_tolerance,
					function_tolerance,
					point_tolerance,
				)
				.map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
		}
	}
}

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
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
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
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
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
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
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
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
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
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
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
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
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
	//     function_tolerance: Option<f64>,
	//     point_tolerance: Option<f64>,
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
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
	) -> PyResult<PyOptimizationResult>;

	/// Optimize on a Euclidean manifold
	fn optimize_euclidean_impl(
		&mut self,
		py: Python<'_>,
		cost_function: &PyCostFunction,
		euclidean: &crate::py_manifolds::euclidean::PyEuclidean,
		initial_point: PyReadonlyArray1<'_, f64>,
		max_iterations: usize,
		gradient_tolerance: Option<f64>,
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
	) -> PyResult<PyOptimizationResult>;

	/// Optimize on a FixedRank manifold
	fn optimize_fixed_rank_impl(
		&mut self,
		py: Python<'_>,
		cost_function: &PyCostFunction,
		fixed_rank: &crate::py_manifolds::fixed_rank::PyFixedRank,
		initial_point: PyReadonlyArray2<'_, f64>,
		max_iterations: usize,
		gradient_tolerance: Option<f64>,
		function_tolerance: Option<f64>,
		point_tolerance: Option<f64>,
	) -> PyResult<PyOptimizationResult>;

	/// Try to run optimization with a native Rust cost function.
	/// Returns Ok(Some(result)) if native, Ok(None) to fall back to Python callbacks.
	#[allow(clippy::too_many_arguments, dead_code)]
	fn try_native_optimize(
		&mut self,
		py: Python<'_>,
		cost_function: &Py<PyAny>,
		manifold: &Py<PyAny>,
		initial_point: &Py<PyAny>,
		max_iterations: usize,
		gradient_tolerance: Option<f64>,
	) -> PyResult<Option<Py<PyAny>>>;
}

/// Macro to generate optimize methods for all manifolds
/// This macro should be invoked inside a #[pymethods] impl block
#[macro_export]
macro_rules! impl_optimizer_methods {
    () => {
            /// Unified optimize method accepting both native Rust and Python cost functions.
            ///
            /// Native cost functions (RayleighQuotient, TraceMinimization, etc.) run
            /// entirely in Rust with zero Python callback overhead. Python cost functions
            /// (created via create_cost_function) are still supported via callbacks.
            #[pyo3(signature = (cost_function, manifold, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize(
                &mut self,
                py: Python<'_>,
                cost_function: Py<PyAny>,
                manifold: Py<PyAny>,
                initial_point: Py<PyAny>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                use crate::py_optimizers::generic::optimize_dispatcher;

                // Try native cost functions first (pure Rust, no GIL overhead)
                if let Some(result) = self.try_native_optimize(
                    py,
                    &cost_function,
                    &manifold,
                    &initial_point,
                    max_iterations,
                    gradient_tolerance,
                )? {
                    return Ok(result);
                }

                // Fall back to Python callback cost function
                let py_cf = cost_function.extract::<PyRef<'_, PyCostFunction>>(py)?;
                optimize_dispatcher(
                    self,
                    py,
                    py_cf,
                    manifold,
                    initial_point,
                    max_iterations,
                    gradient_tolerance,
                    function_tolerance,
                    point_tolerance,
                    None, None, None,
                )
            }
            /// Optimize on a Sphere manifold
            #[pyo3(signature = (cost_function, sphere, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_sphere(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                sphere: PyRef<'_, PySphere>,
                initial_point: PyReadonlyArray1<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_sphere_impl(
                    py, &*cost_function, &*sphere, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }

            /// Optimize on a Stiefel manifold
            #[pyo3(signature = (cost_function, stiefel, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_stiefel(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                stiefel: PyRef<'_, PyStiefel>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_stiefel_impl(
                    py, &*cost_function, &*stiefel, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }
            /// Optimize on a Grassmann manifold
            #[pyo3(signature = (cost_function, grassmann, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_grassmann(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                grassmann: PyRef<'_, PyGrassmann>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_grassmann_impl(
                    py, &*cost_function, &*grassmann, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }

            /// Optimize on an SPD manifold
            #[pyo3(signature = (cost_function, spd, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_spd(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                spd: PyRef<'_, PySPD>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_spd_impl(
                    py, &*cost_function, &*spd, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }

            /// Optimize on a Hyperbolic manifold
            #[pyo3(signature = (cost_function, hyperbolic, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_hyperbolic(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                hyperbolic: PyRef<'_, PyHyperbolic>,
                initial_point: PyReadonlyArray1<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_hyperbolic_impl(
                    py, &*cost_function, &*hyperbolic, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }

            /// Optimize on an Oblique manifold
            #[pyo3(signature = (cost_function, oblique, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_oblique(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                oblique: PyRef<'_, PyOblique>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_oblique_impl(
                    py, &*cost_function, &*oblique, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }

            /// Optimize on a FixedRank manifold
            #[pyo3(signature = (cost_function, fixed_rank, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_fixed_rank(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                fixed_rank: PyRef<'_, crate::py_manifolds::fixed_rank::PyFixedRank>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_fixed_rank_impl(
                    py, &*cost_function, &*fixed_rank, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }

            /// Optimize on a PSDCone manifold
            #[pyo3(signature = (cost_function, psd_cone, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_psd_cone(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                psd_cone: PyRef<'_, PyPSDCone>,
                initial_point: PyReadonlyArray2<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_psd_cone_impl(
                    py, &*cost_function, &*psd_cone, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }

            /// Optimize on a Euclidean manifold
            #[pyo3(signature = (cost_function, euclidean, initial_point, max_iterations, gradient_tolerance=None, function_tolerance=None, point_tolerance=None))]
            pub fn optimize_euclidean(
                &mut self,
                py: Python<'_>,
                cost_function: PyRef<'_, PyCostFunction>,
                euclidean: PyRef<'_, crate::py_manifolds::euclidean::PyEuclidean>,
                initial_point: PyReadonlyArray1<'_, f64>,
                max_iterations: usize,
                gradient_tolerance: Option<f64>,
                function_tolerance: Option<f64>,
                point_tolerance: Option<f64>,
            ) -> PyResult<Py<PyAny>> {
                self.optimize_euclidean_impl(
                    py, &*cost_function, &*euclidean, initial_point,
                    max_iterations, gradient_tolerance, function_tolerance, point_tolerance
                ).map(|r| { let bound = r.into_pyobject(py).unwrap(); bound.into_any().unbind() })
            }
    };
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
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionSphere;

				let x0 = numpy_to_vec(initial_point)?;
				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionSphere::new(cost_function);

				let result = py
					.detach(|| optimizer.optimize(&cost_fn, &sphere.inner, &x0, &criterion))
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					Ok(vec_to_numpy(py, point)?.into())
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
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionStiefel;

				let x0 = numpy_to_mat(initial_point)?;
				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionStiefel::new(cost_function);
				// Note: CachedDynamicCostFunction only works with DVector types, not DMatrix

				let result = py
					.detach(|| optimizer.optimize(&cost_fn, &stiefel.inner, &x0, &criterion))
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					Ok(mat_to_numpy(py, point)?.into())
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
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionMatrix;

				let x0 = numpy_to_mat(initial_point)?;
				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionMatrix::new(cost_function);

				let result = py
					.detach(|| {
						optimizer.optimize(&cost_fn, &grassmann.inner, &x0, &criterion)
					})
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					Ok(mat_to_numpy(py, point)?.into())
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
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionMatrix;

				let x0 = numpy_to_mat(initial_point)?;
				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionMatrix::new(cost_function);

				let result = py
					.detach(|| optimizer.optimize(&cost_fn, &spd.inner, &x0, &criterion))
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					Ok(mat_to_numpy(py, point)?.into())
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
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionVector;

				let x0 = numpy_to_vec(initial_point)?;
				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionVector::new(cost_function);

				let result = py
					.detach(|| {
						optimizer.optimize(&cost_fn, &hyperbolic.inner, &x0, &criterion)
					})
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					Ok(vec_to_numpy(py, point)?.into())
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
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionMatrix;

				let x0 = numpy_to_mat(initial_point)?;
				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionMatrix::new(cost_function);

				let result = py
					.detach(|| optimizer.optimize(&cost_fn, &oblique.inner, &x0, &criterion))
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					Ok(mat_to_numpy(py, point)?.into())
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
			//     function_tolerance: Option<f64>,
			//     point_tolerance: Option<f64>,
			// ) -> PyResult<PyOptimizationResult> {
			//     use crate::py_cost::PyCostFunctionMatrix;
			//
			//     let x0 = numpy_to_mat(initial_point)?;
			//     let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
			//     if let Some(tol) = gradient_tolerance {
			//         criterion = criterion.with_gradient_tolerance(tol);
			//     }
			//     if let Some(tol) = function_tolerance {
			//         criterion = criterion.with_function_tolerance(tol);
			//     }
			//     if let Some(tol) = point_tolerance {
			//         criterion = criterion.with_point_tolerance(tol);
			//     }
			//
			//     let config: $config_type = $create_config(self);
			//     let mut optimizer = <$rust_optimizer>::new(config);
			//     let cost_fn = PyCostFunctionMatrix::new(cost_function);
			//
			//     let result = py.detach(|| {
			//         optimizer.optimize(&cost_fn, &fixed_rank.inner, &x0, &criterion)
			//     }).map_err(to_py_err)?;
			//
			//     PyOptimizationResult::from_rust_result(py, result, |point| {
			//         Ok(mat_to_numpy(py, point)?.into())
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
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionPSDCone;

				let x0_mat = numpy_to_mat(initial_point)?;
				// Convert matrix to vector representation
				let mut x0: riemannopt_core::linalg::Vec<f64> = riemannopt_core::linalg::VectorOps::zeros(psd_cone.n * (psd_cone.n + 1) / 2);
				psd_cone.inner.matrix_to_vector::<f64>(&x0_mat, &mut x0);

				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionPSDCone::new(cost_function, psd_cone.n);

				let result = py
					.detach(|| {
						optimizer.optimize(&cost_fn, &psd_cone.inner, &x0, &criterion)
					})
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					// Convert vector back to matrix
					let mut point_mat: riemannopt_core::linalg::Mat<f64> = riemannopt_core::linalg::MatrixOps::zeros(psd_cone.n, psd_cone.n);
					psd_cone.inner.vector_to_matrix::<f64>(point, &mut point_mat);
					Ok(mat_to_numpy(py, &point_mat)?.into())
				})
			}

			fn optimize_euclidean_impl(
				&mut self,
				py: Python<'_>,
				cost_function: &PyCostFunction,
				euclidean: &crate::py_manifolds::euclidean::PyEuclidean,
				initial_point: PyReadonlyArray1<'_, f64>,
				max_iterations: usize,
				gradient_tolerance: Option<f64>,
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionVector;

				let x0 = numpy_to_vec(initial_point)?;
				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionVector::new(cost_function);

				let result = py
					.detach(|| {
						optimizer.optimize(&cost_fn, &euclidean.inner, &x0, &criterion)
					})
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					Ok(vec_to_numpy(py, point)?.into())
				})
			}

			fn optimize_fixed_rank_impl(
				&mut self,
				py: Python<'_>,
				cost_function: &PyCostFunction,
				fixed_rank: &crate::py_manifolds::fixed_rank::PyFixedRank,
				initial_point: PyReadonlyArray2<'_, f64>,
				max_iterations: usize,
				gradient_tolerance: Option<f64>,
				function_tolerance: Option<f64>,
				point_tolerance: Option<f64>,
			) -> PyResult<PyOptimizationResult> {
				use crate::py_cost::PyCostFunctionFixedRank;
				use riemannopt_manifolds::fixed_rank::FixedRankPoint;

				let x0_mat = numpy_to_mat(initial_point)?;
				let x0 = FixedRankPoint::from_matrix(&x0_mat, fixed_rank.inner.rank())
					.map_err(to_py_err)?;

				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}
				if let Some(tol) = function_tolerance {
					criterion = criterion.with_function_tolerance(tol);
				}
				if let Some(tol) = point_tolerance {
					criterion = criterion.with_point_tolerance(tol);
				}

				let config: $config_type = $create_config(self);
				let mut optimizer = <$rust_optimizer>::new(config);
				let cost_fn = PyCostFunctionFixedRank::new(cost_function);

				let result = py
					.detach(|| {
						optimizer.optimize(&cost_fn, &fixed_rank.inner, &x0, &criterion)
					})
					.map_err(to_py_err)?;

				PyOptimizationResult::from_rust_result(py, result, |point| {
					let mat = point.to_matrix();
					Ok(mat_to_numpy(py, &mat)?.into())
				})
			}

			#[allow(clippy::too_many_arguments)]
			fn try_native_optimize(
				&mut self,
				py: Python<'_>,
				cost_function: &Py<PyAny>,
				manifold: &Py<PyAny>,
				initial_point: &Py<PyAny>,
				max_iterations: usize,
				gradient_tolerance: Option<f64>,
			) -> PyResult<Option<Py<PyAny>>> {
				use crate::native_cost::*;
				use crate::py_optimizers::generic::ManifoldKind;
				use numpy::{PyArray1, PyArray2, PyArrayMethods};

				let kind = match ManifoldKind::extract(py, manifold) {
					Ok(k) => k,
					Err(_) => return Ok(None),
				};

				let mut criterion = StoppingCriterion::new().with_max_iterations(max_iterations);
				if let Some(tol) = gradient_tolerance {
					criterion = criterion.with_gradient_tolerance(tol);
				}

				// Helper macro to reduce repetition
				macro_rules! run_native_vec {
					($cost_fn:expr, $manifold_inner:expr, $x0:expr) => {{
						let config: $config_type = $create_config(self);
						let mut optimizer = <$rust_optimizer>::new(config);
						let cost_fn = $cost_fn;
						let x0 = $x0;
						let manifold_ref = $manifold_inner;
						let result = py
							.detach(|| {
								optimizer.optimize(&cost_fn, manifold_ref, &x0, &criterion)
							})
							.map_err(to_py_err)?;
						let py_result =
							PyOptimizationResult::from_rust_result(py, result, |point| {
								Ok(vec_to_numpy(py, point)?.into())
							})?;
						return Ok(Some(Py::new(py, py_result)?.into_any()));
					}};
				}

				macro_rules! run_native_mat {
					($cost_fn:expr, $manifold_inner:expr, $x0:expr) => {{
						let config: $config_type = $create_config(self);
						let mut optimizer = <$rust_optimizer>::new(config);
						let cost_fn = $cost_fn;
						let x0 = $x0;
						let manifold_ref = $manifold_inner;
						let result = py
							.detach(|| {
								optimizer.optimize(&cost_fn, manifold_ref, &x0, &criterion)
							})
							.map_err(to_py_err)?;
						let py_result =
							PyOptimizationResult::from_rust_result(py, result, |point| {
								Ok(mat_to_numpy(py, point)?.into())
							})?;
						return Ok(Some(Py::new(py, py_result)?.into_any()));
					}};
				}

				// ── RayleighQuotient + Sphere ──
				if let Ok(cf) = cost_function.extract::<PyRef<PyRayleighQuotient>>(py) {
					if let ManifoldKind::Sphere(sphere) = kind {
						let arr =
							initial_point
								.downcast_bound::<PyArray1<f64>>(py)
								.map_err(|_| {
									PyErr::new::<pyo3::exceptions::PyTypeError, _>(
										"initial_point must be a 1D array",
									)
								})?;
						let x0 = numpy_to_vec(arr.readonly())?;
						run_native_vec!(cf.clone(), &sphere.inner, x0);
					}
				}

				// ── Quadratic + Euclidean ──
				if let Ok(cf) = cost_function.extract::<PyRef<PyQuadratic>>(py) {
					if let ManifoldKind::Euclidean(euclidean) = kind {
						let arr =
							initial_point
								.downcast_bound::<PyArray1<f64>>(py)
								.map_err(|_| {
									PyErr::new::<pyo3::exceptions::PyTypeError, _>(
										"initial_point must be a 1D array",
									)
								})?;
						let x0 = numpy_to_vec(arr.readonly())?;
						run_native_vec!(cf.clone(), &euclidean.inner, x0);
					}
				}

				// ── TraceMinimization + Grassmann/Stiefel ──
				if let Ok(cf) = cost_function.extract::<PyRef<PyTraceMinimization>>(py) {
					let arr = initial_point
						.downcast_bound::<PyArray2<f64>>(py)
						.map_err(|_| {
							PyErr::new::<pyo3::exceptions::PyTypeError, _>(
								"initial_point must be a 2D array",
							)
						})?;
					let x0 = numpy_to_mat(arr.readonly())?;
					match kind {
						ManifoldKind::Grassmann(gr) => {
							run_native_mat!(cf.clone(), &gr.inner, x0);
						}
						ManifoldKind::Stiefel(st) => {
							run_native_mat!(cf.clone(), &st.inner, x0);
						}
						_ => {}
					}
				}

				// ── Brockett + Stiefel ──
				if let Ok(cf) = cost_function.extract::<PyRef<PyBrockett>>(py) {
					if let ManifoldKind::Stiefel(st) = kind {
						let arr =
							initial_point
								.downcast_bound::<PyArray2<f64>>(py)
								.map_err(|_| {
									PyErr::new::<pyo3::exceptions::PyTypeError, _>(
										"initial_point must be a 2D array",
									)
								})?;
						let x0 = numpy_to_mat(arr.readonly())?;
						run_native_mat!(cf.clone(), &st.inner, x0);
					}
				}

				// ── LogDetDivergence + SPD ──
				if let Ok(cf) = cost_function.extract::<PyRef<PyLogDetDivergence>>(py) {
					if let ManifoldKind::SPD(spd) = kind {
						let arr =
							initial_point
								.downcast_bound::<PyArray2<f64>>(py)
								.map_err(|_| {
									PyErr::new::<pyo3::exceptions::PyTypeError, _>(
										"initial_point must be a 2D array",
									)
								})?;
						let x0 = numpy_to_mat(arr.readonly())?;
						run_native_mat!(cf.clone(), &spd.inner, x0);
					}
				}

				// Not a native cost function
				Ok(None)
			}
		}
	};
}
