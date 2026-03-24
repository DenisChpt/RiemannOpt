//! # SIMD-Optimized Sphere Operations
//!
//! This module provides SIMD-accelerated implementations for sphere operations
//! when working with f32 or f64 types. These optimizations leverage CPU vector
//! instructions (SSE, AVX, AVX-512) for significant performance improvements on
//! large-dimensional spheres.
//!
//! ## Performance Characteristics
//!
//! - **f64**: SIMD enabled for vectors with ≥ 4 elements (4-way parallelism with AVX)
//! - **f32**: SIMD enabled for vectors with ≥ 8 elements (8-way parallelism with AVX)
//! - **Speedup**: Typically 2-8x faster for large dimensions (n > 100)
//!
//! ## Supported Operations
//!
//! 1. **Point projection**: Normalizing vectors to unit length
//! 2. **Dot products**: Computing inner products of tangent vectors
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use riemannopt_manifolds::{Sphere, sphere_simd::SphereSimdExt};
//! use riemannopt_core::linalg::{self, VectorOps};
//!
//! // Create a high-dimensional sphere
//! let sphere = Sphere::<f64>::new(1000)?;
//!
//! // Project a point using SIMD
//! let point = linalg::Vec::<f64>::from_slice(&vec![1.0; 1000]);
//! let projected = sphere.project_point_simd_f64(&point);
//! assert!((VectorOps::norm(&projected) - 1.0).abs() < 1e-10);
//!
//! // Compute dot product using SIMD
//! let v1 = linalg::Vec::<f64>::from_slice(&vec![1.0; 1000]);
//! let v2 = linalg::Vec::<f64>::from_slice(&vec![2.0; 1000]);
//! let dot = sphere.tangent_dot_simd_f64(&v1, &v2);
//! # Ok::<(), riemannopt_core::error::ManifoldError>(())
//! ```
//!
//! ## Implementation Notes
//!
//! - Automatically falls back to scalar operations for small vectors
//! - Handles edge cases (zero vectors) correctly
//! - Maintains numerical accuracy comparable to scalar implementation
//! - Thread-safe and can be used in parallel algorithms

use crate::Sphere;
use riemannopt_core::{
	compute::{get_dispatcher, SimdBackend},
	linalg::{self, VectorOps},
};
/// Extension trait for SIMD-accelerated sphere operations.
///
/// This trait provides optimized implementations of common sphere operations
/// using SIMD instructions when available. The implementations automatically
/// select the best available instruction set at runtime.
pub trait SphereSimdExt {
	/// Projects a point onto the sphere using SIMD operations.
	///
	/// # Arguments
	///
	/// * `point` - Vector to project onto the unit sphere
	///
	/// # Returns
	///
	/// Normalized vector with unit length
	///
	/// # Performance
	///
	/// Uses SIMD for vectors with ≥ 4 elements. Falls back to scalar
	/// operations for smaller vectors.
	fn project_point_simd_f64(&self, point: &linalg::Vec<f64>) -> linalg::Vec<f64>;

	/// Projects a point onto the sphere using SIMD operations (f32 version).
	///
	/// # Arguments
	///
	/// * `point` - Vector to project onto the unit sphere
	///
	/// # Returns
	///
	/// Normalized vector with unit length
	///
	/// # Performance
	///
	/// Uses SIMD for vectors with ≥ 8 elements. Falls back to scalar
	/// operations for smaller vectors.
	fn project_point_simd_f32(&self, point: &linalg::Vec<f32>) -> linalg::Vec<f32>;

	/// Computes dot product of tangent vectors using SIMD.
	///
	/// # Arguments
	///
	/// * `v1` - First tangent vector
	/// * `v2` - Second tangent vector
	///
	/// # Returns
	///
	/// The Euclidean inner product v1^T v2
	///
	/// # Performance
	///
	/// Uses SIMD for vectors with ≥ 4 elements.
	fn tangent_dot_simd_f64(&self, v1: &linalg::Vec<f64>, v2: &linalg::Vec<f64>) -> f64;

	/// Computes dot product of tangent vectors using SIMD (f32 version).
	///
	/// # Arguments
	///
	/// * `v1` - First tangent vector
	/// * `v2` - Second tangent vector
	///
	/// # Returns
	///
	/// The Euclidean inner product v1^T v2
	///
	/// # Performance
	///
	/// Uses SIMD for vectors with ≥ 8 elements.
	fn tangent_dot_simd_f32(&self, v1: &linalg::Vec<f32>, v2: &linalg::Vec<f32>) -> f32;
}

/// Helper: normalize a vector in-place via the SIMD dispatcher, returning the original norm.
///
/// Bridges `linalg::Vec<T>` to the SIMD dispatcher which expects `DVector<T>`.
/// We construct a temporary `DVector` from the slice, normalize it, and copy back.
fn simd_normalize_inplace<T>(v: &mut linalg::Vec<T>) -> T
where
	T: riemannopt_core::types::Scalar + riemannopt_core::compute::ScalarDispatch,
	linalg::DefaultBackend: riemannopt_core::linalg::LinAlgBackend<T>,
{
	let mut dv = nalgebra::DVector::from_column_slice(v.as_slice());
	let dispatcher = get_dispatcher::<T>();
	let norm = dispatcher.normalize(&mut dv);
	v.as_mut_slice().copy_from_slice(dv.as_slice());
	norm
}

/// Helper: compute dot product via the SIMD dispatcher.
fn simd_dot<T>(a: &linalg::Vec<T>, b: &linalg::Vec<T>) -> T
where
	T: riemannopt_core::types::Scalar + riemannopt_core::compute::ScalarDispatch,
	linalg::DefaultBackend: riemannopt_core::linalg::LinAlgBackend<T>,
{
	let da = nalgebra::DVector::from_column_slice(a.as_slice());
	let db = nalgebra::DVector::from_column_slice(b.as_slice());
	let dispatcher = get_dispatcher::<T>();
	dispatcher.dot_product(&da, &db)
}

impl SphereSimdExt for Sphere {
	fn project_point_simd_f64(&self, point: &linalg::Vec<f64>) -> linalg::Vec<f64> {
		if point.len() >= 4 {
			// Use SIMD for larger vectors
			let mut result = point.clone();
			let norm: f64 = simd_normalize_inplace(&mut result);
			if norm < f64::EPSILON {
				// Handle zero vector
				let mut result = linalg::Vec::<f64>::zeros(self.ambient_dimension());
				*result.get_mut(0) = 1.0;
				return result;
			}
			result
		} else {
			// Fall back to standard implementation for small vectors
			let norm = VectorOps::norm(point);
			if norm < f64::EPSILON {
				let mut result = linalg::Vec::<f64>::zeros(self.ambient_dimension());
				*result.get_mut(0) = 1.0;
				result
			} else {
				let mut result = point.clone();
				result.div_scalar_mut(norm);
				result
			}
		}
	}

	fn project_point_simd_f32(&self, point: &linalg::Vec<f32>) -> linalg::Vec<f32> {
		if point.len() >= 8 {
			// Use SIMD for larger vectors
			let mut result = point.clone();
			let norm: f32 = simd_normalize_inplace(&mut result);
			if norm < f32::EPSILON {
				// Handle zero vector
				let mut result = linalg::Vec::<f32>::zeros(self.ambient_dimension());
				*result.get_mut(0) = 1.0;
				return result;
			}
			result
		} else {
			// Fall back to standard implementation for small vectors
			let norm = VectorOps::norm(point);
			if norm < f32::EPSILON {
				let mut result = linalg::Vec::<f32>::zeros(self.ambient_dimension());
				*result.get_mut(0) = 1.0;
				result
			} else {
				let mut result = point.clone();
				result.div_scalar_mut(norm);
				result
			}
		}
	}

	fn tangent_dot_simd_f64(&self, v1: &linalg::Vec<f64>, v2: &linalg::Vec<f64>) -> f64 {
		if v1.len() >= 4 && v1.len() == v2.len() {
			simd_dot(v1, v2)
		} else {
			v1.dot(v2)
		}
	}

	fn tangent_dot_simd_f32(&self, v1: &linalg::Vec<f32>, v2: &linalg::Vec<f32>) -> f32 {
		if v1.len() >= 8 && v1.len() == v2.len() {
			simd_dot(v1, v2)
		} else {
			v1.dot(v2)
		}
	}
}
