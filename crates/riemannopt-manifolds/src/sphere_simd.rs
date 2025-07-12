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
//! use nalgebra::DVector;
//!
//! // Create a high-dimensional sphere
//! let sphere = Sphere::<f64>::new(1000)?;
//!
//! // Project a point using SIMD
//! let point = DVector::from_vec(vec![1.0; 1000]);
//! let projected = sphere.project_point_simd_f64(&point);
//! assert!((projected.norm() - 1.0).abs() < 1e-10);
//!
//! // Compute dot product using SIMD
//! let v1 = DVector::from_vec(vec![1.0; 1000]);
//! let v2 = DVector::from_vec(vec![2.0; 1000]);
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
    types::DVector,
};
#[cfg(test)]
use riemannopt_core::core::Manifold;

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
    fn project_point_simd_f64(&self, point: &DVector<f64>) -> DVector<f64>;
    
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
    fn project_point_simd_f32(&self, point: &DVector<f32>) -> DVector<f32>;
    
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
    fn tangent_dot_simd_f64(&self, v1: &DVector<f64>, v2: &DVector<f64>) -> f64;
    
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
    fn tangent_dot_simd_f32(&self, v1: &DVector<f32>, v2: &DVector<f32>) -> f32;
}

impl SphereSimdExt for Sphere {
    fn project_point_simd_f64(&self, point: &DVector<f64>) -> DVector<f64> {
        if point.len() >= 4 {
            // Use SIMD for larger vectors
            let mut result = point.clone();
            let dispatcher = get_dispatcher::<f64>();
            let norm = dispatcher.normalize(&mut result);
            if norm < f64::EPSILON {
                // Handle zero vector
                result = DVector::zeros(self.ambient_dimension());
                result[0] = 1.0;
            }
            result
        } else {
            // Fall back to standard implementation for small vectors
            let norm = point.norm();
            if norm < f64::EPSILON {
                let mut result = DVector::zeros(self.ambient_dimension());
                result[0] = 1.0;
                result
            } else {
                point / norm
            }
        }
    }
    
    fn project_point_simd_f32(&self, point: &DVector<f32>) -> DVector<f32> {
        if point.len() >= 8 {
            // Use SIMD for larger vectors
            let mut result = point.clone();
            let dispatcher = get_dispatcher::<f32>();
            let norm = dispatcher.normalize(&mut result);
            if norm < f32::EPSILON {
                // Handle zero vector
                result = DVector::zeros(self.ambient_dimension());
                result[0] = 1.0;
            }
            result
        } else {
            // Fall back to standard implementation for small vectors
            let norm = point.norm();
            if norm < f32::EPSILON {
                let mut result = DVector::zeros(self.ambient_dimension());
                result[0] = 1.0;
                result
            } else {
                point / norm
            }
        }
    }
    
    fn tangent_dot_simd_f64(&self, v1: &DVector<f64>, v2: &DVector<f64>) -> f64 {
        if v1.len() >= 4 && v1.len() == v2.len() {
            let dispatcher = get_dispatcher::<f64>();
            dispatcher.dot_product(v1, v2)
        } else {
            v1.dot(v2)
        }
    }
    
    fn tangent_dot_simd_f32(&self, v1: &DVector<f32>, v2: &DVector<f32>) -> f32 {
        if v1.len() >= 8 && v1.len() == v2.len() {
            let dispatcher = get_dispatcher::<f32>();
            dispatcher.dot_product(v1, v2)
        } else {
            v1.dot(v2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use riemannopt_core::memory::workspace::Workspace;
    
    #[test]
    fn test_simd_projection_f64() {
        let sphere = Sphere::<f64>::new(100).unwrap();
        let point = DVector::from_vec(vec![1.0; 100]);
        
        let mut proj_standard = DVector::zeros(100);
        let mut workspace = Workspace::<f64>::new();
        <Sphere<f64> as Manifold<f64>>::project_point(&sphere, &point, &mut proj_standard, &mut workspace);
        let proj_simd = sphere.project_point_simd_f64(&point);
        
        assert_relative_eq!(proj_standard, proj_simd, epsilon = 1e-10);
        assert_relative_eq!(proj_simd.norm(), 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_simd_projection_f32() {
        let sphere = Sphere::<f64>::new(100).unwrap();
        let point = DVector::from_vec(vec![1.0f32; 100]);
        
        let proj_simd = sphere.project_point_simd_f32(&point);
        
        // Just verify the SIMD result is normalized
        assert_relative_eq!(proj_simd.norm(), 1.0f32, epsilon = 1e-6);
    }
    
    #[test]
    fn test_simd_tangent_dot() {
        let sphere = Sphere::<f64>::new(100).unwrap();
        let v1 = DVector::from_vec(vec![1.0; 100]);
        let v2 = DVector::from_vec(vec![2.0; 100]);
        
        let dot_standard = v1.dot(&v2);
        let dot_simd = sphere.tangent_dot_simd_f64(&v1, &v2);
        
        assert_relative_eq!(dot_standard, dot_simd, epsilon = 1e-10);
    }
}