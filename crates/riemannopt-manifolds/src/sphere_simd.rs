//! SIMD-optimized sphere operations.
//!
//! This module provides SIMD-accelerated implementations for sphere operations
//! when working with f32 or f64 types.

use crate::Sphere;
use riemannopt_core::{
    compute::{get_dispatcher, SimdBackend},
    types::DVector,
};
#[cfg(test)]
use riemannopt_core::core::Manifold;

/// Extension trait for SIMD-accelerated sphere operations.
pub trait SphereSimdExt {
    /// Project a point onto the sphere using SIMD operations.
    fn project_point_simd_f64(&self, point: &DVector<f64>) -> DVector<f64>;
    
    /// Project a point onto the sphere using SIMD operations.
    fn project_point_simd_f32(&self, point: &DVector<f32>) -> DVector<f32>;
    
    /// Compute dot product of tangent vectors using SIMD.
    fn tangent_dot_simd_f64(&self, v1: &DVector<f64>, v2: &DVector<f64>) -> f64;
    
    /// Compute dot product of tangent vectors using SIMD.
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