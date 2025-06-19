//! Optimized implementations for small fixed dimensions.
//!
//! This module provides hand-optimized implementations for common small
//! dimensions (2D, 3D, 4D) that avoid dynamic dispatch and use unrolled loops.

use crate::types::Scalar;
use nalgebra::{Vector2, Vector3};
use num_traits::Float;

/// Trait for small dimension operations.
pub trait SmallDimOps<T: Scalar> {
    /// Compute dot product for small vectors.
    fn dot_small(&self, a: &[T], b: &[T]) -> T;
    
    /// Compute norm for small vectors.
    fn norm_small(&self, v: &[T]) -> T;
    
    /// Normalize small vectors in-place.
    fn normalize_small(&self, v: &mut [T]);
    
    /// Matrix-vector product for small dimensions.
    fn matvec_small(&self, a: &[T], x: &[T], y: &mut [T]);
}

/// Optimized operations for 2D vectors.
pub struct Ops2D;

impl<T: Scalar> SmallDimOps<T> for Ops2D {
    #[inline(always)]
    fn dot_small(&self, a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), 2);
        debug_assert_eq!(b.len(), 2);
        
        // Unrolled dot product
        a[0] * b[0] + a[1] * b[1]
    }
    
    #[inline(always)]
    fn norm_small(&self, v: &[T]) -> T {
        debug_assert_eq!(v.len(), 2);
        
        // Unrolled norm computation
        Float::sqrt(v[0] * v[0] + v[1] * v[1])
    }
    
    #[inline(always)]
    fn normalize_small(&self, v: &mut [T]) {
        debug_assert_eq!(v.len(), 2);
        
        let norm = self.norm_small(v);
        if norm > T::zero() {
            let inv_norm = T::one() / norm;
            v[0] = v[0] * inv_norm;
            v[1] = v[1] * inv_norm;
        }
    }
    
    #[inline(always)]
    fn matvec_small(&self, a: &[T], x: &[T], y: &mut [T]) {
        debug_assert_eq!(a.len(), 4); // 2x2 matrix
        debug_assert_eq!(x.len(), 2);
        debug_assert_eq!(y.len(), 2);
        
        // Unrolled matrix-vector product
        y[0] = a[0] * x[0] + a[1] * x[1];
        y[1] = a[2] * x[0] + a[3] * x[1];
    }
}

/// Optimized operations for 3D vectors.
pub struct Ops3D;

impl<T: Scalar> SmallDimOps<T> for Ops3D {
    #[inline(always)]
    fn dot_small(&self, a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), 3);
        debug_assert_eq!(b.len(), 3);
        
        // Unrolled dot product
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }
    
    #[inline(always)]
    fn norm_small(&self, v: &[T]) -> T {
        debug_assert_eq!(v.len(), 3);
        
        // Unrolled norm computation
        Float::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
    }
    
    #[inline(always)]
    fn normalize_small(&self, v: &mut [T]) {
        debug_assert_eq!(v.len(), 3);
        
        let norm = self.norm_small(v);
        if norm > T::zero() {
            let inv_norm = T::one() / norm;
            v[0] = v[0] * inv_norm;
            v[1] = v[1] * inv_norm;
            v[2] = v[2] * inv_norm;
        }
    }
    
    #[inline(always)]
    fn matvec_small(&self, a: &[T], x: &[T], y: &mut [T]) {
        debug_assert_eq!(a.len(), 9); // 3x3 matrix
        debug_assert_eq!(x.len(), 3);
        debug_assert_eq!(y.len(), 3);
        
        // Unrolled matrix-vector product
        y[0] = a[0] * x[0] + a[1] * x[1] + a[2] * x[2];
        y[1] = a[3] * x[0] + a[4] * x[1] + a[5] * x[2];
        y[2] = a[6] * x[0] + a[7] * x[1] + a[8] * x[2];
    }
}

/// Optimized operations for 4D vectors.
pub struct Ops4D;

impl<T: Scalar> SmallDimOps<T> for Ops4D {
    #[inline(always)]
    fn dot_small(&self, a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), 4);
        debug_assert_eq!(b.len(), 4);
        
        // Unrolled dot product
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    }
    
    #[inline(always)]
    fn norm_small(&self, v: &[T]) -> T {
        debug_assert_eq!(v.len(), 4);
        
        // Unrolled norm computation
        Float::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3])
    }
    
    #[inline(always)]
    fn normalize_small(&self, v: &mut [T]) {
        debug_assert_eq!(v.len(), 4);
        
        let norm = self.norm_small(v);
        if norm > T::zero() {
            let inv_norm = T::one() / norm;
            v[0] = v[0] * inv_norm;
            v[1] = v[1] * inv_norm;
            v[2] = v[2] * inv_norm;
            v[3] = v[3] * inv_norm;
        }
    }
    
    #[inline(always)]
    fn matvec_small(&self, a: &[T], x: &[T], y: &mut [T]) {
        debug_assert_eq!(a.len(), 16); // 4x4 matrix
        debug_assert_eq!(x.len(), 4);
        debug_assert_eq!(y.len(), 4);
        
        // Unrolled matrix-vector product
        y[0] = a[0] * x[0] + a[1] * x[1] + a[2] * x[2] + a[3] * x[3];
        y[1] = a[4] * x[0] + a[5] * x[1] + a[6] * x[2] + a[7] * x[3];
        y[2] = a[8] * x[0] + a[9] * x[1] + a[10] * x[2] + a[11] * x[3];
        y[3] = a[12] * x[0] + a[13] * x[1] + a[14] * x[2] + a[15] * x[3];
    }
}

/// Specialized retraction for 2D sphere.
#[inline(always)]
pub fn retract_sphere_2d<T: Scalar>(
    point: &Vector2<T>,
    tangent: &Vector2<T>,
    t: T,
) -> Vector2<T> {
    // Compute point + t * tangent
    let moved = Vector2::new(
        point[0] + t * tangent[0],
        point[1] + t * tangent[1],
    );
    
    // Normalize
    let norm = Float::sqrt(moved[0] * moved[0] + moved[1] * moved[1]);
    if norm > T::zero() {
        let inv_norm = T::one() / norm;
        Vector2::new(moved[0] * inv_norm, moved[1] * inv_norm)
    } else {
        *point
    }
}

/// Specialized retraction for 3D sphere.
#[inline(always)]
pub fn retract_sphere_3d<T: Scalar>(
    point: &Vector3<T>,
    tangent: &Vector3<T>,
    t: T,
) -> Vector3<T> {
    // Compute point + t * tangent
    let moved = Vector3::new(
        point[0] + t * tangent[0],
        point[1] + t * tangent[1],
        point[2] + t * tangent[2],
    );
    
    // Normalize
    let norm = Float::sqrt(moved[0] * moved[0] + moved[1] * moved[1] + moved[2] * moved[2]);
    if norm > T::zero() {
        let inv_norm = T::one() / norm;
        Vector3::new(
            moved[0] * inv_norm,
            moved[1] * inv_norm,
            moved[2] * inv_norm,
        )
    } else {
        *point
    }
}

/// Specialized projection to tangent space for 2D sphere.
#[inline(always)]
pub fn project_tangent_sphere_2d<T: Scalar>(
    point: &Vector2<T>,
    vector: &mut Vector2<T>,
) {
    // Project: v - <v, p> * p
    let dot = vector[0] * point[0] + vector[1] * point[1];
    vector[0] = vector[0] - dot * point[0];
    vector[1] = vector[1] - dot * point[1];
}

/// Specialized projection to tangent space for 3D sphere.
#[inline(always)]
pub fn project_tangent_sphere_3d<T: Scalar>(
    point: &Vector3<T>,
    vector: &mut Vector3<T>,
) {
    // Project: v - <v, p> * p
    let dot = vector[0] * point[0] + vector[1] * point[1] + vector[2] * point[2];
    vector[0] = vector[0] - dot * point[0];
    vector[1] = vector[1] - dot * point[1];
    vector[2] = vector[2] - dot * point[2];
}

/// Fast path selector for small dimensions.
pub struct SmallDimSelector;

impl SmallDimSelector {
    /// Check if a dimension qualifies for small dimension optimization.
    #[inline(always)]
    pub fn is_small_dim(dim: usize) -> bool {
        dim <= 4
    }
    
    /// Select the appropriate small dimension operations.
    #[inline(always)]
    pub fn select_ops<T: Scalar>(dim: usize) -> Option<Box<dyn SmallDimOps<T>>> {
        match dim {
            2 => Some(Box::new(Ops2D)),
            3 => Some(Box::new(Ops3D)),
            4 => Some(Box::new(Ops4D)),
            _ => None,
        }
    }
    
    /// Compute dot product with fast path for small dimensions.
    #[inline]
    pub fn dot_with_fast_path<T: Scalar>(a: &[T], b: &[T]) -> T {
        debug_assert_eq!(a.len(), b.len());
        
        match a.len() {
            2 => Ops2D.dot_small(a, b),
            3 => Ops3D.dot_small(a, b),
            4 => Ops4D.dot_small(a, b),
            _ => {
                // Fallback to generic implementation
                let mut sum = T::zero();
                for i in 0..a.len() {
                    sum = sum + a[i] * b[i];
                }
                sum
            }
        }
    }
    
    /// Compute norm with fast path for small dimensions.
    #[inline]
    pub fn norm_with_fast_path<T: Scalar>(v: &[T]) -> T {
        match v.len() {
            2 => Ops2D.norm_small(v),
            3 => Ops3D.norm_small(v),
            4 => Ops4D.norm_small(v),
            _ => {
                // Fallback to generic implementation
                let mut sum = T::zero();
                for i in 0..v.len() {
                    sum = sum + v[i] * v[i];
                }
                Float::sqrt(sum)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Vector2, Vector3};
    
    #[test]
    fn test_ops_2d() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        
        let ops = Ops2D;
        assert_eq!(ops.dot_small(&a, &b), 11.0);
        assert!((ops.norm_small(&a) - 5.0_f64.sqrt()).abs() < 1e-10);
        
        let mut v = vec![3.0, 4.0];
        ops.normalize_small(&mut v);
        assert!((v[0] - 0.6).abs() < 1e-10);
        assert!((v[1] - 0.8).abs() < 1e-10);
    }
    
    #[test]
    fn test_ops_3d() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        
        let ops = Ops3D;
        assert_eq!(ops.dot_small(&a, &b), 32.0);
        assert!((ops.norm_small(&a) - 14.0_f64.sqrt()).abs() < 1e-10);
        
        let mut v = vec![1.0, 0.0, 0.0];
        ops.normalize_small(&mut v);
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 0.0);
        assert_eq!(v[2], 0.0);
    }
    
    #[test]
    fn test_retract_sphere_2d() {
        let point = Vector2::<f64>::new(1.0, 0.0);
        let tangent = Vector2::<f64>::new(0.0, 1.0);
        
        let result = retract_sphere_2d(&point, &tangent, 0.5);
        assert!((result.norm() - 1.0).abs() < 1e-10);
    }
    
    #[test]
    fn test_project_tangent_sphere_3d() {
        let point = Vector3::new(1.0, 0.0, 0.0);
        let mut vector = Vector3::new(1.0, 1.0, 1.0);
        
        project_tangent_sphere_3d(&point, &mut vector);
        
        // Should be orthogonal to point
        assert!((vector.dot(&point)).abs() < 1e-10);
        // Should preserve y and z components
        assert_eq!(vector[1], 1.0);
        assert_eq!(vector[2], 1.0);
    }
    
    #[test]
    fn test_small_dim_selector() {
        assert!(SmallDimSelector::is_small_dim(2));
        assert!(SmallDimSelector::is_small_dim(3));
        assert!(SmallDimSelector::is_small_dim(4));
        assert!(!SmallDimSelector::is_small_dim(5));
        
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert_eq!(SmallDimSelector::dot_with_fast_path(&a, &b), 32.0);
        
        let v = vec![3.0, 4.0];
        assert!((SmallDimSelector::norm_with_fast_path(&v) - 5.0).abs() < 1e-10);
    }
}