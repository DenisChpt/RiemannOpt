//! SIMD backend using the `wide` crate for portable SIMD operations.

use super::SimdBackend;
use crate::compute::cpu::simd::{SimdOps, SimdVector, SimdVectorOps, SimdMatrixOps};
use nalgebra::{DMatrix, DVector};
use std::marker::PhantomData;

/// Backend that uses the `wide` crate for SIMD operations.
pub struct WideBackend<T> {
    _phantom: PhantomData<T>,
}

impl<T> WideBackend<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for WideBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: SimdOps> SimdBackend<T> for WideBackend<T> {
    fn dot_product(&self, a: &DVector<T>, b: &DVector<T>) -> T {
        SimdVectorOps::dot_product(a.as_view(), b.as_view())
    }
    
    fn norm(&self, v: &DVector<T>) -> T {
        SimdVectorOps::norm(v.as_view())
    }
    
    fn norm_squared(&self, v: &DVector<T>) -> T {
        // More efficient than computing norm and squaring
        let n = v.len();
        let simd_width = T::SIMD_WIDTH;
        let simd_end = n - (n % simd_width);
        
        let v_slice = v.as_slice();
        
        // SIMD part
        let mut sum = T::SimdVector::splat(T::zero());
        for i in (0..simd_end).step_by(simd_width) {
            let vv = T::SimdVector::from_slice(&v_slice[i..]);
            sum = vv.mul_add(vv, sum);
        }
        
        let mut result = sum.horizontal_sum();
        
        // Scalar remainder
        for i in simd_end..n {
            result = result + v_slice[i] * v_slice[i];
        }
        
        result
    }
    
    fn add(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
        SimdVectorOps::add(a, b, result)
    }
    
    fn axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>) {
        assert_eq!(x.len(), y.len(), "Vectors must have same length");
        
        let n = x.len();
        let simd_width = T::SIMD_WIDTH;
        let simd_end = n - (n % simd_width);
        
        let x_slice = x.as_slice();
        let y_slice = y.as_mut_slice();
        let alpha_vec = T::SimdVector::splat(alpha);
        
        // SIMD part
        for i in (0..simd_end).step_by(simd_width) {
            let vx = T::SimdVector::from_slice(&x_slice[i..]);
            let vy = T::SimdVector::from_slice(&y_slice[i..]);
            let result = vx.mul_add(alpha_vec, vy);
            result.store_to_slice(&mut y_slice[i..]);
        }
        
        // Scalar remainder
        for i in simd_end..n {
            y_slice[i] = y_slice[i] + alpha * x_slice[i];
        }
    }
    
    fn scale(&self, v: &mut DVector<T>, scalar: T) {
        SimdVectorOps::scale(v, scalar)
    }
    
    fn normalize(&self, v: &mut DVector<T>) -> T {
        SimdVectorOps::normalize(v)
    }
    
    fn gemv(
        &self,
        a: &DMatrix<T>,
        x: &DVector<T>,
        y: &mut DVector<T>,
        alpha: T,
        beta: T,
    ) {
        SimdMatrixOps::gemv(a, x, y, alpha, beta)
    }
    
    fn frobenius_norm(&self, a: &DMatrix<T>) -> T {
        SimdMatrixOps::frobenius_norm(a)
    }
    
    fn hadamard_product(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result must have same length");
        
        let n = a.len();
        let simd_width = T::SIMD_WIDTH;
        let simd_end = n - (n % simd_width);
        
        let a_slice = a.as_slice();
        let b_slice = b.as_slice();
        let result_slice = result.as_mut_slice();
        
        // SIMD part
        for i in (0..simd_end).step_by(simd_width) {
            let va = T::SimdVector::from_slice(&a_slice[i..]);
            let vb = T::SimdVector::from_slice(&b_slice[i..]);
            let prod = va.mul(vb);
            prod.store_to_slice(&mut result_slice[i..]);
        }
        
        // Scalar remainder
        for i in simd_end..n {
            result_slice[i] = a_slice[i] * b_slice[i];
        }
    }
    
    fn is_efficient_for_size(&self, size: usize) -> bool {
        // Wide backend is efficient for vectors larger than SIMD width
        size >= T::SIMD_WIDTH * 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_wide_backend_operations() {
        let backend = WideBackend::<f32>::new();
        
        let a = DVector::from_vec(vec![1.0; 100]);
        let b = DVector::from_vec(vec![2.0; 100]);
        
        // Test dot product
        let dot = backend.dot_product(&a, &b);
        assert_relative_eq!(dot, 200.0, epsilon = 1e-6);
        
        // Test norm squared
        let norm_sq = backend.norm_squared(&a);
        assert_relative_eq!(norm_sq, 100.0, epsilon = 1e-6);
        
        // Test axpy
        let mut y = DVector::from_vec(vec![1.0; 100]);
        backend.axpy(0.5, &a, &mut y);
        for i in 0..100 {
            assert_relative_eq!(y[i], 1.5, epsilon = 1e-6);
        }
        
        // Test hadamard product
        let mut result = DVector::zeros(100);
        backend.hadamard_product(&a, &b, &mut result);
        for i in 0..100 {
            assert_relative_eq!(result[i], 2.0, epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_efficiency_check() {
        let backend = WideBackend::<f64>::new();
        
        // Should be inefficient for small sizes
        assert!(!backend.is_efficient_for_size(3));
        
        // Should be efficient for larger sizes
        assert!(backend.is_efficient_for_size(100));
    }
}