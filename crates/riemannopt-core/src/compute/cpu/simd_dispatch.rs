//! Runtime SIMD dispatch based on CPU capabilities.
//!
//! This module provides a backend-agnostic interface for SIMD operations
//! that automatically selects the best implementation based on detected
//! CPU features.

use crate::config::features::simd_config;
use crate::types::Scalar;
use nalgebra::{DMatrix, DVector};
use std::marker::PhantomData;

/// Trait for SIMD backend implementations.
pub trait SimdBackend<T: Scalar>: Send + Sync {
    /// Compute dot product between two vectors.
    fn dot_product(&self, a: &DVector<T>, b: &DVector<T>) -> T;
    
    /// Compute vector norm.
    fn norm(&self, v: &DVector<T>) -> T;
    
    /// Compute squared norm (more efficient than norm for some operations).
    fn norm_squared(&self, v: &DVector<T>) -> T;
    
    /// Element-wise vector addition: result = a + b.
    fn add(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>);
    
    /// Scaled vector addition: result = a + alpha * b.
    fn axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>);
    
    /// Scale vector in-place: v *= scalar.
    fn scale(&self, v: &mut DVector<T>, scalar: T);
    
    /// Normalize vector in-place, returns original norm.
    fn normalize(&self, v: &mut DVector<T>) -> T;
    
    /// Matrix-vector multiplication: y = alpha * A * x + beta * y.
    fn gemv(
        &self,
        a: &DMatrix<T>,
        x: &DVector<T>,
        y: &mut DVector<T>,
        alpha: T,
        beta: T,
    );
    
    /// Frobenius norm of a matrix.
    fn frobenius_norm(&self, a: &DMatrix<T>) -> T;
    
    /// Element-wise vector multiplication: result[i] = a[i] * b[i].
    fn hadamard_product(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>);
    
    /// Check if this backend supports the given vector size efficiently.
    fn is_efficient_for_size(&self, size: usize) -> bool;
    
    /// Compute the maximum absolute difference between two vectors.
    /// Returns the L∞ norm of (a - b).
    fn max_abs_diff(&self, a: &DVector<T>, b: &DVector<T>) -> T;
}

/// Scalar (non-SIMD) backend implementation.
pub struct ScalarBackend<T> {
    _phantom: PhantomData<T>,
}

impl<T> ScalarBackend<T> {
    pub fn new() -> Self {
        Self {
            _phantom: PhantomData,
        }
    }
}

impl<T> Default for ScalarBackend<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> SimdBackend<T> for ScalarBackend<T> {
    fn dot_product(&self, a: &DVector<T>, b: &DVector<T>) -> T {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        let mut sum = T::zero();
        for i in 0..a.len() {
            sum = sum + a[i] * b[i];
        }
        sum
    }
    
    fn norm(&self, v: &DVector<T>) -> T {
        <T as num_traits::Float>::sqrt(self.norm_squared(v))
    }
    
    fn norm_squared(&self, v: &DVector<T>) -> T {
        let mut sum = T::zero();
        for i in 0..v.len() {
            sum = sum + v[i] * v[i];
        }
        sum
    }
    
    fn add(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result must have same length");
        for i in 0..a.len() {
            result[i] = a[i] + b[i];
        }
    }
    
    fn axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>) {
        assert_eq!(x.len(), y.len(), "Vectors must have same length");
        for i in 0..x.len() {
            y[i] = y[i] + alpha * x[i];
        }
    }
    
    fn scale(&self, v: &mut DVector<T>, scalar: T) {
        for i in 0..v.len() {
            v[i] = v[i] * scalar;
        }
    }
    
    fn normalize(&self, v: &mut DVector<T>) -> T {
        let norm = self.norm(v);
        if norm > T::zero() {
            let inv_norm = T::one() / norm;
            self.scale(v, inv_norm);
        }
        norm
    }
    
    fn gemv(
        &self,
        a: &DMatrix<T>,
        x: &DVector<T>,
        y: &mut DVector<T>,
        alpha: T,
        beta: T,
    ) {
        assert_eq!(a.ncols(), x.len(), "Dimension mismatch");
        assert_eq!(a.nrows(), y.len(), "Dimension mismatch");
        
        // Scale y by beta
        if beta != T::one() {
            self.scale(y, beta);
        }
        
        // Compute alpha * A * x
        for i in 0..a.nrows() {
            let mut sum = T::zero();
            for j in 0..a.ncols() {
                sum = sum + a[(i, j)] * x[j];
            }
            y[i] = y[i] + alpha * sum;
        }
    }
    
    fn frobenius_norm(&self, a: &DMatrix<T>) -> T {
        let mut sum = T::zero();
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                let val = a[(i, j)];
                sum = sum + val * val;
            }
        }
        <T as num_traits::Float>::sqrt(sum)
    }
    
    fn hadamard_product(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        assert_eq!(a.len(), result.len(), "Result must have same length");
        for i in 0..a.len() {
            result[i] = a[i] * b[i];
        }
    }
    
    fn is_efficient_for_size(&self, _size: usize) -> bool {
        true // Scalar backend works for any size
    }
    
    fn max_abs_diff(&self, a: &DVector<T>, b: &DVector<T>) -> T {
        assert_eq!(a.len(), b.len(), "Vectors must have same length");
        
        let mut max_diff = T::zero();
        for i in 0..a.len() {
            let diff = a[i] - b[i];
            let abs_diff = <T as num_traits::Signed>::abs(&diff);
            if abs_diff > max_diff {
                max_diff = abs_diff;
            }
        }
        max_diff
    }
}

// Use the wide backend from the parent module
use super::wide_backend::WideBackend;

/// SIMD dispatcher that selects the best backend at runtime.
pub struct SimdDispatcher<T> {
    backend: Box<dyn SimdBackend<T>>,
}

impl<T: Scalar + 'static> SimdDispatcher<T> {
    /// Create a dispatcher with a specific backend.
    pub fn with_backend(backend: Box<dyn SimdBackend<T>>) -> Self {
        Self { backend }
    }
    
    /// Get the underlying backend.
    pub fn backend(&self) -> &dyn SimdBackend<T> {
        &*self.backend
    }
}

impl SimdDispatcher<f32> {
    /// Create a new dispatcher with automatic backend selection for f32.
    pub fn new() -> Self {
        let config = simd_config();
        
        // Select the best backend based on CPU features and configuration
        let backend: Box<dyn SimdBackend<f32>> = if !config.enabled {
            Box::new(ScalarBackend::new())
        } else {
            // Use the Wide backend which provides portable SIMD
            Box::new(WideBackend::new())
        };
        
        Self { backend }
    }
}

impl Default for SimdDispatcher<f32> {
    fn default() -> Self {
        Self::new()
    }
}

impl SimdDispatcher<f64> {
    /// Create a new dispatcher with automatic backend selection for f64.
    pub fn new() -> Self {
        let config = simd_config();
        
        // Select the best backend based on CPU features and configuration
        let backend: Box<dyn SimdBackend<f64>> = if !config.enabled {
            Box::new(ScalarBackend::new())
        } else {
            // Use the Wide backend which provides portable SIMD
            Box::new(WideBackend::new())
        };
        
        Self { backend }
    }
}

impl Default for SimdDispatcher<f64> {
    fn default() -> Self {
        Self::new()
    }
}

// Implement SimdBackend for SimdDispatcher by delegating to the selected backend
impl<T: Scalar> SimdBackend<T> for SimdDispatcher<T> {
    #[inline]
    fn dot_product(&self, a: &DVector<T>, b: &DVector<T>) -> T {
        self.backend.dot_product(a, b)
    }
    
    #[inline]
    fn norm(&self, v: &DVector<T>) -> T {
        self.backend.norm(v)
    }
    
    #[inline]
    fn norm_squared(&self, v: &DVector<T>) -> T {
        self.backend.norm_squared(v)
    }
    
    #[inline]
    fn add(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
        self.backend.add(a, b, result)
    }
    
    #[inline]
    fn axpy(&self, alpha: T, x: &DVector<T>, y: &mut DVector<T>) {
        self.backend.axpy(alpha, x, y)
    }
    
    #[inline]
    fn scale(&self, v: &mut DVector<T>, scalar: T) {
        self.backend.scale(v, scalar)
    }
    
    #[inline]
    fn normalize(&self, v: &mut DVector<T>) -> T {
        self.backend.normalize(v)
    }
    
    #[inline]
    fn gemv(
        &self,
        a: &DMatrix<T>,
        x: &DVector<T>,
        y: &mut DVector<T>,
        alpha: T,
        beta: T,
    ) {
        self.backend.gemv(a, x, y, alpha, beta)
    }
    
    #[inline]
    fn frobenius_norm(&self, a: &DMatrix<T>) -> T {
        self.backend.frobenius_norm(a)
    }
    
    #[inline]
    fn hadamard_product(&self, a: &DVector<T>, b: &DVector<T>, result: &mut DVector<T>) {
        self.backend.hadamard_product(a, b, result)
    }
    
    #[inline]
    fn is_efficient_for_size(&self, size: usize) -> bool {
        self.backend.is_efficient_for_size(size)
    }
    
    #[inline]
    fn max_abs_diff(&self, a: &DVector<T>, b: &DVector<T>) -> T {
        self.backend.max_abs_diff(a, b)
    }
}

/// Global SIMD dispatcher for f32.
static F32_DISPATCHER: once_cell::sync::Lazy<SimdDispatcher<f32>> = 
    once_cell::sync::Lazy::new(|| SimdDispatcher::<f32>::new());

/// Global SIMD dispatcher for f64.
static F64_DISPATCHER: once_cell::sync::Lazy<SimdDispatcher<f64>> = 
    once_cell::sync::Lazy::new(|| SimdDispatcher::<f64>::new());

/// Get the global SIMD dispatcher for a given type.
pub fn get_dispatcher<T: Scalar + 'static>() -> &'static SimdDispatcher<T> {
    // This is a bit of a hack, but it works for our use case
    use std::any::TypeId;
    
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        unsafe { &*(&*F32_DISPATCHER as *const SimdDispatcher<f32> as *const SimdDispatcher<T>) }
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        unsafe { &*(&*F64_DISPATCHER as *const SimdDispatcher<f64> as *const SimdDispatcher<T>) }
    } else {
        panic!("SIMD dispatcher only supports f32 and f64");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_scalar_backend() {
        let backend = ScalarBackend::<f64>::new();
        
        let a = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let b = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        
        // Test dot product
        let dot = backend.dot_product(&a, &b);
        assert_relative_eq!(dot, 32.0, epsilon = 1e-10);
        
        // Test norm
        let norm = backend.norm(&a);
        assert_relative_eq!(norm, (14.0_f64).sqrt(), epsilon = 1e-10);
        
        // Test add
        let mut result = DVector::zeros(3);
        backend.add(&a, &b, &mut result);
        assert_eq!(result[0], 5.0);
        assert_eq!(result[1], 7.0);
        assert_eq!(result[2], 9.0);
    }
    
    #[test]
    fn test_dispatcher() {
        let dispatcher = SimdDispatcher::<f32>::new();
        
        let a = DVector::from_vec(vec![1.0; 100]);
        let b = DVector::from_vec(vec![2.0; 100]);
        
        let dot = dispatcher.dot_product(&a, &b);
        assert_relative_eq!(dot, 200.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_global_dispatcher() {
        let dispatcher = get_dispatcher::<f64>();
        
        let v = DVector::from_vec(vec![3.0, 4.0]);
        let norm = dispatcher.norm(&v);
        assert_relative_eq!(norm, 5.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_max_abs_diff() {
        let backend = ScalarBackend::<f64>::new();
        
        let a = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = DVector::from_vec(vec![1.1, 1.9, 3.3, 3.7]);
        
        // Max abs diff should be |3.3 - 3.0| = 0.3
        let max_diff = backend.max_abs_diff(&a, &b);
        assert_relative_eq!(max_diff, 0.3, epsilon = 1e-10);
        
        // Test with dispatcher
        let dispatcher = get_dispatcher::<f64>();
        let max_diff2 = dispatcher.max_abs_diff(&a, &b);
        assert_relative_eq!(max_diff2, 0.3, epsilon = 1e-10);
        
        // Test with f32 and SIMD backend
        let a_f32 = DVector::from_vec(vec![1.0_f32; 100]);
        let b_f32 = DVector::from_vec(vec![1.5_f32; 100]);
        
        let dispatcher_f32 = get_dispatcher::<f32>();
        let max_diff_f32 = dispatcher_f32.max_abs_diff(&a_f32, &b_f32);
        assert_relative_eq!(max_diff_f32, 0.5, epsilon = 1e-6);
    }
}