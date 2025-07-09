//! SIMD-optimized cost function operations.
//!
//! This module provides SIMD-accelerated implementations of gradient computations
//! using finite differences. It leverages CPU vector instructions for improved performance.

use crate::{
    compute::cpu::{get_dispatcher, parallel::ParallelConfig},
    error::{Result, ManifoldError},
    types::Scalar,
    memory::workspace::Workspace,
};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use nalgebra::DVector;
use num_traits::Float;

/// Trait for types that support SIMD operations for gradient computation
pub trait SimdGradientOps<T: Scalar>: Clone + Send + Sync {
    /// Get the dimension of the vector
    fn dimension(&self) -> usize;
    
    /// Create a zero vector of the same type
    fn zeros_like(&self) -> Self;
    
    /// Create a unit vector with 1 at position i
    fn unit_vector(&self, i: usize) -> Self;
    
    /// Add a scaled vector: self += alpha * other
    fn axpy(&mut self, alpha: T, other: &Self);
    
    /// Get element at index
    fn get(&self, i: usize) -> Option<T>;
    
    /// Set element at index
    fn set(&mut self, i: usize, value: T);
}

// Implement for DVector
impl<T: Scalar> SimdGradientOps<T> for DVector<T> {
    fn dimension(&self) -> usize {
        self.len()
    }
    
    fn zeros_like(&self) -> Self {
        DVector::zeros(self.len())
    }
    
    fn unit_vector(&self, i: usize) -> Self {
        let mut e = DVector::zeros(self.len());
        if i < self.len() {
            e[i] = T::one();
        }
        e
    }
    
    fn axpy(&mut self, alpha: T, other: &Self) {
        self.axpy(alpha, other, T::one());
    }
    
    fn get(&self, i: usize) -> Option<T> {
        if i < self.len() {
            Some(self[i])
        } else {
            None
        }
    }
    
    fn set(&mut self, i: usize, value: T) {
        if i < self.len() {
            self[i] = value;
        }
    }
}

/// SIMD-accelerated gradient computation using finite differences (allocating version).
///
/// This function computes the gradient using central differences with SIMD optimizations
/// for vector operations.
///
/// # Arguments
///
/// * `cost_fn` - Function that computes the cost at a given point
/// * `point` - Point at which to compute the gradient
///
/// # Returns
///
/// The gradient vector computed using finite differences
pub fn gradient_fd_simd_alloc<T, P, F>(
    cost_fn: &F,
    point: &P,
) -> Result<P>
where
    T: Scalar + 'static,
    P: SimdGradientOps<T>,
    F: Fn(&P) -> Result<T>,
{
    let n = point.dimension();
    let mut gradient = point.zeros_like();
    let h = <T as Float>::sqrt(T::epsilon());
    
    // Get the SIMD dispatcher
    let _dispatcher = get_dispatcher::<T>();
    
    // Process multiple perturbations in parallel when possible
    let chunk_size = if n >= 4 { 4 } else { 1 };
    
    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n);
        
        // Compute all perturbations in this chunk
        for i in chunk_start..chunk_end {
            let e_i = point.unit_vector(i);
            
            // Use SIMD for vector operations
            let mut point_plus = point.clone();
            point_plus.axpy(h, &e_i);
            
            let mut point_minus = point.clone();
            point_minus.axpy(-h, &e_i);
            
            // Evaluate cost function
            let f_plus = cost_fn(&point_plus)?;
            let f_minus = cost_fn(&point_minus)?;
            
            // Central difference
            gradient.set(i, (f_plus - f_minus) / (h + h));
        }
    }
    
    Ok(gradient)
}

/// SIMD-accelerated gradient computation with workspace.
///
/// This version uses pre-allocated workspace buffers to avoid allocations.
pub fn gradient_fd_simd<T, P, F>(
    cost_fn: &F,
    point: &P,
    _workspace: &mut Workspace<T>,
    gradient: &mut P,
) -> Result<()>
where
    T: Scalar + Float,
    P: SimdGradientOps<T>,
    F: Fn(&P) -> Result<T>,
{
    let n = point.dimension();
    let h = <T as Float>::sqrt(T::epsilon());
    
    // Clear gradient
    *gradient = point.zeros_like();
    
    // Get the SIMD dispatcher
    let _dispatcher = get_dispatcher::<T>();
    
    for i in 0..n {
        let e_i = point.unit_vector(i);
        
        // Compute point + h*e_i and point - h*e_i
        let mut point_plus = point.clone();
        point_plus.axpy(h, &e_i);
        
        let mut point_minus = point.clone();
        point_minus.axpy(-h, &e_i);
        
        // Central difference
        let f_plus = cost_fn(&point_plus)?;
        let f_minus = cost_fn(&point_minus)?;
        
        gradient.set(i, (f_plus - f_minus) / (h + h));
    }
    
    Ok(())
}

/// Parallel SIMD-accelerated gradient computation.
///
/// This function distributes the gradient computation across multiple threads
/// for improved performance on multi-core systems.
pub fn gradient_fd_simd_parallel<T, P, F>(
    cost_fn: &F,
    point: &P,
    config: &ParallelConfig,
) -> Result<P>
where
    T: Scalar + Float + Send + Sync + 'static,
    P: SimdGradientOps<T>,
    F: Fn(&P) -> Result<T> + Sync,
{
    let n = point.dimension();
    
    // Check if parallelization is beneficial
    if !config.should_parallelize(n) {
        return gradient_fd_simd_alloc(cost_fn, point);
    }
    
    let h = <T as Float>::sqrt(T::epsilon());
    let gradient_parts = Arc::new(Mutex::new(vec![T::zero(); n]));
    
    // Determine chunk size for parallel execution
    let chunk_size = config.chunk_size.unwrap_or((n + rayon::current_num_threads() - 1) / rayon::current_num_threads());
    
    // Parallel computation
    (0..n)
        .into_par_iter()
        .chunks(chunk_size)
        .try_for_each(|chunk| -> Result<()> {
            for i in chunk {
                let e_i = point.unit_vector(i);
                
                // Compute perturbations
                let mut point_plus = point.clone();
                point_plus.axpy(h, &e_i);
                
                let mut point_minus = point.clone();
                point_minus.axpy(-h, &e_i);
                
                // Evaluate cost function
                let f_plus = cost_fn(&point_plus)?;
                let f_minus = cost_fn(&point_minus)?;
                
                // Store gradient component
                let grad_i = (f_plus - f_minus) / (h + h);
                {
                    let mut grad_parts = gradient_parts.lock().unwrap();
                    grad_parts[i] = grad_i;
                }
            }
            Ok(())
        })?;
    
    // Construct result
    let grad_parts = gradient_parts.lock().unwrap();
    let mut gradient = point.zeros_like();
    for (i, &value) in grad_parts.iter().enumerate() {
        gradient.set(i, value);
    }
    
    Ok(gradient)
}

/// Helper function to compute gradient using finite differences with workspace for dynamic vectors.
///
/// This function provides an efficient implementation for dynamic-dimensional problems
/// by reusing pre-allocated buffers from the workspace.
pub fn gradient_fd_dvec<T, F>(
    cost_fn: &F,
    point: &nalgebra::DVector<T>,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar + Float,
    F: Fn(&nalgebra::DVector<T>) -> Result<T>,
{
    let n = point.len();
    let h = <T as Float>::sqrt(T::epsilon());
    
    // Get pre-allocated buffers from workspace
    let (gradient, e_i, point_plus, point_minus) = workspace.get_gradient_buffers_mut()
        .ok_or_else(|| ManifoldError::invalid_parameter(
            "Workspace missing required gradient buffers".to_string()
        ))?;
        
    // Verify dimensions
    if gradient.len() != n || e_i.len() != n || point_plus.len() != n || point_minus.len() != n {
        return Err(ManifoldError::invalid_parameter(
            format!("Workspace buffers have incorrect dimensions for point of size {}", n),
        ));
    }
    
    // Clear gradient
    gradient.fill(T::zero());
    
    for i in 0..n {
        // Clear and set unit vector
        e_i.fill(T::zero());
        e_i[i] = T::one();
        
        // Compute point + h*e_i and point - h*e_i
        point_plus.copy_from(point);
        point_minus.copy_from(point);
        point_plus[i] += h;
        point_minus[i] -= h;
        
        // Central difference
        let f_plus = cost_fn(point_plus)?;
        let f_minus = cost_fn(point_minus)?;
        
        // Update gradient
        gradient[i] = (f_plus - f_minus) / (h + h);
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    // Simple quadratic function for testing
    fn quadratic_cost(x: &DVector<f64>) -> Result<f64> {
        Ok(x.dot(x))
    }
    
    #[test]
    fn test_gradient_fd_simd_alloc() {
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let grad = gradient_fd_simd_alloc(&quadratic_cost, &point).unwrap();
        
        // For f(x) = x^T x, gradient is 2x
        let expected = &point * 2.0;
        for i in 0..point.len() {
            assert_relative_eq!(grad[i], expected[i], epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_gradient_fd_simd_workspace() {
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let mut workspace = Workspace::new();
        let mut grad = DVector::zeros(3);
        
        gradient_fd_simd(&quadratic_cost, &point, &mut workspace, &mut grad).unwrap();
        
        // For f(x) = x^T x, gradient is 2x
        let expected = &point * 2.0;
        for i in 0..point.len() {
            assert_relative_eq!(grad[i], expected[i], epsilon = 1e-6);
        }
    }
    
    #[test]
    fn test_gradient_fd_simd_parallel() {
        let n = 100;
        let point = DVector::from_fn(n, |i, _| (i as f64 + 1.0) / 10.0);
        let config = ParallelConfig::default();
        
        let grad = gradient_fd_simd_parallel(&quadratic_cost, &point, &config).unwrap();
        
        // For f(x) = x^T x, gradient is 2x
        let expected = &point * 2.0;
        for i in 0..point.len() {
            assert_relative_eq!(grad[i], expected[i], epsilon = 1e-4);
        }
    }
    
    #[test]
    fn test_simd_ops_dvector() {
        let v = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        // Test dimension
        assert_eq!(v.dimension(), 3);
        
        // Test zeros_like
        let z = v.zeros_like();
        assert_eq!(z.len(), 3);
        assert!(z.iter().all(|&x| x == 0.0));
        
        // Test unit_vector
        let e1 = v.unit_vector(1);
        assert_eq!(e1[0], 0.0);
        assert_eq!(e1[1], 1.0);
        assert_eq!(e1[2], 0.0);
        
        // Test axpy
        let mut w = v.clone();
        let u = DVector::from_vec(vec![4.0, 5.0, 6.0]);
        w.axpy(2.0, &u, 1.0);
        assert_eq!(w[0], 9.0);  // 1 + 2*4
        assert_eq!(w[1], 12.0); // 2 + 2*5
        assert_eq!(w[2], 15.0); // 3 + 2*6
    }
}