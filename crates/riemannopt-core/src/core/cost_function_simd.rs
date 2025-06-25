//! SIMD-optimized cost function operations.

use crate::{
    compute::cpu::{SimdBackend, get_dispatcher, parallel::ParallelConfig},
    error::{Result, ManifoldError},
    manifold::{Point, TangentVector},
    types::Scalar,
    memory::workspace::Workspace,
};
use std::sync::{Arc, Mutex};
use rayon::prelude::*;
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DVector};
use num_traits::Float;

/// SIMD-accelerated gradient computation using finite differences for dynamic vectors (allocating version).
pub fn gradient_fd_simd_dvec_alloc<T, F>(
    cost_fn: &F,
    point: &DVector<T>,
) -> Result<DVector<T>>
where
    T: Scalar + 'static,
    F: Fn(&DVector<T>) -> Result<T>,
{
    let n = point.len();
    let mut gradient = DVector::zeros(n);
    let h = <T as Float>::sqrt(T::epsilon());
    
    // Get the SIMD dispatcher
    let dispatcher = get_dispatcher::<T>();
    
    // Process multiple perturbations in parallel when possible
    let chunk_size = if n >= 4 { 4 } else { 1 };
    
    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n);
        
        // Compute all perturbations in this chunk
        for i in chunk_start..chunk_end {
            let mut e_i = DVector::zeros(n);
            e_i[i] = T::one();
            
            // Use SIMD for vector operations
            let mut point_plus = point.clone();
            dispatcher.axpy(h, &e_i, &mut point_plus);
            
            let mut point_minus = point.clone();
            dispatcher.axpy(-h, &e_i, &mut point_minus);
            
            // Evaluate cost function
            let f_plus = cost_fn(&point_plus)?;
            let f_minus = cost_fn(&point_minus)?;
            
            gradient[i] = (f_plus - f_minus) / (h + h);
        }
    }
    
    Ok(gradient)
}

/// SIMD-accelerated Hessian-vector product computation for dynamic vectors.
pub fn hessian_vector_product_simd_dvec<T, F>(
    gradient_fn: &F,
    point: &DVector<T>,
    vector: &DVector<T>,
) -> Result<DVector<T>>
where
    T: Scalar + 'static,
    F: Fn(&DVector<T>) -> Result<DVector<T>>,
{
    let dispatcher = get_dispatcher::<T>();
    
    // Use SIMD to compute vector norm
    let norm = dispatcher.norm(vector);
    
    if norm < T::epsilon() {
        return Ok(DVector::zeros(point.len()));
    }
    
    let eps = <T as Float>::sqrt(T::epsilon());
    let t = eps / norm;
    
    // Use SIMD for scaling and addition
    let mut perturbed = point.clone();
    dispatcher.axpy(t, vector, &mut perturbed);
    
    let grad1 = gradient_fn(point)?;
    let grad2 = gradient_fn(&perturbed)?;
    
    // Use SIMD for gradient difference
    let mut diff = grad2.clone();
    dispatcher.axpy(-T::one(), &grad1, &mut diff);
    dispatcher.scale(&mut diff, T::one() / t);
    
    Ok(diff)
}

/// Generic wrapper to apply SIMD gradient computation to any dimension.
pub fn gradient_fd_simd<T, D, F>(
    cost_fn: &F,
    point: &Point<T, D>,
) -> Result<TangentVector<T, D>>
where
    T: Scalar + 'static,
    D: Dim,
    DefaultAllocator: Allocator<D>,
    F: Fn(&Point<T, D>) -> Result<T>,
{
    // For now, use the regular finite difference without SIMD optimization
    // for generic dimensions. SIMD optimization is available through
    // gradient_fd_simd_dvec for dynamic vectors.
    let n = point.len();
    let mut gradient = TangentVector::zeros_generic(point.shape_generic().0, nalgebra::U1);
    let h = <T as Float>::sqrt(T::epsilon());
    
    for i in 0..n {
        let mut e_i = TangentVector::zeros_generic(point.shape_generic().0, nalgebra::U1);
        e_i[i] = T::one();
        
        let point_plus = point + &e_i * h;
        let point_minus = point - &e_i * h;
        
        let f_plus = cost_fn(&point_plus)?;
        let f_minus = cost_fn(&point_minus)?;
        
        gradient[i] = (f_plus - f_minus) / (h + h);
    }
    
    Ok(gradient)
}

/// SIMD-accelerated parallel gradient computation using finite differences.
///
/// This combines SIMD operations for vector arithmetic with parallel execution
/// across gradient components for maximum performance.
pub fn gradient_fd_simd_parallel<T, F>(
    cost_fn: &F,
    point: &DVector<T>,
    config: &ParallelConfig,
) -> Result<DVector<T>>
where
    T: Scalar + 'static,
    F: Fn(&DVector<T>) -> Result<T> + Sync,
{
    let n = point.len();
    let h = <T as Float>::sqrt(T::epsilon());
    
    // Check if we should use parallel execution
    if !config.should_parallelize(n) {
        return gradient_fd_simd_dvec_alloc(cost_fn, point);
    }
    
    // Get adaptive strategy for chunk size if not provided
    let strategy = crate::compute::cpu::parallel_strategy::get_adaptive_strategy();
    
    // Get the SIMD dispatcher
    let dispatcher = get_dispatcher::<T>();
    
    // Create thread-safe gradient vector
    let gradient = Arc::new(Mutex::new(DVector::zeros(n)));
    let point_arc = Arc::new(point.clone());
    
    // Determine chunk size for parallel execution
    let chunk_size = config.chunk_size
        .unwrap_or_else(|| strategy.optimal_chunk_size(n));
    
    // Execute in parallel
    let indices: Vec<usize> = (0..n).collect();
    indices
        .par_chunks(chunk_size)
        .try_for_each(|chunk| -> Result<()> {
            // Process this chunk
            let mut local_results = Vec::with_capacity(chunk.len());
            
            for &i in chunk {
                // Pre-allocate buffers outside the inner loop if possible
                // For now, we allocate per iteration but could optimize further
                let mut e_i = DVector::zeros(n);
                e_i[i] = T::one();
                
                // Use SIMD for vector operations
                let mut point_plus = (*point_arc).clone();
                dispatcher.axpy(h, &e_i, &mut point_plus);
                
                let mut point_minus = (*point_arc).clone();
                dispatcher.axpy(-h, &e_i, &mut point_minus);
                
                // Evaluate cost function
                let f_plus = cost_fn(&point_plus)?;
                let f_minus = cost_fn(&point_minus)?;
                
                let grad_i = (f_plus - f_minus) / (h + h);
                local_results.push((i, grad_i));
            }
            
            // Update gradient with local results
            let mut grad_guard = gradient.lock().unwrap();
            for (i, value) in local_results {
                grad_guard[i] = value;
            }
            
            Ok(())
        })?;
    
    // Extract the gradient
    let gradient_vec = Arc::try_unwrap(gradient)
        .map(|mutex| mutex.into_inner().unwrap())
        .unwrap_or_else(|arc| arc.lock().unwrap().clone());
    Ok(gradient_vec)
}

/// SIMD-accelerated gradient computation using finite differences with workspace.
///
/// This version avoids allocations by using pre-allocated buffers from the workspace.
pub fn gradient_fd_simd_dvec<T, F>(
    cost_fn: &F,
    point: &DVector<T>,
    workspace: &mut Workspace<T>,
    gradient: &mut DVector<T>,
) -> Result<()>
where
    T: Scalar + Float + 'static,
    F: Fn(&DVector<T>) -> Result<T>,
{
    let n = point.len();
    let h = <T as Float>::sqrt(T::epsilon());
    let dispatcher = get_dispatcher::<T>();
    
    // Get pre-allocated buffers from workspace
    let (e_i, point_plus, point_minus) = workspace.get_gradient_fd_buffers_mut(n)
        .ok_or_else(|| ManifoldError::invalid_parameter(
            "Workspace missing required gradient buffers".to_string()
        ))?;
        
    // Clear gradient
    gradient.fill(T::zero());
    
    // Process in chunks for better cache utilization
    let chunk_size = 64;
    for chunk_start in (0..n).step_by(chunk_size) {
        let chunk_end = (chunk_start + chunk_size).min(n);
        
        // Compute all perturbations in this chunk
        for i in chunk_start..chunk_end {
            // Set unit vector
            e_i.fill(T::zero());
            e_i[i] = T::one();
            
            // Use SIMD for vector operations
            point_plus.copy_from(point);
            dispatcher.axpy(h, e_i, point_plus);
            
            point_minus.copy_from(point);
            dispatcher.axpy(-h, e_i, point_minus);
            
            // Evaluate cost function
            let f_plus = cost_fn(point_plus)?;
            let f_minus = cost_fn(point_minus)?;
            
            gradient[i] = (f_plus - f_minus) / (h + h);
        }
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{DMatrix, DVector};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_gradient_fd_simd_dvec() -> Result<()> {
        let n = 100;
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let a = DMatrix::<f64>::from_fn(n, n, |_, _| rng.gen::<f64>());
        let a = &a.transpose() * &a; // Make positive definite
        let b = DVector::from_fn(n, |_, _| rng.gen::<f64>());
        
        let x = DVector::from_fn(n, |_, _| rng.gen::<f64>());
        
        // Compare SIMD and analytical gradient
        let cost_fn = |p: &DVector<f64>| -> Result<f64> {
            Ok(0.5 * p.dot(&(&a * p)) - b.dot(p))
        };
        
        let grad_simd = gradient_fd_simd_dvec_alloc(&cost_fn, &x)?;
        let grad_analytical = &a * &x - &b;
        
        // Check that they are close (finite differences have some error)
        for i in 0..n {
            assert_relative_eq!(grad_simd[i], grad_analytical[i], epsilon = 1e-3);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_hessian_vector_product_simd() -> Result<()> {
        let n = 50;
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let a = DMatrix::<f64>::from_fn(n, n, |_, _| rng.gen::<f64>());
        let a = &a.transpose() * &a; // Make positive definite
        let b = DVector::from_fn(n, |_, _| rng.gen::<f64>());
        
        let x = DVector::from_fn(n, |_, _| rng.gen::<f64>());
        let v = DVector::from_fn(n, |_, _| rng.gen::<f64>());
        
        let gradient_fn = |p: &DVector<f64>| -> Result<DVector<f64>> {
            Ok(&a * p - &b)
        };
        
        let hvp_simd = hessian_vector_product_simd_dvec(&gradient_fn, &x, &v)?;
        let hvp_exact = &a * &v;
        
        // Check that they are close
        for i in 0..n {
            assert_relative_eq!(hvp_simd[i], hvp_exact[i], epsilon = 1e-4);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_gradient_fd_simd_parallel() -> Result<()> {
        use crate::compute::cpu::parallel::ParallelConfig;
        
        let n = 300; // Large dimension to test parallel execution
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let a = DMatrix::<f64>::from_fn(n, n, |_, _| rng.gen::<f64>());
        let a = &a.transpose() * &a; // Make positive definite
        let b = DVector::from_fn(n, |_, _| rng.gen::<f64>());
        
        let x = DVector::from_fn(n, |_, _| rng.gen::<f64>());
        
        let cost_fn = |p: &DVector<f64>| -> Result<f64> {
            Ok(0.5 * p.dot(&(&a * p)) - b.dot(p))
        };
        
        // Sequential SIMD gradient
        let grad_simd_seq = gradient_fd_simd_dvec_alloc(&cost_fn, &x)?;
        
        // Parallel SIMD gradient
        let config = ParallelConfig::default();
        let grad_simd_par = gradient_fd_simd_parallel(&cost_fn, &x, &config)?;
        
        // They should be very close (within floating point tolerance)
        for i in 0..n {
            assert_relative_eq!(grad_simd_par[i], grad_simd_seq[i], epsilon = 1e-10);
        }
        
        Ok(())
    }
    
    #[test]
    fn test_gradient_fd_simd_parallel_custom_config() -> Result<()> {
        use crate::compute::cpu::parallel::ParallelConfig;
        
        let n = 150;
        let a = DMatrix::<f64>::identity(n, n) * 3.0;
        let b = DVector::zeros(n);
        
        let x = DVector::from_element(n, 1.0);
        
        let cost_fn = |p: &DVector<f64>| -> Result<f64> {
            Ok(0.5 * p.dot(&(&a * p)) - b.dot(p))
        };
        
        // Custom configuration
        let config = ParallelConfig::new()
            .with_min_dimension(100)
            .with_chunk_size(25);
        
        let grad_par = gradient_fd_simd_parallel(&cost_fn, &x, &config)?;
        let grad_analytical = &a * &x - &b;
        
        // Check accuracy
        for i in 0..n {
            assert_relative_eq!(grad_par[i], grad_analytical[i], epsilon = 1e-4);
        }
        
        Ok(())
    }
}