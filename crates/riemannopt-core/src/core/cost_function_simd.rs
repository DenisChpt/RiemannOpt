//! SIMD-optimized cost function operations.

use crate::{
    compute::cpu::{SimdBackend, get_dispatcher},
    error::Result,
    manifold::{Point, TangentVector},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DVector};
use num_traits::Float;

/// SIMD-accelerated gradient computation using finite differences for dynamic vectors.
pub fn gradient_fd_simd_dvec<T, F>(
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
        
        let grad_simd = gradient_fd_simd_dvec(&cost_fn, &x)?;
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
}