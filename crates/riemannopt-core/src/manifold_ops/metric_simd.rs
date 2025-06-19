//! SIMD-optimized Riemannian metric operations.

use crate::{
    compute::cpu::{SimdBackend, get_dispatcher},
    error::Result,
    types::{Scalar, DVector, DMatrix},
};
use num_traits::Float;

/// SIMD-accelerated inner product computation using a metric tensor for dynamic vectors.
pub fn metric_inner_product_simd_dvec<T>(
    metric_matrix: &DMatrix<T>,
    u: &DVector<T>,
    v: &DVector<T>,
) -> Result<T>
where
    T: Scalar + 'static,
{
    let dispatcher = get_dispatcher::<T>();
    
    // Compute metric_matrix * v using SIMD
    let mut mv = DVector::zeros(v.len());
    dispatcher.gemv(metric_matrix, v, &mut mv, T::one(), T::zero());
    
    // Compute u^T * (metric_matrix * v) using SIMD
    Ok(dispatcher.dot_product(u, &mv))
}

/// SIMD-accelerated norm computation using a metric tensor for dynamic vectors.
pub fn metric_norm_simd_dvec<T>(
    metric_matrix: &DMatrix<T>,
    v: &DVector<T>,
) -> Result<T>
where
    T: Scalar + 'static,
{
    let inner = metric_inner_product_simd_dvec(metric_matrix, v, v)?;
    Ok(<T as Float>::sqrt(inner))
}

/// SIMD-accelerated weighted metric computation.
pub fn weighted_metric_simd<T>(
    weights: &DVector<T>,
    u: &DVector<T>,
    v: &DVector<T>,
) -> T
where
    T: Scalar + 'static,
{
    let dispatcher = get_dispatcher::<T>();
    
    // Compute element-wise: weights[i] * u[i] * v[i]
    let mut weighted_u = u.clone();
    let mut result = DVector::zeros(u.len());
    
    // First multiply u by weights
    dispatcher.hadamard_product(weights, u, &mut weighted_u);
    
    // Then multiply by v
    dispatcher.hadamard_product(&weighted_u, v, &mut result);
    
    // Sum all elements
    let mut sum = T::zero();
    for i in 0..result.len() {
        sum = sum + result[i];
    }
    sum
}

/// SIMD-accelerated Christoffel symbol computation for dynamic dimensions.
///
/// Computes Γ^k_{ij} = 0.5 * g^{kl} * (∂_i g_{jl} + ∂_j g_{il} - ∂_l g_{ij})
pub fn christoffel_symbols_simd_dvec<T, F>(
    point: &DVector<T>,
    metric_fn: &F,
) -> Result<Vec<DMatrix<T>>>
where
    T: Scalar + 'static,
    F: Fn(&DVector<T>) -> Result<DMatrix<T>>,
{
    let n = point.len();
    let h = <T as Float>::sqrt(T::epsilon());
    let dispatcher = get_dispatcher::<T>();
    
    // Get metric at current point
    let g = metric_fn(point)?;
    
    // Compute metric inverse using SIMD-accelerated operations
    let g_inv = g.clone().try_inverse()
        .ok_or_else(|| crate::error::ManifoldError::numerical_error("Metric tensor is singular"))?;
    
    // Compute metric derivatives
    let mut dg = vec![DMatrix::zeros(n, n); n];
    
    for k in 0..n {
        let mut ek = DVector::zeros(n);
        ek[k] = T::one();
        
        // Forward and backward points
        let mut point_plus = point.clone();
        let mut point_minus = point.clone();
        
        dispatcher.axpy(h, &ek, &mut point_plus);
        dispatcher.axpy(-h, &ek, &mut point_minus);
        
        let g_plus = metric_fn(&point_plus)?;
        let g_minus = metric_fn(&point_minus)?;
        
        // Use SIMD for matrix subtraction and scaling
        for i in 0..n {
            for j in 0..n {
                dg[k][(i, j)] = (g_plus[(i, j)] - g_minus[(i, j)]) / (T::one() + T::one()) / h;
            }
        }
    }
    
    // Compute Christoffel symbols
    let mut gamma = vec![DMatrix::zeros(n, n); n];
    
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..n {
                    sum = sum + g_inv[(k, l)] * 
                        (dg[i][(j, l)] + dg[j][(i, l)] - dg[l][(i, j)]);
                }
                gamma[k][(i, j)] = sum * <T as crate::types::Scalar>::from_f64(0.5);
            }
        }
    }
    
    Ok(gamma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use nalgebra::DVector;
    
    #[test]
    fn test_metric_inner_product_simd() {
        let n = 50;
        let metric = DMatrix::<f64>::identity(n, n) * 2.0;
        let u = DVector::from_element(n, 1.0);
        let v = DVector::from_element(n, 1.0);
        
        let result = metric_inner_product_simd_dvec(&metric, &u, &v).unwrap();
        let expected = 2.0 * n as f64;
        
        assert_relative_eq!(result, expected, epsilon = 1e-10);
    }
    
    #[test]
    fn test_weighted_metric_simd() {
        let n = 100;
        let weights = DVector::from_element(n, 2.0_f32);
        let u = DVector::from_element(n, 1.0_f32);
        let v = DVector::from_element(n, 3.0_f32);
        
        let result = weighted_metric_simd(&weights, &u, &v);
        let expected = 2.0 * 1.0 * 3.0 * n as f32;
        
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }
}