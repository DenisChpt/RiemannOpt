//! SIMD-optimized tangent space operations.

use crate::{
    compute::cpu::{SimdBackend, get_dispatcher},
    error::Result,
    types::{Scalar, DVector, DMatrix},
};
use num_traits::Float;

/// SIMD-accelerated Gram-Schmidt orthogonalization.
///
/// Orthogonalizes a set of vectors using the modified Gram-Schmidt algorithm
/// with SIMD acceleration for better numerical stability and performance.
pub fn gram_schmidt_simd<T>(vectors: &mut [DVector<T>]) -> Result<()>
where
    T: Scalar + 'static,
{
    if vectors.is_empty() {
        return Ok(());
    }
    
    let dispatcher = get_dispatcher::<T>();
    
    for i in 0..vectors.len() {
        // Normalize vector i
        let norm_i = dispatcher.normalize(&mut vectors[i]);
        
        if norm_i < T::epsilon() {
            return Err(crate::error::ManifoldError::numerical_error(
                "Gram-Schmidt: vector has zero norm",
            ));
        }
        
        // Orthogonalize remaining vectors against vector i
        for j in (i + 1)..vectors.len() {
            // Use split_at_mut to get non-overlapping mutable references
            let (left, right) = vectors.split_at_mut(j);
            let vi = &left[i];
            let vj = &mut right[0];
            
            // Compute projection coefficient using SIMD
            let dot = dispatcher.dot_product(vi, vj);
            
            // Subtract projection using SIMD axpy
            dispatcher.axpy(-dot, vi, vj);
        }
    }
    
    Ok(())
}

/// SIMD-accelerated tangent space projection for dynamic vectors.
///
/// Projects a vector onto the tangent space at a point, ensuring it satisfies
/// the manifold's tangent space constraints.
pub fn project_tangent_simd_dvec<T>(
    vector: &mut DVector<T>,
    basis: &[DVector<T>],
) -> Result<()>
where
    T: Scalar + 'static,
{
    if basis.is_empty() {
        return Ok(());
    }
    
    let dispatcher = get_dispatcher::<T>();
    
    // Project out components in the normal space
    for basis_vec in basis {
        // Compute projection coefficient
        let coeff = dispatcher.dot_product(vector, basis_vec);
        
        // Subtract projection
        dispatcher.axpy(-coeff, basis_vec, vector);
    }
    
    Ok(())
}

/// SIMD-accelerated tangent vector normalization for dynamic vectors.
///
/// Normalizes a tangent vector using the appropriate metric.
pub fn normalize_tangent_simd_dvec<T>(
    vector: &mut DVector<T>,
    metric: Option<&DMatrix<T>>,
) -> Result<T>
where
    T: Scalar + 'static,
{
    let dispatcher = get_dispatcher::<T>();
    
    let norm = if let Some(m) = metric {
        // Compute sqrt(v^T * M * v)
        let mut mv = DVector::zeros(vector.len());
        dispatcher.gemv(m, vector, &mut mv, T::one(), T::zero());
        <T as Float>::sqrt(dispatcher.dot_product(vector, &mv))
    } else {
        // Standard Euclidean norm
        dispatcher.norm(vector)
    };
    
    if norm > T::epsilon() {
        dispatcher.scale(vector, T::one() / norm);
    }
    
    Ok(norm)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_gram_schmidt_simd() {
        let mut vectors = vec![
            DVector::from_vec(vec![1.0_f64, 0.0, 0.0]),
            DVector::from_vec(vec![1.0_f64, 1.0, 0.0]),
            DVector::from_vec(vec![1.0_f64, 1.0, 1.0]),
        ];
        
        gram_schmidt_simd(&mut vectors).unwrap();
        
        // Check orthonormality
        for i in 0..vectors.len() {
            // Check normalization
            assert_relative_eq!(vectors[i].norm(), 1.0, epsilon = 1e-10);
            
            // Check orthogonality
            for j in (i + 1)..vectors.len() {
                assert_relative_eq!(vectors[i].dot(&vectors[j]), 0.0, epsilon = 1e-10);
            }
        }
    }
    
    #[test]
    fn test_project_tangent_simd() {
        let n = 50;
        
        // Create an orthonormal basis for the normal space
        let normal = DVector::from_fn(n, |i, _| if i == 0 { 1.0_f32 } else { 0.0 });
        let basis = vec![normal];
        
        // Vector to project
        let mut vector = DVector::from_element(n, 1.0_f32);
        
        project_tangent_simd_dvec(&mut vector, &basis).unwrap();
        
        // Check that projection removed component in normal direction
        assert_relative_eq!(vector.dot(&basis[0]), 0.0, epsilon = 1e-6);
    }
    
    #[test]
    fn test_normalize_tangent_simd() {
        let n = 20;
        let mut vector = DVector::from_element(n, 2.0_f64);
        
        // Test without metric
        let norm = normalize_tangent_simd_dvec(&mut vector, None).unwrap();
        assert_relative_eq!(norm, 2.0 * (n as f64).sqrt(), epsilon = 1e-10);
        assert_relative_eq!(vector.norm(), 1.0, epsilon = 1e-10);
        
        // Test with identity metric
        let mut vector2 = DVector::from_element(n, 2.0_f64);
        let metric = DMatrix::identity(n, n);
        let norm2 = normalize_tangent_simd_dvec(&mut vector2, Some(&metric)).unwrap();
        assert_relative_eq!(norm2, 2.0 * (n as f64).sqrt(), epsilon = 1e-10);
        assert_relative_eq!(vector2.norm(), 1.0, epsilon = 1e-10);
    }
}