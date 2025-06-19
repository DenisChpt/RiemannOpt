//! Metric operations using pre-allocated workspace.

use crate::{
    error::Result,
    manifold::{Point, TangentVector},
    memory::{Workspace, BufferId},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DVector};
use num_traits::Float;

/// Compute the inner product between two tangent vectors using workspace.
pub fn inner_product_with_workspace<T, D>(
    _point: &Point<T, D>,
    u: &TangentVector<T, D>,
    v: &TangentVector<T, D>,
    _workspace: &mut Workspace<T>,
) -> Result<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    // For Euclidean metric, this is just the dot product
    // More complex manifolds might need workspace for computation
    Ok(u.dot(v))
}

/// Compute the norm of a tangent vector using workspace.
pub fn norm_with_workspace<T, D>(
    point: &Point<T, D>,
    v: &TangentVector<T, D>,
    workspace: &mut Workspace<T>,
) -> Result<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let inner = inner_product_with_workspace(point, v, v, workspace)?;
    Ok(<T as Float>::sqrt(inner))
}

/// Normalize a tangent vector in-place using workspace.
pub fn normalize_in_place_with_workspace<T, D>(
    point: &Point<T, D>,
    v: &mut TangentVector<T, D>,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let norm = norm_with_workspace(point, v, workspace)?;
    if norm > T::epsilon() {
        *v /= norm;
    }
    Ok(())
}

/// Compute the Gram matrix for a set of tangent vectors using workspace.
pub fn gram_matrix_with_workspace<T>(
    _point: &DVector<T>,
    tangent_vecs: &[DVector<T>],
    workspace: &mut Workspace<T>,
) -> Result<crate::types::DMatrix<T>>
where
    T: Scalar,
{
    let n = tangent_vecs.len();
    let gram = workspace.get_or_create_matrix(BufferId::Custom(0), n, n);
    
    for i in 0..n {
        for j in i..n {
            let inner = tangent_vecs[i].dot(&tangent_vecs[j]);
            gram[(i, j)] = inner;
            if i != j {
                gram[(j, i)] = inner;
            }
        }
    }
    
    Ok(gram.clone())
}

/// Extension trait for metric operations with workspace.
pub trait MetricWorkspace<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Compute inner product with workspace.
    fn inner_product_workspace(
        &self,
        point: &Point<T, D>,
        u: &TangentVector<T, D>,
        v: &TangentVector<T, D>,
        workspace: &mut Workspace<T>,
    ) -> Result<T>;
    
    /// Compute norm with workspace.
    fn norm_workspace(
        &self,
        point: &Point<T, D>,
        v: &TangentVector<T, D>,
        workspace: &mut Workspace<T>,
    ) -> Result<T>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::DVector;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_inner_product_workspace() {
        let n = 10;
        let point = DVector::<f64>::zeros(n);
        let u = DVector::from_element(n, 1.0);
        let v = DVector::from_element(n, 2.0);
        let mut workspace = Workspace::with_size(n);
        
        let inner = inner_product_with_workspace(&point, &u, &v, &mut workspace).unwrap();
        assert_relative_eq!(inner, 20.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_normalize_workspace() {
        let n = 5;
        let point = DVector::<f64>::zeros(n);
        let mut v = DVector::from_element(n, 3.0);
        let mut workspace = Workspace::with_size(n);
        
        normalize_in_place_with_workspace(&point, &mut v, &mut workspace).unwrap();
        
        let norm = norm_with_workspace(&point, &v, &mut workspace).unwrap();
        assert_relative_eq!(norm, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_gram_matrix_workspace() {
        let n = 3;
        let point = DVector::<f64>::zeros(n);
        let vecs = vec![
            DVector::from_vec(vec![1.0, 0.0, 0.0]),
            DVector::from_vec(vec![0.0, 1.0, 0.0]),
            DVector::from_vec(vec![0.0, 0.0, 1.0]),
        ];
        let mut workspace = Workspace::with_size(n);
        
        let gram = gram_matrix_with_workspace(&point, &vecs, &mut workspace).unwrap();
        
        // Should be identity matrix
        for i in 0..3 {
            for j in 0..3 {
                if i == j {
                    assert_relative_eq!(gram[(i, j)], 1.0, epsilon = 1e-10);
                } else {
                    assert_relative_eq!(gram[(i, j)], 0.0, epsilon = 1e-10);
                }
            }
        }
    }
}