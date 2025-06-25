//! Metric operations using pre-allocated workspace.

use crate::{
    error::Result,
    manifold::{Point, TangentVector},
    memory::Workspace,
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DVector};
use num_traits::Float;

/// Compute the inner product between two tangent vectors using workspace.
pub fn inner_product<T, D>(
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
pub fn norm<T, D>(
    point: &Point<T, D>,
    v: &TangentVector<T, D>,
    workspace: &mut Workspace<T>,
) -> Result<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let inner = inner_product(point, v, v, workspace)?;
    Ok(<T as Float>::sqrt(inner))
}

/// Normalize a tangent vector in-place using workspace.
pub fn normalize_in_place<T, D>(
    point: &Point<T, D>,
    v: &mut TangentVector<T, D>,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let norm = norm(point, v, workspace)?;
    if norm > T::epsilon() {
        *v /= norm;
    }
    Ok(())
}

/// Compute the Gram matrix for a set of tangent vectors using workspace.
pub fn gram_matrix<T>(
    _point: &DVector<T>,
    tangent_vecs: &[DVector<T>],
    _workspace: &mut Workspace<T>,
    gram: &mut crate::types::DMatrix<T>,
) -> Result<()>
where
    T: Scalar,
{
    let n = tangent_vecs.len();
    assert_eq!(gram.nrows(), n, "Gram matrix must have correct dimensions");
    assert_eq!(gram.ncols(), n, "Gram matrix must be square");
    
    for i in 0..n {
        for j in i..n {
            let inner = tangent_vecs[i].dot(&tangent_vecs[j]);
            gram[(i, j)] = inner;
            if i != j {
                gram[(j, i)] = inner;
            }
        }
    }
    
    Ok(())
}

/// Compute the inner product (allocating version).
pub fn inner_product_alloc<T, D>(
    point: &Point<T, D>,
    u: &TangentVector<T, D>,
    v: &TangentVector<T, D>,
) -> Result<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let mut workspace = Workspace::new();
    inner_product(point, u, v, &mut workspace)
}

/// Compute the norm (allocating version).
pub fn norm_alloc<T, D>(
    point: &Point<T, D>,
    v: &TangentVector<T, D>,
) -> Result<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let mut workspace = Workspace::new();
    norm(point, v, &mut workspace)
}

/// Normalize a tangent vector (allocating version).
pub fn normalize_alloc<T, D>(
    point: &Point<T, D>,
    v: &TangentVector<T, D>,
) -> Result<TangentVector<T, D>>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let mut workspace = Workspace::new();
    let mut result = v.clone();
    normalize_in_place(point, &mut result, &mut workspace)?;
    Ok(result)
}

/// Compute the Gram matrix (allocating version).
pub fn gram_matrix_alloc<T>(
    point: &DVector<T>,
    tangent_vecs: &[DVector<T>],
) -> Result<crate::types::DMatrix<T>>
where
    T: Scalar,
{
    let n = tangent_vecs.len();
    let mut workspace = Workspace::new();
    let mut gram = crate::types::DMatrix::zeros(n, n);
    gram_matrix(point, tangent_vecs, &mut workspace, &mut gram)?;
    Ok(gram)
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
        
        let inner = inner_product(&point, &u, &v, &mut workspace).unwrap();
        assert_relative_eq!(inner, 20.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_normalize_workspace() {
        let n = 5;
        let point = DVector::<f64>::zeros(n);
        let mut v = DVector::from_element(n, 3.0);
        let mut workspace = Workspace::with_size(n);
        
        normalize_in_place(&point, &mut v, &mut workspace).unwrap();
        
        let norm = norm(&point, &v, &mut workspace).unwrap();
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
        
        let mut gram = crate::types::DMatrix::<f64>::zeros(3, 3);
        gram_matrix(&point, &vecs, &mut workspace, &mut gram).unwrap();
        
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