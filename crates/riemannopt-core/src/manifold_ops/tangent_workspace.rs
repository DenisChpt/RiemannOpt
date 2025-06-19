//! Tangent space operations using pre-allocated workspace.

use crate::{
    error::Result,
    manifold::{Point, TangentVector},
    memory::{Workspace, BufferId},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DVector};

/// Project a vector onto the tangent space using workspace.
pub fn project_with_workspace<T, D>(
    _point: &Point<T, D>,
    vector: &TangentVector<T, D>,
    _workspace: &mut Workspace<T>,
    result: &mut TangentVector<T, D>,
) -> Result<()>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    // For Euclidean manifold, projection is identity
    // More complex manifolds would use workspace for computation
    result.copy_from(vector);
    Ok(())
}

/// Project a vector onto the tangent space in-place using workspace.
pub fn project_in_place_with_workspace<T, D>(
    _point: &Point<T, D>,
    _vector: &mut TangentVector<T, D>,
    _workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    // For Euclidean manifold, projection is identity
    // More complex manifolds would modify vector in-place using workspace
    Ok(())
}

/// Compute the orthogonal projection matrix for the tangent space using workspace.
pub fn projection_matrix_with_workspace<T>(
    point: &DVector<T>,
    _workspace: &mut Workspace<T>,
    proj: &mut crate::types::DMatrix<T>,
) -> Result<()>
where
    T: Scalar,
{
    let n = point.len();
    assert_eq!(proj.nrows(), n, "Projection matrix must have correct dimensions");
    assert_eq!(proj.ncols(), n, "Projection matrix must be square");
    
    // For Euclidean space, projection matrix is identity
    proj.fill(T::zero());
    proj.fill_diagonal(T::one());
    
    Ok(())
}

/// Orthogonalize a set of tangent vectors using workspace (Gram-Schmidt).
pub fn orthogonalize_with_workspace<T>(
    point: &DVector<T>,
    vectors: &mut [DVector<T>],
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
{
    let n = vectors.len();
    if n == 0 {
        return Ok(());
    }
    
    // Use workspace for temporary storage
    let temp = workspace.get_or_create_vector(BufferId::Temp3, point.len());
    
    for i in 1..n {
        // Copy current vector to temp
        temp.copy_from(&vectors[i]);
        
        // Project out components in directions of previous vectors
        for j in 0..i {
            let dot = temp.dot(&vectors[j]);
            let norm_sq = vectors[j].dot(&vectors[j]);
            
            if norm_sq > T::epsilon() {
                let scale = dot / norm_sq;
                // Update vectors[i] using temp value, not its own value
                temp.axpy(-scale, &vectors[j], T::one());
            }
        }
        
        // Copy result back
        vectors[i].copy_from(&*temp);
    }
    
    Ok(())
}

/// Compute the parallel transport of a vector along a curve using workspace.
pub fn parallel_transport_with_workspace<T, D>(
    _from_point: &Point<T, D>,
    _to_point: &Point<T, D>,
    vector: &TangentVector<T, D>,
    _workspace: &mut Workspace<T>,
    result: &mut TangentVector<T, D>,
) -> Result<()>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    // For Euclidean manifold, parallel transport is identity
    // More complex manifolds would use workspace for computation
    result.copy_from(vector);
    Ok(())
}

/// Extension trait for tangent space operations with workspace.
pub trait TangentWorkspace<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Project onto tangent space with workspace.
    fn project_workspace(
        &self,
        point: &Point<T, D>,
        vector: &TangentVector<T, D>,
        workspace: &mut Workspace<T>,
        result: &mut TangentVector<T, D>,
    ) -> Result<()>;
    
    /// Parallel transport with workspace.
    fn parallel_transport_workspace(
        &self,
        from: &Point<T, D>,
        to: &Point<T, D>,
        vector: &TangentVector<T, D>,
        workspace: &mut Workspace<T>,
        result: &mut TangentVector<T, D>,
    ) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_project_workspace() {
        let n = 5;
        let point = DVector::<f64>::zeros(n);
        let vector = DVector::from_element(n, 1.0);
        let mut workspace = Workspace::with_size(n);
        
        let mut projected = DVector::<f64>::zeros(n);
        project_with_workspace(&point, &vector, &mut workspace, &mut projected).unwrap();
        
        // For Euclidean space, projection is identity
        for i in 0..n {
            assert_relative_eq!(projected[i], vector[i], epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_orthogonalize_workspace() {
        let point = DVector::<f64>::zeros(3);
        let mut vectors = vec![
            DVector::from_vec(vec![1.0, 1.0, 0.0]),
            DVector::from_vec(vec![1.0, 0.0, 1.0]),
            DVector::from_vec(vec![0.0, 1.0, 1.0]),
        ];
        let mut workspace = Workspace::with_size(3);
        
        orthogonalize_with_workspace(&point, &mut vectors, &mut workspace).unwrap();
        
        // Check orthogonality
        for i in 0..3 {
            for j in 0..i {
                let dot = vectors[i].dot(&vectors[j]);
                assert!(dot.abs() < 1e-10, "Vectors {} and {} not orthogonal: dot = {}", i, j, dot);
            }
        }
    }
    
    #[test]
    fn test_parallel_transport_workspace() {
        let n = 4;
        let from_point = DVector::<f64>::zeros(n);
        let to_point = DVector::from_element(n, 1.0);
        let vector = DVector::from_element(n, 2.0);
        let mut workspace = Workspace::with_size(n);
        
        let mut transported = DVector::<f64>::zeros(n);
        parallel_transport_with_workspace(
            &from_point,
            &to_point,
            &vector,
            &mut workspace,
            &mut transported
        ).unwrap();
        
        // For Euclidean space, parallel transport is identity
        for i in 0..n {
            assert_relative_eq!(transported[i], vector[i], epsilon = 1e-10);
        }
    }
}