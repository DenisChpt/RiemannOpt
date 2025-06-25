//! Retraction operations using pre-allocated workspace.

use crate::{
    error::Result,
    manifold::{Point, TangentVector},
    memory::{Workspace, BufferId},
    types::Scalar,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, DVector};

/// Perform exponential map using workspace.
pub fn exponential_map<T, D>(
    point: &Point<T, D>,
    direction: &TangentVector<T, D>,
    _workspace: &mut Workspace<T>,
) -> Result<Point<T, D>>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    // For Euclidean manifold, exponential map is just addition
    // More complex manifolds would use workspace for computation
    Ok(point + direction)
}

/// Perform retraction using workspace.
pub fn retract<T, D>(
    point: &Point<T, D>,
    direction: &TangentVector<T, D>,
    t: T,
    _workspace: &mut Workspace<T>,
) -> Result<Point<T, D>>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    // For Euclidean manifold, retraction is linear
    // More complex manifolds would use workspace
    Ok(point + direction * t)
}

/// Compute inverse retraction (logarithm map) using workspace.
pub fn inverse_retract<T, D>(
    base_point: &Point<T, D>,
    target_point: &Point<T, D>,
    _workspace: &mut Workspace<T>,
) -> Result<TangentVector<T, D>>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    // For Euclidean manifold, this is just subtraction
    // More complex manifolds would use workspace
    Ok(target_point - base_point)
}

/// Compute geodesic distance using workspace.
pub fn geodesic_distance<T, D>(
    point1: &Point<T, D>,
    point2: &Point<T, D>,
    workspace: &mut Workspace<T>,
) -> Result<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let diff = inverse_retract(point1, point2, workspace)?;
    Ok(diff.norm())
}

/// Compute multiple retractions efficiently using workspace.
/// The results vector must be pre-allocated with the correct number of elements.
pub fn batch_retract<T>(
    point: &DVector<T>,
    directions: &[DVector<T>],
    t: T,
    workspace: &mut Workspace<T>,
    results: &mut [DVector<T>],
) -> Result<()>
where
    T: Scalar,
{
    assert_eq!(directions.len(), results.len(), "Results vector must have same length as directions");
    
    let n = point.len();
    let temp = workspace.get_or_create_vector(BufferId::Temp1, n);
    
    for (direction, result) in directions.iter().zip(results.iter_mut()) {
        // temp = point + t * direction
        temp.copy_from(point);
        temp.axpy(t, direction, T::one());
        result.copy_from(&*temp);
    }
    
    Ok(())
}

/// Perform exponential map (allocating version).
pub fn exponential_map_alloc<T, D>(
    point: &Point<T, D>,
    direction: &TangentVector<T, D>,
) -> Result<Point<T, D>>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let mut workspace = Workspace::new();
    exponential_map(point, direction, &mut workspace)
}

/// Perform retraction (allocating version).
pub fn retract_alloc<T, D>(
    point: &Point<T, D>,
    direction: &TangentVector<T, D>,
    t: T,
) -> Result<Point<T, D>>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let mut workspace = Workspace::new();
    retract(point, direction, t, &mut workspace)
}

/// Compute inverse retraction (allocating version).
pub fn inverse_retract_alloc<T, D>(
    base_point: &Point<T, D>,
    target_point: &Point<T, D>,
) -> Result<TangentVector<T, D>>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let mut workspace = Workspace::new();
    inverse_retract(base_point, target_point, &mut workspace)
}

/// Compute geodesic distance (allocating version).
pub fn geodesic_distance_alloc<T, D>(
    point1: &Point<T, D>,
    point2: &Point<T, D>,
) -> Result<T>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    let mut workspace = Workspace::new();
    geodesic_distance(point1, point2, &mut workspace)
}

/// Extension trait for retraction operations with workspace.
pub trait RetractionWorkspace<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D>,
{
    /// Exponential map with workspace.
    fn exponential_map_workspace(
        &self,
        point: &Point<T, D>,
        direction: &TangentVector<T, D>,
        workspace: &mut Workspace<T>,
    ) -> Result<Point<T, D>>;
    
    /// Retraction with workspace.
    fn retract_workspace(
        &self,
        point: &Point<T, D>,
        direction: &TangentVector<T, D>,
        t: T,
        workspace: &mut Workspace<T>,
    ) -> Result<Point<T, D>>;
    
    /// Inverse retraction with workspace.
    fn inverse_retract_workspace(
        &self,
        base: &Point<T, D>,
        target: &Point<T, D>,
        workspace: &mut Workspace<T>,
    ) -> Result<TangentVector<T, D>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_retract_workspace() {
        let n = 5;
        let point = DVector::<f64>::zeros(n);
        let direction = DVector::from_element(n, 1.0);
        let t = 0.5;
        let mut workspace = Workspace::with_size(n);
        
        let result = retract(&point, &direction, t, &mut workspace).unwrap();
        
        // For Euclidean space: result = point + t * direction
        for i in 0..n {
            assert_relative_eq!(result[i], 0.5, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_geodesic_distance_workspace() {
        let n = 3;
        let point1 = DVector::<f64>::from_vec(vec![0.0, 0.0, 0.0]);
        let point2 = DVector::from_vec(vec![3.0, 4.0, 0.0]);
        let mut workspace = Workspace::with_size(n);
        
        let dist = geodesic_distance(&point1, &point2, &mut workspace).unwrap();
        
        // Euclidean distance: sqrt(3^2 + 4^2) = 5
        assert_relative_eq!(dist, 5.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_batch_retract_workspace() {
        let n = 4;
        let point = DVector::<f64>::zeros(n);
        let directions = vec![
            DVector::from_element(n, 1.0),
            DVector::from_element(n, 2.0),
            DVector::from_element(n, 3.0),
        ];
        let t = 0.1;
        let mut workspace = Workspace::with_size(n);
        
        let mut results = vec![DVector::<f64>::zeros(n); 3];
        batch_retract(&point, &directions, t, &mut workspace, &mut results).unwrap();
        
        assert_eq!(results.len(), 3);
        for (i, result) in results.iter().enumerate() {
            let expected = (i + 1) as f64 * 0.1;
            for j in 0..n {
                assert_relative_eq!(result[j], expected, epsilon = 1e-10);
            }
        }
    }
}