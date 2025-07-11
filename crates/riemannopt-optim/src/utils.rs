//! Utility functions for optimizers.
//!
//! This module provides common functionality used across different optimizers
//! to reduce code duplication and improve maintainability.

use riemannopt_core::{
    core::manifold::Manifold,
    types::Scalar,
    memory::workspace::Workspace,
    error::Result,
};

/// Helper trait for workspace-based operations.
///
/// This trait provides common patterns for workspace management in optimizers.
#[allow(dead_code)]
pub trait WorkspaceOps<T: Scalar> {
    /// Gets a mutable reference to the workspace.
    fn workspace_mut(&mut self) -> &mut Workspace<T>;
    
    /// Executes a closure with the workspace, handling borrowing correctly.
    fn with_workspace<F, R>(&mut self, f: F) -> R
    where
        F: FnOnce(&mut Workspace<T>) -> R,
    {
        let workspace = self.workspace_mut();
        f(workspace)
    }
}

/// Performs a workspace-scoped operation on a manifold.
///
/// This function helps manage the workspace borrowing pattern common in optimizers.
#[allow(dead_code)]
pub fn with_manifold_workspace<T, M, F, R>(
    manifold: &M,
    workspace: &mut Workspace<T>,
    f: F,
) -> Result<R>
where
    T: Scalar,
    M: Manifold<T>,
    F: FnOnce(&M, &mut Workspace<T>) -> Result<R>,
{
    f(manifold, workspace)
}

/// Scales a tangent vector by a scalar using workspace.
#[allow(dead_code)]
pub fn scale_tangent_with_workspace<T, M>(
    manifold: &M,
    point: &M::Point,
    scalar: T,
    tangent: &M::TangentVector,
    result: &mut M::TangentVector,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
    M: Manifold<T>,
{
    let _ = workspace; // Suppress unused warning
    manifold.scale_tangent(point, scalar, tangent, result)
}

/// Performs retraction using workspace.
#[allow(dead_code)]
pub fn retract_with_workspace<T, M>(
    manifold: &M,
    point: &M::Point,
    tangent: &M::TangentVector,
    result: &mut M::Point,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
    M: Manifold<T>,
{
    let _ = workspace; // Suppress unused warning
    manifold.retract(point, tangent, result)
}

/// Transports a tangent vector from one point to another using workspace.
#[allow(dead_code)]
pub fn transport_with_workspace<T, M>(
    manifold: &M,
    from: &M::Point,
    to: &M::Point,
    tangent: &M::TangentVector,
    result: &mut M::TangentVector,
    workspace: &mut Workspace<T>,
) -> Result<()>
where
    T: Scalar,
    M: Manifold<T>,
{
    let _ = workspace; // Suppress unused warning
    manifold.parallel_transport(from, to, tangent, result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use riemannopt_core::memory::workspace::Workspace;
    
    struct MockOptimizer<T: Scalar> {
        workspace: Workspace<T>,
    }
    
    impl<T: Scalar> WorkspaceOps<T> for MockOptimizer<T> {
        fn workspace_mut(&mut self) -> &mut Workspace<T> {
            &mut self.workspace
        }
    }
    
    #[test]
    fn test_workspace_ops() {
        let mut optimizer = MockOptimizer::<f64> {
            workspace: Workspace::new(),
        };
        
        // Test with_workspace
        let result = optimizer.with_workspace(|ws| {
            ws.preallocate_vector(riemannopt_core::memory::workspace::BufferId::Gradient, 10);
            42
        });
        
        assert_eq!(result, 42);
        assert!(optimizer.workspace.get_vector(riemannopt_core::memory::workspace::BufferId::Gradient).is_some());
    }
}