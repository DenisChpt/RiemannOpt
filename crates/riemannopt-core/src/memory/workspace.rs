//! Pre-allocated workspace for optimization algorithms.
//!
//! This module provides a workspace structure that pre-allocates all temporary
//! buffers needed by optimization algorithms, eliminating allocations in hot paths.

use crate::{
    memory::pool::{VectorPool, MatrixPool, PooledVector, PooledMatrix},
    types::{Scalar, DVector, DMatrix},
};
use std::collections::HashMap;

/// Identifier for workspace buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferId {
    /// Gradient vector
    Gradient,
    /// Search direction
    Direction,
    /// Previous gradient
    PreviousGradient,
    /// Momentum buffer
    Momentum,
    /// Second moment estimate (for Adam)
    SecondMoment,
    /// Temporary vector 1
    Temp1,
    /// Temporary vector 2
    Temp2,
    /// Temporary vector 3
    Temp3,
    /// Hessian approximation
    Hessian,
    /// Preconditioner
    Preconditioner,
    /// Custom buffer with user-defined ID
    Custom(u32),
}

/// Pre-allocated workspace for optimization algorithms.
pub struct Workspace<T: Scalar> {
    /// Pre-allocated vectors indexed by buffer ID
    vectors: HashMap<BufferId, DVector<T>>,
    /// Pre-allocated matrices indexed by buffer ID
    matrices: HashMap<BufferId, DMatrix<T>>,
    /// Vector pool for dynamic allocations
    vector_pool: VectorPool<T>,
    /// Matrix pool for dynamic allocations
    matrix_pool: MatrixPool<T>,
}

impl<T: Scalar> Workspace<T> {
    /// Create a new empty workspace.
    pub fn new() -> Self {
        Self {
            vectors: HashMap::new(),
            matrices: HashMap::new(),
            vector_pool: VectorPool::new(8),
            matrix_pool: MatrixPool::new(4),
        }
    }
    
    /// Create a workspace with pre-allocated buffers for a specific problem size.
    pub fn with_size(n: usize) -> Self {
        let mut workspace = Self::new();
        
        // Pre-allocate common vector buffers
        workspace.vectors.insert(BufferId::Gradient, DVector::zeros(n));
        workspace.vectors.insert(BufferId::Direction, DVector::zeros(n));
        workspace.vectors.insert(BufferId::PreviousGradient, DVector::zeros(n));
        workspace.vectors.insert(BufferId::Temp1, DVector::zeros(n));
        workspace.vectors.insert(BufferId::Temp2, DVector::zeros(n));
        
        workspace
    }
    
    /// Pre-allocate a vector buffer.
    pub fn preallocate_vector(&mut self, id: BufferId, size: usize) {
        self.vectors.insert(id, DVector::zeros(size));
    }
    
    /// Pre-allocate a matrix buffer.
    pub fn preallocate_matrix(&mut self, id: BufferId, rows: usize, cols: usize) {
        self.matrices.insert(id, DMatrix::zeros(rows, cols));
    }
    
    /// Get a mutable reference to a pre-allocated vector buffer.
    pub fn get_vector_mut(&mut self, id: BufferId) -> Option<&mut DVector<T>> {
        self.vectors.get_mut(&id)
    }
    
    /// Get a reference to a pre-allocated vector buffer.
    pub fn get_vector(&self, id: BufferId) -> Option<&DVector<T>> {
        self.vectors.get(&id)
    }
    
    /// Get a mutable reference to a pre-allocated matrix buffer.
    pub fn get_matrix_mut(&mut self, id: BufferId) -> Option<&mut DMatrix<T>> {
        self.matrices.get_mut(&id)
    }
    
    /// Get a reference to a pre-allocated matrix buffer.
    pub fn get_matrix(&self, id: BufferId) -> Option<&DMatrix<T>> {
        self.matrices.get(&id)
    }
    
    /// Get or create a vector buffer of the specified size.
    pub fn get_or_create_vector(&mut self, id: BufferId, size: usize) -> &mut DVector<T> {
        self.vectors.entry(id).or_insert_with(|| DVector::zeros(size))
    }
    
    /// Get or create a matrix buffer of the specified dimensions.
    pub fn get_or_create_matrix(&mut self, id: BufferId, rows: usize, cols: usize) -> &mut DMatrix<T> {
        self.matrices.entry(id).or_insert_with(|| DMatrix::zeros(rows, cols))
    }
    
    /// Acquire a temporary vector from the pool.
    pub fn acquire_temp_vector(&self, size: usize) -> PooledVector<T> {
        self.vector_pool.acquire(size)
    }
    
    /// Acquire a temporary matrix from the pool.
    pub fn acquire_temp_matrix(&self, rows: usize, cols: usize) -> PooledMatrix<T> {
        self.matrix_pool.acquire(rows, cols)
    }
    
    /// Clear all buffers (fill with zeros).
    pub fn clear(&mut self) {
        for (_, vec) in self.vectors.iter_mut() {
            vec.fill(T::zero());
        }
        for (_, mat) in self.matrices.iter_mut() {
            mat.fill(T::zero());
        }
    }
    
    /// Get the total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let vector_bytes: usize = self.vectors.values()
            .map(|v| v.len() * std::mem::size_of::<T>())
            .sum();
        let matrix_bytes: usize = self.matrices.values()
            .map(|m| m.len() * std::mem::size_of::<T>())
            .sum();
        vector_bytes + matrix_bytes
    }
}

impl<T: Scalar> Default for Workspace<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating workspaces with specific buffer configurations.
pub struct WorkspaceBuilder<T: Scalar> {
    workspace: Workspace<T>,
}

impl<T: Scalar> WorkspaceBuilder<T> {
    /// Create a new workspace builder.
    pub fn new() -> Self {
        Self {
            workspace: Workspace::new(),
        }
    }
    
    /// Add a vector buffer.
    pub fn with_vector(mut self, id: BufferId, size: usize) -> Self {
        self.workspace.preallocate_vector(id, size);
        self
    }
    
    /// Add a matrix buffer.
    pub fn with_matrix(mut self, id: BufferId, rows: usize, cols: usize) -> Self {
        self.workspace.preallocate_matrix(id, rows, cols);
        self
    }
    
    /// Add standard optimization buffers for the given problem size.
    pub fn with_standard_buffers(mut self, n: usize) -> Self {
        self.workspace.preallocate_vector(BufferId::Gradient, n);
        self.workspace.preallocate_vector(BufferId::Direction, n);
        self.workspace.preallocate_vector(BufferId::PreviousGradient, n);
        self.workspace.preallocate_vector(BufferId::Temp1, n);
        self.workspace.preallocate_vector(BufferId::Temp2, n);
        self
    }
    
    /// Add momentum buffers for momentum-based methods.
    pub fn with_momentum_buffers(mut self, n: usize) -> Self {
        self.workspace.preallocate_vector(BufferId::Momentum, n);
        self
    }
    
    /// Add buffers for Adam optimizer.
    pub fn with_adam_buffers(mut self, n: usize) -> Self {
        self.workspace.preallocate_vector(BufferId::Momentum, n);
        self.workspace.preallocate_vector(BufferId::SecondMoment, n);
        self
    }
    
    /// Add buffers for quasi-Newton methods.
    pub fn with_quasi_newton_buffers(mut self, n: usize) -> Self {
        self.workspace.preallocate_matrix(BufferId::Hessian, n, n);
        self
    }
    
    /// Build the workspace.
    pub fn build(self) -> Workspace<T> {
        self.workspace
    }
}

impl<T: Scalar> Default for WorkspaceBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_workspace_basic() {
        let mut workspace = Workspace::<f64>::with_size(10);
        
        // Check pre-allocated buffers exist
        assert!(workspace.get_vector(BufferId::Gradient).is_some());
        assert!(workspace.get_vector(BufferId::Direction).is_some());
        
        // Modify a buffer
        if let Some(grad) = workspace.get_vector_mut(BufferId::Gradient) {
            grad[0] = 1.0;
        }
        
        // Verify modification
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap()[0], 1.0);
        
        // Clear workspace
        workspace.clear();
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap()[0], 0.0);
    }
    
    #[test]
    fn test_workspace_builder() {
        let workspace = WorkspaceBuilder::<f32>::new()
            .with_standard_buffers(20)
            .with_momentum_buffers(20)
            .with_matrix(BufferId::Hessian, 20, 20)
            .build();
        
        assert_eq!(workspace.get_vector(BufferId::Gradient).unwrap().len(), 20);
        assert_eq!(workspace.get_vector(BufferId::Momentum).unwrap().len(), 20);
        assert_eq!(workspace.get_matrix(BufferId::Hessian).unwrap().nrows(), 20);
    }
    
    #[test]
    fn test_get_or_create() {
        let mut workspace = Workspace::<f64>::new();
        
        // Buffer doesn't exist yet
        assert!(workspace.get_vector(BufferId::Temp3).is_none());
        
        // Get or create it
        let temp = workspace.get_or_create_vector(BufferId::Temp3, 15);
        temp[0] = 5.0;
        
        // Now it exists
        assert_eq!(workspace.get_vector(BufferId::Temp3).unwrap()[0], 5.0);
    }
    
    #[test]
    fn test_memory_pools() {
        let workspace = Workspace::<f64>::new();
        
        // Acquire temporary buffers from pools
        let mut temp1 = workspace.acquire_temp_vector(100);
        let mut temp2 = workspace.acquire_temp_matrix(10, 10);
        
        temp1[0] = 1.0;
        temp2[(0, 0)] = 2.0;
        
        // Buffers are automatically returned to pool when dropped
        drop(temp1);
        drop(temp2);
        
        // Acquire again - should get the same buffers (zeroed)
        let temp3 = workspace.acquire_temp_vector(100);
        assert_eq!(temp3[0], 0.0);
    }
    
    #[test]
    fn test_memory_usage() {
        let workspace = WorkspaceBuilder::<f64>::new()
            .with_vector(BufferId::Gradient, 100)
            .with_matrix(BufferId::Hessian, 10, 10)
            .build();
        
        let expected = 100 * 8 + 100 * 8; // 100 f64s + 10x10 f64s
        assert_eq!(workspace.memory_usage(), expected);
    }
}