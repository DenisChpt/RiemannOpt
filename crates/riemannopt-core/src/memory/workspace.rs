//! Pre-allocated workspace for optimization algorithms.
//!
//! This module provides a workspace structure that pre-allocates all temporary
//! buffers needed by optimization algorithms, eliminating allocations in hot paths.

use crate::{
    memory::pool::{VectorPool, MatrixPool, PooledVector, PooledMatrix},
    types::{Scalar, DVector, DMatrix},
};
use std::collections::HashMap;
use std::any::Any;

/// Trait for buffers that can be stored in the workspace.
///
/// This trait serves as a contract for anything that can be stored in the workspace.
/// It provides methods for memory management and type-safe access.
pub trait WorkspaceBuffer<T: Scalar>: Any + Send + Sync {
    /// Clear the buffer to a neutral state (e.g., fill with zeros).
    fn clear(&mut self);

    /// Calculate the memory usage of this buffer in bytes.
    fn size_bytes(&self) -> usize;

    /// Clone the buffer into a boxed trait object.
    fn dyn_clone(&self) -> Box<dyn WorkspaceBuffer<T>>;

    /// Get a mutable reference to the underlying Any type for downcasting.
    fn as_any_mut(&mut self) -> &mut dyn Any;

    /// Get a reference to the underlying Any type for downcasting.
    fn as_any(&self) -> &dyn Any;
}

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
    /// Unit vector for finite differences
    UnitVector,
    /// Point plus perturbation for finite differences
    PointPlus,
    /// Point minus perturbation for finite differences
    PointMinus,
    /// Hessian approximation
    Hessian,
    /// Preconditioner
    Preconditioner,
    /// Custom buffer with user-defined ID
    Custom(u32),
}

/// Implementation of WorkspaceBuffer for DVector.
impl<T: Scalar> WorkspaceBuffer<T> for DVector<T> {
    fn clear(&mut self) {
        self.fill(T::zero());
    }

    fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn dyn_clone(&self) -> Box<dyn WorkspaceBuffer<T>> {
        Box::new(self.clone())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Implementation of WorkspaceBuffer for DMatrix.
impl<T: Scalar> WorkspaceBuffer<T> for DMatrix<T> {
    fn clear(&mut self) {
        self.fill(T::zero());
    }

    fn size_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn dyn_clone(&self) -> Box<dyn WorkspaceBuffer<T>> {
        Box::new(self.clone())
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Pre-allocated workspace for optimization algorithms.
pub struct Workspace<T: Scalar> {
    /// Pre-allocated buffers indexed by buffer ID
    buffers: HashMap<BufferId, Box<dyn WorkspaceBuffer<T>>>,
    /// Vector pool for dynamic allocations
    vector_pool: VectorPool<T>,
    /// Matrix pool for dynamic allocations
    matrix_pool: MatrixPool<T>,
}

impl<T: Scalar> Workspace<T> {
    /// Create a new empty workspace.
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            vector_pool: VectorPool::new(8),
            matrix_pool: MatrixPool::new(4),
        }
    }
    
    /// Create a workspace with pre-allocated buffers for a specific problem size.
    pub fn with_size(n: usize) -> Self {
        let mut workspace = Self::new();
        
        // Pre-allocate common vector buffers
        workspace.buffers.insert(BufferId::Gradient, Box::new(DVector::<T>::zeros(n)) as Box<dyn WorkspaceBuffer<T>>);
        workspace.buffers.insert(BufferId::Direction, Box::new(DVector::<T>::zeros(n)) as Box<dyn WorkspaceBuffer<T>>);
        workspace.buffers.insert(BufferId::PreviousGradient, Box::new(DVector::<T>::zeros(n)) as Box<dyn WorkspaceBuffer<T>>);
        workspace.buffers.insert(BufferId::Temp1, Box::new(DVector::<T>::zeros(n)) as Box<dyn WorkspaceBuffer<T>>);
        workspace.buffers.insert(BufferId::Temp2, Box::new(DVector::<T>::zeros(n)) as Box<dyn WorkspaceBuffer<T>>);
        workspace.buffers.insert(BufferId::UnitVector, Box::new(DVector::<T>::zeros(n)) as Box<dyn WorkspaceBuffer<T>>);
        workspace.buffers.insert(BufferId::PointPlus, Box::new(DVector::<T>::zeros(n)) as Box<dyn WorkspaceBuffer<T>>);
        workspace.buffers.insert(BufferId::PointMinus, Box::new(DVector::<T>::zeros(n)) as Box<dyn WorkspaceBuffer<T>>);
        
        workspace
    }
    
    /// Pre-allocate a generic buffer.
    pub fn preallocate_buffer<B: WorkspaceBuffer<T> + 'static>(&mut self, id: BufferId, buffer: B) {
        self.buffers.insert(id, Box::new(buffer));
    }
    
    /// Pre-allocate a vector buffer.
    pub fn preallocate_vector(&mut self, id: BufferId, size: usize) {
        self.preallocate_buffer(id, DVector::<T>::zeros(size));
    }
    
    /// Pre-allocate a matrix buffer.
    pub fn preallocate_matrix(&mut self, id: BufferId, rows: usize, cols: usize) {
        self.preallocate_buffer(id, DMatrix::<T>::zeros(rows, cols));
    }
    
    /// Get a mutable reference to a buffer.
    pub fn get_buffer_mut<'a, B: 'static>(&'a mut self, id: BufferId) -> Option<&'a mut B> {
        self.buffers.get_mut(&id)?.as_any_mut().downcast_mut::<B>()
    }
    
    /// Get a reference to a buffer.
    pub fn get_buffer<'a, B: 'static>(&'a self, id: BufferId) -> Option<&'a B> {
        self.buffers.get(&id)?.as_any().downcast_ref::<B>()
    }
    
    /// Get or create a buffer.
    pub fn get_or_create_buffer<B: WorkspaceBuffer<T> + 'static>(&mut self, id: BufferId, default: impl FnOnce() -> B) -> &mut B {
        self.buffers
            .entry(id)
            .or_insert_with(|| Box::new(default()))
            .as_any_mut()
            .downcast_mut::<B>()
            .unwrap() // Panic if the type is incorrect, which is a logic error.
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
        for (_, buffer) in self.buffers.iter_mut() {
            buffer.clear();
        }
    }
    
    /// Get the total memory usage in bytes.
    pub fn memory_usage(&self) -> usize {
        let mut total_bytes = 0;
        for (_, buffer) in self.buffers.iter() {
            total_bytes += buffer.size_bytes();
        }
        total_bytes
    }
    
}

impl<T: Scalar> Default for Workspace<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Scalar> Clone for Workspace<T> {
    fn clone(&self) -> Self {
        let mut new_workspace = Self {
            buffers: HashMap::new(),
            vector_pool: self.vector_pool.clone(),
            matrix_pool: self.matrix_pool.clone(),
        };
        
        // Clone all buffers using the dyn_clone method
        for (id, buffer) in &self.buffers {
            new_workspace.buffers.insert(*id, buffer.dyn_clone());
        }
        
        new_workspace
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
    
    /// Add a generic buffer.
    pub fn with_buffer<B: WorkspaceBuffer<T> + 'static>(mut self, id: BufferId, buffer: B) -> Self {
        self.workspace.preallocate_buffer(id, buffer);
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
        assert!(workspace.get_buffer::<DVector<f64>>(BufferId::Gradient).is_some());
        assert!(workspace.get_buffer::<DVector<f64>>(BufferId::Direction).is_some());
        
        // Modify a buffer
        if let Some(grad) = workspace.get_buffer_mut::<DVector<f64>>(BufferId::Gradient) {
            grad[0] = 1.0;
        }
        
        // Verify modification
        assert_eq!(workspace.get_buffer::<DVector<f64>>(BufferId::Gradient).unwrap()[0], 1.0);
        
        // Clear workspace
        workspace.clear();
        assert_eq!(workspace.get_buffer::<DVector<f64>>(BufferId::Gradient).unwrap()[0], 0.0);
    }
    
    #[test]
    fn test_workspace_builder() {
        let workspace = WorkspaceBuilder::<f32>::new()
            .with_buffer(BufferId::Gradient, DVector::<f32>::zeros(20))
            .with_buffer(BufferId::Direction, DVector::<f32>::zeros(20))
            .with_buffer(BufferId::PreviousGradient, DVector::<f32>::zeros(20))
            .with_buffer(BufferId::Temp1, DVector::<f32>::zeros(20))
            .with_buffer(BufferId::Temp2, DVector::<f32>::zeros(20))
            .with_buffer(BufferId::Momentum, DVector::<f32>::zeros(20))
            .with_buffer(BufferId::Hessian, DMatrix::<f32>::zeros(20, 20))
            .build();
        
        assert_eq!(workspace.get_buffer::<DVector<f32>>(BufferId::Gradient).unwrap().len(), 20);
        assert_eq!(workspace.get_buffer::<DVector<f32>>(BufferId::Momentum).unwrap().len(), 20);
        assert_eq!(workspace.get_buffer::<DMatrix<f32>>(BufferId::Hessian).unwrap().nrows(), 20);
    }
    
    #[test]
    fn test_get_or_create() {
        let mut workspace = Workspace::<f64>::new();
        
        // Buffer doesn't exist yet
        assert!(workspace.get_buffer::<DVector<f64>>(BufferId::Temp3).is_none());
        
        // Get or create it
        let temp = workspace.get_or_create_buffer(BufferId::Temp3, || DVector::<f64>::zeros(15));
        temp[0] = 5.0;
        
        // Now it exists
        assert_eq!(workspace.get_buffer::<DVector<f64>>(BufferId::Temp3).unwrap()[0], 5.0);
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
            .with_buffer(BufferId::Gradient, DVector::<f64>::zeros(100))
            .with_buffer(BufferId::Hessian, DMatrix::<f64>::zeros(10, 10))
            .build();
        
        let expected = 100 * 8 + 100 * 8; // 100 f64s + 10x10 f64s
        assert_eq!(workspace.memory_usage(), expected);
    }
}