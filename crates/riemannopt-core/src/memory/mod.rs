//! Memory management utilities for optimization algorithms.

pub mod pool;
pub mod workspace;

// Future modules will be added here:
// pub mod cache;

// Re-export key items
pub use pool::{
    VectorPool, MatrixPool, PooledVector, PooledMatrix,
    get_pooled_vector, get_pooled_matrix,
};
pub use workspace::{
    Workspace, WorkspaceBuilder, BufferId,
};