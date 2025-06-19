//! Memory management utilities for optimization algorithms.

pub mod pool;
pub mod workspace;
pub mod cache;

// Re-export key items
pub use pool::{
    VectorPool, MatrixPool, PooledVector, PooledMatrix,
    get_pooled_vector, get_pooled_matrix,
};
pub use workspace::{
    Workspace, WorkspaceBuilder, BufferId,
};
pub use cache::{
    CacheKey, Cacheable, CachedValue, 
    L1Cache, L2Cache, MultiLevelCache,
    CacheStats,
};