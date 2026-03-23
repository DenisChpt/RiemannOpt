//! Memory management utilities for optimization algorithms.

pub mod cache;
pub mod pool;
pub mod workspace;

// Re-export key items
pub use cache::{CacheKey, CacheStats, Cacheable, CachedValue, L1Cache, L2Cache, MultiLevelCache};
pub use pool::{
	get_pooled_matrix, get_pooled_vector, MatrixPool, PooledMatrix, PooledVector, VectorPool,
};
pub use workspace::{BufferId, Workspace, WorkspaceBuilder};
