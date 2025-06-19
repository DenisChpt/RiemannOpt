//! Specialized computation implementations for specific cases.

pub mod small_dim;
pub mod sparse;
pub mod sparse_backend;

// Re-export key types
pub use small_dim::*;
pub use sparse::*;
pub use sparse_backend::*;