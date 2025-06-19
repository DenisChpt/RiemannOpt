//! Manifold-specific operations.

pub mod fisher;
pub mod metric;
pub mod retraction;
pub mod tangent;

// Re-export manifold operations
pub use fisher::*;
pub use metric::*;
pub use retraction::*;
pub use tangent::*;