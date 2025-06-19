//! Manifold-specific operations.

pub mod fisher;
pub mod metric;
pub mod metric_simd;
pub mod retraction;
pub mod tangent;
pub mod tangent_simd;

// Re-export manifold operations
pub use fisher::*;
pub use metric::*;
pub use metric_simd::*;
pub use retraction::*;
pub use tangent::*;
pub use tangent_simd::*;