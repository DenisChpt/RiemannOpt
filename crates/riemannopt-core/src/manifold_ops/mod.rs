//! Manifold-specific operations.

pub mod fisher;
pub mod metric;
pub mod metric_simd;
pub mod metric_workspace;
pub mod retraction;
pub mod retraction_workspace;
pub mod tangent;
pub mod tangent_simd;
pub mod tangent_workspace;

// Re-export manifold operations
pub use fisher::*;
pub use metric::*;
pub use metric_simd::*;
pub use metric_workspace::*;
pub use retraction::*;
pub use retraction_workspace::*;
pub use tangent::*;
pub use tangent_simd::*;
pub use tangent_workspace::*;