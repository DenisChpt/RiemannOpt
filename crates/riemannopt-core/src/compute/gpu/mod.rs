//! GPU computation backend infrastructure.
//!
//! This module provides the foundation for GPU-accelerated computation.
//! Currently, it defines the interfaces and detection mechanisms.
//! Actual GPU implementations (CUDA, ROCm, Metal) will be added later.

pub mod selector;
pub mod traits;

// Re-export key types
pub use selector::*;
pub use traits::*;