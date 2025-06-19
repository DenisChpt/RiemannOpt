//! CPU-based computation implementations.

pub mod parallel;
pub mod simd;

// Re-export all CPU operations
pub use parallel::*;
pub use simd::*;