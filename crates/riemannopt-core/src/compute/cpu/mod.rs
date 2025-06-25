//! CPU-based computation implementations.

pub mod batch_ops;
pub mod parallel;
pub mod parallel_strategy;
pub mod simd;
pub mod simd_dispatch;
pub mod wide_backend;

// Re-export all CPU operations
pub use batch_ops::*;
pub use parallel::*;
pub use parallel_strategy::*;
pub use simd::*;
pub use simd_dispatch::{SimdBackend, SimdDispatcher, get_dispatcher};