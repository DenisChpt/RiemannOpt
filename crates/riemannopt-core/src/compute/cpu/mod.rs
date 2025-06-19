//! CPU-based computation implementations.

pub mod parallel;
pub mod simd;
pub mod simd_dispatch;
pub mod wide_backend;

// Re-export all CPU operations
pub use parallel::*;
pub use simd::*;
pub use simd_dispatch::{SimdBackend, SimdDispatcher, get_dispatcher};