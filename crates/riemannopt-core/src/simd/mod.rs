//! SIMD-accelerated computational backends.
//!
//! This module provides SIMD (Single Instruction, Multiple Data) optimized
//! implementations of common linear algebra operations using the `wide` crate,
//! along with a runtime dispatch mechanism that selects the best backend.

pub mod dispatch;
pub mod ops;
pub mod wide_backend;

// Re-export key types
pub use dispatch::{get_dispatcher, ScalarBackend, ScalarDispatch, SimdBackend, SimdDispatcher};
pub use ops::{SimdOps, SimdVector};
