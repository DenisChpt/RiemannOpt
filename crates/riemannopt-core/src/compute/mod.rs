//! Computational backends and utilities.

pub mod backend;
pub mod cpu;
pub mod gpu;
pub mod specialized;

// Re-export backend types
pub use backend::{BackendSelector, BackendSelection, ComputeBackend};

// Re-export CPU operations
pub use cpu::*;