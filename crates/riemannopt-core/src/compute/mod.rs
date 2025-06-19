//! Computational backends and utilities.

pub mod cpu;
pub mod specialized;

// Re-export CPU operations
pub use cpu::*;