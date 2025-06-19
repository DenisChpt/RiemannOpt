//! Configuration utilities for the optimization library.

pub mod features;

// Future modules will be added here:
// pub mod runtime;
// pub mod tuning;

// Re-export key items
pub use features::{cpu_features, simd_config, CpuFeatures, SimdConfig, SimdConfigBuilder};