//! Performance profiling and metrics collection.
//!
//! This module provides tools for monitoring and optimizing the performance
//! of Riemannian optimization algorithms.

pub mod metrics;
pub mod auto_tuner;

// Re-export commonly used types
pub use metrics::{
    PerformanceCollector, MetricType, MetricsSummary, 
    InMemoryCollector, AtomicCollector, TimerGuard,
};
pub use auto_tuner::{AutoTuner, AutoTunable, TunableParameter, TuningResult};