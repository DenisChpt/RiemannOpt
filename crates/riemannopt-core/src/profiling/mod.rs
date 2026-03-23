//! Performance profiling and metrics collection.
//!
//! This module provides tools for monitoring and optimizing the performance
//! of Riemannian optimization algorithms.

pub mod auto_tuner;
pub mod metrics;

// Re-export commonly used types
pub use auto_tuner::{AutoTunable, AutoTuner, TunableParameter, TuningResult};
pub use metrics::{
	AtomicCollector, InMemoryCollector, MetricType, MetricsSummary, PerformanceCollector,
	TimerGuard,
};
