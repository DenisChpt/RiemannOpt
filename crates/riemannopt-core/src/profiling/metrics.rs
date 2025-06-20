//! Performance metrics collection for optimization algorithms.
//!
//! This module provides a lightweight system for collecting performance metrics
//! during optimization, including timing information, operation counts, and
//! memory usage statistics.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicU64, Ordering};

/// A performance metric that can be collected.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Time spent in function evaluations
    CostFunctionTime,
    /// Time spent in gradient computations
    GradientTime,
    /// Time spent in retraction operations
    RetractionTime,
    /// Time spent in vector transport
    VectorTransportTime,
    /// Time spent in line search
    LineSearchTime,
    /// Number of function evaluations
    FunctionEvals,
    /// Number of gradient evaluations
    GradientEvals,
    /// Number of retraction operations
    Retractions,
    /// Memory allocated (bytes)
    MemoryAllocated,
    /// Custom metric with a name
    Custom(String),
}

/// A single metric measurement.
#[derive(Debug, Clone)]
pub struct Measurement {
    /// The type of metric
    pub metric_type: MetricType,
    /// The value (duration in nanoseconds for time metrics, count for others)
    pub value: u64,
    /// When the measurement was taken
    pub timestamp: Instant,
}

/// Trait for types that can collect performance metrics.
pub trait PerformanceCollector: Send + Sync {
    /// Records a duration for a specific metric.
    fn record_duration(&self, metric: MetricType, duration: Duration);
    
    /// Records a count for a specific metric.
    fn record_count(&self, metric: MetricType, count: u64);
    
    /// Increments a counter metric by 1.
    fn increment(&self, metric: MetricType) {
        self.record_count(metric, 1);
    }
    
    /// Returns a summary of all collected metrics.
    fn summary(&self) -> MetricsSummary;
    
    /// Clears all collected metrics.
    fn clear(&self);
}

/// Summary statistics for a metric.
#[derive(Debug, Clone)]
pub struct MetricStats {
    /// Total count of measurements
    pub count: usize,
    /// Sum of all values
    pub sum: u64,
    /// Minimum value
    pub min: u64,
    /// Maximum value
    pub max: u64,
    /// Average value
    pub avg: f64,
}

/// Summary of all collected metrics.
#[derive(Debug, Clone)]
pub struct MetricsSummary {
    /// Statistics for each metric type
    pub stats: HashMap<MetricType, MetricStats>,
    /// Total runtime
    pub total_time: Duration,
}

/// A simple in-memory metrics collector.
#[derive(Debug)]
pub struct InMemoryCollector {
    measurements: Arc<Mutex<Vec<Measurement>>>,
    start_time: Instant,
}

impl InMemoryCollector {
    /// Creates a new in-memory metrics collector.
    pub fn new() -> Self {
        Self {
            measurements: Arc::new(Mutex::new(Vec::new())),
            start_time: Instant::now(),
        }
    }
}

impl Default for InMemoryCollector {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceCollector for InMemoryCollector {
    fn record_duration(&self, metric: MetricType, duration: Duration) {
        let measurement = Measurement {
            metric_type: metric,
            value: duration.as_nanos() as u64,
            timestamp: Instant::now(),
        };
        self.measurements.lock().unwrap().push(measurement);
    }
    
    fn record_count(&self, metric: MetricType, count: u64) {
        let measurement = Measurement {
            metric_type: metric,
            value: count,
            timestamp: Instant::now(),
        };
        self.measurements.lock().unwrap().push(measurement);
    }
    
    fn summary(&self) -> MetricsSummary {
        let measurements = self.measurements.lock().unwrap();
        let mut stats_map: HashMap<MetricType, Vec<u64>> = HashMap::new();
        
        // Group measurements by metric type
        for measurement in measurements.iter() {
            stats_map
                .entry(measurement.metric_type.clone())
                .or_insert_with(Vec::new)
                .push(measurement.value);
        }
        
        // Calculate statistics for each metric
        let mut stats = HashMap::new();
        for (metric_type, values) in stats_map {
            if !values.is_empty() {
                let count = values.len();
                let sum: u64 = values.iter().sum();
                let min = *values.iter().min().unwrap();
                let max = *values.iter().max().unwrap();
                let avg = sum as f64 / count as f64;
                
                stats.insert(metric_type, MetricStats {
                    count,
                    sum,
                    min,
                    max,
                    avg,
                });
            }
        }
        
        MetricsSummary {
            stats,
            total_time: self.start_time.elapsed(),
        }
    }
    
    fn clear(&self) {
        self.measurements.lock().unwrap().clear();
    }
}

/// A thread-safe atomic counter collector for high-performance scenarios.
#[derive(Debug)]
pub struct AtomicCollector {
    counters: HashMap<MetricType, Arc<AtomicU64>>,
    start_time: Instant,
}

impl AtomicCollector {
    /// Creates a new atomic collector with predefined counters.
    pub fn new(metrics: Vec<MetricType>) -> Self {
        let mut counters = HashMap::new();
        for metric in metrics {
            counters.insert(metric, Arc::new(AtomicU64::new(0)));
        }
        
        Self {
            counters,
            start_time: Instant::now(),
        }
    }
}

impl PerformanceCollector for AtomicCollector {
    fn record_duration(&self, metric: MetricType, duration: Duration) {
        if let Some(counter) = self.counters.get(&metric) {
            counter.fetch_add(duration.as_nanos() as u64, Ordering::Relaxed);
        }
    }
    
    fn record_count(&self, metric: MetricType, count: u64) {
        if let Some(counter) = self.counters.get(&metric) {
            counter.fetch_add(count, Ordering::Relaxed);
        }
    }
    
    fn summary(&self) -> MetricsSummary {
        let mut stats = HashMap::new();
        
        for (metric_type, counter) in &self.counters {
            let value = counter.load(Ordering::Relaxed);
            if value > 0 {
                // For atomic counters, we only have the total
                stats.insert(metric_type.clone(), MetricStats {
                    count: 1,
                    sum: value,
                    min: value,
                    max: value,
                    avg: value as f64,
                });
            }
        }
        
        MetricsSummary {
            stats,
            total_time: self.start_time.elapsed(),
        }
    }
    
    fn clear(&self) {
        for counter in self.counters.values() {
            counter.store(0, Ordering::Relaxed);
        }
    }
}

/// A timer guard that automatically records duration when dropped.
pub struct TimerGuard<'a> {
    collector: &'a dyn PerformanceCollector,
    metric: MetricType,
    start: Instant,
}

impl<'a> TimerGuard<'a> {
    /// Creates a new timer guard.
    pub fn new(collector: &'a dyn PerformanceCollector, metric: MetricType) -> Self {
        Self {
            collector,
            metric,
            start: Instant::now(),
        }
    }
}

impl<'a> Drop for TimerGuard<'a> {
    fn drop(&mut self) {
        let duration = self.start.elapsed();
        self.collector.record_duration(self.metric.clone(), duration);
    }
}

/// Convenience macro for timing a block of code.
#[macro_export]
macro_rules! time_operation {
    ($collector:expr, $metric:expr, $block:expr) => {{
        let _guard = $crate::profiling::metrics::TimerGuard::new($collector, $metric);
        $block
    }};
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_in_memory_collector() {
        let collector = InMemoryCollector::new();
        
        // Record some measurements
        collector.record_duration(MetricType::CostFunctionTime, Duration::from_millis(10));
        collector.record_duration(MetricType::CostFunctionTime, Duration::from_millis(20));
        collector.record_count(MetricType::FunctionEvals, 5);
        collector.increment(MetricType::GradientEvals);
        
        let summary = collector.summary();
        
        // Check cost function time stats
        let cost_stats = &summary.stats[&MetricType::CostFunctionTime];
        assert_eq!(cost_stats.count, 2);
        assert_eq!(cost_stats.min, 10_000_000); // 10ms in nanoseconds
        assert_eq!(cost_stats.max, 20_000_000); // 20ms in nanoseconds
        
        // Check function evals
        let eval_stats = &summary.stats[&MetricType::FunctionEvals];
        assert_eq!(eval_stats.count, 1);
        assert_eq!(eval_stats.sum, 5);
        
        // Check gradient evals
        let grad_stats = &summary.stats[&MetricType::GradientEvals];
        assert_eq!(grad_stats.count, 1);
        assert_eq!(grad_stats.sum, 1);
    }
    
    #[test]
    fn test_timer_guard() {
        let collector = InMemoryCollector::new();
        
        // Use timer guard
        {
            let _timer = TimerGuard::new(&collector, MetricType::LineSearchTime);
            std::thread::sleep(Duration::from_millis(1));
        }
        
        let summary = collector.summary();
        let stats = &summary.stats[&MetricType::LineSearchTime];
        assert_eq!(stats.count, 1);
        assert!(stats.sum > 1_000_000); // Should be > 1ms in nanoseconds
    }
}