//! Auto-tuning system for optimization parameters.
//!
//! This module provides functionality to automatically tune optimization
//! parameters based on observed performance characteristics.

use crate::{
    profiling::metrics::{PerformanceCollector, MetricType},
    compute::cpu::parallel::ParallelConfig,
};
use std::collections::HashMap;
use std::time::Duration;

/// Parameters that can be auto-tuned.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum TunableParameter {
    /// Chunk size for SIMD operations
    SimdChunkSize,
    /// Threshold for enabling parallel operations
    ParallelThreshold,
    /// Number of threads for parallel operations
    ThreadCount,
    /// Cache size for memoization
    CacheSize,
    /// Line search initial step size
    InitialStepSize,
    /// Custom parameter
    Custom(String),
}

/// Result of an auto-tuning run.
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// The parameter that was tuned
    pub parameter: TunableParameter,
    /// The original value
    pub original_value: f64,
    /// The optimal value found
    pub optimal_value: f64,
    /// Performance improvement (as a ratio, 1.0 = no improvement)
    pub improvement: f64,
    /// Time taken to tune
    pub tuning_time: Duration,
}

/// Trait for auto-tunable components.
pub trait AutoTunable {
    /// Returns the list of tunable parameters.
    fn tunable_parameters(&self) -> Vec<TunableParameter>;
    
    /// Gets the current value of a parameter.
    fn get_parameter(&self, param: &TunableParameter) -> Option<f64>;
    
    /// Sets the value of a parameter.
    fn set_parameter(&mut self, param: &TunableParameter, value: f64) -> Result<(), String>;
    
    /// Runs a benchmark with the current parameters.
    fn benchmark(&self, collector: &dyn PerformanceCollector) -> Result<f64, String>;
}

/// Auto-tuner that optimizes parameters based on benchmarking.
pub struct AutoTuner {
    /// Number of iterations for each parameter value
    pub benchmark_iterations: usize,
    /// Relative tolerance for considering improvements
    pub improvement_tolerance: f64,
    /// Maximum time to spend tuning (per parameter)
    pub max_tuning_time: Duration,
}

impl AutoTuner {
    /// Creates a new auto-tuner with default settings.
    pub fn new() -> Self {
        Self {
            benchmark_iterations: 10,
            improvement_tolerance: 0.05, // 5% improvement threshold
            max_tuning_time: Duration::from_secs(30),
        }
    }
    
    /// Tunes a single parameter using grid search.
    pub fn tune_parameter<T: AutoTunable>(
        &self,
        component: &mut T,
        parameter: &TunableParameter,
        values: &[f64],
        collector: &dyn PerformanceCollector,
    ) -> Result<TuningResult, String> {
        let start_time = std::time::Instant::now();
        
        // Get original value
        let original_value = component.get_parameter(parameter)
            .ok_or_else(|| format!("Parameter {:?} not found", parameter))?;
        
        // Benchmark with original value
        collector.clear();
        let mut original_score = 0.0;
        for _ in 0..self.benchmark_iterations {
            original_score += component.benchmark(collector)?;
        }
        original_score /= self.benchmark_iterations as f64;
        
        let mut best_value = original_value;
        let mut best_score = original_score;
        
        // Try each value
        for &value in values {
            if start_time.elapsed() > self.max_tuning_time {
                break;
            }
            
            // Set parameter
            component.set_parameter(parameter, value)?;
            
            // Benchmark
            collector.clear();
            let mut score = 0.0;
            for _ in 0..self.benchmark_iterations {
                score += component.benchmark(collector)?;
            }
            score /= self.benchmark_iterations as f64;
            
            // Check if better (lower is better for timing)
            if score < best_score {
                best_score = score;
                best_value = value;
            }
        }
        
        // Restore best value
        component.set_parameter(parameter, best_value)?;
        
        let improvement = original_score / best_score;
        
        Ok(TuningResult {
            parameter: parameter.clone(),
            original_value,
            optimal_value: best_value,
            improvement,
            tuning_time: start_time.elapsed(),
        })
    }
    
    /// Tunes all parameters of a component.
    pub fn tune_all<T: AutoTunable>(
        &self,
        component: &mut T,
        parameter_values: &HashMap<TunableParameter, Vec<f64>>,
        collector: &dyn PerformanceCollector,
    ) -> Vec<TuningResult> {
        let mut results = Vec::new();
        
        for parameter in component.tunable_parameters() {
            if let Some(values) = parameter_values.get(&parameter) {
                match self.tune_parameter(component, &parameter, values, collector) {
                    Ok(result) => {
                        if result.improvement > 1.0 + self.improvement_tolerance {
                            println!("Tuned {:?}: {:.2} -> {:.2} ({}% improvement)",
                                parameter,
                                result.original_value,
                                result.optimal_value,
                                (result.improvement - 1.0) * 100.0
                            );
                        }
                        results.push(result);
                    }
                    Err(e) => {
                        eprintln!("Failed to tune {:?}: {}", parameter, e);
                    }
                }
            }
        }
        
        results
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

/// Example implementation for ParallelConfig
impl AutoTunable for ParallelConfig {
    fn tunable_parameters(&self) -> Vec<TunableParameter> {
        vec![
            TunableParameter::ThreadCount,
            TunableParameter::ParallelThreshold,
        ]
    }
    
    fn get_parameter(&self, param: &TunableParameter) -> Option<f64> {
        match param {
            TunableParameter::ThreadCount => {
                // If None, return the default number of threads (rayon's default)
                Some(self.num_threads.unwrap_or(rayon::current_num_threads()) as f64)
            },
            TunableParameter::ParallelThreshold => Some(self.min_dimension_for_parallel as f64),
            _ => None,
        }
    }
    
    fn set_parameter(&mut self, param: &TunableParameter, value: f64) -> Result<(), String> {
        match param {
            TunableParameter::ThreadCount => {
                if value < 1.0 {
                    return Err("Thread count must be at least 1".to_string());
                }
                self.num_threads = Some(value as usize);
                Ok(())
            }
            TunableParameter::ParallelThreshold => {
                if value < 0.0 {
                    return Err("Threshold must be non-negative".to_string());
                }
                self.min_dimension_for_parallel = value as usize;
                Ok(())
            }
            _ => Err(format!("Parameter {:?} not supported", param)),
        }
    }
    
    fn benchmark(&self, collector: &dyn PerformanceCollector) -> Result<f64, String> {
        // This would run an actual benchmark
        // For now, return a dummy value
        use std::time::Instant;
        let start = Instant::now();
        
        // Simulate some work
        std::thread::sleep(Duration::from_micros(10));
        
        let elapsed = start.elapsed();
        collector.record_duration(MetricType::Custom("benchmark".to_string()), elapsed);
        
        Ok(elapsed.as_secs_f64())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profiling::metrics::InMemoryCollector;
    
    #[test]
    fn test_auto_tuner() {
        let mut config = ParallelConfig::default();
        // Set an initial thread count so we have something to tune from
        config.num_threads = Some(4);
        
        let collector = InMemoryCollector::new();
        let tuner = AutoTuner::new();
        
        // Tune thread count
        let thread_counts = vec![1.0, 2.0, 4.0, 8.0];
        let result = tuner.tune_parameter(
            &mut config,
            &TunableParameter::ThreadCount,
            &thread_counts,
            &collector,
        ).unwrap();
        
        assert_eq!(result.parameter, TunableParameter::ThreadCount);
        assert!(thread_counts.contains(&result.optimal_value));
    }
}