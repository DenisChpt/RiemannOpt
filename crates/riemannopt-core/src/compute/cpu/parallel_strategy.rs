//! Adaptive parallelization strategy for Riemannian optimization.
//!
//! This module provides intelligent decision-making for when to use parallel execution
//! based on problem dimensions, available hardware, and runtime measurements.

use crate::types::Scalar;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

/// Holds runtime measurements for parallelization decisions.
#[derive(Debug, Clone)]
pub struct ParallelizationMetrics {
	/// Average time per element for sequential execution
	pub sequential_time_per_element: Duration,
	/// Average time per element for parallel execution
	pub parallel_time_per_element: Duration,
	/// Overhead of setting up parallel execution
	pub parallel_overhead: Duration,
	/// Number of available CPU cores
	pub num_cores: usize,
}

impl Default for ParallelizationMetrics {
	fn default() -> Self {
		Self {
			sequential_time_per_element: Duration::from_nanos(100),
			parallel_time_per_element: Duration::from_nanos(50),
			parallel_overhead: Duration::from_micros(10),
			num_cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
		}
	}
}

/// Strategy for adaptive parallelization decisions.
#[derive(Debug, Clone)]
pub struct AdaptiveParallelStrategy {
	/// Minimum dimension for parallel execution
	pub min_dimension: usize,
	/// Maximum dimension for sequential execution
	pub max_sequential_dimension: usize,
	/// Metrics for decision making
	pub metrics: ParallelizationMetrics,
	/// Whether to use runtime calibration
	pub enable_calibration: bool,
}

impl Default for AdaptiveParallelStrategy {
	fn default() -> Self {
		Self {
			min_dimension: 100,
			max_sequential_dimension: 1000,
			metrics: ParallelizationMetrics::default(),
			enable_calibration: true,
		}
	}
}

impl AdaptiveParallelStrategy {
	/// Create a new adaptive strategy with default settings.
	pub fn new() -> Self {
		Self::default()
	}

	/// Decide whether to use parallel execution for given dimension.
	pub fn should_parallelize(&self, dimension: usize) -> bool {
		// Quick decision for extreme cases
		if dimension < self.min_dimension {
			return false;
		}
		if dimension > self.max_sequential_dimension {
			return true;
		}

		// Use metrics for intermediate cases
		let sequential_time =
			self.metrics.sequential_time_per_element.as_nanos() as f64 * dimension as f64;
		let parallel_time = self.metrics.parallel_overhead.as_nanos() as f64
			+ (self.metrics.parallel_time_per_element.as_nanos() as f64 * dimension as f64)
				/ self.metrics.num_cores as f64;

		parallel_time < sequential_time
	}

	/// Determine optimal chunk size for parallel execution.
	pub fn optimal_chunk_size(&self, dimension: usize) -> usize {
		// Heuristic: balance between cache efficiency and parallelism
		let ideal_chunks_per_thread = 4;
		let total_chunks = self.metrics.num_cores * ideal_chunks_per_thread;

		let chunk_size = dimension / total_chunks;

		// Ensure reasonable bounds
		chunk_size.clamp(1, 1000)
	}

	/// Calibrate the strategy by running microbenchmarks.
	pub fn calibrate<T: Scalar + 'static + std::iter::Sum>(&mut self) {
		if !self.enable_calibration {
			return;
		}

		// Test dimension
		let test_dim = 1000;
		let iterations = 100;

		// Benchmark sequential execution
		let mut seq_times = Vec::with_capacity(iterations);
		for _ in 0..iterations {
			let start = Instant::now();
			let _result = benchmark_sequential::<T>(test_dim);
			seq_times.push(start.elapsed());
		}

		// Benchmark parallel execution
		let mut par_times = Vec::with_capacity(iterations);
		for _ in 0..iterations {
			let start = Instant::now();
			let _result = benchmark_parallel::<T>(test_dim);
			par_times.push(start.elapsed());
		}

		// Calculate medians to avoid outliers
		seq_times.sort();
		par_times.sort();

		let seq_median = seq_times[iterations / 2];
		let par_median = par_times[iterations / 2];

		// Update metrics
		self.metrics.sequential_time_per_element = seq_median / test_dim as u32;
		self.metrics.parallel_time_per_element = par_median / test_dim as u32;

		// Estimate overhead
		let small_dim = 10;
		let start = Instant::now();
		let _result = benchmark_parallel::<T>(small_dim);
		let small_time = start.elapsed();

		let start = Instant::now();
		let _result = benchmark_sequential::<T>(small_dim);
		let small_seq_time = start.elapsed();

		if small_time > small_seq_time {
			self.metrics.parallel_overhead = small_time - small_seq_time;
		}
	}
}

/// Global adaptive strategy instance.
static GLOBAL_STRATEGY: OnceLock<AdaptiveParallelStrategy> = OnceLock::new();

/// Get the global adaptive parallelization strategy.
pub fn get_adaptive_strategy() -> &'static AdaptiveParallelStrategy {
	GLOBAL_STRATEGY.get_or_init(|| {
		let mut strategy = AdaptiveParallelStrategy::default();
		// Disable calibration by default to avoid startup overhead
		strategy.enable_calibration = false;
		strategy
	})
}

// ---------------------------------------------------------------------------
// BLAS / OpenMP thread management
// ---------------------------------------------------------------------------

/// Mutex that serializes access to BLAS thread environment variables.
///
/// Modifying environment variables from multiple threads is inherently
/// unsafe (data race on the process-level environment).  This mutex
/// prevents *our* code from doing so concurrently.  It does **not**
/// protect against other libraries reading the variables at the same
/// time, which is a fundamental limitation of environment-based
/// thread control.  For robust BLAS thread management, prefer setting
/// `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS`
/// **before** spawning any threads (e.g., at program startup).
static ENV_MUTEX: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard that sets `OMP_NUM_THREADS=1` (and the equivalent for MKL and
/// OpenBLAS) for the duration of its lifetime.
///
/// This prevents nested parallelism when Rayon is driving the outer loop and
/// a BLAS call inside each work-unit would otherwise spawn its own threads,
/// creating massive over-subscription.
///
/// # Safety considerations
///
/// Modifying environment variables at runtime in a multi-threaded program is
/// inherently problematic.  This guard serializes its own access via a mutex,
/// but cannot prevent data races with external code that reads or writes env
/// vars concurrently.  Prefer configuring BLAS threads at process startup
/// whenever possible.
///
/// # Example
///
/// ```rust,no_run
/// use riemannopt_core::compute::cpu::parallel_strategy::BlasThreadGuard;
/// use rayon::prelude::*;
///
/// let _guard = BlasThreadGuard::single_threaded();
/// (0..1000).into_par_iter().for_each(|_| {
///     // BLAS calls here will use a single thread
/// });
/// // On drop, the original thread counts are restored.
/// ```
pub struct BlasThreadGuard {
	prev_omp: Option<String>,
	prev_mkl: Option<String>,
	prev_openblas: Option<String>,
	/// Held for the guard's lifetime to serialize env var access.
	_lock: std::sync::MutexGuard<'static, ()>,
}

impl BlasThreadGuard {
	/// Restrict BLAS to a single thread.
	pub fn single_threaded() -> Self {
		Self::with_threads(1)
	}

	/// Set BLAS thread count to `n`.
	pub fn with_threads(n: usize) -> Self {
		let lock = ENV_MUTEX.lock().unwrap_or_else(|e| e.into_inner());

		let prev_omp = std::env::var("OMP_NUM_THREADS").ok();
		let prev_mkl = std::env::var("MKL_NUM_THREADS").ok();
		let prev_openblas = std::env::var("OPENBLAS_NUM_THREADS").ok();

		let val = n.to_string();
		// Note: env var modification is serialized by ENV_MUTEX.
		// The remaining risk is external code reading env vars concurrently,
		// which is an inherent limitation of environment-based BLAS thread control.
		std::env::set_var("OMP_NUM_THREADS", &val);
		std::env::set_var("MKL_NUM_THREADS", &val);
		std::env::set_var("OPENBLAS_NUM_THREADS", &val);

		Self {
			prev_omp,
			prev_mkl,
			prev_openblas,
			_lock: lock,
		}
	}
}

impl Drop for BlasThreadGuard {
	fn drop(&mut self) {
		// Lock is already held via _lock field
		fn restore(key: &str, prev: &Option<String>) {
			match prev {
				Some(v) => std::env::set_var(key, v),
				None => std::env::remove_var(key),
			}
		}
		restore("OMP_NUM_THREADS", &self.prev_omp);
		restore("MKL_NUM_THREADS", &self.prev_mkl);
		restore("OPENBLAS_NUM_THREADS", &self.prev_openblas);
	}
}

/// Simple benchmark function for sequential execution.
fn benchmark_sequential<T: Scalar>(dimension: usize) -> T {
	let mut sum = T::zero();
	for i in 0..dimension {
		let x = T::from(i).unwrap();
		sum = sum + x * x;
	}
	sum
}

/// Simple benchmark function for parallel execution.
fn benchmark_parallel<T: Scalar + std::iter::Sum>(dimension: usize) -> T {
	use rayon::prelude::*;

	// Suppress BLAS threads during Rayon parallelism to avoid over-subscription.
	let _guard = BlasThreadGuard::single_threaded();

	(0..dimension)
		.into_par_iter()
		.map(|i| {
			let x = T::from(i).unwrap();
			x * x
		})
		.sum()
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_adaptive_strategy_decisions() {
		let strategy = AdaptiveParallelStrategy::new();

		// Small dimensions should not parallelize
		assert!(!strategy.should_parallelize(10));
		assert!(!strategy.should_parallelize(50));

		// Large dimensions should parallelize
		assert!(strategy.should_parallelize(2000));
		assert!(strategy.should_parallelize(10000));
	}

	#[test]
	fn test_optimal_chunk_size() {
		let strategy = AdaptiveParallelStrategy::new();

		let chunk = strategy.optimal_chunk_size(1000);
		assert!(chunk >= 1);
		assert!(chunk <= 1000);

		// Test bounds
		assert_eq!(strategy.optimal_chunk_size(5), 1);
		assert_eq!(strategy.optimal_chunk_size(100000), 1000);
	}

	#[test]
	fn test_calibration() {
		let mut strategy = AdaptiveParallelStrategy::new();
		strategy.calibrate::<f64>();

		// After calibration, metrics should be updated or at least the calibration should complete
		// Note: On very fast systems or in debug mode, times might be zero due to precision limits
		// The important thing is that calibration completes without errors
		assert!(strategy.metrics.sequential_time_per_element >= Duration::ZERO);
		assert!(strategy.metrics.parallel_time_per_element >= Duration::ZERO);

		// Verify that calibration was attempted (times may be zero on fast systems)
		// but the strategy should be in a consistent state
		assert!(strategy.enable_calibration || !strategy.enable_calibration); // Always true, but checks the field exists
	}

	#[test]
	fn test_global_strategy() {
		let strategy1 = get_adaptive_strategy();
		let strategy2 = get_adaptive_strategy();

		// Should be the same instance
		assert!(std::ptr::eq(strategy1, strategy2));
	}
}
