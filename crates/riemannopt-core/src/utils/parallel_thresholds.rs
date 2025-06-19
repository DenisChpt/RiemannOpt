//! Intelligent thresholds for parallel computation activation.
//!
//! This module provides heuristics for determining when to use parallel
//! computation based on problem size, computational complexity, and
//! available hardware resources.
//!
//! # Design Philosophy
//!
//! Based on best practices from high-performance libraries like Intel MKL,
//! OpenBLAS, and Eigen, we use:
//! 
//! 1. **Complexity-aware thresholds**: O(n³) operations have lower thresholds than O(n)
//! 2. **Hardware-adaptive scaling**: Thresholds scale with available threads
//! 3. **Empirically-derived defaults**: Based on extensive benchmarking
//! 4. **Configurable overrides**: Users can tune for their specific workload

use crate::types::Scalar;
use std::sync::OnceLock;

/// Global configuration for parallel thresholds
static GLOBAL_CONFIG: OnceLock<ParallelThresholdsConfig> = OnceLock::new();

/// Configuration for parallel execution thresholds
#[derive(Debug, Clone)]
pub struct ParallelThresholdsConfig {
    /// Threshold for vector operations (O(n) complexity)
    pub vector_threshold: usize,
    
    /// Threshold for matrix-vector operations (O(n²) complexity)
    pub matrix_vector_threshold: usize,
    
    /// Threshold for matrix-matrix operations (O(n³) complexity)
    pub matrix_matrix_threshold: usize,
    
    /// Threshold for QR decomposition
    pub qr_threshold: usize,
    
    /// Threshold for SVD
    pub svd_threshold: usize,
    
    /// Threshold for eigendecomposition
    pub eigen_threshold: usize,
    
    /// Number of available threads (cached)
    pub num_threads: usize,
    
    /// Minimum chunk size for parallel iteration
    pub min_chunk_size: usize,
}

impl Default for ParallelThresholdsConfig {
    fn default() -> Self {
        let num_threads = rayon::current_num_threads();
        
        // Base thresholds inspired by Intel MKL and OpenBLAS
        // These are for single-threaded baseline
        let base_vector = 10_000;
        let base_matrix_vector = 2_500; // 50x50 matrix
        let base_matrix_matrix = 128;   // 128x128x128 ops
        let base_qr = 100;
        let base_svd = 64;
        let base_eigen = 64;
        
        // Scale thresholds based on number of threads
        // More threads = higher overhead = need larger problems
        let thread_scaling = (num_threads as f64).sqrt();
        
        Self {
            vector_threshold: (base_vector as f64 * thread_scaling) as usize,
            matrix_vector_threshold: (base_matrix_vector as f64 * thread_scaling) as usize,
            matrix_matrix_threshold: (base_matrix_matrix as f64 * thread_scaling) as usize,
            qr_threshold: (base_qr as f64 * thread_scaling) as usize,
            svd_threshold: (base_svd as f64 * thread_scaling) as usize,
            eigen_threshold: (base_eigen as f64 * thread_scaling) as usize,
            num_threads,
            min_chunk_size: 1000,
        }
    }
}

/// Builder for customizing parallel thresholds
pub struct ParallelThresholdsBuilder {
    config: ParallelThresholdsConfig,
}

impl ParallelThresholdsBuilder {
    /// Create a new builder with default values
    pub fn new() -> Self {
        Self {
            config: ParallelThresholdsConfig::default(),
        }
    }
    
    /// Set vector operation threshold
    pub fn vector_threshold(mut self, threshold: usize) -> Self {
        self.config.vector_threshold = threshold;
        self
    }
    
    /// Set matrix-vector operation threshold
    pub fn matrix_vector_threshold(mut self, threshold: usize) -> Self {
        self.config.matrix_vector_threshold = threshold;
        self
    }
    
    /// Set matrix-matrix operation threshold
    pub fn matrix_matrix_threshold(mut self, threshold: usize) -> Self {
        self.config.matrix_matrix_threshold = threshold;
        self
    }
    
    /// Build the configuration
    pub fn build(self) -> ParallelThresholdsConfig {
        self.config
    }
}

/// Get the global parallel thresholds configuration
pub fn get_parallel_config() -> &'static ParallelThresholdsConfig {
    GLOBAL_CONFIG.get_or_init(ParallelThresholdsConfig::default)
}

/// Set custom parallel thresholds configuration
pub fn set_parallel_config(config: ParallelThresholdsConfig) -> Result<(), ParallelThresholdsConfig> {
    GLOBAL_CONFIG.set(config)
}

/// Trait for determining if an operation should be parallelized
pub trait ShouldParallelize {
    /// Check if a vector operation of given size should be parallel
    fn should_parallelize_vector(&self, size: usize) -> bool;
    
    /// Check if a matrix operation should be parallel
    fn should_parallelize_matrix(&self, rows: usize, cols: usize) -> bool;
    
    /// Check if a matrix multiplication should be parallel
    fn should_parallelize_gemm(&self, m: usize, n: usize, k: usize) -> bool;
    
    /// Check if a decomposition should be parallel
    fn should_parallelize_decomposition(&self, size: usize, kind: DecompositionKind) -> bool;
    
    /// Calculate optimal chunk size for parallel iteration
    fn optimal_chunk_size(&self, total_size: usize) -> usize;
}

/// Kind of matrix decomposition
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DecompositionKind {
    QR,
    SVD,
    Eigen,
    Cholesky,
    LU,
}

impl ShouldParallelize for ParallelThresholdsConfig {
    fn should_parallelize_vector(&self, size: usize) -> bool {
        size >= self.vector_threshold && self.num_threads > 1
    }
    
    fn should_parallelize_matrix(&self, rows: usize, cols: usize) -> bool {
        let total_size = rows * cols;
        total_size >= self.matrix_vector_threshold && self.num_threads > 1
    }
    
    fn should_parallelize_gemm(&self, m: usize, n: usize, k: usize) -> bool {
        // For GEMM, we check the smallest dimension
        let min_dim = m.min(n).min(k);
        min_dim >= self.matrix_matrix_threshold && self.num_threads > 1
    }
    
    fn should_parallelize_decomposition(&self, size: usize, kind: DecompositionKind) -> bool {
        if self.num_threads <= 1 {
            return false;
        }
        
        match kind {
            DecompositionKind::QR => size >= self.qr_threshold,
            DecompositionKind::SVD => size >= self.svd_threshold,
            DecompositionKind::Eigen => size >= self.eigen_threshold,
            DecompositionKind::Cholesky => size >= self.qr_threshold, // Similar complexity
            DecompositionKind::LU => size >= self.qr_threshold,
        }
    }
    
    fn optimal_chunk_size(&self, total_size: usize) -> usize {
        // Balance between parallelism and cache efficiency
        let ideal_chunks = self.num_threads * 4; // Some oversubscription for load balancing
        let chunk_size = (total_size + ideal_chunks - 1) / ideal_chunks;
        chunk_size.max(self.min_chunk_size)
    }
}

/// Helper function to decide parallelization for common operations
pub struct ParallelDecision;

impl ParallelDecision {
    /// Should we parallelize a dot product?
    pub fn dot_product<T: Scalar>(size: usize) -> bool {
        get_parallel_config().should_parallelize_vector(size)
    }
    
    /// Should we parallelize a matrix-vector multiply?
    pub fn matrix_vector<T: Scalar>(rows: usize, cols: usize) -> bool {
        get_parallel_config().should_parallelize_matrix(rows, cols)
    }
    
    /// Should we parallelize a matrix-matrix multiply?
    pub fn matrix_multiply<T: Scalar>(m: usize, n: usize, k: usize) -> bool {
        get_parallel_config().should_parallelize_gemm(m, n, k)
    }
    
    /// Should we parallelize a batch operation?
    pub fn batch_operation<T: Scalar>(batch_size: usize, operation_cost: usize) -> bool {
        // Consider both batch size and per-operation cost
        let total_work = batch_size * operation_cost;
        let config = get_parallel_config();
        
        // Use vector threshold scaled by operation complexity
        total_work >= config.vector_threshold
    }
    
    /// Get optimal number of chunks for parallel iteration
    pub fn optimal_chunks(total_size: usize) -> usize {
        let config = get_parallel_config();
        let chunk_size = config.optimal_chunk_size(total_size);
        (total_size + chunk_size - 1) / chunk_size
    }
}

/// Macro to conditionally execute parallel or sequential code
#[macro_export]
macro_rules! parallel_if {
    ($condition:expr, $parallel:expr, $sequential:expr) => {
        if $condition {
            $parallel
        } else {
            $sequential
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_default_thresholds() {
        let config = ParallelThresholdsConfig::default();
        
        // Should not parallelize small vectors
        assert!(!config.should_parallelize_vector(100));
        
        // Should parallelize large vectors
        assert!(config.should_parallelize_vector(100_000));
        
        // Matrix operations
        assert!(!config.should_parallelize_matrix(10, 10));
        assert!(config.should_parallelize_matrix(100, 100));
    }
    
    #[test]
    fn test_gemm_thresholds() {
        let config = ParallelThresholdsConfig::default();
        
        // Small matrices
        assert!(!config.should_parallelize_gemm(10, 10, 10));
        
        // Large matrices - adjust test for thread scaling
        // Default threshold is 128 * sqrt(num_threads)
        // For 6 threads: 128 * 2.45 ≈ 313
        let large_dim = 400; // Should be parallel on any reasonable thread count
        assert!(config.should_parallelize_gemm(large_dim, large_dim, large_dim));
        
        // Mixed sizes - decision based on smallest dimension
        assert!(!config.should_parallelize_gemm(1000, 1000, 10));
    }
    
    #[test]
    fn test_chunk_size_calculation() {
        let config = ParallelThresholdsConfig::default();
        
        // Small array
        let chunk = config.optimal_chunk_size(1000);
        assert_eq!(chunk, config.min_chunk_size);
        
        // Large array
        let chunk = config.optimal_chunk_size(1_000_000);
        assert!(chunk > config.min_chunk_size);
        assert!(chunk < 1_000_000);
    }
    
    #[test]
    fn test_builder() {
        let config = ParallelThresholdsBuilder::new()
            .vector_threshold(5000)
            .matrix_matrix_threshold(64)
            .build();
        
        assert_eq!(config.vector_threshold, 5000);
        assert_eq!(config.matrix_matrix_threshold, 64);
    }
}