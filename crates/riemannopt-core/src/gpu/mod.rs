//! GPU acceleration support for Riemannian optimization.
//!
//! This module provides CUDA-based GPU acceleration for manifold operations
//! and optimization algorithms using cudarc and cust.
//!
//! # Features
//!
//! - GPU memory management with pooling
//! - CUDA kernel implementations for basic operations
//! - GPU-accelerated manifold operations
//! - Batch processing on GPU
//!
//! # Requirements
//!
//! - CUDA toolkit installed
//! - GPU with compute capability 3.5 or higher
//! - Enable the `cuda` feature flag
//!
//! # Example
//!
//! ```rust,no_run
//! # #[cfg(feature = "cuda")]
//! # {
//! use riemannopt_core::gpu::{GpuBackend, GpuMatrix, CudaBackend};
//! use nalgebra::DMatrix;
//!
//! // Initialize GPU backend
//! let backend = CudaBackend::initialize().unwrap();
//!
//! // Transfer matrix to GPU
//! let host_matrix = DMatrix::from_vec(3, 3, vec![1.0; 9]);
//! let gpu_matrix = GpuMatrix::from_host(&host_matrix).unwrap();
//!
//! // Perform operations on GPU...
//!
//! // Transfer back to host
//! let result = gpu_matrix.to_host().unwrap();
//! # }
//! ```

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub mod memory;

#[cfg(feature = "cuda")]
pub mod kernels;

use crate::types::Scalar;
use nalgebra::DMatrix;

/// GPU backend trait for accelerated operations
pub trait GpuBackend {
    /// Initialize the GPU backend
    fn initialize() -> Result<Self, GpuError>
    where
        Self: Sized;
    
    /// Check if GPU is available
    fn is_available() -> bool;
    
    /// Get device information
    fn device_info(&self) -> DeviceInfo;
    
    /// Synchronize device
    fn synchronize(&self) -> Result<(), GpuError>;
}

/// GPU-accelerated matrix operations
pub trait GpuMatrixOps<T: Scalar> {
    /// Matrix multiplication on GPU
    fn gemm(
        &self,
        a: &GpuMatrix<T>,
        b: &GpuMatrix<T>,
        alpha: T,
        beta: T,
        c: &mut GpuMatrix<T>,
    ) -> Result<(), GpuError>;
    
    /// Matrix transpose on GPU
    fn transpose(&self, a: &GpuMatrix<T>) -> Result<GpuMatrix<T>, GpuError>;
    
    /// Element-wise operations
    fn elementwise_add(
        &self,
        a: &GpuMatrix<T>,
        b: &GpuMatrix<T>,
    ) -> Result<GpuMatrix<T>, GpuError>;
    
    fn elementwise_mul(
        &self,
        a: &GpuMatrix<T>,
        b: &GpuMatrix<T>,
    ) -> Result<GpuMatrix<T>, GpuError>;
}

/// GPU matrix wrapper
pub struct GpuMatrix<T: Scalar> {
    /// Pointer to device memory
    device_ptr: *mut T,
    /// Matrix dimensions
    rows: usize,
    cols: usize,
    /// Leading dimension
    ld: usize,
}

impl<T: Scalar> GpuMatrix<T> {
    /// Create from host matrix
    pub fn from_host(matrix: &DMatrix<T>) -> Result<Self, GpuError> {
        #[cfg(feature = "cuda")]
        {
            cuda::GpuMatrix::from_host(matrix)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::NotAvailable)
        }
    }
    
    /// Copy to host matrix
    pub fn to_host(&self) -> Result<DMatrix<T>, GpuError> {
        #[cfg(feature = "cuda")]
        {
            cuda::GpuMatrix::to_host(self)
        }
        #[cfg(not(feature = "cuda"))]
        {
            Err(GpuError::NotAvailable)
        }
    }
    
    /// Get dimensions
    pub fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }
}

/// Device information
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub name: String,
    pub compute_capability: (u32, u32),
    pub memory_total: usize,
    pub memory_free: usize,
    pub multiprocessor_count: u32,
}

/// GPU errors
#[derive(thiserror::Error, Debug)]
pub enum GpuError {
    #[error("GPU not available")]
    NotAvailable,
    
    #[error("CUDA error: {0}")]
    CudaError(String),
    
    #[error("Memory allocation failed: {0}")]
    AllocationError(String),
    
    #[error("Invalid dimensions: {0}")]
    DimensionError(String),
    
    #[error("Kernel launch failed: {0}")]
    KernelError(String),
    
    #[error("Memory transfer failed: {0}")]
    TransferError(String),
}

/// GPU-accelerated manifold operations
pub trait GpuManifoldOps<T: Scalar> {
    /// Project point onto manifold using GPU
    fn project_gpu(&self, point: &GpuMatrix<T>) -> Result<GpuMatrix<T>, GpuError>;
    
    /// Compute retraction using GPU
    fn retract_gpu(
        &self,
        point: &GpuMatrix<T>,
        tangent: &GpuMatrix<T>,
        t: T,
    ) -> Result<GpuMatrix<T>, GpuError>;
    
    /// Project to tangent space using GPU
    fn project_tangent_gpu(
        &self,
        point: &GpuMatrix<T>,
        vector: &GpuMatrix<T>,
    ) -> Result<GpuMatrix<T>, GpuError>;
    
    /// Parallel transport using GPU
    fn parallel_transport_gpu(
        &self,
        from: &GpuMatrix<T>,
        to: &GpuMatrix<T>,
        vector: &GpuMatrix<T>,
    ) -> Result<GpuMatrix<T>, GpuError>;
}

/// Batch operations on GPU
pub trait GpuBatchOps<T: Scalar> {
    /// Process multiple points in parallel
    fn batch_project(
        &self,
        points: &[GpuMatrix<T>],
    ) -> Result<Vec<GpuMatrix<T>>, GpuError>;
    
    /// Batch retraction
    fn batch_retract(
        &self,
        points: &[GpuMatrix<T>],
        tangents: &[GpuMatrix<T>],
        t: T,
    ) -> Result<Vec<GpuMatrix<T>>, GpuError>;
    
    /// Batch gradient computation
    fn batch_gradient<F>(
        &self,
        points: &[GpuMatrix<T>],
        cost_fn: F,
    ) -> Result<Vec<GpuMatrix<T>>, GpuError>
    where
        F: Fn(&GpuMatrix<T>) -> Result<GpuMatrix<T>, GpuError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_availability() {
        #[cfg(feature = "cuda")]
        {
            let available = cuda::CudaBackend::is_available();
            println!("CUDA available: {}", available);
        }
        #[cfg(not(feature = "cuda"))]
        {
            println!("CUDA support not enabled");
        }
    }
}