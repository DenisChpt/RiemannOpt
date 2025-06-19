//! Traits for GPU operations.
//!
//! This module defines the traits that GPU backends must implement
//! to integrate with the optimization library.

use crate::{
    compute::backend::ComputeBackend,
    error::Result,
    types::Scalar,
};
use nalgebra::DVector;
use std::fmt::Debug;

/// Trait for GPU-specific operations.
pub trait GpuBackend<T: Scalar>: ComputeBackend<T> {
    /// Gets the device index this backend is using.
    fn device_index(&self) -> usize;
    
    /// Gets the total device memory in bytes.
    fn total_memory(&self) -> usize;
    
    /// Gets the available device memory in bytes.
    fn available_memory(&self) -> usize;
    
    /// Synchronizes the device (waits for all operations to complete).
    fn synchronize(&self) -> Result<()>;
    
    /// Allocates device memory.
    fn allocate(&self, size: usize) -> Result<GpuBuffer<T>>;
    
    /// Transfers data to the device.
    fn upload(&self, data: &[T], buffer: &mut GpuBuffer<T>) -> Result<()>;
    
    /// Transfers data from the device.
    fn download(&self, buffer: &GpuBuffer<T>, data: &mut [T]) -> Result<()>;
    
    /// Checks if the backend supports a specific precision.
    fn supports_precision(&self) -> bool;
}

/// A buffer on the GPU device.
#[derive(Debug)]
pub struct GpuBuffer<T> {
    /// Pointer to device memory (opaque)
    ptr: *mut u8,
    /// Size in elements
    size: usize,
    /// Device index
    device: usize,
    _phantom: std::marker::PhantomData<T>,
}

// Safety: GpuBuffer can be sent between threads
unsafe impl<T> Send for GpuBuffer<T> {}
unsafe impl<T> Sync for GpuBuffer<T> {}

impl<T> GpuBuffer<T> {
    /// Creates a new GPU buffer (used by backend implementations).
    pub fn new(ptr: *mut u8, size: usize, device: usize) -> Self {
        Self {
            ptr,
            size,
            device,
            _phantom: std::marker::PhantomData,
        }
    }
    
    /// Gets the device pointer.
    pub fn as_ptr(&self) -> *mut u8 {
        self.ptr
    }
    
    /// Gets the size in elements.
    pub fn size(&self) -> usize {
        self.size
    }
    
    /// Gets the device index.
    pub fn device(&self) -> usize {
        self.device
    }
}

/// Trait for GPU kernel operations.
pub trait GpuKernels<T: Scalar> {
    /// Launches a custom kernel.
    fn launch_kernel(
        &self,
        kernel_name: &str,
        grid_size: (u32, u32, u32),
        block_size: (u32, u32, u32),
        args: &[GpuKernelArg],
    ) -> Result<()>;
    
    /// Compiles a kernel from source.
    fn compile_kernel(&self, source: &str, kernel_name: &str) -> Result<()>;
    
    /// Checks if a kernel is available.
    fn has_kernel(&self, kernel_name: &str) -> bool;
}

/// Arguments for GPU kernels.
#[derive(Debug)]
pub enum GpuKernelArg<'a> {
    /// Scalar value
    Scalar(f64),
    /// Integer value
    Int(i32),
    /// Unsigned integer value
    UInt(u32),
    /// GPU buffer
    Buffer(&'a GpuBuffer<f64>),
    /// Size value
    Size(usize),
}

/// Extension trait for GPU-accelerated manifold operations.
pub trait GpuManifoldOps<T: Scalar, const D: usize> {
    /// Performs batch retraction on GPU.
    fn gpu_batch_retract(
        &self,
        backend: &dyn GpuBackend<T>,
        points: &[DVector<T>],
        tangents: &[DVector<T>],
        t: T,
    ) -> Result<Vec<DVector<T>>>;
    
    /// Performs batch projection on GPU.
    fn gpu_batch_project(
        &self,
        backend: &dyn GpuBackend<T>,
        points: &[DVector<T>],
        vectors: &mut [DVector<T>],
    ) -> Result<()>;
    
    /// Computes batch inner products on GPU.
    fn gpu_batch_inner_product(
        &self,
        backend: &dyn GpuBackend<T>,
        points: &[DVector<T>],
        v1s: &[DVector<T>],
        v2s: &[DVector<T>],
    ) -> Result<Vec<T>>;
}

/// Memory pool for GPU allocations.
pub trait GpuMemoryPool {
    /// Acquires a buffer from the pool.
    fn acquire<T: Scalar>(&self, size: usize) -> Result<GpuBuffer<T>>;
    
    /// Returns a buffer to the pool.
    fn release<T: Scalar>(&self, buffer: GpuBuffer<T>);
    
    /// Gets the total allocated memory.
    fn allocated_memory(&self) -> usize;
    
    /// Gets the number of active allocations.
    fn active_allocations(&self) -> usize;
    
    /// Clears the pool, releasing all memory.
    fn clear(&self);
}

/// Configuration for GPU operations.
#[derive(Debug, Clone)]
pub struct GpuConfig {
    /// Maximum memory to use (in bytes)
    pub max_memory: Option<usize>,
    /// Preferred block size for kernels
    pub block_size: u32,
    /// Whether to use persistent kernels
    pub use_persistent_kernels: bool,
    /// Whether to use tensor cores (if available)
    pub use_tensor_cores: bool,
    /// Stream priority (0 = default, higher = more priority)
    pub stream_priority: i32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            max_memory: None,
            block_size: 256,
            use_persistent_kernels: false,
            use_tensor_cores: true,
            stream_priority: 0,
        }
    }
}

/// Performance metrics for GPU operations.
#[derive(Debug, Clone, Default)]
pub struct GpuMetrics {
    /// Total kernel execution time (microseconds)
    pub kernel_time_us: u64,
    /// Total memory transfer time (microseconds)
    pub transfer_time_us: u64,
    /// Number of kernel launches
    pub kernel_launches: usize,
    /// Bytes transferred to device
    pub bytes_to_device: usize,
    /// Bytes transferred from device
    pub bytes_from_device: usize,
    /// Peak memory usage
    pub peak_memory_usage: usize,
}

impl GpuMetrics {
    /// Computes the average kernel execution time.
    pub fn avg_kernel_time_us(&self) -> f64 {
        if self.kernel_launches > 0 {
            self.kernel_time_us as f64 / self.kernel_launches as f64
        } else {
            0.0
        }
    }
    
    /// Computes the total transfer bandwidth (MB/s).
    pub fn transfer_bandwidth_mbps(&self) -> f64 {
        let total_bytes = self.bytes_to_device + self.bytes_from_device;
        let total_time_s = self.transfer_time_us as f64 / 1_000_000.0;
        
        if total_time_s > 0.0 {
            (total_bytes as f64 / 1_048_576.0) / total_time_s
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_config_default() {
        let config = GpuConfig::default();
        assert_eq!(config.block_size, 256);
        assert!(config.use_tensor_cores);
        assert_eq!(config.stream_priority, 0);
    }
    
    #[test]
    fn test_gpu_metrics() {
        let mut metrics = GpuMetrics::default();
        metrics.kernel_time_us = 1000;
        metrics.kernel_launches = 10;
        
        assert_eq!(metrics.avg_kernel_time_us(), 100.0);
        
        metrics.bytes_to_device = 1024 * 1024;  // 1 MB
        metrics.bytes_from_device = 1024 * 1024;  // 1 MB
        metrics.transfer_time_us = 1000;  // 1 ms
        
        // 2 MB in 1 ms = 2000 MB/s
        assert_eq!(metrics.transfer_bandwidth_mbps(), 2000.0);
    }
}