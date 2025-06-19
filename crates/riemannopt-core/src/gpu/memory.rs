//! GPU memory management utilities.

use super::GpuError;
use crate::types::Scalar;
use cudarc::driver::{CudaDevice, DevicePtr};
use std::sync::Arc;
use std::collections::HashMap;
use std::sync::Mutex;

/// Memory pool for efficient GPU memory allocation
pub struct GpuMemoryPool<T: Scalar> {
    device: Arc<CudaDevice>,
    /// Available memory blocks by size
    free_blocks: Mutex<HashMap<usize, Vec<DevicePtr<T>>>>,
    /// Total allocated memory
    total_allocated: Mutex<usize>,
    /// Maximum memory limit
    max_memory: usize,
}

impl<T: Scalar + cudarc::driver::DeviceRepr> GpuMemoryPool<T> {
    /// Create a new memory pool
    pub fn new(device: Arc<CudaDevice>, max_memory: usize) -> Self {
        Self {
            device,
            free_blocks: Mutex::new(HashMap::new()),
            total_allocated: Mutex::new(0),
            max_memory,
        }
    }
    
    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize) -> Result<DevicePtr<T>, GpuError> {
        // Check if we have a free block of the right size
        {
            let mut free_blocks = self.free_blocks.lock().unwrap();
            if let Some(blocks) = free_blocks.get_mut(&size) {
                if let Some(block) = blocks.pop() {
                    return Ok(block);
                }
            }
        }
        
        // Check memory limit
        let mut total = self.total_allocated.lock().unwrap();
        let new_size = size * std::mem::size_of::<T>();
        
        if *total + new_size > self.max_memory {
            return Err(GpuError::AllocationError(
                format!("Memory limit exceeded: {} + {} > {}", 
                    *total, new_size, self.max_memory)
            ));
        }
        
        // Allocate new block
        let ptr = self.device.alloc::<T>(size)
            .map_err(|e| GpuError::AllocationError(format!("CUDA allocation failed: {}", e)))?;
        
        *total += new_size;
        
        Ok(ptr)
    }
    
    /// Return memory to the pool
    pub fn deallocate(&self, ptr: DevicePtr<T>, size: usize) {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        free_blocks.entry(size)
            .or_insert_with(Vec::new)
            .push(ptr);
    }
    
    /// Clear all cached memory
    pub fn clear(&self) {
        let mut free_blocks = self.free_blocks.lock().unwrap();
        let mut total = self.total_allocated.lock().unwrap();
        
        // Calculate freed memory
        let freed: usize = free_blocks.iter()
            .map(|(size, blocks)| size * blocks.len() * std::mem::size_of::<T>())
            .sum();
        
        // Clear all blocks
        free_blocks.clear();
        
        // Update total
        *total = total.saturating_sub(freed);
    }
    
    /// Get memory statistics
    pub fn stats(&self) -> MemoryStats {
        let free_blocks = self.free_blocks.lock().unwrap();
        let total = self.total_allocated.lock().unwrap();
        
        let cached_blocks: usize = free_blocks.values()
            .map(|v| v.len())
            .sum();
        
        let cached_memory: usize = free_blocks.iter()
            .map(|(size, blocks)| size * blocks.len() * std::mem::size_of::<T>())
            .sum();
        
        MemoryStats {
            total_allocated: *total,
            cached_memory,
            cached_blocks,
            max_memory: self.max_memory,
        }
    }
}

/// Memory statistics
#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub cached_memory: usize,
    pub cached_blocks: usize,
    pub max_memory: usize,
}

/// Pinned host memory for efficient transfers
pub struct PinnedMemory<T> {
    ptr: *mut T,
    size: usize,
}

impl<T: Scalar + cudarc::driver::DeviceRepr> PinnedMemory<T> {
    /// Allocate pinned host memory
    pub fn new(device: &CudaDevice, size: usize) -> Result<Self, GpuError> {
        let ptr = device.alloc_host::<T>(size)
            .map_err(|e| GpuError::AllocationError(format!("Pinned allocation failed: {}", e)))?;
        
        Ok(Self {
            ptr: ptr.as_ptr() as *mut T,
            size,
        })
    }
    
    /// Get a slice of the pinned memory
    pub fn as_slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(self.ptr, self.size)
        }
    }
    
    /// Get a mutable slice of the pinned memory
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        unsafe {
            std::slice::from_raw_parts_mut(self.ptr, self.size)
        }
    }
}

impl<T> Drop for PinnedMemory<T> {
    fn drop(&mut self) {
        // Pinned memory is automatically freed when DevicePtr is dropped
    }
}

/// Unified memory that can be accessed from both host and device
pub struct UnifiedMemory<T> {
    ptr: *mut T,
    size: usize,
    device: Arc<CudaDevice>,
}

impl<T: Scalar + cudarc::driver::DeviceRepr> UnifiedMemory<T> {
    /// Allocate unified memory
    pub fn new(device: Arc<CudaDevice>, size: usize) -> Result<Self, GpuError> {
        // Note: cudarc doesn't directly expose unified memory API
        // This would need custom CUDA calls
        // For now, return an error
        Err(GpuError::NotAvailable)
    }
}

/// Memory transfer utilities
pub struct MemoryTransfer;

impl MemoryTransfer {
    /// Asynchronous host to device copy
    pub fn h2d_async<T: cudarc::driver::DeviceRepr>(
        device: &CudaDevice,
        src: &[T],
        dst: &mut DevicePtr<T>,
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), GpuError> {
        device.htod_copy_into(src, dst)
            .map_err(|e| GpuError::TransferError(format!("H2D async failed: {}", e)))
    }
    
    /// Asynchronous device to host copy
    pub fn d2h_async<T: cudarc::driver::DeviceRepr>(
        device: &CudaDevice,
        src: &DevicePtr<T>,
        dst: &mut [T],
        stream: &cudarc::driver::CudaStream,
    ) -> Result<(), GpuError> {
        device.dtoh_copy_into(src, dst)
            .map_err(|e| GpuError::TransferError(format!("D2H async failed: {}", e)))
    }
    
    /// Device to device copy
    pub fn d2d<T: cudarc::driver::DeviceRepr>(
        device: &CudaDevice,
        src: &DevicePtr<T>,
        dst: &mut DevicePtr<T>,
        size: usize,
    ) -> Result<(), GpuError> {
        // Would use cudaMemcpy device to device
        // For now, go through host
        let mut temp = vec![T::default(); size];
        device.dtoh_copy_into(src, &mut temp)
            .map_err(|e| GpuError::TransferError(format!("D2D copy failed: {}", e)))?;
        device.htod_copy_into(&temp, dst)
            .map_err(|e| GpuError::TransferError(format!("D2D copy failed: {}", e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_memory_pool() {
        if !super::super::cuda::CudaBackend::is_available() {
            return;
        }
        
        let device = Arc::new(CudaDevice::new(0).unwrap());
        let pool: GpuMemoryPool<f32> = GpuMemoryPool::new(
            device.clone(),
            1024 * 1024 * 1024, // 1GB limit
        );
        
        // Allocate some memory
        let ptr1 = pool.allocate(1000);
        assert!(ptr1.is_ok());
        
        let stats = pool.stats();
        assert!(stats.total_allocated > 0);
        
        // Return memory to pool
        if let Ok(ptr) = ptr1 {
            pool.deallocate(ptr, 1000);
            
            let stats = pool.stats();
            assert_eq!(stats.cached_blocks, 1);
        }
        
        // Allocate again - should reuse
        let ptr2 = pool.allocate(1000);
        assert!(ptr2.is_ok());
        
        let stats = pool.stats();
        assert_eq!(stats.cached_blocks, 0);
    }
    
    #[test]
    fn test_pinned_memory() {
        if !super::super::cuda::CudaBackend::is_available() {
            return;
        }
        
        let device = CudaDevice::new(0).unwrap();
        let mut pinned: PinnedMemory<f32> = PinnedMemory::new(&device, 1000).unwrap();
        
        // Write to pinned memory
        let slice = pinned.as_mut_slice();
        for (i, val) in slice.iter_mut().enumerate() {
            *val = i as f32;
        }
        
        // Read back
        let slice = pinned.as_slice();
        assert_eq!(slice[0], 0.0);
        assert_eq!(slice[999], 999.0);
    }
}