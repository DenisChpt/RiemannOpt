//! CUDA backend implementation for GPU acceleration.

use super::{DeviceInfo, GpuBackend, GpuError, GpuMatrix, GpuMatrixOps};
use crate::types::Scalar;
use cudarc::driver::{CudaDevice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::cublas::{CudaBlas, Gemm};
use nalgebra::DMatrix;
use std::sync::Arc;

/// CUDA backend for GPU operations
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    stream: CudaStream,
    blas: Arc<CudaBlas>,
}

impl GpuBackend for CudaBackend {
    fn initialize() -> Result<Self, GpuError> {
        let device = CudaDevice::new(0)
            .map_err(|e| GpuError::CudaError(format!("Failed to create device: {}", e)))?;
        let device = Arc::new(device);
        
        let stream = device.fork_default_stream()
            .map_err(|e| GpuError::CudaError(format!("Failed to create stream: {}", e)))?;
        
        let blas = Arc::new(CudaBlas::new(device.clone())
            .map_err(|e| GpuError::CudaError(format!("Failed to create CUBLAS: {}", e)))?);
        
        Ok(Self {
            device,
            stream,
            blas,
        })
    }
    
    fn is_available() -> bool {
        CudaDevice::new(0).is_ok()
    }
    
    fn device_info(&self) -> DeviceInfo {
        let ordinal = self.device.ordinal();
        
        // Get device properties
        let name = format!("CUDA Device {}", ordinal);
        let compute_capability = (8, 0); // Default, would need actual query
        
        // Query memory info
        let (memory_free, memory_total) = self.device.memory_info();
        
        DeviceInfo {
            name,
            compute_capability,
            memory_total,
            memory_free,
            multiprocessor_count: 80, // Default, would need actual query
        }
    }
    
    fn synchronize(&self) -> Result<(), GpuError> {
        self.stream.synchronize()
            .map_err(|e| GpuError::CudaError(format!("Synchronization failed: {}", e)))
    }
}

/// CUDA implementation of GPU matrix
impl<T: Scalar + cudarc::driver::DeviceRepr> GpuMatrix<T> {
    /// Create from host matrix
    pub fn from_host_cuda(matrix: &DMatrix<T>, device: &Arc<CudaDevice>) -> Result<Self, GpuError> {
        let rows = matrix.nrows();
        let cols = matrix.ncols();
        let data = matrix.as_slice();
        
        // Allocate device memory
        let device_ptr = device.alloc::<T>(rows * cols)
            .map_err(|e| GpuError::AllocationError(format!("Failed to allocate: {}", e)))?;
        
        // Copy data to device
        device.htod_copy_into(data, &device_ptr)
            .map_err(|e| GpuError::TransferError(format!("H2D copy failed: {}", e)))?;
        
        Ok(Self {
            device_ptr: device_ptr.as_ptr() as *mut T,
            rows,
            cols,
            ld: rows, // Column-major layout
        })
    }
    
    /// Copy to host matrix
    pub fn to_host_cuda(&self, device: &Arc<CudaDevice>) -> Result<DMatrix<T>, GpuError> {
        let size = self.rows * self.cols;
        let mut host_data = vec![T::zero(); size];
        
        // Create device pointer wrapper
        let device_ptr = unsafe {
            DevicePtr::from_raw(self.device_ptr as *mut T, size)
        };
        
        // Copy data from device
        device.dtoh_copy_into(&device_ptr, &mut host_data)
            .map_err(|e| GpuError::TransferError(format!("D2H copy failed: {}", e)))?;
        
        // Create matrix from data
        Ok(DMatrix::from_vec(self.rows, self.cols, host_data))
    }
}

impl<T: Scalar + cudarc::driver::DeviceRepr + Gemm> GpuMatrixOps<T> for CudaBackend {
    fn gemm(
        &self,
        a: &GpuMatrix<T>,
        b: &GpuMatrix<T>,
        alpha: T,
        beta: T,
        c: &mut GpuMatrix<T>,
    ) -> Result<(), GpuError> {
        // Check dimensions
        if a.cols != b.rows || a.rows != c.rows || b.cols != c.cols {
            return Err(GpuError::DimensionError(
                "Invalid dimensions for matrix multiplication".to_string()
            ));
        }
        
        // Perform GEMM using cuBLAS
        // C = alpha * A * B + beta * C
        unsafe {
            self.blas.gemm(
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                cudarc::cublas::sys::cublasOperation_t::CUBLAS_OP_N,
                a.rows as i32,
                b.cols as i32,
                a.cols as i32,
                &alpha,
                a.device_ptr,
                a.ld as i32,
                b.device_ptr,
                b.ld as i32,
                &beta,
                c.device_ptr,
                c.ld as i32,
            ).map_err(|e| GpuError::CudaError(format!("GEMM failed: {}", e)))?;
        }
        
        Ok(())
    }
    
    fn transpose(&self, a: &GpuMatrix<T>) -> Result<GpuMatrix<T>, GpuError> {
        let rows = a.cols;
        let cols = a.rows;
        
        // Allocate output matrix
        let device_ptr = self.device.alloc::<T>(rows * cols)
            .map_err(|e| GpuError::AllocationError(format!("Failed to allocate: {}", e)))?;
        
        let mut result = GpuMatrix {
            device_ptr: device_ptr.as_ptr() as *mut T,
            rows,
            cols,
            ld: rows,
        };
        
        // Launch transpose kernel
        let block_size = 16;
        let grid_x = (cols + block_size - 1) / block_size;
        let grid_y = (rows + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_x as u32, grid_y as u32, 1),
            block_dim: (block_size as u32, block_size as u32, 1),
            shared_mem_bytes: 0,
        };
        
        // Would launch actual kernel here
        // For now, use a simple CPU fallback
        let host_a = a.to_host_cuda(&self.device)?;
        let transposed = host_a.transpose();
        result = GpuMatrix::from_host_cuda(&transposed, &self.device)?;
        
        Ok(result)
    }
    
    fn elementwise_add(
        &self,
        a: &GpuMatrix<T>,
        b: &GpuMatrix<T>,
    ) -> Result<GpuMatrix<T>, GpuError> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(GpuError::DimensionError(
                "Matrices must have same dimensions".to_string()
            ));
        }
        
        let size = a.rows * a.cols;
        
        // Allocate output
        let device_ptr = self.device.alloc::<T>(size)
            .map_err(|e| GpuError::AllocationError(format!("Failed to allocate: {}", e)))?;
        
        let result = GpuMatrix {
            device_ptr: device_ptr.as_ptr() as *mut T,
            rows: a.rows,
            cols: a.cols,
            ld: a.rows,
        };
        
        // Launch elementwise addition kernel
        let block_size = 256;
        let grid_size = (size + block_size - 1) / block_size;
        
        let config = LaunchConfig {
            grid_dim: (grid_size as u32, 1, 1),
            block_dim: (block_size as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        
        // Would launch actual kernel here
        // For now, use CPU fallback
        let host_a = a.to_host_cuda(&self.device)?;
        let host_b = b.to_host_cuda(&self.device)?;
        let sum = &host_a + &host_b;
        let result = GpuMatrix::from_host_cuda(&sum, &self.device)?;
        
        Ok(result)
    }
    
    fn elementwise_mul(
        &self,
        a: &GpuMatrix<T>,
        b: &GpuMatrix<T>,
    ) -> Result<GpuMatrix<T>, GpuError> {
        if a.rows != b.rows || a.cols != b.cols {
            return Err(GpuError::DimensionError(
                "Matrices must have same dimensions".to_string()
            ));
        }
        
        let size = a.rows * a.cols;
        
        // Allocate output
        let device_ptr = self.device.alloc::<T>(size)
            .map_err(|e| GpuError::AllocationError(format!("Failed to allocate: {}", e)))?;
        
        let result = GpuMatrix {
            device_ptr: device_ptr.as_ptr() as *mut T,
            rows: a.rows,
            cols: a.cols,
            ld: a.rows,
        };
        
        // Would launch elementwise multiplication kernel
        // For now, use CPU fallback
        let host_a = a.to_host_cuda(&self.device)?;
        let host_b = b.to_host_cuda(&self.device)?;
        let product = host_a.component_mul(&host_b);
        let result = GpuMatrix::from_host_cuda(&product, &self.device)?;
        
        Ok(result)
    }
}

/// GPU-accelerated sphere operations
pub struct GpuSphere {
    backend: CudaBackend,
    dim: usize,
}

impl GpuSphere {
    pub fn new(dim: usize) -> Result<Self, GpuError> {
        let backend = CudaBackend::initialize()?;
        Ok(Self { backend, dim })
    }
    
    /// Project point onto sphere using GPU
    pub fn project_gpu<T: Scalar + cudarc::driver::DeviceRepr>(
        &self,
        point: &GpuMatrix<T>,
    ) -> Result<GpuMatrix<T>, GpuError> {
        // Compute norm
        // Then normalize
        // For now, use CPU fallback
        let host_point = point.to_host_cuda(&self.backend.device)?;
        let norm = host_point.norm();
        
        if norm > T::zero() {
            let normalized = host_point / norm;
            GpuMatrix::from_host_cuda(&normalized, &self.backend.device)
        } else {
            Ok(GpuMatrix {
                device_ptr: point.device_ptr,
                rows: point.rows,
                cols: point.cols,
                ld: point.ld,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cuda_initialization() {
        if CudaBackend::is_available() {
            let backend = CudaBackend::initialize();
            assert!(backend.is_ok());
            
            if let Ok(backend) = backend {
                let info = backend.device_info();
                println!("Device: {}", info.name);
                println!("Memory: {} MB free / {} MB total", 
                    info.memory_free / (1024 * 1024),
                    info.memory_total / (1024 * 1024)
                );
            }
        }
    }
    
    #[test]
    fn test_matrix_transfer() {
        if !CudaBackend::is_available() {
            return;
        }
        
        let backend = CudaBackend::initialize().unwrap();
        let host_matrix = DMatrix::from_vec(3, 3, vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ]);
        
        // Transfer to GPU
        let gpu_matrix = GpuMatrix::from_host_cuda(&host_matrix, &backend.device);
        assert!(gpu_matrix.is_ok());
        
        // Transfer back
        if let Ok(gpu_matrix) = gpu_matrix {
            let back = gpu_matrix.to_host_cuda(&backend.device);
            assert!(back.is_ok());
            
            if let Ok(back) = back {
                assert_eq!(host_matrix, back);
            }
        }
    }
}