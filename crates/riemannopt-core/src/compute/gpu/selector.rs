//! GPU backend selection and detection.
//!
//! This module provides functionality to detect available GPU devices
//! and select the appropriate backend based on the hardware and problem
//! characteristics.

use std::fmt;

/// Types of GPU backends supported.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackendType {
    /// NVIDIA CUDA
    Cuda,
    /// AMD ROCm
    Rocm,
    /// Apple Metal
    Metal,
    /// Intel OneAPI
    OneApi,
    /// OpenCL (fallback)
    OpenCl,
}

impl fmt::Display for GpuBackendType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cuda => write!(f, "CUDA"),
            Self::Rocm => write!(f, "ROCm"),
            Self::Metal => write!(f, "Metal"),
            Self::OneApi => write!(f, "OneAPI"),
            Self::OpenCl => write!(f, "OpenCL"),
        }
    }
}

/// Information about a GPU device.
#[derive(Debug, Clone)]
pub struct GpuDevice {
    /// Device index
    pub index: usize,
    /// Device name
    pub name: String,
    /// Backend type
    pub backend_type: GpuBackendType,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Compute capability (for CUDA) or equivalent
    pub compute_capability: (u32, u32),
    /// Whether the device supports double precision
    pub supports_double: bool,
}

impl GpuDevice {
    /// Checks if this device is suitable for a given problem size.
    pub fn is_suitable_for(&self, required_memory: usize) -> bool {
        self.available_memory >= required_memory
    }
    
    /// Estimates the required memory for a problem.
    pub fn estimate_memory_requirement(dimension: usize, precision: usize) -> usize {
        // Rough estimate: need space for several vectors and matrices
        // This is a conservative estimate
        let vector_size = dimension * precision;
        let matrix_size = dimension * dimension * precision;
        
        // Assume we need at least:
        // - 10 vectors (for various intermediate results)
        // - 2 matrices (for Hessian approximations)
        10 * vector_size + 2 * matrix_size
    }
}

/// GPU selector that chooses the best available GPU.
#[derive(Debug)]
pub struct GpuSelector {
    devices: Vec<GpuDevice>,
    preferred_backend: Option<GpuBackendType>,
}

impl GpuSelector {
    /// Creates a new GPU selector.
    pub fn new() -> Self {
        let devices = Self::detect_devices();
        Self {
            devices,
            preferred_backend: None,
        }
    }
    
    /// Sets a preferred backend type.
    pub fn with_preferred_backend(mut self, backend: GpuBackendType) -> Self {
        self.preferred_backend = Some(backend);
        self
    }
    
    /// Detects available GPU devices.
    fn detect_devices() -> Vec<GpuDevice> {
        #[allow(unused_mut)]
        let mut devices = Vec::new();
        
        // Try CUDA detection
        #[cfg(feature = "cuda")]
        if let Ok(cuda_devices) = Self::detect_cuda_devices() {
            devices.extend(cuda_devices);
        }
        
        // Try Metal detection on macOS
        #[cfg(target_os = "macos")]
        if let Ok(metal_devices) = Self::detect_metal_devices() {
            devices.extend(metal_devices);
        }
        
        // Try ROCm detection (commented out until feature is added to Cargo.toml)
        // #[cfg(feature = "rocm")]
        // if let Ok(rocm_devices) = Self::detect_rocm_devices() {
        //     devices.extend(rocm_devices);
        // }
        
        devices
    }
    
    /// Detects CUDA devices (stub for now).
    #[cfg(feature = "cuda")]
    fn detect_cuda_devices() -> crate::error::Result<Vec<GpuDevice>> {
        // TODO: Implement actual CUDA detection
        Ok(vec![])
    }
    
    /// Detects Metal devices (stub for now).
    #[cfg(target_os = "macos")]
    fn detect_metal_devices() -> crate::error::Result<Vec<GpuDevice>> {
        // TODO: Implement actual Metal detection
        Ok(vec![])
    }
    
    // /// Detects ROCm devices (stub for now).
    // #[cfg(feature = "rocm")]
    // fn detect_rocm_devices() -> crate::error::Result<Vec<GpuDevice>> {
    //     // TODO: Implement actual ROCm detection
    //     Ok(vec![])
    // }
    
    /// Lists all available devices.
    pub fn available_devices(&self) -> &[GpuDevice] {
        &self.devices
    }
    
    /// Checks if any GPU is available.
    pub fn has_gpu(&self) -> bool {
        !self.devices.is_empty()
    }
    
    /// Selects the best GPU for a given problem.
    pub fn select_device(&self, dimension: usize, precision: usize) -> Option<&GpuDevice> {
        let required_memory = GpuDevice::estimate_memory_requirement(dimension, precision);
        
        // Filter suitable devices
        let mut suitable_devices: Vec<_> = self.devices.iter()
            .filter(|d| d.is_suitable_for(required_memory))
            .collect();
        
        if suitable_devices.is_empty() {
            return None;
        }
        
        // Sort by preference
        suitable_devices.sort_by_key(|d| {
            let backend_priority = match (&self.preferred_backend, &d.backend_type) {
                (Some(pref), actual) if pref == actual => 0,
                _ => 1,
            };
            
            // Sort by: preferred backend, then by available memory
            (backend_priority, std::cmp::Reverse(d.available_memory))
        });
        
        suitable_devices.first().copied()
    }
    
    /// Checks if a specific backend type is available.
    pub fn has_backend(&self, backend_type: GpuBackendType) -> bool {
        self.devices.iter().any(|d| d.backend_type == backend_type)
    }
}

/// Decision logic for when to use GPU vs CPU.
#[derive(Debug, Clone)]
pub struct GpuDecisionMaker {
    /// Minimum dimension to consider GPU
    pub min_dimension: usize,
    /// Minimum speedup factor expected
    pub min_speedup_factor: f64,
    /// Whether to use GPU for double precision
    pub use_gpu_for_double: bool,
}

impl Default for GpuDecisionMaker {
    fn default() -> Self {
        Self {
            min_dimension: 1000,  // GPU typically beneficial for larger problems
            min_speedup_factor: 2.0,
            use_gpu_for_double: true,
        }
    }
}

impl GpuDecisionMaker {
    /// Decides whether to use GPU for a given problem.
    pub fn should_use_gpu(
        &self,
        dimension: usize,
        is_double_precision: bool,
        gpu_available: bool,
    ) -> bool {
        if !gpu_available {
            return false;
        }
        
        if dimension < self.min_dimension {
            return false;
        }
        
        if is_double_precision && !self.use_gpu_for_double {
            return false;
        }
        
        // Additional heuristics can be added here
        true
    }
    
    /// Estimates the speedup factor for GPU.
    pub fn estimate_speedup(&self, dimension: usize) -> f64 {
        // Simple heuristic: speedup grows with problem size
        if dimension < 100 {
            0.5  // GPU overhead dominates
        } else if dimension < 1000 {
            1.0 + (dimension as f64 - 100.0) / 900.0  // Linear growth from 1x to 2x
        } else {
            2.0 + (dimension as f64).log10()  // Logarithmic growth beyond 2x
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_backend_type_display() {
        assert_eq!(format!("{}", GpuBackendType::Cuda), "CUDA");
        assert_eq!(format!("{}", GpuBackendType::Metal), "Metal");
    }
    
    #[test]
    fn test_gpu_device_memory_estimation() {
        // For dimension 1000 with f64 (8 bytes)
        let memory = GpuDevice::estimate_memory_requirement(1000, 8);
        
        // Should be at least 10 vectors + 2 matrices
        let expected = 10 * 1000 * 8 + 2 * 1000 * 1000 * 8;
        assert_eq!(memory, expected);
    }
    
    #[test]
    fn test_gpu_decision_maker() {
        let decision_maker = GpuDecisionMaker::default();
        
        // Small problem, GPU available
        assert!(!decision_maker.should_use_gpu(100, true, true));
        
        // Large problem, GPU available
        assert!(decision_maker.should_use_gpu(2000, true, true));
        
        // Large problem, no GPU
        assert!(!decision_maker.should_use_gpu(2000, true, false));
        
        // Test speedup estimation
        assert!(decision_maker.estimate_speedup(50) < 1.0);
        assert!((decision_maker.estimate_speedup(500) - 1.5).abs() < 0.1);
        assert!(decision_maker.estimate_speedup(10000) > 3.0);
    }
    
    #[test]
    fn test_gpu_selector() {
        let selector = GpuSelector::new();
        
        // Without actual GPU detection, should have no devices
        assert!(selector.available_devices().is_empty());
        assert!(!selector.has_gpu());
        
        // Test device selection with no devices
        assert!(selector.select_device(1000, 8).is_none());
    }
}