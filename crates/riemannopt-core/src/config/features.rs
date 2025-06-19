//! CPU feature detection and runtime configuration.
//!
//! This module provides runtime detection of available CPU instruction sets
//! and SIMD capabilities, allowing the library to dynamically select the most
//! efficient implementations for the current hardware.

use once_cell::sync::Lazy;
use std::sync::Arc;

/// Available CPU features that can be detected at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CpuFeatures {
    /// AVX2 support (256-bit SIMD)
    pub avx2: bool,
    /// AVX-512 support (512-bit SIMD)
    pub avx512f: bool,
    /// FMA (Fused Multiply-Add) support
    pub fma: bool,
    /// SSE4.1 support
    pub sse41: bool,
    /// SSE4.2 support
    pub sse42: bool,
    /// NEON support (ARM)
    pub neon: bool,
}

impl CpuFeatures {
    /// Detect CPU features at runtime.
    pub fn detect() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                avx2: is_x86_feature_detected!("avx2"),
                avx512f: is_x86_feature_detected!("avx512f"),
                fma: is_x86_feature_detected!("fma"),
                sse41: is_x86_feature_detected!("sse4.1"),
                sse42: is_x86_feature_detected!("sse4.2"),
                neon: false,
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            Self {
                avx2: false,
                avx512f: false,
                fma: false,
                sse41: false,
                sse42: false,
                neon: std::arch::is_aarch64_feature_detected!("neon"),
            }
        }
        
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self {
                avx2: false,
                avx512f: false,
                fma: false,
                sse41: false,
                sse42: false,
                neon: false,
            }
        }
    }
    
    /// Check if any SIMD support is available.
    pub fn has_simd(&self) -> bool {
        self.avx2 || self.avx512f || self.sse41 || self.sse42 || self.neon
    }
    
    /// Get the maximum vector width in bits supported by this CPU.
    pub fn max_vector_width(&self) -> usize {
        if self.avx512f {
            512
        } else if self.avx2 {
            256
        } else if self.sse41 || self.sse42 {
            128
        } else if self.neon {
            128
        } else {
            64 // Scalar fallback
        }
    }
    
    /// Get the recommended vector size for f32 operations.
    pub fn recommended_f32_vector_size(&self) -> usize {
        self.max_vector_width() / 32
    }
    
    /// Get the recommended vector size for f64 operations.
    pub fn recommended_f64_vector_size(&self) -> usize {
        self.max_vector_width() / 64
    }
}

/// Global CPU features detected at program startup.
pub static CPU_FEATURES: Lazy<Arc<CpuFeatures>> = Lazy::new(|| Arc::new(CpuFeatures::detect()));

/// Get the detected CPU features.
pub fn cpu_features() -> &'static CpuFeatures {
    &CPU_FEATURES
}

/// Configuration for SIMD operations.
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Whether to use SIMD operations.
    pub enabled: bool,
    /// Minimum vector length to use SIMD (below this, use scalar).
    pub min_vector_length: usize,
    /// Whether to use AVX-512 if available.
    pub use_avx512: bool,
    /// Whether to use FMA instructions if available.
    pub use_fma: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        let features = cpu_features();
        Self {
            enabled: features.has_simd(),
            min_vector_length: 16, // Reasonable default
            use_avx512: features.avx512f,
            use_fma: features.fma,
        }
    }
}

/// Global SIMD configuration.
pub static SIMD_CONFIG: Lazy<Arc<SimdConfig>> = Lazy::new(|| Arc::new(SimdConfig::default()));

/// Get the current SIMD configuration.
pub fn simd_config() -> &'static SimdConfig {
    &SIMD_CONFIG
}

/// Builder for creating a custom SIMD configuration.
pub struct SimdConfigBuilder {
    config: SimdConfig,
}

impl SimdConfigBuilder {
    /// Create a new builder with default settings.
    pub fn new() -> Self {
        Self {
            config: SimdConfig::default(),
        }
    }
    
    /// Enable or disable SIMD operations.
    pub fn enabled(mut self, enabled: bool) -> Self {
        self.config.enabled = enabled;
        self
    }
    
    /// Set the minimum vector length for SIMD operations.
    pub fn min_vector_length(mut self, length: usize) -> Self {
        self.config.min_vector_length = length;
        self
    }
    
    /// Enable or disable AVX-512 usage.
    pub fn use_avx512(mut self, use_avx512: bool) -> Self {
        self.config.use_avx512 = use_avx512;
        self
    }
    
    /// Enable or disable FMA usage.
    pub fn use_fma(mut self, use_fma: bool) -> Self {
        self.config.use_fma = use_fma;
        self
    }
    
    /// Build the configuration.
    pub fn build(self) -> SimdConfig {
        self.config
    }
}

impl Default for SimdConfigBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro to conditionally compile code based on target architecture.
#[macro_export]
macro_rules! arch_specific {
    (x86_64: $x86_64:expr, aarch64: $aarch64:expr, default: $default:expr) => {
        {
            #[cfg(target_arch = "x86_64")]
            { $x86_64 }
            
            #[cfg(target_arch = "aarch64")]
            { $aarch64 }
            
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            { $default }
        }
    };
}

/// Macro to conditionally compile SIMD code based on available features.
#[macro_export]
macro_rules! simd_dispatch {
    ($scalar:expr, avx2: $avx2:expr, avx512: $avx512:expr, neon: $neon:expr) => {
        {
            let features = $crate::config::features::cpu_features();
            let config = $crate::config::features::simd_config();
            
            if !config.enabled {
                $scalar
            } else if config.use_avx512 && features.avx512f {
                #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx512f"))]
                { $avx512 }
                #[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx512f")))]
                { $scalar }
            } else if features.avx2 {
                #[cfg(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))]
                { $avx2 }
                #[cfg(not(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2")))]
                { $scalar }
            } else if features.neon {
                #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
                { $neon }
                #[cfg(not(all(target_arch = "aarch64", target_feature = "neon")))]
                { $scalar }
            } else {
                $scalar
            }
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_cpu_feature_detection() {
        let features = CpuFeatures::detect();
        println!("Detected CPU features: {:?}", features);
        
        // Basic sanity checks
        assert!(features.max_vector_width() >= 64);
        assert!(features.recommended_f32_vector_size() >= 2);
        assert!(features.recommended_f64_vector_size() >= 1);
    }
    
    #[test]
    fn test_simd_config_builder() {
        let config = SimdConfigBuilder::new()
            .enabled(true)
            .min_vector_length(32)
            .use_avx512(false)
            .use_fma(true)
            .build();
        
        assert!(config.enabled);
        assert_eq!(config.min_vector_length, 32);
        assert!(!config.use_avx512);
        assert!(config.use_fma);
    }
    
    #[test]
    fn test_global_cpu_features() {
        let features1 = cpu_features();
        let features2 = cpu_features();
        
        // Should return the same instance
        assert!(std::ptr::eq(features1, features2));
    }
    
    #[test]
    fn test_arch_specific_macro() {
        let result = arch_specific!(
            x86_64: "x86_64",
            aarch64: "aarch64",
            default: "other"
        );
        
        #[cfg(target_arch = "x86_64")]
        assert_eq!(result, "x86_64");
        
        #[cfg(target_arch = "aarch64")]
        assert_eq!(result, "aarch64");
        
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        assert_eq!(result, "other");
    }
}