//! Cached wrapper for cost functions with dynamic dimensions.
//!
//! This module provides a specialized caching layer for cost functions
//! that work with dynamic vectors (DVector), which is common in many
//! optimization problems.

use crate::{
    core::cost_function::CostFunction,
    error::Result,
    memory::cache::{MultiLevelCache, CacheKey, Cacheable},
    types::Scalar,
};
use nalgebra::{DVector, DMatrix, Dyn};
use std::fmt::Debug;
use std::hash::{Hash, Hasher};
use std::sync::Arc;
use std::time::Duration;

/// A hash-based cache key for dynamic vectors.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct DynamicPointKey {
    /// Hash of the point coordinates
    hash: u64,
    /// Dimension for verification
    dim: usize,
}

impl CacheKey for DynamicPointKey {}

impl<T: Scalar> From<&DVector<T>> for DynamicPointKey {
    fn from(point: &DVector<T>) -> Self {
        use std::collections::hash_map::DefaultHasher;
        
        let mut hasher = DefaultHasher::new();
        
        // Hash dimension
        point.len().hash(&mut hasher);
        
        // Hash each component
        for i in 0..point.len() {
            let val = point[i].to_f64();
            val.to_bits().hash(&mut hasher);
        }
        
        Self {
            hash: hasher.finish(),
            dim: point.len(),
        }
    }
}

/// Cached results for dynamic vectors.
#[derive(Clone, Debug)]
pub struct CachedDynamicResult<T: Scalar> {
    /// Function value
    pub value: Option<T>,
    /// Gradient vector
    pub gradient: Option<DVector<T>>,
    /// Combined cost and gradient
    pub cost_and_gradient: Option<(T, DVector<T>)>,
}

impl<T: Scalar> Cacheable for CachedDynamicResult<T> {
    fn size_bytes(&self) -> usize {
        let base = std::mem::size_of::<Self>();
        let grad_size = self.gradient.as_ref()
            .map(|g| g.len() * std::mem::size_of::<T>())
            .unwrap_or(0);
        let cg_size = self.cost_and_gradient.as_ref()
            .map(|(_, g)| g.len() * std::mem::size_of::<T>() + std::mem::size_of::<T>())
            .unwrap_or(0);
        base + grad_size + cg_size
    }
}

/// A cached cost function for dynamic vectors.
pub struct CachedDynamicCostFunction<F>
where
    F: CostFunction<f64, Dyn>,
{
    /// The underlying cost function
    inner: F,
    /// Multi-level cache
    cache: Arc<MultiLevelCache<DynamicPointKey, CachedDynamicResult<f64>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Statistics
    stats: Arc<parking_lot::Mutex<CacheStatistics>>,
}

/// Configuration for cached cost function.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Maximum entries in L1 cache
    pub l1_max_entries: usize,
    /// Maximum bytes in L1 cache
    pub l1_max_bytes: usize,
    /// Maximum entries in L2 cache
    pub l2_max_entries: usize,
    /// Maximum bytes in L2 cache
    pub l2_max_bytes: usize,
    /// Time-to-live for L2 cache entries
    pub l2_ttl: Duration,
    /// Whether to cache gradients
    pub cache_gradients: bool,
    /// Whether to cache combined cost and gradient
    pub cache_combined: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            l1_max_entries: 100,
            l1_max_bytes: 10 * 1024 * 1024, // 10 MB
            l2_max_entries: 1000,
            l2_max_bytes: 100 * 1024 * 1024, // 100 MB
            l2_ttl: Duration::from_secs(300), // 5 minutes
            cache_gradients: true,
            cache_combined: true,
        }
    }
}

impl<F> Debug for CachedDynamicCostFunction<F>
where
    F: CostFunction<f64, Dyn>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedDynamicCostFunction")
            .field("inner", &self.inner)
            .field("config", &self.config)
            .finish()
    }
}

impl<F> CachedDynamicCostFunction<F>
where
    F: CostFunction<f64, Dyn>,
{
    /// Creates a new cached cost function with default configuration.
    pub fn new(inner: F) -> Self {
        Self::with_config(inner, CacheConfig::default())
    }

    /// Creates a new cached cost function with custom configuration.
    pub fn with_config(inner: F, config: CacheConfig) -> Self {
        let cache = Arc::new(MultiLevelCache::new(
            config.l1_max_entries,
            config.l1_max_bytes,
            config.l2_max_entries,
            config.l2_max_bytes,
            config.l2_ttl,
        ));
        
        Self {
            inner,
            cache,
            config,
            stats: Arc::new(parking_lot::Mutex::new(CacheStatistics::default())),
        }
    }

    /// Returns cache statistics.
    pub fn cache_stats(&self) -> CacheStatistics {
        self.stats.lock().clone()
    }

    /// Clears the cache.
    pub fn clear_cache(&self) {
        self.cache.clear();
        self.stats.lock().reset();
    }

    /// Performs cache maintenance.
    pub fn maintain_cache(&self) {
        self.cache.maintain();
    }
}

impl<F> CostFunction<f64, Dyn> for CachedDynamicCostFunction<F>
where
    F: CostFunction<f64, Dyn>,
{
    fn cost(&self, point: &DVector<f64>) -> Result<f64> {
        let key = DynamicPointKey::from(point);
        let mut stats = self.stats.lock();
        
        // Check cache
        if let Some(cached) = self.cache.get(&key) {
            if let Some(value) = cached.value {
                stats.cost_hits += 1;
                return Ok(value);
            }
        }
        stats.cost_misses += 1;
        drop(stats);
        
        // Compute and cache
        let value = self.inner.cost(point)?;
        
        let mut cached = self.cache.get(&key).unwrap_or(CachedDynamicResult {
            value: None,
            gradient: None,
            cost_and_gradient: None,
        });
        cached.value = Some(value);
        self.cache.insert(key, cached);
        
        Ok(value)
    }

    fn cost_and_gradient(&self, point: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
        if !self.config.cache_combined {
            return self.inner.cost_and_gradient(point);
        }
        
        let key = DynamicPointKey::from(point);
        let mut stats = self.stats.lock();
        
        // Check cache
        if let Some(cached) = self.cache.get(&key) {
            if let Some((cost, grad)) = &cached.cost_and_gradient {
                stats.combined_hits += 1;
                return Ok((*cost, grad.clone()));
            }
        }
        stats.combined_misses += 1;
        drop(stats);
        
        // Compute and cache
        let (cost, gradient) = self.inner.cost_and_gradient(point)?;
        
        let mut cached = self.cache.get(&key).unwrap_or(CachedDynamicResult {
            value: None,
            gradient: None,
            cost_and_gradient: None,
        });
        cached.value = Some(cost);
        cached.gradient = Some(gradient.clone());
        cached.cost_and_gradient = Some((cost, gradient.clone()));
        self.cache.insert(key, cached);
        
        Ok((cost, gradient))
    }

    fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
        if !self.config.cache_gradients {
            return self.inner.gradient(point);
        }
        
        let key = DynamicPointKey::from(point);
        let mut stats = self.stats.lock();
        
        // Check cache
        if let Some(cached) = self.cache.get(&key) {
            if let Some(gradient) = &cached.gradient {
                stats.gradient_hits += 1;
                return Ok(gradient.clone());
            }
        }
        stats.gradient_misses += 1;
        drop(stats);
        
        // Try to use cost_and_gradient if available
        let gradient = if let Ok((cost, grad)) = self.inner.cost_and_gradient(point) {
            // Cache both results
            let mut cached = self.cache.get(&key).unwrap_or(CachedDynamicResult {
                value: None,
                gradient: None,
                cost_and_gradient: None,
            });
            cached.value = Some(cost);
            cached.gradient = Some(grad.clone());
            cached.cost_and_gradient = Some((cost, grad.clone()));
            self.cache.insert(key, cached);
            grad
        } else {
            // Fall back to gradient only
            let grad = self.inner.gradient(point)?;
            let mut cached = self.cache.get(&key).unwrap_or(CachedDynamicResult {
                value: None,
                gradient: None,
                cost_and_gradient: None,
            });
            cached.gradient = Some(grad.clone());
            self.cache.insert(key, cached);
            grad
        };
        
        Ok(gradient)
    }

    fn hessian(&self, point: &DVector<f64>) -> Result<DMatrix<f64>> {
        // Hessians are typically too large to cache effectively
        self.inner.hessian(point)
    }

    fn hessian_vector_product(
        &self,
        point: &DVector<f64>,
        vector: &DVector<f64>,
    ) -> Result<DVector<f64>> {
        // HVP results depend on both point and vector, making caching complex
        self.inner.hessian_vector_product(point, vector)
    }
}

/// Statistics for cache performance.
#[derive(Debug, Clone, Default)]
pub struct CacheStatistics {
    /// Number of cost function cache hits
    pub cost_hits: usize,
    /// Number of cost function cache misses
    pub cost_misses: usize,
    /// Number of gradient cache hits
    pub gradient_hits: usize,
    /// Number of gradient cache misses
    pub gradient_misses: usize,
    /// Number of combined cost and gradient cache hits
    pub combined_hits: usize,
    /// Number of combined cost and gradient cache misses
    pub combined_misses: usize,
}

impl CacheStatistics {
    /// Returns the overall hit rate.
    pub fn hit_rate(&self) -> f64 {
        let total_hits = self.cost_hits + self.gradient_hits + self.combined_hits;
        let total_accesses = total_hits + self.cost_misses + self.gradient_misses + self.combined_misses;
        
        if total_accesses > 0 {
            total_hits as f64 / total_accesses as f64
        } else {
            0.0
        }
    }
    
    /// Resets all statistics.
    pub fn reset(&mut self) {
        *self = Self::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cost_function::QuadraticCost;
    use nalgebra::DMatrix;

    #[test]
    fn test_cached_dynamic_cost_function() {
        let n = 3;
        let a = DMatrix::<f64>::identity(n, n);
        let b = DVector::zeros(n);
        let inner = QuadraticCost::new(a, b, 0.0);
        
        let cached = CachedDynamicCostFunction::new(inner);
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        // First call - cache miss
        let cost1 = cached.cost(&x).unwrap();
        assert_eq!(cached.cache_stats().cost_misses, 1);
        assert_eq!(cached.cache_stats().cost_hits, 0);
        
        // Second call - cache hit
        let cost2 = cached.cost(&x).unwrap();
        assert_eq!(cost1, cost2);
        assert_eq!(cached.cache_stats().cost_misses, 1);
        assert_eq!(cached.cache_stats().cost_hits, 1);
        
        // Test gradient caching
        let _grad1 = cached.gradient(&x).unwrap();
        assert_eq!(cached.cache_stats().gradient_misses, 1); // First gradient call is a miss
        
        // Second gradient call should hit
        let _grad2 = cached.gradient(&x).unwrap();
        assert_eq!(cached.cache_stats().gradient_hits, 1);
        
        // Different point
        let y = DVector::from_vec(vec![2.0, 3.0, 4.0]);
        let _ = cached.cost(&y).unwrap();
        assert_eq!(cached.cache_stats().cost_misses, 2);
    }

    #[test]
    fn test_cache_hit_rate() {
        let mut stats = CacheStatistics::default();
        assert_eq!(stats.hit_rate(), 0.0);
        
        stats.cost_hits = 3;
        stats.cost_misses = 1;
        stats.gradient_hits = 2;
        stats.gradient_misses = 2;
        
        // Total hits = 3 + 2 = 5
        // Total accesses = 5 + 1 + 2 = 8
        // Hit rate = 5 / 8 = 0.625
        assert_eq!(stats.hit_rate(), 0.625);
    }
}