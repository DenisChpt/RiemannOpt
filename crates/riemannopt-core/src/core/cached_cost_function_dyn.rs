//! Cached wrapper for cost functions with dynamic dimensions.
//!
//! This module provides a specialized caching layer for cost functions
//! that work with dynamic vectors (DVector), which is common in many
//! optimization problems.

use crate::{
    core::cost_function::CostFunction,
    error::Result,
    memory::Workspace,
    types::{Scalar, constants},
    compute::cpu::{get_dispatcher, SimdBackend},
};
use nalgebra::{DVector, DMatrix, Dyn};
use std::fmt::Debug;
use std::sync::Arc;

/// Simple last-point cache entry
#[derive(Clone, Debug)]
struct LastPointCache<T: Scalar> {
    /// The cached point
    point: DVector<T>,
    /// Function value
    value: Option<T>,
    /// Gradient vector
    gradient: Option<DVector<T>>,
}

impl<T: Scalar> LastPointCache<T> {
    fn new(point: DVector<T>) -> Self {
        Self {
            point,
            value: None,
            gradient: None,
        }
    }
    
    /// Check if this cache entry matches the given point within tolerance
    fn matches_with_tolerance(&self, point: &DVector<T>, tolerance: T) -> bool {
        // First check dimensions for fast rejection
        if self.point.len() != point.len() {
            return false;
        }
        
        // For f64, we can use the dispatcher directly
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // SAFETY: We've verified T is f64
            unsafe {
                let self_point_f64 = &*(&self.point as *const DVector<T> as *const DVector<f64>);
                let point_f64 = &*(point as *const DVector<T> as *const DVector<f64>);
                let tolerance_f64 = *(&tolerance as *const T as *const f64);
                
                let dispatcher = get_dispatcher::<f64>();
                let max_diff = dispatcher.max_abs_diff(self_point_f64, point_f64);
                return max_diff <= tolerance_f64;
            }
        }
        
        // Fallback for other types (shouldn't happen in practice since this is used with f64)
        for i in 0..self.point.len() {
            let diff = self.point[i] - point[i];
            if num_traits::Signed::abs(&diff) > tolerance {
                return false;
            }
        }
        
        true
    }
}

/// A cached cost function for dynamic vectors.
/// 
/// This implementation uses a simple last-point cache which is optimal for
/// the common pattern of calling cost() and gradient() on the same point
/// in sequence, without the overhead of hashing large vectors.
/// 
/// Points are compared using a tolerance-based approach to handle
/// floating-point precision issues.
pub struct CachedDynamicCostFunction<F>
where
    F: CostFunction<f64, Dyn>,
{
    /// The underlying cost function
    inner: F,
    /// Simple last-point cache
    cache: Arc<parking_lot::RwLock<Option<LastPointCache<f64>>>>,
    /// Cache configuration
    config: CacheConfig,
    /// Statistics
    stats: Arc<parking_lot::Mutex<CacheStatistics>>,
    /// Comparison tolerance
    tolerance: f64,
}

/// Configuration for cached cost function.
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// Whether to cache values
    pub cache_values: bool,
    /// Whether to cache gradients
    pub cache_gradients: bool,
    /// Tolerance for point comparison (None uses default_tolerance)
    pub comparison_tolerance: Option<f64>,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            cache_values: true,
            cache_gradients: true,
            comparison_tolerance: None,
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
        let tolerance = config.comparison_tolerance
            .unwrap_or_else(constants::default_tolerance::<f64>);
        
        Self {
            inner,
            cache: Arc::new(parking_lot::RwLock::new(None)),
            config,
            stats: Arc::new(parking_lot::Mutex::new(CacheStatistics::default())),
            tolerance,
        }
    }

    /// Returns cache statistics.
    pub fn cache_stats(&self) -> CacheStatistics {
        self.stats.lock().clone()
    }

    /// Clears the cache.
    pub fn clear_cache(&self) {
        *self.cache.write() = None;
        self.stats.lock().reset();
    }
}

impl<F> CostFunction<f64, Dyn> for CachedDynamicCostFunction<F>
where
    F: CostFunction<f64, Dyn>,
{
    fn cost(&self, point: &DVector<f64>) -> Result<f64> {
        if !self.config.cache_values {
            return self.inner.cost(point);
        }
        
        // Try to read from cache first
        {
            let cache_read = self.cache.read();
            if let Some(ref cache_entry) = *cache_read {
                if cache_entry.matches_with_tolerance(point, self.tolerance) {
                    if let Some(value) = cache_entry.value {
                        self.stats.lock().cost_hits += 1;
                        return Ok(value);
                    }
                }
            }
        }
        
        // Cache miss - compute value
        self.stats.lock().cost_misses += 1;
        let value = self.inner.cost(point)?;
        
        // Update cache
        let mut cache_write = self.cache.write();
        match cache_write.as_mut() {
            Some(cache_entry) if cache_entry.matches_with_tolerance(point, self.tolerance) => {
                // Same point, just update value
                cache_entry.value = Some(value);
            }
            _ => {
                // Different point or no cache, create new entry
                let mut new_cache = LastPointCache::new(point.clone());
                new_cache.value = Some(value);
                *cache_write = Some(new_cache);
            }
        }
        
        Ok(value)
    }

    fn cost_and_gradient(
        &self,
        point: &DVector<f64>,
        workspace: &mut Workspace<f64>,
        gradient: &mut DVector<f64>,
    ) -> Result<f64> {
        if !self.config.cache_values && !self.config.cache_gradients {
            return self.inner.cost_and_gradient(point, workspace, gradient);
        }
        
        // Try to read from cache first
        {
            let cache_read = self.cache.read();
            if let Some(ref cache_entry) = *cache_read {
                if cache_entry.matches_with_tolerance(point, self.tolerance) {
                    if let (Some(value), Some(ref cached_grad)) = (cache_entry.value, &cache_entry.gradient) {
                        self.stats.lock().combined_hits += 1;
                        gradient.copy_from(cached_grad);
                        return Ok(value);
                    }
                }
            }
        }
        
        // Cache miss - compute both
        self.stats.lock().combined_misses += 1;
        let cost = self.inner.cost_and_gradient(point, workspace, gradient)?;
        
        // Update cache
        let mut cache_write = self.cache.write();
        let mut new_cache = LastPointCache::new(point.clone());
        new_cache.value = Some(cost);
        new_cache.gradient = Some(gradient.clone());
        *cache_write = Some(new_cache);
        
        Ok(cost)
    }

    fn cost_and_gradient_alloc(&self, point: &DVector<f64>) -> Result<(f64, DVector<f64>)> {
        if !self.config.cache_values && !self.config.cache_gradients {
            return self.inner.cost_and_gradient_alloc(point);
        }
        
        // Try to read from cache first
        {
            let cache_read = self.cache.read();
            if let Some(ref cache_entry) = *cache_read {
                if cache_entry.matches_with_tolerance(point, self.tolerance) {
                    if let (Some(value), Some(ref gradient)) = (cache_entry.value, &cache_entry.gradient) {
                        self.stats.lock().combined_hits += 1;
                        return Ok((value, gradient.clone()));
                    }
                }
            }
        }
        
        // Cache miss - compute both
        self.stats.lock().combined_misses += 1;
        let (cost, gradient) = self.inner.cost_and_gradient_alloc(point)?;
        
        // Update cache
        let mut cache_write = self.cache.write();
        let mut new_cache = LastPointCache::new(point.clone());
        new_cache.value = Some(cost);
        new_cache.gradient = Some(gradient.clone());
        *cache_write = Some(new_cache);
        
        Ok((cost, gradient))
    }

    fn gradient(&self, point: &DVector<f64>) -> Result<DVector<f64>> {
        if !self.config.cache_gradients {
            return self.inner.gradient(point);
        }
        
        // Try to read from cache first
        {
            let cache_read = self.cache.read();
            if let Some(ref cache_entry) = *cache_read {
                if cache_entry.matches_with_tolerance(point, self.tolerance) {
                    if let Some(ref gradient) = cache_entry.gradient {
                        self.stats.lock().gradient_hits += 1;
                        return Ok(gradient.clone());
                    }
                }
            }
        }
        
        // Cache miss
        self.stats.lock().gradient_misses += 1;
        
        // Try to use cost_and_gradient_alloc if it's more efficient
        let (gradient, cost_opt) = if let Ok((cost, grad)) = self.inner.cost_and_gradient_alloc(point) {
            (grad, Some(cost))
        } else {
            (self.inner.gradient(point)?, None)
        };
        
        // Update cache
        let mut cache_write = self.cache.write();
        match cache_write.as_mut() {
            Some(cache_entry) if cache_entry.matches_with_tolerance(point, self.tolerance) => {
                // Same point, update gradient and possibly cost
                cache_entry.gradient = Some(gradient.clone());
                if let Some(cost) = cost_opt {
                    cache_entry.value = Some(cost);
                }
            }
            _ => {
                // Different point or no cache, create new entry
                let mut new_cache = LastPointCache::new(point.clone());
                new_cache.gradient = Some(gradient.clone());
                new_cache.value = cost_opt;
                *cache_write = Some(new_cache);
            }
        }
        
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
    
    #[test]
    fn test_cache_tolerance_comparison() {
        let n = 3;
        let a = DMatrix::<f64>::identity(n, n);
        let b = DVector::zeros(n);
        let inner = QuadraticCost::new(a, b, 0.0);
        
        // Configure cache with a specific tolerance
        let config = CacheConfig {
            cache_values: true,
            cache_gradients: true,
            comparison_tolerance: Some(1e-8),
        };
        
        let cached = CachedDynamicCostFunction::with_config(inner, config);
        
        // Original point
        let x = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        // First evaluation - cache miss
        let cost1 = cached.cost(&x).unwrap();
        assert_eq!(cached.cache_stats().cost_misses, 1);
        assert_eq!(cached.cache_stats().cost_hits, 0);
        
        // Point with tiny perturbation within tolerance
        let x_perturbed = DVector::from_vec(vec![1.0 + 1e-9, 2.0 - 1e-9, 3.0 + 1e-9]);
        
        // Should be a cache hit because difference is within tolerance
        let cost2 = cached.cost(&x_perturbed).unwrap();
        assert_eq!(cached.cache_stats().cost_hits, 1);
        assert_eq!(cached.cache_stats().cost_misses, 1);
        
        // Due to tolerance, we get the cached value
        assert_eq!(cost1, cost2);
        
        // Point with larger perturbation outside tolerance
        let x_different = DVector::from_vec(vec![1.0 + 1e-7, 2.0, 3.0]);
        
        // Should be a cache miss because difference exceeds tolerance
        let _ = cached.cost(&x_different).unwrap();
        assert_eq!(cached.cache_stats().cost_misses, 2);
        assert_eq!(cached.cache_stats().cost_hits, 1);
    }
}