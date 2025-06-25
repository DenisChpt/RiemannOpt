//! Cached cost function wrapper for eliminating redundant computations.
//!
//! This module provides a caching wrapper around cost functions to avoid
//! recomputing values when called multiple times with the same point.
//! This is critical for optimization performance as algorithms often need
//! the cost, gradient, and sometimes Hessian at the same point.

use crate::{
    error::Result,
    manifold::{Point, TangentVector},
    memory::Workspace,
    types::Scalar,
    core::cost_function::CostFunction,
};
use nalgebra::{allocator::Allocator, DefaultAllocator, Dim, OMatrix};
use num_traits::Float;
use std::cell::RefCell;
use std::fmt::Debug;

/// A caching wrapper for cost functions that eliminates redundant computations.
///
/// This wrapper tracks the last evaluated point and caches the cost, gradient,
/// and Hessian values. When called with the same point, it returns cached values
/// instead of recomputing them.
///
/// # Type Parameters
///
/// * `C` - The underlying cost function type
/// * `T` - The scalar type (f32, f64)
/// * `D` - The dimension type
///
/// # Example
///
/// ```rust
/// use riemannopt_core::core::{CachedCostFunction, QuadraticCost, CostFunction};
/// use nalgebra::{DVector, Dyn};
/// 
/// let cost_fn = QuadraticCost::<f64, Dyn>::simple(Dyn(3));
/// let cached = CachedCostFunction::new(&cost_fn);
/// 
/// let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
/// 
/// // First call computes the value
/// let cost1 = cached.cost(&point).unwrap();
/// 
/// // Second call with same point returns cached value
/// let cost2 = cached.cost(&point).unwrap();
/// assert_eq!(cost1, cost2);
/// ```
#[derive(Debug)]
pub struct CachedCostFunction<'a, C, T, D>
where
    C: CostFunction<T, D>,
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// The underlying cost function
    inner: &'a C,
    /// Cache storage wrapped in RefCell for interior mutability
    cache: RefCell<CacheStorage<T, D>>,
}

/// Internal cache storage
#[derive(Debug)]
struct CacheStorage<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// The last point that was evaluated
    point_cache: Option<Point<T, D>>,
    /// Cached cost value
    cost_cache: Option<T>,
    /// Cached gradient value
    grad_cache: Option<TangentVector<T, D>>,
    /// Cached Hessian value
    hess_cache: Option<OMatrix<T, D, D>>,
    /// Number of cache hits for cost
    cost_hits: usize,
    /// Number of cache misses for cost
    cost_misses: usize,
    /// Number of cache hits for gradient
    grad_hits: usize,
    /// Number of cache misses for gradient
    grad_misses: usize,
    /// Number of cache hits for Hessian
    hess_hits: usize,
    /// Number of cache misses for Hessian
    hess_misses: usize,
}

impl<T, D> Default for CacheStorage<T, D>
where
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    fn default() -> Self {
        Self {
            point_cache: None,
            cost_cache: None,
            grad_cache: None,
            hess_cache: None,
            cost_hits: 0,
            cost_misses: 0,
            grad_hits: 0,
            grad_misses: 0,
            hess_hits: 0,
            hess_misses: 0,
        }
    }
}

impl<'a, C, T, D> CachedCostFunction<'a, C, T, D>
where
    C: CostFunction<T, D>,
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    /// Creates a new cached cost function wrapper.
    pub fn new(inner: &'a C) -> Self {
        Self {
            inner,
            cache: RefCell::new(CacheStorage::default()),
        }
    }

    /// Returns the underlying cost function.
    pub fn inner(&self) -> &C {
        self.inner
    }

    /// Returns cache statistics as (hits, misses) for (cost, gradient, hessian).
    pub fn cache_stats(&self) -> ((usize, usize), (usize, usize), (usize, usize)) {
        let cache = self.cache.borrow();
        (
            (cache.cost_hits, cache.cost_misses),
            (cache.grad_hits, cache.grad_misses),
            (cache.hess_hits, cache.hess_misses),
        )
    }

    /// Resets the cache, clearing all stored values.
    pub fn reset_cache(&self) {
        let mut cache = self.cache.borrow_mut();
        cache.point_cache = None;
        cache.cost_cache = None;
        cache.grad_cache = None;
        cache.hess_cache = None;
    }

    /// Resets cache statistics.
    pub fn reset_stats(&self) {
        let mut cache = self.cache.borrow_mut();
        cache.cost_hits = 0;
        cache.cost_misses = 0;
        cache.grad_hits = 0;
        cache.grad_misses = 0;
        cache.hess_hits = 0;
        cache.hess_misses = 0;
    }

    /// Checks if the cache is valid for the given point and invalidates if necessary.
    fn check_and_invalidate(&self, point: &Point<T, D>) {
        let mut cache = self.cache.borrow_mut();
        
        // Check if point has changed
        let point_changed = if let Some(ref cached_point) = cache.point_cache {
            // Compare points element by element
            point.len() != cached_point.len() || 
            point.iter().zip(cached_point.iter())
                .any(|(a, b)| <T as Float>::abs(*a - *b) > T::epsilon())
        } else {
            true
        };

        if point_changed {
            // Point has changed, invalidate all caches
            cache.point_cache = Some(point.clone());
            cache.cost_cache = None;
            cache.grad_cache = None;
            cache.hess_cache = None;
        }
    }
}

impl<'a, C, T, D> CostFunction<T, D> for CachedCostFunction<'a, C, T, D>
where
    C: CostFunction<T, D>,
    T: Scalar,
    D: Dim,
    DefaultAllocator: Allocator<D> + Allocator<D, D>,
{
    fn cost(&self, point: &Point<T, D>) -> Result<T> {
        self.check_and_invalidate(point);
        
        let mut cache = self.cache.borrow_mut();
        
        if let Some(cost) = cache.cost_cache {
            cache.cost_hits += 1;
            Ok(cost)
        } else {
            cache.cost_misses += 1;
            let cost = self.inner.cost(point)?;
            cache.cost_cache = Some(cost);
            Ok(cost)
        }
    }

    fn cost_and_gradient(
        &self,
        point: &Point<T, D>,
        workspace: &mut Workspace<T>,
        gradient: &mut TangentVector<T, D>,
    ) -> Result<T> {
        self.check_and_invalidate(point);
        
        let mut cache = self.cache.borrow_mut();
        
        // Check if both are cached
        let cost_cached = cache.cost_cache;
        let grad_cached = cache.grad_cache.clone();
        
        match (cost_cached, grad_cached) {
            (Some(cost), Some(grad)) => {
                cache.cost_hits += 1;
                cache.grad_hits += 1;
                gradient.copy_from(&grad);
                Ok(cost)
            }
            _ => {
                // Need to compute at least one
                cache.cost_misses += 1;
                cache.grad_misses += 1;
                
                // Drop the borrow before calling inner function
                drop(cache);
                
                let cost = self.inner.cost_and_gradient(point, workspace, gradient)?;
                
                // Re-borrow to update cache
                let mut cache = self.cache.borrow_mut();
                cache.cost_cache = Some(cost);
                cache.grad_cache = Some(gradient.clone());
                
                Ok(cost)
            }
        }
    }

    fn cost_and_gradient_alloc(&self, point: &Point<T, D>) -> Result<(T, TangentVector<T, D>)> {
        self.check_and_invalidate(point);
        
        let mut cache = self.cache.borrow_mut();
        
        // Check if both are cached
        let cost_cached = cache.cost_cache;
        let grad_cached = cache.grad_cache.clone();
        
        match (cost_cached, grad_cached) {
            (Some(cost), Some(grad)) => {
                cache.cost_hits += 1;
                cache.grad_hits += 1;
                Ok((cost, grad))
            }
            _ => {
                // Need to compute at least one
                cache.cost_misses += 1;
                cache.grad_misses += 1;
                
                // Drop the borrow before calling inner function
                drop(cache);
                
                let (cost, grad) = self.inner.cost_and_gradient_alloc(point)?;
                
                // Re-borrow to update cache
                let mut cache = self.cache.borrow_mut();
                cache.cost_cache = Some(cost);
                cache.grad_cache = Some(grad.clone());
                
                Ok((cost, grad))
            }
        }
    }

    fn gradient(&self, point: &Point<T, D>) -> Result<TangentVector<T, D>> {
        self.check_and_invalidate(point);
        
        let mut cache = self.cache.borrow_mut();
        
        if let Some(grad) = cache.grad_cache.clone() {
            cache.grad_hits += 1;
            Ok(grad)
        } else {
            cache.grad_misses += 1;
            
            let cost_is_cached = cache.cost_cache.is_some();
            
            // Drop the borrow before calling inner function
            drop(cache);
            
            // Try to get both cost and gradient if the inner function provides an efficient implementation
            let grad = if !cost_is_cached {
                // Cost is not cached, might as well get both
                let (cost, grad) = self.inner.cost_and_gradient_alloc(point)?;
                let mut cache = self.cache.borrow_mut();
                cache.cost_cache = Some(cost);
                cache.grad_cache = Some(grad.clone());
                grad
            } else {
                // Cost is already cached, just get gradient
                let grad = self.inner.gradient(point)?;
                let mut cache = self.cache.borrow_mut();
                cache.grad_cache = Some(grad.clone());
                grad
            };
            
            Ok(grad)
        }
    }

    fn hessian(&self, point: &Point<T, D>) -> Result<OMatrix<T, D, D>>
    where
        DefaultAllocator: Allocator<D, D>,
    {
        self.check_and_invalidate(point);
        
        let mut cache = self.cache.borrow_mut();
        
        if let Some(hess) = cache.hess_cache.clone() {
            cache.hess_hits += 1;
            Ok(hess)
        } else {
            cache.hess_misses += 1;
            
            // Drop the borrow before calling inner function
            drop(cache);
            
            let hess = self.inner.hessian(point)?;
            
            // Re-borrow to update cache
            let mut cache = self.cache.borrow_mut();
            cache.hess_cache = Some(hess.clone());
            
            Ok(hess)
        }
    }

    fn hessian_vector_product(
        &self,
        point: &Point<T, D>,
        vector: &TangentVector<T, D>,
    ) -> Result<TangentVector<T, D>> {
        // For Hessian-vector products, we don't cache as the vector changes each time
        // But we might benefit from having the Hessian cached
        self.check_and_invalidate(point);
        
        // Try to use cached Hessian if available and inner doesn't have specialized implementation
        let cache = self.cache.borrow();
        if let Some(ref hess) = cache.hess_cache {
            Ok(hess * vector)
        } else {
            drop(cache);
            // Use inner's implementation which might be more efficient
            self.inner.hessian_vector_product(point, vector)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cost_function::QuadraticCost;
    use approx::assert_relative_eq;
    use nalgebra::{DVector, Dyn};

    #[test]
    fn test_cached_cost_function_basic() {
        let inner = QuadraticCost::<f64, Dyn>::simple(Dyn(3));
        let cached = CachedCostFunction::new(&inner);
        
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        // First call should miss
        let cost1 = cached.cost(&point).unwrap();
        let ((cost_hits, cost_misses), _, _) = cached.cache_stats();
        assert_eq!(cost_hits, 0);
        assert_eq!(cost_misses, 1);
        
        // Second call should hit
        let cost2 = cached.cost(&point).unwrap();
        let ((cost_hits, cost_misses), _, _) = cached.cache_stats();
        assert_eq!(cost_hits, 1);
        assert_eq!(cost_misses, 1);
        
        assert_relative_eq!(cost1, cost2);
    }

    #[test]
    fn test_cached_gradient() {
        let inner = QuadraticCost::<f64, Dyn>::simple(Dyn(3));
        let cached = CachedCostFunction::new(&inner);
        
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        // Get gradient twice
        let grad1 = cached.gradient(&point).unwrap();
        let grad2 = cached.gradient(&point).unwrap();
        
        let (_, (grad_hits, grad_misses), _) = cached.cache_stats();
        assert_eq!(grad_hits, 1);
        assert_eq!(grad_misses, 1);
        
        assert_relative_eq!(grad1, grad2);
    }

    #[test]
    fn test_cache_invalidation() {
        let inner = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let cached = CachedCostFunction::new(&inner);
        
        let point1 = DVector::from_vec(vec![1.0, 2.0]);
        let point2 = DVector::from_vec(vec![2.0, 3.0]);
        
        // Compute at point1
        let _ = cached.cost(&point1).unwrap();
        let _ = cached.gradient(&point1).unwrap();
        
        // Compute at point2 - should invalidate cache
        let _ = cached.cost(&point2).unwrap();
        
        let ((cost_hits, cost_misses), (grad_hits, grad_misses), _) = cached.cache_stats();
        assert_eq!(cost_hits, 0);
        assert_eq!(cost_misses, 2);
        assert_eq!(grad_hits, 0);
        assert_eq!(grad_misses, 1);
    }

    #[test]
    fn test_cost_and_gradient_caching() {
        let inner = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let cached = CachedCostFunction::new(&inner);
        
        let point = DVector::from_vec(vec![1.0, 2.0]);
        
        // Get both cost and gradient
        let (cost1, grad1) = cached.cost_and_gradient_alloc(&point).unwrap();
        
        // Get them separately - should use cache
        let cost2 = cached.cost(&point).unwrap();
        let grad2 = cached.gradient(&point).unwrap();
        
        let ((cost_hits, cost_misses), (grad_hits, grad_misses), _) = cached.cache_stats();
        assert_eq!(cost_hits, 1);
        assert_eq!(cost_misses, 1);
        assert_eq!(grad_hits, 1);
        assert_eq!(grad_misses, 1);
        
        assert_relative_eq!(cost1, cost2);
        assert_relative_eq!(grad1, grad2);
    }

    #[test]
    fn test_cache_reset() {
        let inner = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let cached = CachedCostFunction::new(&inner);
        
        let point = DVector::from_vec(vec![1.0, 2.0]);
        
        // Populate cache
        let _ = cached.cost(&point).unwrap();
        let _ = cached.gradient(&point).unwrap();
        
        // Reset cache
        cached.reset_cache();
        
        // Next calls should miss
        let _ = cached.cost(&point).unwrap();
        let ((cost_hits, cost_misses), _, _) = cached.cache_stats();
        assert_eq!(cost_hits, 0);
        assert_eq!(cost_misses, 2);
    }

    #[test]
    fn test_hessian_caching() {
        let inner = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let cached = CachedCostFunction::new(&inner);
        
        let point = DVector::from_vec(vec![1.0, 2.0]);
        
        // Get Hessian twice
        let hess1 = cached.hessian(&point).unwrap();
        let hess2 = cached.hessian(&point).unwrap();
        
        let (_, _, (hess_hits, hess_misses)) = cached.cache_stats();
        assert_eq!(hess_hits, 1);
        assert_eq!(hess_misses, 1);
        
        assert_relative_eq!(hess1, hess2);
    }
}