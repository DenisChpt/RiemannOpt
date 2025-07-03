//! Cached cost function wrapper for eliminating redundant computations.
//!
//! This module provides a caching wrapper around cost functions to avoid
//! recomputing values when called multiple times with the same point.
//! This is critical for optimization performance as algorithms often need
//! the cost, gradient, and sometimes Hessian at the same point.

use crate::{
    error::Result,
    memory::Workspace,
    types::Scalar,
    core::cost_function::CostFunction,
};
use nalgebra::OMatrix;
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
/// * `T` - The scalar type
///
/// # Example
///
/// ```rust,ignore
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
pub struct CachedCostFunction<'a, C, T>
where
    C: CostFunction<T> + ?Sized,
    T: Scalar,
{
    /// The underlying cost function
    inner: &'a C,
    /// Cache storage wrapped in RefCell for interior mutability
    cache: RefCell<CacheStorage<T, C::Point, C::TangentVector>>,
}

impl<'a, C, T> Debug for CachedCostFunction<'a, C, T>
where
    C: CostFunction<T> + ?Sized,
    T: Scalar,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CachedCostFunction")
            .field("inner", &self.inner)
            .finish()
    }
}

/// Internal cache storage
#[derive(Debug)]
struct CacheStorage<T, P, TV>
where
    T: Scalar,
{
    /// The last point that was evaluated
    point_cache: Option<P>,
    /// Cached cost value
    cost_cache: Option<T>,
    /// Cached gradient value
    grad_cache: Option<TV>,
    /// Cached Hessian value
    hess_cache: Option<OMatrix<T, nalgebra::Dyn, nalgebra::Dyn>>,
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

impl<T, P, TV> Default for CacheStorage<T, P, TV>
where
    T: Scalar,
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

impl<'a, C, T> CachedCostFunction<'a, C, T>
where
    C: CostFunction<T> + ?Sized,
    T: Scalar,
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
}

/// Trait for types that can be compared for cache validity
pub trait CacheComparable {
    /// Check if two values are approximately equal for caching purposes
    fn cache_equal(&self, other: &Self, tolerance: f64) -> bool;
}

// Implement for common types
impl<T: Scalar> CacheComparable for nalgebra::DVector<T> {
    fn cache_equal(&self, other: &Self, tolerance: f64) -> bool {
        self.len() == other.len() && 
        self.iter().zip(other.iter())
            .all(|(a, b)| {
                let diff = num_traits::Float::abs(*a - *b);
                diff.to_f64() < tolerance
            })
    }
}

impl<T: Scalar> CacheComparable for nalgebra::DMatrix<T> {
    fn cache_equal(&self, other: &Self, tolerance: f64) -> bool {
        self.shape() == other.shape() && 
        self.iter().zip(other.iter())
            .all(|(a, b)| {
                let diff = num_traits::Float::abs(*a - *b);
                diff.to_f64() < tolerance
            })
    }
}

impl<'a, C, T> CachedCostFunction<'a, C, T>
where
    C: CostFunction<T> + ?Sized,
    T: Scalar,
    C::Point: CacheComparable + Clone,
    C::TangentVector: Clone,
{
    /// Checks if the cache is valid for the given point and invalidates if necessary.
    fn check_and_invalidate(&self, point: &C::Point) {
        let mut cache = self.cache.borrow_mut();
        
        // Check if point has changed
        let point_changed = if let Some(ref cached_point) = cache.point_cache {
            !cached_point.cache_equal(point, T::epsilon().to_f64())
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

impl<'a, C, T> CostFunction<T> for CachedCostFunction<'a, C, T>
where
    C: CostFunction<T> + ?Sized,
    T: Scalar,
    C::Point: CacheComparable + Clone,
    C::TangentVector: Clone,
{
    type Point = C::Point;
    type TangentVector = C::TangentVector;
    
    fn cost(&self, point: &Self::Point) -> Result<T> {
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
        point: &Self::Point,
        workspace: &mut Workspace<T>,
        gradient: &mut Self::TangentVector,
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
                gradient.clone_from(&grad);
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

    fn cost_and_gradient_alloc(&self, point: &Self::Point) -> Result<(T, Self::TangentVector)> {
        self.check_and_invalidate(point);
        
        let mut cache = self.cache.borrow_mut();
        
        let cost_cached = cache.cost_cache;
        let grad_cached = cache.grad_cache.clone();
        
        match (cost_cached, grad_cached) {
            (Some(cost), Some(grad)) => {
                cache.cost_hits += 1;
                cache.grad_hits += 1;
                Ok((cost, grad))
            }
            _ => {
                cache.cost_misses += 1;
                cache.grad_misses += 1;
                
                drop(cache);
                
                let (cost, grad) = self.inner.cost_and_gradient_alloc(point)?;
                
                let mut cache = self.cache.borrow_mut();
                cache.cost_cache = Some(cost);
                cache.grad_cache = Some(grad.clone());
                
                Ok((cost, grad))
            }
        }
    }

    fn gradient(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        self.check_and_invalidate(point);
        
        let mut cache = self.cache.borrow_mut();
        
        if let Some(grad) = cache.grad_cache.clone() {
            cache.grad_hits += 1;
            Ok(grad)
        } else {
            cache.grad_misses += 1;
            drop(cache);
            
            let grad = self.inner.gradient(point)?;
            
            let mut cache = self.cache.borrow_mut();
            cache.grad_cache = Some(grad.clone());
            
            Ok(grad)
        }
    }

    fn hessian(&self, point: &Self::Point) -> Result<OMatrix<T, nalgebra::Dyn, nalgebra::Dyn>> {
        self.check_and_invalidate(point);
        
        let mut cache = self.cache.borrow_mut();
        
        if let Some(hess) = cache.hess_cache.clone() {
            cache.hess_hits += 1;
            Ok(hess)
        } else {
            cache.hess_misses += 1;
            drop(cache);
            
            let hess = self.inner.hessian(point)?;
            
            let mut cache = self.cache.borrow_mut();
            cache.hess_cache = Some(hess.clone());
            
            Ok(hess)
        }
    }

    fn hessian_vector_product(
        &self,
        point: &Self::Point,
        vector: &Self::TangentVector,
    ) -> Result<Self::TangentVector> {
        // Note: we don't cache Hessian-vector products as they depend on both point and vector
        self.inner.hessian_vector_product(point, vector)
    }
    
    fn gradient_fd_alloc(&self, point: &Self::Point) -> Result<Self::TangentVector> {
        // For finite differences, we don't use caching since it would interfere
        // with the numerical approximation
        self.inner.gradient_fd_alloc(point)
    }
    
    fn gradient_fd(
        &self,
        point: &Self::Point,
        workspace: &mut Workspace<T>,
        gradient: &mut Self::TangentVector,
    ) -> Result<()> {
        // For finite differences, we don't use caching
        self.inner.gradient_fd(point, workspace, gradient)
    }
    
    fn gradient_fd_parallel(
        &self,
        point: &Self::Point,
        _config: &crate::compute::cpu::parallel::ParallelConfig,
    ) -> Result<Self::TangentVector> 
    where 
        Self: Sync,
    {
        // For finite differences, we don't use caching, but we can't call
        // the parallel method on inner if it's not Sync, so fall back to sequential
        self.gradient_fd_alloc(point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::cost_function::QuadraticCost;
    use nalgebra::{DVector, Dyn};
    use approx::assert_relative_eq;

    #[test]
    fn test_cache_basic() {
        let cost_fn = QuadraticCost::<f64, Dyn>::simple(Dyn(3));
        let cached = CachedCostFunction::new(&cost_fn);
        let point = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        
        // First call should miss
        let cost1 = cached.cost(&point).unwrap();
        let (cost_stats, _, _) = cached.cache_stats();
        assert_eq!(cost_stats, (0, 1)); // 0 hits, 1 miss
        
        // Second call should hit
        let cost2 = cached.cost(&point).unwrap();
        let (cost_stats, _, _) = cached.cache_stats();
        assert_eq!(cost_stats, (1, 1)); // 1 hit, 1 miss
        
        assert_eq!(cost1, cost2);
    }
    
    #[test]
    fn test_cache_invalidation() {
        let cost_fn = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let cached = CachedCostFunction::new(&cost_fn);
        
        let point1 = DVector::from_vec(vec![1.0, 2.0]);
        let point2 = DVector::from_vec(vec![3.0, 4.0]);
        
        // Compute at point1
        let _cost1 = cached.cost(&point1).unwrap();
        
        // Compute at point2 - should invalidate cache
        let _cost2 = cached.cost(&point2).unwrap();
        
        let (cost_stats, _, _) = cached.cache_stats();
        assert_eq!(cost_stats, (0, 2)); // 0 hits, 2 misses
    }
    
    #[test]
    fn test_gradient_caching() {
        let cost_fn = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let cached = CachedCostFunction::new(&cost_fn);
        
        let point = DVector::from_vec(vec![1.0, 2.0]);
        
        // First gradient call
        let grad1 = cached.gradient(&point).unwrap();
        let (_, grad_stats, _) = cached.cache_stats();
        assert_eq!(grad_stats, (0, 1));
        
        // Second gradient call - should hit cache
        let grad2 = cached.gradient(&point).unwrap();
        let (_, grad_stats, _) = cached.cache_stats();
        assert_eq!(grad_stats, (1, 1));
        
        assert_relative_eq!(grad1, grad2);
    }
    
    #[test]
    fn test_cost_and_gradient() {
        let cost_fn = QuadraticCost::<f64, Dyn>::simple(Dyn(2));
        let cached = CachedCostFunction::new(&cost_fn);
        
        let point = DVector::from_vec(vec![1.0, 2.0]);
        
        // First call computes both
        let (cost1, grad1) = cached.cost_and_gradient_alloc(&point).unwrap();
        
        // Individual calls should now hit cache
        let cost2 = cached.cost(&point).unwrap();
        let grad2 = cached.gradient(&point).unwrap();
        
        assert_eq!(cost1, cost2);
        assert_relative_eq!(grad1, grad2);
        
        let ((cost_hits, _), (grad_hits, _), _) = cached.cache_stats();
        assert_eq!(cost_hits, 1);
        assert_eq!(grad_hits, 1);
    }
}