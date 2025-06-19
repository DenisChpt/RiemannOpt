//! Memory pool for efficient allocation reuse.
//!
//! This module provides thread-safe memory pools for vectors and matrices,
//! reducing allocation overhead in hot paths by reusing previously allocated memory.

use crate::types::{Scalar, DVector, DMatrix};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// A pool of reusable vectors of a specific type and size.
#[derive(Debug)]
struct VectorPoolInner<T: Scalar> {
    /// Pools organized by vector size
    pools: HashMap<usize, Vec<DVector<T>>>,
    /// Maximum number of vectors to keep per size
    max_per_size: usize,
}

impl<T: Scalar> VectorPoolInner<T> {
    fn new(max_per_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_per_size,
        }
    }
    
    fn acquire(&mut self, size: usize) -> DVector<T> {
        if let Some(pool) = self.pools.get_mut(&size) {
            if let Some(mut vec) = pool.pop() {
                // Clear the vector before returning
                vec.fill(T::zero());
                return vec;
            }
        }
        // No vector available, create a new one
        DVector::zeros(size)
    }
    
    fn release(&mut self, mut vec: DVector<T>) {
        let size = vec.len();
        let pool = self.pools.entry(size).or_insert_with(Vec::new);
        
        // Only keep the vector if we haven't reached the limit
        if pool.len() < self.max_per_size {
            // Clear sensitive data
            vec.fill(T::zero());
            pool.push(vec);
        }
        // Otherwise, let the vector be dropped
    }
}

/// Thread-safe vector pool.
#[derive(Clone)]
pub struct VectorPool<T: Scalar> {
    inner: Arc<Mutex<VectorPoolInner<T>>>,
}

impl<T: Scalar> VectorPool<T> {
    /// Create a new vector pool with the specified maximum vectors per size.
    pub fn new(max_per_size: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(VectorPoolInner::new(max_per_size))),
        }
    }
    
    /// Acquire a vector from the pool or create a new one.
    pub fn acquire(&self, size: usize) -> PooledVector<T> {
        let vec = self.inner.lock().acquire(size);
        PooledVector {
            vector: Some(vec),
            pool: self.clone(),
        }
    }
    
    /// Get the number of pooled vectors for a specific size.
    pub fn pool_size(&self, size: usize) -> usize {
        self.inner.lock().pools.get(&size).map_or(0, |p| p.len())
    }
    
    /// Clear all pooled vectors.
    pub fn clear(&self) {
        self.inner.lock().pools.clear();
    }
}

impl<T: Scalar> Default for VectorPool<T> {
    fn default() -> Self {
        Self::new(16) // Default to keeping 16 vectors per size
    }
}

/// A vector borrowed from a pool that automatically returns when dropped.
pub struct PooledVector<T: Scalar> {
    vector: Option<DVector<T>>,
    pool: VectorPool<T>,
}

impl<T: Scalar> PooledVector<T> {
    /// Take ownership of the vector, preventing it from returning to the pool.
    pub fn take(mut self) -> DVector<T> {
        self.vector.take().expect("Vector already taken")
    }
}

impl<T: Scalar> Deref for PooledVector<T> {
    type Target = DVector<T>;
    
    fn deref(&self) -> &Self::Target {
        self.vector.as_ref().expect("Vector already taken")
    }
}

impl<T: Scalar> DerefMut for PooledVector<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.vector.as_mut().expect("Vector already taken")
    }
}

impl<T: Scalar> Drop for PooledVector<T> {
    fn drop(&mut self) {
        if let Some(vec) = self.vector.take() {
            self.pool.inner.lock().release(vec);
        }
    }
}

/// A pool of reusable matrices of a specific type.
#[derive(Debug)]
struct MatrixPoolInner<T: Scalar> {
    /// Pools organized by (rows, cols) dimensions
    pools: HashMap<(usize, usize), Vec<DMatrix<T>>>,
    /// Maximum number of matrices to keep per dimension
    max_per_size: usize,
}

impl<T: Scalar> MatrixPoolInner<T> {
    fn new(max_per_size: usize) -> Self {
        Self {
            pools: HashMap::new(),
            max_per_size,
        }
    }
    
    fn acquire(&mut self, rows: usize, cols: usize) -> DMatrix<T> {
        let key = (rows, cols);
        if let Some(pool) = self.pools.get_mut(&key) {
            if let Some(mut mat) = pool.pop() {
                // Clear the matrix before returning
                mat.fill(T::zero());
                return mat;
            }
        }
        // No matrix available, create a new one
        DMatrix::zeros(rows, cols)
    }
    
    fn release(&mut self, mut mat: DMatrix<T>) {
        let rows = mat.nrows();
        let cols = mat.ncols();
        let key = (rows, cols);
        let pool = self.pools.entry(key).or_insert_with(Vec::new);
        
        // Only keep the matrix if we haven't reached the limit
        if pool.len() < self.max_per_size {
            // Clear sensitive data
            mat.fill(T::zero());
            pool.push(mat);
        }
        // Otherwise, let the matrix be dropped
    }
}

/// Thread-safe matrix pool.
#[derive(Clone)]
pub struct MatrixPool<T: Scalar> {
    inner: Arc<Mutex<MatrixPoolInner<T>>>,
}

impl<T: Scalar> MatrixPool<T> {
    /// Create a new matrix pool with the specified maximum matrices per dimension.
    pub fn new(max_per_size: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(MatrixPoolInner::new(max_per_size))),
        }
    }
    
    /// Acquire a matrix from the pool or create a new one.
    pub fn acquire(&self, rows: usize, cols: usize) -> PooledMatrix<T> {
        let mat = self.inner.lock().acquire(rows, cols);
        PooledMatrix {
            matrix: Some(mat),
            pool: self.clone(),
        }
    }
    
    /// Get the number of pooled matrices for specific dimensions.
    pub fn pool_size(&self, rows: usize, cols: usize) -> usize {
        self.inner.lock().pools.get(&(rows, cols)).map_or(0, |p| p.len())
    }
    
    /// Clear all pooled matrices.
    pub fn clear(&self) {
        self.inner.lock().pools.clear();
    }
}

impl<T: Scalar> Default for MatrixPool<T> {
    fn default() -> Self {
        Self::new(8) // Default to keeping 8 matrices per dimension
    }
}

/// A matrix borrowed from a pool that automatically returns when dropped.
pub struct PooledMatrix<T: Scalar> {
    matrix: Option<DMatrix<T>>,
    pool: MatrixPool<T>,
}

impl<T: Scalar> PooledMatrix<T> {
    /// Take ownership of the matrix, preventing it from returning to the pool.
    pub fn take(mut self) -> DMatrix<T> {
        self.matrix.take().expect("Matrix already taken")
    }
}

impl<T: Scalar> Deref for PooledMatrix<T> {
    type Target = DMatrix<T>;
    
    fn deref(&self) -> &Self::Target {
        self.matrix.as_ref().expect("Matrix already taken")
    }
}

impl<T: Scalar> DerefMut for PooledMatrix<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.matrix.as_mut().expect("Matrix already taken")
    }
}

impl<T: Scalar> Drop for PooledMatrix<T> {
    fn drop(&mut self) {
        if let Some(mat) = self.matrix.take() {
            self.pool.inner.lock().release(mat);
        }
    }
}

// Global thread-local pools for common types.
thread_local! {
    static F32_VECTOR_POOL: VectorPool<f32> = VectorPool::default();
    static F64_VECTOR_POOL: VectorPool<f64> = VectorPool::default();
    static F32_MATRIX_POOL: MatrixPool<f32> = MatrixPool::default();
    static F64_MATRIX_POOL: MatrixPool<f64> = MatrixPool::default();
}

/// Get a pooled vector of the specified size.
pub fn get_pooled_vector<T: Scalar + 'static>(size: usize) -> PooledVector<T> {
    use std::any::TypeId;
    
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        F32_VECTOR_POOL.with(|pool| {
            unsafe { std::mem::transmute(pool.acquire(size)) }
        })
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        F64_VECTOR_POOL.with(|pool| {
            unsafe { std::mem::transmute(pool.acquire(size)) }
        })
    } else {
        panic!("Memory pool only supports f32 and f64");
    }
}

/// Get a pooled matrix of the specified dimensions.
pub fn get_pooled_matrix<T: Scalar + 'static>(rows: usize, cols: usize) -> PooledMatrix<T> {
    use std::any::TypeId;
    
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        F32_MATRIX_POOL.with(|pool| {
            unsafe { std::mem::transmute(pool.acquire(rows, cols)) }
        })
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        F64_MATRIX_POOL.with(|pool| {
            unsafe { std::mem::transmute(pool.acquire(rows, cols)) }
        })
    } else {
        panic!("Memory pool only supports f32 and f64");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_vector_pool_basic() {
        let pool = VectorPool::<f64>::new(2);
        
        // Acquire vectors
        let mut v1 = pool.acquire(10);
        let mut v2 = pool.acquire(10);
        let mut v3 = pool.acquire(20);
        
        // Modify them
        v1[0] = 1.0;
        v2[1] = 2.0;
        v3[2] = 3.0;
        
        // Check pool is empty
        assert_eq!(pool.pool_size(10), 0);
        assert_eq!(pool.pool_size(20), 0);
        
        // Drop vectors to return to pool
        drop(v1);
        drop(v2);
        drop(v3);
        
        // Check pool now contains vectors
        assert_eq!(pool.pool_size(10), 2);
        assert_eq!(pool.pool_size(20), 1);
        
        // Acquire again and verify they're zeroed
        let v4 = pool.acquire(10);
        assert_eq!(v4[0], 0.0);
        assert_eq!(v4[1], 0.0);
    }
    
    #[test]
    fn test_matrix_pool_basic() {
        let pool = MatrixPool::<f32>::new(2);
        
        // Acquire matrices
        let mut m1 = pool.acquire(5, 5);
        let mut m2 = pool.acquire(5, 5);
        let mut m3 = pool.acquire(3, 4);
        
        // Modify them
        m1[(0, 0)] = 1.0;
        m2[(1, 1)] = 2.0;
        m3[(2, 3)] = 3.0;
        
        // Drop to return to pool
        drop(m1);
        drop(m2);
        drop(m3);
        
        // Check pool sizes
        assert_eq!(pool.pool_size(5, 5), 2);
        assert_eq!(pool.pool_size(3, 4), 1);
        
        // Acquire again and verify they're zeroed
        let m4 = pool.acquire(5, 5);
        assert_eq!(m4[(0, 0)], 0.0);
        assert_eq!(m4[(1, 1)], 0.0);
    }
    
    #[test]
    fn test_pool_limit() {
        let pool = VectorPool::<f64>::new(2);
        
        // Create 5 vectors without dropping them immediately
        let vectors: Vec<_> = (0..5).map(|_| pool.acquire(10)).collect();
        
        // Now drop all vectors - pool should only keep 2
        drop(vectors);
        
        // Pool should only keep 2
        assert_eq!(pool.pool_size(10), 2);
    }
    
    #[test]
    fn test_take_ownership() {
        let pool = VectorPool::<f64>::new(2);
        
        let v1 = pool.acquire(10);
        let owned = v1.take();
        
        assert_eq!(owned.len(), 10);
        
        // Dropping owned vector shouldn't return it to pool
        drop(owned);
        assert_eq!(pool.pool_size(10), 0);
    }
    
    #[test]
    fn test_global_pools() {
        let v1 = get_pooled_vector::<f64>(100);
        let v2 = get_pooled_vector::<f32>(50);
        let m1 = get_pooled_matrix::<f64>(10, 10);
        let m2 = get_pooled_matrix::<f32>(5, 5);
        
        assert_eq!(v1.len(), 100);
        assert_eq!(v2.len(), 50);
        assert_eq!(m1.nrows(), 10);
        assert_eq!(m2.ncols(), 5);
    }
}