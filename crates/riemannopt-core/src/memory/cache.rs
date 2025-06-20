//! Multi-level cache infrastructure for optimization.
//!
//! This module provides a two-level caching system:
//! - L1: Thread-local cache with LRU eviction
//! - L2: Concurrent shared cache using dashmap
//!
//! The cache is designed to store expensive computation results,
//! particularly cost function and gradient evaluations.

use std::collections::HashMap;
use std::hash::Hash;
use std::sync::Arc;
use std::time::{Duration, Instant};
use dashmap::DashMap;
use lru::LruCache;
use std::num::NonZeroUsize;
use std::cell::RefCell;

thread_local! {
    /// Thread-local L1 cache storage
    /// 
    /// Note: RefCell is safe here because each thread gets its own instance
    /// of the cache, preventing any data races. This is the standard pattern
    /// for thread-local storage in Rust.
    static L1_CACHE: RefCell<HashMap<std::any::TypeId, Box<dyn std::any::Any>>> = RefCell::new(HashMap::new());
}

/// Trait for types that can be used as cache keys.
pub trait CacheKey: Hash + Eq + Clone + Send + Sync + 'static {}

/// Trait for types that can be cached.
pub trait Cacheable: Clone + Send + Sync + 'static {
    /// Returns the size in bytes for cache eviction decisions.
    fn size_bytes(&self) -> usize {
        std::mem::size_of_val(self)
    }
}

/// A cached value with metadata.
#[derive(Clone, Debug)]
pub struct CachedValue<V> {
    /// The cached value
    pub value: V,
    /// When the value was cached
    pub timestamp: Instant,
    /// Number of times this value has been accessed
    pub access_count: usize,
    /// Size in bytes
    pub size_bytes: usize,
}

impl<V: Cacheable> CachedValue<V> {
    /// Creates a new cached value.
    pub fn new(value: V) -> Self {
        let size_bytes = value.size_bytes();
        Self {
            value,
            timestamp: Instant::now(),
            access_count: 0,
            size_bytes,
        }
    }

    /// Returns the age of the cached value.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }

    /// Increments the access count and returns the value.
    pub fn access(&mut self) -> &V {
        self.access_count += 1;
        &self.value
    }
}

/// Thread-local L1 cache with LRU eviction.
pub struct L1Cache<K, V> {
    cache: LruCache<K, CachedValue<V>>,
    max_bytes: usize,
    current_bytes: usize,
}

impl<K: CacheKey, V: Cacheable> L1Cache<K, V> {
    /// Creates a new L1 cache with specified capacity.
    pub fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            cache: LruCache::new(NonZeroUsize::new(max_entries).unwrap()),
            max_bytes,
            current_bytes: 0,
        }
    }

    /// Inserts a value into the cache.
    pub fn insert(&mut self, key: K, value: V) {
        let cached = CachedValue::new(value);
        let size = cached.size_bytes;
        
        // Evict entries if necessary
        while self.current_bytes + size > self.max_bytes && self.cache.len() > 0 {
            if let Some((_key, evicted)) = self.cache.pop_lru() {
                self.current_bytes -= evicted.size_bytes;
            }
        }
        
        // Insert new value
        self.cache.push(key, cached);
        self.current_bytes += size;
    }

    /// Gets a value from the cache.
    pub fn get(&mut self, key: &K) -> Option<V> {
        self.cache.get_mut(key).map(|cached| {
            cached.access_count += 1;
            cached.value.clone()
        })
    }

    /// Clears the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
        self.current_bytes = 0;
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.cache.len(),
            bytes: self.current_bytes,
            capacity_entries: self.cache.cap().get(),
            capacity_bytes: self.max_bytes,
        }
    }
}

/// Shared L2 cache using concurrent hashmap.
pub struct L2Cache<K, V> {
    cache: Arc<DashMap<K, CachedValue<V>>>,
    max_entries: usize,
    max_bytes: usize,
    ttl: Duration,
}

impl<K: CacheKey, V: Cacheable> L2Cache<K, V> {
    /// Creates a new L2 cache.
    pub fn new(max_entries: usize, max_bytes: usize, ttl: Duration) -> Self {
        Self {
            cache: Arc::new(DashMap::new()),
            max_entries,
            max_bytes,
            ttl,
        }
    }

    /// Inserts a value into the cache.
    pub fn insert(&self, key: K, value: V) {
        let cached = CachedValue::new(value);
        
        // Simple eviction: remove oldest entries if at capacity
        if self.cache.len() >= self.max_entries {
            self.evict_oldest();
        }
        
        self.cache.insert(key, cached);
    }

    /// Gets a value from the cache.
    pub fn get(&self, key: &K) -> Option<V> {
        self.cache.get_mut(key).and_then(|mut entry| {
            let cached = entry.value_mut();
            
            // Check TTL
            if cached.age() > self.ttl {
                drop(entry);
                self.cache.remove(key);
                return None;
            }
            
            cached.access_count += 1;
            Some(cached.value.clone())
        })
    }

    /// Evicts the oldest entry.
    fn evict_oldest(&self) {
        if let Some(oldest_key) = self.cache
            .iter()
            .min_by_key(|entry| entry.value().timestamp)
            .map(|entry| entry.key().clone())
        {
            self.cache.remove(&oldest_key);
        }
    }

    /// Clears the cache.
    pub fn clear(&self) {
        self.cache.clear();
    }

    /// Removes expired entries.
    pub fn cleanup(&self) {
        let now = Instant::now();
        self.cache.retain(|_, cached| {
            now.duration_since(cached.timestamp) <= self.ttl
        });
    }

    /// Returns cache statistics.
    pub fn stats(&self) -> CacheStats {
        let total_bytes: usize = self.cache
            .iter()
            .map(|entry| entry.value().size_bytes)
            .sum();
            
        CacheStats {
            entries: self.cache.len(),
            bytes: total_bytes,
            capacity_entries: self.max_entries,
            capacity_bytes: self.max_bytes,
        }
    }
}

/// Multi-level cache combining L1 and L2.
pub struct MultiLevelCache<K, V> {
    l2: L2Cache<K, V>,
    l1_config: (usize, usize), // (max_entries, max_bytes)
}

impl<K: CacheKey, V: Cacheable> MultiLevelCache<K, V> {
    /// Creates a new multi-level cache.
    pub fn new(
        l1_max_entries: usize,
        l1_max_bytes: usize,
        l2_max_entries: usize,
        l2_max_bytes: usize,
        l2_ttl: Duration,
    ) -> Self {
        Self {
            l2: L2Cache::new(l2_max_entries, l2_max_bytes, l2_ttl),
            l1_config: (l1_max_entries, l1_max_bytes),
        }
    }

    /// Gets a value, checking L1 first, then L2.
    pub fn get(&self, key: &K) -> Option<V> {
        // Get or create thread-local L1 cache
        L1_CACHE.with(|cache_map| {
            let mut map = cache_map.borrow_mut();
            let type_id = std::any::TypeId::of::<L1Cache<K, V>>();
            
            let l1 = map.entry(type_id)
                .or_insert_with(|| {
                    Box::new(L1Cache::<K, V>::new(self.l1_config.0, self.l1_config.1))
                });
                
            let l1_cache = l1.downcast_mut::<L1Cache<K, V>>().unwrap();
            
            // Check L1
            if let Some(value) = l1_cache.get(key) {
                return Some(value);
            }
            
            // Check L2
            if let Some(value) = self.l2.get(key) {
                // Promote to L1
                l1_cache.insert(key.clone(), value.clone());
                return Some(value);
            }
            
            None
        })
    }

    /// Inserts a value into both cache levels.
    pub fn insert(&self, key: K, value: V) {
        // Insert into L2
        self.l2.insert(key.clone(), value.clone());
        
        // Insert into L1
        L1_CACHE.with(|cache_map| {
            let mut map = cache_map.borrow_mut();
            let type_id = std::any::TypeId::of::<L1Cache<K, V>>();
            
            let l1 = map.entry(type_id)
                .or_insert_with(|| {
                    Box::new(L1Cache::<K, V>::new(self.l1_config.0, self.l1_config.1))
                });
                
            let l1_cache = l1.downcast_mut::<L1Cache<K, V>>().unwrap();
            l1_cache.insert(key, value);
        });
    }

    /// Clears both cache levels.
    pub fn clear(&self) {
        self.l2.clear();
        
        L1_CACHE.with(|cache_map| {
            cache_map.borrow_mut().clear();
        });
    }

    /// Performs maintenance operations.
    pub fn maintain(&self) {
        self.l2.cleanup();
    }
}

/// Cache statistics.
#[derive(Debug, Clone)]
pub struct CacheStats {
    /// Number of entries
    pub entries: usize,
    /// Total size in bytes
    pub bytes: usize,
    /// Maximum number of entries
    pub capacity_entries: usize,
    /// Maximum size in bytes
    pub capacity_bytes: usize,
}

impl CacheStats {
    /// Returns the hit ratio if tracking is enabled.
    pub fn utilization(&self) -> f64 {
        if self.capacity_entries > 0 {
            self.entries as f64 / self.capacity_entries as f64
        } else {
            0.0
        }
    }
}

// Implement CacheKey for common types
impl<T: Hash + Eq + Clone + Send + Sync + 'static> CacheKey for Vec<T> {}
impl CacheKey for String {}
impl CacheKey for u64 {}
impl CacheKey for (u64, u64) {}

// Default implementations would conflict with specific implementations
// Users should implement Cacheable for their specific types

#[cfg(test)]
mod tests {
    use super::*;
    
    // Implement Cacheable for f64 in tests
    impl Cacheable for f64 {}
    
    #[test]
    fn test_l1_cache_basic() {
        let mut cache = L1Cache::<String, f64>::new(2, 1024);
        
        cache.insert("a".to_string(), 1.0);
        cache.insert("b".to_string(), 2.0);
        
        assert_eq!(cache.get(&"a".to_string()), Some(1.0));
        assert_eq!(cache.get(&"b".to_string()), Some(2.0));
        assert_eq!(cache.get(&"c".to_string()), None);
    }
    
    #[test]
    fn test_l1_cache_lru_eviction() {
        let mut cache = L1Cache::<String, f64>::new(2, 1024);
        
        cache.insert("a".to_string(), 1.0);
        cache.insert("b".to_string(), 2.0);
        cache.insert("c".to_string(), 3.0); // Should evict "a"
        
        assert_eq!(cache.get(&"a".to_string()), None);
        assert_eq!(cache.get(&"b".to_string()), Some(2.0));
        assert_eq!(cache.get(&"c".to_string()), Some(3.0));
    }
    
    #[test]
    fn test_l2_cache_ttl() {
        let cache = L2Cache::<String, f64>::new(10, 1024, Duration::from_millis(100));
        
        cache.insert("a".to_string(), 1.0);
        assert_eq!(cache.get(&"a".to_string()), Some(1.0));
        
        // Wait for TTL to expire
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(cache.get(&"a".to_string()), None);
    }
    
    #[test]
    fn test_multi_level_cache() {
        let cache = MultiLevelCache::<String, f64>::new(2, 1024, 10, 10240, Duration::from_secs(60));
        
        // Insert and retrieve
        cache.insert("a".to_string(), 1.0);
        assert_eq!(cache.get(&"a".to_string()), Some(1.0));
        
        // Clear and verify
        cache.clear();
        assert_eq!(cache.get(&"a".to_string()), None);
    }
}