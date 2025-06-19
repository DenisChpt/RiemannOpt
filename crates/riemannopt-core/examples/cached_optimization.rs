//! Example demonstrating the multi-level cache system.

use nalgebra::{DVector, DMatrix};
use riemannopt_core::{
    core::{
        cost_function::{CostFunction, QuadraticCost},
        cached_cost_function_dyn::{CachedDynamicCostFunction, CacheConfig},
    },
};
use std::time::{Duration, Instant};

/// A slow cost function that simulates expensive computation.
#[derive(Debug)]
struct SlowQuadraticCost {
    inner: QuadraticCost<f64, nalgebra::Dyn>,
    delay_ms: u64,
}

impl SlowQuadraticCost {
    fn new(n: usize, delay_ms: u64) -> Self {
        let a = DMatrix::identity(n, n) * 2.0;
        let b = DVector::zeros(n);
        Self {
            inner: QuadraticCost::new(a, b, 0.0),
            delay_ms,
        }
    }
}

impl CostFunction<f64, nalgebra::Dyn> for SlowQuadraticCost {
    fn cost(&self, point: &DVector<f64>) -> riemannopt_core::error::Result<f64> {
        // Simulate expensive computation
        std::thread::sleep(Duration::from_millis(self.delay_ms));
        self.inner.cost(point)
    }

    fn cost_and_gradient(&self, point: &DVector<f64>) -> riemannopt_core::error::Result<(f64, DVector<f64>)> {
        // Simulate expensive computation
        std::thread::sleep(Duration::from_millis(self.delay_ms));
        self.inner.cost_and_gradient(point)
    }

    fn gradient(&self, point: &DVector<f64>) -> riemannopt_core::error::Result<DVector<f64>> {
        // Simulate expensive computation
        std::thread::sleep(Duration::from_millis(self.delay_ms));
        self.inner.gradient(point)
    }
}

fn main() {
    println!("Multi-Level Cache Example");
    println!("========================\n");

    let n = 50;
    let delay_ms = 10; // 10ms per evaluation
    
    // Create a slow cost function
    let slow_cost = SlowQuadraticCost::new(n, delay_ms);
    
    // Configure cache
    let cache_config = CacheConfig {
        l1_max_entries: 50,
        l1_max_bytes: 1024 * 1024, // 1 MB
        l2_max_entries: 200,
        l2_max_bytes: 10 * 1024 * 1024, // 10 MB
        l2_ttl: Duration::from_secs(60),
        cache_gradients: true,
        cache_combined: true,
    };
    
    // Wrap with cache
    let cached_cost = CachedDynamicCostFunction::with_config(slow_cost, cache_config);
    
    // Test points
    let test_points = vec![
        DVector::from_element(n, 1.0),
        DVector::from_element(n, 2.0),
        DVector::from_element(n, 3.0),
        DVector::from_element(n, 1.0), // Repeat first point
        DVector::from_element(n, 2.0), // Repeat second point
    ];
    
    println!("Testing cost function evaluations:");
    println!("---------------------------------");
    
    for (i, point) in test_points.iter().enumerate() {
        let start = Instant::now();
        let cost = cached_cost.cost(point).unwrap();
        let elapsed = start.elapsed();
        
        println!("Point {}: cost = {:.6}, time = {:?}", i + 1, cost, elapsed);
    }
    
    let stats = cached_cost.cache_stats();
    println!("\nCache Statistics:");
    println!("----------------");
    println!("Cost hits: {}", stats.cost_hits);
    println!("Cost misses: {}", stats.cost_misses);
    println!("Hit rate: {:.2}%", stats.hit_rate() * 100.0);
    
    // Test gradient caching
    println!("\nTesting gradient evaluations:");
    println!("----------------------------");
    
    for (i, point) in test_points[..3].iter().enumerate() {
        let start = Instant::now();
        let grad = cached_cost.gradient(point).unwrap();
        let elapsed = start.elapsed();
        
        println!("Point {}: gradient norm = {:.6}, time = {:?}", 
                 i + 1, grad.norm(), elapsed);
    }
    
    let stats = cached_cost.cache_stats();
    println!("\nUpdated Cache Statistics:");
    println!("-----------------------");
    println!("Gradient hits: {}", stats.gradient_hits);
    println!("Gradient misses: {}", stats.gradient_misses);
    println!("Overall hit rate: {:.2}%", stats.hit_rate() * 100.0);
    
    // Demonstrate cache clearing
    println!("\nClearing cache...");
    cached_cost.clear_cache();
    
    let start = Instant::now();
    let _ = cached_cost.cost(&test_points[0]).unwrap();
    let elapsed = start.elapsed();
    println!("After clear - first evaluation time: {:?}", elapsed);
    
    // Test combined cost and gradient
    println!("\nTesting combined cost and gradient:");
    println!("----------------------------------");
    
    let point = DVector::from_element(n, 4.0);
    
    let start = Instant::now();
    let (cost, grad) = cached_cost.cost_and_gradient(&point).unwrap();
    let elapsed1 = start.elapsed();
    
    let start = Instant::now();
    let cost2 = cached_cost.cost(&point).unwrap();
    let grad2 = cached_cost.gradient(&point).unwrap();
    let elapsed2 = start.elapsed();
    
    println!("First call (combined): time = {:?}", elapsed1);
    println!("Second calls (separate, cached): time = {:?}", elapsed2);
    println!("Cost match: {}, Gradient match: {}", 
             cost == cost2, grad == grad2);
    
    let final_stats = cached_cost.cache_stats();
    println!("\nFinal Cache Statistics:");
    println!("---------------------");
    println!("Total cost accesses: {}", 
             final_stats.cost_hits + final_stats.cost_misses);
    println!("Total gradient accesses: {}", 
             final_stats.gradient_hits + final_stats.gradient_misses);
    println!("Total combined accesses: {}", 
             final_stats.combined_hits + final_stats.combined_misses);
    println!("Overall hit rate: {:.2}%", final_stats.hit_rate() * 100.0);
}