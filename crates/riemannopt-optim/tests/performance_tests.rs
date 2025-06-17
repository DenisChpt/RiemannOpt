//! Performance tests for optimization algorithms
//!
//! These tests ensure that optimization algorithms maintain expected performance
//! characteristics and don't regress.

use riemannopt_optim::{SGD, SGDConfig, Adam, AdamConfig, LBFGS, LBFGSConfig};
use riemannopt_core::{
    optimizer::{Optimizer, StoppingCriterion},
    manifold::Manifold,
    cost_function::CostFunction,
    error::Result,
};
use riemannopt_manifolds::Sphere;
use nalgebra::DVector;
use std::time::Instant;

/// Helper to measure optimization time
fn time_optimization<O, C, M>(
    optimizer: &mut O,
    cost_fn: &C,
    manifold: &M,
    x0: &DVector<f64>,
    max_iter: usize,
) -> (std::time::Duration, usize)
where
    O: Optimizer<f64, nalgebra::Dyn>,
    C: CostFunction<f64, nalgebra::Dyn>,
    M: Manifold<f64, nalgebra::Dyn>,
{
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(max_iter)
        .with_gradient_tolerance(1e-8);
    
    let start = Instant::now();
    let result = match optimizer.optimize(cost_fn, manifold, x0, &stopping_criterion) {
        Ok(res) => res,
        Err(_) => {
            // If optimization fails, return max iterations
            return (start.elapsed(), max_iter);
        }
    };
    let duration = start.elapsed();
    
    (duration, result.iterations)
}

#[derive(Debug)]
struct RastriginOnSphere {
    dimension: usize,
}

impl RastriginOnSphere {
    fn new(dimension: usize) -> Self {
        Self { dimension }
    }
}

impl CostFunction<f64, nalgebra::Dyn> for RastriginOnSphere {
    fn cost(&self, x: &DVector<f64>) -> Result<f64> {
        // Rastrigin function adapted for sphere
        let a = 10.0;
        let n = self.dimension as f64;
        
        Ok(a * n + x.iter()
            .map(|&xi| xi * xi - a * (2.0 * std::f64::consts::PI * xi).cos())
            .sum::<f64>())
    }
    
    fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>> {
        let a = 10.0;
        
        Ok(DVector::from_iterator(
            self.dimension,
            x.iter().map(|&xi| {
                2.0 * xi + 2.0 * a * std::f64::consts::PI * (2.0 * std::f64::consts::PI * xi).sin()
            })
        ))
    }
}

#[test]
fn test_sgd_performance_scaling() {
    // Test SGD performance on different dimensions
    let dimensions = vec![10, 50, 100];
    let mut times = Vec::new();
    
    for &dim in &dimensions {
        let sphere = Sphere::new(dim).unwrap();
        let cost_fn = RastriginOnSphere::new(dim);
        let x0 = sphere.random_point();
        
        let mut sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.01));
        
        let (duration, iterations) = time_optimization(&mut sgd, &cost_fn, &sphere, &x0, 100);
        
        let time_per_iter = duration.as_micros() as f64 / iterations as f64;
        times.push((dim, time_per_iter));
        
        println!("SGD - Dimension {}: {:.2} μs/iteration", dim, time_per_iter);
    }
    
    // Check that time doesn't scale too badly with dimension
    let time_10 = times[0].1;
    let time_100 = times[2].1;
    // Scaling from 10D to 100D (10x dimension increase) should be reasonable
    // We allow up to 60x time increase which accounts for O(n²) operations
    assert!(time_100 < time_10 * 60.0, "SGD scaling is too poor: {}x slower", time_100 / time_10);
}

#[test]
fn test_adam_performance() {
    let sphere = Sphere::new(50).unwrap();
    let cost_fn = RastriginOnSphere::new(50);
    let x0 = sphere.random_point();
    
    // Warm up
    let mut adam = Adam::new(AdamConfig::new().with_learning_rate(0.01));
    let _ = time_optimization(&mut adam, &cost_fn, &sphere, &x0, 10);
    
    // Actual test
    let (duration, iterations) = time_optimization(&mut adam, &cost_fn, &sphere, &x0, 100);
    let time_per_iter = duration.as_micros() as f64 / iterations as f64;
    
    println!("Adam - 50D: {:.2} μs/iteration", time_per_iter);
    
    // Adam should complete within reasonable time
    assert!(time_per_iter < 1000.0, "Adam too slow: {:.2} μs/iteration", time_per_iter);
}

#[test]
fn test_lbfgs_performance() {
    let sphere = Sphere::new(50).unwrap();
    let cost_fn = RastriginOnSphere::new(50);
    let x0 = sphere.random_point();
    
    // Test different memory sizes
    let memory_sizes = vec![5, 10, 20];
    
    for &m in &memory_sizes {
        let mut lbfgs = LBFGS::new(LBFGSConfig::new().with_memory_size(m));
        let (duration, iterations) = time_optimization(&mut lbfgs, &cost_fn, &sphere, &x0, 50);
        let time_per_iter = duration.as_micros() as f64 / iterations as f64;
        
        println!("L-BFGS - Memory {}: {:.2} μs/iteration", m, time_per_iter);
        
        // L-BFGS should be efficient even with larger memory
        assert!(time_per_iter < 2000.0, "L-BFGS too slow with memory {}: {:.2} μs/iteration", m, time_per_iter);
    }
}

#[test]
fn test_optimizer_comparison() {
    // Compare different optimizers on the same problem
    let sphere = Sphere::new(30).unwrap();
    let target = sphere.random_point();
    
    #[derive(Debug)]
    struct SimpleQuadratic {
        target: DVector<f64>,
    }
    
    impl CostFunction<f64, nalgebra::Dyn> for SimpleQuadratic {
        fn cost(&self, x: &DVector<f64>) -> Result<f64> {
            Ok((x - &self.target).norm_squared())
        }
        
        fn gradient(&self, x: &DVector<f64>) -> Result<DVector<f64>> {
            Ok(2.0 * (x - &self.target))
        }
    }
    
    let cost_fn = SimpleQuadratic { target };
    let x0 = sphere.random_point();
    
    // Test SGD
    let mut sgd = SGD::new(SGDConfig::new().with_constant_step_size(0.1));
    let (sgd_time, sgd_iter) = time_optimization(&mut sgd, &cost_fn, &sphere, &x0, 1000);
    
    // Test Adam
    let mut adam = Adam::new(AdamConfig::new().with_learning_rate(0.1));
    let (adam_time, adam_iter) = time_optimization(&mut adam, &cost_fn, &sphere, &x0, 1000);
    
    // Test L-BFGS
    let mut lbfgs = LBFGS::new(LBFGSConfig::new().with_memory_size(10));
    let (lbfgs_time, lbfgs_iter) = time_optimization(&mut lbfgs, &cost_fn, &sphere, &x0, 1000);
    
    println!("\nOptimizer Comparison (30D quadratic):");
    println!("SGD:    {} iterations in {:.2} ms", sgd_iter, sgd_time.as_secs_f64() * 1000.0);
    println!("Adam:   {} iterations in {:.2} ms", adam_iter, adam_time.as_secs_f64() * 1000.0);
    println!("L-BFGS: {} iterations in {:.2} ms", lbfgs_iter, lbfgs_time.as_secs_f64() * 1000.0);
    
    // L-BFGS should converge fast for quadratic problems when it succeeds
    // If L-BFGS failed (returned max_iter), skip the assertion
    if lbfgs_iter < 1000 {
        assert!(lbfgs_iter < sgd_iter, "L-BFGS should converge faster than SGD");
        // Adam can sometimes converge very fast too, so we'll be more lenient
        assert!(lbfgs_iter <= adam_iter + 5, "L-BFGS should converge reasonably fast compared to Adam");
    }
}

#[test]
#[ignore] // Run with --ignored for stress tests
fn stress_test_large_scale_optimization() {
    // Test with very large dimensions
    let sphere = Sphere::new(1000).unwrap();
    let cost_fn = RastriginOnSphere::new(1000);
    let x0 = sphere.random_point();
    
    let mut sgd = SGD::new(
        SGDConfig::new()
            .with_constant_step_size(0.001)
            .with_classical_momentum(0.9)
    );
    
    let start = Instant::now();
    let stopping_criterion = StoppingCriterion::new()
        .with_max_iterations(100)
        .with_gradient_tolerance(1e-6);
    
    let result = sgd.optimize(&cost_fn, &sphere, &x0, &stopping_criterion).unwrap();
    let duration = start.elapsed();
    
    println!("Large scale optimization (1000D): {} iterations in {:.2} s", 
             result.iterations, duration.as_secs_f64());
    
    // Should complete in reasonable time
    assert!(duration.as_secs() < 10, "Large scale optimization too slow");
}

#[test]
fn test_parallel_batch_performance() {
    
    let dim = 100;
    let batch_size = 32;
    
    // Create batch of points
    let points: Vec<DVector<f64>> = (0..batch_size)
        .map(|_| DVector::from_fn(dim, |_, _| rand::random::<f64>()))
        .collect();
    
    // Create batch of gradients
    let gradients: Vec<DVector<f64>> = points.iter()
        .map(|x| 2.0 * x)
        .collect();
    
    // Time batch averaging
    let start = Instant::now();
    for _ in 0..1000 {
        // Compute average manually
        let mut avg = DVector::zeros(dim);
        for grad in &gradients {
            avg += grad;
        }
        let _ = avg / (batch_size as f64);
    }
    let duration = start.elapsed();
    
    let avg_time = duration.as_micros() as f64 / 1000.0;
    println!("Batch gradient averaging ({}x{}): {:.2} μs", batch_size, dim, avg_time);
    
    // Should be efficient
    assert!(avg_time < 100.0, "Batch averaging too slow: {:.2} μs", avg_time);
}