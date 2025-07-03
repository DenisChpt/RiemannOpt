//! Performance tests for manifold operations
//!
//! These tests ensure that manifold operations maintain expected performance
//! characteristics and don't regress.

use riemannopt_manifolds::{Sphere, Stiefel, Grassmann};
use riemannopt_core::{manifold::Manifold, memory::workspace::Workspace};
use nalgebra::{DVector, DMatrix};
use std::time::Instant;

/// Helper to measure operation time
fn time_operation<F: FnOnce()>(f: F) -> std::time::Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

#[test]
fn test_sphere_projection_performance() {
    let sphere = Sphere::<f64>::new(1000).unwrap();
    let point = sphere.random_point();
    let mut result = DVector::zeros(1000);
    let mut workspace = Workspace::<f64>::new();
    
    // Warm up
    for _ in 0..10 {
        sphere.project_point(&point, &mut result, &mut workspace);
    }
    
    // Measure
    let iterations = 1000;
    let duration = time_operation(|| {
        for _ in 0..iterations {
            sphere.project_point(&point, &mut result, &mut workspace);
        }
    });
    
    let avg_time = duration.as_micros() as f64 / iterations as f64;
    println!("Sphere projection (n=1000): {:.2} μs", avg_time);
    
    // Performance assertion - projection should be fast
    assert!(avg_time < 10.0, "Sphere projection too slow: {:.2} μs", avg_time);
}

#[test]
fn test_stiefel_retraction_performance() {
    let stiefel = Stiefel::<f64>::new(50, 10).unwrap();
    let point = stiefel.random_point();
    let mut tangent = DMatrix::zeros(50, 10);
    let mut workspace = Workspace::<f64>::new();
    stiefel.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
    tangent *= 0.1; // Small step
    
    let mut result = DMatrix::zeros(50, 10);
    
    // Warm up
    for _ in 0..10 {
        stiefel.retract(&point, &tangent, &mut result, &mut workspace).unwrap();
    }
    
    // Measure
    let iterations = 100;
    let duration = time_operation(|| {
        for _ in 0..iterations {
            stiefel.retract(&point, &tangent, &mut result, &mut workspace).unwrap();
        }
    });
    
    let avg_time = duration.as_micros() as f64 / iterations as f64;
    println!("Stiefel retraction (50x10): {:.2} μs", avg_time);
    
    // QR-based retraction should be reasonably fast
    assert!(avg_time < 500.0, "Stiefel retraction too slow: {:.2} μs", avg_time);
}

#[test]
fn test_grassmann_inner_product_performance() {
    let grassmann = Grassmann::<f64>::new(100, 20).unwrap();
    let point = grassmann.random_point();
    let mut u = DMatrix::zeros(100, 20);
    let mut v = DMatrix::zeros(100, 20);
    let mut workspace = Workspace::<f64>::new();
    
    grassmann.random_tangent(&point, &mut u, &mut workspace).unwrap();
    grassmann.random_tangent(&point, &mut v, &mut workspace).unwrap();
    
    // Warm up
    for _ in 0..10 {
        let _ = grassmann.inner_product(&point, &u, &v).unwrap();
    }
    
    // Measure
    let iterations = 1000;
    let duration = time_operation(|| {
        for _ in 0..iterations {
            let _ = grassmann.inner_product(&point, &u, &v).unwrap();
        }
    });
    
    let avg_time = duration.as_nanos() as f64 / iterations as f64;
    println!("Grassmann inner product (100x20): {:.2} ns", avg_time);
    
    // Inner product is just trace computation, should be very fast
    assert!(avg_time < 1000.0, "Grassmann inner product too slow: {:.2} ns", avg_time);
}

#[test]
fn test_memory_allocation_overhead() {
    let sphere = Sphere::<f64>::new(1000).unwrap();
    let mut workspace = Workspace::<f64>::new();
    
    // Test that workspace prevents allocations
    let point = sphere.random_point();
    let mut tangent = DVector::zeros(1000);
    
    // First call might allocate workspace memory
    sphere.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
    
    // Subsequent calls should not allocate
    let iterations = 1000;
    let duration = time_operation(|| {
        for _ in 0..iterations {
            sphere.random_tangent(&point, &mut tangent, &mut workspace).unwrap();
        }
    });
    
    let avg_time = duration.as_micros() as f64 / iterations as f64;
    println!("Random tangent generation with workspace (n=1000): {:.2} μs", avg_time);
    
    // With workspace, should be fast
    assert!(avg_time < 50.0, "Random tangent generation too slow: {:.2} μs", avg_time);
}

#[test]
fn test_large_scale_operations() {
    // Test that operations scale well with dimension
    let sizes = vec![(10, 5), (50, 20), (100, 30)];
    let mut times = Vec::new();
    
    for (n, p) in sizes.iter() {
        let stiefel = Stiefel::<f64>::new(*n, *p).unwrap();
        let mut workspace = Workspace::<f64>::new();
        let x = stiefel.random_point();
        let y = stiefel.random_point();
        
        let start = Instant::now();
        let _ = stiefel.distance(&x, &y, &mut workspace).unwrap();
        let elapsed = start.elapsed();
        
        times.push(elapsed.as_micros());
        println!("Stiefel distance ({}x{}): {} μs", n, p, elapsed.as_micros());
    }
    
    // Check that scaling is reasonable (not exponential)
    // Time should increase roughly with matrix size
    let ratio1 = times[1] as f64 / times[0] as f64;
    let ratio2 = times[2] as f64 / times[1] as f64;
    
    // Ratios should be reasonable (not exponential growth)
    assert!(ratio1 < 30.0, "Scaling from 10x5 to 50x20 too steep: {:.2}x", ratio1);
    assert!(ratio2 < 10.0, "Scaling from 50x20 to 100x30 too steep: {:.2}x", ratio2);
}