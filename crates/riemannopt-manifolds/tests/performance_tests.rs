//! Performance tests for manifold operations
//!
//! These tests ensure that manifold operations maintain expected performance
//! characteristics and don't regress.

use riemannopt_manifolds::{Sphere, Stiefel, Grassmann};
use riemannopt_core::manifold::Manifold;
use std::time::Instant;

/// Helper to measure operation time
fn time_operation<F: FnOnce()>(f: F) -> std::time::Duration {
    let start = Instant::now();
    f();
    start.elapsed()
}

#[test]
fn test_sphere_projection_performance() {
    let sphere = Sphere::new(1000).unwrap();
    let point: nalgebra::DVector<f64> = sphere.random_point();
    
    // Warm up
    for _ in 0..10 {
        sphere.project_point(&point);
    }
    
    // Measure
    let iterations = 1000;
    let duration = time_operation(|| {
        for _ in 0..iterations {
            sphere.project_point(&point);
        }
    });
    
    let avg_time = duration.as_micros() as f64 / iterations as f64;
    println!("Sphere projection (n=1000): {:.2} μs", avg_time);
    
    // Performance assertion - projection should be fast
    assert!(avg_time < 10.0, "Sphere projection too slow: {:.2} μs", avg_time);
}

#[test]
fn test_stiefel_retraction_performance() {
    let stiefel = Stiefel::new(50, 10).unwrap();
    let point: nalgebra::DVector<f64> = stiefel.random_point();
    let tangent = stiefel.random_tangent(&point).unwrap();
    
    // Warm up
    for _ in 0..10 {
        let _ = stiefel.retract(&point, &tangent);
    }
    
    // Measure
    let iterations = 100;
    let duration = time_operation(|| {
        for _ in 0..iterations {
            let _ = stiefel.retract(&point, &tangent);
        }
    });
    
    let avg_time = duration.as_millis() as f64 / iterations as f64;
    println!("Stiefel retraction (50x10): {:.2} ms", avg_time);
    
    // QR decomposition should complete in reasonable time
    assert!(avg_time < 5.0, "Stiefel retraction too slow: {:.2} ms", avg_time);
}

#[test]
fn test_grassmann_distance_performance() {
    let grassmann = Grassmann::new(20, 5).unwrap();
    let x: nalgebra::DVector<f64> = grassmann.random_point();
    let y: nalgebra::DVector<f64> = grassmann.random_point();
    
    // Warm up
    for _ in 0..10 {
        let _ = grassmann.distance(&x, &y);
    }
    
    // Measure
    let iterations = 100;
    let duration = time_operation(|| {
        for _ in 0..iterations {
            let _ = grassmann.distance(&x, &y);
        }
    });
    
    let avg_time = duration.as_millis() as f64 / iterations as f64;
    println!("Grassmann distance (20x5): {:.2} ms", avg_time);
    
    // SVD-based distance should be reasonable
    assert!(avg_time < 10.0, "Grassmann distance too slow: {:.2} ms", avg_time);
}

#[test]
#[ignore] // Run with --ignored for stress tests
fn stress_test_large_manifolds() {
    // Test with very large dimensions
    let sphere = Sphere::new(10000).unwrap();
    let _: nalgebra::DVector<f64> = sphere.random_point();
    
    let stiefel = Stiefel::new(100, 20).unwrap();
    let _: nalgebra::DVector<f64> = stiefel.random_point();
    
    println!("Large manifold stress test passed");
}