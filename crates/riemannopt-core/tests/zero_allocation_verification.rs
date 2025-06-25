//! Simple test to verify the zero-allocation API works correctly

use nalgebra::{DMatrix, DVectorView, DVectorViewMut};
use riemannopt_core::compute::cpu::parallel::ParallelBatch;

#[test]
fn test_gradient_api() {
    // Setup test data
    let n_points = 100;
    let dim = 50;
    let points = DMatrix::<f64>::from_fn(dim, n_points, |i, j| (i * n_points + j) as f64 * 0.1);
    let mut output = DMatrix::<f64>::zeros(dim, n_points);
    
    // Define in-place gradient function that doesn't allocate
    let grad_func = |x: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
        // Simple gradient: 2x (gradient of f(x) = x^T x)
        // Using manual loop to avoid any potential allocations
        for i in 0..x.len() {
            out[i] = 2.0 * x[i];
        }
    };
    
    // Call the function
    ParallelBatch::gradient(&points, &mut output, grad_func).unwrap();
    
    // Verify correctness
    for j in 0..n_points {
        for i in 0..dim {
            let expected = 2.0 * points[(i, j)];
            assert!((output[(i, j)] - expected).abs() < 1e-10, 
                    "Incorrect result at ({}, {}): expected {}, got {}", 
                    i, j, expected, output[(i, j)]);
        }
    }
}

#[test]
fn test_map_api() {
    // Setup test data
    let n_points = 100;
    let dim = 50;
    let points = DMatrix::<f64>::from_fn(dim, n_points, |i, j| (i * n_points + j) as f64 * 0.1 + 0.1);
    let mut output = DMatrix::<f64>::zeros(dim, n_points);
    
    // Define in-place operation that doesn't allocate
    let op = |x: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
        // Normalize vector manually to avoid allocations
        let mut norm_squared = 0.0;
        for i in 0..x.len() {
            norm_squared += x[i] * x[i];
        }
        let norm = norm_squared.sqrt();
        
        if norm > 0.0 {
            for i in 0..x.len() {
                out[i] = x[i] / norm;
            }
        } else {
            for i in 0..x.len() {
                out[i] = x[i];
            }
        }
    };
    
    // Call the function
    ParallelBatch::map(&points, &mut output, op).unwrap();
    
    // Verify all output vectors are normalized
    for j in 0..n_points {
        let mut norm_squared = 0.0;
        for i in 0..dim {
            norm_squared += output[(i, j)] * output[(i, j)];
        }
        let norm = norm_squared.sqrt();
        assert!((norm - 1.0).abs() < 1e-10, 
                "Vector {} is not normalized: norm = {}", j, norm);
    }
}

#[test]
fn test_map_pairs_api() {
    // Setup test data
    let n_points = 100;
    let dim = 50;
    let points = DMatrix::<f64>::from_fn(dim, n_points, |i, j| (i * n_points + j) as f64 * 0.1);
    let tangents = DMatrix::<f64>::from_fn(dim, n_points, |i, j| (i + j) as f64 * 0.01);
    let mut output = DMatrix::<f64>::zeros(dim, n_points);
    
    // Define in-place operation that doesn't allocate
    let op = |p: DVectorView<f64>, t: DVectorView<f64>, mut out: DVectorViewMut<f64>| {
        // Simple retraction: p + t
        for i in 0..p.len() {
            out[i] = p[i] + t[i];
        }
    };
    
    // Call the function
    ParallelBatch::map_pairs(&points, &tangents, &mut output, op).unwrap();
    
    // Verify correctness
    for j in 0..n_points {
        for i in 0..dim {
            let expected = points[(i, j)] + tangents[(i, j)];
            assert!((output[(i, j)] - expected).abs() < 1e-10, 
                    "Incorrect result at ({}, {}): expected {}, got {}", 
                    i, j, expected, output[(i, j)]);
        }
    }
}

