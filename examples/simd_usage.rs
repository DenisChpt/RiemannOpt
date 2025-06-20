//! Example demonstrating SIMD dispatcher usage in RiemannOpt
//!
//! This example shows how to leverage the SIMD dispatcher for optimized
//! vector operations in manifold implementations.

use riemannopt_core::{
    compute::{get_dispatcher, SimdBackend},
    types::{DVector, Scalar},
};
use nalgebra::DVector as NalgebraDVector;
use std::time::Instant;

/// Example function using SIMD dispatcher for efficient vector operations
#[allow(dead_code)]
fn compute_gradient_norm<T: Scalar + 'static>(gradient: &DVector<T>) -> T {
    // Get the global SIMD dispatcher for type T
    let dispatcher = get_dispatcher::<T>();
    
    // Use SIMD-optimized norm computation
    dispatcher.norm(gradient)
}

/// Example of projecting a vector using SIMD operations
fn project_to_unit_sphere<T: Scalar + 'static>(vector: &mut DVector<T>) {
    let dispatcher = get_dispatcher::<T>();
    
    // Compute norm using SIMD
    let norm = dispatcher.norm(vector);
    
    if norm > T::epsilon() {
        // Scale vector in-place using SIMD
        dispatcher.scale(vector, T::one() / norm);
    }
}

/// Example of computing inner products with SIMD
fn gram_schmidt_step<T: Scalar + 'static>(
    v: &mut DVector<T>,
    u: &DVector<T>,
) {
    let dispatcher = get_dispatcher::<T>();
    
    // Compute inner product <v, u> using SIMD
    let inner = dispatcher.dot_product(v, u);
    
    // Update v = v - inner * u using SIMD axpy operation
    dispatcher.axpy(-inner, u, v);
}

/// Benchmark comparison between naive and SIMD operations
fn benchmark_operations() {
    println!("Benchmarking SIMD operations...\n");
    
    // Test different vector sizes
    let sizes = vec![100, 1000, 10000, 100000];
    
    for &size in &sizes {
        println!("Vector size: {}", size);
        
        // Create random vectors
        let v1 = NalgebraDVector::<f64>::from_fn(size, |_, _| rand::random::<f64>());
        let v2 = NalgebraDVector::<f64>::from_fn(size, |_, _| rand::random::<f64>());
        
        // Benchmark dot product
        let start = Instant::now();
        let naive_dot = v1.dot(&v2);
        let naive_time = start.elapsed();
        
        let dispatcher = get_dispatcher::<f64>();
        let start = Instant::now();
        let simd_dot = dispatcher.dot_product(&v1, &v2);
        let simd_time = start.elapsed();
        
        println!("  Dot product:");
        println!("    Naive: {:?}", naive_time);
        println!("    SIMD:  {:?} (speedup: {:.2}x)", 
                 simd_time, 
                 naive_time.as_secs_f64() / simd_time.as_secs_f64());
        
        // Verify results match
        assert!((naive_dot - simd_dot).abs() < 1e-10);
        
        // Benchmark normalization
        let mut v_naive = v1.clone();
        let start = Instant::now();
        let norm = v_naive.norm();
        if norm > 0.0 {
            v_naive /= norm;
        }
        let naive_time = start.elapsed();
        
        let mut v_simd = v1.clone();
        let start = Instant::now();
        project_to_unit_sphere(&mut v_simd);
        let simd_time = start.elapsed();
        
        println!("  Normalization:");
        println!("    Naive: {:?}", naive_time);
        println!("    SIMD:  {:?} (speedup: {:.2}x)", 
                 simd_time, 
                 naive_time.as_secs_f64() / simd_time.as_secs_f64());
        
        println!();
    }
}

/// Example usage in a manifold context
fn manifold_example() {
    println!("Example: Sphere manifold operations with SIMD\n");
    
    // Create a point on sphere
    let mut point = NalgebraDVector::<f64>::from_vec(vec![3.0, 4.0, 0.0]);
    println!("Original point: {:?}", point);
    
    // Project to sphere using SIMD
    project_to_unit_sphere(&mut point);
    println!("Projected to sphere: {:?}", point);
    println!("Norm check: {}", point.norm());
    
    // Create a tangent vector
    let mut tangent = NalgebraDVector::<f64>::from_vec(vec![1.0, 0.0, 1.0]);
    println!("\nOriginal tangent: {:?}", tangent);
    
    // Project to tangent space using SIMD
    gram_schmidt_step(&mut tangent, &point);
    println!("Projected to tangent space: {:?}", tangent);
    println!("Orthogonality check: {}", point.dot(&tangent));
}

fn main() {
    println!("=== SIMD Dispatcher Usage Example ===\n");
    
    // Check if SIMD features are available
    let dispatcher = get_dispatcher::<f64>();
    println!("SIMD dispatcher initialized for f64");
    println!("Backend efficiency check for size 1000: {}\n", 
             dispatcher.is_efficient_for_size(1000));
    
    // Run examples
    manifold_example();
    println!("\n{}\n", "=".repeat(50));
    benchmark_operations();
    
    println!("\nNote: Actual speedups depend on CPU features and vector sizes.");
    println!("The dispatcher automatically selects the best backend.");
}