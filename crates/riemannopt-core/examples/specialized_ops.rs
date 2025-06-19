//! Example demonstrating specialized optimizations for small dimensions and sparse matrices.

use nalgebra::{DVector, DMatrix, Vector3};
use riemannopt_core::compute::{
    SmallDimSelector, Ops3D, SmallDimOps,
    CsrMatrix, CooMatrix,
    SparseAwareBackend,
    ComputeBackend,
};
use std::time::Instant;

fn main() {
    println!("Specialized Operations Example");
    println!("=============================\n");
    
    demonstrate_small_dim_ops();
    println!();
    demonstrate_sparse_ops();
    println!();
    benchmark_comparison();
}

fn demonstrate_small_dim_ops() {
    println!("Small Dimension Optimizations");
    println!("----------------------------");
    
    // Test 3D operations
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    
    let ops = Ops3D;
    let dot = ops.dot_small(&a, &b);
    println!("3D dot product: {} (unrolled)", dot);
    
    let norm = ops.norm_small(&a);
    println!("3D norm: {:.6} (unrolled)", norm);
    
    // Test sphere operations
    let point = Vector3::new(1.0, 0.0, 0.0);
    let tangent = Vector3::new(0.0, 1.0, 0.0);
    
    let retracted = riemannopt_core::compute::retract_sphere_3d(&point, &tangent, 0.5);
    println!("Retracted point: [{:.4}, {:.4}, {:.4}]", 
             retracted[0], retracted[1], retracted[2]);
    println!("Norm after retraction: {:.6}", retracted.norm());
    
    // Test fast path selector
    let sizes = vec![2, 3, 4, 5, 10];
    println!("\nFast path availability:");
    for size in sizes {
        let has_fast_path = SmallDimSelector::is_small_dim(size);
        println!("  Dimension {}: {}", size, 
                 if has_fast_path { "✓ Fast path" } else { "✗ Generic path" });
    }
}

fn demonstrate_sparse_ops() {
    println!("Sparse Matrix Operations");
    println!("-----------------------");
    
    // Create a sparse matrix using COO format
    let n = 1000;
    let mut coo = CooMatrix::<f64>::new(n, n);
    
    // Add diagonal and some off-diagonal elements
    for i in 0..n {
        coo.push(i, i, 2.0).unwrap();
        if i > 0 {
            coo.push(i, i-1, -1.0).unwrap();
        }
        if i < n-1 {
            coo.push(i, i+1, -1.0).unwrap();
        }
    }
    
    let csr = coo.to_csr();
    println!("Sparse matrix: {}x{}", csr.nrows(), csr.ncols());
    println!("Non-zero elements: {} ({:.2}% sparse)", 
             csr.nnz(), csr.sparsity() * 100.0);
    
    // Perform sparse matrix-vector multiplication
    let x = DVector::from_element(n, 1.0);
    let mut y = DVector::zeros(n);
    
    let start = Instant::now();
    csr.spmv(&x, &mut y).unwrap();
    let sparse_time = start.elapsed();
    
    println!("SpMV time: {:?}", sparse_time);
    println!("Result norm: {:.6}", y.norm());
    
    // Compare with dense for small matrix
    if n <= 10 {
        let dense = csr.to_dense();
        let y_dense = &dense * &x;
        let diff = (&y - &y_dense).norm();
        println!("Difference from dense: {:.2e}", diff);
    }
}

fn benchmark_comparison() {
    println!("Performance Comparison");
    println!("---------------------");
    
    // Compare small dimension operations
    let iterations = 1_000_000;
    let a = vec![1.0, 2.0, 3.0];
    let b = vec![4.0, 5.0, 6.0];
    
    // Specialized implementation
    let ops = Ops3D;
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = ops.dot_small(&a, &b);
    }
    let specialized_time = start.elapsed();
    
    // Generic implementation
    let a_vec = DVector::from_vec(a.clone());
    let b_vec = DVector::from_vec(b.clone());
    let start = Instant::now();
    for _ in 0..iterations {
        let _ = a_vec.dot(&b_vec);
    }
    let generic_time = start.elapsed();
    
    println!("\n3D Dot Product ({} iterations):", iterations);
    println!("  Specialized: {:?}", specialized_time);
    println!("  Generic: {:?}", generic_time);
    println!("  Speedup: {:.2}x", 
             generic_time.as_secs_f64() / specialized_time.as_secs_f64());
    
    // Compare sparse vs dense operations
    let n = 500;
    let mut dense = DMatrix::<f64>::zeros(n, n);
    
    // Create tridiagonal matrix
    for i in 0..n {
        dense[(i, i)] = 2.0;
        if i > 0 {
            dense[(i, i-1)] = -1.0;
        }
        if i < n-1 {
            dense[(i, i+1)] = -1.0;
        }
    }
    
    let sparse = CsrMatrix::from_dense(&dense, 1e-10);
    let x = DVector::from_element(n, 1.0);
    let mut y_sparse = DVector::zeros(n);
    
    let iterations = 100;
    
    // Sparse multiplication
    let start = Instant::now();
    for _ in 0..iterations {
        sparse.spmv(&x, &mut y_sparse).unwrap();
    }
    let sparse_time = start.elapsed();
    
    // Dense multiplication (allocating version for fair comparison)
    let start = Instant::now();
    for _ in 0..iterations {
        let _y_dense = &dense * &x;
    }
    let dense_time = start.elapsed();
    
    println!("\nMatrix-Vector Product ({}x{}, {} iterations):", n, n, iterations);
    println!("  Sparse ({} nnz): {:?}", sparse.nnz(), sparse_time);
    println!("  Dense: {:?}", dense_time);
    println!("  Speedup: {:.2}x", 
             dense_time.as_secs_f64() / sparse_time.as_secs_f64());
    
    // Test sparse-aware backend
    let backend = SparseAwareBackend::<f64>::new()
        .with_threshold(0.9);
    
    let mut y_backend = DVector::zeros(n);
    let start = Instant::now();
    for _ in 0..iterations {
        backend.gemv(1.0, &dense, &x, 0.0, &mut y_backend).unwrap();
    }
    let backend_time = start.elapsed();
    
    println!("  Sparse-aware backend: {:?}", backend_time);
    println!("  Auto-detection overhead: {:.2}%", 
             ((backend_time.as_secs_f64() / sparse_time.as_secs_f64()) - 1.0) * 100.0);
}